"""Train, resume and jointly evaluate ordinary MAPPO and continuous MAAC.

Run as ``python -m experiments.public_baselines train ...`` or ``evaluate ...``.
Both algorithms use measured current states and directly command both thrusters.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

from citf.baselines import SourceMAPPO, SourceMAAC
from citf.bridge import World, source_modules
from citf.contracts import (Protocol, LayoutRegistry, validate_task_banks,
    immutable_run_contract, require_same_contract)
from citf.evaluate import atomic_json
from citf.train import atomic_save, read_journal


OBSERVATION = 'measured-body-frame17-peer7-v1'


def observation(world):
    state=world.state();own=state[:-1];enemy=state[-1];n=len(own)
    c,s=np.cos(own[:,2]),np.sin(own[:,2])
    def local(vector):
        return np.stack((c*vector[:,0]+s*vector[:,1],-s*vector[:,0]+c*vector[:,1]),-1)
    relative_yaw=enemy[2]-own[:,2]
    base=np.concatenate((local(-own[:,:2])/60.,local(enemy[:2]-own[:,:2])/60.,
        local(np.broadcast_to(enemy[3:5],(n,2)))/5.,local(own[:,3:5])/5.,
        np.sin(own[:,2:3]),np.cos(own[:,2:3]),own[:,5:6],
        np.sin(relative_yaw[:,None]),np.cos(relative_yaw[:,None]),np.full((n,1),enemy[5]),
        np.full((n,1),max(0.,1-world.env.Current_T/world.protocol.time_limit)),
        np.full((n,1),n/6.),np.full((n,1),world.protocol.target_radius/60.)),axis=-1)
    peers=[]
    for i in range(n):
        other=np.delete(own,i,axis=0);delta=other[:,:2]-own[i,:2];velocity=other[:,3:5]-own[i,3:5]
        angle=other[:,2]-own[i,2]
        peers.append(np.concatenate((np.stack((c[i]*delta[:,0]+s[i]*delta[:,1],
            -s[i]*delta[:,0]+c[i]*delta[:,1]),-1)/60.,
            np.stack((c[i]*velocity[:,0]+s[i]*velocity[:,1],-s[i]*velocity[:,0]+c[i]*velocity[:,1]),-1)/5.,
            np.sin(angle[:,None]),np.cos(angle[:,None]),other[:,5:6]),axis=-1).reshape(-1))
    return np.concatenate((base,np.stack(peers)),axis=-1).astype(np.float32)


def agent_state(agent):
    value={name:getattr(agent,name).state_dict() for name in ('actor','critic','actor_optimizer','critic_optimizer')}
    if isinstance(agent,SourceMAPPO):
        value.update(value_mean=agent.value_mean,value_scale=agent.value_scale,
            normalizer_updates=agent.normalizer_updates,rng=agent.rng.bit_generator.state)
    else:
        value.update(target_actor=agent.target_actor.state_dict(),target_critic=agent.target_critic.state_dict())
    return value


def restore_agent(agent,state,training=True):
    for name in ('actor','critic'):
        getattr(agent,name).load_state_dict(state[name])
    if training:
        for name in ('actor_optimizer','critic_optimizer'):
            getattr(agent,name).load_state_dict(state[name])
    if isinstance(agent,SourceMAPPO):
        for name in ('value_mean','value_scale','normalizer_updates'):setattr(agent,name,state[name])
        agent.rng.bit_generator.state=state['rng']
    else:
        agent.target_actor.load_state_dict(state['target_actor'])
        agent.target_critic.load_state_dict(state['target_critic'])


def make_agent(config,device):
    cls=SourceMAPPO if config['algorithm']=='MAPPO' else SourceMAAC
    return cls(device=device,width=config['width'],seed=config['seed'],public_state=True)


def load_policy(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')!='citf-public-baseline-v1' or payload['config']['observation']!=OBSERVATION:
        raise ValueError('Expected a public-baseline checkpoint with its measured-state interface')
    if payload['config']['algorithm'] not in ('MAPPO','MAAC'):
        raise ValueError('Unknown public algorithm')
    agent=make_agent(payload['config'],device);restore_agent(agent,payload['state'],training=False)
    agent.actor.eval();agent.critic.eval()
    agent.metadata={key:payload[key] for key in ('schema','config','physical_protocol','update','environment_steps')}
    return agent


class PublicController:
    supported_selections=frozenset(('joint',))
    def __init__(self,source,model,device='cpu',*unused,**kwargs):self.model=model

    def act(self,world):
        value=self.model.act(observation(world)[None],deterministic=True)
        action=value[0][0] if isinstance(self.model,SourceMAPPO) else value[0]
        force=np.clip(250+750*action,-500,1000).astype(np.float32)
        return force,dict(evaluated_unique=0,forward_batches=1,physical_thrust=force.tolist(),
            algorithm=self.model.metadata['config']['algorithm'],observation=OBSERVATION,
            source_used_online=False,teacher_used_online=False)


class Replay:
    def __init__(self,capacity):self.capacity=capacity;self.size=0;self.cursor=0;self.arrays=None

    def add(self,transition):
        if self.arrays is None:
            self.arrays=[np.empty((self.capacity,*x.shape[1:]),dtype=np.float32) for x in transition]
        for row in zip(*transition):
            for array,value in zip(self.arrays,row):array[self.cursor]=value
            self.cursor=(self.cursor+1)%self.capacity;self.size=min(self.size+1,self.capacity)

    def batch(self,rng,count):
        indices=rng.integers(self.size,size=count)
        return [x[indices] for x in self.arrays]

    def state(self):
        return dict(capacity=self.capacity,size=self.size,cursor=self.cursor,
            arrays=None if self.arrays is None else [x[:self.size].copy() for x in self.arrays])

    def restore(self,state):
        self.size,self.cursor=state['size'],state['cursor']
        if self.capacity!=state['capacity']:raise ValueError('Replay capacity changed')
        if state['arrays'] is not None:
            self.arrays=[np.empty((self.capacity,*x.shape[1:]),dtype=np.float32) for x in state['arrays']]
            for full,saved in zip(self.arrays,state['arrays']):full[:self.size]=saved


def validate(agent,project,physical,bank):
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    rows=[]
    try:
        for task in bank['tasks']:
            world=World(project,physical,task['defenders'])
            world.reset(task['scene_seed'],task['future_seed'],task['agility'])
            if world.initial_identity['fingerprint']!=task['geometry_family_id']:
                raise ValueError('Validation layout differs from allocation')
            controller=PublicController(None,agent)
            while not world.done:
                force,_=controller.act(world);result=world.step(force)
            rows.append(dict(task=task,**result))
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
    return dict(N=len(rows),success=sum(x['success'] for x in rows),collision=sum(x['collision'] for x in rows),
        capped_capture_time=float(np.mean([x['duration'] if x['capture'] else physical.time_limit for x in rows])),tasks=rows)


def train(argv):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','tasks','validation','registry','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--algorithm',choices=['MAPPO','MAAC'],required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--width',type=int,default=256);parser.add_argument('--environments',type=int,default=4)
    parser.add_argument('--updates',type=int,default=1000);parser.add_argument('--rollout',type=int,default=128)
    parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--minibatch',type=int,default=256)
    parser.add_argument('--replay-capacity',type=int,default=50000);parser.add_argument('--warmup',type=int,default=5000)
    parser.add_argument('--validation-every',type=int,default=10)
    parser.add_argument('--reward',choices=['source','terminal'],default='source')
    parser.add_argument('--time-cost',type=float,default=.00005)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args(argv)
    if min(args.environments,args.updates,args.rollout,args.epochs,args.minibatch,args.replay_capacity,args.validation_every)<1 or args.warmup<0 or args.time_cost<0:
        parser.error('Invalid training dimensions, warmup or time cost')
    bank=json.loads(Path(args.tasks).read_text());validation=json.loads(Path(args.validation).read_text())
    validate_task_banks(('train',bank),('validation',validation))
    registry=LayoutRegistry(args.registry)
    for subset in (bank,validation):
        for task in subset['tasks']:
            record=registry.data['families'].get(task['geometry_family_id'])
            if not record or record['split']!=subset['role']:raise ValueError('Task split differs from registered layout')
    sizes={x['defenders'] for x in bank['tasks']}
    if len(sizes)!=1 or min(sizes)<3:raise ValueError('Train one team size (at least three); evaluate cross-team generalization separately')
    physical=Protocol(**bank['protocol']);out=Path(args.out);out.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed+71)
    config=dict(algorithm=args.algorithm,width=args.width,seed=args.seed,observation=OBSERVATION,
        action='direct normalized two thrusters mapped to[-500,1000]N',reward=args.reward,time_cost=args.time_cost)
    parameters={k:v for k,v in vars(args).items() if k not in ('resume','out','updates','device')}
    contract=immutable_run_contract(dict(arguments=parameters,config=config,protocol=asdict(physical)),
        files=[args.tasks,args.validation,args.registry]+[Path(args.project)/'train'/p
            for p in ('envs/TADgame.py','envs/modules.py','policy/networks.py')],code=[__file__])
    agent=make_agent(config,args.device);agent.metadata=dict(config=config)
    worlds=[];active=[];episode_count=steps=update=0;best=None;replay=Replay(args.replay_capacity)
    def reset(index):
        nonlocal episode_count
        base=bank['tasks'][int(rng.integers(len(bank['tasks'])))];future=int(rng.integers(100000000,2000000000))
        world=World(args.project,physical,base['defenders']);world.reset(base['scene_seed'],future,base['agility'])
        if world.initial_identity['fingerprint']!=base['geometry_family_id']:raise ValueError('Training layout differs from allocation')
        worlds[index]=world;active[index]=dict(task=dict(base,future_seed=future,allocated_task_id=base['task_id']),
            episode=episode_count,steps=0,returns=0.);episode_count+=1
    source_modules(str(Path(args.project).resolve()))
    if args.resume:
        payload=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        require_same_contract(payload.get('contract'),contract,'Public baseline resume')
        restore_agent(agent,payload['state']);rng.bit_generator.state=payload['numpy_rng']
        torch.set_rng_state(payload['torch_rng'])
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all(payload['cuda_rng'])
        worlds,active=pickle.loads(payload['worlds']);replay.restore(payload['replay'])
        update,steps,episode_count,best=payload['update'],payload['environment_steps'],payload['allocated_episodes'],payload['best']
        if args.updates<update:raise ValueError('Requested training endpoint precedes the restored checkpoint')
        for name in ('episodes.jsonl','learning.jsonl'):
            read_journal(out/name,committed_update=update)
    else:
        worlds=[None]*args.environments;active=[None]*args.environments
        for index in range(args.environments):reset(index)
        atomic_json(dict(config=config,protocol=asdict(physical),arguments=vars(args),contract=contract,
            selection='fixed validation bank: success,collision,capped capture time,earlier',
            algorithms='shared decentralized actors; MAPPO centralized value or MAAC attention Q; no predictive CITF module'),out/'protocol.json')
    def save(name):
        atomic_save(dict(schema='citf-public-baseline-v1',config=config,physical_protocol=asdict(physical),contract=contract,
            state=agent_state(agent),update=update,environment_steps=steps,allocated_episodes=episode_count,best=best,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
            worlds=pickle.dumps((worlds,active),protocol=pickle.HIGHEST_PROTOCOL),replay=replay.state()),out/name)
    started=time.time()
    with (out/'episodes.jsonl').open('a' if args.resume else 'w',encoding='utf-8') as episode_log, \
            (out/'learning.jsonl').open('a' if args.resume else 'w',encoding='utf-8') as learning_log:
        if not args.resume:save('last.pt')
        for update in range(update+1,args.updates+1):
            trajectory=[[] for _ in range(6)];metrics=[]
            for tick in range(args.rollout):
                obs=np.stack([observation(w) for w in worlds])
                if args.algorithm=='MAPPO':action,latent,logp,value=agent.act(obs)
                elif steps<args.warmup:action=rng.uniform(-1.,1.,size=(len(worlds),next(iter(sizes)),2)).astype(np.float32)
                else:action=agent.act(obs)
                rewards=[];done=[];following=[]
                for index,world in enumerate(worlds):
                    result=world.step(np.clip(250+750*action[index],-500,1000));following.append(observation(world))
                    reward=(world.last_reward if args.reward=='source' else
                        np.full(world.env.defender_num,(1. if result['success'] else -1.)*world.done-args.time_cost,np.float32))
                    rewards.append(reward);done.append(float(world.done));active[index]['steps']+=1
                    active[index]['returns']+=float(np.mean(reward))
                    if world.done:
                        episode_log.write(json.dumps(dict(update=update,**active[index],**result))+'\n');reset(index)
                rewards=np.stack(rewards);done=np.asarray(done,np.float32);steps+=len(worlds)
                if args.algorithm=='MAPPO':
                    for series,x in zip(trajectory,(obs,latent,logp,value,rewards,done)):series.append(x)
                else:
                    replay.add((obs,action,rewards,np.stack(following),done))
                    if steps>=args.warmup and replay.size>=args.minibatch:metrics.append(agent.update(replay.batch(rng,args.minibatch)))
            if args.algorithm=='MAPPO':
                bootstrap=agent.act(np.stack([observation(w) for w in worlds]),deterministic=True)[3]
                metrics.append(agent.update([np.asarray(x) for x in trajectory]+[bootstrap],args.epochs,args.minibatch))
            report=dict(update=update,environment_steps=steps,
                losses={key:float(np.mean([row[key] for row in metrics])) for key in metrics[0]} if metrics else {},
                elapsed_seconds=time.time()-started)
            selected=False
            if update%args.validation_every==0 or update==args.updates:
                result=validate(agent,args.project,physical,validation)
                score=[result['success'],-result['collision'],-result['capped_capture_time'],-update]
                report['validation']={k:v for k,v in result.items() if k!='tasks'}
                atomic_json(dict(update=update,**result),out/f'validation_{update:06d}.json')
                if best is None or tuple(score)>tuple(best['score']):best=dict(update=update,score=score);selected=True
            episode_log.flush();learning_log.write(json.dumps(report)+'\n');learning_log.flush()
            if selected:save('best.pt')
            save('last.pt')
            atomic_json(dict(completed=update==args.updates,best=best,**report),out/'status.json')
            print(json.dumps(report),flush=True)
    save('endpoint.pt')
    atomic_json(dict(completed=True,algorithm=args.algorithm,updates=update,environment_steps=steps,best=best,
        protocol=asdict(physical),checkpoint='best.pt',endpoint='endpoint.pt'),out/'summary.json')


def evaluate(argv):
    import citf.evaluate as evaluation
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--checkpoint',required=True)
    args,_=parser.parse_known_args(argv)
    original=evaluation.runtime_identity
    cached={}
    class UnusedSource:
        def __init__(self,*args,**kwargs):pass
    def load(checkpoint,source,world,device):
        key=(checkpoint,str(device))
        if key not in cached:cached[key]=load_policy(checkpoint,device)
        agent=cached[key]
        trained=asdict(Protocol(**agent.metadata['physical_protocol']));actual=asdict(world.protocol)
        changed={key:dict(training=trained[key],execution=value) for key,value in actual.items() if trained[key]!=value}
        if set(changed)-{'name','apf_obstacle_rule','radial_early_failure','risk_weight','history_steps','prediction_steps','risk_clearance','prior_scale'}:
            raise ValueError('Baseline evaluation changed physical dynamics or task duration')
        agent.protocol_transfer=changed
        return agent
    def identity(*positional,**kwargs):
        value=original(*positional,**kwargs);payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        value.update(method=payload['config']['algorithm'],schema=payload['schema'],source_sha256=None,
            history_features=[],context_features=[],observation=OBSERVATION)
        transfer=value['controller'].get('initialization_protocol_transfer',{})
        value['controller']=dict(algorithm=payload['config']['algorithm'],action=payload['config']['action'],
            observation=OBSERVATION,source_used_online=False,teacher_used_online=False,
            training_environment_steps=payload['environment_steps'],training_update=payload['update'],
            reward=payload['config']['reward'],initialization_protocol_transfer=transfer,
            policy='deterministic tanh mean; shared decentralized actors')
        return value
    evaluation.Source=UnusedSource;evaluation.Controller=PublicController
    evaluation.load_behavior=load;evaluation.runtime_identity=identity
    if '--source' not in argv and not any(x.startswith('--source=') for x in argv):argv=[*argv,'--source',args.checkpoint]
    sys.argv=[sys.argv[0],*argv];evaluation.main()


def main():
    if len(sys.argv)<2 or sys.argv[1] not in ('train','evaluate'):
        raise SystemExit('Usage: python -m experiments.public_baselines {train|evaluate} ...')
    (train if sys.argv[1]=='train' else evaluate)(sys.argv[2:])


if __name__=='__main__':main()
