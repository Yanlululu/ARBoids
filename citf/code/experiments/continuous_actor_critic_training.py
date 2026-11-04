"""On-policy whole-task learning in the full common thrust domain.

Each rollout sample is a joint action that was actually executed. A centralized
critic assigns credit across the complete task. Old completed training episodes
warm the value head; actor means initialize at the strong feedback controller.
"""
import argparse
from dataclasses import asdict
import datetime
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from citf.bridge import Source,World
from citf.contracts import Protocol
from citf.train import atomic_save
from experiments.contact_event_policy import retained_teacher
from experiments.continuous_actor_critic import ContinuousConfig,ContinuousActorCritic,observed_history
from experiments.interaction_trajectories import atomic_json
from experiments.synchronized_configuration_policy import joint


def environment_worker(connection,arguments,index,restored=None):
    try:
        torch.set_num_threads(1)
        source=Source(arguments['project'],arguments['source'])
        teacher=joint.JointPolicyController(source,retained_teacher(arguments['teacher']),bank=12,budget=96,rounds=1)
        bank=json.loads(Path(arguments['tasks']).read_text())
        protocol=Protocol(**bank['protocol']);tasks=bank['tasks']
        rng=np.random.default_rng(arguments['seed']*1000+index)
        world=None;episode=0;steps=0;episode_return=0.;current_task=None
        def reset():
            nonlocal world,episode,steps,episode_return,current_task
            base=tasks[int(rng.integers(len(tasks)))];future=int(rng.integers(100000000,2000000000))
            current_task=dict(base,allocated_task_id=base['task_id'],future_seed=future,
                task_id=f"actor{arguments['seed']}-worker{index}-episode{episode}")
            world=World(arguments['project'],protocol,base['defenders'])
            world.reset(base['scene_seed'],future,base['agility']);episode+=1;steps=0;episode_return=0.
        def observation():
            reference,_=teacher.act(world)
            return dict(history=observed_history(world),remaining=protocol.time_limit-world.env.Current_T,
                reference=np.asarray(reference,np.float32))
        if restored is None:reset()
        else:
            world,episode,steps,episode_return,current_task,state=pickle.loads(restored)
            rng.bit_generator.state=state
        current=observation();connection.send(('ready',current))
        while True:
            command,payload=connection.recv()
            if command=='close':break
            if command=='snapshot':
                connection.send(('snapshot',pickle.dumps((world,episode,steps,episode_return,current_task,
                    rng.bit_generator.state),protocol=pickle.HIGHEST_PROTOCOL)))
                continue
            if command!='step':raise ValueError('Unknown environment command')
            force=np.clip(current['reference']+arguments['residual_scale']*np.tanh(payload),-500.,1000.)
            result=world.step(force);steps+=1
            reward=-arguments['time_cost']
            if world.done:reward+=1. if result['success'] else -1.
            episode_return+=reward;done=world.done;completed=None
            if done:
                completed=dict(task=current_task,worker=index,steps=steps,episode_return=episode_return,**result)
                reset()
            current=observation()
            connection.send(('transition',dict(observation=current,reward=reward,done=done,episode=completed)))
    except BaseException:
        try:connection.send(('error',traceback.format_exc()))
        except BaseException:pass
    finally:connection.close()


class Environments:
    def __init__(self,args,restored=None):
        context=mp.get_context('spawn');self.processes=[];self.connections=[]
        for index in range(args.workers):
            parent,child=context.Pipe()
            process=context.Process(target=environment_worker,args=(child,vars(args),index,
                None if restored is None else restored[index]))
            process.start();child.close();self.processes.append(process);self.connections.append(parent)
        self.current=[self.receive(pipe,'ready') for pipe in self.connections]

    @staticmethod
    def receive(pipe,expected):
        kind,value=pipe.recv()
        if kind=='error':raise RuntimeError(value)
        if kind!=expected:raise RuntimeError('Unexpected worker response:'+kind)
        return value

    def step(self,latent):
        for pipe,action in zip(self.connections,latent):pipe.send(('step',action))
        results=[self.receive(pipe,'transition') for pipe in self.connections]
        self.current=[x['observation'] for x in results]
        return results

    def snapshot(self):
        for pipe in self.connections:pipe.send(('snapshot',None))
        return [self.receive(pipe,'snapshot') for pipe in self.connections]

    def close(self):
        for pipe in self.connections:
            try:pipe.send(('close',None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join(timeout=5)
        for pipe in self.connections:pipe.close()


class ValueEpisodes:
    def __init__(self,directory):
        directory=Path(directory)
        protocol=json.loads((directory/'protocol.json').read_text())
        summary=json.loads((directory/'summary.json').read_text())
        if protocol['role']!='train' or not summary['completed']:
            raise ValueError('Value initialization requires completed training-only trajectories')
        rows=[json.loads(x) for x in (directory/'episodes.jsonl').read_text().splitlines()]
        self.data=[];self.families=set()
        for row in rows:
            if row['behavior']!='strong_physical':continue
            with np.load(directory/'episodes'/row['episode']) as arrays:state=arrays['state'].copy()
            self.data.append((state,1. if row['result']['success'] else -1.))
            self.families.add(row['task']['geometry_family_id'])
        self.description=dict(directory=str(directory),episodes=len(self.data),layouts=len(self.families),
            actual_steps=sum(len(x[0])-1 for x in self.data),behavior='strong_physical',
            use='value-head warm start from real complete returns; no validation or future input',
            collection_revision=protocol['revision'])

    def batch(self,rng,count,device,time_cost):
        histories=[];remaining=[];returns=[]
        for _ in range(count):
            state,reward=self.data[int(rng.integers(len(self.data)))];length=len(state)-1
            tick=int(rng.integers(length));indices=np.maximum(0,np.arange(tick-7,tick+1))
            histories.append(state[indices]);remaining.append(80.-.2*tick)
            returns.append(reward-time_cost*(length-tick))
        return (torch.as_tensor(np.stack(histories),device=device),
            torch.tensor(remaining,dtype=torch.float32,device=device),
            torch.tensor(returns,dtype=torch.float32,device=device))


def observation_batch(current,device):
    return tuple(torch.as_tensor(np.stack([x[key] for x in current]),dtype=torch.float32,device=device)
        for key in ['history','remaining','reference'])


def update_policy(model,optimizer,rollout,bootstrap,args,rng):
    reward=np.asarray(rollout['reward'],np.float32);done=np.asarray(rollout['done'],np.float32)
    old_value=np.asarray(rollout['value'],np.float32)
    advantage=np.zeros_like(reward);following=bootstrap;tail=np.zeros_like(bootstrap)
    for tick in reversed(range(len(reward))):
        alive=1.-done[tick]
        delta=reward[tick]+alive*following-old_value[tick]
        tail=delta+args.gae_lambda*alive*tail;advantage[tick]=tail;following=old_value[tick]
    returns=advantage+old_value
    advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
    def tensor(value):
        array=np.asarray(value)
        return torch.as_tensor(array.reshape(-1,*array.shape[2:]),dtype=torch.float32,device=args.device)
    data={key:tensor(rollout[key]) for key in ['history','remaining','reference','latent','logp','value']}
    data.update(advantage=tensor(advantage),returns=tensor(returns))
    records=[];early=False
    for epoch in range(args.epochs):
        order=rng.permutation(len(data['returns']))
        for start in range(0,len(order),args.minibatch):
            indices=order[start:start+args.minibatch]
            distribution,value=model.distribution(data['history'][indices],data['remaining'][indices],data['reference'][indices])
            # The latent Gaussian is the stochastic policy; the deterministic
            # tanh and hardware clip are part of its executed action mapping.
            logp=distribution.log_prob(data['latent'][indices]).sum((-1,-2))
            delta=logp-data['logp'][indices];ratio=delta.clamp(-20,20).exp()
            kl=float(((ratio-1)-delta).mean().detach())
            if kl>args.target_kl and records:early=True;break
            adv=data['advantage'][indices]
            policy=-torch.minimum(ratio*adv,ratio.clamp(1-args.clip,1+args.clip)*adv).mean()
            clipped=data['value'][indices]+(value-data['value'][indices]).clamp(-.2,.2)
            value_loss=.5*torch.maximum((value-data['returns'][indices]).square(),
                (clipped-data['returns'][indices]).square()).mean()
            loss=policy+.5*value_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite actual actor-critic training objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            with torch.no_grad():model.log_std.clamp_(-5.,-1.)
            records.append(dict(policy=float(policy.detach()),value_loss=float(value_loss.detach()),
                kl=kl,gradient=float(gradient),clip_fraction=float((abs(ratio-1)>args.clip).float().mean().detach())))
        if early:break
    predicted=data['value'];target=data['returns']
    report={key:float(np.mean([x[key] for x in records])) for key in records[0]}
    report.update(minibatch_updates=len(records),kl_stopped=early,
        value_explained_variance=float((1-(target-predicted).var()/target.var().clamp_min(1e-8)).detach()),
        exploration_std=model.log_std.exp().detach().cpu().tolist())
    return report


def main():
    parser=argparse.ArgumentParser()
    for key in ['project','source','teacher','encoder','tasks','value-data','out']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--seed',type=int,default=9401);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--workers',type=int,default=16);parser.add_argument('--rollout',type=int,default=128)
    parser.add_argument('--updates',type=int,default=96);parser.add_argument('--stages',nargs='+',type=int,default=[16,48,96])
    parser.add_argument('--value-pretrain',type=int,default=500);parser.add_argument('--minibatch',type=int,default=512)
    parser.add_argument('--epochs',type=int,default=4);parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--encoder-lr',type=float,default=1e-5);parser.add_argument('--value-lr',type=float,default=3e-4)
    parser.add_argument('--clip',type=float,default=.2);parser.add_argument('--target-kl',type=float,default=.025)
    parser.add_argument('--gae-lambda',type=float,default=.97);parser.add_argument('--time-cost',type=float,default=.00005)
    parser.add_argument('--residual-scale',type=float,default=1500.);parser.add_argument('--initial-std',type=float,default=.06)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=args.resume)
    bank=json.loads(Path(args.tasks).read_text())
    if bank['role']!='train':raise ValueError('On-policy learning requires training layouts')
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    rng=np.random.default_rng(args.seed+41)
    cfg=ContinuousConfig(teacher_checkpoint=args.teacher,encoder_checkpoint=args.encoder,
        residual_scale=args.residual_scale,initial_std=args.initial_std)
    model=ContinuousActorCritic(cfg).to(args.device);model.initialize_encoder(args.encoder)
    actor_params=[*model.command.parameters(),*model.action_context.parameters(),*model.actor.parameters(),model.log_std]
    optimizer=torch.optim.Adam([dict(params=model.encoder.parameters(),lr=0.),
        dict(params=actor_params,lr=args.lr),dict(params=model.critic.parameters(),lr=args.value_lr)],eps=1e-5)
    payload=None;start_update=0;steps=0;episodes=0
    if args.resume:
        payload=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(payload['state']);optimizer.load_state_dict(payload['optimizer'])
        rng.bit_generator.state=payload['numpy_rng'];torch.set_rng_state(payload['torch_rng'])
        if args.device.startswith('cuda'):torch.cuda.set_rng_state_all(payload['cuda_rng'])
        start_update=payload['update'];steps=payload['environment_steps'];episodes=payload['episodes']
    values=ValueEpisodes(args.value_data);started=time.time()
    protocol=dict(arguments=vars(args),config=asdict(cfg),value_initialization=values.description,
        train_layouts=len({x['geometry_family_id'] for x in bank['tasks']}),
        action='reference force+1500*tanh(joint latent),clipped only to common[-500,1000]N hardware domain',
        goal='complete task success first; terminal+1/-1,per actual control step cost0.00005,gamma1',
        learning='on-policy joint12dim GaussianPPO and centralized state value,GAE0.97',
        zero_mean='initial greedy action exactly retains strong reference; no activation threshold',
        scopes='fixed training layout pool with fresh episode futures; source/teacher/dynamics-encoder pretraining costs retained; action domain expanded',
        selection='declared16/48/96 full independent256 evaluations,success/collision/cappedtime/earlier; common768 and repeats according to potential',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if not args.resume:atomic_json(protocol,out/'protocol.json')
    envs=None
    try:
        with (out/'learning.jsonl').open('a' if args.resume else 'w') as log,(out/'episodes.jsonl').open('a' if args.resume else 'w') as episode_log:
            if not args.resume:
                for iteration in range(1,args.value_pretrain+1):
                    history,remaining,target=values.batch(rng,args.minibatch,args.device,args.time_cost)
                    loss=F.mse_loss(model(history,remaining)['value'],target)
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.critic.parameters(),1.);optimizer.step()
                    if iteration%100==0:
                        row=dict(phase='value_initialization',update=iteration,loss=float(loss.detach()))
                        log.write(json.dumps(row)+'\n');log.flush();atomic_json(dict(completed=False,**row),out/'status.json')
                        print(json.dumps(row),flush=True)
            optimizer.param_groups[0]['lr']=args.encoder_lr
            envs=Environments(args,None if payload is None else payload['workers'])
            def checkpoint(update):
                saved=dict(config=asdict(cfg),state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(),update=update,environment_steps=steps,episodes=episodes,
                    workers=envs.snapshot(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],protocol=protocol)
                atomic_save(saved,out/f'stage_{update:06d}.pt');atomic_save(saved,out/'last.pt')
            if not args.resume:checkpoint(0)
            for update in range(start_update+1,args.updates+1):
                rollout={key:[] for key in ['history','remaining','reference','latent','logp','value','reward','done']}
                recent=[];rollout_started=time.time();model.eval()
                for tick in range(args.rollout):
                    history,remaining,reference=observation_batch(envs.current,args.device)
                    with torch.no_grad():
                        distribution,value=model.distribution(history,remaining,reference)
                        latent=distribution.sample();logp=distribution.log_prob(latent).sum((-1,-2))
                    for key,item in [('history',history),('remaining',remaining),('reference',reference),
                        ('latent',latent),('logp',logp),('value',value)]:rollout[key].append(item.cpu().numpy())
                    transition=envs.step(latent.cpu().numpy())
                    rollout['reward'].append([x['reward'] for x in transition]);rollout['done'].append([x['done'] for x in transition])
                    for item in transition:
                        if item['episode'] is not None:
                            row=item['episode'];episodes+=1;recent.append(row)
                            episode_log.write(json.dumps(dict(update=update,**row))+'\n')
                episode_log.flush();steps+=args.workers*args.rollout
                with torch.no_grad():
                    history,remaining,reference=observation_batch(envs.current,args.device)
                    bootstrap=model(history,remaining)['value'].cpu().numpy()
                rollout_seconds=time.time()-rollout_started;model.train()
                detail=update_policy(model,optimizer,rollout,bootstrap,args,rng)
                row=dict(phase='on_policy',update=update,total_updates=args.updates,environment_steps=steps,
                    completed_episodes=episodes,rollout_completed_episodes=len(recent),
                    rollout_success=sum(x['success'] for x in recent),rollout_collision=sum(x['collision'] for x in recent),
                    rollout_radial=sum(x['radial_only_failure'] for x in recent),
                    rollout_mean_return=float(np.mean([x['episode_return'] for x in recent])) if recent else None,
                    rollout_steps_per_second=args.workers*args.rollout/rollout_seconds,
                    elapsed_seconds=time.time()-started,**detail)
                log.write(json.dumps(row)+'\n');log.flush();atomic_json(dict(completed=False,**row),out/'status.json')
                print(json.dumps(row),flush=True)
                if update in args.stages:checkpoint(update)
            report=dict(completed=True,updates=args.updates,environment_steps=steps,completed_episodes=episodes,
                training_layouts=protocol['train_layouts'],elapsed_seconds=time.time()-started,
                closed_loop_evaluation='declared checkpoints pending independent complete256 evaluation')
            atomic_json(report,out/'summary.json');atomic_json(report,out/'status.json')
    finally:
        if envs is not None:envs.close()


if __name__=='__main__':main()
