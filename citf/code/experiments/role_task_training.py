"""Whole episodes train the same outcome readout used for joint role selection."""
import argparse
import concurrent.futures
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from citf.bridge import Source, World
from citf.contracts import Protocol, validate_task_banks
from citf.evaluate import atomic_json
from experiments.contact_event_policy import retained_teacher
from experiments.fusion_capture_data import save_tensor
from experiments.fusion_capture_train import targets, hurdle_nll
from experiments.role_task_policy import (RoleConfig, RoleOutcomeTransformer,
    RoleTaskController, load_role_policy, policy_probabilities, joint)


WORKER = {}


def initialize(arguments, physical):
    torch.set_num_threads(1)
    source=Source(arguments['project'],arguments['source'])
    teacher=joint.JointPolicyController(source,retained_teacher(arguments['teacher']),'cpu',12,96,1)
    WORKER.update(arguments=arguments,physical=physical,source=source,teacher=teacher,checkpoint=None)


def finish_interval(record, positions, commands, events, cfg):
    length=len(positions);xy=np.stack(positions,axis=-2)
    if length<cfg.horizon:
        xy=np.concatenate((xy,np.repeat(xy[...,-1:,:],cfg.horizon-length,axis=-2)),axis=-2)
    valid=torch.arange(cfg.horizon)<length
    record.update(real_positions=torch.from_numpy(xy.astype(np.float32)),time_valid=valid,
                  actual_commands=np.asarray(commands,dtype=np.float32),events=events)
    record['target']=targets(dict(real_positions=record['real_positions'][None,None],
        time_valid=valid[None,None],batch=record['batch']),cfg)
    return record


def episode(item):
    checkpoint,task,episode_id,epsilon=item;a=WORKER['arguments']
    destination=Path(a['out'])/'episodes'/f'{episode_id:06d}.pt'
    if destination.exists():
        saved=torch.load(destination,map_location='cpu',weights_only=False)
        if saved['task']!=task or saved['checkpoint']!=checkpoint or saved['epsilon']!=epsilon:
            raise ValueError('Saved episode differs from the resumed collection assignment')
        return str(destination)
    if WORKER['checkpoint']!=checkpoint:
        WORKER.update(model=load_role_policy(checkpoint),checkpoint=checkpoint)
    model=WORKER['model'];cfg=model.cfg
    world=World(a['project'],Protocol(**WORKER['physical']),task['defenders'])
    world.reset(task['scene_seed'],task['future_seed'],task['agility'])
    if world.initial_identity['fingerprint']!=task['geometry_family_id']:
        raise ValueError('Training episode differs from its allocated geometry')
    controller=RoleTaskController(WORKER['source'],model,'cpu',12,96,1,epsilon=epsilon,
        sampling_seed=a['seed']*100000+episode_id,teacher=WORKER['teacher'])
    records=[];record=None;steps=0;started=time.time()
    while not world.done:
        force,trace=controller.act(world)
        if trace['replanned']:
            if record is not None:records.append(finish_interval(record,positions,commands,events,cfg))
            record=dict(batch=controller.last_batch,action=controller.last_choice,
                old_probabilities=torch.as_tensor(controller.last_probabilities.copy()),
                old_value=controller.last_value,epsilon=epsilon,elapsed=float(world.env.Current_T))
            positions=[];commands=[];events=[];rotation=controller.last_rotation
        result=world.step(force);steps+=1
        positions.append(world.state()[:,:2]@rotation);commands.append(force)
        events.append([result['capture'],result['physical_collision'],result['physical_breach']])
    records.append(finish_interval(record,positions,commands,events,cfg))
    category=0 if result['capture'] else (1 if result['success'] else 2)
    capture_time=result['duration']/world.protocol.time_limit
    reward=4+.2*(1-capture_time) if category==0 else (3. if category==1 else -12.)
    for record in records:record.update(category=category,reward=reward,capture_time=capture_time)
    save_tensor(dict(task=task,episode_id=episode_id,checkpoint=checkpoint,epsilon=epsilon,
                steps=steps,seconds=time.time()-started,result=result,reward=reward,records=records),destination)
    return str(destination)


def minibatch(records,indices,device):
    rows=[records[int(i)] for i in indices]
    batch={key:torch.stack([row['batch'][key] for row in rows]).to(device) for key in rows[0]['batch']}
    truth={key:torch.stack([row['target'][key] for row in rows]).to(device) for key in rows[0]['target']}
    scalar={key:torch.as_tensor([row[key] for row in rows],device=device) for key in
            ['action','old_value','epsilon','category','reward','capture_time','advantage']}
    scalar['old_probabilities']=torch.stack([row['old_probabilities'] for row in rows]).to(device)
    return batch,truth,scalar


def loss_terms(model,batch,truth,row):
    output=model(batch);index=torch.arange(len(row['action']),device=row['action'].device)
    action=row['action'];epsilon=row['epsilon'][:,None]
    probabilities=policy_probabilities(output,model.cfg,epsilon).clamp_min(1e-9)
    old=row['old_probabilities'].clamp_min(1e-9)
    ratio=probabilities[index,action]/old[index,action]
    advantage=row['advantage']
    actor=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
    value=F.mse_loss(output['value'],row['reward'])
    outcome=F.cross_entropy(output['outcome_logits'][index,action],row['category'])
    captured=(row['category']==0)
    timing=((output['terminal_capture_time'][index,action]-row['capture_time']).square()*captured).sum()/captured.sum().clamp_min(1)
    selected={key:val[index,action][:,None] for key,val in output.items()
              if key.startswith(('pair_','team_','radius_')) or key=='xy'}
    geometry=(hurdle_nll(truth['pair'],selected,'pair',5.)
              +.5*hurdle_nll(truth['team'],selected,'team',5.)
              +.2*hurdle_nll(truth['radius'],selected,'radius',15.))
    mask=truth['xy_valid'][:,:,None,:,None].expand_as(truth['xy'])
    motion=F.smooth_l1_loss(selected['xy']/5,truth['xy']/5,reduction='none')[mask].mean()
    entropy=-(probabilities*probabilities.log()).sum(-1).mean()
    kl=(old*(old.log()-probabilities.log())).sum(-1).mean()
    loss=actor+.25*value+.5*outcome+.1*timing+.05*(geometry+.2*motion)-.005*entropy
    metrics=dict(loss=loss,actor=actor,value=value,outcome=outcome,timing=timing,
                 geometry=geometry,motion=motion,entropy=entropy,kl=kl)
    return loss,{key:float(val.detach()) for key,val in metrics.items()}


def main():
    parser=argparse.ArgumentParser()
    for name in ['project','source','teacher','tasks','initialize-from','out']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',type=int,default=9701)
    parser.add_argument('--iterations',type=int,default=24)
    parser.add_argument('--episodes-per-iteration',type=int,default=32)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--minibatch',type=int,default=48)
    parser.add_argument('--learning-rate',type=float,default=2e-4)
    parser.add_argument('--future-base',type=int,default=95051000)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    out=Path(args.out);bank=json.loads(Path(args.tasks).read_text())
    validate_task_banks(('train',bank))
    if bank['role']!='train':raise ValueError('Only allocated training layouts may enter learning')
    if out.exists() and not args.resume:raise ValueError('Refuse to overwrite an existing training run')
    out.mkdir(parents=True,exist_ok=True);(out/'rollouts').mkdir(exist_ok=True);(out/'episodes').mkdir(exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    physical=asdict(Protocol(**bank['protocol']));cfg=RoleConfig(teacher_checkpoint=args.teacher)
    model=RoleOutcomeTransformer(cfg).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    history=[];start=0;total_steps=0;started=time.time()
    protocol=dict(arguments={k:v for k,v in vars(args).items() if k!='resume'},configuration=asdict(cfg),
        physical_protocol=physical,task_bank=args.tasks,task_count=len(bank['tasks']),
        effective_layouts=len({task['geometry_family_id'] for task in bank['tasks']}),
        reward='capture:4+0.2*(1-duration/80);timeout:3;failure:-12',
        credit='complete episode Monte Carlo return, gamma1; on-policy clipped PPO, outcome likelihood, actual selected-role stopped-distance supervision',
        model_selection='initial and fixed final checkpoint on all256 independent controller-selection tasks; expand only with retained defense and improved capture',
        initialization='distance Transformer transferred; live-feedback role candidates use fresh selected-option supervision; new terminal-outcome head starts at declared guard-retention prior')
    if args.resume:
        saved=json.loads((out/'protocol.json').read_text())
        if {k:v for k,v in saved.items() if k!='transferred_missing_parameters'}!=protocol:
            raise ValueError('Resume arguments differ from the recorded training run')
        protocol=saved
        payload=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(payload['state']);optimizer.load_state_dict(payload['optimizer'])
        history=payload['history'];start=payload['iteration'];total_steps=payload['environment_steps']
        random.setstate(payload['python_rng']);np.random.set_state(payload['numpy_rng'])
        torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
    else:
        base=torch.load(args.initialize_from,map_location='cpu',weights_only=False)
        missing=model.load_state_dict(base['state'],strict=False)
        protocol['transferred_missing_parameters']=list(missing.missing_keys)
        atomic_json(protocol,out/'protocol.json')
    def save(iteration):
        payload=dict(state={key:val.detach().cpu() for key,val in model.state_dict().items()},
            config=asdict(cfg),physical_protocol=physical,optimizer=optimizer.state_dict(),iteration=iteration,
            environment_steps=total_steps,episodes=iteration*args.episodes_per_iteration,history=history,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),protocol=protocol)
        save_tensor(payload,out/'latest.pt')
        if iteration in (0,8,16,args.iterations):save_tensor(payload,out/f'stage_{iteration:03d}.pt')
    if not args.resume:save(0)
    arguments=vars(args).copy()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
            initargs=(arguments,physical)) as pool:
        for iteration in range(start,args.iterations):
            begin=time.time();epsilon=.2-.15*iteration/max(args.iterations-1,1)
            # Checkpoint path includes the iteration so persistent workers reload the new behavior.
            behavior=out/f'behavior_{iteration:03d}.pt'
            save_tensor(dict(config=asdict(cfg),physical_protocol=physical,
                state={k:v.detach().cpu() for k,v in model.state_dict().items()}),behavior)
            items=[]
            for k in range(args.episodes_per_iteration):
                episode_id=iteration*args.episodes_per_iteration+k
                task=dict(bank['tasks'][np.random.randint(len(bank['tasks']))])
                task['future_seed']=args.future_base+episode_id
                items.append((str(behavior),task,episode_id,epsilon))
            atomic_json(dict(completed=False,iteration=iteration,phase='complete_episode_collection',
                completed_episodes=iteration*args.episodes_per_iteration,epsilon=epsilon),out/'status.json')
            paths=list(pool.map(episode,items))
            episodes=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
            records=[row for ep in episodes for row in ep['records']]
            atomic_json(dict(iteration=iteration+1,episode_paths=paths,physical_protocol=physical),
                        out/'rollouts'/f'iteration_{iteration+1:03d}.json')
            advantages=np.asarray([row['reward']-row['old_value'] for row in records],np.float32)
            advantages=(advantages-advantages.mean())/max(advantages.std(),1e-6)
            for record,advantage in zip(records,advantages):record['advantage']=float(advantage)
            metrics=[];stop=False;model.train()
            for epoch in range(args.epochs):
                permutation=np.random.permutation(len(records))
                for offset in range(0,len(records),args.minibatch):
                    batch,truth,scalar=minibatch(records,permutation[offset:offset+args.minibatch],args.device)
                    loss,values=loss_terms(model,batch,truth,scalar)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite whole-task learning loss')
                    if values['kl']>.03:stop=True;break
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();metrics.append(values)
                if stop:break
            model.eval();total_steps+=sum(ep['steps'] for ep in episodes)
            action_counts=np.bincount([row['action'] for row in records],minlength=9).tolist()
            result=dict(iteration=iteration+1,episodes=len(episodes),decisions=len(records),
                success=sum(ep['result']['success'] for ep in episodes),capture=sum(ep['result']['capture'] for ep in episodes),
                collision=sum(ep['result']['physical_collision'] for ep in episodes),
                breach=sum(ep['result']['physical_breach'] for ep in episodes),
                mean_reward=float(np.mean([ep['reward'] for ep in episodes])),environment_steps=total_steps,
                epsilon=epsilon,action_counts=action_counts,gradient_steps=len(metrics),kl_stopped=stop,
                metrics={key:float(np.mean([m[key] for m in metrics])) for key in metrics[0]} if metrics else {},
                seconds=time.time()-begin)
            history.append(result);save(iteration+1);atomic_json(result,out/'status.json');print(json.dumps(result),flush=True)
    summary=dict(completed=True,iterations=args.iterations,episodes=args.iterations*args.episodes_per_iteration,
        effective_layouts=len({task['geometry_family_id'] for task in bank['tasks']}),environment_steps=total_steps,
        seconds=time.time()-started,history=history,final_checkpoint=str(out/f'stage_{args.iterations:03d}.pt'))
    atomic_json(summary,out/'summary.json');atomic_json(summary,out/'status.json');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
