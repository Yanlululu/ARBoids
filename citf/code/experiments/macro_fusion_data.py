"""Paired complete-task labels for short joint fusion decisions in all phases.

This is training data, not a value model or an online oracle. Three candidates
share eight future draws at each prospectively timed observed root. The short
candidate is executed for three control periods, followed by the frozen strong
joint controller. All returns, adverse outcomes and short state labels survive.
"""
import argparse
import concurrent.futures
import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.bridge import Source,World
from citf.contracts import Protocol
from experiments.contact_event_policy import retained_teacher
from experiments.interception_fusion import InterceptionFusionController
from experiments.interaction_dynamics import load as load_dynamics
from experiments.interaction_trajectories import atomic_json
from experiments.synchronized_configuration_policy import joint


WORKER={}


def initialize(project,source,teacher,dynamics,out,futures,root_ticks):
    torch.set_num_threads(1)
    source_model=Source(project,source)
    teacher_model=retained_teacher(teacher)
    reference=joint.JointPolicyController(source_model,teacher_model,bank=12,budget=96,rounds=1)
    proposed=InterceptionFusionController(source_model,load_dynamics(dynamics),budget=96,rounds=1)
    WORKER.update(project=project,source=source_model,teacher=reference,proposed=proposed,
        out=Path(out),futures=futures,root_ticks=root_ticks)


@torch.no_grad()
def branch_rollout(root,factor,plan,future_seed,reference=False):
    world=root.clone();world.rng=np.random.RandomState(future_seed)
    states=[world.state().copy()];commands=[];attack=[];events=[];steps=0
    while not world.done:
        if not reference and steps<3:
            current=dict(factor)
            current['learned'],current['rules'],current['prior']=WORKER['source'].commands(world.env)
            value=torch.as_tensor(world.state(),dtype=torch.float32)[None]
            if steps:
                current['intercept_target']=value[0,-1,:2][None]+current['lead_seconds'][:,None]*value[0,-1,3:5][None]
            force=WORKER['proposed'].forces(value,value[...,3:5],current,
                torch.as_tensor(plan,dtype=torch.long)[None],0)[0].numpy()
        else:
            force,_=WORKER['teacher'].act(world)
        result=world.step(force)
        if steps<16:
            states.append(world.state().copy());commands.append(force)
            attack.append(world.env.action_to_thrust(world.env.att_action,world.env.attacker.agility))
            events.append([result[k] for k in ('capture','collision','radial_only_failure','physical_breach','timeout')])
        steps+=1
    duration=result['duration'] if result['capture'] else 80.
    reward=(1. if result['success'] else -1.)-.25*result['collision']+.02*result['capture']-.0005*duration
    return dict(result=result,reward=reward,steps=steps,states=np.asarray(states,np.float32),
        command=np.asarray(commands,np.float32),attacker=np.asarray(attack,np.float32),events=np.asarray(events,np.uint8))


@torch.no_grad()
def collect(job):
    index,task,protocol=job
    world=World(WORKER['project'],Protocol(**protocol),task['defenders'])
    world.reset(task['scene_seed'],task['future_seed'],task['agility'])
    rows=[];tick=0
    while not world.done:
        force,teacher_trace=WORKER['teacher'].act(world)
        if tick in WORKER['root_ticks']:
            ordinal=WORKER['root_ticks'].index(tick);root_id=f'{index:04d}_{ordinal:02d}'
            meta=WORKER['out']/'roots'/(root_id+'.json')
            if meta.exists():
                rows.append(json.loads(meta.read_text()))
            else:
                WORKER['proposed'].cached=None
                _,prediction_trace=WORKER['proposed'].act(world)
                factor=WORKER['proposed'].last_factor
                proposed=np.asarray(prediction_trace['selected_indices'],np.int64)
                original=np.asarray(teacher_trace['selected_indices'],np.int64)
                alternatives=np.asarray(teacher_trace['configurations'],np.int64)
                ranking=np.argsort(np.asarray(teacher_trace['scores']),kind='stable')
                alternative=next((alternatives[k] for k in ranking
                    if not np.array_equal(alternatives[k],original) and not np.array_equal(alternatives[k],proposed)),original)
                plans=[original,proposed,alternative]
                future_seeds=[87051000+index*1000+ordinal*100+i for i in range(WORKER['futures'])]
                rollout=[]
                for action,plan in enumerate(plans):
                    rollout.append([branch_rollout(world,factor,plan,seed,reference=action==0) for seed in future_seeds])
                history=np.stack(world.history).astype(np.float32)
                history=np.concatenate((np.repeat(history[:1],8-len(history),0),history),0)
                shape=(3,WORKER['futures']);entities=task['defenders']+1
                state=np.zeros((*shape,17,entities,6),np.float32)
                command=np.zeros((*shape,16,entities-1,2),np.float32)
                attacker=np.zeros((*shape,16,2),np.float32)
                event=np.zeros((*shape,16,5),np.uint8);valid=np.zeros((*shape,16),bool)
                reward=np.zeros(shape,np.float32);endpoints=[]
                for action,group in enumerate(rollout):
                    for future,result in enumerate(group):
                        length=len(result['command']);state[action,future,:length+1]=result['states']
                        state[action,future,length+1:]=result['states'][-1]
                        command[action,future,:length]=result['command'];attacker[action,future,:length]=result['attacker']
                        event[action,future,:length]=result['events'];valid[action,future,:length]=True
                        reward[action,future]=result['reward']
                        endpoints.append(dict(action=action,future_replica=future,future_seed=future_seeds[future],
                            environment_steps=result['steps'],reward=result['reward'],**result['result']))
                path=meta.with_suffix('.npz');pending=path.with_suffix('.pending')
                with pending.open('wb') as handle:
                    np.savez_compressed(handle,history=history,state=state,command=command,
                        attacker_thrust_label=attacker,events=event,valid=valid,reward=reward,
                        queried_plans=np.asarray(plans),all_plans=np.asarray(prediction_trace['configurations']),
                        all_scores=np.asarray(prediction_trace['scores']),
                        learned_source=factor['learned'].numpy(),boids_source=factor['rules'].numpy(),
                        grid=factor['grid'].numpy(),prior=factor['prior'].numpy(),
                        lead_seconds=factor['lead_seconds'].numpy(),intercept_target=factor['intercept_target'].numpy())
                pending.replace(path)
                row=dict(root_id=root_id,task=task,elapsed=float(world.env.Current_T),tick=tick,
                    futures=WORKER['futures'],candidate_count=len(prediction_trace['configurations']),
                    queried_actions=['reference teacher feedback','predicted winning fusion3ticks','leading distinct teacher-bank fusion3ticks'],
                    queried_plans=np.asarray(plans).tolist(),mean_returns=reward.mean(-1).tolist(),
                    paired_return_differences=(reward[1:]-reward[:1]).tolist(),
                    branch_environment_steps=sum(x['environment_steps'] for x in endpoints),endpoints=endpoints,
                    prefix='frozen717 joint controller on an existing training layout',
                    continuation='same frozen717 controller after the3tick decision',
                    online_information='observations only; future draws introduced after candidate selection',
                    short_labels='first16actual steps, through real termination; no padded steps counted')
                atomic_json(row,meta);rows.append(row)
        world.step(force);tick+=1
        if tick>max(WORKER['root_ticks']):break
    return dict(task=task,roots=rows,prefix_steps=tick,requested_roots=len(WORKER['root_ticks']))


def main():
    parser=argparse.ArgumentParser()
    for key in ('project','source','teacher','dynamics','tasks','out'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--layouts',type=int,default=16)
    parser.add_argument('--layout-offset',type=int,default=0)
    parser.add_argument('--futures',type=int,default=8)
    parser.add_argument('--root-times',type=float,nargs='+',default=[4.,10.,16.])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=args.resume)
    (out/'roots').mkdir(exist_ok=args.resume)
    bank=json.loads(Path(args.tasks).read_text());families={}
    if bank['role']!='train':raise ValueError('This policy-label collection requires the allocated training pool')
    for task in bank['tasks']:families.setdefault(task['geometry_family_id'],[]).append(task)
    allocated=list(families.values())[args.layout_offset:args.layout_offset+args.layouts]
    tasks=[group[(i+args.layout_offset)%len(group)] for i,group in enumerate(allocated)]
    ticks=[int(round(x/.2)) for x in args.root_times]
    protocol=dict(arguments=vars(args),role='training pilot, all returned branches retained',
        independent_layouts=len(tasks),maximum_roots=len(tasks)*len(ticks),
        declared_future_count=args.futures,decision_duration_ticks=3,
        source_domain='original SAC/Boids fusion only',
        terminal_returns='policy-comparison labels only, no CV/value/success head',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if not args.resume:atomic_json(protocol,out/'protocol.json')
    rows=[];started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
        mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
        initargs=(args.project,args.source,args.teacher,args.dynamics,str(out),args.futures,ticks)) as executor:
        jobs=[executor.submit(collect,(i+args.layout_offset,task,bank['protocol'])) for i,task in enumerate(tasks)]
        for completed in concurrent.futures.as_completed(jobs):
            rows.append(completed.result());roots=[root for row in rows for root in row['roots']]
            status=dict(completed=False,layouts=len(rows),expected_layouts=len(tasks),roots=len(roots),
                complete_branches=len(roots)*3*args.futures,branch_steps=sum(x['branch_environment_steps'] for x in roots),
                elapsed_seconds=time.time()-started)
            atomic_json(status,out/'status.json');print(json.dumps(status),flush=True)
    roots=sorted([root for row in rows for root in row['roots']],key=lambda x:x['root_id'])
    (out/'roots.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in roots))
    report=dict(completed=True,layouts=len(tasks),roots=len(roots),requested_roots=len(tasks)*len(ticks),
        futures=args.futures,complete_branches=len(roots)*3*args.futures,
        branch_environment_steps=sum(x['branch_environment_steps'] for x in roots),
        elapsed_seconds=time.time()-started,scope='training roots with common futures; not independent complete-bank performance')
    atomic_json(report,out/'summary.json');atomic_json(report,out/'status.json');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
