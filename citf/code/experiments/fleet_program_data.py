"""Common-random-future, complete-task supervision of every deployed program."""
import argparse
import concurrent.futures
from dataclasses import asdict
import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.bridge import Source, World
from citf.contracts import Protocol, validate_task_banks, immutable_run_contract, require_same_contract, source_identity
from citf.evaluate import atomic_json
from citf.opponents import load_opponent
from experiments.fusion_capture_data import save_tensor
from experiments.fleet_program_policy import (FleetConfig, FleetExecutor, PROGRAMS, coverage_parameters,
    initial_guard, make_input, load_teacher, program_bank)


WORKER={}


def initialize(arguments,configuration):
    torch.set_num_threads(1)
    source=Source(arguments['project'],arguments['source'])
    teacher=load_teacher(source,arguments['teacher'],arguments['teacher_sha256'])
    WORKER.update(arguments=arguments,source=source,teacher=teacher,
        executor=FleetExecutor(source,teacher,configuration.get('feedback_mode','opening_teacher')),cfg=FleetConfig(**configuration),
        opponent=load_opponent(arguments.get('opponent_spec')))


def task_utility(result,limit,time_weight=.2):
    return (4+time_weight*(1-result['duration']/limit) if result['capture'] else
            3. if result['timeout'] else -12.)


def unresolved(values):
    """Extend whole-bank common futures when an observed difference is uncertain."""
    values=np.asarray(values,dtype=float)  # futures,candidates
    difference=values[:,:,None]-values[:,None,:]
    mean=difference.mean(0);sem=difference.std(0,ddof=1)/np.sqrt(len(values))
    return bool(((sem>.04)&(np.abs(mean)<1.64*sem)).any())


def collect(item):
    index,task,physical,root_time,behavior_program=item;a=WORKER['arguments'];out=Path(a['out'])
    root_id=f'{index:05d}';meta=out/'roots'/(root_id+'.json')
    if meta.exists():
        row=json.loads(meta.read_text())
        if row['task']!=task:raise ValueError('Resumed fleet root differs from allocated task')
        return row
    started=time.time();cfg=WORKER['cfg'];executor=WORKER['executor'];opponent=WORKER['opponent']
    world=World(a['project'],Protocol(**physical),task['defenders'])
    world.reset(task['scene_seed'],task['future_seed'],task['agility'])
    if world.initial_identity['fingerprint']!=task['geometry_family_id']:
        raise ValueError('Fleet root layout differs from allocation')
    guard=initial_guard(world);executor.configure(coverage_parameters(world,guard));prefix_steps=0;prefix_commands=[]
    behavior_started=False
    while not world.done and world.env.Current_T+1e-7<root_time:
        if not behavior_started and world.env.Current_T+1e-7>=cfg.decision_time:
            executor.configure(program_bank(world,guard,cfg.feedback_mode)[behavior_program]);behavior_started=True
        force=executor.act(world)[0];prefix_commands.append(force)
        result=world.step(force,attacker_action=opponent.act(world));prefix_steps+=1
    if world.done:
        row=dict(root_id=root_id,task=task,reached=False,prefix_steps=prefix_steps,
            prefix_endpoint=result,branch_steps=0,futures=0,seconds=time.time()-started)
        atomic_json(row,meta);return row
    batch,parameters,first,closed,rotation,_=make_input(world,WORKER['source'],WORKER['teacher'],executor,guard,cfg)
    C=len(PROGRAMS);E=task['defenders']+1;H=cfg.horizon
    positions=[];valids=[];returns=[];classes=[];durations=[];endpoints=[];seeds=[];steps=0
    max_ticks=int(round((world.protocol.time_limit-world.env.Current_T)/world.protocol.control_dt))+1
    commands=[];events=[];full_pairs=[];full_radii=[];full_times=[];pair_i,pair_j=np.triu_indices(E,1)
    for replica in range(a['max_futures']):
        seed=a['future_base']+index*a['max_futures']+replica;seeds.append(seed)
        xy=np.zeros((C,E,H,2),np.float32);valid=np.zeros((C,H),bool)
        command_log=np.zeros((C,max_ticks,E-1,2),np.float32);event_log=np.zeros((C,max_ticks,3),bool)
        minimum=np.full((C,len(pair_i)),np.inf,np.float32);radius=np.full(C,np.inf,np.float32)
        closest=np.full(C,np.inf,np.float32);time_at=np.zeros(C,np.float32)
        utility=[];category=[];duration=[]
        for c in range(C):
            branch=world.clone();branch.rng=np.random.RandomState(seed)
            executor.configure(parameters[c],bool(closed[c]));tick=0
            while not branch.done:
                force=first[c].numpy() if tick==0 else executor.act(branch)[0]
                result=branch.step(force,attacker_action=opponent.act(branch))
                position=branch.state()[:,:2];distance=np.linalg.norm(position[pair_i]-position[pair_j],axis=-1)
                minimum[c]=np.minimum(minimum[c],distance);radius[c]=min(radius[c],np.linalg.norm(position[-1]))
                contact=float(distance[pair_j==E-1].min())
                if contact<closest[c]:closest[c]=contact;time_at[c]=result['duration']/world.protocol.time_limit
                if tick<H:
                    xy[c,:,tick]=position@rotation;valid[c,tick]=True
                command_log[c,tick]=force
                event_log[c,tick]=[result['capture'],result['physical_collision'],result['physical_breach']]
                tick+=1
            used=min(tick,H)
            xy[c,:,used:]=xy[c,:,used-1:used];steps+=tick
            utility.append(task_utility(result,world.protocol.time_limit,cfg.capture_time_weight))
            category.append(0 if result['capture'] else 1 if result['timeout'] else 2)
            duration.append(result['duration']/world.protocol.time_limit)
            endpoints.append(dict(configuration=c,future_replica=replica,future_seed=seed,steps=tick,**result))
        positions.append(xy);valids.append(valid);returns.append(utility);classes.append(category);durations.append(duration)
        commands.append(command_log);events.append(event_log)
        full_pairs.append(minimum);full_radii.append(radius);full_times.append(time_at)
        if replica+1==a['futures'] and not unresolved(returns):break
    payload=dict(batch={k:v[0] for k,v in batch.items()},
        real_positions=torch.from_numpy(np.stack(positions,1)),time_valid=torch.from_numpy(np.stack(valids,1)),
        terminal_utility=torch.tensor(np.asarray(returns).T,dtype=torch.float32),
        terminal_class=torch.tensor(np.asarray(classes).T,dtype=torch.long),
        terminal_duration=torch.tensor(np.asarray(durations).T,dtype=torch.float32),
        actual_commands=torch.from_numpy(np.stack(commands,1)),events=torch.from_numpy(np.stack(events,1)),
        prefix_commands=torch.as_tensor(np.asarray(prefix_commands),dtype=torch.float32),behavior_program=behavior_program,
        parameters=parameters,first_commands=first,initial_closed=closed,
        task=task,root_id=root_id,elapsed=world.env.Current_T,guard=guard,future_seeds=seeds,
        physical_protocol=physical,opponent_identity=opponent.identity)
    save_tensor(payload,meta.with_suffix('.pt'))
    minimum=torch.from_numpy(np.stack(full_pairs,1))
    save_tensor(dict(root_id=root_id,task=task,future_seeds=seeds,physical_protocol=physical,pair=minimum,
        team=minimum[...,pair_j==E-1].amin(-1),radius=torch.from_numpy(np.stack(full_radii,1)),
        time_at_minimum=torch.from_numpy(np.stack(full_times,1)),interval='root through actual terminal transition'),
        out/'full_future/roots'/meta.with_suffix('.pt').name)
    row=dict(root_id=root_id,task=task,reached=True,prefix_steps=prefix_steps,candidates=C,futures=len(seeds),
        branch_steps=steps,terminal_branches=C*len(seeds),endpoints=endpoints,
        root_time=root_time,behavior_program=behavior_program,
        program_mean_utility=np.mean(returns,0).tolist(),uncertain_after_sampling=unresolved(returns),seconds=time.time()-started)
    atomic_json(row,meta);return row


def main():
    parser=argparse.ArgumentParser()
    for name in ['project','source','teacher','tasks','out']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--layouts',type=int,default=128)
    parser.add_argument('--futures',type=int,default=6)
    parser.add_argument('--max-futures',type=int,default=12)
    parser.add_argument('--future-base',type=int,default=94051000)
    parser.add_argument('--decision-time',type=float,default=2.)
    parser.add_argument('--root-times',type=float,nargs='+')
    parser.add_argument('--feedback-mode',choices=['opening_teacher','persistent_interception'],default='opening_teacher')
    parser.add_argument('--capture-time-weight',type=float,default=.2)
    parser.add_argument('--opponent-spec')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=Path(args.out)
    if min(args.workers,args.layouts)<1 or args.futures<2 or args.max_futures<args.futures:
        parser.error('Positive workers/layouts and 2 <= futures <= max-futures required')
    bank=json.loads(Path(args.tasks).read_text())
    if bank['role'] not in ('train','validation'):raise ValueError('Only allocated development supervision')
    validate_task_banks((bank['role'],bank));physical=Protocol(**bank['protocol'])
    if not 0<=args.decision_time<physical.time_limit or abs(args.decision_time/physical.control_dt-round(args.decision_time/physical.control_dt))>1e-6:
        parser.error('Decision time must be a legal control tick')
    families={}
    for task in bank['tasks']:families.setdefault(task['geometry_family_id'],[]).append(task)
    tasks=[group[i%len(group)] for i,group in enumerate(list(families.values())[:args.layouts])]
    root_times=args.root_times or [args.decision_time]
    if len(set(root_times))!=len(root_times) or any(t<args.decision_time or t>=physical.time_limit
            or abs(t/physical.control_dt-round(t/physical.control_dt))>1e-6 for t in root_times):
        parser.error('Distinct root times must be legal ticks after the common observation prefix')
    if args.root_times and args.feedback_mode!='persistent_interception':
        parser.error('Multiple state roots require persistent feedback programs')
    items=[(i*len(root_times)+j,t,bank['protocol'],when,i%len(PROGRAMS))
           for i,t in enumerate(tasks) for j,when in enumerate(root_times)]
    if not tasks or args.future_base<0 or args.future_base+len(items)*args.max_futures>2**32:
        parser.error('Empty task bank or invalid branch seed range')
    cfg=FleetConfig(teacher_checkpoint=args.teacher,history=physical.history_steps,decision_time=args.decision_time,
        feedback_mode=args.feedback_mode,capture_time_weight=args.capture_time_weight)
    opponent=load_opponent(args.opponent_spec)
    parameters={k:v for k,v in vars(args).items() if k not in ('resume','out','workers')}
    contract=immutable_run_contract(dict(arguments=parameters,config=asdict(cfg),opponent=opponent.identity),
        files=[args.tasks,args.source,args.teacher]+([args.opponent_spec] if args.opponent_spec else []),
        code=[__file__,Path(__file__).with_name('fleet_program_policy.py'),Path(__file__).with_name('fleet_feedback.py')])
    teacher_sha256=contract['artifacts'][str(Path(args.teacher).resolve())]
    if args.resume:require_same_contract(json.loads((out/'protocol.json').read_text()).get('contract'),contract,'Fleet collection resume')
    out.mkdir(parents=True,exist_ok=args.resume);(out/'roots').mkdir(exist_ok=True)
    (out/'full_future/roots').mkdir(parents=True,exist_ok=True)
    if not args.resume:
        atomic_json(dict(arguments=vars(args),config=asdict(cfg),role=bank['role'],contract=contract,
            teacher_sha256=teacher_sha256,
            physical_protocol=bank['protocol'],source_identity=source_identity(args.project,args.source),
            opponent_identity=opponent.identity,programs=list(PROGRAMS),
            root_times=root_times,behavior='program0 coverage to2s; then layout-index modulo4 persistent program',
            common_future_rule=f'{args.futures} initial; extend all four to{args.max_futures} if a paired mean has SEM>.04 and abs(mean)<1.64SEM',
            execution='same coverage prefix; same projected first command and closed-state; full program to actual terminal',
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),out/'protocol.json')
    rows=[];started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize,initargs=(dict(vars(args),teacher_sha256=teacher_sha256),asdict(cfg))) as pool:
        jobs=[pool.submit(collect,item) for item in items]
        for future in concurrent.futures.as_completed(jobs):
            rows.append(future.result())
            status=dict(completed=len(rows)==len(items),layouts=len({r['task']['geometry_family_id'] for r in rows}),expected_layouts=len(tasks),
                processed_states=len(rows),expected_states=len(items),
                roots=sum(r['reached'] for r in rows),branch_steps=sum(r['branch_steps'] for r in rows),
                prefix_steps=sum(r['prefix_steps'] for r in rows),
                terminal_branches=sum(r.get('terminal_branches',0) for r in rows),
                expanded_roots=sum(r['futures']>args.futures for r in rows),seconds=time.time()-started)
            atomic_json(status,out/'status.json');print(json.dumps(status),flush=True)
    status.update(candidate_count=len(PROGRAMS),scope='complete conditional supervision; not independent deployment performance')
    atomic_json(status,out/'summary.json')
    record=json.loads((out/'protocol.json').read_text())
    atomic_json(dict(source_data=str(out),role=bank['role'],physical_protocol=bank['protocol'],
        source_identity=record['source_identity'],interval='remaining task through actual terminal',
        method='synchronous distances recorded in the actual common-future branches'),out/'full_future/protocol.json')
    atomic_json(dict(completed=True,roots=status['roots'],branches=status['terminal_branches'],steps=status['branch_steps']),out/'full_future/summary.json')


if __name__=='__main__':main()
