"""Derive full stopped-distance targets from the recorded executed commands.

The original common-future rollouts remain immutable. This pass reconstructs
their trajectories without repeating candidate search or policy inference in
the branches; it produces the long-interval labels needed by task selection.
"""
import argparse
import concurrent.futures
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.bridge import Source,World
from citf.contracts import Protocol,immutable_run_contract,require_same_contract
from citf.evaluate import atomic_json
from citf.opponents import load_opponent
from experiments.fleet_program_policy import (FleetConfig,FleetExecutor,initial_guard,coverage_parameters,
    load_teacher,recorded_teacher_identity)
from experiments.fusion_capture_data import save_tensor


WORKER={}


def initialize(arguments,record):
    torch.set_num_threads(1);a=record['arguments'];source=Source(a['project'],a['source'])
    teacher=load_teacher(source,a['teacher'],recorded_teacher_identity(record,a['teacher']))
    WORKER.update(arguments=arguments,record=record,cfg=FleetConfig(**record['config']),
        executor=FleetExecutor(source,teacher),opponent=load_opponent(a.get('opponent_spec')))


def derive(path):
    path=Path(path);out=Path(WORKER['arguments']['out']);target=out/'roots'/path.name
    meta=target.with_suffix('.json')
    if meta.exists():return json.loads(meta.read_text())
    started=time.time();data=torch.load(path,map_location='cpu',weights_only=False)
    row=json.loads(path.with_suffix('.json').read_text());a=WORKER['record']['arguments'];task=data['task']
    world=World(a['project'],Protocol(**data['physical_protocol']),task['defenders'])
    world.reset(task['scene_seed'],task['future_seed'],task['agility'])
    if world.initial_identity['fingerprint']!=task['geometry_family_id']:
        raise ValueError('Full-future replay layout differs from its recorded root')
    executor=WORKER['executor'];executor.configure(coverage_parameters(world,initial_guard(world)))
    opponent=WORKER['opponent']
    while world.env.Current_T+1e-7<data['elapsed']:
        world.step(executor.act(world)[0],attacker_action=opponent.act(world))
    history,positions,history_valid,rotation=world.observation()
    if (world.done or abs(world.env.Current_T-data['elapsed'])>1e-6
            or not torch.equal(history_valid,data['batch']['history_valid'])
            or not torch.allclose(history,data['batch']['history'],atol=1e-5,rtol=1e-6)
            or not torch.allclose(positions,data['batch']['positions'],atol=1e-5,rtol=1e-6)
            or opponent.identity!=data['opponent_identity']):
        raise ValueError('Full-future replay prefix differs from the sampled decision state')
    C,R=data['terminal_utility'].shape;E=task['defenders']+1;i,j=np.triu_indices(E,1)
    minimum=np.full((C,R,len(i)),np.inf,np.float32);radius=np.full((C,R),np.inf,np.float32)
    time_at=np.zeros((C,R),np.float32);steps=0
    endpoints={(e['configuration'],e['future_replica']):e for e in row['endpoints']}
    commands=data['actual_commands'].numpy()
    for c in range(C):
        for r,seed in enumerate(data['future_seeds']):
            branch=world.clone();branch.rng=np.random.RandomState(seed);expected=endpoints[c,r]
            best_contact=float('inf')
            for tick in range(expected['steps']):
                result=branch.step(commands[c,r,tick],attacker_action=opponent.act(branch))
                position=branch.state()[:,:2];distance=np.linalg.norm(position[i]-position[j],axis=-1)
                if tick<data['real_positions'].shape[-2] and data['time_valid'][c,r,tick]:
                    saved=data['real_positions'][c,r,:,tick].numpy()
                    if not np.allclose(position@rotation,saved,atol=1e-5,rtol=1e-6):
                        raise ValueError('Recorded-command replay differs from the saved trajectory: '+path.stem)
                minimum[c,r]=np.minimum(minimum[c,r],distance)
                radius[c,r]=min(radius[c,r],float(np.linalg.norm(position[-1])))
                contact=float(distance[j==E-1].min())
                if contact<best_contact:best_contact=contact;time_at[c,r]=result['duration']/world.protocol.time_limit
                steps+=1
            # These are derived labels of saved interactions, not new policy rollouts.
            if not branch.done or result['outcome']!=expected['outcome'] or abs(result['duration']-expected['duration'])>1e-6:
                raise ValueError('Recorded-command continuation differs from its saved terminal: '+path.stem)
    save_tensor(dict(root_id=data['root_id'],task=task,future_seeds=data['future_seeds'],
        physical_protocol=data['physical_protocol'],pair=torch.from_numpy(minimum),
        team=torch.from_numpy(minimum[...,j==E-1].min(-1)),radius=torch.from_numpy(radius),
        time_at_minimum=torch.from_numpy(time_at),interval='root through actual terminal transition'),target)
    record=dict(root_id=data['root_id'],source=str(path),branches=C*R,steps=steps,seconds=time.time()-started)
    atomic_json(record,meta);return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--workers',type=int,default=16);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();directory=Path(args.data);out=Path(args.out)
    record=json.loads((directory/'protocol.json').read_text());summary=json.loads((directory/'summary.json').read_text())
    if not summary['completed']:raise ValueError('Full-future source collection is incomplete')
    paths=sorted((directory/'roots').glob('*.pt'))
    if len(paths)!=summary['roots']:raise ValueError('Full-future source root coverage differs')
    contract=immutable_run_contract(dict(data=str(directory),config=record['config'],role=record['role']),
        files=[directory/'protocol.json',directory/'summary.json'],code=[__file__])
    if args.resume:require_same_contract(json.loads((out/'protocol.json').read_text()).get('contract'),contract,'Full-future label resume')
    out.mkdir(parents=True,exist_ok=args.resume);(out/'roots').mkdir(exist_ok=True)
    if not args.resume:atomic_json(dict(arguments=vars(args),contract=contract,role=record['role'],
        source_data=str(directory),physical_protocol=record['physical_protocol'],source_identity=record['source_identity'],
        interval='remaining actual task,including the terminal transition',
        method='reconstruct saved applied commands and common random streams; no candidate search repeated'),out/'protocol.json')
    rows=[];started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize,initargs=(vars(args),record)) as pool:
        jobs=[pool.submit(derive,str(p)) for p in paths]
        for future in concurrent.futures.as_completed(jobs):
            rows.append(future.result())
            status=dict(completed=len(rows)==len(paths),roots=len(rows),expected_roots=len(paths),
                branches=sum(r['branches'] for r in rows),steps=sum(r['steps'] for r in rows),seconds=time.time()-started)
            atomic_json(status,out/'status.json');print(json.dumps(status),flush=True)
    atomic_json(status,out/'summary.json')


if __name__=='__main__':main()
