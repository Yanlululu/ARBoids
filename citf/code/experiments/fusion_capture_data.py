"""Real stopped feedback branches for the complete joint configuration bank."""
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
from citf.contracts import Protocol
from citf.evaluate import atomic_json
from experiments.contact_event_policy import retained_teacher
from experiments.persistent_coverage_control import PersistentCoverageController
from experiments.fusion_capture_policy import FusionConfig, ConfigurationExecutor, FusionCaptureController, load_policy, make_input, joint


WORKER = {}


def save_tensor(value, path):
    pending = path.with_suffix('.pending')
    torch.save(value, pending)
    pending.replace(path)


def initialize(arguments):
    torch.set_num_threads(1)
    source = Source(arguments['project'], arguments['source'])
    model = retained_teacher(arguments['teacher'])
    WORKER.update(arguments=arguments, source=source, teacher_model=model,
        teacher=joint.JointPolicyController(source, model, 'cpu', 12, 96, 1),
        executor=ConfigurationExecutor(source), cfg=FusionConfig(teacher_checkpoint=arguments['teacher']))
    if arguments.get('behavior_checkpoint'):
        WORKER['behavior_model']=load_policy(arguments['behavior_checkpoint'])


def collect(item):
    index, task, physical_protocol = item
    a = WORKER['arguments']; out = Path(a['out'])
    task_path = out/'tasks'/f'{index:05d}.json'
    if task_path.exists():
        return json.loads(task_path.read_text())
    world = World(a['project'], Protocol(**physical_protocol), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    behavior = (FusionCaptureController(WORKER['source'],WORKER['behavior_model'],'cpu',12,96,1)
                if a.get('behavior_checkpoint') else
                PersistentCoverageController(WORKER['source'], WORKER['teacher_model'], 'cpu', 12, 96, 1))
    ticks = [round(t/world.protocol.control_dt) for t in a['root_times']]
    tick = 0; rows = []
    while not world.done and tick <= max(ticks):
        prefix_force, _ = behavior.act(world)
        if tick in ticks:
            ordinal = ticks.index(tick); root_id = f'{index:05d}_{ordinal:02d}'
            meta = out/'roots'/(root_id+'.json')
            if meta.exists():
                rows.append(json.loads(meta.read_text()))
            else:
                started = time.time(); cfg = WORKER['cfg']; executor = WORKER['executor']
                batch, gates, parameters, first, rotation, _ = make_input(world, WORKER['source'],
                    WORKER['teacher'], executor, behavior.guard if a.get('behavior_checkpoint') else behavior.teacher.guard, cfg)
                C, R, E, H = len(gates), a['futures'], task['defenders']+1, cfg.horizon
                xy = np.zeros((C, R, E, H, 2), np.float32)
                valid = np.zeros((C, R, H), bool)
                commands = np.zeros((C, R, H, E-1, 2), np.float32)
                events = np.zeros((C, R, H, 3), bool)
                endpoints = []
                future_seeds = [a['future_base']+index*1000+ordinal*100+r for r in range(R)]
                for c in range(C):
                    executor.configure(gates[c], parameters[c])
                    for r, seed in enumerate(future_seeds):
                        branch = world.clone(); branch.rng = np.random.RandomState(seed)
                        for q in range(H):
                            command = first[c].numpy() if q == 0 else executor.act(branch)[0]
                            result = branch.step(command)
                            xy[c, r, :, q] = branch.state()[:, :2] @ rotation
                            valid[c, r, q] = True; commands[c, r, q] = command
                            events[c, r, q] = [result['capture'], result['physical_collision'], result['physical_breach']]
                            if branch.done: break
                        xy[c, r, :, q+1:] = xy[c, r, :, q:q+1]
                        endpoints.append(dict(configuration=c, future_replica=r, future_seed=seed,
                            steps=q+1, stopped=branch.done, **result))
                payload = dict(batch={k:v[0] for k,v in batch.items()},
                    real_positions=torch.from_numpy(xy), time_valid=torch.from_numpy(valid),
                    actual_commands=torch.from_numpy(commands), events=torch.from_numpy(events),
                    gates=gates, feedback_parameters=parameters, task=task, elapsed=world.env.Current_T,
                    root_id=root_id, future_seeds=future_seeds, physical_protocol=physical_protocol)
                save_tensor(payload, meta.with_suffix('.pt'))
                row = dict(root_id=root_id, task=task, elapsed=world.env.Current_T, candidates=C,
                    futures=R, branch_steps=int(valid.sum()), terminal_branches=sum(x['stopped'] for x in endpoints),
                    endpoints=endpoints, seconds=time.time()-started)
                atomic_json(row, meta); rows.append(row)
        if tick == max(ticks): break
        result = world.step(prefix_force); tick += 1
    row = dict(task=task, roots=[r['root_id'] for r in rows], requested_roots=len(ticks),
        unreached_roots=[q for q in ticks if q>tick], prefix_steps=tick, prefix_terminated=world.done,
        prefix_endpoint=result if world.done else None,
        branch_steps=sum(r['branch_steps'] for r in rows), terminal_branches=sum(r['terminal_branches'] for r in rows))
    atomic_json(row, task_path)
    return row


def main():
    parser = argparse.ArgumentParser()
    for name in ['project','source','teacher','tasks','out']: parser.add_argument('--'+name, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--layouts', type=int, default=128)
    parser.add_argument('--futures', type=int, default=4)
    parser.add_argument('--future-base', type=int, default=91051000)
    parser.add_argument('--root-times', type=float, nargs='+', default=[0.,8.,20.,40.])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--behavior-checkpoint')
    args = parser.parse_args(); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    (out/'roots').mkdir(exist_ok=args.resume); (out/'tasks').mkdir(exist_ok=args.resume)
    bank = json.loads(Path(args.tasks).read_text()); families = {}
    if bank['role'] not in ('train','validation'): raise ValueError('Only allocated development data')
    for task in bank['tasks']: families.setdefault(task['geometry_family_id'], []).append(task)
    allocated = list(families.values())[:args.layouts]
    tasks = [group[i%len(group)] for i,group in enumerate(allocated)]
    if not args.resume:
        atomic_json(dict(arguments=vars(args), config=asdict(FusionConfig(teacher_checkpoint=args.teacher)),
            physical_protocol=bank['protocol'], role=bank['role'], allocated_layouts=len(tasks),
            selection='first allocated layout families,one alternating agility per family',
            behavior=args.behavior_checkpoint or 'persistent inner coverage plus retained joint teacher',
            execution='every candidate uses the same projected first command and refreshing feedback executor',
            supervision='all 12 candidates x 4 common futures; actual stopped interval includes terminal transition',
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),out/'protocol.json')
    started = time.time(); rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
        mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(vars(args),)) as pool:
        jobs = [pool.submit(collect,(i,t,bank['protocol'])) for i,t in enumerate(tasks)]
        for job in concurrent.futures.as_completed(jobs):
            rows.append(job.result())
            status = dict(completed=len(rows)==len(tasks), layouts=len(rows), expected_layouts=len(tasks),
                roots=sum(len(r['roots']) for r in rows), branch_steps=sum(r['branch_steps'] for r in rows),
                terminal_branches=sum(r['terminal_branches'] for r in rows),
                prefix_steps=sum(r['prefix_steps'] for r in rows), seconds=time.time()-started)
            atomic_json(status, out/'status.json'); print(json.dumps(status),flush=True)
    status.update(common_futures=args.futures, candidate_count=12, requested_roots=len(tasks)*len(args.root_times),
                  scope='development supervision,not independent complete-task performance')
    atomic_json(status,out/'summary.json')


if __name__ == '__main__': main()
