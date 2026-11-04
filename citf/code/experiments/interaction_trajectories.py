"""Whole-trajectory observed transitions for continuous joint interception.

Every allocated episode is retained, including early failures. Attacker thrust
is an auxiliary TRAINING LABEL measured after a transition, never an input.
"""
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
from citf.checkpointing import atomic_json, atomic_bytes
from citf.contracts import (Protocol, canonical_hash, file_hash, source_identity,
    immutable_run_contract, require_same_contract, validate_task_banks)
from experiments.root_response_diagnostic import physical_bank, forces
from experiments.synchronized_configuration_policy import SynchronizedPolicyConfig


WORKER = {}


def episode_jobs(bank):
    return [(i * 2 + j, task, behavior) for i, task in enumerate(bank['tasks'])
            for j, behavior in enumerate(('strong_physical', 'source_preference'))]


def validate_episode(row, job, directory, contract):
    index, task, behavior = job
    if (row['index'], row['task'], row['behavior']) != (index, task, behavior):
        raise ValueError('Cached trajectory belongs to a different task or behavior')
    if row.get('collection_identity') != canonical_hash(contract):
        raise ValueError('Cached trajectory belongs to a different collection')
    expected = contract['parameters']['source']
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('Trajectory source or environment differs from its collection')
    if row['episode'] != f'{index:05d}.npz' or row['steps'] < 1 or row['result'].get('outcome') not in (1, 2, 3, 4):
        raise ValueError('Incomplete trajectory record')
    path = Path(directory) / 'episodes' / row['episode']
    if not path.is_file() or row.get('episode_sha256') != file_hash(path):
        raise ValueError('Trajectory payload is missing or differs from its committed record')


def load_collection(directory, role):
    """One reader contract for allocation, completion and episode payloads."""
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    contract = protocol.get('collection_contract')
    if contract is None:
        raise ValueError('Legacy trajectories lack an immutable collection contract; retain them separately')
    identity = canonical_hash(contract)
    bank = contract['parameters']['task_bank']
    families = validate_task_banks((role, bank))
    if protocol['role'] != role or protocol['protocol'] != bank['protocol']:
        raise ValueError('Trajectory header differs from its allocated role or protocol')
    report = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(x) for x in (directory / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    jobs = episode_jobs(bank)
    if (not report.get('completed') or report.get('collection_identity') != identity
            or protocol.get('collection_identity') != identity
            or len(rows) != len(jobs) or report['episodes'] != len(jobs)
            or report['expected_episodes'] != len(jobs) or report['layouts'] != len(families[role])):
        raise ValueError('Trajectory collection is incomplete or its identity differs')
    for row, job in zip(rows, jobs):
        validate_episode(row, job, directory, contract)
    if report['transitions'] != sum(row['steps'] for row in rows):
        raise ValueError('Trajectory transition count differs from committed episodes')
    return protocol, report, rows


def initialize(project, source, out, protocol, contract):
    torch.set_num_threads(1)
    actual = source_identity(project, source)
    require_same_contract(contract['parameters']['source'], actual, 'Collection worker source')
    WORKER.update(project=project, source=Source(project, source), out=Path(out),
                  protocol=Protocol(**protocol), cfg=SynchronizedPolicyConfig(),
                  contract=contract, source_identity=actual)


@torch.no_grad()
def collect(job):
    index, task, behavior = job
    path = WORKER['out'] / 'episodes' / f'{index:05d}.npz'
    meta = path.with_suffix('.json')
    if meta.exists():
        row = json.loads(meta.read_text(encoding='utf-8'))
        validate_episode(row, job, WORKER['out'], WORKER['contract'])
        return row
    world = World(WORKER['project'], WORKER['protocol'], task['defenders'])
    if world.environment_sha256 != WORKER['source_identity']['environment_sha256']:
        raise ValueError('Collection environment changed after the run was registered')
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    if task['geometry_family_id'] != world.initial_identity['fingerprint']:
        raise ValueError('Episode geometry differs from allocated task')
    states, commands, attacker_labels, events = [world.state().copy()], [], [], []
    while not world.done:
        if behavior == 'strong_physical':
            factor, _, _, choice, _ = physical_bank(world, WORKER['source'], WORKER['cfg'])
            force = forces(factor, choice)
        else:
            learned, boids, gate = WORKER['source'].commands(world.env)
            force = (750 * (gate * learned + (1 - gate) * boids) + 250).numpy()
        result = world.step(force)
        commands.append(force)
        attacker_labels.append(world.env.action_to_thrust(world.env.att_action, world.env.attacker.agility))
        states.append(world.state().copy())
        events.append([result[k] for k in ('capture', 'collision', 'radial_only_failure', 'physical_breach', 'timeout')])
    with path.with_suffix('.pending').open('wb') as handle:
        np.savez_compressed(handle, state=np.asarray(states, np.float32),
            command=np.asarray(commands, np.float32), attacker_thrust_label=np.asarray(attacker_labels, np.float32),
            events=np.asarray(events, np.uint8))
    path.with_suffix('.pending').replace(path)
    row = dict(index=index, task=task, behavior=behavior, steps=len(commands), result=result,
               episode=path.name, observed_state='x,y,yaw,updated ground vx,vy,yaw_rate',
               source=WORKER['source'].sha256, environment=world.environment_sha256,
               collection_identity=canonical_hash(WORKER['contract']), episode_sha256=file_hash(path),
               **WORKER['source_identity'])
    atomic_json(row, meta)
    return row


def main():
    parser = argparse.ArgumentParser()
    for key in ('project', 'source', 'tasks', 'out'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out, bank = Path(args.out), json.loads(Path(args.tasks).read_text(encoding='utf-8'))
    validate_task_banks((bank['role'], bank))
    contract = immutable_run_contract(dict(task_bank=bank,
        source=source_identity(args.project, args.source), behavior_config=asdict(SynchronizedPolicyConfig())),
        files=(args.tasks, args.source), code=(__file__,))
    if args.resume:
        recorded = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
        require_same_contract(recorded.get('collection_contract'), contract, 'Trajectory resume')
        if recorded.get('collection_identity') != canonical_hash(contract):
            raise ValueError('Recorded trajectory collection identity is inconsistent')
    out.mkdir(parents=True, exist_ok=args.resume)
    (out / 'episodes').mkdir(exist_ok=args.resume)
    jobs = episode_jobs(bank)
    protocol = dict(revision='CITF-observable-interaction-trajectories-v2', arguments=vars(args),
        collection_contract=contract, collection_identity=canonical_hash(contract),
        expected_episodes=len(jobs), layouts=len({t['geometry_family_id'] for t in bank['tasks']}),
        role=bank['role'], protocol=bank['protocol'],
        behavior='two complete policies on each allocated layout/agility/future; no outcome filtering',
        labels='all observed state transitions and executed attacker thrust; no future action inputs',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if not args.resume:
        atomic_json(protocol, out / 'protocol.json')
    started, rows = time.time(), []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, str(out), bank['protocol'], contract)) as executor:
        pending = [executor.submit(collect, job) for job in jobs]
        for future in concurrent.futures.as_completed(pending):
            rows.append(future.result())
            status = dict(completed=False, episodes=len(rows), expected_episodes=len(jobs),
                          transitions=sum(x['steps'] for x in rows), elapsed_seconds=time.time()-started)
            atomic_json(status, out / 'status.json')
            if len(rows) % 16 == 0:
                print(json.dumps(status), flush=True)
    rows.sort(key=lambda x: x['index'])
    atomic_bytes(out / 'episodes.jsonl', ''.join(json.dumps(x) + '\n' for x in rows).encode('utf-8'))
    report = dict(completed=True, episodes=len(rows), expected_episodes=len(jobs),
        collection_identity=canonical_hash(contract),
        transitions=sum(x['steps'] for x in rows), layouts=protocol['layouts'],
        behaviors={name: dict(episodes=sum(x['behavior']==name for x in rows),
            success=sum(x['result']['success'] for x in rows if x['behavior']==name),
            collision=sum(x['result']['collision'] for x in rows if x['behavior']==name))
            for name in ('strong_physical', 'source_preference')}, elapsed_seconds=time.time()-started)
    atomic_json(report, out / 'summary.json'); atomic_json(report, out / 'status.json')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
