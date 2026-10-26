"""Timing-aware selection of measured conditional-value fusion programs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import fixed_development_cv as fixed

e = fixed.e
OUT = fixed.OUT
native_load_actor = fixed.joint.load_actor


def load_actor(path, development):
    actor = native_load_actor(path, development)
    settings = e.read(Path(path)).get('fixed_pool_value_selection')
    if settings is None:
        return actor
    assert settings['source_sha256'] == e.digest(Path(__file__))
    decision = actor.successor_task_value
    native_eligible = decision.eligible

    def eligible(packet):
        allowed = native_eligible(packet)
        steps = (packet['node'][:, 0, 15] * 400.).round().long()
        allowed &= steps >= settings['minimum_step']
        if settings['maximum_step'] is not None:
            allowed &= steps <= settings['maximum_step']
        return allowed

    decision.eligible = eligible
    return actor


def candidate(args):
    fixed.initialize()
    parent = Path(args.policy)
    target = OUT / (args.name + '.pth')
    if target.exists():
        raise ValueError('A policy identity cannot be overwritten.')
    if args.minimum_step < 0 or (args.maximum_step is not None
                                and args.maximum_step < args.minimum_step):
        raise ValueError('The eligible decision interval must be nonempty.')
    model = e.read(parent)
    model['fixed_pool_value_selection'] = dict(parent_policy=str(parent),
        parent_sha256=e.digest(parent), minimum_step=args.minimum_step,
        maximum_step=args.maximum_step, source_sha256=e.digest(Path(__file__)),
        clock='Observed physical decision clock from the public packet.',
        candidate_execution_unchanged=True, predecessor_ci_unchanged=True,
        role='Selection timing developed on the fixed development validation set.')
    e.save(model, target)
    print(json.dumps(dict(path=str(target), sha256=e.digest(target))), flush=True)


def design(args):
    fixed.initialize()
    target = OUT / (args.name + '.pth')
    if target.exists():
        raise ValueError('A frozen evaluation design cannot be overwritten.')
    specification = e.read(fixed.DESIGN)
    observed = (specification['existing_training_tasks']
                + specification['added_training_tasks']
                + specification['validation_tasks'])
    for path in (fixed.p.PRIOR / 'mixed-design.pth',
                 fixed.p.PRIOR / 'joint-final-new200-design.pth',
                 fixed.p.OUT / 'fresh400-final-design.pth'):
        observed += e.read(path)['tasks']
    used = {int(scene[0]) for scene, _ in observed}
    for policy in args.policies:
        model = e.read(Path(policy))
        used.update(int(scene[0]) for scene in
                    model.get('successor_task_fit', {}).get('training_layouts', []))
    for path in OUT.glob('*independent-design.pth'):
        used.update(int(scene[0]) for scene, _ in e.read(path)['tasks'])
    tasks = []
    index = 0
    while len(tasks) < args.count:
        seed = e.seed32(args.seed, index)
        index += 1
        if seed in used:
            continue
        used.add(seed)
        tasks.append(((seed, 6, 2. if len(tasks) < args.count // 2 else 2.5), None))
    result = dict(tasks=tasks,
        policies={str(Path(path)): e.digest(Path(path)) for path in args.policies},
        frozen_unix=time.time(), source_sha256=e.digest(Path(__file__)),
        fixed_pool_sha256=e.digest(fixed.DESIGN),
        driver_sha256=e.digest(Path(fixed.__file__)),
        joint_driver_sha256=e.digest(Path(fixed.joint.__file__)),
        selected_policy=str(Path(args.selected_policy)),
        role='Independent evaluation after fixed-pool development and policy selection.')
    assert result['selected_policy'] in result['policies']
    e.save(result, target)
    print(json.dumps(dict(path=str(target), count=len(tasks),
                         policies=len(args.policies))), flush=True)


def evaluate(args):
    _, development, _, _, runtime = fixed.initialize()
    if args.design:
        specification = e.read(Path(args.design))
        assert specification['source_sha256'] == e.digest(Path(__file__))
        assert specification['fixed_pool_sha256'] == e.digest(fixed.DESIGN)
        assert specification['driver_sha256'] == e.digest(Path(fixed.__file__))
        assert specification['joint_driver_sha256'] == e.digest(Path(fixed.joint.__file__))
        tasks = specification['tasks']
        role = specification['role']
    else:
        specification = e.read(fixed.DESIGN)
        tasks = specification['validation_tasks']
        role = 'Repeated fixed development validation, used for model development.'
    policy = Path(args.policy)
    if args.design:
        assert specification['policies'][str(policy)] == e.digest(policy)
    full_count = len(tasks)
    assert args.shards > 0 and 0 <= args.shard_index < args.shards
    indices = np.array_split(np.arange(full_count), args.shards)[args.shard_index]
    tasks = [tasks[int(index)] for index in indices]
    target = OUT / f'{args.name}-part{args.shard_index}-n{len(tasks)}-evaluation.pth'
    if target.exists():
        result = e.read(target)
        assert result['completed'] and result['tasks'] == tasks
        assert result['policy_sha256'] == e.digest(policy)
        print(json.dumps(dict(cached=True, path=str(target))), flush=True)
        return
    start = time.monotonic()
    rows = runtime.batched_credit_rows(policy, tasks, batch_size=args.batch,
        device=args.device, actor_override=load_actor(policy, development),
        fixed_inference_width=80)
    result = dict(policy=str(policy), policy_sha256=e.digest(policy),
        tasks=tasks, rows=rows, completed=True,
        success=sum(row['success'] for row in rows),
        elapsed_seconds=time.monotonic() - start, role=role,
        fixed_inference_width=80, full_design_task_count=full_count,
        shard_index=args.shard_index, shards=args.shards,
        source_sha256=e.digest(Path(__file__)),
        evaluation_design_sha256=e.digest(Path(args.design) if args.design else fixed.DESIGN))
    e.save(result, target)
    print(json.dumps(dict(path=str(target), success=result['success'], count=len(rows))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('candidate', 'design', 'evaluate'))
    parser.add_argument('--name', required=True)
    parser.add_argument('--policy')
    parser.add_argument('--policies', nargs='+', default=[])
    parser.add_argument('--selected-policy')
    parser.add_argument('--minimum-step', type=int, default=5)
    parser.add_argument('--maximum-step', type=int)
    parser.add_argument('--design')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=11262350)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--batch', type=int, default=250)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    globals()[args.operation](args)


if __name__ == '__main__':
    main()
