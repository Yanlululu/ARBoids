"""Repeated unified fusion learning on an expanded, explicitly reused development pool."""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import time
from pathlib import Path

import numpy as np
import subject_centered_cv as centered

fixed = centered.fixed
e = centered.e
OLD = centered.OUT
OUT = e.BASE / 'unified-civ-development-20261027'
DESIGN = OUT / 'expanded-design.pth'
PROPOSERS = [OLD / 'pool3000-centered-rank4-seed20261102-conditional.pth',
             OLD / 'pool3000-centered-rank4-seed20261104-conditional.pth',
             OLD / 'pool3000-centered-rank4-seed20261102-ordinary.pth']


def configure():
    fixed.OUT = OUT
    fixed.DESIGN = DESIGN
    fixed.PROPOSERS = PROPOSERS
    fixed.joint.load_actor = centered.load_actor
    centered.OUT = OUT
    OUT.mkdir(exist_ok=True)
    return centered.initialize()


def prepare(args):
    configure()
    if DESIGN.exists():
        print(json.dumps(dict(cached=True, path=str(DESIGN))), flush=True)
        return
    previous = e.read(OLD / 'fixed-development-design.pth')
    cohort_paths = [Path(previous['validation_source']),
                    OLD / 'fixedpool-final1000-independent-design.pth',
                    OLD / 'paper-readiness-b-independent-design.pth',
                    OLD / 'paper-readiness-c-independent-design.pth']
    cohorts = [e.read(path)['tasks'] for path in cohort_paths]
    existing = e.read(OLD / 'combined-factual.pth')['tasks']
    added = [task for tasks in cohorts for task in tasks]
    used = {int(scene[0]) for scene, _ in existing + added}
    fresh = []
    index = 0
    while len(fresh) < 2000:
        seed = e.seed32(11270822, index)
        index += 1
        if seed in used:
            continue
        used.add(seed)
        fresh.append(((seed, 6, 2. if len(fresh) < 1000 else 2.5), None))
    added += fresh
    screening = [tasks[i] for tasks in cohorts for i in list(range(50)) + list(range(500, 550))]
    repeated = [tasks[i] for tasks in cohorts for i in list(range(200)) + list(range(500, 700))]
    result = dict(added_training_tasks=added, existing_training_tasks=existing,
        validation_tasks=repeated, screening_tasks=screening,
        source_cohorts=[dict(path=str(path), sha256=e.digest(path)) for path in cohort_paths],
        newly_generated_tasks=fresh, predecessor=str(fixed.p.BASE),
        predecessor_sha256=e.digest(fixed.p.BASE),
        proposal_policies={str(path): e.digest(path) for path in PROPOSERS},
        collection_shards=12, roots_per_shard=64, branch_repetitions=3,
        fixed_inference_width=80, driver_sha256=e.digest(Path(fixed.__file__)),
        preparation_source_sha256=e.digest(Path(__file__)), frozen_unix=time.time(),
        role='Expanded fixed development pool; former evaluation cohorts explicitly reused for learning.',
        validation_used_for_gradients=True, independent_final_evaluation_claim=False)
    assert len(added) == 6000 and len(existing) == 3000
    assert len({int(scene[0]) for scene, _ in existing + added}) == 9000
    e.save(result, DESIGN)
    print(json.dumps(dict(training_layouts=9000, added_layouts=6000,
        newly_generated=2000, screening=len(screening), repeated=len(repeated))), flush=True)


def collect(args):
    configure()
    fixed.collect(args)


def merge(args):
    _, _, local, _, _ = configure()
    specification = e.read(DESIGN)
    paired_sources = [OLD / 'combined-paired.pth']
    factual_sources = [OLD / 'combined-factual.pth']
    for index in range(specification['collection_shards']):
        paired_sources.append(OUT / f'added-paired-part{index}.pth')
        factual_sources.append(OUT / f'added-factual-part{index}.pth')
    datasets = [e.read(path) for path in paired_sources]
    banks = [e.read(path) for path in factual_sources]
    assert all(item['completed'] for item in datasets + banks)
    for index, bank in enumerate(banks[1:]):
        assert bank['tasks'] == fixed.shard_tasks(specification, index)
    paired = local.merge_paired_data(datasets, allow_distinct_contexts=True)
    paired.update(completed=True, training_data=str(OUT / 'combined-paired.pth'),
        fixed_development_design_sha256=e.digest(DESIGN))
    bank = copy.deepcopy(banks[0])
    bank['records'] = [record for source in banks for record in source['records']]
    bank['tasks'] = [task for source in banks for task in source['tasks']]
    for key in ('simulation_steps', 'trajectories', 'state_labels'):
        bank[key] = sum(source.get(key, 0) for source in banks)
    bank.update(completed=True, fixed_development_design_sha256=e.digest(DESIGN),
        acquisition_sources=[e.digest(path) for path in factual_sources])
    assert len(bank['tasks']) == 9000
    for item, name in ((paired, 'combined-paired.pth'), (bank, 'combined-factual.pth')):
        if (OUT / name).exists():
            raise ValueError('The completed expanded pool already exists.')
        e.save(item, OUT / name)
    print(json.dumps(dict(completed=True, layouts=9000, roots=len(paired['roots']),
        outcomes=sum(len(root['outcomes']) for root in paired['roots']))), flush=True)


def train(args):
    configure()
    native_read = e.read
    native_train = fixed.train

    def training_read(path):
        model = native_read(path)
        if Path(path) == PROPOSERS[0] and args.parameter_budget:
            model['successor_task_fit']['matched_success_training'][
                'paired_parameters_per_member'] = args.parameter_budget
        return model

    e.read = training_read
    if args.objective != 'joint_likelihood':
        namespace = dict(fixed.__dict__)
        source = inspect.getsource(native_train).replace("'joint_likelihood'", repr(args.objective))
        exec(compile(source, '<expanded-pool-objective>', 'exec'), namespace)
        fixed.train = namespace['train']
    try:
        centered.train(args)
    finally:
        e.read = native_read
        fixed.train = native_train
    for arm in (('conditional', 'ordinary') if args.matched_ordinary else ('conditional',)):
        path = OUT / f'{args.name}-{arm}.pth'
        model = e.read(path)
        model['fixed_development_training'].update(validation_used_for_gradients=True,
            expanded_pool=True, independent_final_evaluation_claim=False)
        model['unified_development'] = dict(source_sha256=e.digest(Path(__file__)),
            pool_sha256=e.digest(DESIGN), all_training_scenarios=9000,
            requested_parameter_budget=args.parameter_budget, objective=args.objective)
        e.save(model, path)


def evaluate(args):
    _, development, _, _, runtime = configure()
    specification = e.read(DESIGN)
    tasks = specification['screening_tasks' if args.screen else 'validation_tasks']
    groups = np.array_split(np.arange(len(tasks)), args.shards)
    tasks = [tasks[int(i)] for i in groups[args.shard_index]]
    policy = Path(args.policy)
    target = OUT / f'{args.name}-part{args.shard_index}-n{len(tasks)}-evaluation.pth'
    if target.exists():
        item = e.read(target)
        assert item['completed'] and item['tasks'] == tasks and item['policy_sha256'] == e.digest(policy)
        return
    start = time.monotonic()
    from unified_civ_factual_value import load_actor
    actor = load_actor(policy, development)
    rows = runtime.batched_credit_rows(policy, tasks, batch_size=args.batch,
        device=args.device, actor_override=actor, fixed_inference_width=80)
    e.save(dict(completed=True, tasks=tasks, rows=rows, policy=str(policy),
        policy_sha256=e.digest(policy), success=sum(row['success'] for row in rows),
        design_sha256=e.digest(DESIGN), source_sha256=e.digest(Path(__file__)),
        fixed_inference_width=80, shard_index=args.shard_index, shards=args.shards,
        screen=args.screen, elapsed_seconds=time.monotonic()-start,
        role='Repeated development evaluation; scenarios are in the expanded learning pool.'), target)
    print(json.dumps(dict(completed=True, path=str(target), success=sum(r['success'] for r in rows),
        count=len(rows), elapsed=time.monotonic()-start)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'collect', 'merge', 'train', 'evaluate'))
    parser.add_argument('--name')
    parser.add_argument('--policy')
    parser.add_argument('--seed', type=int, default=20261201)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--batch', type=int, default=250)
    parser.add_argument('--updates', type=int, default=54000)
    parser.add_argument('--parameter-budget', type=int, default=888888)
    parser.add_argument('--objective', choices=('joint_likelihood', 'marginal_events',
                                               'shared_potential'), default='joint_likelihood')
    parser.add_argument('--gain-weight', type=float, default=8.)
    parser.add_argument('--harm-weight', type=float, default=4.)
    parser.add_argument('--rank-weight', type=float, default=4.)
    parser.add_argument('--joint-weight', type=float, default=1.28)
    parser.add_argument('--matched-ordinary', action='store_true')
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    globals()[args.operation](args)


if __name__ == '__main__':
    main()
