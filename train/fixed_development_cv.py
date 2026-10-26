"""Persistent scenario expansion and repeated training for joint conditional value."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import joint_value_policy_improvement as joint

p = joint.previous
e = joint.e
OUT = e.BASE / 'fixed-development-cv-20261026'
DESIGN = OUT / 'fixed-development-design.pth'
PROPOSERS = [joint.OUT / ('jointw1.28-seed20261102-conditional.pth'),
             joint.OUT / ('jointw1.28-seed20261105-conditional.pth'),
             joint.OUT / ('jointw1.28-seed20261102-ordinary.pth')]


def initialize():
    state = joint.initialize()
    OUT.mkdir(exist_ok=True)
    return state


def design(args):
    initialize()
    if DESIGN.exists():
        result = e.read(DESIGN)
        assert len(result['added_training_tasks']) == args.count
        print(json.dumps(dict(cached=True, path=str(DESIGN))), flush=True)
        return
    validation_path = joint.OUT / 'fresh1000-final-design.pth'
    validation = e.read(validation_path)['tasks']
    old_bank = e.read(p.OUT / 'combined-factual.pth')
    existing = old_bank['tasks']
    used = {int(scene[0]) for scene, _ in existing + validation}
    for path in (p.PRIOR / 'mixed-design.pth', p.PRIOR / 'joint-final-new200-design.pth',
                 p.OUT / 'fresh400-final-design.pth'):
        used.update(int(scene[0]) for scene, _ in e.read(path)['tasks'])
    added = []
    index = 0
    while len(added) < args.count:
        seed = e.seed32(11262201, index)
        index += 1
        if seed in used:
            continue
        used.add(seed)
        added.append(((seed, 6, 2. if len(added) < args.count // 2 else 2.5), None))
    result = dict(added_training_tasks=added, existing_training_tasks=existing,
        validation_tasks=validation, validation_source=str(validation_path),
        validation_source_sha256=e.digest(validation_path),
        predecessor=str(p.BASE), predecessor_sha256=e.digest(p.BASE),
        proposal_policies={str(path): e.digest(path) for path in PROPOSERS},
        collection_shards=args.shards, roots_per_shard=args.roots,
        branch_repetitions=args.repeats, fixed_inference_width=80,
        frozen_unix=time.time(), driver_sha256=e.digest(Path(__file__)),
        role='Fixed expanding training pool and reused development validation; no final-test claim.',
        validation_used_for_gradients=False,
        historical_validation_role='The first evaluation preceded this development reuse.')
    e.save(result, DESIGN)
    print(json.dumps(dict(path=str(DESIGN), added_training=len(added),
        existing_training=len(existing), fixed_validation=len(validation))), flush=True)


def shard_tasks(specification, index):
    groups = np.array_split(np.arange(len(specification['added_training_tasks'])),
                           specification['collection_shards'])
    return [specification['added_training_tasks'][int(i)] for i in groups[index]]


def collect(args):
    _, d, local, _, _ = initialize()
    specification = e.read(DESIGN)
    assert specification['driver_sha256'] == e.digest(Path(__file__))
    assert 0 <= args.shard_index < specification['collection_shards']
    for path, sha in specification['proposal_policies'].items():
        assert e.digest(Path(path)) == sha
    tasks = shard_tasks(specification, args.shard_index)
    actors = [joint.load_actor(path, d).to(args.device) for path in PROPOSERS]
    selector = local.FactualProposalRootSelector([a.successor_task_value for a in actors],
        [e.digest(path) for path in PROPOSERS], fixed_inference_width=80)
    bank_path = OUT / f'added-factual-part{args.shard_index}.pth'
    bank = local.collect_onpolicy_factual_roots(p.BASE, tasks, bank_path,
        root_sample_seed=11262202, inference_batch=250, inference_device=args.device,
        root_selector=selector, fixed_inference_width=80)
    paired_design = OUT / f'added-paired-design-part{args.shard_index}.pth'
    if paired_design.exists():
        blueprint = e.read(paired_design)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([11262203, args.shard_index]))
        failures = [i for i, record in enumerate(bank['records']) if not record['values'][0]]
        successes = [i for i, record in enumerate(bank['records']) if record['values'][0]]
        rng.shuffle(failures)
        rng.shuffle(successes)
        count = specification['roots_per_shard']
        indices = (failures[:count // 2] + successes)[:count]
        catalog = {tuple(map(int, option)) for option in
                   e.read(PROPOSERS[0])['successor_task_value']['options']}
        roots = []
        for index in indices:
            record = bank['records'][index]
            snapshot = copy.deepcopy(record['root_snapshot'])
            proposed = [tuple(map(int, option)) for option in
                        snapshot.get('proposal_observation', {}).get('options', [])]
            options = {(0, 0, 0)} | {option for option in proposed if option[0]}
            boats = []
            for option in proposed:
                if option[0]:
                    boats.extend(j for j in range(6) if (option[0] // (10 ** j)) % 10)
            boats = list(dict.fromkeys(boats))
            if not boats:
                boats = [int(rng.integers(6))]
            alternatives = [boat for boat in range(6) if boat != boats[0]]
            selected_boats = [boats[0], int(rng.choice(alternatives))]
            for boat in selected_boats:
                options.update((digit * 10 ** boat, 0, 5) for digit in (1, 3, 7, 9))
            assert options <= catalog
            options = [(0, 0, 0)] + sorted(options - {(0, 0, 0)})
            repetitions = specification['branch_repetitions']
            futures = [e.seed32(11262204, args.shard_index, index, repeat)
                       for repeat in range(repetitions)]
            roots.append(dict(scene=tuple(record['scene']), prefix_future=record['future'],
                root_steps=snapshot['root_steps'], root_snapshot=snapshot, options=options,
                futures=futures, future_weights=np.ones(repetitions, np.float32),
                blueprint_root=index,
                outcomes={(q, f): dict(duration=400)
                          for q in range(len(options)) for f in range(repetitions)}))
        acquisition = dict(kind='breadth_local_pairs', roots=len(roots),
            repetitions=specification['branch_repetitions'], seed=11262204,
            planned_observations=sum(len(root['outcomes']) for root in roots),
            inherited_labels=False, shared_by_all_estimators=True,
            root_selection='Failure and success states at the first joint-value or ordinary proposal.',
            evidence_role='Additional fixed-pool development training scenes.',
            shard_index=args.shard_index, fixed_pool_sha256=e.digest(DESIGN))
        blueprint = dict(roots=roots, acquisition_design=acquisition,
                         selected_task_indices=indices)
        e.save(blueprint, paired_design)
    result = local.recollect_breadth_design(p.BASE, blueprint,
        OUT / f'added-paired-part{args.shard_index}.pth', workers=1,
        inference_batch=240, inference_device=args.device, root_batch_size=8,
        isolate_factual_start=True, physical_response_horizon=5,
        fixed_inference_width=80, response_feature_batch=80)
    if result is None:
        result = e.read(OUT / f'added-paired-part{args.shard_index}.pth')
    assert result['completed']
    print(json.dumps(dict(phase='added_training_acquisition_complete',
        shard=args.shard_index, roots=len(result['roots']), trajectories=len(bank['records']))), flush=True)


def merge(args):
    _, _, local, _, _ = initialize()
    specification = e.read(DESIGN)
    datasets = [e.read(p.OUT / 'combined-paired.pth')]
    banks = [e.read(p.OUT / 'combined-factual.pth')]
    for index in range(specification['collection_shards']):
        dataset = e.read(OUT / f'added-paired-part{index}.pth')
        bank = e.read(OUT / f'added-factual-part{index}.pth')
        assert dataset['completed'] and bank['completed']
        assert bank['tasks'] == shard_tasks(specification, index)
        datasets.append(dataset)
        banks.append(bank)
    paired = local.merge_paired_data(datasets, allow_distinct_contexts=True)
    paired.update(completed=True, training_data=str(OUT / 'combined-paired.pth'),
                  fixed_development_design_sha256=e.digest(DESIGN))
    bank = copy.deepcopy(banks[0])
    bank['records'] = [record for source in banks for record in source['records']]
    bank['tasks'] = [task for source in banks for task in source['tasks']]
    for key in ('simulation_steps', 'trajectories', 'state_labels'):
        bank[key] = sum(source.get(key, 0) for source in banks)
    bank.update(completed=True, fixed_development_design_sha256=e.digest(DESIGN),
        acquisition_sources=[e.digest(p.OUT / 'combined-factual.pth')] +
            [e.digest(OUT / f'added-factual-part{i}.pth')
             for i in range(specification['collection_shards'])])
    train_layouts = {tuple(scene) for scene, _ in bank['tasks']}
    validation_layouts = {tuple(scene) for scene, _ in specification['validation_tasks']}
    assert not train_layouts & validation_layouts
    for data, name in ((paired, 'combined-paired.pth'), (bank, 'combined-factual.pth')):
        if (OUT / name).exists():
            raise ValueError('A completed pool generation cannot be overwritten.')
        e.save(data, OUT / name)
    print(json.dumps(dict(phase='fixed_pool_complete', training_layouts=len(train_layouts),
        validation_layouts=len(validation_layouts), roots=len(paired['roots']),
        observations=sum(len(root['outcomes']) for root in paired['roots']))), flush=True)


def train(args):
    manifest, _, local, _, _ = initialize()
    data_path, bank_path = OUT / 'combined-paired.pth', OUT / 'combined-factual.pth'
    data, bank = e.read(data_path), e.read(bank_path)
    assert data['completed'] and bank['completed']
    data, bank = local.retain_vessel_identity(data, bank)
    code_sha = p.install_gain_objective(manifest, args.gain_weight, args.harm_weight, args.rank_weight)
    reference = e.read(PROPOSERS[0])
    parameter_budget = reference['successor_task_fit']['matched_success_training']['paired_parameters_per_member']
    stem = args.name or f'pool3000-rank{args.rank_weight:g}-seed{args.seed}'
    arms = ('conditional', 'ordinary') if args.matched_ordinary else ('conditional',)
    paths = [OUT / (stem + '-' + arm + '.pth') for arm in arms]
    if any(path.exists() for path in paths):
        raise ValueError('A trained policy identity cannot be overwritten.')
    if args.matched_ordinary:
        models = local.fit_success_intervention_pair(data, None, None, factual_bank=bank,
            updates=args.updates, ensemble_size=3, width=64, parameter_budget=parameter_budget,
            seed=args.seed, device=args.device, architecture='mlp',
            conditional_objective='joint_likelihood',
            network_options=dict(identity_conditioned=True, action_encoding='categorical'))
    else:
        assert args.updates % 3 == 0
        plain = sys.modules['plain_return']
        members = [plain.fit_paired_mlp_return(data, None, factual_bank=bank,
            updates=args.updates // 3, parameter_budget=parameter_budget,
            seed=args.seed + 997 * member, device=args.device, value_target='success',
            intervention_budget=1, gain_margin=.02, root_sampling_fraction=.5,
            identity_conditioned=True, action_encoding='categorical',
            conditional_objective='joint_likelihood', difference_weight=1.)
            for member in range(3)]
        model = members[0]
        model['successor_task_value']['ensemble_states'] = [
            member['successor_task_value']['state'] for member in members[1:]]
        fits = [member['successor_task_fit'] for member in members]
        for field in ('training_observation_sha256', 'training_target_sha256',
                      'root_sampling_prior_sha256', 'factual_sampling_prior_sha256'):
            assert len({fit[field] for fit in fits}) == 1
        layouts = sorted({tuple(record['scene']) for record in data['roots'] + bank['records']})
        model['successor_task_fit'].update(training_layouts=layouts,
            updates=args.updates, total_parameter_updates=args.updates,
            member_updates=[args.updates // 3] * 3, ensemble_members=3,
            member_seeds=[args.seed + 997 * i for i in range(3)],
            member_training_roots=[len(data['roots'])] * 3,
            member_factual_trajectories=[len(bank['records'])] * 3,
            trainable_parameters=sum(fit['trainable_parameters'] for fit in fits),
            member_initialization_parameter_sha256=[fit['initialization_parameter_sha256'] for fit in fits],
            architecture='3 independently initialized dense members; mean predicted probabilities and effects',
            last_loss=float(np.mean([fit['last_loss'] for fit in fits])))
        models = (model,)
    for arm, model, path in zip(arms, models, paths):
        p.support(model, data, local)
        model['successor_task_value'].update(ensemble_gain_penalty=0., gain_margin=.02)
        settings = copy.deepcopy(reference['joint_value_policy_improvement'])
        settings.pop('parent_policy', None)
        settings.pop('parent_sha256', None)
        settings.update(joint_weight=args.joint_weight,
                        decision_rule_source_sha256=e.digest(PROPOSERS[0]))
        model['joint_value_policy_improvement'] = settings
        model['paired_policy_improvement'] = dict(training_data_sha256=e.digest(data_path),
            factual_data_sha256=e.digest(bank_path), seed=args.seed,
            objective_source_sha256=code_sha, gain_weight=args.gain_weight if arm == 'conditional' else 0.,
            false_positive_harm_weight=args.harm_weight if arm == 'conditional' else 0.,
            common_future_ranking_weight=args.rank_weight if arm == 'conditional' else 0.,
            target='mean same-root same-program common-future terminal difference',
            role='Expanded fixed-pool development training.')
        model['fixed_development_training'] = dict(design_sha256=e.digest(DESIGN),
            training_layouts=len(model['successor_task_fit']['training_layouts']),
            validation_used_for_gradients=False, initialization_repeated_from_scratch=True,
            updates=args.updates, code_sha256=e.digest(Path(__file__)))
        e.save(model, path)
    print(json.dumps(dict(phase='fixed_pool_training_complete', seed=args.seed,
        paths=[str(path) for path in paths])), flush=True)


def evaluate(args):
    _, d, _, _, r = initialize()
    specification = e.read(DESIGN)
    all_tasks = specification['validation_tasks']
    groups = np.array_split(np.arange(len(all_tasks)), args.shards)
    tasks = [all_tasks[int(i)] for i in groups[args.shard_index]]
    policy = Path(args.policy)
    path = OUT / f'{args.name}-part{args.shard_index}-n{len(tasks)}-evaluation.pth'
    if path.exists():
        result = e.read(path)
        assert result['tasks'] == tasks and result['policy_sha256'] == e.digest(policy)
        assert result['completed']
        print(json.dumps(dict(cached=True, path=str(path))), flush=True)
        return
    start = time.monotonic()
    rows = r.batched_credit_rows(policy, tasks, batch_size=args.batch, device=args.device,
        actor_override=joint.load_actor(policy, d), fixed_inference_width=80)
    result = dict(policy=str(policy), policy_sha256=e.digest(policy), tasks=tasks, rows=rows,
        completed=True, success=sum(row['success'] for row in rows),
        elapsed_seconds=time.monotonic() - start, fixed_inference_width=80,
        fixed_development_design_sha256=e.digest(DESIGN), driver_sha256=e.digest(Path(__file__)),
        shard_index=args.shard_index, shards=args.shards,
        role='Repeated fixed development validation; this evaluation is used for model development.')
    e.save(result, path)
    print(json.dumps(dict(path=str(path), success=result['success'], count=len(rows))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('design', 'collect', 'merge', 'train', 'evaluate'))
    parser.add_argument('--name')
    parser.add_argument('--policy')
    parser.add_argument('--count', type=int, default=2000)
    parser.add_argument('--roots', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--shards', type=int, default=8)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--batch', type=int, default=250)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--updates', type=int, default=27000)
    parser.add_argument('--gain-weight', type=float, default=8.)
    parser.add_argument('--harm-weight', type=float, default=4.)
    parser.add_argument('--rank-weight', type=float, default=4.)
    parser.add_argument('--joint-weight', type=float, default=1.28)
    parser.add_argument('--matched-ordinary', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    globals()[args.operation](args)


if __name__ == '__main__':
    main()
