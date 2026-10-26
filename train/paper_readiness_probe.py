"""Bounded prospective checks before committing to the full journal matrix."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import subject_centered_cv as centered

e = centered.e
OUT = centered.OUT
DESIGN = OUT / 'paper-readiness-controls-design.pth'


def prepare():
    _, _, _, mechanisms, _ = centered.initialize()
    if DESIGN.exists():
        raise ValueError('The readiness comparison is already frozen.')
    cohort_path = OUT / 'paper-readiness-b-independent-design.pth'
    cohort = e.read(cohort_path)
    indices = list(range(250)) + list(range(500, 750))
    tasks = [cohort['tasks'][i] for i in indices]
    joint_path = centered.fixed.p.BASE
    independent_path = joint_path.parent / 'best-independent-400.pth'
    joint, independent = e.read(joint_path), e.read(independent_path)
    assert set(joint['actor']) == set(independent['actor'])
    assert all(torch.equal(value, independent['actor'][key])
               for key, value in joint['actor'].items())
    first, second = (dict(model['actuator_fusion_refinement'])
                     for model in (joint, independent))
    assert first.pop('selection') == 'joint'
    assert second.pop('selection') == 'independent' and first == second
    assert all(joint.get(key) == independent.get(key)
               for key in ('levels', 'risk_prior', 'risk_horizon', 'risk_clearance',
                           'risk_feedback', 'mechanism_ablation', 'source_sha256'))
    models = {'ci-independent': dict(path=str(independent_path),
        sha256=e.digest(independent_path), joint_reference=str(joint_path),
        joint_sha256=e.digest(joint_path),
        comparison='Same candidates and learned source; joint versus independent selection.')}
    directory = e.BASE / 'independent-training/source-six-arboids-seed42-20261021'
    training, _ = mechanisms.validate_author_source_repeat(directory)
    assert training['completed'] and training['training_steps'] == 1000000
    path = directory / 'best-development.pth'
    models['arboids'] = dict(path=str(path), directory=str(directory), seed=42,
        sha256=e.digest(path), training_steps=training['training_steps'])
    assert models['arboids']['sha256'] == training['selected_checkpoint_sha256']
    directory = e.BASE / 'r20-six-baseline-comparison/mappo-seed42'
    training = e.read(directory / 'training-result.pth')
    path = directory / 'best.pth'
    assert training['completed'] and training['steps'] == training['budget'] == 1000000
    models['mappo'] = dict(path=str(path), seed=42, sha256=e.digest(path),
                          training_steps=training['steps'])
    assert models['mappo']['sha256'] == training['best_sha256']
    baseline_runtime = e.baseline_runtime()
    result = dict(tasks=tasks, cohort_indices=indices, models=models,
        cohort_path=str(cohort_path), cohort_sha256=e.digest(cohort_path),
        source_sha256=e.digest(Path(__file__)),
        centered_source_sha256=e.digest(Path(centered.__file__)),
        baseline_helpers_sha256=baseline_runtime['helpers_sha256'],
        baseline_endpoint_sha256=baseline_runtime['endpoint_sha256'],
        frozen_unix=time.time(), role='Development readiness probe, not the full formal evidence matrix.',
        baseline_seed_selection='Existing seed 42, fixed before readiness outcomes.',
        scope='Three matched controls on 500 balanced tasks; no new baseline training.')
    e.save(result, DESIGN)
    print(json.dumps(dict(design=str(DESIGN), tasks=len(tasks), models=list(models))), flush=True)


def evaluate(args):
    _, development, _, _, runtime = centered.initialize()
    design = e.read(DESIGN)
    assert design['source_sha256'] == e.digest(Path(__file__))
    assert design['centered_source_sha256'] == e.digest(Path(centered.__file__))
    assert design['cohort_sha256'] == e.digest(Path(design['cohort_path']))
    model = design['models'][args.method]
    policy = Path(model['path'])
    assert model['sha256'] == e.digest(policy)
    indices = np.array_split(np.arange(len(design['tasks'])), args.shards)[args.shard_index]
    tasks = [design['tasks'][int(i)] for i in indices]
    target = OUT / f'paper-readiness-control-{args.method}-part{args.shard_index}-evaluation.pth'
    if target.exists():
        old = e.read(target)
        assert old['completed'] and old['tasks'] == tasks
        assert old['design_sha256'] == e.digest(DESIGN)
        return
    start = time.monotonic()
    if args.method == 'ci-independent':
        rows = runtime.batched_credit_rows(policy, tasks, batch_size=125,
            actor_override=centered.load_actor(policy, development),
            device='cuda:0', fixed_inference_width=80)
    else:
        baseline_runtime = e.baseline_runtime()
        assert baseline_runtime['helpers_sha256'] == design['baseline_helpers_sha256']
        assert baseline_runtime['endpoint_sha256'] == design['baseline_endpoint_sha256']
        location = model.get('directory', model['path'])
        rows = [e.baseline_episode((args.method, location, model['sha256'], scene, future))
                for scene, future in tasks]
    assert len(rows) == len(tasks)
    for row, (scene, future) in zip(rows, tasks):
        assert (row['seed'], row['defenders'], row['agility']) == tuple(scene)
        assert future is None and row['environment_protocol'] == 'source'
    e.save(dict(completed=True, tasks=tasks, rows=rows, policy=str(policy),
        policy_sha256=model['sha256'], method=args.method, design_sha256=e.digest(DESIGN),
        shard_index=args.shard_index, shards=args.shards,
        success=sum(row['success'] for row in rows), elapsed_seconds=time.monotonic()-start,
        source_sha256=e.digest(Path(__file__)), role=design['role']), target)
    print(json.dumps(dict(method=args.method, shard=args.shard_index,
                         completed=True, tasks=len(tasks))), flush=True)


def capability():
    _, development, _, _, _ = centered.initialize()
    policy = OUT / 'pool3000-centered-rank4-seed20261104-conditional.pth'
    records = []
    for defenders in (6, 4, 8):
        actor = centered.load_actor(policy, development)
        decision = actor.successor_task_value
        try:
            if defenders != decision.network.defenders:
                decision.configure_fleet(defenders)
            result = dict(defenders=defenders, representation_supported=True,
                          catalog_size=len(decision.options))
        except (AttributeError, ValueError, RuntimeError) as error:
            result = dict(defenders=defenders, representation_supported=False,
                          error_type=type(error).__name__, reason=str(error))
        records.append(result)
    target = OUT / 'paper-readiness-fleet-capability.pth'
    assert not target.exists()
    e.save(dict(completed=True, policy=str(policy), policy_sha256=e.digest(policy),
        records=records, source_sha256=e.digest(Path(__file__)),
        role='Representation deployment check only; no fleet-generalization performance claim.'), target)
    print(json.dumps(records), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'evaluate', 'capability'))
    parser.add_argument('--method', choices=('ci-independent', 'arboids', 'mappo'))
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if args.operation == 'evaluate':
        if args.method is None or args.shards <= 0 or not 0 <= args.shard_index < args.shards:
            parser.error('Evaluation requires a method and valid shard index.')
        evaluate(args)
    else:
        globals()[args.operation]()


if __name__ == '__main__':
    main()
