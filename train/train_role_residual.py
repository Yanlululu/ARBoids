"""Bounded policy-improvement study with matched conditional-return supervision."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import warnings

import numpy as np
import torch

from envs.TADgame import TADEnv
from envs.snapshot import preserved_random_state, seed_random
from interaction_rollout import CandidateRolePolicy, compact_snapshot, execute, public_packet
from policy.role_residual import RoleResidualPolicy
from policy.role_value import ConditionalRoleValue, LearnedRoleResidual, compositions, features
from train_interaction import atomic_json, atomic_torch, episode, outcome_row


def initialize():
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore', message='.*Lgh is zero.*')


def collect(task):
    seed, agility, repetitions, behavior_path, continuation = task
    seed_random(seed)
    env = TADEnv(6, protocol='paper-parameters-v1')
    obs, _ = env.reset(agility, noisy_agility=False)
    np.random.seed(seed+1000000)
    prior, residual = CandidateRolePolicy(), RoleResidualPolicy()
    behavior = prior
    if behavior_path is not None:
        saved = torch.load(behavior_path, map_location='cpu', weights_only=True)
        model = ConditionalRoleValue(interaction=saved['arm'] != 'additive')
        model.load_state_dict(saved['model'])
        behavior = LearnedRoleResidual(model, minimum_gain=3.)
    packet, done, step, records = public_packet(env, obs), 0, 0, []
    masks = compositions(6)
    while not done:
        if step in (0, 20, 50, 90, 130):
            snapshot = compact_snapshot(env)
            controls = residual.candidate_controls(packet)
            repeated_labels = []
            with preserved_random_state():
                for repetition in range(repetitions):
                    labels = []
                    future_seed = None if repetition == 0 else (seed*37+step*1009+repetition*100000007) % 2**32
                    for mask in masks:
                        branch = snapshot.restore(future_seed=future_seed)
                        branch_packet, ended, tick = public_packet(branch), 0, 0
                        while not ended:
                            action = (residual.compose(branch_packet, mask) if tick < 10 or continuation == 'sustained' else
                                      prior.choose_action(branch_packet)[0])
                            branch_packet, _, ended, _, _, _ = execute(branch, branch_packet, action)
                            tick += 1
                        result = outcome_row(branch, ended)
                        labels.append([result[k] for k in ('capture_time', 'capture', 'success', 'collision')])
                    repeated_labels.append(labels)
            repeated_labels = np.asarray(repeated_labels, dtype=np.float32)
            records.append(dict(scene_seed=seed, agility=agility, step=step, time=float(env.Current_T),
                features=features(packet, controls), outcomes=repeated_labels.mean(0),
                outcome_repetitions=repeated_labels, behavior='learned' if behavior_path else 'prior'))
        action, _ = behavior.choose_action(packet)
        packet, _, done, _, _, _ = execute(env, packet, action)
        step += 1
    return records


def fit(records, arm, seed, updates):
    seed_random(seed)
    model = ConditionalRoleValue(interaction=arm != 'additive')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    x = torch.tensor(np.stack([r['features'] for r in records]))
    times = torch.tensor(np.stack([r['outcomes'][:, 0] for r in records]))
    starts = torch.tensor([r['time'] for r in records])[:, None]
    target = -(times-starts)/20.
    masks = torch.from_numpy(compositions(6))
    differences = masks[:, None]-masks[None]
    upper, lower = torch.where((differences.abs().sum(-1) == 1) & (differences.sum(-1) == 1))
    rng = np.random.default_rng(seed+1703)
    last = None
    for _ in range(updates):
        indices = rng.choice(len(x), size=min(32, len(x)), replace=False)
        predicted = model(x[indices], masks)
        y = target[indices]
        if arm == 'absolute':
            loss = torch.nn.functional.mse_loss(predicted, y)
        else:
            loss = torch.nn.functional.mse_loss(predicted[:, upper]-predicted[:, lower], y[:, upper]-y[:, lower])
            loss = loss + .05*torch.nn.functional.mse_loss(predicted[:, 0], y[:, 0])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        last = float(loss.detach())
    return model, last


def evaluate_one(task):
    path, threshold, seed, agility = task
    if path is None:
        policy, arm, training_seed = CandidateRolePolicy(), 'prior', None
    else:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = ConditionalRoleValue(interaction=saved['arm'] != 'additive')
        model.load_state_dict(saved['model'])
        policy = LearnedRoleResidual(model, minimum_gain=threshold)
        arm, training_seed = saved['arm'], saved['seed']
    row, _ = episode(policy, seed, 6, agility)
    return dict(row, arm=arm, training_seed=training_seed, minimum_gain=threshold,
        scene_seed=seed, agility=agility,
        decisions=getattr(policy, 'decisions', 0), interventions=getattr(policy, 'interventions', 0))


def summary(rows):
    result = {}
    for arm, seed, threshold, agility in sorted(
            {(r['arm'], r['training_seed'], r['minimum_gain'], r['agility']) for r in rows}, key=str):
        group = [r for r in rows if (r['arm'], r['training_seed'], r['minimum_gain'], r['agility']) ==
                 (arm, seed, threshold, agility)]
        result[f'{arm}-seed{seed}-gain{threshold:g}-a{agility:g}'] = dict(episodes=len(group), **{
            key: float(np.mean([r[key] for r in group])) for key in
            ('capture_time', 'capture', 'success', 'collision', 'interventions', 'decisions')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-scenes-per-cell', type=int, default=12)
    parser.add_argument('--evaluation-scenes-per-cell', type=int, default=32)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--behavior', type=Path)
    parser.add_argument('--continuation', choices=['prior', 'sustained'], default='prior')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Do not overwrite a previous research result.')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(range(24)))
    initialize()
    code = [Path(__file__), Path('train/policy/role_value.py'), Path('train/policy/role_residual.py'),
            Path('train/interaction_rollout.py'), Path('train/envs/TADgame.py')]
    protocol = dict(training_seed=args.seed, train_scenes_per_cell=args.train_scenes_per_cell,
        evaluation_scenes_per_cell=args.evaluation_scenes_per_cell, updates=args.updates,
        arms=['full', 'absolute', 'additive'], thresholds_seconds=[0., 1., 3.],
        block_steps=10, followup=args.continuation,
        objective='negative capped capture time', common_branch_randomness=True,
        deployment_inputs='public observations and current vessel motions only',
        training_bank=1060000000+args.seed*10000, evaluation_bank=1070000000,
        repetitions=args.repetitions, behavior_checkpoint=None if args.behavior is None else str(args.behavior),
        behavior_sha256=None if args.behavior is None else hashlib.sha256(args.behavior.read_bytes()).hexdigest(),
        stage='development', formal_evidence=False,
        source={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), text=p.read_text()) for p in code})
    atomic_json(args.output/'protocol.json', protocol)
    started, records = time.perf_counter(), []
    tasks = [(protocol['training_bank']+cell*1000000+i, a, args.repetitions,
              args.behavior if i % 2 else None, args.continuation)
             for cell, a in enumerate((4., 6.)) for i in range(args.train_scenes_per_cell)]
    with ProcessPoolExecutor(args.workers, initializer=initialize,
                             mp_context=mp.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(collect, task) for task in tasks]):
            records.extend(future.result())
            print(json.dumps(dict(stage='paired_labels', states=len(records))), flush=True)
        records.sort(key=lambda r: (r['scene_seed'], r['step']))
        np.savez_compressed(args.output/'paired-labels.npz',
            features=np.stack([r['features'] for r in records]), outcomes=np.stack([r['outcomes'] for r in records]),
            outcome_repetitions=np.stack([r['outcome_repetitions'] for r in records]),
            behavior=[r['behavior'] for r in records],
            scene_seed=[r['scene_seed'] for r in records], agility=[r['agility'] for r in records],
            step=[r['step'] for r in records], time=[r['time'] for r in records], masks=compositions(6))
        paths = []
        for arm in protocol['arms']:
            model, loss = fit(records, arm, args.seed, args.updates)
            path = args.output/(arm+'.pth')
            atomic_torch(path, dict(format='conditional-role-value-v1', arm=arm, seed=args.seed,
                model=model.state_dict(), training_loss=loss, protocol_file=str(args.output/'protocol.json')))
            paths.append(path)
            print(json.dumps(dict(stage='trained', arm=arm, states=len(records), loss=loss)), flush=True)
        evaluations = [(path, threshold, protocol['evaluation_bank']+cell*1000000+i, a)
            for cell, a in enumerate((4., 6.)) for i in range(args.evaluation_scenes_per_cell)
            for path, threshold in [(None, 0.)]+[(p, t) for p in paths for t in protocol['thresholds_seconds']]]
        rows = []
        for future in as_completed([pool.submit(evaluate_one, task) for task in evaluations]):
            rows.append(future.result())
            if len(rows)%64 == 0:
                atomic_json(args.output/'results.json', dict(complete=False, episodes=rows, summary=summary(rows)))
                print(json.dumps(dict(stage='tasks', completed=len(rows), total=len(evaluations))), flush=True)
    for name, saved in protocol['source'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != saved['sha256']:
            raise RuntimeError('Frozen source changed: '+name)
    result = dict(complete=True, states=len(records), episodes=rows, summary=summary(rows),
        wall_seconds=time.perf_counter()-started, formal_evidence=False)
    atomic_json(args.output/'results.json', result)
    print(json.dumps(dict(complete=True, summary=result['summary'], wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
