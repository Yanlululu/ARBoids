"""A fixed, paired scene suite for gate-coordination paper tables and ablations."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from diagnose_gate_space import code_hashes, new_output, write_json
from envs.TADgame import TADEnv
from envs.snapshot import SimulationSnapshot, preserved_random_state, seed_random
from gate_diagnostics import FrozenARBoids, GateCase, digest, validate_action
from gate_followup import run_controlled


METHODS = [
    dict(name='ARBoids', mode='baseline', horizon=3., gate=None),
    dict(name='Boids only', mode='constant', horizon=3., gate=0.),
    dict(name='Actor only', mode='constant', horizon=3., gate=1.),
    dict(name='Fixed blend', mode='constant', horizon=3., gate=.5),
    dict(name='Single correction', mode='rolling_once', horizon=3., gate=None),
    dict(name='Rolling H1', mode='rolling', horizon=1., gate=None),
    dict(name='Rolling H2', mode='rolling', horizon=2., gate=None),
    dict(name='Rolling H3', mode='rolling', horizon=3., gate=None),
]
_policy = None


def initialize_worker(checkpoint):
    global _policy
    torch.set_num_threads(1)
    _policy = FrozenARBoids(checkpoint)


def initial_case(policy, scene_seed):
    with preserved_random_state():
        seed_random(scene_seed)
        env = TADEnv(protocol='paper-parameters-v1')
        observation, _ = env.reset(2., False)
        return GateCase(scene_seed, 0, 'scene', SimulationSnapshot.capture(env),
                        observation.copy(), validate_action(policy(observation), env.defender_num))


def run_method(case, policy, method, future_seed):
    theta = None if method['gate'] is None else np.full(case.theta.shape, method['gate'])
    return run_controlled(case, policy, mode=method['mode'], theta=theta, active_steps=None,
                          horizon=method['horizon'], future_seed=future_seed,
                          safe_distance=7., change_penalty=.02)


def scene_job(argument):
    scene_seed, future_seed = argument
    case = initial_case(_policy, scene_seed)
    rows = []
    reference = None
    for method in METHODS:
        start = time.perf_counter()
        result = run_method(case, _policy, method, future_seed).summary
        elapsed = time.perf_counter() - start
        if reference is None:
            reference = result
        row = dict(scene_seed=scene_seed, future_seed=future_seed, method=method['name'],
                   horizon=method['horizon'], wall_seconds=elapsed, **result)
        for key in ('success', 'collision', 'breach', 'capture', 'timeout', 'team_return', 'end_time'):
            row['reference_' + key] = reference[key]
            row[key + '_delta'] = result[key] - reference[key]
        row['reference_trajectory_sha256'] = reference['trajectory_sha256']
        rows.append(row)
    # Repeat the baseline exactly to catch accidental mutation/RNG leakage.
    repeated = run_method(case, _policy, METHODS[0], future_seed).summary
    if repeated['trajectory_sha256'] != reference['trajectory_sha256']:
        raise RuntimeError(f'Baseline replay changed for scene {scene_seed}.')
    return rows


def run(args):
    start = time.monotonic()
    policy = FrozenARBoids(args.checkpoint)
    hashes = code_hashes()
    for name in ('gate_followup.py', 'evaluate_gate_paper.py'):
        hashes[name] = digest(Path(__file__).parent / name)
    specification = dict(checkpoint=str(policy.path), checkpoint_sha256=policy.sha256,
        code_sha256=hashes, source_episodes=args.episodes, first_scene_seed=args.first_scene_seed,
        first_future_seed=args.first_future_seed, protocol='paper-parameters-v1',
        defenders=3, attacker='APF', agility=2., noisy_agility=False,
        control_dt=.2, integration_dt=.05, warning_distance=7., collision_radius=5.,
        change_penalty=.02, grid=[0., .25, .5, .75, 1.], methods=METHODS,
        primary_comparison=['Rolling H3', 'ARBoids'],
        evaluation='All methods run from the same scene start with paired exogenous noise. '
                   'One complete episode per distinct scene per method. No policy training.')
    new_output(args.output_dir)
    write_json(args.output_dir / 'specification.json', specification)
    arguments = [(args.first_scene_seed+i, args.first_future_seed+i) for i in range(args.episodes)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(str(policy.path),)) as pool:
        futures = {pool.submit(scene_job, argument): argument[0] for argument in arguments}
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with (args.output_dir / 'episodes.csv').open('a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=list(result[0]))
                if not rows:
                    writer.writeheader()
                writer.writerows(result)
            rows.extend(result)
            if count % 8 == 0 or count == args.episodes:
                print(f'[EVALUATED] {count}/{args.episodes} scenes; {len(rows)} method episodes.', flush=True)
    if digest(policy.path) != policy.sha256 or any(digest(Path(__file__).parent / name) != value
                                                 for name, value in hashes.items()):
        raise RuntimeError('Code or checkpoint changed during the evaluation.')
    if len(rows) != args.episodes * len(METHODS) or not all(row['terminated'] for row in rows):
        raise RuntimeError('Incomplete evaluation suite.')
    result = dict(passed=True, completed=True, **specification, method_episodes=len(rows),
                  exact_baseline_replays=args.episodes, wall_seconds=time.monotonic()-start)
    write_json(args.output_dir / 'summary.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--first-scene-seed', type=int, default=108000000)
    parser.add_argument('--first-future-seed', type=int, default=118000000)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 10000 or not 1 <= args.workers <= 8:
        parser.error('Use 1-10000 scenes and 1-8 workers.')
    if min(args.first_scene_seed,args.first_future_seed)<0 or max(args.first_scene_seed,args.first_future_seed)+args.episodes>=2**32:
        parser.error('Seeds must fit the NumPy seed range.')
    torch.set_num_threads(1)
    try:
        run(args)
    except Exception as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        raise


if __name__ == '__main__':
    main()
