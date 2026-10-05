"""Evaluate persistent warning-triggered gates on saved states and scene starts."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from diagnose_gate_followup import aggregate_rolling
from diagnose_gate_space import new_output, write_csv, write_json
from envs.TADgame import TADEnv
from envs.snapshot import SimulationSnapshot, preserved_random_state, seed_random
from gate_diagnostics import FrozenARBoids, GateCase, digest, validate_action
from gate_followup import run_controlled


_policy = None


def initialize_worker(checkpoint):
    global _policy
    torch.set_num_threads(1)
    _policy = FrozenARBoids(checkpoint)


def job(arguments):
    scope, case, seeds, horizons, warning, penalty = arguments
    rows = []
    for seed in seeds:
        reference = run_controlled(case, _policy, future_seed=seed).summary
        for horizon in horizons:
            result = run_controlled(case, _policy, mode='rolling', active_steps=None,
                                    future_seed=seed, horizon=horizon, safe_distance=warning,
                                    change_penalty=penalty).summary
            row = dict(scope=scope, case_id=case.case_id, source_seed=case.source_seed,
                       kind=case.kind, lookback_seconds=case.lookback_seconds,
                       start_time=float(case.snapshot.environment.Current_T),
                       horizon=horizon, active_steps=None, future_seed=seed, **result)
            for key in ('success', 'collision', 'breach', 'team_return'):
                row['reference_' + key] = reference[key]
                row[key + '_delta'] = result[key] - reference[key]
            row['reference_end_time'] = reference['end_time']
            row['reference_trajectory_sha256'] = reference['trajectory_sha256']
            row['warning_lead_to_reference_collision'] = (
                reference['end_time'] - result['first_warning_time']
                if reference['collision'] and result['first_warning_time'] is not None else None)
            rows.append(row)
    return rows


def episode_start_cases(policy, source_rows):
    cases = []
    with preserved_random_state():
        for row in source_rows:
            seed = int(row['source_seed'])
            seed_random(seed)
            env = TADEnv(protocol='paper-parameters-v1')
            observation, _ = env.reset(2., False)
            kind = 'collision' if int(row['collision']) else 'normal'
            case = GateCase(seed, 0, kind, SimulationSnapshot.capture(env),
                            observation.copy(), validate_action(policy(observation), env.defender_num),
                            source_trajectory_sha256=row['trajectory_sha256'])
            if run_controlled(case, policy).summary['trajectory_sha256'] != row['trajectory_sha256']:
                raise RuntimeError(f'Original scene {seed} did not replay from its start.')
            cases.append(case)
    return cases


def run(args):
    started = time.monotonic()
    # These are trusted local snapshots emitted by diagnose_gate_followup.py.
    with args.cases.open('rb') as file:
        saved = pickle.load(file)
    original = saved['specification']
    hashes = original['code_sha256'].copy()
    hashes[Path(__file__).name] = digest(Path(__file__))
    code_root = Path(__file__).parent
    if any(digest(code_root / name) != value for name, value in hashes.items()):
        raise RuntimeError('Source code differs from the captured follow-up study.')
    policy = FrozenARBoids(args.checkpoint or original['checkpoint'])
    if policy.sha256 != original['checkpoint_sha256']:
        raise RuntimeError('Checkpoint content differs from the captured study.')
    source_rows = list(csv.DictReader((args.cases.parent / 'sources.csv').open(encoding='utf-8')))
    tasks = [('captured_state', case) for case in saved['cases']]
    starts = episode_start_cases(policy, source_rows)
    tasks.extend(('episode_start', case) for case in starts)
    specification = dict(checkpoint=str(policy.path), checkpoint_sha256=policy.sha256,
                         code_sha256=hashes, cases_sha256=digest(args.cases),
                         source_table_sha256=digest(args.cases.parent / 'sources.csv'),
                         source_seeds=original['source_seeds'], horizons=args.horizons,
                         validation_repeats=original['validation_repeats'],
                         continuation_seed=original['continuation_seed'],
                         warning_distance=args.warning_distance, change_penalty=args.change_penalty,
                         active_steps=None, noise_alignment='absolute control step',
                         statistical_scope='Development scenes reused; not an independent test set. '
                                           'Episode-start mode uses no future collision time.')
    new_output(args.output_dir)
    write_json(args.output_dir / 'specification.json', specification)
    arguments = []
    for scope, case in tasks:
        base = original['continuation_seed'] + original['source_seeds'].index(case.source_seed) * 1000 + 100
        seeds = list(range(base, base + original['validation_repeats']))
        arguments.append((scope, case, seeds, args.horizons, args.warning_distance, args.change_penalty))
    rows = []
    print(f'[REPLAY] {len(starts)} source starts verified; {len(tasks)} evaluation states.', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(str(policy.path),)) as pool:
        futures = {pool.submit(job, argument): (argument[0], argument[1].case_id) for argument in arguments}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with (args.output_dir / 'branches.csv').open('a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=list(result[0]))
                if not rows:
                    writer.writeheader()
                writer.writerows(result)
            rows.extend(result)
            print(f'[DONE] {i}/{len(tasks)} {futures[future]}', flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['scope']].append(row)
    aggregates = [dict(scope=scope, **row) for scope, group in grouped.items()
                  for row in aggregate_rolling(group)]
    write_csv(args.output_dir / 'summary.csv', aggregates)
    if digest(policy.path) != policy.sha256 or any(digest(code_root / name) != value for name, value in hashes.items()):
        raise RuntimeError('Code or checkpoint changed during evaluation.')
    result = dict(passed=True, completed=True, **specification, source_replay_verified=len(starts),
                  states=len(tasks), paired_comparisons=len(rows), wall_seconds=time.monotonic() - started)
    write_json(args.output_dir / 'summary.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--horizons', type=float, nargs='+', default=[1., 2., 3.])
    parser.add_argument('--warning-distance', type=float, default=7.)
    parser.add_argument('--change-penalty', type=float, default=.02)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use 1-8 workers.')
    if (not np.isfinite(args.horizons).all() or min(args.horizons) <= 0
            or len(set(args.horizons)) != len(args.horizons)):
        parser.error('Use distinct, finite, positive horizons.')
    if not np.isfinite([args.warning_distance, args.change_penalty]).all() or args.warning_distance <= 5 or args.change_penalty < 0:
        parser.error('Use a warning distance above 5 and a nonnegative finite penalty.')
    torch.set_num_threads(1)
    try:
        run(args)
    except Exception as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        raise


if __name__ == '__main__':
    main()
