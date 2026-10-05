"""Bounded follow-up: earlier impulses, matched model audits and rolling gates."""
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

from diagnose_gate_space import code_hashes, new_output, write_csv, write_json
from envs.TADgame import TADEnv
from gate_diagnostics import FrozenARBoids, collect_cases, diagnose_case, digest
from gate_followup import audit_case, run_controlled


_policy = None


def initialize_worker(checkpoint):
    global _policy
    torch.set_num_threads(1)
    _policy = FrozenARBoids(checkpoint)


def case_job(arguments):
    case, search_seeds, validation_seeds, audit_repeats, settings, warning, penalty = arguments
    summary, timing = diagnose_case(case, _policy, search_seeds, validation_seeds,
                                    horizon=2., safe_distance=warning, align_future_step=True)
    audits, forecasts = audit_case(case, _policy, validation_seeds[:audit_repeats])
    rolling = []
    for seed in validation_seeds:
        reference = run_controlled(case, _policy, future_seed=seed).summary
        for horizon, active_steps in settings:
            result = run_controlled(case, _policy, mode='rolling', active_steps=active_steps,
                                    future_seed=seed, horizon=horizon, safe_distance=warning,
                                    change_penalty=penalty).summary
            row = dict(case_id=case.case_id, source_seed=case.source_seed, kind=case.kind,
                       lookback_seconds=case.lookback_seconds, horizon=horizon,
                       active_steps=active_steps, future_seed=seed, **result)
            for key in ('success', 'collision', 'breach', 'team_return'):
                row['reference_' + key] = reference[key]
                row[key + '_delta'] = result[key] - reference[key]
            rolling.append(row)
    return dict(timing_cases=[summary], timing_branches=timing, model_audits=audits,
                pulse_forecasts=forecasts, rolling_branches=rolling)


def aggregate_rolling(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['kind'], row['lookback_seconds'], row['horizon'], row['active_steps'])].append(row)
    output = []
    for (kind, lead, horizon, active), members in groups.items():
        sources = defaultdict(list)
        for member in members:
            sources[member['source_seed']].append(member)
        result = dict(kind=kind, lookback_seconds=lead, horizon=horizon, active_steps=active,
                      source_episodes=len(sources), paired_continuations=len(members))
        for key in ('success', 'collision', 'breach', 'team_return',
                    'reference_success', 'reference_collision', 'reference_breach', 'reference_team_return',
                    'success_delta', 'collision_delta', 'breach_delta', 'team_return_delta'):
            result[key] = float(np.mean([np.mean([r[key] for r in group]) for group in sources.values()]))
        output.append(result)
    return output


def run(args):
    started = time.monotonic()
    policy = FrozenARBoids(args.checkpoint)
    source_rows = list(csv.DictReader(args.source_episodes.open(encoding='utf-8')))
    seeds = [int(row['source_seed']) for row in source_rows]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Source episode table must contain distinct scene seeds.')
    expected = {int(row['source_seed']): row['trajectory_sha256'] for row in source_rows}
    hashes = code_hashes()
    hashes.update({name: digest(Path(__file__).parent / name)
                   for name in ('gate_followup.py', 'diagnose_gate_followup.py')})
    cases, sources = collect_cases(policy, seeds,
        env_factory=lambda: TADEnv(protocol='paper-parameters-v1'), lookbacks=(1., 2., 3.),
        collision_episodes=len(seeds), normal_episodes=len(seeds), sampling_seed=42)
    if any(row['trajectory_sha256'] != expected[row['source_seed']] for row in sources):
        raise RuntimeError('The original source episodes did not reproduce exactly.')
    print(f'[REPLAY] {len(sources)} original trajectories verified; {len(cases)} states collected.', flush=True)
    settings = [(2., 1), (2., 5), (2., 10), (1., 10), (3., 10)]
    if args.entire_episode:
        settings.extend([(1., None), (2., None), (3., None)])
    specification = dict(checkpoint=str(policy.path), checkpoint_sha256=policy.sha256,
                         code_sha256=hashes, source_table_sha256=digest(args.source_episodes),
                         source_seeds=seeds, lookbacks=[1., 2., 3.],
                         search_repeats=args.search_repeats, validation_repeats=args.validation_repeats,
                         audit_repeats=args.audit_repeats, continuation_seed=args.continuation_seed,
                         rolling_settings=settings, warning_distance=args.warning_distance,
                         change_penalty=args.change_penalty, noise_alignment='absolute control step',
                         statistical_scope='Development study; states nested in source episodes, '
                                           'future seeds paired across settings. No policy training.')
    new_output(args.output_dir)
    write_csv(args.output_dir / 'sources.csv', sources)
    with (args.output_dir / 'cases.pkl').open('wb') as file:
        pickle.dump(dict(specification=specification, cases=cases), file, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(args.output_dir / 'specification.json', specification)
    arguments = []
    for case in cases:
        base = args.continuation_seed + seeds.index(case.source_seed) * 1000
        arguments.append((case, list(range(base, base + args.search_repeats)),
                          list(range(base + 100, base + 100 + args.validation_repeats)),
                          args.audit_repeats, settings, args.warning_distance, args.change_penalty))
    collected = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize_worker, initargs=(str(policy.path),)) as pool:
        futures = {pool.submit(case_job, argument): argument[0].case_id for argument in arguments}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            for name, rows in result.items():
                path = args.output_dir / (name + '.csv')
                with path.open('a', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    if not collected[name]:
                        writer.writeheader()
                    writer.writerows(rows)
                collected[name].extend(rows)
            print(f'[DONE] {i}/{len(cases)} {futures[future]}', flush=True)
    if digest(policy.path) != policy.sha256 or any(digest(Path(__file__).parent / name) != value
                                                for name, value in hashes.items()):
        raise RuntimeError('Code or checkpoint changed during the follow-up.')
    aggregates = aggregate_rolling(collected['rolling_branches'])
    write_csv(args.output_dir / 'rolling_summary.csv', aggregates)
    matched = [row for row in collected['model_audits'] if row['execution'] == 'matched_zero_current']
    maximum_error = max(row['maximum_position_error'] for row in matched)
    result = dict(passed=maximum_error < 1e-7, completed=True, **specification,
                  source_replay_verified=len(sources), cases=len(cases),
                  collision_cases=sum(case.kind == 'collision' for case in cases),
                  normal_cases=sum(case.kind == 'normal' for case in cases),
                  matched_model_max_position_error=maximum_error,
                  impulse_improvement_cases=[row['case_id'] for row in collected['timing_cases']
                                             if row['validation_observed_improvement']],
                  row_counts={name: len(rows) for name, rows in collected.items()},
                  wall_seconds=time.monotonic() - started)
    write_json(args.output_dir / 'summary.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result['passed']:
        raise RuntimeError('Matched-control nominal model verification failed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-episodes', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--search-repeats', type=int, default=2)
    parser.add_argument('--validation-repeats', type=int, default=8)
    parser.add_argument('--audit-repeats', type=int, default=4)
    parser.add_argument('--continuation-seed', type=int, default=99000000)
    parser.add_argument('--warning-distance', type=float, default=7.)
    parser.add_argument('--change-penalty', type=float, default=.02)
    parser.add_argument('--entire-episode', action='store_true', help='Also enable rolling gates for the remaining episode.')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.search_repeats <= 100 or not 1 <= args.validation_repeats <= 100:
        parser.error('Use 1-8 workers and 1-100 repeats.')
    if not 1 <= args.audit_repeats <= args.validation_repeats or args.continuation_seed < 0:
        parser.error('Audit repeats must fit the validation set; seed must be nonnegative.')
    torch.set_num_threads(1)
    try:
        run(args)
    except Exception as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        raise


if __name__ == '__main__':
    main()
