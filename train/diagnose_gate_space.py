"""CLI for collecting ARBoids snapshots and diagnosing gate-space opportunities."""
import argparse
import csv
import json
from pathlib import Path
import pickle

import torch

from envs.TADgame import TADEnv
from gate_diagnostics import FrozenARBoids, collect_cases, diagnose_case, digest, summarize_cases


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def check_output(path):
    if path.exists() and not path.is_dir():
        raise ValueError('Output path must be a directory.')
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output directory must be empty; existing diagnostic data is not overwritten.')


def new_output(path):
    check_output(path)
    path.mkdir(parents=True, exist_ok=True)


def code_hashes():
    root = Path(__file__).parent
    files = ('diagnose_gate_space.py', 'gate_diagnostics.py', 'envs/snapshot.py',
             'envs/TADgame.py', 'envs/modules.py', 'policy/networks.py', 'policy/interaction_prediction.py')
    return {name: digest(root / name) for name in files}


def collect(args):
    check_output(args.output_dir)
    policy = FrozenARBoids(args.checkpoint)
    hashes = code_hashes()
    cases, sources = collect_cases(
        policy, range(args.first_seed, args.first_seed + args.episodes),
        env_factory=lambda: TADEnv(defender_num=args.defenders, protocol='paper-parameters-v1'),
        lookbacks=args.lookbacks, collision_episodes=args.collision_episodes,
        normal_episodes=args.normal_episodes, sampling_seed=args.sampling_seed, agility=args.agility,
        noisy_agility=args.noisy_agility,
        progress=lambda done, total: print(f'[COLLECT] {done}/{total} source episodes', flush=True)
        if done % 10 == 0 or done == total else None)
    if digest(policy.path) != policy.sha256 or code_hashes() != hashes:
        raise RuntimeError('Baseline checkpoint or simulator code changed during collection.')
    specification = dict(version=1, checkpoint=str(policy.path), checkpoint_sha256=policy.sha256,
                         code_sha256=hashes, protocol='paper-parameters-v1', defenders=args.defenders,
                         agility=args.agility, noisy_agility=args.noisy_agility,
                         first_seed=args.first_seed, source_episodes=args.episodes,
                         lookbacks=args.lookbacks, sampling_seed=args.sampling_seed,
                         collision_episode_quota=args.collision_episodes, normal_episode_quota=args.normal_episodes)
    new_output(args.output_dir)
    with (args.output_dir / 'cases.pkl').open('wb') as file:
        pickle.dump(dict(specification=specification, cases=cases), file, protocol=pickle.HIGHEST_PROTOCOL)
    write_csv(args.output_dir / 'sources.csv', sources)
    summary = dict(passed=True, completed=True, phase='collect', **specification,
                   cases=len(cases), diagnosis_ready=bool(cases),
                   source_outcomes={key: sum(row[key] for row in sources) for key in ('success', 'collision', 'breach')},
                   selected_cases=[dict(case_id=case.case_id, source_seed=case.source_seed,
                       source_step=case.source_step, kind=case.kind, lookback_seconds=case.lookback_seconds,
                       baseline_action=case.action.tolist(),
                       boids_thrust=case.snapshot.environment.boids_actions.tolist()) for case in cases])
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(dict(passed=True, cases=len(cases), source_outcomes=summary['source_outcomes'])), flush=True)


def search(args):
    check_output(args.output_dir)
    # Only consume local trusted cases.pkl files produced by this entry point.
    with args.cases.open('rb') as file:
        bundle = pickle.load(file)
    specification, cases = bundle['specification'], bundle['cases']
    if specification.get('version') != 1 or specification['code_sha256'] != code_hashes():
        raise ValueError('Snapshot code version differs; recollect with the current simulator.')
    if not cases:
        raise ValueError('No source cases were collected; inspect source outcomes and quotas.')
    policy = FrozenARBoids(args.checkpoint or specification['checkpoint'])
    if policy.sha256 != specification['checkpoint_sha256']:
        raise ValueError('Search requires the exact frozen checkpoint used for collection.')
    block = args.search_repeats + args.validation_repeats
    if args.continuation_seed + len(cases) * block > 2**32:
        raise ValueError('Continuation seed range exceeds the NumPy seed limit.')
    new_output(args.output_dir)
    summaries = []
    with (args.output_dir / 'branches.csv').open('w', newline='', encoding='utf-8') as file:
        writer = None
        for index, case in enumerate(cases):
            seed = args.continuation_seed + index * block
            def progress(done, total):
                if done % 50 == 0 or done == total:
                    print(f'[SEARCH] case {index+1}/{len(cases)}, branches {done}/{total}', flush=True)
            summary, rows = diagnose_case(
                case, policy, range(seed, seed + args.search_repeats),
                range(seed + args.search_repeats, seed + block), grid=args.grid,
                horizon=args.horizon, safe_distance=args.safe_distance,
                max_combinations=args.max_combinations, progress=progress)
            summaries.append(summary)
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            file.flush()
    if digest(policy.path) != policy.sha256 or specification['code_sha256'] != code_hashes():
        raise RuntimeError('Baseline checkpoint or simulator code changed during diagnosis.')
    write_csv(args.output_dir / 'cases.csv', summaries)
    result = dict(passed=True, completed=True, phase='search',
                  checkpoint_sha256=policy.sha256, source_cases_sha256=digest(args.cases),
                  code_sha256=specification['code_sha256'], cases=len(cases), grid=args.grid,
                  continuation_seed=args.continuation_seed, search_repeats=args.search_repeats,
                  validation_repeats=args.validation_repeats, prediction_horizon=args.horizon,
                  prediction_safe_distance=args.safe_distance, intervention_control_steps=1,
                  distance_sample_period_seconds=cases[0].snapshot.environment.Action_T,
                  prediction_sample_period_seconds=.05,
                  selection_rule='On search futures: preserve observed success/breach; minimize collision; '
                                 'then maximize success, minimize breach, maximize return and short clearance; '
                                 'then prefer less gate change.',
                  interpretation='Offline state-conditioned opportunity diagnosis. Validation uses new '
                                 'future seeds at the same states, not new deployment episodes. '
                                 'Deltas are descriptive and grouped by source episode.',
                  strata=summarize_cases(summaries))
    write_json(args.output_dir / 'summary.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect_parser = commands.add_parser('collect', help='Save frozen baseline cases and full snapshots.')
    collect_parser.add_argument('--checkpoint', type=Path, required=True)
    collect_parser.add_argument('--output-dir', type=Path, required=True)
    collect_parser.add_argument('--first-seed', type=int, required=True)
    collect_parser.add_argument('--episodes', type=int, default=100)
    collect_parser.add_argument('--defenders', type=int, default=3)
    collect_parser.add_argument('--agility', type=float, default=2.)
    collect_parser.add_argument('--noisy-agility', action='store_true')
    collect_parser.add_argument('--collision-episodes', type=int, default=5)
    collect_parser.add_argument('--normal-episodes', type=int, default=5)
    collect_parser.add_argument('--lookbacks', type=float, nargs='+', default=[1., 2., 3.])
    collect_parser.add_argument('--sampling-seed', type=int, default=0)
    search_parser = commands.add_parser('search', help='Enumerate fixed candidates and validate selected gates.')
    search_parser.add_argument('--cases', type=Path, required=True)
    search_parser.add_argument('--checkpoint', type=Path, help='Relocated identical baseline checkpoint.')
    search_parser.add_argument('--output-dir', type=Path, required=True)
    search_parser.add_argument('--continuation-seed', type=int, required=True)
    search_parser.add_argument('--search-repeats', type=int, default=4)
    search_parser.add_argument('--validation-repeats', type=int, default=8)
    search_parser.add_argument('--grid', type=float, nargs='+', default=[0., .25, .5, .75, 1.])
    search_parser.add_argument('--horizon', type=float, default=2.)
    search_parser.add_argument('--safe-distance', type=float, default=7.)
    search_parser.add_argument('--max-combinations', type=int, default=4096)
    args = parser.parse_args()
    if args.command == 'collect':
        if args.episodes < 1 or args.defenders < 2 or not 0 <= args.first_seed < args.first_seed + args.episodes <= 2**32:
            parser.error('Use positive episodes, at least two defenders and valid source seeds.')
    elif min(args.search_repeats, args.validation_repeats, args.max_combinations) < 1 or args.continuation_seed < 0:
        parser.error('Use positive repeat/combination counts and a nonnegative continuation seed.')
    torch.set_num_threads(1)
    try:
        (collect if args.command == 'collect' else search)(args)
    except Exception as error:
        print(json.dumps(dict(passed=False, error=str(error))), flush=True)
        raise


if __name__ == '__main__':
    main()
