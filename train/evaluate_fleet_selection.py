"""Finite fleet-size/parameter screen, then frozen paired new-scene validation."""
import study_runtime
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

import numpy as np

import evaluate_feedback_joint as feedback_study
import evaluate_gate_contribution as original_study
from fleet_selection_control import FleetSelectionController
from gate_mechanism_diagnostics import stable_number
from source_arboids import sha256, verify_source


PRIOR_ROOT = Path('train/experiments/feedback-joint-20261006')
PROTOCOL = Path('docs/fleet-selection-study.md')
CONFIGS = {f'h{h}_p{round(p*1000):03d}': dict(block_steps=h, departure_penalty=p)
           for h in (5, 10) for p in (0., .01, .03)}
METHODS = ('original', 'cbf', 'current', 'tuned', 'held_tuned', 'gate_tuned')
_SELECTED = {}
_CONTROLLERS = {}


def initialize(checkpoint, selected):
    global _SELECTED
    feedback_study.initialize(checkpoint)
    _SELECTED = selected
    # Reuse the frozen executor verbatim, injecting only the controller factory.
    feedback_study.new_controller = new_controller


def configuration(n, method):
    if method in CONFIGS:
        return CONFIGS[method]
    if method == 'current':
        return CONFIGS['h10_p000']
    return CONFIGS[_SELECTED[str(n)]['configuration']]


def new_controller(n, method, block_steps):
    config = configuration(n, method)
    if block_steps != config['block_steps']:
        raise AssertionError('Executor horizon differs from frozen controller horizon.')
    key = n, method
    if key not in _CONTROLLERS:
        _CONTROLLERS[key] = FleetSelectionController(n, feedback_study._POLICY, **config,
            prediction='held' if method == 'held_tuned' else 'feedback',
            gate_only=method == 'gate_tuned')
    return _CONTROLLERS[key]


def scene_job(job):
    scene, methods = job
    results = []
    order = np.random.default_rng(scene['scene_seed'] + 821).permutation(len(methods))
    for i in order:
        method = methods[i]
        if method in ('original', 'cbf'):
            row = original_study.rollout(scene, method)
            if method == 'original':
                row['original_equivalent'] = (row['trajectory_sha256'] ==
                    original_study.direct_original(scene, feedback_study._POLICY))
                if not row['original_equivalent']:
                    raise AssertionError('Original-source replay mismatch.')
        else:
            config = configuration(scene['defenders'], method)
            row = feedback_study.new_rollout(scene, method, config['block_steps'])
            row.update(config)
        results.append(row)
    return results


def development_scenes():
    groups = defaultdict(list)
    with (PRIOR_ROOT / 'confirmation.csv').open(encoding='utf-8') as file:
        for row in csv.DictReader(file):
            if row['method'] == 'original' and not row['cell'].startswith('main-'):
                scene = dict(cell=row['cell'], scene_seed=int(row['scene_seed']),
                    noise_seed=int(row['noise_seed']), defenders=int(row['defenders']),
                    agility=float(row['agility']))
                groups[row['cell']].append(scene)
    scenes = [s for cell in sorted(groups) for s in sorted(groups[cell],
        key=lambda s: stable_number(f"fleet-selection-development-{s['scene_seed']}"))[:4]]
    for i, agility in enumerate((1.5, 2., 2.5, 3.)):
        for repeat in range(4):
            seed = 161000000 + i * 10000 + repeat
            scenes.append(dict(cell=f'n6-a{agility:g}', scene_seed=seed, noise_seed=seed+1000000,
                               defenders=6, agility=agility))
    if len(scenes) != 80:
        raise AssertionError('Development roster must contain 80 scenes.')
    return scenes


def confirmation_scenes():
    scenes = []
    for i, (n, agility) in enumerate((n, a) for n in (2, 3, 4, 5, 6) for a in (1.5, 2., 2.5, 3.)):
        for repeat in range(16):
            seed = 163000000 + i * 10000 + repeat
            scenes.append(dict(cell=f'n{n}-a{agility:g}', scene_seed=seed, noise_seed=seed+1000000,
                               defenders=n, agility=agility))
    return scenes


def run(jobs, workers, selected):
    results = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
            initargs=(str(feedback_study.CHECKPOINT), selected)) as pool:
        pending = {pool.submit(scene_job, job): i for i, job in enumerate(jobs)}
        for future in as_completed(pending):
            results.append((pending[future], future.result()))
            print(f'[SCENES] {len(results)}/{len(jobs)}; {time.perf_counter()-start:.1f}s', flush=True)
    return [row for _, group in sorted(results) for row in group]


def frozen_inputs():
    prior = json.loads((PRIOR_ROOT / 'specification.json').read_text(encoding='utf-8'))
    code = feedback_study.code_hashes()
    if code != prior['code'] or sha256(feedback_study.CHECKPOINT) != prior['checkpoint_sha256']:
        raise RuntimeError('Prior frozen implementations or checkpoint changed.')
    source = verify_source()
    if source != prior['source']:
        raise RuntimeError('Untouched source changed.')
    for name in ('feedback_joint_fast.py', 'fleet_selection_control.py',
                 'evaluate_fleet_selection.py', 'study_runtime.py'):
        code[name] = sha256(Path(__file__).with_name(name))
    return dict(code=code, source=source, checkpoint=str(feedback_study.CHECKPOINT),
        checkpoint_sha256=sha256(feedback_study.CHECKPOINT), protocol_sha256=sha256(PROTOCOL),
        prior_confirmation_sha256=sha256(PRIOR_ROOT / 'confirmation.csv'), configurations=CONFIGS,
        methods=METHODS, development=development_scenes(), confirmation=confirmation_scenes(),
        selection_rule='per N: most successes, fewest collisions, highest mean return, current setting, deterministic id')


def select(rows):
    summaries = []
    selected = {}
    for n in (2, 3, 4, 5, 6):
        candidates = []
        for method in (*CONFIGS, 'cbf'):
            group = [r for r in rows if r['defenders'] == n and r['method'] == method]
            item = dict(defenders=n, configuration=method, episodes=len(group),
                successes=sum(r['success'] for r in group), collisions=sum(r['collision'] for r in group),
                mean_return=float(np.mean([r['team_return'] for r in group])))
            summaries.append(item)
            if method != 'cbf':
                candidates.append(item)
        selected[str(n)] = min(candidates, key=lambda r: (-r['successes'], r['collisions'],
            -r['mean_return'], r['configuration'] != 'h10_p000', r['configuration']))
    return selected, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('develop', 'confirm'), required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('train/experiments/fleet-selection-20261006'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    spec = frozen_inputs()
    path = root / 'specification.json'
    normalized = json.loads(json.dumps(spec))
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != normalized:
            raise RuntimeError('Frozen study inputs changed.')
    else:
        path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    if (root / f'{args.stage}.csv').exists():
        raise RuntimeError('This completed stage already exists; no automatic rerun or overwrite.')
    start = time.perf_counter()
    if args.stage == 'develop':
        rows = run([(s, (*CONFIGS, 'cbf')) for s in spec['development']], args.workers, {})
        selected, summaries = select(rows)
        frozen = dict(selected=selected, code=spec['code'], protocol_sha256=spec['protocol_sha256'],
                      selection_rule=spec['selection_rule'])
        (root / 'frozen_parameters.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
        original_study.write_csv(root / 'development_summary.csv', summaries)
        print('[FROZEN] ' + json.dumps(selected), flush=True)
    else:
        frozen = json.loads((root / 'frozen_parameters.json').read_text(encoding='utf-8'))
        if frozen['code'] != spec['code'] or frozen['protocol_sha256'] != spec['protocol_sha256']:
            raise RuntimeError('Selected method or protocol changed.')
        selected = frozen['selected']
        rows = run([(s, METHODS) for s in spec['confirmation']], args.workers, selected)
    original_study.write_csv(root / f'{args.stage}.csv', rows)
    expected = 560 if args.stage == 'develop' else 1920
    passed = frozen_inputs() == spec and len(rows) == expected
    summary = dict(passed=passed, stage=args.stage, scenes=len(rows)//(7 if args.stage=='develop' else 6),
                   rows=len(rows), elapsed_seconds=time.perf_counter()-start,
                   code_source_checkpoint_protocol_unchanged=frozen_inputs()==spec)
    (root / f'{args.stage}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
