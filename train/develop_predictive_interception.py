"""Finite prototype comparison on already observed development scenes."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

import numpy as np

import evaluate_feedback_joint as old
import evaluate_gate_contribution as source
from gate_mechanism_diagnostics import stable_number
from predictive_interception import PredictiveInterceptionController, MISSION_TEMPLATES
from source_arboids import sha256


_CONTROLLERS = {}
CONFIGS = {f'mission_h{h}_b{int(b*100)}': dict(block_steps=h, blend=b)
           for h in (10, 20) for b in (.5, 1.)}
FIXED = {f'fixed_{t}_b{int(b*100)}': dict(fixed_template=t, blend=b, block_steps=10)
         for t in MISSION_TEMPLATES[1:] for b in (.5, 1.)}


def configuration(method):
    return CONFIGS[method] if method in CONFIGS else FIXED[method]


def factory(n, method, steps):
    key = n, method
    if key not in _CONTROLLERS:
        _CONTROLLERS[key] = PredictiveInterceptionController(n, old._POLICY, **configuration(method))
    return _CONTROLLERS[key]


def initialize():
    old.initialize(str(old.CHECKPOINT))
    old.new_controller = factory


def job(scene):
    rows = []
    methods = ('cbf', *CONFIGS, *FIXED)
    for i in np.random.default_rng(scene['scene_seed'] + 398).permutation(len(methods)):
        method = methods[i]
        if method == 'cbf':
            row = source.rollout(scene, method)
        else:
            row = old.new_rollout(scene, method, configuration(method)['block_steps'])
        row['failure_capped_capture_time'] = row['duration'] if row['capture'] else 80.
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--per-cell', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'episodes.csv').exists():
        raise RuntimeError('Prototype output already exists.')
    input_path = Path('train/experiments/fleet-selection-20261006/confirm.csv')
    rows = list(csv.DictReader(input_path.open(encoding='utf-8')))
    scenes = []
    for n in (3, 4, 5, 6):
        for agility in (1.5, 2., 2.5, 3.):
            group = [r for r in rows if r['method'] == 'original' and int(r['defenders']) == n
                     and float(r['agility']) == agility]
            for r in sorted(group, key=lambda r: stable_number('mission-prototype-' + r['scene_seed']))[:args.per_cell]:
                scenes.append(dict(cell=r['cell'], scene_seed=int(r['scene_seed']), noise_seed=int(r['noise_seed']),
                                   defenders=n, agility=agility))
    code = {name: sha256(Path(__file__).with_name(name)) for name in
            ('predictive_interception.py', 'develop_predictive_interception.py')}
    spec = dict(code=code, scenes=scenes, configurations=CONFIGS, fixed_templates=FIXED,
                development_only=True, checkpoint_sha256=sha256(old.CHECKPOINT), input_sha256=sha256(input_path))
    (root / 'specification.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    started = time.perf_counter()
    groups = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        pending = {pool.submit(job, scene): i for i, scene in enumerate(scenes)}
        for task in as_completed(pending):
            groups.append((pending[task], task.result()))
            print(f'[PROTOTYPE] {len(groups)}/{len(scenes)}; {time.perf_counter()-started:.1f}s', flush=True)
    rows = [r for _, group in sorted(groups) for r in group]
    source.write_csv(root / 'episodes.csv', rows)
    summary = []
    for method in ('cbf', *CONFIGS, *FIXED):
        group = [r for r in rows if r['method'] == method]
        summary.append(dict(method=method, episodes=len(group), success=sum(r['success'] for r in group),
            collision=sum(r['collision'] for r in group), source_loss=sum(r['source_loss'] for r in group),
            capture=sum(r['capture'] for r in group),
            capped_capture_time=float(np.mean([r['failure_capped_capture_time'] for r in group])),
            mean_return=float(np.mean([r['team_return'] for r in group]))))
    source.write_csv(root / 'summary.csv', summary)
    passed = code == {n: sha256(Path(__file__).with_name(n)) for n in code}
    output = dict(passed=passed, scenes=len(scenes), rows=len(rows), elapsed_seconds=time.perf_counter()-started,
                  summary=summary)
    (root / 'summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
