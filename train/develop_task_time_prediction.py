"""Second bounded prototype with preserved navigation and causal observation."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np

import evaluate_feedback_joint as old
import evaluate_gate_contribution as source
from predictive_interception_v2 import TaskTimePredictiveController, TEMPLATES
from source_arboids import sha256, verify_source


CONFIGS = {f'task_h{h}_b{int(b*100)}': dict(block_steps=h, blend=b)
           for h in (10, 20) for b in (.5, 1.)}
FIXED = {f'fixed_{t}_b{int(b*100)}': dict(fixed_template=t, blend=b, block_steps=10,
          adaptive_attacker=False) for t in TEMPLATES[1:] for b in (.5, 1.)}
_RULES = {}


def configuration(method):
    return CONFIGS[method] if method in CONFIGS else FIXED[method]


def factory(n, method, h):
    if (n, method) not in _RULES:
        _RULES[n, method] = TaskTimePredictiveController(n, old._POLICY, **configuration(method))
    return _RULES[n, method]


def initialize():
    old.initialize(str(old.CHECKPOINT))
    old.new_controller = factory


def job(scene):
    rows = []
    methods = ('cbf', *CONFIGS, *FIXED)
    for i in np.random.default_rng(scene['scene_seed']+511).permutation(len(methods)):
        m = methods[i]
        row = source.rollout(scene, m) if m == 'cbf' else old.new_rollout(scene, m, configuration(m)['block_steps'])
        row['failure_capped_capture_time'] = row['duration'] if row['capture'] else 80.
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('train/experiments/predictive-interception-20261006/prototype2'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    if (root/'episodes.csv').exists():
        raise RuntimeError('Prototype already exists.')
    scenes = json.loads(Path('train/experiments/predictive-interception-20261006/prototype/specification.json').read_text(encoding='utf-8'))['scenes']
    names = ('predictive_interception_v2.py', 'attacker_motion_observer.py', 'develop_task_time_prediction.py',
             'predictive_interception.py', 'feedback_joint_fast.py')
    code = {n: sha256(Path(__file__).with_name(n)) for n in names}
    original = verify_source()
    spec = dict(code=code, scenes=scenes, configurations=CONFIGS, fixed_templates=FIXED,
                development_only=True, source=original, checkpoint_sha256=sha256(old.CHECKPOINT))
    (root/'specification.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    start = time.perf_counter()
    groups = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        pending = {pool.submit(job, s): i for i, s in enumerate(scenes)}
        for task in as_completed(pending):
            groups.append((pending[task], task.result()))
            print(f'[PROTOTYPE2] {len(groups)}/{len(scenes)}; {time.perf_counter()-start:.1f}s', flush=True)
    rows = [r for _, group in sorted(groups) for r in group]
    source.write_csv(root/'episodes.csv', rows)
    summaries = []
    for method in ('cbf', *CONFIGS, *FIXED):
        group = [r for r in rows if r['method']==method]
        summaries.append(dict(method=method, episodes=len(group),
            **{k: sum(r[k] for r in group) for k in ('success', 'collision', 'source_loss', 'capture')},
            capped_capture_time=float(np.mean([r['failure_capped_capture_time'] for r in group])),
            mean_return=float(np.mean([r['team_return'] for r in group]))))
    source.write_csv(root/'summary.csv', summaries)
    passed = verify_source()==original and code=={n:sha256(Path(__file__).with_name(n)) for n in names}
    output = dict(passed=passed, rows=len(rows), elapsed_seconds=time.perf_counter()-start, summary=summaries)
    (root/'summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
