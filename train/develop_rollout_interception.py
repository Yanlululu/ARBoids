"""A bounded continuation-value prototype on the same 16 old scenes."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import numpy as np

import evaluate_feedback_joint as old
import evaluate_gate_contribution as source
from rollout_interception import RolloutInterceptionController
from source_arboids import sha256, verify_source


CONFIGS = {f'rollout_t{tail}_b{int(b*100)}': dict(block_steps=10, blend=b, tail_steps=tail)
           for tail in (40, 100) for b in (.5, 1.)}
RULES = {}


def factory(n, method, h):
    if (n, method) not in RULES:
        RULES[n, method] = RolloutInterceptionController(n, old._POLICY, **CONFIGS[method])
    return RULES[n, method]


def initialize():
    old.initialize(str(old.CHECKPOINT))
    old.new_controller = factory


def job(scene):
    rows = []
    methods = ('cbf', *CONFIGS)
    for i in np.random.default_rng(scene['scene_seed']+679).permutation(len(methods)):
        method = methods[i]
        row = source.rollout(scene, method) if method == 'cbf' else old.new_rollout(scene, method, 10)
        row['failure_capped_capture_time'] = row['duration'] if row['capture'] else 80.
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path('train/experiments/predictive-interception-20261006/prototype3')
    root.mkdir(parents=True, exist_ok=True)
    if (root/'episodes.csv').exists():
        raise RuntimeError('Prototype already exists.')
    scenes = json.loads(Path('train/experiments/predictive-interception-20261006/prototype/specification.json').read_text(encoding='utf-8'))['scenes']
    names = ('rollout_interception.py', 'develop_rollout_interception.py',
             'predictive_interception_v2.py', 'attacker_motion_observer.py',
             'predictive_interception.py', 'feedback_joint_fast.py')
    code = {name:sha256(Path(__file__).with_name(name)) for name in names}
    original = verify_source()
    spec = dict(code=code, scenes=scenes, configurations=CONFIGS, development_only=True,
                source=original, checkpoint_sha256=sha256(old.CHECKPOINT))
    (root/'specification.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    started = time.perf_counter()
    groups = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        pending = {pool.submit(job, scene):i for i, scene in enumerate(scenes)}
        for task in as_completed(pending):
            groups.append((pending[task], task.result()))
            print(f'[CONTINUATION] {len(groups)}/{len(scenes)}; {time.perf_counter()-started:.1f}s', flush=True)
    rows = [row for _, group in sorted(groups) for row in group]
    source.write_csv(root/'episodes.csv', rows)
    summary = []
    for method in ('cbf', *CONFIGS):
        group = [row for row in rows if row['method']==method]
        summary.append(dict(method=method, episodes=len(group),
            **{key:sum(row[key] for row in group) for key in ('success','collision','source_loss','capture')},
            capped_capture_time=float(np.mean([r['failure_capped_capture_time'] for r in group])),
            mean_return=float(np.mean([r['team_return'] for r in group]))))
    source.write_csv(root/'summary.csv', summary)
    passed = original == verify_source() and code == {n:sha256(Path(__file__).with_name(n)) for n in names}
    output = dict(passed=passed, rows=len(rows), elapsed_seconds=time.perf_counter()-started, summary=summary)
    (root/'summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
