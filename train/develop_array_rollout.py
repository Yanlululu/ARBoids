"""Finite continuation-policy development using already observed scenes only."""
import study_runtime
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import time

import numpy as np

from array_rollout import ArrayRolloutController, BatchedSourcePolicy
import evaluate_predictive_interception as prior
from predictive_interception_v2 import TaskTimePredictiveController
from source_arboids import sha256


ROOT = Path('train/experiments/robust-predictive-interception-20261006/development1')
CONFIGS = {f'{kind}_h{h}_m{int(margin*10):02d}':
           dict(block_steps=h, blend=1., tail_steps=0 if kind == 'short' else 100,
                tail_policy='candidate' if kind == 'sustain' else 'baseline', capture_margin=margin,
                failure_cost='delay')
           for kind in ('base', 'sustain', 'short') for h in (5, 10) for margin in (.5, 1.5)}
CONFIGS['legacy_base_h10_m05'] = dict(block_steps=10, blend=1., tail_steps=100,
                                     tail_policy='baseline', capture_margin=.5, failure_cost='legacy')
METHODS = ('cbf', 'best_fixed', *CONFIGS)
RULES = {}
BATCHED = None


def initialize():
    global BATCHED
    prior.old.initialize(str(prior.old.CHECKPOINT))
    BATCHED = BatchedSourcePolicy(prior.old._POLICY)
    prior.old.new_controller = factory


def factory(n, method, h):
    key = n, method
    if key not in RULES:
        if method == 'best_fixed':
            RULES[key] = TaskTimePredictiveController(n, prior.old._POLICY,
                fixed_template='lead_guard', blend=1., block_steps=10, adaptive_attacker=False)
        else:
            RULES[key] = ArrayRolloutController(n, prior.old._POLICY,
                prediction_policy=BATCHED, **CONFIGS[method])
    return RULES[key]


def scenes():
    primary = [dict(s, development_group='old_primary') for s in prior.development_scenes()
               if s['defenders'] == 6]
    failures = [dict(s, development_group='failure_cases') for s in prior.confirmation_scenes()
                if s['scene_seed'] in (173180012, 173190003)]
    return primary + failures


def job(scene):
    rows = []
    for i in np.random.default_rng(scene['scene_seed'] + 9482).permutation(len(METHODS)):
        method = METHODS[i]
        if method == 'cbf':
            row = prior.source.rollout(scene, method)
        else:
            h = CONFIGS[method]['block_steps'] if method in CONFIGS else 10
            row = prior.old.new_rollout(scene, method, h)
        row['failure_capped_capture_time'] = row['duration'] if row['capture'] else 80.
        rows.append(row)
    return rows


def summarize(rows):
    table = []
    for group in ('all', 'old_primary', 'failure_cases'):
        for method in METHODS:
            selected = [r for r in rows if r['method'] == method and
                        (group == 'all' or r['development_group'] == group)]
            table.append(dict(group=group, method=method, episodes=len(selected),
                **{key:sum(r[key] for r in selected) for key in ('success','capture','collision','source_loss','timeout')},
                capped_capture_time=float(np.mean([r['failure_capped_capture_time'] for r in selected]))))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'episodes.csv').exists():
        raise RuntimeError('This development stage already exists.')
    previous = prior.frozen_inputs()
    code = dict(previous['code'])
    for name in ('array_rollout.py','compiled_source_policy.py','develop_array_rollout.py'):
        code[name] = sha256(Path(__file__).with_name(name))
    spec = dict(code=code, source=previous['source'], checkpoint_sha256=previous['checkpoint_sha256'],
                configurations=CONFIGS, scenes=scenes(), development_only=True,
                scope='32 old development scenes plus two previously observed failures; no unseen confirmation data.')
    (ROOT / 'specification.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    started = time.perf_counter()
    groups = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        pending = {pool.submit(job, s):i for i, s in enumerate(scenes())}
        for task in as_completed(pending):
            groups.append((pending[task], task.result()))
            print(f'[ARRAY-DEVELOP] {len(groups)}/34; {time.perf_counter()-started:.1f}s', flush=True)
    rows = [row for _, group in sorted(groups) for row in group]
    table = summarize(rows)
    prior.source.write_csv(ROOT / 'episodes.csv', rows)
    prior.source.write_csv(ROOT / 'summary.csv', table)
    passed = len(rows) == 34 * len(METHODS) and all(
        sha256(Path(__file__).with_name(name)) == expected for name, expected in code.items())
    output = dict(passed=passed, rows=len(rows), elapsed_seconds=time.perf_counter()-started, summary=table)
    (ROOT / 'summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
