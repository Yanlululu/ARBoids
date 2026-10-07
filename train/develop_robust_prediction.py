"""Bounded development on the two completed six-vessel confirmation cohorts."""
import study_runtime
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import time

import numpy as np

from array_rollout import BatchedSourcePolicy
import evaluate_predictive_interception as prior
import evaluate_short_continuation as second
from jit_rollout import JitRolloutController
from predictive_interception_v2 import TaskTimePredictiveController
from source_arboids import sha256


ROOT = Path('train/experiments/robust-predictive-interception-20261006/development2')
CONFIGS = {
    'base_h10': dict(block_steps=10, tail_steps=100, tail_policy='baseline'),
    'sustain_h5': dict(block_steps=5, tail_steps=100, tail_policy='candidate'),
    'sustain_h10': dict(block_steps=10, tail_steps=100, tail_policy='candidate'),
    'short_h5': dict(block_steps=5, tail_steps=0, tail_policy='candidate'),
    'short_h10': dict(block_steps=10, tail_steps=0, tail_policy='candidate'),
    'legacy_base_h10': dict(block_steps=10, tail_steps=100, tail_policy='baseline', failure_cost='legacy')}
CONFIGS = {name: dict(blend=1., capture_margin=.5, failure_cost='delay', **{
    key:value for key,value in settings.items() if key != 'failure_cost'}) |
    {'failure_cost':settings.get('failure_cost','delay')} for name,settings in CONFIGS.items()}
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
            rule = TaskTimePredictiveController(n, prior.old._POLICY,
                fixed_template='lead_guard', blend=1., block_steps=10, adaptive_attacker=False)
        else:
            rule = JitRolloutController(n, prior.old._POLICY, prediction_policy=BATCHED, **CONFIGS[method])
        RULES[key] = rule
    return RULES[key]


def scenes():
    return [dict(s, development_group='old_first') for s in prior.confirmation_scenes() if s['defenders']==6] + [
        dict(s, development_group='old_second') for s in second.scenes()]


def job(scene):
    rows = []
    for i in np.random.default_rng(scene['scene_seed']+47292).permutation(len(METHODS)):
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
    for group in ('all', 'old_first', 'old_second'):
        for method in METHODS:
            selected = [r for r in rows if r['method']==method and (group=='all' or r['development_group']==group)]
            if selected:
                table.append(dict(group=group, method=method, episodes=len(selected),
                    **{key:sum(r[key] for r in selected) for key in ('success','capture','collision','source_loss','timeout')},
                    capped_capture_time=float(np.mean([r['failure_capped_capture_time'] for r in selected]))))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'episodes.csv').exists():
        raise RuntimeError('Completed development must not be overwritten.')
    previous = prior.frozen_inputs()
    code = dict(previous['code'])
    for name in ('array_rollout.py','compiled_source_policy.py','jit_nominal_environment.py',
                 'jit_rollout.py','develop_robust_prediction.py'):
        code[name] = sha256(Path(__file__).with_name(name))
    spec = dict(code=code, source=previous['source'], checkpoint_sha256=previous['checkpoint_sha256'],
                configurations=CONFIGS, methods=METHODS, scenes=scenes(), development_only=True,
                scope='All 256 already observed six-vessel scenarios; these are development data.')
    (ROOT/'specification.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    started = time.perf_counter()
    groups = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        pending = {pool.submit(job, scene):i for i,scene in enumerate(scenes())}
        for task in as_completed(pending):
            groups.append((pending[task], task.result()))
            print(f'[ROBUST-DEVELOP] {len(groups)}/256; {time.perf_counter()-started:.1f}s', flush=True)
            if len(groups)%32 == 0:
                print(json.dumps([r for r in summarize([r for _,g in groups for r in g]) if r['group']=='all']), flush=True)
    rows = [r for _,group in sorted(groups) for r in group]
    table = summarize(rows)
    prior.source.write_csv(ROOT/'episodes.csv', rows)
    prior.source.write_csv(ROOT/'summary.csv', table)
    passed = len(rows)==256*len(METHODS) and all(
        sha256(Path(__file__).with_name(name))==expected for name,expected in code.items())
    result = dict(passed=passed, rows=len(rows), elapsed_seconds=time.perf_counter()-started, summary=table)
    (ROOT/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
