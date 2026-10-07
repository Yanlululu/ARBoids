"""One causal cadence rule on all previously observed six-vessel scenarios."""
import study_runtime
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from adaptive_interval_rollout import AdaptiveIntervalController
from array_rollout import BatchedSourcePolicy
import develop_robust_prediction as old_development
import evaluate_robust_prediction as third
import evaluate_predictive_interception as prior
from source_arboids import sha256


ROOT = Path('train/experiments/adaptive-prediction-20261006/development')
CONFIGS = {method:dict(block_steps=5, blend=1., capture_margin=.5, failure_cost='delay',
    tail_steps=100 if method=='predictive' else 0, tail_policy='candidate', agility_threshold=1.75)
    for method in ('predictive','short_value')}
METHODS = ('cbf','best_fixed','strongest_short','previous','predictive','short_value')
RULES = {}
BATCHED = None


def scenes():
    return old_development.scenes()+[dict(s,development_group='old_third') for s in third.scenes()]


def initialize():
    global BATCHED
    prior.old.initialize(str(prior.old.CHECKPOINT))
    BATCHED = BatchedSourcePolicy(prior.old._POLICY)
    prior.old.new_controller = factory


def factory(n,method,h):
    if (n,method) not in RULES:
        RULES[n,method] = AdaptiveIntervalController(n,prior.old._POLICY,
            prediction_policy=BATCHED,**CONFIGS[method])
    return RULES[n,method]


def job(scene):
    methods = list(CONFIGS)
    rows = []
    for i in np.random.default_rng(scene['scene_seed']+53470).permutation(len(methods)):
        method = methods[i]
        row = prior.old.new_rollout(scene,method,5)
        row['failure_capped_capture_time'] = row['duration'] if row['capture'] else 80.
        rule = RULES[scene['defenders'],method]
        row.update(one_second_plans=rule.interval_plan_counts[5],two_second_plans=rule.interval_plan_counts[10],
                   final_estimated_agility=rule.attacker_agility, reused_reference=False)
        rows.append(row)
    return rows


def references():
    rows = []
    for path,mapping in (
        (old_development.ROOT/'episodes.csv',dict(cbf='cbf',best_fixed='best_fixed',
            short_h10='strongest_short',sustain_h5='previous')),
        (third.ROOT/'episodes.csv',dict(cbf='cbf',best_fixed='best_fixed',
            strongest_short='strongest_short',predictive='previous'))):
        with path.open(encoding='utf-8',newline='') as file:
            for row in csv.DictReader(file):
                if row['method'] in mapping:
                    row['method'] = mapping[row['method']]
                    row.setdefault('development_group','old_third')
                    row.update(reference_file=str(path),reused_reference=True)
                    row['scene_seed'] = int(row['scene_seed'])
                    rows.append(row)
    return rows


def summarize(rows):
    table = []
    for group in ('all','old_first','old_second','old_third'):
        for method in METHODS:
            r = [r for r in rows if r['method']==method and (group=='all' or r['development_group']==group)]
            table.append(dict(group=group,method=method,episodes=len(r),
                **{k:sum(int(x[k]) for x in r) for k in ('success','capture','collision','source_loss','timeout')},
                capped_capture_time=float(np.mean([float(x['failure_capped_capture_time']) for x in r]))))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'episodes.csv').exists():
        raise RuntimeError('Completed development must not be overwritten.')
    previous = json.loads((third.ROOT/'specification.json').read_text(encoding='utf-8'))
    third.verify(previous)
    code = dict(previous['code'])
    for name in ('adaptive_interval_rollout.py','develop_adaptive_prediction.py'):
        code[name] = sha256(Path(__file__).with_name(name))
    inputs = {str(path):sha256(path) for path in (old_development.ROOT/'episodes.csv',third.ROOT/'episodes.csv')}
    spec = dict(code=code,source=previous['source'],checkpoint_sha256=previous['checkpoint_sha256'],
                configurations=CONFIGS,scenes=scenes(),development_only=True,reused_inputs_sha256=inputs)
    (ROOT/'specification.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    began = time.perf_counter()
    groups = []
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        pending = {pool.submit(job,s):i for i,s in enumerate(scenes())}
        for task in as_completed(pending):
            groups.append((pending[task],task.result()))
            print(f'[CADENCE-DEVELOP] {len(groups)}/768; {time.perf_counter()-began:.1f}s',flush=True)
    rows = [r for _,group in sorted(groups) for r in group]+references()
    table = summarize(rows)
    prior.source.write_csv(ROOT/'episodes.csv',rows)
    prior.source.write_csv(ROOT/'summary.csv',table)
    passed = len(rows)==768*len(METHODS) and len({(r['scene_seed'],r['method']) for r in rows})==len(rows)
    passed = passed and all(sha256(Path(__file__).with_name(k))==v for k,v in code.items())
    passed = passed and all(sha256(Path(k))==v for k,v in inputs.items())
    output = dict(passed=passed,new_episodes=1536,reused_episodes=3072,
                  elapsed_seconds=time.perf_counter()-began,summary=table)
    (ROOT/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output),flush=True)
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
