"""One fixed cadence rule, followed by 1024 independent paired scenarios."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

import numpy as np

from adaptive_interval_rollout import AdaptiveIntervalController
from array_rollout import BatchedSourcePolicy
import develop_adaptive_prediction as development
import evaluate_robust_prediction as third
import evaluate_predictive_interception as prior
from jit_rollout import JitRolloutController
from predictive_interception_v2 import TaskTimePredictiveController
from source_arboids import sha256


ROOT = Path('train/experiments/adaptive-prediction-20261006/confirmation')
PROTOCOL = Path('docs/adaptive-prediction-study.md')
CONFIDENCE = 1. - .05/4.
METHODS = ('original','cbf','predictive','short_value','strongest_short','best_fixed','previous')
PRIMARY = ('cbf','short_value','strongest_short','best_fixed')
ANALYSIS_FILES = ('gen_fig_adaptive_prediction.py','gen_fig_predictive_interception.py','gen_fig_gate_contribution.py')
SETTINGS = {}
BATCHED = None
RULES = {}


def scenes():
    return [dict(cell=f'n6-a{a:g}',defenders=6,agility=a,
                 scene_seed=203160000+i*10000+r,noise_seed=204160000+i*10000+r)
            for i,a in enumerate((1.5,2.,2.5,3.)) for r in range(256)]


def selected():
    previous = json.loads((third.ROOT/'specification.json').read_text(encoding='utf-8'))
    settings = dict(development.CONFIGS)
    settings.update(strongest_short=dict(previous['selected']['settings']['strongest_short']),
                    previous=dict(previous['selected']['settings']['predictive']))
    return dict(predictive='adaptive',settings=settings,methods=METHODS,primary_references=PRIMARY)


def input_code():
    previous = json.loads((third.ROOT/'specification.json').read_text(encoding='utf-8'))
    third.verify(previous)
    code = dict(previous['code'])
    for name in ('adaptive_interval_rollout.py','parallel_adaptive_prediction.py',
                 'develop_adaptive_prediction.py','benchmark_adaptive_prediction.py','evaluate_adaptive_prediction.py'):
        code[name] = sha256(Path(__file__).with_name(name))
    return previous,code


def verify(spec):
    previous,code = input_code()
    if code != spec['code'] or previous['source'] != spec['source'] or \
       previous['checkpoint_sha256'] != spec['checkpoint_sha256'] or sha256(PROTOCOL)!=spec['protocol_sha256']:
        raise RuntimeError('Frozen source, weights, controller or protocol changed.')
    for name,expected in spec['evidence_sha256'].items():
        if sha256(Path(name))!=expected:
            raise RuntimeError('Frozen evidence changed: '+name)
    for name,expected in spec['analysis_sha256'].items():
        if sha256(Path(__file__).parent/'figures'/name)!=expected:
            raise RuntimeError('Frozen analysis changed: '+name)


def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'specification.json').exists():
        raise RuntimeError('Confirmation already frozen.')
    development_result = json.loads((development.ROOT/'summary.json').read_text(encoding='utf-8'))
    if not development_result['passed']:
        raise RuntimeError('Development failed its execution checks.')
    table = {r['method']:r for r in development_result['summary'] if r['group']=='all'}
    a = table['predictive']
    for method in PRIMARY:
        b = table[method]
        if a['success']<b['success'] or a['collision']>b['collision'] or \
           a['capped_capture_time']>.9*b['capped_capture_time']:
            raise RuntimeError('Development is not sufficiently favorable: '+method)
    selection = selected()
    timing_path = ROOT.parent/'runtime-adaptive.json'
    timing = json.loads(timing_path.read_text(encoding='utf-8'))
    if not timing['passed'] or timing['settings']!=selection['settings']['predictive']:
        raise RuntimeError('Matched runtime evidence is required.')
    planning = next(r for r in timing['groups'] if r['engine']=='parallel' and r['phase']=='planning')
    if planning['p95_ms']>=200.:
        raise RuntimeError('The runtime requirement is not met.')
    previous,code = input_code()
    history = set()
    for path in Path('train/experiments').rglob('*.csv'):
        if ROOT in path.parents:
            continue
        with path.open(encoding='utf-8-sig',newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames and 'scene_seed' in reader.fieldnames:
                history.update(int(r['scene_seed']) for r in reader if r.get('scene_seed'))
    if {s['scene_seed'] for s in scenes()} & history:
        raise RuntimeError('Confirmation overlaps old scenes.')
    inputs = (development.ROOT/'episodes.csv',development.ROOT/'summary.json',
              development.ROOT/'specification.json',third.ROOT/'episodes.csv',timing_path)
    spec = dict(selected=selection,code=code,source=previous['source'],checkpoint_sha256=previous['checkpoint_sha256'],
        protocol_sha256=sha256(PROTOCOL),evidence_sha256={str(p):sha256(p) for p in inputs},
        analysis_sha256={n:sha256(Path(__file__).parent/'figures'/n) for n in ANALYSIS_FILES},
        confirmation=scenes(),historical_unique_scenes=len(history),confidence_level=CONFIDENCE,
        bootstrap_repeats=10000,primary_defenders=6,
        strong_advantage='Every primary reference: mean capped-time reduction >=10%, paired stratified CI excludes zero, '
            'observed success not lower, collisions not higher, conservative paired success lower bound >=-3pp.')
    verify(spec)
    (ROOT/'specification.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,stage='freeze',selected=selection,scenes=1024,
                         historical_unique_scenes=len(history))),flush=True)


def initialize(settings):
    global SETTINGS,BATCHED
    SETTINGS = settings
    prior.old.initialize(str(prior.old.CHECKPOINT))
    BATCHED = BatchedSourcePolicy(prior.old._POLICY)
    prior.old.new_controller = factory


def factory(n,method,h):
    if (n,method) not in RULES:
        if method=='best_fixed':
            rule = TaskTimePredictiveController(n,prior.old._POLICY,fixed_template='lead_guard',
                                               blend=1.,block_steps=10,adaptive_attacker=False)
        else:
            cls = AdaptiveIntervalController if method in ('predictive','short_value') else JitRolloutController
            rule = cls(n,prior.old._POLICY,prediction_policy=BATCHED,**SETTINGS[method])
        RULES[n,method] = rule
    return RULES[n,method]


def job(scene):
    rows = []
    for i in np.random.default_rng(scene['scene_seed']+58705).permutation(len(METHODS)):
        method = METHODS[i]
        if method in ('original','cbf'):
            row = prior.source.rollout(scene,method)
            if method=='original':
                row['original_equivalent'] = row['trajectory_sha256']==prior.source.direct_original(scene,prior.old._POLICY)
                if not row['original_equivalent']:
                    raise AssertionError('Original direct source replay differs.')
        else:
            h = SETTINGS[method]['block_steps'] if method in SETTINGS else 10
            row = prior.old.new_rollout(scene,method,h)
            if method in ('predictive','short_value'):
                rule = RULES[scene['defenders'],method]
                row.update(one_second_plans=rule.interval_plan_counts[5],two_second_plans=rule.interval_plan_counts[10],
                           final_estimated_agility=rule.attacker_agility)
        row['failure_capped_capture_time'] = row['duration'] if row['capture'] else 80.
        rows.append(row)
    return rows


def confirm(workers):
    spec = json.loads((ROOT/'specification.json').read_text(encoding='utf-8'))
    verify(spec)
    if (ROOT/'episodes.csv').exists():
        raise RuntimeError('Completed confirmation must not be overwritten.')
    began = time.perf_counter()
    groups = []
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(spec['selected']['settings'],)) as pool:
        pending = {pool.submit(job,s):i for i,s in enumerate(spec['confirmation'])}
        for task in as_completed(pending):
            groups.append((pending[task],task.result()))
            print(f'[CADENCE-CONFIRM] {len(groups)}/1024; {time.perf_counter()-began:.1f}s',flush=True)
    rows = [r for _,group in sorted(groups) for r in group]
    verify(spec)
    passed = len(rows)==1024*len(METHODS) and len({(r['scene_seed'],r['method']) for r in rows})==len(rows)
    if not passed:
        raise AssertionError('Incomplete paired confirmation.')
    prior.source.write_csv(ROOT/'episodes.csv',rows)
    output = dict(passed=passed,rows=len(rows),elapsed_seconds=time.perf_counter()-began)
    (ROOT/'execution.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('freeze','confirm'),required=True)
    parser.add_argument('--workers',type=int,default=8)
    args = parser.parse_args()
    freeze() if args.stage=='freeze' else confirm(args.workers)


if __name__=='__main__':
    main()
