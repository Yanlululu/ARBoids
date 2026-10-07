"""Freeze one developed controller, then evaluate a new paired 512-scene set."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

import numpy as np

from array_rollout import BatchedSourcePolicy
import develop_robust_prediction as development
import evaluate_predictive_interception as prior
from jit_rollout import JitRolloutController
from predictive_interception_v2 import TaskTimePredictiveController
from source_arboids import sha256


ROOT = Path('train/experiments/robust-predictive-interception-20261006/confirmation')
PROTOCOL = Path('docs/robust-predictive-interception-study.md')
CONFIDENCE = 1. - .05/3.
ANALYSIS_FILES = ('gen_fig_robust_prediction.py','gen_fig_predictive_interception.py','gen_fig_gate_contribution.py')
SETTINGS = {}
BATCHED = None
RULES = {}


def scenes():
    return [dict(cell=f'n6-a{agility:g}', defenders=6, agility=agility,
                 scene_seed=193160000+i*10000+repeat, noise_seed=194160000+i*10000+repeat)
            for i,agility in enumerate((1.5,2.,2.5,3.)) for repeat in range(128)]


def selection():
    result = json.loads((development.ROOT/'summary.json').read_text(encoding='utf-8'))
    if not result['passed']:
        raise RuntimeError('Development did not pass execution checks.')
    index = {r['method']:r for r in result['summary'] if r['group']=='all'}
    def score(name):
        r = index[name]
        return -r['success'], r['collision'], r['capped_capture_time'], name
    predictive = min(('base_h10','sustain_h5','sustain_h10'), key=score)
    strongest_short = min(('short_h5','short_h10'), key=score)
    selected = dict(development.CONFIGS[predictive])
    matched = dict(selected, tail_steps=0)
    settings = dict(predictive=selected, short_value=matched,
                    strongest_short=dict(development.CONFIGS[strongest_short]),
                    legacy_value=dict(development.CONFIGS['legacy_base_h10']))
    methods = ['original','cbf','predictive','short_value','best_fixed','legacy_value']
    references = ['cbf','short_value','best_fixed']
    if settings['strongest_short']['block_steps'] != matched['block_steps']:
        methods.insert(4, 'strongest_short')
        references.append('strongest_short')
    return dict(predictive=predictive, strongest_short=strongest_short,
                settings=settings, methods=methods, primary_references=references)


def input_code():
    previous = prior.frozen_inputs()
    code = dict(previous['code'])
    for name in ('array_rollout.py','compiled_source_policy.py','jit_nominal_environment.py','jit_rollout.py',
                 'parallel_jit_rollout.py','develop_robust_prediction.py','evaluate_robust_prediction.py',
                 'benchmark_robust_prediction.py'):
        code[name] = sha256(Path(__file__).with_name(name))
    return previous, code


def verify(spec):
    previous, code = input_code()
    if code != spec['code'] or previous['source'] != spec['source'] or \
       previous['checkpoint_sha256'] != spec['checkpoint_sha256'] or \
       sha256(PROTOCOL) != spec['protocol_sha256']:
        raise RuntimeError('Frozen implementation, model, weights or protocol changed.')
    for path,expected in spec['evidence_sha256'].items():
        if sha256(Path(path)) != expected:
            raise RuntimeError('Frozen development or benchmark evidence changed: '+path)
    for name,expected in spec['analysis_sha256'].items():
        if sha256(Path(__file__).parent/'figures'/name) != expected:
            raise RuntimeError('Frozen analysis changed: '+name)


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT/'specification.json'
    if path.exists():
        raise RuntimeError('A confirmation has already been frozen.')
    selected = selection()
    runtime_path = ROOT.parent/('runtime-'+selected['predictive']+'.json')
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    if not runtime['passed'] or runtime['settings'] != selected['settings']['predictive']:
        raise RuntimeError('Equivalent final-controller runtime evidence is required before confirmation.')
    planning = next(r for r in runtime['groups'] if r['engine']=='parallel' and r['phase']=='planning')
    if planning['p95_ms'] >= 200.:
        raise RuntimeError('The selected controller has not met the runtime requirement.')
    previous, code = input_code()
    fresh = scenes()
    fresh_seeds = {s['scene_seed'] for s in fresh}
    historical_seeds = set()
    inputs = [development.ROOT/'episodes.csv', development.ROOT/'specification.json',
              development.ROOT/'summary.json', runtime_path]
    for csv_path in Path('train/experiments').rglob('*.csv'):
        if ROOT in csv_path.parents:
            continue
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames and 'scene_seed' in reader.fieldnames:
                for row in reader:
                    if row.get('scene_seed'):
                        historical_seeds.add(int(row['scene_seed']))
    if fresh_seeds & historical_seeds:
        raise RuntimeError('Fresh confirmation overlaps previously recorded scenes.')
    if len(fresh_seeds) != 512:
        raise AssertionError('Confirmation roster is not unique.')
    spec = dict(selected=selected, code=code, source=previous['source'],
        checkpoint_sha256=previous['checkpoint_sha256'], protocol_sha256=sha256(PROTOCOL),
        evidence_sha256={str(p):sha256(p) for p in inputs}, confirmation=fresh,
        analysis_sha256={name:sha256(Path(__file__).parent/'figures'/name) for name in ANALYSIS_FILES},
        historical_unique_scenes=len(historical_seeds), confidence_level=CONFIDENCE,
        bootstrap_repeats=10000, primary_defenders=6,
        strong_advantage='Every declared primary reference: >=10% mean capped-time reduction, '
            'paired stratified CI excludes zero, observed success is not lower, collisions do not increase, '
            'and conservative success difference lower bound is at least -3 percentage points.',
        engine='JitRolloutController serial; ParallelJitController equivalence verified independently')
    verify(spec)
    path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=True, stage='freeze', selected=selected, scenes=len(fresh),
                         historical_unique_scenes=len(historical_seeds))), flush=True)


def initialize(settings):
    global SETTINGS, BATCHED
    SETTINGS = settings
    prior.old.initialize(str(prior.old.CHECKPOINT))
    BATCHED = BatchedSourcePolicy(prior.old._POLICY)
    prior.old.new_controller = factory


def factory(n, method, h):
    if (n,method) not in RULES:
        if method == 'best_fixed':
            rule = TaskTimePredictiveController(n, prior.old._POLICY, fixed_template='lead_guard',
                                               blend=1., block_steps=10, adaptive_attacker=False)
        else:
            rule = JitRolloutController(n, prior.old._POLICY, prediction_policy=BATCHED, **SETTINGS[method])
        RULES[n,method] = rule
    return RULES[n,method]


def job(scene, methods):
    rows = []
    for i in np.random.default_rng(scene['scene_seed']+79243).permutation(len(methods)):
        method = methods[i]
        if method in ('original','cbf'):
            row = prior.source.rollout(scene, method)
            if method == 'original':
                row['original_equivalent'] = row['trajectory_sha256']==prior.source.direct_original(scene, prior.old._POLICY)
                if not row['original_equivalent']:
                    raise AssertionError('Original source replay differs.')
        else:
            h = SETTINGS[method]['block_steps'] if method in SETTINGS else 10
            row = prior.old.new_rollout(scene, method, h)
        row['failure_capped_capture_time'] = row['duration'] if row['capture'] else 80.
        rows.append(row)
    return rows


def confirm(workers):
    spec = json.loads((ROOT/'specification.json').read_text(encoding='utf-8'))
    verify(spec)
    if (ROOT/'episodes.csv').exists():
        raise RuntimeError('Completed confirmation must not be overwritten.')
    started = time.perf_counter()
    groups = []
    methods = spec['selected']['methods']
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=(spec['selected']['settings'],)) as pool:
        pending = {pool.submit(job,scene,methods):i for i,scene in enumerate(spec['confirmation'])}
        for task in as_completed(pending):
            groups.append((pending[task], task.result()))
            print(f'[ROBUST-CONFIRM] {len(groups)}/512; {time.perf_counter()-started:.1f}s', flush=True)
    rows = [r for _,group in sorted(groups) for r in group]
    verify(spec)
    passed = len(rows)==512*len(methods) and len({(r['scene_seed'],r['method']) for r in rows})==len(rows)
    if not passed:
        raise AssertionError('Incomplete paired confirmation.')
    prior.source.write_csv(ROOT/'episodes.csv', rows)
    output = dict(passed=passed, rows=len(rows), elapsed_seconds=time.perf_counter()-started)
    (ROOT/'execution.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('freeze','confirm'), required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    freeze() if args.stage=='freeze' else confirm(args.workers)


if __name__ == '__main__':
    main()
