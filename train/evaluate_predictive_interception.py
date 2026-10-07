"""Frozen, paired confirmation of task-level feedback policy improvement."""
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
from develop_rollout_interception import CONFIGS
from feedback_joint_fast import FastFeedbackJointController
from gate_mechanism_diagnostics import stable_number
from predictive_interception_v2 import TaskTimePredictiveController, TEMPLATES
from rollout_interception import RolloutInterceptionController
from source_arboids import sha256, verify_source


ROOT = Path('train/experiments/predictive-interception-20261006')
PROTOCOL = Path('docs/predictive-interception-study.md')
OLD_DATA = Path('train/experiments/fleet-selection-20261006/confirm.csv')
FIXED = {f'fixed_{t}_b{int(b*100)}': dict(fixed_template=t, blend=b, block_steps=10,
         adaptive_attacker=False) for t in TEMPLATES[1:] for b in (.5, 1.)}
METHODS = ('original', 'cbf', 'previous', 'predictive', 'short_value', 'held_prefix', 'best_fixed')
SELECTED = {}
RULES = {}


def initialize(selected):
    global SELECTED
    old.initialize(str(old.CHECKPOINT))
    old.new_controller = factory
    SELECTED = selected


def factory(n, method, h):
    key = n, method
    if key not in RULES:
        if method == 'previous':
            controller = FastFeedbackJointController(n, old._POLICY, block_steps=10)
        elif method in FIXED or method == 'best_fixed':
            name = method if method in FIXED else SELECTED['fixed']
            controller = TaskTimePredictiveController(n, old._POLICY, **FIXED[name])
        else:
            name = method if method in CONFIGS else SELECTED['predictive']
            settings = dict(CONFIGS[name])
            if method == 'short_value':
                settings.pop('tail_steps')
                controller = TaskTimePredictiveController(n, old._POLICY, **settings)
            else:
                controller = RolloutInterceptionController(n, old._POLICY, **settings,
                    prediction='held' if method == 'held_prefix' else 'feedback')
        RULES[key] = controller
    return RULES[key]


def scene_job(job):
    scene, methods = job
    rows = []
    for i in np.random.default_rng(scene['scene_seed']+894).permutation(len(methods)):
        method = methods[i]
        if method in ('original', 'cbf') or (method == 'best_fixed' and SELECTED['fixed'] == 'cbf'):
            row = source.rollout(scene, 'cbf' if method == 'best_fixed' else method)
            row['method'] = method
            if method == 'original':
                row['original_equivalent'] = row['trajectory_sha256'] == source.direct_original(scene, old._POLICY)
                if not row['original_equivalent']:
                    raise AssertionError('Direct author-code replay differs.')
        else:
            row = old.new_rollout(scene, method, 10)
        row['failure_capped_capture_time'] = row['duration'] if row['capture'] else 80.
        rows.append(row)
    return rows


def development_scenes():
    with OLD_DATA.open(encoding='utf-8') as file:
        rows = [r for r in csv.DictReader(file) if r['method'] == 'original']
    scenes = []
    for n in (2, 3, 4, 5, 6):
        for agility in (1.5, 2., 2.5, 3.):
            group = [r for r in rows if int(r['defenders']) == n and float(r['agility']) == agility]
            for row in sorted(group, key=lambda r:stable_number('mission-prototype-'+r['scene_seed']))[:8 if n==6 else 3]:
                scenes.append(dict(cell=row['cell'], scene_seed=int(row['scene_seed']),
                    noise_seed=int(row['noise_seed']), defenders=n, agility=agility))
    if len(scenes) != 80:
        raise AssertionError('Expected 80 old development scenes.')
    return scenes


def confirmation_scenes():
    scenes = []
    for i, (n, agility) in enumerate((n,a) for n in (2,3,4,5,6) for a in (1.5,2.,2.5,3.)):
        for repeat in range(32 if n==6 else 8):
            seed = 173000000+i*10000+repeat
            scenes.append(dict(cell=f'n{n}-a{agility:g}', scene_seed=seed, noise_seed=seed+1000000,
                               defenders=n, agility=agility))
    return scenes


def frozen_inputs():
    prior = json.loads(Path('train/experiments/fleet-selection-20261006/specification.json').read_text(encoding='utf-8'))
    for name, expected in prior['code'].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise RuntimeError('Previously frozen comparison changed: '+name)
    if sha256(old.CHECKPOINT) != prior['checkpoint_sha256']:
        raise RuntimeError('Actor checkpoint changed.')
    code = dict(prior['code'])
    for name in ('predictive_interception.py', 'predictive_interception_v2.py', 'attacker_motion_observer.py',
                 'rollout_interception.py', 'develop_rollout_interception.py', 'evaluate_predictive_interception.py'):
        code[name] = sha256(Path(__file__).with_name(name))
    return dict(code=code, source=verify_source(), checkpoint_sha256=sha256(old.CHECKPOINT),
        protocol_sha256=sha256(PROTOCOL), development_input_sha256=sha256(OLD_DATA),
        configurations=CONFIGS, fixed=FIXED, methods=list(METHODS),
        development=development_scenes(), confirmation=confirmation_scenes(),
        selection='primary N=6: max success, min collision, min capped capture time; then all scenes; then name')


def summarize(rows):
    results = []
    for method in ('cbf', *CONFIGS, *FIXED):
        for group_name, n in (('primary', 6), ('all', None)):
            group = [r for r in rows if r['method'] == method and (n is None or r['defenders']==n)]
            results.append(dict(method=method, group=group_name, episodes=len(group),
                **{key:sum(r[key] for r in group) for key in ('success','capture','collision','source_loss')},
                capped_capture_time=float(np.mean([r['failure_capped_capture_time'] for r in group]))))
    return results


def selection(table):
    index = {(r['method'],r['group']):r for r in table}
    def score(method):
        result = []
        for group in ('primary','all'):
            r = index[method,group]
            result += [-r['success'],r['collision'],r['capped_capture_time']]
        return (*result,method)
    return dict(predictive=min(CONFIGS,key=score), fixed=min(('cbf',*FIXED),key=score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('develop','confirm'), required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = ROOT/'validation'
    root.mkdir(parents=True,exist_ok=True)
    spec = json.loads(json.dumps(frozen_inputs()))
    spec_path = root/'specification.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding='utf-8')) != spec:
            raise RuntimeError('Frozen study changed.')
    else:
        spec_path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    if (root/f'{args.stage}.csv').exists():
        raise RuntimeError('Completed stage exists; no automatic overwrite.')
    selected = {} if args.stage=='develop' else json.loads((root/'frozen_parameters.json').read_text(encoding='utf-8'))['selected']
    methods = ('cbf',*CONFIGS,*FIXED) if args.stage=='develop' else METHODS
    scenes = spec['development' if args.stage=='develop' else 'confirmation']
    if {s['scene_seed'] for s in spec['development']} & {s['scene_seed'] for s in spec['confirmation']}:
        raise AssertionError('Development and confirmation overlap.')
    started = time.perf_counter()
    groups = []
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(selected,)) as pool:
        pending = {pool.submit(scene_job,(scene,methods)):i for i,scene in enumerate(scenes)}
        for task in as_completed(pending):
            groups.append((pending[task],task.result()))
            print(f'[{args.stage.upper()}] {len(groups)}/{len(scenes)}; {time.perf_counter()-started:.1f}s',flush=True)
    rows = [r for _,group in sorted(groups) for r in group]
    source.write_csv(root/f'{args.stage}.csv',rows)
    if args.stage=='develop':
        table = summarize(rows)
        selected = selection(table)
        source.write_csv(root/'development_summary.csv',table)
        (root/'frozen_parameters.json').write_text(json.dumps(dict(selected=selected,code=spec['code'],
            protocol_sha256=spec['protocol_sha256']),indent=2),encoding='utf-8')
        print('[FROZEN] '+json.dumps(selected),flush=True)
    passed = json.loads(json.dumps(frozen_inputs()))==spec and len(rows)==len(scenes)*len(methods)
    summary = dict(passed=passed,scenes=len(scenes),rows=len(rows),selected=selected,
                   elapsed_seconds=time.perf_counter()-started)
    (root/f'{args.stage}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
