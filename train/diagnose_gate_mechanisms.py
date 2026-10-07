"""Fixed, bounded mechanism diagnostics on the existing source-protocol data."""
import study_runtime
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import pickle
import time

import torch

from source_arboids import SourcePolicy, sha256, verify_source, verify_source_config
from evaluate_gate_contribution import write_csv
from gate_mechanism_diagnostics import (capture_scene, diagnose_case, recover_case,
                                       stable_number, HORIZON_STEPS, RECOVERY_METHODS)


_POLICY = None


def initialize(checkpoint):
    global _POLICY
    torch.set_num_threads(1)
    _POLICY = SourcePolicy(checkpoint)
    verify_source()


def worker(stage, data):
    if stage == 'capture':
        return capture_scene(data, _POLICY)
    if stage == 'diagnose':
        return diagnose_case(data, _POLICY)
    if stage == 'recover':
        return recover_case(data[0], data[1], _POLICY)
    raise ValueError(stage)


def parallel(stage, jobs, checkpoint, workers):
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=(str(checkpoint),)) as pool:
        futures = {pool.submit(worker, stage, item): index for index, item in enumerate(jobs)}
        for future in as_completed(futures):
            results.append((futures[future], future.result()))
            print(f'[{stage.upper()}] {len(results)}/{len(jobs)} complete; '
                  f'{time.perf_counter()-started:.1f}s', flush=True)
    return [result for _, result in sorted(results)]


def source_rows(root):
    rows = []
    for suite in ('mechanism', 'generalization'):
        for row in csv.DictReader((root/suite/'episodes.csv').open(encoding='utf-8')):
            row['suite'] = suite
            for key in ('scene_seed','noise_seed','defenders','outcome','success','collision','source_loss'):
                row[key] = int(row[key])
            row['agility'] = float(row['agility'])
            rows.append(row)
    return rows


def scene_of(row):
    return {key: row[key] for key in ('cell','scene_seed','noise_seed','defenders','agility')}


def select_jobs(rows):
    paired = defaultdict(dict)
    for row in rows:
        paired[(row['suite'], row['scene_seed'])][row['method']] = row
    natural = defaultdict(list)
    failures = defaultdict(list)
    for (suite, seed), methods in paired.items():
        original, predictive = methods['original'], methods['predictive_joint']
        if suite == 'generalization':
            natural[original['cell']].append(original)
        if not predictive['success'] and methods['cbf']['success']:
            failures[(predictive['defenders'], predictive['outcome'])].append(predictive)
    jobs = []
    for kind, groups, maximum, method in (
            ('natural', natural, 2, 'original'), ('failure', failures, 3, 'predictive_joint')):
        for group in sorted(groups):
            ordered = sorted(groups[group], key=lambda r: stable_number(f"mechanism-diagnostic-{r['suite']}-{r['scene_seed']}"))
            for row in ordered[:maximum]:
                jobs.append(dict(scene=scene_of(row), suite=row['suite'], kind=kind,
                                 method=method, expected=row))
    counts = dict(eligible_failure_scenes=sum(map(len, failures.values())),
                  failure_strata={str(k):len(v) for k,v in sorted(failures.items())},
                  selected_natural=sum(j['kind']=='natural' for j in jobs),
                  selected_failure=sum(j['kind']=='failure' for j in jobs))
    return jobs, counts


def transitions(rows):
    pairs = [('original','reactive_joint'), ('original','predictive_joint'), ('original','cbf'),
             ('reactive_joint','predictive_joint'), ('predictive_joint','cbf'), ('reactive_joint','cbf')]
    index = defaultdict(dict)
    for row in rows:
        index[(row['suite'], row['scene_seed'])][row['method']] = row
    result = []
    for suite in ('mechanism', 'generalization'):
        for before, after in pairs:
            count = Counter((v[before]['outcome'],v[after]['outcome'])
                            for (s,_),v in index.items() if s==suite)
            for (a,b),n in sorted(count.items()):
                result.append(dict(suite=suite, before=before, after=after,
                                   before_outcome=a, after_outcome=b, scenes=n))
    return result


def current_code():
    names = ['diagnose_gate_mechanisms.py', 'gate_mechanism_diagnostics.py',
             'source_arboids.py', 'gate_study_control.py', 'cbf_source_baseline.py',
             'evaluate_gate_contribution.py', 'study_runtime.py']
    return {name: sha256(Path(__file__).with_name(name)) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('train/experiments/gate-contribution-source-20261006'))
    parser.add_argument('--output-dir', type=Path, default=Path('train/experiments/gate-mechanism-diagnostics-20261006'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stage', choices=['all','capture','diagnose','recover'], default='all')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    source_spec = json.loads((args.source_root/'mechanism/specification.json').read_text(encoding='utf-8'))
    checkpoint = Path(source_spec['checkpoint'])
    if sha256(checkpoint) != source_spec['checkpoint_sha256']:
        raise RuntimeError('Recorded checkpoint changed.')
    verify_source_config(checkpoint.with_name('config.yaml'))
    for name, expected in source_spec['code'].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise RuntimeError(f'Recorded source-study implementation changed: {name}')
    rows = source_rows(args.source_root)
    jobs, counts = select_jobs(rows)
    code = current_code()
    specification = dict(source=verify_source(), checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint), code=code, source_root=str(args.source_root.resolve()),
        source_data={suite:sha256(args.source_root/suite/'episodes.csv') for suite in ('mechanism','generalization')},
        protocol_sha256=sha256(Path(__file__).resolve().parents[1]/'docs/gate-mechanism-diagnostics.md'),
        horizons_steps=HORIZON_STEPS, candidate_budget=96, held_grid='original + 0,.25,.5,.75,1',
        fine_grid='original + 0,.125,...,1', recovery_methods=RECOVERY_METHODS, unseen_futures=3,
        counts=counts, scene_unit='Existing development scene; repeated states/candidates/futures nested',
        selection=[{k:v for k,v in j.items() if k!='expected'} for j in jobs])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specification_path = args.output_dir/'specification.json'
    if specification_path.exists():
        prior = json.loads(specification_path.read_text(encoding='utf-8'))
        if prior['code'] != code or prior['source_data'] != specification['source_data']:
            raise RuntimeError('The staged run code or input changed; use a new output directory.')
    else:
        specification_path.write_text(json.dumps(specification, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(args.output_dir/'outcome_transitions.csv', transitions(rows))
    started = time.perf_counter()
    if args.stage in ('all','capture'):
        results = parallel('capture', jobs, checkpoint, args.workers)
        cases = [case for saved,_ in results for case in saved]
        audits = [audit for _,audit in results]
        with (args.output_dir/'states.pkl').open('wb') as file:
            pickle.dump(cases, file, protocol=5)
        write_csv(args.output_dir/'states.csv', [c['meta'] for c in cases])
        write_csv(args.output_dir/'source_replays.csv', audits)
        print(f'[CAPTURED] {len(cases)} states; {len(audits)} recorded trajectories verified.', flush=True)
    else:
        with (args.output_dir/'states.pkl').open('rb') as file:
            cases = pickle.load(file)
    if args.stage in ('all','diagnose'):
        results = parallel('diagnose', cases, checkpoint, args.workers)
        for field,name in [('metrics','prediction_metrics.csv'),('selections','selected_candidates.csv'),
                           ('candidates','candidate_outcomes.csv')]:
            write_csv(args.output_dir/name, [row for result in results for row in result[field]])
        write_csv(args.output_dir/'integrator_checks.csv', [r['audit'] for r in results])
        oracles = {r['oracle']['case_id']:r['oracle'] for r in results}
        (args.output_dir/'selected_diagnostic_actions.json').write_text(json.dumps(oracles, indent=2), encoding='utf-8')
    if args.stage in ('all','recover'):
        oracles = json.loads((args.output_dir/'selected_diagnostic_actions.json').read_text(encoding='utf-8'))
        tasks = [(c,oracles[c['meta']['case_id']]) for c in cases if c['meta']['category']=='failure']
        results = parallel('recover', tasks, checkpoint, args.workers)
        write_csv(args.output_dir/'recovery_episodes.csv', [row for rows in results for row in rows])
    unchanged = current_code()==code and sha256(checkpoint)==source_spec['checkpoint_sha256']
    unchanged &= verify_source()==specification['source']
    required = ['states.pkl','states.csv','source_replays.csv','prediction_metrics.csv','selected_candidates.csv',
                'candidate_outcomes.csv','integrator_checks.csv','selected_diagnostic_actions.json','recovery_episodes.csv']
    complete = all((args.output_dir/name).exists() for name in required)
    summary = dict(passed=bool(unchanged and complete), stage=args.stage, code_source_checkpoint_unchanged=unchanged,
                   complete=complete, states=len(cases), failure_states=sum(c['meta']['category']=='failure' for c in cases),
                   counts=counts, elapsed_this_stage_seconds=time.perf_counter()-started)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not unchanged or (args.stage=='all' and not complete):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
