"""Evaluate the locked SAC controls and compare independent paired training seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from evidence_eval import sha256, suites
from evidence_matrix import SEEDS
from evidence_stats import paired_fixed, seed_comparison, holm
from finish_evidence import atomic_json, load_rows


def complete_training(root, wait):
    while True:
        paths = [root / 'results/sac' / f'arboids-seed{s}' / 'complete.json' for s in SEEDS]
        if all(p.exists() for p in paths):
            data = [json.loads(p.read_text()) for p in paths]
            if not all(d['passed'] and d['steps'] == 1000000 and d['seed'] == s
                       for d, s in zip(data, SEEDS)):
                raise RuntimeError('SAC completion records do not match the fixed protocol')
            for p, d in zip(paths, data):
                if sha256(p.parent / 'adares-best.pth') != d['sha256']:
                    raise RuntimeError('Selected SAC checkpoint changed after training')
            return data
        if not wait:
            raise RuntimeError('All eight SAC seeds must finish before final testing')
        time.sleep(30)


def run_tests(root, jobs):
    queue, active, failures = list(SEEDS), [], []
    out = root / 'results/sac-test'
    out.mkdir(parents=True, exist_ok=True)
    while queue or active:
        while queue and len(active) < jobs:
            seed = queue.pop(0)
            destination = out / f'arboids-seed{seed}'
            log = (out / f'arboids-seed{seed}.log').open('a', encoding='utf-8')
            command = [sys.executable, '-X', 'utf8', '-u', str(root / 'train/evidence_eval.py'),
                       '--kind', 'arboids', '--checkpoint', str(root / 'results/sac' /
                       f'arboids-seed{seed}/adares-best.pth'), '--cohort', 'matrix', '--suite', 'all',
                       '--output', str(destination), '--device', 'cpu', '--parallel', '8']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
            active.append((seed, subprocess.Popen(command, env=env, stdout=log,
                                                  stderr=subprocess.STDOUT), log))
            print(f'[SAC FINAL TEST START] seed={seed}', flush=True)
        remaining = []
        for seed, process, log in active:
            code = process.poll()
            if code is None:
                remaining.append((seed, process, log))
            else:
                log.close()
                if code:
                    failures.append(dict(seed=seed, returncode=code))
                    queue.clear()
                print(f'[SAC FINAL TEST END] seed={seed} returncode={code}', flush=True)
        active = remaining
        atomic_json(out / 'status.json', dict(queued=len(queue), failures=failures,
                    active=[dict(seed=s, pid=p.pid) for s, p, _ in active]))
        if active:
            time.sleep(10)
    if failures:
        raise RuntimeError(f'SAC test jobs failed: {failures}')


def analyze(root):
    results = {}
    for spec in suites('matrix'):
        name, candidate, reference = spec['name'], [], []
        for seed in SEEDS:
            candidate.append(load_rows(root, 'full', seed, name))
            path = root / 'results/sac-test' / f'arboids-seed{seed}' / f'{name}.json'
            data = json.loads(path.read_text())
            completion = json.loads((root / 'results/sac' / f'arboids-seed{seed}/complete.json').read_text())
            if not data['passed'] or data['suite'] != spec or data['policy']['checkpoint_sha256'] != completion['sha256']:
                raise RuntimeError(f'SAC test inputs do not match: {path}')
            reference.append(sorted(data['rows'], key=lambda r: r['seed']))
            paired_fixed(candidate[-1], reference[-1], 2)
        events = dict(success=lambda r: r['success'], collision=lambda r: r['outcome_code'] == 2,
                      breach=lambda r: r['outcome_code'] == 1)
        results[name] = {metric: seed_comparison([[f(r) for r in rows] for rows in candidate],
                                               [[f(r) for r in rows] for rows in reference])
                         for metric, f in events.items()}
    adjusted = iter(holm([v['exact_paired_sign_flip_two_sided_p'] for c in results.values() for v in c.values()]))
    for condition in results.values():
        for value in condition.values():
            value['holm_48_secondary_p'] = next(adjusted)
    report = dict(passed=True, comparison='Full MAPPO minus from-scratch ARBoids SAC',
                  training_seeds=list(SEEDS), conditions=results,
                  interpretation='Whole training pipelines; MAPPO includes a shared teacher, demonstrations, '
                                 'collision-start revisits and a collision-cost critic. '
                                 'This is not an isolated architectural effect or an equal-total-cost comparison.',
                  analysis_sha256=sha256(__file__))
    atomic_json(root / 'artifacts/sac-analysis.json', report)
    lines = ['# Independent ARBoids SAC comparison', '', report['interpretation'], '',
             '| Condition | Success difference (pp) | Collision difference (pp) | Breach difference (pp) |',
             '|---|---:|---:|---:|']
    for name, condition in results.items():
        values = [f'{100 * condition[k]["paired_difference"]:.3f}' for k in ('success', 'collision', 'breach')]
        lines.append(f'| {name} | ' + ' | '.join(values) + ' |')
    lines += ['', 'All training-seed differences, crossed-bootstrap intervals and corrected tests are in sac-analysis.json.', '']
    (root / 'artifacts/sac-results.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = dict(source_sha256=sha256(__file__), evaluator_sha256=sha256(root / 'train/evidence_eval.py'),
                    statistics_sha256=sha256(root / 'train/evidence_stats.py'), seeds=list(SEEDS),
                    correction='Holm across all 48 secondary outcome/condition contrasts')
    lock = root / 'artifacts/sac-analysis-protocol.json'
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise RuntimeError('The locked SAC analysis protocol changed')
    atomic_json(lock, protocol)
    complete_training(root, args.wait)
    run_tests(root, args.jobs)
    while not (root / 'artifacts/evidence-complete.json').exists():
        if not args.wait:
            raise RuntimeError('The matched MAPPO final tests have not finished')
        time.sleep(30)
    analyze(root)
    atomic_json(root / 'artifacts/sac-evidence-complete.json', dict(passed=True, completed_unix=time.time()))


if __name__ == '__main__':
    main()
