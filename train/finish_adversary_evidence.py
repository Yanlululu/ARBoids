"""Fixed-panel opponent tests, with a barrier before every defender cohort."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.stats import binomtest

from evidence_adversary import ATTACKER_SEEDS
from evidence_eval import sha256
from evidence_matrix import SEEDS
from evidence_stats import paired_fixed, seed_comparison, holm
from finish_evidence import atomic_json, training_barrier
from finish_sac_evidence import complete_training


def attacker_barrier(root, wait):
    while True:
        records = []
        for seed in ATTACKER_SEEDS:
            path = root / 'results/attackers' / f'seed{seed}/complete.json'
            if path.exists():
                d = json.loads(path.read_text())
                if not d['passed'] or d['steps'] != 500000 or d['seed'] != seed or sha256(path.parent / 'best.pth') != d['sha256']:
                    raise RuntimeError(f'Attacker completion record does not match: {path}')
                records.append(d)
        if len(records) == len(ATTACKER_SEEDS):
            return records
        if not wait:
            raise RuntimeError('All three attackers must finish before final testing')
        marker = root / 'jobs/attackers.json'
        if marker.exists() and os.name == 'posix':
            pid = json.loads(marker.read_text())['pid']
            proc = Path('/proc') / str(pid)
            if not proc.exists() or b'evidence_adversary.py' not in (proc / 'cmdline').read_bytes():
                raise RuntimeError('The attacker training queue stopped before completion')
        time.sleep(30)


def run_panel(root, cohort, defenders, jobs):
    out = root / 'results/adversary-test' / cohort
    out.mkdir(parents=True, exist_ok=True)
    queue = [(name, kind, checkpoint, opponent) for name, kind, checkpoint in defenders
             for opponent in ('apf', *ATTACKER_SEEDS)]
    active, failures = [], []
    while queue or active:
        while queue and len(active) < jobs:
            name, kind, checkpoint, opponent = queue.pop(0)
            attacker = root / 'results/attackers' / f'seed{opponent}/best.pth' if opponent != 'apf' else None
            output = out / f'{name}-opponent{opponent}.json'
            if output.exists():
                cached = json.loads(output.read_text())
                if (cached['passed'] and len(cached['rows']) == 2048
                        and cached['defender']['checkpoint_sha256'] == sha256(checkpoint)
                        and cached['evaluator_sha256'] == sha256(root / 'train/evidence_adversary.py')
                        and cached['attacker'].get('sha256') == (sha256(attacker) if attacker else None)):
                    continue
                raise RuntimeError(f'Existing opponent result differs from frozen inputs: {output}')
            command = [sys.executable, '-X', 'utf8', '-u', str(root / 'train/evidence_adversary.py'),
                       'evaluate', '--root', str(root), '--kind', kind, '--checkpoint', str(checkpoint),
                       '--output', str(output), '--eval-seed', '9050000', '--episodes', '2048']
            if attacker:
                command += ['--attacker', str(attacker)]
            log = output.with_suffix('.log').open('a', encoding='utf-8')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append((output.name, process, log))
            print(f'[OPPONENT TEST START] {cohort}/{output.name}', flush=True)
        remaining = []
        for name, process, log in active:
            code = process.poll()
            if code is None:
                remaining.append((name, process, log))
            else:
                log.close()
                if code:
                    failures.append(dict(name=name, returncode=code))
                    queue.clear()
                print(f'[OPPONENT TEST END] {name} returncode={code}', flush=True)
        active = remaining
        atomic_json(out / 'status.json', dict(queued=len(queue), failures=failures,
                    active=[dict(name=n, pid=p.pid) for n, p, _ in active]))
        if active:
            time.sleep(10)
    if failures:
        raise RuntimeError(f'Opponent tests failed: {failures}')


def load_rows(root, cohort, defender, opponent):
    data = json.loads((root / 'results/adversary-test' / cohort / f'{defender}-opponent{opponent}.json').read_text())
    if not data['passed'] or len(data['rows']) != 2048:
        raise RuntimeError('Incomplete adversarial evaluation')
    return sorted(data['rows'], key=lambda r: r['seed'])


def fixed_analysis(root):
    results, values = {}, []
    for opponent in ('apf', *ATTACKER_SEEDS):
        a = load_rows(root, 'frozen', 'channel', opponent)
        b = load_rows(root, 'frozen', 'reference', opponent)
        results[str(opponent)] = {}
        for metric, code in (('success', None), ('collision', 2), ('breach', 1)):
            left = [dict(r, outcome_code=r['success']) for r in a] if code is None else a
            right = [dict(r, outcome_code=r['success']) for r in b] if code is None else b
            test = paired_fixed(left, right, 1 if code is None else code)
            n = test['episodes']
            x = binomtest(test['candidate_only'], n).proportion_ci(confidence_level=.975)
            y = binomtest(test['reference_only'], n).proportion_ci(confidence_level=.975)
            test['conservative_paired_95_interval'] = [x.low-y.high, x.high-y.low]
            results[str(opponent)][metric] = test
            values.append(test)
    for value, p in zip(values, holm([v['exact_mcnemar_two_sided_p'] for v in values])):
        value['holm_12_exploratory_p'] = p
    atomic_json(root / 'artifacts/adversary-frozen-analysis.json', dict(passed=True, comparisons=results,
                scope='Two fixed defenders, each against the same fixed attacker panel; conditional inference'))


def matrix_analysis(root):
    results, all_values = {}, []
    events = dict(success=lambda r: r['success'], collision=lambda r: r['outcome_code'] == 2,
                  breach=lambda r: r['outcome_code'] == 1)
    for control in ('scalar', 'arboids'):
        contrasts, arrays = {}, {}
        for opponent in ('apf', *ATTACKER_SEEDS):
            candidates, references = [], []
            for seed in SEEDS:
                a = load_rows(root, 'matrix', f'full-seed{seed}', opponent)
                b = load_rows(root, 'matrix', f'{control}-seed{seed}', opponent)
                paired_fixed(a, b, 2)
                candidates.append(a)
                references.append(b)
            contrasts[str(opponent)] = {}
            arrays[opponent] = {}
            for metric, f in events.items():
                a = np.asarray([[f(r) for r in rows] for rows in candidates], dtype=float)
                b = np.asarray([[f(r) for r in rows] for rows in references], dtype=float)
                arrays[opponent][metric] = (a, b)
                result = seed_comparison(a, b)
                contrasts[str(opponent)][metric] = result
                all_values.append(result)
        contrasts['fixed_panel_mean'] = {}
        for metric in events:
            a = np.mean([values[metric][0] for values in arrays.values()], axis=0)
            b = np.mean([values[metric][1] for values in arrays.values()], axis=0)
            result = seed_comparison(a, b)
            result['interpretation'] = 'Equal weighting of APF and the three fixed attackers; '
            result['interpretation'] += 'bootstrap preserves opponent pairing and resamples training seeds and initial scenarios.'
            contrasts['fixed_panel_mean'][metric] = result
            all_values.append(result)
        results[control] = contrasts
    for value, p in zip(all_values, holm([v['exact_paired_sign_flip_two_sided_p'] for v in all_values])):
        value['holm_30_exploratory_p'] = p
    atomic_json(root / 'artifacts/adversary-matrix-analysis.json', dict(passed=True, comparisons=results,
                interpretation='Generalization to this shared panel; no alternating defender adaptation or minimax claim'))
    lines = ['# Learned-opponent evidence', '', 'Differences are full minus control, in percentage points.', '',
             '| Control | Opponent | Success | Collision | Target breach |', '|---|---|---:|---:|---:|']
    for control, conditions in results.items():
        for opponent, result in conditions.items():
            scores = [f'{100 * result[k]["paired_difference"]:.3f}' for k in events]
            lines.append(f'| {control} | {opponent} | ' + ' | '.join(scores) + ' |')
    lines += ['', 'Seed-level effects, uncertainty intervals and all corrected exploratory tests are retained in JSON.',
              'These tests do not reproduce alternating attacker/defender training.', '']
    (root / 'artifacts/adversary-results.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = dict(analysis_sha256=sha256(__file__), evaluator_sha256=sha256(root / 'train/evidence_adversary.py'),
                    attacker_seeds=list(ATTACKER_SEEDS), defender_seeds=list(SEEDS), test_seed=9050000, episodes=2048,
                    contrasts=['full-scalar', 'full-arboids'], correction='Holm over all 30 exploratory matrix contrasts')
    lock = root / 'artifacts/adversary-analysis-protocol.json'
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise RuntimeError('The locked opponent analysis protocol changed')
    atomic_json(lock, protocol)
    attacker_barrier(root, args.wait)
    run_panel(root, 'frozen', [('channel', 'channel', root / 'inputs/channel-frozen.pth'),
                             ('reference', 'arboids', root / 'inputs/arboids-reference.pth')], args.jobs)
    fixed_analysis(root)
    training_barrier(root, args.wait)
    complete_training(root, args.wait)
    defenders = []
    for seed in SEEDS:
        for arm in ('full', 'scalar'):
            defenders.append((f'{arm}-seed{seed}', 'channel', root / 'results/matrix' / f'{arm}-seed{seed}/channel-mappo-best1.pth'))
        defenders.append((f'arboids-seed{seed}', 'arboids', root / 'results/sac' / f'arboids-seed{seed}/adares-best.pth'))
    run_panel(root, 'matrix', defenders, args.jobs)
    matrix_analysis(root)
    atomic_json(root / 'artifacts/adversary-evidence-complete.json', dict(passed=True, completed_unix=time.time()))


if __name__ == '__main__':
    main()
