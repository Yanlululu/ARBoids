"""Run a fixed development bank in bounded, independently committed shards."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def merge(out):
    from citf.contracts import canonical_hash, file_hash
    out = Path(out)
    state = json.loads((out / 'status.json').read_text())
    if not state['jobs'] or any(job['status'] != 'completed' for job in state['jobs']):
        raise ValueError('Cannot merge an incomplete evaluation')
    arguments = state['arguments']
    bank = json.loads(Path(arguments['tasks']).read_text())
    rows = []; identity = None
    for job in state['jobs']:
        directory = out / job['name']
        summary = json.loads((directory / 'summary.json').read_text())
        if not summary['completed']:
            raise ValueError('Cannot merge an incomplete shard: ' + job['name'])
        if identity is None:
            identity = summary['runtime']
        elif canonical_hash(identity) != canonical_hash(summary['runtime']):
            raise ValueError('Runtime identity differs across evaluation shards')
        for line in (directory / 'tasks.jsonl').read_text().splitlines():
            row = json.loads(line)
            row['trace'] = (Path(job['name']) / row['trace']).as_posix()
            rows.append(row)
    if [r['task'] for r in rows] != bank['tasks']:
        raise ValueError('Evaluation merge differs from the prespecified bank')
    pending = out / 'tasks.jsonl.pending'
    pending.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    pending.replace(out / 'tasks.jsonl')
    counts = {key: sum(r[key] for r in rows) for key in
              ('success', 'capture', 'collision', 'physical_collision', 'physical_breach', 'radial_only_failure', 'timeout')}
    for key in ('ever_radial_disadvantage','radial_recovery'):
        if all(key in row for row in rows):counts[key]=sum(row[key] for row in rows)
    summary = dict(completed=True, phase='development', N=len(rows), expected_N=len(rows), **counts,
                   runtime=identity, tasks_sha256=file_hash(arguments['tasks']),
                   registry_sha256=file_hash(arguments['registry']),
                   capped_capture_time=sum(r['capped_capture_time'] for r in rows)/len(rows),
                   latency_mean_ms=sum(r['latency_mean_ms']*r['steps'] for r in rows)/sum(r['steps'] for r in rows),
                   deadline_fraction=sum(r['deadline_fraction']*r['steps'] for r in rows)/sum(r['steps'] for r in rows),
                   mean_unique_candidates=sum(r['mean_unique_candidates']*r['steps'] for r in rows)/sum(r['steps'] for r in rows),
                   evidence_role='fixed-source development validation', arguments=arguments)
    pending = out / 'summary.json.pending'
    pending.write_text(json.dumps(summary, indent=2))
    pending.replace(out / 'summary.json')
    return summary


def main():
    ap = argparse.ArgumentParser()
    for key in ('project', 'tasks', 'registry', 'out'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--source', help='Required by source-based evaluators; unused by joint direct control')
    ap.add_argument('--checkpoint')
    ap.add_argument('--evaluator', default='citf.evaluate')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--opponent-spec', help='Frozen opponent used by every evaluation shard')
    ap.add_argument('--nominal-agility', type=float)
    ap.add_argument('--mixture-alpha', type=float)
    ap.add_argument('--guard-weight', type=float)
    ap.add_argument('--guard-margin', type=float)
    ap.add_argument('--guard-sigma-scale', type=float)
    ap.add_argument('--guard-sigma-floor', type=float)
    ap.add_argument('--encounter-weight', type=float)
    ap.add_argument('--corridor-ratio', type=float)
    ap.add_argument('--pursuit-weight', type=float)
    ap.add_argument('--collision-source', choices=['nominal', 'learned'])
    ap.add_argument('--impulse-reference', action='store_true')
    ap.add_argument('--nominal-slack', type=float)
    ap.add_argument('--ensemble-checkpoints', nargs='+')
    ap.add_argument('--guard-combination', choices=['envelope', 'learned'])
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--shard-size', type=int, default=4)
    ap.add_argument('--deadline', default='2026-11-02T22:40:28+00:00')
    ap.add_argument('--risk-weight', type=float)
    ap.add_argument('--budget', type=int, default=96)
    ap.add_argument('--bank', type=int, default=12)
    ap.add_argument('--rounds', type=int, default=1)
    ap.add_argument('--risk', default='expected', choices=['expected', 'mean_distance'])
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    if not a.source and a.evaluator != 'experiments.joint_replanning_policy':
        ap.error('--source is required by the selected evaluator')
    out = Path(a.out).resolve()
    bank = json.loads(Path(a.tasks).read_text())
    if bank['role'] != 'validation' or a.workers < 1 or a.shard_size < 1:
        raise ValueError('A fixed development validation bank and positive counts are required')
    out.mkdir(parents=True, exist_ok=a.resume)
    deadline = datetime.datetime.fromisoformat(a.deadline).timestamp()
    if a.resume:
        state = json.loads((out / 'status.json').read_text())
        stored = state['arguments']
        for key, value in vars(a).items():
            if key not in ('resume', 'workers', 'deadline') and stored.get(key) != value:
                raise ValueError('Evaluation manager resume argument changed: ' + key)
        # Even a legacy completed state may precede a missing/failed merge.
        # Completed shards are reused below; always finish the aggregate commit.
        state['completed'] = False
        state.setdefault('resume_history', []).append(dict(
            previous_pid=state['pid'], pid=os.getpid(), workers=a.workers, deadline=a.deadline,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        state['pid'] = os.getpid()
    else:
        arguments = {key: value for key, value in vars(a).items() if key != 'resume'}
        state = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'pid': os.getpid(), 'arguments': arguments, 'jobs': [], 'completed': False}
    for start in ([] if a.resume else range(0, len(bank['tasks']), a.shard_size)):
        name = f'shard_{start:04d}'
        path = out / (name + '.json')
        path.write_text(json.dumps(dict(bank, tasks=bank['tasks'][start:start+a.shard_size]), indent=2))
        command = [sys.executable, '-u', '-m', a.evaluator, '--project', a.project,
                   '--registry', a.registry, '--tasks', str(path),
                   '--out', str(out / name), '--budget', str(a.budget), '--bank', str(a.bank),
                   '--rounds', str(a.rounds), '--risk', a.risk, '--device', a.device]
        if a.source:
            command += ['--source', a.source]
        if a.nominal_agility is not None:
            command += ['--nominal-agility', str(a.nominal_agility)]
        if a.checkpoint:
            command += ['--checkpoint', a.checkpoint]
        if a.opponent_spec:
            command += ['--opponent-spec', a.opponent_spec]
        if a.risk_weight is not None:
            command += ['--risk-weight', str(a.risk_weight)]
        if a.mixture_alpha is not None:
            command += ['--mixture-alpha', str(a.mixture_alpha)]
        if a.guard_weight is not None:
            command += ['--guard-weight', str(a.guard_weight)]
        if a.guard_margin is not None:
            command += ['--guard-margin', str(a.guard_margin)]
        if a.guard_sigma_scale is not None:
            command += ['--guard-sigma-scale', str(a.guard_sigma_scale)]
        if a.guard_sigma_floor is not None:
            command += ['--guard-sigma-floor', str(a.guard_sigma_floor)]
        if a.encounter_weight is not None:
            command += ['--encounter-weight', str(a.encounter_weight)]
        if a.corridor_ratio is not None:
            command += ['--corridor-ratio', str(a.corridor_ratio)]
        if a.pursuit_weight is not None:
            command += ['--pursuit-weight', str(a.pursuit_weight)]
        if a.collision_source is not None:
            command += ['--collision-source', a.collision_source]
        if a.impulse_reference:
            command += ['--impulse-reference']
        if a.nominal_slack is not None:
            command += ['--nominal-slack', str(a.nominal_slack)]
        if a.ensemble_checkpoints:
            command += ['--ensemble-checkpoints', *a.ensemble_checkpoints]
        if a.guard_combination is not None:
            command += ['--guard-combination', a.guard_combination]
        state['jobs'].append({'name': name, 'command': command, 'status': 'pending'})
    lock = threading.Lock()
    def save():
        tmp = out / 'status.pending'
        tmp.write_text(json.dumps(state, indent=2)); tmp.replace(out / 'status.json')
    def execute(job):
        if job['status'] == 'completed':
            return
        remaining = deadline - time.time()
        if remaining <= 0:
            with lock:
                job['status'] = 'deadline_not_started'; save()
            return
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        command = list(job['command'])
        if a.resume and (out / job['name'] / 'evaluation.json').is_file():
            command.append('--resume')
        attempt = len(state.get('resume_history', []))
        suffix = f'.resume{attempt}.log' if a.resume else '.log'
        with (out / (job['name'] + suffix)).open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                if a.resume:
                    job.setdefault('attempts', []).append(dict(
                        previous_status=job['status'], command=command, log=job['name'] + suffix))
                job.update(status='running', pid=process.pid, started=time.time()); save()
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                code = -9
        path = out / job['name'] / 'summary.json'
        valid = path.exists() and json.loads(path.read_text()).get('completed') is True
        with lock:
            job.update(status='completed' if code == 0 and valid else 'failed',
                       returncode=code, finished=time.time()); save()
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(execute, state['jobs']))
    jobs_completed = all(job['status'] == 'completed' for job in state['jobs'])
    state['completed'] = False
    save()
    if not jobs_completed:
        raise RuntimeError('Some evaluation shards did not complete; retain all partial results')
    summary = merge(out)
    state['completed'] = True
    state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
