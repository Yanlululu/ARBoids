"""Extend a prespecified reusable development pool with bounded parallel shards."""
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


def main():
    ap = argparse.ArgumentParser()
    for key in ('project', 'source', 'run', 'registry', 'train', 'validation'):
        ap.add_argument('--' + key, required=True)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--candidates', type=int, default=24)
    ap.add_argument('--futures', type=int, default=3)
    ap.add_argument('--behavior-checkpoint')
    ap.add_argument('--collector', default='citf.collect')
    ap.add_argument('--collector-subcommand')
    ap.add_argument('--root-times', type=float, nargs='+', default=[0, 2, 4, 8, 12, 20, 30, 40, 60])
    ap.add_argument('--deadline', default='2026-11-02T22:25:28+00:00')
    a = ap.parse_args(); out = Path(a.run).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'collection_status.json'
    if state_path.exists():
        raise ValueError('Collection identity already exists')
    deadline = datetime.datetime.fromisoformat(a.deadline).timestamp()
    state = {'arguments': vars(a), 'pid': os.getpid(), 'jobs': [], 'completed': False}
    # Alternate the first half of training and validation so a short pilot can
    # start before the entire reusable pool finishes.
    banks = [(role, path, json.loads(Path(path).read_text())) for role, path in
             [('train', a.train), ('validation', a.validation)]]
    for first, last in [(0, 32), (32, 1000000)]:
        for role, path, bank in banks:
            if bank['role'] != role:
                raise ValueError('Wrong declared data role')
            for start in range(first, min(last, len(bank['tasks'])), 2):
                name = f'{role}_{start:04d}'
                cmd = [sys.executable, '-u', '-m', a.collector,
                       *([a.collector_subcommand] if a.collector_subcommand else []), '--project', a.project,
                       '--source', a.source, '--tasks', path, '--registry', a.registry,
                       '--out', str(out / (name + '.pt')), '--layout-start', str(start),
                       '--layout-count', '2', '--candidates', str(a.candidates), '--futures', str(a.futures),
                       '--max-per-layout', str(len(a.root_times)), '--root-times', *map(str, a.root_times)]
                if a.behavior_checkpoint:
                    cmd += ['--behavior-checkpoint', a.behavior_checkpoint]
                state['jobs'].append({'name': name, 'command': cmd, 'status': 'pending'})
    lock = threading.Lock()
    def save():
        tmp = state_path.with_suffix('.pending')
        tmp.write_text(json.dumps(state, indent=2)); tmp.replace(state_path)
    def execute(job):
        remaining = deadline - time.time()
        if remaining <= 0:
            with lock:
                job['status'] = 'deadline_not_started'; save()
            return
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (out / (job['name'] + '.log')).open('w') as log:
            process = subprocess.Popen(job['command'], env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock:
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
        with lock:
            valid = (out / (job['name'] + '.pt')).is_file() and (out / (job['name'] + '.json')).is_file()
            job.update(status='completed' if code == 0 and valid else 'failed',
                       returncode=code, finished=time.time()); save()
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(execute, state['jobs']))
    state['completed'] = all(job['status'] == 'completed' for job in state['jobs'])
    state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
    print(json.dumps({'completed': state['completed'], 'jobs': len(state['jobs'])}), flush=True)


if __name__ == '__main__':
    main()
