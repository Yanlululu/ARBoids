"""Bounded parallel first-wave development collection and matched references."""
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
    root = Path(__file__).resolve().parents[2]
    run = root / 'runs/v0'; source = root / 'source_archive'
    checkpoint = source / 'train/experiments/author-source-fusion-20261011/two-hour-development-20261026/best-m1-400.pth'
    common = ['--project', str(source), '--source', str(checkpoint), '--registry', str(run / 'layout_registry.json')]
    jobs = []
    def add(name, args, output):
        jobs.append({'id': name, 'argv': [sys.executable, '-u', *args], 'output': str(output), 'status': 'pending'})
    for bank_name, size, module in [('closed_pilot', 4, 'citf.evaluate'), ('diagnostic_roots', 2, 'experiments.branch_diagnosis')]:
        bank = json.loads((run / (bank_name + '.json')).read_text())
        for start in range(0, len(bank['tasks']), size):
            name = bank_name + f'_{start:03d}'; path = run / (name + '.json')
            path.write_text(json.dumps(dict(bank, tasks=bank['tasks'][start:start + size]), indent=2))
            output = run / name
            extra = ['--pool', '96', '--futures', '3', '--root-times', '0', '2', '4', '8', '12', '20'] if module.endswith('diagnosis') else []
            add(name, ['-m', module, *common, '--tasks', str(path), '--out', str(output), *extra], output / 'summary.json')
    # The first 16 train / 8 validation layouts become available early. The
    # rest extend the same prespecified reusable data pool, with no seed culling.
    for role, start, end in [('train', 0, 16), ('validation', 0, 8), ('train', 16, 64), ('validation', 8, 24)]:
        bank = run / ('train_pool.json' if role == 'train' else 'label_validation.json')
        for offset in range(start, end, 2):
            output = run / f'{role}_{offset:03d}.pt'
            add(f'collect_{role}_{offset:03d}', ['-m', 'citf.collect', *common, '--tasks', str(bank),
                '--out', str(output), '--layout-start', str(offset), '--layout-count', '2',
                '--candidates', '24', '--futures', '3', '--max-per-layout', '6',
                '--root-times', '0', '2', '4', '8', '12', '20'], output)
    state = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'deadline_utc': '2026-11-02T22:40:28+00:00', 'pid': os.getpid(), 'slots': 16,
             'evidence_role': 'development only', 'jobs': jobs}
    deadline = datetime.datetime.fromisoformat(state['deadline_utc']).timestamp()
    lock = threading.Lock(); status = run / 'wave1_status.json'
    def save():
        tmp = status.with_suffix('.pending'); tmp.write_text(json.dumps(state, indent=2)); tmp.replace(status)
    def execute(job):
        remaining = deadline - time.time()
        if remaining <= 0:
            with lock: job['status'] = 'deadline_not_started'; save()
            return
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (run / (job['id'] + '.log')).open('w') as log:
            process = subprocess.Popen(job['argv'], cwd=root / 'code', env=env, stdout=log, stderr=subprocess.STDOUT)
            with lock: job.update(status='running', pid=process.pid, started=time.time()); save()
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                code = -9
        output = Path(job['output']); valid = output.is_file()
        if valid and output.name == 'summary.json':
            valid = json.loads(output.read_text()).get('completed') is True
        with lock:
            job.update(status='completed' if code == 0 and valid else 'failed', returncode=code, finished=time.time())
            save(); print(job['id'], job['status'], flush=True)
    if status.exists():
        raise ValueError('First-wave driver already exists; inspect it before any resume')
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(execute, jobs))
    state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()


if __name__ == '__main__':
    main()
