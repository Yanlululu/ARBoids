"""One-shot launch/status helper for the second authorized development window.

Only explicitly queued experiments are launched. This process exits after
one pass; it is not a recurring scheduler or an autonomous research loop.
"""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import subprocess


BASE = Path('/root/autodl-tmp/citf-10h-20261102')
RUN = BASE / 'runs/innovation_20261103'


def read(path, fallback):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def active_slots(with_policy=False):
    processes, managers = {}, {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = [x.decode(errors='replace') for x in (path / 'cmdline').read_bytes().split(b'\0') if x]
            if '-m' not in args or str(BASE) not in ' '.join(args):
                continue
            module = args[args.index('-m') + 1]
            if not module.startswith(('experiments.', 'citf.')) or module == __name__:
                continue
            parent = int(next(line.split(':', 1)[1] for line in (path / 'status').read_text().splitlines() if line.startswith('PPid:')))
            pid = int(path.name)
            processes[pid] = (module, parent)
            if (module in ('experiments.parallel_evaluate', 'experiments.collect_development',
                           'experiments.diagnose_radial_choices')
                    or module.endswith('configuration_policy')) and '--workers' in args:
                workers = int(args[args.index('--workers') + 1])
                # Policy rollout and GPU optimization are sequential phases:
                # the workers are idle while the parent optimizes a collected batch.
                flag, filename = ('--run', 'collection_status.json') if module.endswith('collect_development') else ('--out', 'status.json')
                state = read(Path(args[args.index(flag) + 1]) / filename, {})
                if 'jobs' in state:
                    workers = min(workers, sum(j['status'] in ('pending', 'running') for j in state['jobs']))
                managers[pid] = workers
        except (OSError, StopIteration):
            continue
    count = sum(managers.values()) + sum(pid not in managers and parent not in managers for pid, (_, parent) in processes.items())
    policy_count = sum(managers.get(pid, 0) for pid, (module, _) in processes.items()
                       if module.endswith('configuration_policy'))
    return (count, policy_count) if with_policy else count


def prepare_declared_deployment_checkpoints(queue):
    """Materialize only the already registered fixed-scale policy evaluations."""
    records = []
    targets = sorted({filename for job in queue for filename in job.get('required_files', [])
                      if filename.endswith('/selected_scale2.pt')})
    for filename in targets:
        target = Path(filename)
        if target.is_file() or not read(target.parent / 'status.json', {}).get('completed'):
            continue
        source = target.parent / 'best.pt'
        if not source.is_file():
            continue
        import torch
        from citf.train import atomic_save
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        record = dict(source_checkpoint=str(source), derived_checkpoint=str(target),
            selected_update=checkpoint['update'], original_policy_logit_scale=checkpoint['config']['policy_logit_scale'],
            policy_logit_scale=2., selected_weights_unchanged=True,
            role='declared deployment scale for original24-validation-selected weights; not a new training run',
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        checkpoint['config'] = dict(checkpoint['config'], policy_logit_scale=2.)
        checkpoint['derived_deployment'] = record
        atomic_save(checkpoint, target)
        records.append(record)
        print(json.dumps(dict(derived_deployment=record)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    protocol = read(RUN / 'protocol.json', {})
    records = read(RUN / 'launches.json', [])
    now = datetime.datetime.now(datetime.timezone.utc)
    count, policy_count = active_slots(with_policy=True)
    print(json.dumps(dict(utc=now.isoformat(), compute_slots=count,
        memory_gib=int(Path('/sys/fs/cgroup/memory.current').read_text()) / 2**30)))
    if args.status:
        for record in records:
            name = record['name']
            directory = RUN / name
            summary = read(directory / 'summary.json', {})
            state = read(directory / 'status.json', {})
            if not state:
                state = read(directory / 'collection_status.json', {})
            if summary.get('completed'):
                print(json.dumps(dict(name=name, success=summary['success'], N=summary['N'],
                    collision=summary['collision'], capture=summary['capture'],
                    capped_capture_time=summary['capped_capture_time'])))
            elif 'jobs' in state:
                print(json.dumps(dict(name=name, jobs=dict(collections.Counter(j['status'] for j in state['jobs'])))))
            elif state.get('completed'):
                print(json.dumps(dict(name=name, training_complete=True, update=state['update'],
                                     best_update=state.get('best', {}).get('update'))))
            else:
                log = directory / 'learning.jsonl'
                if log.is_file():
                    last = json.loads(log.read_text().splitlines()[-1])
                    print(json.dumps(dict(name=name, update=last['update'])))
                else:
                    log = RUN / (name + '.log')
                    if log.exists() and 'Traceback' in log.read_text():
                        print(json.dumps(dict(name=name, error=log.read_text()[-1800:])))
        return
    if now >= datetime.datetime.fromisoformat(protocol['deadline_utc']) - datetime.timedelta(minutes=20):
        raise SystemExit('Final launch cutoff reached')
    env = os.environ.copy()
    env.update(PYTHONPATH=str(BASE / 'code'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    names = {x['name'] for x in records}
    queue = read(RUN / 'queue.json', [])
    derived = prepare_declared_deployment_checkpoints(queue)
    if derived:
        protocol.setdefault('declared_deployment_derivations', []).extend(derived)
        (RUN / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    # Keep the GPU supplied while CPU workers execute the explicit evaluations.
    order = {'training': 0, 'collection': 1, 'evaluation': 2}
    waiting_training = False
    for job in sorted(queue, key=lambda item: order.get(item.get('kind'), 2)):
        if job['name'] in names or job.get('deferred_in_window'):
            continue
        if any(not read(Path(dependency), {}).get('completed') for dependency in job.get('dependencies', [])):
            continue
        if any(not Path(filename).is_file() for filename in job.get('required_files', [])):
            continue
        command = job['command']
        if '--deadline' in command and now >= datetime.datetime.fromisoformat(
                command[command.index('--deadline') + 1]):
            continue
        slots = int(command[command.index('--workers') + 1]) if '--workers' in command else 1
        is_policy = command[command.index('-m') + 1].endswith('configuration_policy')
        if is_policy:
            # Keep complete-bank feedback moving while long policy runs train.
            if policy_count + slots > protocol.get('policy_training_slots', 12):
                continue
        if count + slots > protocol['cpu_slots']:
            waiting_training |= job.get('kind') == 'training'
            continue
        if waiting_training and job.get('kind') != 'training':
            continue
        target_flag = '--run' if '--run' in command else '--out'
        target = Path(command[command.index(target_flag) + 1])
        if target.exists():
            raise ValueError('Unregistered existing experiment output: ' + str(target))
        with (RUN / (job['name'] + '.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        records.append(dict(job, pid=process.pid, started_utc=now.isoformat()))
        (RUN / 'launches.json').write_text(json.dumps(records, indent=2))
        names.add(job['name'])
        count += slots
        if is_policy:
            policy_count += slots
        print(json.dumps(dict(launched=job['name'], pid=process.pid, slots=slots)))


if __name__ == '__main__':
    main()
