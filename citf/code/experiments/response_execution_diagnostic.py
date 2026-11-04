"""Matched stochastic-versus-greedy task execution for frozen response policies.

No optimization is performed. All declared checkpoints, layouts, and action
seeds are retained; physical future seeds equal the recorded greedy evaluation.
"""
import argparse
import concurrent.futures
import datetime
import json
import multiprocessing
from pathlib import Path

import torch

from experiments.response_policy import initialize, rollout


def task_key(task):
    return tuple(task[k] for k in ('scene_seed', 'future_seed', 'agility', 'defenders'))


def main():
    parser = argparse.ArgumentParser()
    for key in ('project', 'source', 'tasks', 'out', 'deadline'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--replicates', type=int, default=1)
    parser.add_argument('--action-seed', type=int, default=93000000)
    parser.add_argument('--execution', choices=['sample', 'mean'], default='sample')
    args = parser.parse_args(); torch.set_num_threads(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    bank = json.loads(Path(args.tasks).read_text())
    if bank['role'] != 'validation':
        raise ValueError('Execution diagnosis uses the registered validation bank')
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    configs, greedy, jobs = [], {}, []
    for path in map(Path, args.checkpoints):
        value = torch.load(path, map_location='cpu', weights_only=False)
        configs.append(value['config']); label = f'{path.parent.name}_u{value["update"]:06d}'
        stages = [json.loads(x) for x in (path.parent / 'validation.jsonl').read_text().splitlines()]
        stage = next(x for x in stages if x['update'] == value['update'] and x['completed'])
        greedy[label] = {task_key(x['task']): x for x in stage['tasks']}
        parameters = {k: v.numpy() for k, v in value['state'].items()}
        for i, task in enumerate(bank['tasks']):
            if task_key(task) not in greedy[label]:
                raise ValueError('The frozen stage did not evaluate this task')
            for repeat in range(args.replicates):
                job = dict(task=task, state=parameters, training=True, record_transitions=False,
                    execution=args.execution,
                    policy_seed=args.action_seed + i * 100 + repeat,
                    protocol=bank.get('protocol', {}), deadline=deadline,
                    stop_file=str(out / 'STOP_EXECUTION_DIAGNOSTIC'))
                jobs.append((label, repeat, job))
    if any(c != configs[0] for c in configs):
        raise ValueError('Use one matched configuration family per diagnosis')
    status = dict(completed=False, arguments=vars(args), expected_jobs=len(jobs), finished_jobs=0)
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, configs[0])) as pool:
        pending = {pool.submit(rollout, job): (label, repeat) for label, repeat, job in jobs}
        for future in concurrent.futures.as_completed(pending):
            label, repeat = pending[future]
            try:
                result = future.result()
                row = dict(model=label, replicate=repeat, **result.get('record', result))
            except Exception as error:
                row = dict(model=label, replicate=repeat, completed=False, error=repr(error))
            records.append(row)
            with (out / 'tasks.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            status['finished_jobs'] = len(records)
            (out / 'status.json').write_text(json.dumps(status, indent=2))
            if len(records) % 16 == 0:
                print(json.dumps(dict(finished_jobs=len(records), expected_jobs=len(jobs))), flush=True)
    status['completed'] = len(records) == len(jobs) and all(x['completed'] for x in records)
    status['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    if not status['completed']:
        return
    summary = {}
    for label, reference in greedy.items():
        rows = [x for x in records if x['model'] == label]
        summary[label] = dict(N=len(rows), task_count=len(reference),
            execution=args.execution,
            physical_layouts=len({x['task']['scene_seed'] for x in reference.values()}),
            executed_success=sum(x['success'] for x in rows),
            executed_collision=sum(x['collision'] for x in rows),
            executed_radial_failure=sum(x['radial_only_failure'] for x in rows),
            greedy_success=sum(x['success'] for x in reference.values()),
            greedy_collision=sum(x['collision'] for x in reference.values()),
            greedy_radial_failure=sum(x['radial_only_failure'] for x in reference.values()),
            rescues=sum(x['success'] > reference[task_key(x['task'])]['success'] for x in rows),
            harms=sum(x['success'] < reference[task_key(x['task'])]['success'] for x in rows))
    (out / 'summary.json').write_text(json.dumps(dict(completed=True,
        scope='all declared frozen stages and initializations; validation task execution diagnosis',
        methods=summary), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
