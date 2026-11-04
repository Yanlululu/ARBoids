"""Remove one actual event correction and retain the same learned continuation.

These are selected development-task causal diagnostics. Their outcomes are not
training labels and cannot replace the complete common-task endpoint results.
"""
import argparse
import concurrent.futures
import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.bridge import Source, World
from citf.contracts import Protocol
from citf.search import candidate_grid
from experiments.capture_window_diagnostic import geometry
from experiments.contact_event_policy import (ContactEventConfig, ContactEventModel,
    ContactEventController, retained_teacher)


worker = {}


def initialize(project, source, checkpoint, deadline):
    torch.set_num_threads(1)
    value = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = ContactEventModel(ContactEventConfig(**value['config']))
    model.load_state_dict(value['state'])
    worker.update(project=project, source=Source(project, source), model=model.eval(),
        teacher=retained_teacher(model.cfg.teacher_checkpoint), deadline=deadline)


@torch.no_grad()
def run(task, protocol, suppress=None):
    world = World(worker['project'], Protocol(**protocol), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    controller = ContactEventController(worker['source'], worker['model'], teacher=worker['teacher'])
    events, ordinal, steps = [], 0, 0
    while not world.done:
        if steps % 20 == 0 and time.time() >= worker['deadline']:
            return dict(completed=False, events=events, steps=steps, reason='deadline')
        before = geometry(world.state())
        force, trace = controller.act(world)
        transition = controller.transition
        if transition is not None:
            event = dict(ordinal=ordinal, step=steps, time=float(world.env.Current_T),
                phases=transition['phases'], changed=transition['changed'], observed=before,
                selected=trace['selected_indices'], teacher=trace['retained_teacher_indices'],
                physical=trace['configurations'][0], suppressed=ordinal == suppress)
            if ordinal == suppress:
                a, b, prior = worker['source'].commands(world.env)
                grid, _ = candidate_grid(prior)
                gate = grid[torch.arange(len(a)), torch.as_tensor(trace['retained_teacher_indices'])]
                force = (750 * (gate * a + (1 - gate) * b) + 250).numpy()
            events.append(event)
            ordinal += 1
        result = world.step(force)
        steps += 1
    return dict(completed=True, events=events, steps=steps, **result)


def attribute(job):
    baseline = run(job['task'], job['protocol'])
    if not baseline['completed']:
        return dict(completed=False, job=job, baseline=baseline, interventions=[])
    matched = all(baseline[k] == job['saved_result'][k]
                  for k in ('success', 'capture', 'collision', 'radial_only_failure', 'physical_breach'))
    interventions = []
    if matched:
        for event in baseline['events']:
            if not event['changed']:
                continue
            result = run(job['task'], job['protocol'], event['ordinal'])
            interventions.append(dict(removed_event=event, result=result))
            if not result['completed']:
                break
    return dict(completed=matched and all(x['result']['completed'] for x in interventions),
        job=job, baseline_matches_saved=matched, baseline=baseline, interventions=interventions)


def main():
    parser = argparse.ArgumentParser()
    for key in ('project', 'source', 'checkpoint', 'jobs', 'out', 'deadline'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    done = []

    def status():
        record = dict(completed=len(done) == len(jobs) and all(x['completed'] for x in done),
            expected_tasks=len(jobs), finished_tasks=len(done),
            matched_replays=sum(x.get('baseline_matches_saved', False) for x in done),
            completed_interventions=sum(sum(y['result']['completed'] for y in x['interventions']) for x in done),
            arguments=vars(args), updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (out / 'status.json').write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)

    status()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, args.checkpoint, deadline)) as pool:
        pending = {pool.submit(attribute, job): i for i, job in enumerate(jobs)}
        for future in concurrent.futures.as_completed(pending):
            record = future.result()
            (out / f'{pending[future]:04d}.json').write_text(json.dumps(record, indent=2))
            done.append(record)
            status()
    (out / 'tasks.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in done))


if __name__ == '__main__':
    main()
