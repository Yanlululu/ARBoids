"""Revisit the prospectively collected roots with executed one-tick conditioning."""
import argparse
import concurrent.futures
import datetime
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from citf.bridge import World
from citf.contracts import Protocol
from experiments import root_response_diagnostic as diagnostic
from experiments.response_data import sample_response


def initialize(project, source, checkpoint, out, deadline):
    diagnostic.initialize(project, source, checkpoint, out, deadline)
    diagnostic.worker['response_hold'] = 1


@torch.no_grad()
def recollect(path):
    old = torch.load(path, map_location='cpu', weights_only=False)
    meta = dict(old['metadata']); task = meta['task']; w = diagnostic.worker
    world = World(w['project'], Protocol(), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    for force in meta['prefix_forces']:
        world.step(force)
    tensors, metadata, steps = sample_response(world,
        np.asarray(meta['candidate_indices'], dtype=np.int64), meta['future_seeds'], meta)
    metadata['original_held_root'] = str(path)
    target = w['out'] / Path(path).name
    torch.save(dict(tensors=tensors, metadata=metadata), target)
    return dict(completed=True, root=Path(path).name, branch_transitions=steps,
                task=task, source_root=str(path))


def main():
    parser = argparse.ArgumentParser()
    for key in ['project', 'source', 'checkpoint', 'roots', 'out', 'deadline']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    paths = sorted(Path(args.roots).glob('*.pt'))
    status = dict(completed=False, arguments=vars(args), jobs=[dict(name=p.name, status='pending') for p in paths])
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, args.checkpoint, args.out, deadline)) as pool:
        pending = {pool.submit(recollect, str(path)): i for i, path in enumerate(paths)}
        for future in concurrent.futures.as_completed(pending):
            i = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(completed=False, error=repr(error), root=paths[i].name)
            with (out / 'records.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            status['jobs'][i].update(status='completed' if result['completed'] else 'incomplete', detail=result.get('error'))
            (out / 'status.json').write_text(json.dumps(status, indent=2))
            print(json.dumps(result), flush=True)
    status.update(completed=all(j['status'] == 'completed' for j in status['jobs']),
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
