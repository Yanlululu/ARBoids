"""Add a matched-horizon one-shot ablation to a completed paired gate suite."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path
import time

import torch

from diagnose_gate_space import new_output, write_json
from evaluate_gate_paper import initial_case, run_method
from gate_diagnostics import FrozenARBoids, digest


_policy = None


def initialize(checkpoint):
    global _policy
    torch.set_num_threads(1)
    _policy = FrozenARBoids(checkpoint)


def evaluate(reference):
    seed, future = int(reference['scene_seed']), int(reference['future_seed'])
    case = initial_case(_policy, seed)
    method = dict(name='Single H2', mode='rolling_once', horizon=2., gate=None)
    start = time.perf_counter()
    result = run_method(case, _policy, method, future).summary
    row = dict(scene_seed=seed, future_seed=future, method=method['name'], horizon=2.,
               wall_seconds=time.perf_counter()-start, **result)
    for key in ('success','collision','breach','capture','timeout','team_return','end_time'):
        value = float(reference[key])
        row['reference_'+key] = value
        row[key+'_delta'] = result[key]-value
    row['reference_trajectory_sha256'] = reference['trajectory_sha256']
    replay = run_method(case, _policy, dict(mode='baseline',horizon=3.,gate=None),future)
    if replay.summary['trajectory_sha256'] != reference['trajectory_sha256']:
        raise RuntimeError(f'Scene {seed} no longer matches the paired baseline.')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use 1-8 workers.')
    started=time.monotonic()
    original=json.loads((args.source_run/'summary.json').read_text(encoding='utf-8'))
    if not original['passed'] or not original['completed']:
        raise RuntimeError('Source evaluation did not pass.')
    root=Path(__file__).parent
    hashes=original['code_sha256'].copy()
    hashes[Path(__file__).name]=digest(Path(__file__))
    if any(digest(root/name)!=value for name,value in hashes.items()):
        raise RuntimeError('Evaluation source code changed.')
    if digest(original['checkpoint'])!=original['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed.')
    references=[row for row in csv.DictReader((args.source_run/'episodes.csv').open(encoding='utf-8'))
                if row['method']=='ARBoids']
    new_output(args.output_dir)
    specification=dict(source_summary_sha256=digest(args.source_run/'summary.json'),code_sha256=hashes,
                       checkpoint_sha256=original['checkpoint_sha256'],method='Single H2',
                       horizon=2.,mode='rolling_once',warning_distance=7.,change_penalty=.02)
    write_json(args.output_dir/'specification.json',specification)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=initialize,initargs=(original['checkpoint'],)) as pool:
        futures=[pool.submit(evaluate,reference) for reference in references]
        for i,future in enumerate(as_completed(futures),1):
            row=future.result()
            with (args.output_dir/'episodes.csv').open('a',newline='',encoding='utf-8') as file:
                writer=csv.DictWriter(file,fieldnames=list(row))
                if not rows: writer.writeheader()
                writer.writerow(row)
            rows.append(row)
            if i%32==0 or i==len(references): print(f'[ABLATION] {i}/{len(references)}',flush=True)
    if any(digest(root/name)!=value for name,value in hashes.items()) or digest(original['checkpoint'])!=original['checkpoint_sha256']:
        raise RuntimeError('Evaluation code or checkpoint changed during the ablation.')
    result=dict(passed=True,completed=True,**specification,episodes=len(rows),
                exact_baseline_replays=len(references),wall_seconds=time.monotonic()-started)
    write_json(args.output_dir/'summary.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
