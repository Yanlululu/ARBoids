"""Common-task original baselines for the unified CIV development runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import torch

from civ_rl_experiment import make_tasks
import mechanism_controls as m
import r20_baseline_comparison as b


def initialize_baselines():
    global b
    import r20_independent_evidence as evidence
    evidence.baseline_runtime()
    b=sys.modules['r20_endpoint_runtime']
    torch.set_num_threads(1)


def episode(job):
    family,location,identity,task=job
    torch.set_num_threads(1)
    if family in ('arboids','rp','sac'):
        result=m.author_source_repeat_episode((tuple(task['scene']),task['future'],location,identity))
    else:
        result=b.evaluation_episode((family,location,tuple(task['scene']),task['future']))
    return dict(result,scene=task['scene'],future=task['future'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True);p.add_argument('--tasks',type=int,default=160)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--output-directory',required=True)
    args=p.parse_args();initialize_baselines()
    base=Path(__file__).resolve().parent/'experiments/author-source-fusion-20261011'
    output=Path(args.output_directory);tasks=make_tasks(args.seed,args.tasks,6)
    jobs=[];metadata={}
    for family in ('arboids','rp','sac','boids','mappo'):
        identity=location=None;training=None
        if family in ('arboids','rp','sac'):
            directory=base/'independent-training'/f'source-six-{family}-seed42-20261021'
            record,_=m.validate_author_source_repeat(directory)
            identity=record['selected_checkpoint_sha256'];location=str(directory)
            training=record['training_steps']
        elif family=='mappo':
            directory=base/'r20-six-baseline-comparison/mappo-seed42'
            record=torch.load(directory/'training-result.pth',map_location='cpu',weights_only=False)
            if not record['completed'] or record['steps']!=record['budget']:raise ValueError('A complete MAPPO endpoint is required.')
            location=str(directory/'best.pth');identity=hashlib.sha256(Path(location).read_bytes()).hexdigest()
            if identity!=record['best_sha256']:raise ValueError('MAPPO checkpoint identity changed.')
            training=record['steps']
        metadata[family]=dict(policy=location,policy_sha256=identity,training_steps=training)
        jobs.extend((family,location,identity,task) for task in tasks)
    rows={family:[] for family in metadata}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_baselines) as pool:
        for job,result in zip(jobs,pool.map(episode,jobs,chunksize=4)):
            family=job[0];rows[family].append(result)
            if len(rows[family])==len(tasks):
                record=dict(completed=True,role='Fixed development comparison of completed original baselines',
                    method=family,tasks=tasks,rows=rows[family],**metadata[family],
                    success=sum(x['success'] for x in rows[family]),collision=sum(x['collision'] for x in rows[family]))
                path=output/f'hour-baseline-{family}-fixed{args.tasks}.json'
                path.write_text(json.dumps(record,indent=2),encoding='utf-8')
                print(json.dumps(dict(completed=family,success=record['success'],collision=record['collision'])),flush=True)


if __name__=='__main__':main()
