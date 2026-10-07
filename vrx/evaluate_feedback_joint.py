"""Four fixed real-Gazebo integration pairs for the frozen feedback controller."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'train'))
import study_runtime
import numpy as np
from source_arboids import sha256, verify_source
from evaluate_gate_contribution import write_csv


SCENES=[dict(cell='dock-n3',seed=157000000,setting=0,defenders=3,agility=2.25),
        dict(cell='ocean-n3',seed=157000001,setting=1,defenders=3,agility=2.25),
        dict(cell='dock-n4',seed=157000002,setting=0,defenders=4,agility=2.25),
        dict(cell='dock-n5',seed=157000003,setting=0,defenders=5,agility=2.25)]
METHODS=('cbf','feedback_joint')
DOMAINS=Queue()


def job(scene,method,args):
    run_id=f'{scene["cell"]}-{scene["seed"]}-{method}'
    output=args.output_dir/run_id
    command=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('run_feedback_joint.py')),
        '--method',method,'--frozen-method',str(args.frozen_method),'--checkpoint',str(args.checkpoint),
        '--prediction-engine',args.prediction_engine,
        '--controller','AdaRes','--termination-rule','source','--setting',str(scene['setting']),
        '--num-robots',str(scene['defenders']+1),'--agility',str(scene['agility']),
        '--duration','100','--seed',str(scene['seed']),'--headless','--assets-dir',str(args.assets_dir),
        '--output-dir',str(args.output_dir),'--run-id',run_id]
    domain=DOMAINS.get()
    try:
        env=dict(os.environ,ROS_DOMAIN_ID=str(domain),ROS_LOCALHOST_ONLY='1')
        with (args.output_dir/(run_id+'.log')).open('w',encoding='utf-8') as stream:
            process=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
    finally:
        DOMAINS.put(domain)
    if not (output/'result.json').exists():
        raise RuntimeError(f'Missing VRX result: {run_id}')
    result=json.loads((output/'result.json').read_text())
    if process.returncode or not result.get('passed'):
        raise RuntimeError(f'VRX execution failed: {run_id}: {result.get("error")}')
    if (result['source']!=verify_source() or result['checkpoint_sha256']!=sha256(args.checkpoint)
            or not result['feedback_code_unchanged']):
        raise RuntimeError('VRX source/checkpoint validation failed.')
    trajectory=np.load(output/'trajectory.npz')
    n=scene['defenders']
    positions=trajectory['DefPos'].reshape(-1,n,2)
    i,j=np.triu_indices(n,1)
    terminal=np.asarray(result['terminal_positions'])[1:]
    dmin=min(np.linalg.norm(positions[:,i]-positions[:,j],axis=-1).min(),
             np.linalg.norm(terminal[i]-terminal[j],axis=-1).min())
    control=1000*trajectory['ControlWallSeconds']
    return dict(**scene,method=method,run_id=run_id,passed=True,success=int(result['success']),
        collision=int(result['outcome_code']==2),source_loss=int(result['outcome_code']==1),
        outcome=result['outcome_code'],minimum_distance=float(dmin),steps=result['control_steps'],
        duration=result['simulation_seconds'],control_median_ms=float(np.median(control)),
        control_p95_ms=float(np.quantile(control,.95)),control_p99_ms=float(np.quantile(control,.99)),
        control_max_ms=float(control.max()),deadline_misses=int(np.sum(control>200.)),
        first_positions=json.dumps([trajectory['AttPos'][0].tolist(),*positions[0].tolist()]),
        first_simulation_stamp=float(trajectory['SimulationStamp'][0]),
        initial_poses=result['initial_poses'],source_unchanged=result['source_unchanged'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('train/experiments/feedback-joint-20261006/vrx'))
    parser.add_argument('--frozen-method',type=Path,default=Path('train/experiments/feedback-joint-20261006/frozen_method.json'))
    parser.add_argument('--checkpoint',type=Path,default=Path('/mnt/d/ARBoids/train/experiments/main-seed42-20260921/adares1.pth'))
    parser.add_argument('--assets-dir',type=Path,default=Path('/mnt/d/ARBoids/.vrx-assets'))
    parser.add_argument('--prediction-engine',choices=['reference','vectorized'],default='reference')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    spec=dict(scenes=SCENES,methods=METHODS,source=verify_source(),prediction_engine=args.prediction_engine,
              checkpoint_sha256=sha256(args.checkpoint),frozen_method_sha256=sha256(args.frozen_method),
              paired_start_tolerance_m=.01,scope='Bounded real-Gazebo integration; four new fixed pairs',
              code={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('run_feedback_joint.py'))})
    path=args.output_dir/'specification.json'
    if path.exists():
        raise FileExistsError('Integration roster already ran; inspect its outputs before any retry.')
    path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    for domain in (41,42):
        DOMAINS.put(domain)
    started=time.monotonic()
    rows=[]
    with ThreadPoolExecutor(2) as pool:
        pending=[pool.submit(job,s,m,args) for s in SCENES for m in METHODS]
        for future in as_completed(pending):
            row=future.result()
            rows.append(row)
            print(f'[VRX] {len(rows)}/8 {row["cell"]} {row["method"]} outcome={row["outcome"]}',flush=True)
    rows.sort(key=lambda r:(r['seed'],r['method']))
    pairs=[]
    for scene in SCENES:
        group=[r for r in rows if r['seed']==scene['seed']]
        positions=[np.asarray(json.loads(r['first_positions'])) for r in group]
        error=float(np.linalg.norm(positions[0]-positions[1],axis=-1).max())
        same_poses=len({r['initial_poses'] for r in group})==1
        pairs.append(dict(seed=scene['seed'],cell=scene['cell'],initial_error_m=error,
                          generated_poses_identical=same_poses,
                          paired_start_passed=bool(same_poses and error<=.01)))
    write_csv(args.output_dir/'episodes.csv',rows)
    write_csv(args.output_dir/'pairs.csv',pairs)
    summary=dict(passed=all(r['passed'] for r in rows),trials=len(rows),
                 paired_start_passed=all(p['paired_start_passed'] for p in pairs),pairs=pairs,
                 deadline_misses=sum(r['deadline_misses'] for r in rows),elapsed_seconds=time.monotonic()-started)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
