"""Bounded paired VRX study, with separate ROS domains and Gazebo partitions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'train'))
import study_runtime
import numpy as np
from source_arboids import verify_source, verify_source_config, sha256


METHODS = ('original','predictive_independent','predictive_joint','cbf')
DOMAINS = Queue()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def trial_job(item, args):
    scene, method, order = item
    base_id = f'{scene["cell"]}-{scene["seed"]}-{method}'
    run_id = base_id
    output = args.output_dir/run_id
    attempt = 1
    while args.resume and output.exists() and (output/'result.json').exists():
        previous=json.loads((output/'result.json').read_text())
        if previous.get('passed'):
            break
        # Preserve an infrastructure failure and retry the same fixed scene,
        # never a valid collision/breach outcome. Limit recovery to two attempts.
        error=previous.get('error','')
        if previous.get('control_steps',0) or 'Gazebo launch exited' not in error:
            raise RuntimeError(f'Existing failure needs investigation before retry: {run_id}: {error}')
        attempt+=1
        if attempt>2:
            raise RuntimeError(f'Repeated VRX startup failure: {base_id}')
        run_id=f'{base_id}-attempt-{attempt}'
        output=args.output_dir/run_id
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        result = json.loads((output/'result.json').read_text())
        if not result.get('passed') or result.get('method') != method:
            raise RuntimeError(f'Existing trial is invalid: {output}')
    else:
        runner_name=('run_gate_contribution_sync_start.py' if getattr(args,'synchronized_start',False)
                     else 'run_gate_contribution.py')
        command = [sys.executable,'-X','utf8','-u', str(Path(__file__).with_name(runner_name)),
            '--method',method,'--checkpoint',str(args.checkpoint), '--controller','AdaRes',
            '--termination-rule','source','--setting',str(scene['setting']),
            '--num-robots',str(scene['defenders']+1),'--agility',str(scene['agility']),
            '--duration','100','--seed',str(scene['seed']),'--headless',
            '--assets-dir',str(args.assets_dir),'--output-dir',str(args.output_dir),'--run-id',run_id]
        if scene['seed']==args.seed and method=='predictive_joint' and args.capture_first:
            command.append('--capture-frames')
        # Every live trial has its own ROS domain. The child runner separately
        # creates a unique GZ_PARTITION, avoiding cross-trial pose/thrust mixing.
        domain = DOMAINS.get()
        env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY='1')
        try:
            with (args.output_dir/f'{run_id}.log').open('w', encoding='utf-8') as stream:
                process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        finally:
            DOMAINS.put(domain)
        if not (output/'result.json').exists():
            raise RuntimeError(f'Trial produced no result: {run_id}; see its log')
        result = json.loads((output/'result.json').read_text())
        result['returncode'] = process.returncode
        result['ros_domain_id'] = domain
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        if process.returncode or not result.get('passed'):
            raise RuntimeError(f'VRX infrastructure failure: {run_id}: {result.get("error")}')
    source = verify_source()
    if result.get('source') != source or result.get('checkpoint_sha256') != sha256(args.checkpoint):
        raise RuntimeError(f'Original source/checkpoint provenance mismatch: {run_id}')
    trajectory = np.load(output/'trajectory.npz')
    positions = trajectory['DefPos'].reshape(-1,scene['defenders'],2)
    a,b = np.triu_indices(scene['defenders'],1)
    distance = np.linalg.norm(positions[:,a]-positions[:,b],axis=-1)
    terminal = np.asarray(result['terminal_positions'],dtype=float)[1:]
    terminal_distance = np.linalg.norm(terminal[a]-terminal[b],axis=-1)
    return dict(**scene, method=method, run_id=run_id, attempt=attempt, success=int(result['success']),
        collision=int(result['defender_collision']), breach=int(result['outcome_code']==1),
        outcome=int(result['outcome_code']), minimum_distance=float(min(distance.min(),terminal_distance.min())),
        duration=float(result['simulation_seconds']), control_steps=result['control_steps'],
        warning_steps=int(trajectory['GateWarning'].sum()),
        mean_policy_ms=1000*result['inference_seconds']['mean'],
        p95_policy_ms=1000*result['inference_seconds']['p95'],
        deadline_misses=result['control_deadline_misses'], wall_seconds=result['wall_seconds'],
        ros_domain_id=result.get('ros_domain_id'),
        start_simulation_stamp=float(trajectory['SimulationStamp'][0]),
        first_defender_positions=json.dumps(positions[0].tolist()),
        first_attacker_position=json.dumps(trajectory['AttPos'][0].tolist()),
        initial_poses=result['initial_poses'], source_unchanged=result['source_unchanged'])


def pairing_error(rows):
    if len(rows)!=len(METHODS) or {r['method'] for r in rows}!=set(METHODS):
        raise RuntimeError('A pairing block must contain all four methods exactly once.')
    if len({r['initial_poses'] for r in rows})!=1:
        raise RuntimeError('Generated poses differ within a pairing block.')
    starts=[np.asarray([json.loads(r['first_attacker_position']),*json.loads(r['first_defender_positions'])])
            for r in rows]
    return max(float(np.linalg.norm(p-starts[0],axis=-1).max()) for p in starts)


def repair_initial_pairing(rows,scenes,args):
    """Replace an entire block only for the fixed pre-control pairing criterion.

    Outcomes are not used for selection. All four methods are rerun together,
    under the unchanged protocol, with at most two repair blocks per scene.
    Original result directories remain in place and replacements are explicit.
    """
    replacements=[]
    for scene in scenes:
        old=[r for r in rows if r['cell']==scene['cell'] and r['seed']==scene['seed']]
        error=pairing_error(old)
        if error<1e-7:
            continue
        accepted=None
        synchronized=getattr(args,'synchronized_start_repair',False)
        for attempt in ((1,) if synchronized else (1,2)):
            prefix='pairing-sync' if synchronized else 'pairing-repair'
            folder_name=f'{prefix}-{scene["cell"]}-{scene["seed"]}-attempt-{attempt}'
            child=argparse.Namespace(**vars(args))
            child.synchronized_start=synchronized
            child.output_dir=args.output_dir/folder_name
            child.output_dir.mkdir(parents=True,exist_ok=args.resume)
            methods=np.random.default_rng(scene['seed']+824).permutation(METHODS)
            candidate=[]
            with ThreadPoolExecutor(args.workers) as pool:
                pending=[pool.submit(trial_job,(scene,str(method),i),child) for i,method in enumerate(methods)]
                for completed,future in enumerate(as_completed(pending),1):
                    candidate.append(future.result())
                    print(f'[PAIRING REPAIR] {scene["cell"]} seed={scene["seed"]} '
                          f'block={attempt} completed={completed}/4',flush=True)
            candidate.sort(key=lambda r:r['method'])
            new_error=pairing_error(candidate)
            if {r['initial_poses'] for r in candidate}!={r['initial_poses'] for r in old}:
                raise RuntimeError('A pairing repair changed the generated source initial state.')
            if new_error<1e-7:
                accepted=candidate
                for row in accepted:
                    row['replaces_run_id']=next(r['run_id'] for r in old if r['method']==row['method'])
                    row['run_id']=f'{folder_name}/{row["run_id"]}'
                    row['pairing_repair_attempt']=attempt
                    row['original_pairing_error_m']=error
                replacements.append(dict(cell=scene['cell'],seed=scene['seed'],
                    original_error_m=error,accepted_error_m=new_error,repair_attempt=attempt,
                    synchronized_model_start=synchronized,
                    replaced_methods=list(METHODS),reason='First-control position mismatch > 1e-7 m'))
                break
            print(f'[PAIRING REPAIR] block still differs by {new_error:.9g} m',flush=True)
        if accepted is None:
            bound='one synchronized-start block' if synchronized else 'two whole-block attempts'
            raise RuntimeError(f'Pairing still failed after {bound}: {scene}')
        rows=[r for r in rows if not(r['cell']==scene['cell'] and r['seed']==scene['seed'])]+accepted
    return rows,replacements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--assets-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--dock-scenes',type=int,default=16)
    parser.add_argument('--ocean-scenes',type=int,default=8)
    parser.add_argument('--team-scenes',type=int,default=8)
    parser.add_argument('--seed',type=int,default=139000000)
    parser.add_argument('--capture-first',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--repair-pairing',action='store_true',
                        help='Rerun complete four-method blocks that fail the fixed first-position criterion')
    parser.add_argument('--synchronized-start-repair',action='store_true',
                        help='Start physics after all models are inserted in pairing-repair blocks')
    args=parser.parse_args()
    if args.workers not in (1,2):
        parser.error('Use one or two isolated VRX workers.')
    for domain in range(30,30+args.workers):
        DOMAINS.put(domain)
    args.checkpoint=args.checkpoint.resolve()
    args.assets_dir=args.assets_dir.resolve()
    args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    source=verify_source()
    verify_source_config(args.checkpoint.with_name('config.yaml'))
    scenes=[]
    for cell,setting,n,agility,count,offset in (
        ('dock-n3',1,3,2.25,args.dock_scenes,0),
        ('ocean-n3',0,3,2.25,args.ocean_scenes,10000),
        ('dock-n4',1,4,2.5,args.team_scenes,20000)):
        for i in range(count):
            scenes.append(dict(cell=cell,setting=setting,defenders=n,agility=agility,seed=args.seed+offset+i))
    jobs=[]
    for scene in scenes:
        methods=np.random.default_rng(scene['seed']+824).permutation(METHODS)
        offset=len(jobs)
        jobs.extend((scene,str(method),offset+i) for i,method in enumerate(methods))
    specification=dict(source=source,checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),
        methods=METHODS,scenes=scenes,workers=args.workers,paired_unit='Initial pose seed within configuration',
        capture_radius=5.5,collision_radius=5.,duration_limit=100.,original_observation_padding=True,
        common_settle_seconds=2.,
        common_prelaunch_warmup=True,
        original_velocity_callback=True,independent_ros_domains=True,
        source_runner_sha256=sha256(Path(__file__).with_name('run_gate_contribution.py')))
    spec_path=args.output_dir/'specification.json'
    if args.resume and spec_path.exists() and json.loads(spec_path.read_text()) != json.loads(json.dumps(specification)):
        raise ValueError('Resume requires the same fixed specification.')
    spec_path.write_text(json.dumps(specification,indent=2)+'\n')
    write_csv(args.output_dir/'scenes.csv',scenes)
    rows=[]
    started=time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        pending={pool.submit(trial_job,job,args):job for job in jobs}
        try:
            for number,future in enumerate(as_completed(pending),1):
                row=future.result()
                rows.append(row)
                write_csv(args.output_dir/'episodes.csv',rows)
                print(f'[VRX] {number}/{len(jobs)} {row["cell"]} {row["method"]} '
                      f'success={row["success"]} collision={row["collision"]}; '
                      f'{time.monotonic()-started:.1f}s',flush=True)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    pre_repair_outcomes=[dict(method=method,episodes=sum(r['method']==method for r in rows),
        successes=sum(r['success'] for r in rows if r['method']==method),
        collisions=sum(r['collision'] for r in rows if r['method']==method),
        breaches=sum(r['breach'] for r in rows if r['method']==method)) for method in METHODS]
    replacements=[]
    if args.repair_pairing:
        rows,replacements=repair_initial_pairing(rows,scenes,args)
    rows.sort(key=lambda r:(r['cell'],r['seed'],r['method']))
    write_csv(args.output_dir/'episodes.csv',rows)
    paired=all(len({r['initial_poses'] for r in rows if r['cell']==s['cell'] and r['seed']==s['seed']})==1
               for s in scenes)
    start_errors=[]
    for scene in scenes:
        select=[r for r in rows if r['cell']==scene['cell'] and r['seed']==scene['seed']]
        starts=[np.asarray([json.loads(r['first_attacker_position']),*json.loads(r['first_defender_positions'])])
                for r in select]
        start_errors.append(max(float(np.linalg.norm(p-starts[0],axis=-1).max()) for p in starts))
    start_positions_match=max(start_errors,default=0.)<1e-7
    table=[]
    for cell in sorted({s['cell'] for s in scenes})+['all']:
        for method in METHODS:
            select=[r for r in rows if r['method']==method and (cell=='all' or r['cell']==cell)]
            if select:
                table.append(dict(cell=cell,method=method,episodes=len(select),
                    successes=sum(r['success'] for r in select),collisions=sum(r['collision'] for r in select),
                    breaches=sum(r['breach'] for r in select),
                    mean_policy_ms=float(np.mean([r['mean_policy_ms'] for r in select])),
                    deadline_misses=sum(r['deadline_misses'] for r in select)))
    write_csv(args.output_dir/'main_table.csv',table)
    summary=dict(passed=bool(paired and start_positions_match and all(r['source_unchanged'] for r in rows)),
        paired_initial_poses_verified=paired,scenes=len(scenes),method_episodes=len(rows),
        paired_control_start_positions_verified=start_positions_match,
        maximum_start_position_difference=max(start_errors,default=0.),
        whole_block_pairing_replacements=replacements,
        outcomes_before_pairing_recovery=pre_repair_outcomes,
        elapsed_seconds=time.monotonic()-started,main_table=table)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
