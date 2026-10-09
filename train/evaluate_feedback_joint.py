"""Bounded development, frozen new-scene validation, and repair diagnostics."""
import study_runtime
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

import evaluate_gate_contribution as original_study
from source_arboids import sha256, verify_source, snapshot, source_mixture
from feedback_joint_control import FeedbackJointController, observe, project_thrust
from gate_mechanism_diagnostics import restore, stable_number


CHECKPOINT=Path(r'D:\ARBoids\train\experiments\main-seed42-20260921\adares1.pth')
OLD_ROOT=Path('train/experiments/gate-contribution-source-20261006')
DIAGNOSTIC_ROOT=Path('train/experiments/gate-mechanism-diagnostics-20261006')
BASELINES=('original','reactive_joint','predictive_joint','cbf')
ADDITIONS=('feedback_joint','held_joint','feedback_gate')
METHODS=BASELINES+ADDITIONS
_POLICY=None
_RULES={}


def initialize(checkpoint):
    global _POLICY
    original_study.initialize(checkpoint)
    _POLICY=original_study._POLICY


def new_controller(n,method,block_steps):
    key=n,method,block_steps
    if key not in _RULES:
        _RULES[key]=FeedbackJointController(n,_POLICY,block_steps,
            gate_only=method=='feedback_gate',prediction='held' if method=='held_joint' else 'feedback')
    return _RULES[key]


def new_rollout(scene,method,block_steps,case=None,future_seed=None,trace=False):
    if case is None:
        env,observations=original_study.setup_scene(scene)
        previous=None
    else:
        env,observations=restore(case['state'])
        previous=np.array([[b.left_thrust,b.right_thrust] for b in env.defender_list])
        if future_seed is not None:
            np.random.seed(future_seed)
            for _ in range(case['meta']['control_step']):
                np.random.normal(0.,.3,2)
                for _ in range(env.defender_num+1):
                    env.generate_random_current()
    rule=new_controller(env.defender_num,method,block_steps)
    rule.reset(previous)
    rng=np.random.get_state()
    warm_started=time.perf_counter()
    # Explicit warmup is outside per-tick measurements and cannot affect noise.
    rule.feedback(observe(env,observations),'baseline')
    warm_seconds=time.perf_counter()-warm_started
    np.random.set_state(rng)
    done=steps=changes=warnings=plans=0
    team_return=0.
    nearest=original_study.minimum_distance(env)
    start_time=float(env.Current_T)
    times=[]
    planned_times=[]
    regular_times=[]
    position_errors=[]
    thrust_projection=[]
    candidates=0
    max_violation=0.
    digest=hashlib.sha256()
    rows=[]
    while not done:
        measurement=observe(env,observations)
        started=time.perf_counter()
        command,info=rule.control(measurement)
        seconds=time.perf_counter()-started
        times.append(seconds)
        (planned_times if info['planned'] else regular_times).append(seconds)
        plans+=int(info['planned'])
        candidates+=info['candidate_policies']
        raw=_POLICY(observations)
        reference=source_mixture(raw,env.boids_actions)
        projection=project_thrust(command,raw,env.boids_actions)
        thrust_projection.append(float(np.linalg.norm(command-projection,axis=-1).max()))
        changes+=int(np.max(np.abs(command-reference))>1e-3)
        warnings+=int(info['warning'])
        max_violation=max(max_violation,info['cbf_constraint_violation'])
        passed=env.thrust_to_action(command)
        observations,reward,done,_=env.step(passed,'RL')
        if not np.isfinite(observations).all() or not np.isfinite(reward).all():
            raise FloatingPointError('Non-finite actual feedback rollout.')
        distance=original_study.minimum_distance(env)
        nearest=min(nearest,distance)
        team_return+=float(np.mean(reward))
        steps+=1
        predicted=info['predicted_next_state']
        if predicted is not None:
            position_errors.append(float(np.sqrt(np.mean(np.sum((snapshot(env)[:,:2]-predicted[:,:2])**2,axis=-1)))))
        original_study.digest_transition(digest,passed,env,observations,reward,done)
        if trace:
            rows.append(dict(time=float(env.Current_T),positions=snapshot(env).tolist(),
                              attacker=env.attacker.pos.tolist(),command=command.tolist(),
                              template=info['template'],planned=info['planned'],
                              minimum_distance=distance))
        if steps>401:
            raise RuntimeError('Source horizon exceeded.')
    result=dict(**scene,method=method,block_steps=block_steps,success=int(done>2),collision=int(done==2),
        source_loss=int(done==1),outcome=int(done),capture=int(done==3),timeout=int(done==4),
        target_breach=int(np.linalg.norm(env.attacker.pos)<env.Target_R),steps=steps,
        duration=float(env.Current_T-start_time),team_return=team_return,minimum_distance=nearest,
        altered_steps=changes,warning_steps=warnings,plans=plans,candidate_policies=candidates,
        policy_seconds_mean=float(np.mean(times)),policy_seconds_median=float(np.median(times)),
        policy_seconds_p95=float(np.quantile(times,.95)),policy_seconds_p99=float(np.quantile(times,.99)),
        policy_seconds_max=float(np.max(times)),deadline_misses=sum(t>.2 for t in times),
        planning_seconds_mean=float(np.mean(planned_times)),
        feedback_seconds_mean=float(np.mean(regular_times)) if regular_times else 0.,
        warmup_seconds=warm_seconds,max_cbf_constraint_violation=max_violation,
        prediction_position_rmse_mean=float(np.mean(position_errors)) if position_errors else None,
        max_projection_residual=float(max(thrust_projection)),
        out_of_segment_steps=sum(x>1e-3 for x in thrust_projection),trajectory_sha256=digest.hexdigest())
    return (result,rows) if trace else result


def scene_job(job):
    scene,methods,block_steps=job
    order=np.random.default_rng(scene['scene_seed']+194).permutation(len(methods))
    results=[]
    for i in order:
        method=methods[i]
        if method in BASELINES:
            result=original_study.rollout(scene,method)
            if method=='original':
                result['original_equivalent']=(result['trajectory_sha256']==original_study.direct_original(scene,_POLICY))
                if not result['original_equivalent']:
                    raise AssertionError('Untouched original source changed.')
        else:
            result=new_rollout(scene,method,block_steps)
        results.append(result)
    return results


def recovery_job(job):
    case,block_steps=job
    scene={k:case['meta'][k] for k in ('cell','scene_seed','noise_seed','defenders','agility')}
    result=[]
    for repeat in range(4):
        seed=None if repeat==0 else stable_number(f"recovery-{scene['scene_seed']}-{repeat-1}")%(2**32)
        for method in ADDITIONS:
            row=new_rollout(scene,method,block_steps,case=case,future_seed=seed)
            result.append({**case['meta'],**row,'future':'recorded' if repeat==0 else 'unseen',
                           'repeat':repeat,'future_noise_seed':seed})
    return result


def prediction_job(job):
    case,block_steps=job
    env,obs=restore(case['state'])
    past=np.array([[b.left_thrust,b.right_thrust] for b in env.defender_list])
    actual=new_controller(env.defender_num,'feedback_joint',block_steps)
    held=new_controller(env.defender_num,'held_joint',block_steps)
    actual.reset(past)
    held.reset(past)
    measurement=observe(env,obs)
    command,info=actual.control(measurement)
    prediction=actual.plan
    fixed=held.predict(measurement,actual.template)
    actual_states=[]
    done=0
    for step in range(block_steps):
        if step:
            command,_=actual.control(observe(env,obs))
        obs,_,done,_=env.step(env.thrust_to_action(command),'RL')
        actual_states.append(snapshot(env))
        if done:
            break
    common=min(len(actual_states),len(prediction['positions'])-1,len(fixed['positions'])-1)
    truth=np.asarray(actual_states[:common])
    rows=[]
    for mode,pred in [('feedback',prediction),('held',fixed)]:
        values=pred['positions'][1:common+1]
        i,j=np.triu_indices(env.defender_num,1)
        actual_d=float(np.linalg.norm(truth[:,i,:2]-truth[:,j,:2],axis=-1).min())
        predicted_d=float(np.linalg.norm(values[:,i,:2]-values[:,j,:2],axis=-1).min())
        rows.append(dict(**case['meta'],prediction=mode,block_steps=block_steps,observed_steps=common,
                         position_rmse=float(np.sqrt(np.mean(np.sum((truth[:,:,:2]-values[:,:,:2])**2,axis=-1)))),
                         dmin_error=predicted_d-actual_d,actual_dmin=actual_d,predicted_dmin=predicted_d,
                         actual_outcome=int(done),template=actual.template))
    return rows


def run_parallel(stage,jobs,workers):
    function={'scenes':scene_job,'recovery':recovery_job,'prediction':prediction_job}[stage]
    results=[]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(str(CHECKPOINT),)) as pool:
        pending={pool.submit(function,j):i for i,j in enumerate(jobs)}
        for task in as_completed(pending):
            results.append((pending[task],task.result()))
            print(f'[{stage.upper()}] {len(results)}/{len(jobs)}; {time.perf_counter()-started:.1f}s',flush=True)
    return [row for _,group in sorted(results) for row in group]


def code_hashes():
    names=('feedback_joint_control.py','evaluate_feedback_joint.py','gate_study_control.py',
           'source_arboids.py','cbf_source_baseline.py','evaluate_gate_contribution.py','gate_mechanism_diagnostics.py')
    return {name:sha256(Path(__file__).with_name(name)) for name in names}


def development_scenes():
    groups=defaultdict(list)
    for row in csv.DictReader((OLD_ROOT/'generalization/episodes.csv').open(encoding='utf-8')):
        if row['method']=='original':
            scene=dict(cell=row['cell'],scene_seed=int(row['scene_seed']),noise_seed=int(row['noise_seed']),
                       defenders=int(row['defenders']),agility=float(row['agility']))
            groups[row['cell']].append(scene)
    return [r for cell in sorted(groups) for r in sorted(groups[cell],
             key=lambda s:stable_number(f"feedback-repair-development-{s['scene_seed']}"))[:4]]


def confirmation_scenes():
    scenes=[]
    for i in range(128):
        seed=151000000+i
        scenes.append(dict(cell='main-n3-a2.25',scene_seed=seed,noise_seed=seed+1000000,defenders=3,agility=2.25))
    for i,(n,agility) in enumerate((n,a) for n in (2,3,4,5) for a in (1.5,2.,2.5,3.)):
        for repeat in range(16):
            seed=153000000+i*10000+repeat
            scenes.append(dict(cell=f'n{n}-a{agility:g}',scene_seed=seed,noise_seed=seed+1000000,
                               defenders=n,agility=agility))
    return scenes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('train/experiments/feedback-joint-20261006'))
    parser.add_argument('--stage',choices=['develop','confirm','recover','prediction'],required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    old_spec=json.loads((OLD_ROOT/'mechanism/specification.json').read_text(encoding='utf-8'))
    for name,expected in old_spec['code'].items():
        if sha256(Path(__file__).with_name(name))!=expected:
            raise RuntimeError('Original comparator changed: '+name)
    if sha256(CHECKPOINT)!=old_spec['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed.')
    code=code_hashes()
    source=verify_source()
    spec=dict(source=source,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha256(CHECKPOINT),code=code,
              protocol_sha256=sha256(Path('docs/feedback-joint-control.md')),
              source_data_sha256=sha256(OLD_ROOT/'generalization/episodes.csv'),
              new_scene_seeds=[s['scene_seed'] for s in confirmation_scenes()],
              new_scene_count=384,methods=list(METHODS),development_block_steps=[5,10])
    path=args.output_dir/'specification.json'
    frozen_path=args.output_dir/'frozen_method.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf-8'))
        if previous!=spec:
            raise RuntimeError('Frozen run inputs changed.')
    else:
        path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    started=time.perf_counter()
    if args.stage=='develop':
        scenes=development_scenes()
        jobs=[(s,('feedback_joint',),h) for h in (5,10) for s in scenes]
        rows=run_parallel('scenes',jobs,args.workers)
        original_study.write_csv(args.output_dir/'development.csv',rows)
        candidates=[]
        for h in (5,10):
            group=[r for r in rows if r['block_steps']==h]
            candidates.append(dict(block_steps=h,successes=sum(r['success'] for r in group),
                collisions=sum(r['collision'] for r in group),episodes=len(group),
                mean_control_seconds=float(np.mean([r['policy_seconds_mean'] for r in group]))))
        selected=min(candidates,key=lambda r:(-r['successes'],r['collisions'],r['mean_control_seconds']))
        frozen=dict(selected=selected,candidates=candidates,code=code,selection_rule='successes, collisions, runtime',
                    final_method='feedback_joint',checkpoint_sha256=spec['checkpoint_sha256'])
        frozen_path.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
        print('[FROZEN] '+json.dumps(frozen['selected']),flush=True)
    else:
        frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
        if frozen['code']!=code:
            raise RuntimeError('Selected method implementation changed.')
        h=frozen['selected']['block_steps']
        if args.stage=='confirm':
            rows=run_parallel('scenes',[(s,METHODS,h) for s in confirmation_scenes()],args.workers)
            original_study.write_csv(args.output_dir/'confirmation.csv',rows)
        else:
            with (DIAGNOSTIC_ROOT/'states.pkl').open('rb') as file:
                cases=pickle.load(file)
            if args.stage=='recover':
                jobs=[(c,h) for c in cases if c['meta']['category']=='failure']
                rows=run_parallel('recovery',jobs,args.workers)
                original_study.write_csv(args.output_dir/'recovery.csv',rows)
            else:
                rows=run_parallel('prediction',[(c,h) for c in cases],args.workers)
                original_study.write_csv(args.output_dir/'prediction.csv',rows)
    unchanged=code_hashes()==code and verify_source()==source and sha256(CHECKPOINT)==spec['checkpoint_sha256']
    summary=dict(passed=bool(unchanged),stage=args.stage,rows=len(rows),
                  code_source_checkpoint_unchanged=bool(unchanged),elapsed_seconds=time.perf_counter()-started)
    (args.output_dir/(args.stage+'_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    if not unchanged:
        raise SystemExit(1)


if __name__=='__main__':
    main()
