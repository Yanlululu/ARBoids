"""Run the existing six-defender experiments in independent scene shards."""
import os

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import develop_source_fusion as d

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / 'train/experiments/six-defender-paper-method-20261014'
SOURCE = REPO / 'train/experiments/main-seed42-20260921/adares1.pth'
REFERENCE = RUNS / 'six-trained-interaction-185/remote-paired-futures.json'
SCENES = d.bank(4151000000, 32, ((6, 2.), (6, 2.5)))


def initialize():
    torch.set_num_threads(1)
    d.initialize(str(SOURCE))


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def shortlist(data):
    scenes, programs, selection = [], [], []
    for roll in data['rollouts']:
        row = roll['row']
        samples = {}
        for branch in roll['branches']:
            if branch['observed_future']:
                continue
            reference = branch['factual_outcome']
            repetition = branch['future_repetition']
            for side, name in (('left', 'before'), ('right', 'after')):
                gates = tuple(float(x) for x in branch[side][:, 3])
                if all(g < 0 for g in gates):
                    continue
                value = branch[name]
                samples.setdefault(gates, {})[repetition] = (
                    value['success'] - reference['success'],
                    value['success'] * np.exp(-value['duration'] / 20.)
                    - reference['success'] * np.exp(-reference['duration'] / 20.))
        ranked = sorted([(float(np.mean([x[0] for x in values.values()])),
                          float(np.mean([x[1] for x in values.values()])), gates)
                         for gates, values in samples.items()], reverse=True)
        first = ranked[0]
        subject = next(i for i, g in enumerate(first[2]) if g >= 0)
        second = next((x for x in ranked[1:] if x[2][subject] < 0), ranked[1])
        programs.append([[-1.] * row['defenders'], list(first[2]), list(second[2])])
        scenes.append((row['seed'], row['defenders'], row['agility']))
        selection.append(dict(seed=row['seed'], ranked=[first, second]))
    return scenes, programs, selection


def collect_shard(task):
    start, scenes, output, repetitions, future_start, programs, device = task
    d.collect_initial_roles_batched(start, scenes, output, repetitions=repetitions,
        future_start=future_start, programs=programs, batch_size=128 if device.startswith('cuda') else 64, device=device)
    return str(Path(output) / 'training-data.pth')


def collect(start, scenes, output, repetitions, future_start, programs, device, workers=8):
    output.mkdir(parents=True, exist_ok=True)
    master = output / 'training-data.pth'
    previous = torch.load(master, map_location='cpu', weights_only=False) if master.exists() else None
    if previous and previous.get('completed'):
        return previous
    tasks = []
    for rank in range(workers):
        indices = list(range(rank, len(scenes), workers))
        part = output / 'shards' / str(rank)
        part.mkdir(parents=True, exist_ok=True)
        shard_file = part / 'training-data.pth'
        if previous and not shard_file.exists():
            mapping = {old: new for new, old in enumerate(indices)}
            signature = previous['sampling_signature']
            shard = dict(previous,
                sampling_signature=(signature[0], signature[1], tuple(signature[2][i] for i in indices)),
                sampling_jobs=[(mapping[i], repetition, arm, future)
                    for i, repetition, arm, future in previous['sampling_jobs'] if i in mapping],
                partial_results={(mapping[i], repetition, arm): value
                    for (i, repetition, arm), value in previous['partial_results'].items() if i in mapping},
                factual_steps={mapping[i]: value for i, value in previous['factual_steps'].items() if i in mapping},
                factual_rows={mapping[i]: value for i, value in previous['factual_rows'].items() if i in mapping})
            atomic_save(shard, shard_file)
        tasks.append((str(start), [scenes[i] for i in indices], str(part), repetitions,
                      future_start, None if programs is None else [programs[i] for i in indices], device))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        paths = list(pool.map(collect_shard, tasks))
    rollouts = []
    for path in paths:
        shard = torch.load(path, map_location='cpu', weights_only=False)
        if not shard.get('completed'):
            raise RuntimeError('A scene shard did not finish.')
        rollouts.extend(shard['rollouts'])
    order = {scene[0]: i for i, scene in enumerate(scenes)}
    rollouts.sort(key=lambda r: order[r['row']['seed']])
    if len(rollouts) != len(scenes) or len({r['row']['seed'] for r in rollouts}) != len(scenes):
        raise RuntimeError('Missing or repeated scene evidence.')
    model = torch.load(start, map_location='cpu', weights_only=False)
    data = dict(model=model, rollouts=rollouts, simulation_steps=sum(r['simulation_steps'] for r in rollouts),
        source_sha256=model.get('source_sha256'), completed=True, future_start=future_start,
        use='Same paired continuations, distributed by complete scene; no sampling protocol changes.')
    atomic_save(data, master)
    d.save_json(output / 'development.json', dict(collection=d.summarize([r['row'] for r in rollouts]),
        rows=[r['row'] for r in rollouts], simulation_steps=data['simulation_steps'], repetitions=repetitions,
        future_start=future_start, inference_device=device, scene_shards=workers))
    return data


def compare(left, right):
    if [(r['seed'], r.get('future_seed')) for r in left] != [(r['seed'], r.get('future_seed')) for r in right]:
        raise ValueError('Paired comparisons require the same scenes and future noise.')
    x = np.array([r['success'] for r in left]).reshape(4, 64)
    y = np.array([r['success'] for r in right]).reshape(4, 64)
    delta = (x - y).mean(0)
    rng = np.random.default_rng(20261015)
    interval = np.quantile(delta[rng.integers(64, size=(20000, 64))].mean(1), [.025, .975])
    return dict(gain=float(delta.mean()), rescues=int(((x == 1) & (y == 0)).sum()),
        regressions=int(((x == 0) & (y == 1)).sum()), cluster_bootstrap_interval=interval.tolist(),
        per_noise_gain=(x - y).mean(1).tolist())


def command(arguments):
    subprocess.run([sys.executable, '-X', 'utf8', '-u', str(REPO / 'train/develop_source_fusion.py'),
                    *arguments, '--source', str(SOURCE)], cwd=REPO, check=True)


@torch.no_grad()
def batched_primitive_program(actor, action):
    """Resolve a single value stage's delayed prefix to native tick control."""
    if action.shape[-1] != 14:
        return action, 0
    delay = int(action[0, 8]); depth = int(action[0, 13])
    if delay > 0 and depth >= 0:
        value = actor.successor_task_value
        if depth == 1:
            if value is None or value.prior is not None:
                raise ValueError('Batched delayed evaluation requires one value stage over a native predecessor.')
        elif depth != 0 or value is not None:
            raise ValueError('Batched delayed evaluation cannot collapse a value-stage chain.')
        if any(getattr(actor, key, None) is not None for key in (
                'cooperative_role_credit', 'finite_temporal_credit', 'continuous_program_value',
                'joint_intervention_value')) or len(getattr(actor, 'joint_intervention_values', [])):
            raise ValueError('The delayed predecessor must use native per-tick decisions.')
    return action[:, :13].copy(), max(delay, 0) if depth >= 0 else 0


def batched_credit_rows(policy,tasks,batch_size=32,device='cuda:0',keep_steps=False,sample_policy=False,initial_jobs=None,
                        response_tick=None,actor_override=None,collect_fleet=False,root_sample_seed=None):
    """Batch deterministic policy inference, retaining each episode's RNG."""
    from envs.TADgame import TADEnv
    initialize();torch.backends.cuda.matmul.allow_tf32=False
    actor=(d.joint_load(str(policy),Path(policy).stat().st_mtime_ns) if actor_override is None else actor_override).to(device)
    fleet=getattr(actor,'feedback_fleet_policy',None)
    if collect_fleet and (fleet is None or not fleet.sample or not fleet.collect):
        raise ValueError('Fleet training requires a sampling whole-fleet policy and decision collection.')
    if root_sample_seed is not None and (initial_jobs is not None or sample_policy or collect_fleet
                                        or actor.successor_task_value is not None):
        raise ValueError('Decision-root sampling needs complete deterministic factual episodes without M2.')
    if actor.gate_period!=1 or actor.intent_strength:
        raise ValueError('Batched credit evaluation requires per-step soft-preference control.')
    results=[]
    for begin in range(0,len(tasks),batch_size):
        current=tasks[begin:begin+batch_size];environments=[];observations=[];random_states=[]
        for local,(scene,future) in enumerate(current):
            seed,n,agility=scene;np.random.seed(seed)
            if initial_jobs is None:
                env=TADEnv(n,protocol=actor.environment_protocol);obs,_=env.reset(agility=agility,noisy_agility=False)
            else:
                job=initial_jobs[begin+local]
                env=copy.deepcopy(job['environment']);obs=job['observation'].copy()
                np.random.set_state(job['random_state'])
            if future is not None:np.random.seed(future)
            environments.append(env);observations.append(obs);random_states.append(np.random.get_state())
        active=list(range(len(current)));options=[None]*len(current)
        remaining=np.zeros(len(current),int);ticks=np.zeros(len(current),int)
        midpoint_checked=np.zeros(len(current),bool)
        intervention_pending=np.zeros(len(current),bool)
        intervention_activation=np.zeros(len(current),int)
        if initial_jobs is not None:
            for i,job in enumerate(initial_jobs[begin:begin+len(current)]):
                state=job.get('controller_state')
                if state is None:continue
                # A suffix intervention changes the remaining program, not
                # the predecessor's observation clock. Restore its temporal
                # state so the paired factual branch is the actual policy.
                action=job['action'].copy();elapsed=int(state['ticks'])
                if action.shape[-1]==6:
                    action=np.concatenate((action,np.zeros_like(action[:,:2]),
                        np.full_like(action[:,:1],d.joint_phase_boundary(action))),-1)
                action[:,8]+=elapsed;remaining[i]=int(action[0,4])
                action[:,4]+=elapsed;options[i]=action;ticks[i]=elapsed
                midpoint_checked[i]=state['midpoint_checked']
        returns=np.zeros(len(current));task_returns=np.zeros(len(current));rows=[None]*len(current)
        intervention_events=[[] for _ in current]
        decision_records=[[] for _ in current] if keep_steps else None
        trajectories=[[] for _ in current] if keep_steps else None
        fleet_trajectories=[[] for _ in current] if collect_fleet else None
        sampled_roots=[None]*len(current) if root_sample_seed is not None else None
        root_counts=np.zeros(len(current),int)
        root_rngs=([np.random.default_rng(np.random.SeedSequence(
            [int(root_sample_seed),int(scene[0]),0 if future is None else int(future),39797]))
            for scene,future in current] if sampled_roots is not None else None)
        responses=[None]*len(current) if response_tick is not None else None
        response_ticks = (None if response_tick is None else
            [int(response_tick)]*len(current) if np.isscalar(response_tick) else
            response_tick[begin:begin+len(current)])
        def packets(indices):
            records=[d.joint_packet(environments[i],observations[i]) for i in indices]
            for record in records:record.setdefault('_conditional_count',np.zeros(1,np.float32))
            return {key:torch.as_tensor(np.stack([record[key] for record in records]),device=device) for key in records[0]}
        for step in range(400):
            if not active:break
            if sampled_roots is not None and step%5==0:
                for i in active:
                    if remaining[i]:continue
                    root_counts[i]+=1
                    if root_rngs[i].integers(root_counts[i])==0:
                        sampled_roots[i]=dict(environment=copy.deepcopy(environments[i]),
                            observation=observations[i].copy(),random_state=copy.deepcopy(random_states[i]),
                            root_steps=step)
            controls={};fresh=[i for i in active if remaining[i]==0]
            if fresh:
                decisions=(np.stack([initial_jobs[begin+i]['action'] for i in fresh])
                    if initial_jobs is not None and step==0 else
                    actor(packets(fresh),deterministic=not sample_policy)[0].cpu().numpy())
                for i,action in zip(fresh,decisions):
                    if keep_steps:decision_records[i].append(dict(step=step,action=action.copy()))
                    value_program=action.shape[-1]==14
                    action,activation=batched_primitive_program(actor,action)
                    controls[i]=action[:,:3]
                    intervene=(action.shape[-1]>3 and ((action[:,3]>=0.).any()
                        or (action.shape[-1]>5 and (action[:,5]>=0.).any())))
                    intervene=intervene or (action.shape[-1]==13 and (action[:,9:11]>=0.).any())
                    if value_program:intervene=intervene or (action[:,9:13]>=0.).any()
                    waiting=(actor.cooperative_role_credit is not None and actor.cooperative_role_credit.wait_span)
                    waiting=waiting or (action.shape[-1]>5 and (action[:,3]<0.).all() and (action[:,5]==-2.).all())
                    waiting=waiting or action.shape[-1]==9
                    if action.shape[-1]>3 and (intervene or waiting):
                        options[i]=action;remaining[i]=int(action[0,4]);ticks[i]=0
                        midpoint_checked[i]=False
                        intervention_pending[i]=intervene
                        intervention_activation[i]=activation
            for i in active:
                if intervention_pending[i] and ticks[i]>=intervention_activation[i]:
                    action=options[i]
                    preferences=(np.where((ticks[i]>=action[:,8])[:,None],action[:,11:13],action[:,9:11])
                        if action.shape[-1]==13 else action[:,3])
                    selected=(preferences>=0.).any(-1) if preferences.ndim==2 else preferences>=0.
                    environments[i]._conditional_option_count=getattr(environments[i],'_conditional_option_count',0)+1
                    intervention_events[i].append(dict(time=float(environments[i].Current_T),
                        duration_steps=int(remaining[i]),subjects=np.flatnonzero(selected).tolist(),
                        coefficients=preferences[selected].tolist()))
                    intervention_pending[i]=False
            committed=[i for i in active if remaining[i]>0]
            if committed:
                batch=packets(committed)
                fixed=torch.as_tensor(np.stack([d.joint_program_gates(options[i],ticks[i])
                    for i in committed]),device=device)
                if actor.preference_strength:batch['_option_preference']=fixed
                elif actor.conditional_commitment:
                    planning=fixed.clone()
                    if not actor.immediate_commitment:
                        planning[torch.tensor([ticks[i]==0 for i in committed],device=device)]=-1.
                    batch['_committed_gates']=planning
                if not actor.preference_strength:batch['_held_gates']=fixed
                if any(options[i].shape[-1]>6 for i in committed):
                    batch['_proposal_offset']=torch.as_tensor(np.stack([
                        options[i][:,6:8] if options[i].shape[-1]>6 else np.zeros_like(options[i][:,:2])
                        for i in committed]),device=device)
                decisions=actor(batch,deterministic=True,gate_only=True)[0].cpu().numpy()
                if actor.midpoint_intervention_values is not None and not actor.preference_strength:
                    local=[j for j,i in enumerate(committed) if not midpoint_checked[i]
                        and not (step==0 and initial_jobs is not None
                            and initial_jobs[begin+i].get('controller_state') is not None)
                        and options[i].shape[-1] in (6,9)
                        and d.joint_refinement_due(options[i][0,4],ticks[i],actor.option_refinement_fraction,
                            actor.option_refinement_steps,actor.option_refinement_checkpoints)
                        and ticks[i]>0 and (options[i][:,5]>=0.).any()
                        and (actor.phase_preserving_refinement or actor.option_refinement_fraction==.5
                             or np.array_equal(options[i][:,3],options[i][:,5]))]
                    if local:
                        selected=[committed[j] for j in local]
                        suffix=np.stack([d.joint_remaining_program(decisions[j],options[i],ticks[i],
                            actor.phase_preserving_refinement)
                            for j,i in zip(local,selected)])
                        revised=actor.midpoint_intervention_values(packets(selected),
                            torch.as_tensor(suffix,device=device)).cpu().numpy()
                        changed=[]
                        for j,i,prior,action in zip(local,selected,suffix,revised):
                            midpoint_checked[i]=not actor.option_refinement_checkpoints
                            if not np.array_equal(action[:,3:],prior[:,3:]):
                                midpoint_checked[i]=True
                                options[i]=action;ticks[i]=0
                                fixed[j]=torch.as_tensor(action[:,3],device=device)
                                changed.append(j)
                        if changed and actor.immediate_commitment:
                            updated=packets([committed[j] for j in changed])
                            updated['_committed_gates']=fixed[changed]
                            updated['_held_gates']=fixed[changed]
                            decisions[changed]=actor(updated,deterministic=True,gate_only=True)[0].cpu().numpy()
                if not actor.preference_strength:
                    commanded=fixed.cpu().numpy()
                    decisions[:,:,2]=np.where(commanded>=0.,commanded,decisions[:,:,2])
                for i,action in zip(committed,decisions):controls[i]=action[:,:3]
            following=[]
            actuator_preferences=any(remaining[i]>0 and options[i].shape[-1]==13 for i in active)
            source_controls=(np.stack([controls[i] for i in active])
                if getattr(actor,'actuator_fusion_refinement',None) or actuator_preferences else None)
            if actor.response_feedback_value is not None:
                physical=actor.response_feedback_value(packets(active),torch.as_tensor(
                    np.stack([controls[i] for i in active]),device=device)).cpu().numpy()
                for i,control in zip(active,physical):controls[i]=control
            if actuator_preferences:
                from conditional_fusion_value import apply_actuator_preferences
                preferences=np.stack([np.where((ticks[i]>=options[i][:,8])[:,None],
                    options[i][:,11:13],options[i][:,9:11]) if remaining[i]>0 and options[i].shape[-1]==13
                    else np.full((len(controls[i]),2),-1.,dtype=np.float32) for i in active])
                physical=apply_actuator_preferences(packets(active),torch.as_tensor(source_controls,device=device),
                    torch.as_tensor(np.stack([controls[i] for i in active]),device=device),
                    torch.as_tensor(preferences,device=device)).cpu().numpy()
                for i,control in zip(active,physical):controls[i]=control
            if getattr(actor,'actuator_fusion_refinement',None):
                from conditional_fusion_value import refine_actuator_interactions
                refinement_packet=packets(active)
                held_preferences=np.stack([
                    (np.where((ticks[i]>=options[i][:,8])[:,None],options[i][:,11:13],options[i][:,9:11])
                     if options[i].shape[-1]==13 else
                     np.repeat(d.joint_program_gates(options[i],ticks[i])[:,None],2,axis=-1))
                    if remaining[i]>0 else np.full((len(controls[i]),2),-1.,dtype=np.float32)
                    for i in active])
                refinement_packet['_committed_actuator_preferences']=torch.as_tensor(held_preferences,device=device)
                if actor.response_feedback_value is None:
                    native=np.broadcast_to(source_controls[:,:,2:3],held_preferences.shape)
                    coefficients=(np.where(preferences>=0.,preferences,native)
                        if actuator_preferences else native)
                    refinement_packet['_preferred_actuator_coefficients']=torch.as_tensor(
                        np.array(coefficients,copy=True),device=device)
                physical=refine_actuator_interactions(actor,refinement_packet,
                    torch.as_tensor(source_controls,device=device),
                    torch.as_tensor(np.stack([controls[i] for i in active]),device=device),
                    **actor.actuator_fusion_refinement).cpu().numpy()
                for i,control in zip(active,physical):controls[i]=control
                if collect_fleet:
                    decision=fleet.last_decision
                    for j,i in enumerate(active):
                        fleet_trajectories[i].append({key:value[j].clone() for key,value in decision.items()})
            for i in active:
                env=environments[i];np.random.set_state(random_states[i])
                before=d.joint_packet(env,observations[i]) if keep_steps else None
                obs,reward,done,sr=d.joint_task_step(env,controls[i],actor.training_gamma,actor.training_step_cost)
                if collect_fleet:
                    fleet_trajectories[i][-1].update(reward=float(reward),done=bool(done),
                        discount=float(actor.training_gamma))
                if responses is not None and responses[i] is None and (done or step+1==response_ticks[i]):
                    responses[i]=dict(packet=d.joint_packet(env,obs),terminal=int(done),steps=step+1)
                if keep_steps:
                    trajectories[i].append(dict(packet=before,action=controls[i].copy(),done=done,physical_steps=1,
                        reward=reward,source_reward=sr,discount=actor.training_gamma,
                        controller_state=None if remaining[i]<=0 else dict(program=options[i].copy(),
                            ticks=int(ticks[i]),remaining=int(remaining[i]),
                            midpoint_checked=bool(midpoint_checked[i]))))
                random_states[i]=np.random.get_state();observations[i]=obs;returns[i]+=(.99**step)*sr
                task_returns[i]+=(actor.training_gamma**step)*reward
                if remaining[i]>0:remaining[i]-=1;ticks[i]+=1
                if done:
                    scene,future=current[i];seed,n,agility=scene
                    row=dict(seed=int(seed),defenders=n,agility=agility,**d.outcome(env,done),discounted_return=float(returns[i]),
                        environment_protocol=actor.environment_protocol,
                        terminal=int(done),physical_steps=step+1,
                        task_return=float(task_returns[i]),task_discount=actor.training_gamma**(step+1),
                        conditional_interventions=int(getattr(env,'_conditional_option_count',0)),
                        interventions=intervention_events[i])
                    if future is not None:row['future_seed']=int(future)
                    if responses is not None:row['response']=responses[i]
                    rows[i]=row
                else:following.append(i)
            active=following
        if active:raise RuntimeError('Evaluation did not reach the unchanged task terminal.')
        if collect_fleet:
            results.extend(dict(row=row,fleet_steps=steps) for row,steps in zip(rows,fleet_trajectories))
        elif keep_steps:
            for row,steps,decisions in zip(rows,trajectories,decision_records):
                continuation=0.
                for record in reversed(steps):
                    continuation=record['source_reward']+.99*continuation
                    record['source_return']=continuation
                results.append(dict(row=row,steps=steps,decisions=decisions,branches=[],value_steps=[],simulation_steps=len(steps)))
        else:results.extend(rows)
        if sampled_roots is not None:
            for i,snapshot in enumerate(sampled_roots):
                snapshot['eligible_states']=int(root_counts[i])
                position=len(results)-len(current)+i
                if not keep_steps:results[position]=dict(row=results[position])
                results[position]['root_snapshot']=snapshot
        print(dict(phase='batched_credit_evaluation',model=str(policy),episodes=len(results),total=len(tasks)),flush=True)
    actor.cpu()
    return results


def fit_feedback_fleet_policy(predecessor,output,*,seed=42,iterations=24,episodes_per_iteration=32,
                              batch_size=32,width=48,epochs=4,minibatch=64,learning_rate=3e-4,
                              analytic_temperature=2.,entropy_weight=.01,device='cuda:0',
                              candidate_contrast=False,sampling_temperature=1.):
    """Optimize candidate selection on actual complete source episodes.

    SAC, Boids, dynamics and the task reward stay frozen. Each action is one
    complete feedback forecast from the shared fleet candidate library, or
    independent per-vessel reference choices for the matched learned control.
    Both use the same graph and receive real on-policy team-return credit.
    """
    from conditional_fusion_value import FeedbackFleetPolicy
    initialize();predecessor=Path(predecessor);output=Path(output)
    source_hash=hashlib.sha256(predecessor.read_bytes()).hexdigest()
    source=torch.load(predecessor,map_location='cpu',weights_only=False)
    config=source.get('actuator_fusion_refinement') or {}
    selection=config.get('selection')
    if not config.get('task_feedback_prediction') or selection not in ('joint','single_reference'):
        raise ValueError('Fleet learning requires a complete-feedback joint or independent predecessor.')
    if source.get('successor_task_value') is not None:
        raise ValueError('Mechanism-one learning must precede conditional intervention learning.')
    specification=dict(seed=seed,episodes_per_iteration=episodes_per_iteration,batch_size=batch_size,
        width=width,epochs=epochs,minibatch=minibatch,learning_rate=learning_rate,
        analytic_temperature=analytic_temperature,entropy_weight=entropy_weight,
        predecessor=str(predecessor),predecessor_sha256=source_hash,
        objective='on-policy clipped joint-fleet task-return policy gradient with a learned state baseline',
        candidate_budget='same 81 balanced fleet plans plus every single-vessel reference change',
        source_policies_frozen=True,training_task_base=4166500000,training_defenders=6)
    if candidate_contrast:
        specification['candidate_contrast']=True
        specification['objective']='within-state candidate-contrast clipped joint-fleet task-return policy gradient'
    if selection=='single_reference':
        specification['selection']=selection
        specification['objective']='factorized single-vessel reference policy gradient on the same complete-feedback queries and team returns'
    if sampling_temperature!=1.:
        specification['sampling_temperature']=sampling_temperature
    if config.get('independent_feedback_proposal'):
        specification['candidate_budget']+=' and its complete independent recombination'
    if config.get('nominal_feedback_proposals'):
        specification['candidate_budget']+=' plus jointly and independently proposed nominal fleet plans'
    if config.get('feedback_refinement_rounds'):
        specification['feedback_refinement_rounds']=config['feedback_refinement_rounds']
        specification['candidate_budget']+=' with common complete-feedback single-vessel neighborhood expansions'
    model=torch.load(output,map_location='cpu',weights_only=False) if output.exists() else copy.deepcopy(source)
    history=model.get('fleet_learning',{})
    if history and history['specification']!=specification:
        raise ValueError('Cannot resume fleet learning with changed data, budget or source parameters.')
    torch.manual_seed(seed);np.random.seed(seed)
    actor=copy.deepcopy(d.joint_load(str(predecessor),predecessor.stat().st_mtime_ns)).to(device)
    learner=FeedbackFleetPolicy(width,analytic_temperature,
        (model.get('feedback_fleet_policy') or {}).get('state'),candidate_contrast,
        sampling_temperature).to(device)
    actor.feedback_fleet_policy=learner
    optimizer=torch.optim.Adam(learner.parameters(),lr=learning_rate)
    if history:
        optimizer.load_state_dict(history['optimizer'])
        torch.set_rng_state(history['torch_rng'])
        if device.startswith('cuda'):torch.cuda.set_rng_state(history['cuda_rng'],device)
        np.random.set_state(history['numpy_rng'])
    completed=int(history.get('completed_iterations',0));learning=list(history.get('iterations',[]))
    for iteration in range(completed,iterations):
        started=time.monotonic();learner.sample=True;learner.collect=True;learner.eval()
        tasks=[(scene,None) for scene in d.bank(4166500000+seed*100000+iteration*episodes_per_iteration,
            max(1,(episodes_per_iteration+1)//2),((6,2.),(6,2.5)))][:episodes_per_iteration]
        with torch.no_grad():
            rollouts=batched_credit_rows(predecessor,tasks,batch_size=batch_size,device=device,
                actor_override=actor,collect_fleet=True)
        learner.to(device);learner.sample=False;learner.collect=False;learner.train()
        records=[];advantages=[];targets=[]
        for rollout in rollouts:
            following_value=0.;advantage=0.;steps=rollout['fleet_steps']
            local_advantages=[];local_targets=[]
            for record in reversed(steps):
                value=float(record['value']);discount=record['discount']*(not record['done'])
                delta=record['reward']+discount*following_value-value
                advantage=delta+discount*.95*advantage
                local_advantages.append(advantage);local_targets.append(advantage+value)
                following_value=value
            records.extend(steps);advantages.extend(reversed(local_advantages));targets.extend(reversed(local_targets))
        advantages=torch.tensor(advantages,dtype=torch.float32,device=device)
        advantages=(advantages-advantages.mean())/advantages.std().clamp_min(1e-6)
        targets=torch.tensor(targets,dtype=torch.float32,device=device)
        losses=[];divergences=[];stop=False;updates=0
        with torch.enable_grad():
            for epoch in range(epochs):
                for begin in range(0,len(records),minibatch):
                    if begin==0:permutation=torch.randperm(len(records)).tolist()
                    indices=permutation[begin:begin+minibatch]
                    selected=[records[index] for index in indices]
                    features=torch.stack([record['features'] for record in selected]).to(device,torch.float32)
                    costs=torch.stack([record['costs'] for record in selected]).to(device)
                    choices=torch.stack([record['index'] for record in selected]).to(device)
                    previous=torch.stack([record['logp'] for record in selected]).to(device)
                    logits,values=learner(features,costs)
                    distribution=learner.policy_distribution(logits,selection,features.shape[2],
                        temperature=learner.sampling_temperature)
                    log_ratio=distribution.log_prob(choices)-previous
                    ratio=log_ratio.exp();advantage=advantages[indices]
                    policy_loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                    value_loss=torch.nn.functional.smooth_l1_loss(values,targets[indices])
                    loss=policy_loss+.5*value_loss-entropy_weight*distribution.entropy().mean()
                    divergence=((ratio-1)-log_ratio).mean().detach()
                    if epoch>0 and float(divergence)>.03:
                        stop=True;break
                    optimizer.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(learner.parameters(),.5);optimizer.step();updates+=1
                    losses.append(float(loss.detach()));divergences.append(float(divergence))
                if stop:break
        record=dict(iteration=iteration+1,episodes=len(rollouts),physical_steps=len(records),updates=updates,
            success=sum(rollout['row']['success'] for rollout in rollouts),
            task_return=float(np.mean([rollout['row']['task_return'] for rollout in rollouts])),
            loss=float(np.mean(losses)),mean_kl=float(np.mean(divergences)),
            elapsed_seconds=time.monotonic()-started,tasks=tasks)
        learning.append(record)
        model['feedback_fleet_policy']=dict(width=width,analytic_temperature=analytic_temperature,
            state={key:value.detach().cpu().clone() for key,value in learner.state_dict().items()})
        if candidate_contrast:model['feedback_fleet_policy']['candidate_contrast']=True
        if sampling_temperature!=1.:
            model['feedback_fleet_policy']['sampling_temperature']=sampling_temperature
        model['fleet_learning']=dict(specification=specification,completed_iterations=iteration+1,
            iterations=learning,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
            numpy_rng=np.random.get_state(),total_training_episodes=sum(row['episodes'] for row in learning),
            total_physical_steps=sum(row['physical_steps'] for row in learning))
        atomic_save(model,output)
        print(dict(phase='feedback_fleet_learning',policy=str(output),seed=seed,
            **{key:value for key,value in record.items() if key!='tasks'}),flush=True)
        del records,rollouts,features,costs,logits,values,loss
        actor.to(device)
    actor.cpu();learner.eval();learner.sample=False;learner.collect=False
    return model


def collect_target_feedback_data(predecessor,factual_bank,output,*,seed=42,steps=12,
                                 branches=3,batch_size=64,device='cuda:0',record_sequences=False):
    """Observe short candidate continuations for public target-motion fitting."""
    initialize();d.numerical_source()
    predecessor=Path(predecessor);factual_bank=Path(factual_bank);output=Path(output)
    if steps<1 or branches<1 or batch_size<1:
        raise ValueError('Target-response collection needs positive continuation and batch sizes.')
    digest=hashlib.sha256(predecessor.read_bytes()).hexdigest()
    actor=d.joint_load(str(predecessor),predecessor.stat().st_mtime_ns).to(device)
    if actor.successor_task_value is not None:
        raise ValueError('Target-response fitting must use the mechanism-one predecessor.')
    bank=torch.load(factual_bank,map_location='cpu',weights_only=False)
    if not bank.get('completed') or bank['predecessor_sha256']!=digest:
        raise ValueError('Use completed factual states from this exact predecessor.')
    specification=dict(predecessor=str(predecessor),predecessor_sha256=digest,
        factual_bank=str(factual_bank),factual_bank_sha256=hashlib.sha256(factual_bank.read_bytes()).hexdigest(),
        seed=seed,steps=steps,branches=branches,batch_size=batch_size,inference_device=device,
        control_seconds=.2,absolute_frame=True,
        inputs='public defender poses, velocities and Boids commands; public target position and velocity',
        excluded_inputs='attacker agility, hidden water velocity, attacker controller and realized future',
        continuation='fixed actuator fusion coefficients with refreshed unchanged SAC and Boids commands')
    if record_sequences:specification['record_sequences']=True
    result=(torch.load(output,map_location='cpu',weights_only=False) if output.exists() else
        dict(specification=specification,next_task=0,nodes=[],global_state=[],targets=[],root_ids=[],
            physical_steps=0,completed=False))
    if record_sequences and not result['next_task']:
        result.update({key:[] for key in ('sequence_ids','sequence_ticks','public_motion','public_boids',
            'public_target','public_velocity','next_public_target','next_public_velocity')})
    if result['specification']!=specification:
        raise ValueError('Cannot resume public dynamics acquisition with changed states or sampling.')
    jobs=[(root,branch) for root in range(len(bank['records'])) for branch in range(branches)]
    if result['completed']:
        if result['next_task']!=len(jobs):raise ValueError('Completed dynamics data have an incomplete task prefix.')
        actor.cpu();return result
    start=int(result['next_task'])
    if start%batch_size or not 0<=start<len(jobs):
        raise ValueError('Target-response acquisition must resume at a complete native batch.')
    for begin in range(start,len(jobs),batch_size):
        current=jobs[begin:begin+batch_size];environments=[];observations=[];random_states=[];coefficients=[]
        for root,branch in current:
            snapshot=bank['records'][root]['root_snapshot']
            environments.append(copy.deepcopy(snapshot['environment']))
            observations.append(snapshot['observation'].copy())
            future=int(np.random.SeedSequence([seed,root,branch,1776]).generate_state(1)[0])
            random_states.append(np.random.RandomState(future).get_state())
            n=len(snapshot['observation'])
            codes=(np.full(n,4) if branch==0 else
                np.random.default_rng(np.random.SeedSequence([seed,root,branch,1777])).integers(9,size=n))
            coefficients.append(np.stack((codes//3,codes%3),-1).astype(np.float32)*.5)
        active=list(range(len(current)))
        for tick in range(steps):
            if not active:break
            packets=[d.joint_packet(environments[i],observations[i]) for i in active]
            packet={key:torch.as_tensor(np.stack([item[key] for item in packets]),device=device)
                for key in packets[0]}
            with torch.no_grad():
                nodes,global_state,target,velocity,axis=d.PublicTargetMotion.encode(packet,True)
                source=actor(packet,deterministic=True,gate_only=True)[0][...,:2]
                gates=torch.as_tensor(np.stack([coefficients[i] for i in active]),device=device)
                action=torch.cat((gates*source+(1.-gates)*packet['node'][...,11:13],
                    torch.ones_like(source[...,:1])),-1).cpu().numpy()
            following=[];after=[]
            for local,i in enumerate(active):
                np.random.set_state(random_states[i])
                obs,_,done,_=d.joint_task_step(environments[i],action[local],actor.training_gamma,actor.training_step_cost)
                random_states[i]=np.random.get_state();observations[i]=obs
                after.append(d.joint_packet(environments[i],obs))
                if not done:following.append(i)
            with torch.no_grad():
                next_packet={key:torch.as_tensor(np.stack([item[key] for item in after]),device=device)
                    for key in ('node','motion')}
                next_target,next_velocity=actor.public_target_motion(next_packet)
                def rotate(vector):
                    return torch.stack(((vector*axis).sum(-1),axis[:,0]*vector[:,1]-axis[:,1]*vector[:,0]),-1)
                labels=torch.cat((rotate(next_target-target-.2*velocity),rotate(next_velocity-velocity)),-1)
            result['nodes'].append(nodes.cpu().numpy());result['global_state'].append(global_state.cpu().numpy())
            result['targets'].append(labels.cpu().numpy())
            result['root_ids'].append(np.asarray([current[i][0] for i in active],np.int64))
            if record_sequences:
                result['sequence_ids'].append(np.asarray([begin+i for i in active],np.int64))
                result['sequence_ticks'].append(np.full(len(active),tick,np.int64))
                for key,value in (('public_motion',packet['motion']),('public_boids',packet['node'][...,11:13]),
                                  ('public_target',target),('public_velocity',velocity),
                                  ('next_public_target',next_target),('next_public_velocity',next_velocity)):
                    result[key].append(value.cpu().numpy())
            result['physical_steps']+=len(active);active=following
        result['next_task']=begin+len(current);result['completed']=result['next_task']==len(jobs)
        atomic_save(result,output)
        print(dict(phase='public_target_feedback_collection',seed=seed,tasks=result['next_task'],
            total=len(jobs),physical_steps=result['physical_steps']),flush=True)
    actor.cpu();return result


def fit_target_feedback_model(data,output,*,updates=6000,ensemble_size=3,width=48,seed=42,device='cuda:0'):
    """Fit measured one-step public responses, validating on separate roots."""
    if not data.get('completed') or updates<ensemble_size or updates%ensemble_size:
        raise ValueError('Target-response learning needs complete data and equal positive member budgets.')
    specification=data['specification'];predecessor=Path(specification['predecessor'])
    if hashlib.sha256(predecessor.read_bytes()).hexdigest()!=specification['predecessor_sha256']:
        raise ValueError('The target-response predecessor changed after observation collection.')
    roots=np.concatenate(data['root_ids']);unique=np.unique(roots)
    if len(unique)<5:raise ValueError('Public response validation requires at least five distinct factual roots.')
    shuffled=np.random.default_rng(seed).permutation(unique)
    held_roots=shuffled[:max(1,len(unique)//5)]
    held=np.isin(roots,held_roots);train_ids=np.flatnonzero(~held);held_ids=np.flatnonzero(held)
    nodes=torch.as_tensor(np.concatenate(data['nodes']),device=device)
    global_state=torch.as_tensor(np.concatenate(data['global_state']),device=device)
    target=torch.as_tensor(np.concatenate(data['targets']),device=device)
    if nodes.dtype!=torch.float32 or global_state.dtype!=torch.float32 or target.dtype!=torch.float32:
        raise ValueError('Public target-response observations must retain their native float32 representation.')
    def root_weights(indices):
        _,inverse,counts=np.unique(roots[indices],return_inverse=True,return_counts=True)
        weights=1./counts[inverse];return torch.as_tensor(weights/weights.sum(),device=device,dtype=torch.float32)
    probability=root_weights(train_ids);validation_weights=root_weights(held_ids)
    train_ids=torch.as_tensor(train_ids,device=device);held_ids=torch.as_tensor(held_ids,device=device)
    scales=target.new_tensor([.1,.1,.5,.5]);times=target.new_full((len(target),),.2)
    baseline=((target[held_ids]/scales).square().mean(-1)*validation_weights).sum().item()
    states=[];records=[];per_member=updates//ensemble_size
    for member in range(ensemble_size):
        torch.manual_seed(seed+1009*member)
        network=d.PublicTargetMotion(width,True).to(device)
        optimizer=torch.optim.Adam(network.parameters(),lr=1e-3)
        best_loss=baseline;best_step=0
        best_state={key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
        with torch.enable_grad():
            for step in range(per_member):
                selected=train_ids[torch.multinomial(probability,min(256,len(train_ids)),replacement=True)]
                prediction=torch.cat(network(nodes[selected],global_state[selected],times[selected]),-1)
                loss=((prediction-target[selected])/scales).square().mean()
                rate=1e-4+9e-4*.5*(1.+np.cos(np.pi*step/max(1,per_member-1)))
                for group in optimizer.param_groups:group['lr']=rate
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
                optimizer.step()
                if (step+1)%100==0 or step+1==per_member:
                    with torch.no_grad():
                        estimate=torch.cat(network(nodes[held_ids],global_state[held_ids],times[held_ids]),-1)
                        validation=(((estimate-target[held_ids])/scales).square().mean(-1)*validation_weights).sum().item()
                    if validation<best_loss:
                        best_loss=validation;best_step=step+1
                        best_state={key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
        states.append(best_state);records.append(dict(member=member,updates=per_member,
            selected_update=best_step,validation_normalized_mse=best_loss))
        print(dict(phase='public_target_feedback_fit',seed=seed,**records[-1],
            constant_velocity_normalized_mse=baseline),flush=True)
        del network,optimizer
    predictor=d.PublicTargetFeedback(width,states,True).to(device)
    with torch.no_grad():
        predictions=[torch.cat(network(nodes[held_ids],global_state[held_ids],times[held_ids]),-1)
            for network in predictor.models]
        predicted=torch.stack(predictions).mean(0)
        normalized=(((predicted-target[held_ids])/scales).square().mean(-1)*validation_weights).sum().item()
        coordinate_mse=((predicted-target[held_ids]).square()*validation_weights[:,None]).sum(0).cpu().tolist()
        baseline_mse=(target[held_ids].square()*validation_weights[:,None]).sum(0).cpu().tolist()
    model=copy.deepcopy(torch.load(predecessor,map_location='cpu',weights_only=False))
    model['target_feedback']=dict(width=width,states=states,absolute_frame=True,control_step=.2)
    model['target_feedback_fit']=dict(completed=True,seed=seed,updates=updates,ensemble_size=ensemble_size,
        width=width,parameters_per_member=sum(value.numel() for value in predictor.models[0].parameters()),
        observations=len(target),training_roots=int(len(unique)-len(held_roots)),validation_roots=held_roots.tolist(),
        acquisition=specification,physical_steps=data['physical_steps'],members=records,
        validation_normalized_mse=normalized,constant_velocity_normalized_mse=baseline,
        validation_coordinate_mse=coordinate_mse,constant_velocity_coordinate_mse=baseline_mse,
        validation_scope='development prediction validation split by actual factual root; no final evaluation tasks',
        candidate_conditioning='repeated public state updates inside each complete candidate fleet path')
    atomic_save(model,output)
    del predictor
    return model


def fit_target_feedback_rollout_model(data,output,*,initial_model=None,updates=6000,
                                     ensemble_size=3,width=48,seed=42,device='cuda:0',mixture=False):
    """Fit recursive public target dynamics on measured candidate continuations.

    Each rollout uses measured public fleet states as exogenous inputs. The
    target position and velocity are the model's own preceding predictions;
    later target observations enter the loss only. Deployment supplies the
    corresponding complete fleet forecasts instead of measured fleet states.
    """
    if (not data.get('completed') or not data['specification'].get('record_sequences')
            or ensemble_size<1 or updates<ensemble_size or updates%ensemble_size
            or (mixture and ensemble_size<2)):
        raise ValueError('Recursive response fitting needs complete sequences and equal positive member budgets.')
    specification=data['specification'];predecessor=Path(specification['predecessor'])
    if hashlib.sha256(predecessor.read_bytes()).hexdigest()!=specification['predecessor_sha256']:
        raise ValueError('The target-response predecessor changed after sequence acquisition.')
    sequence=np.concatenate(data['sequence_ids']);ticks=np.concatenate(data['sequence_ticks'])
    root_ids=np.concatenate(data['root_ids']);unique_sequences=np.unique(sequence)
    horizon=int(specification['steps']);count=len(unique_sequences)
    if not np.array_equal(unique_sequences,np.arange(data['next_task'])):
        raise ValueError('Every acquired continuation must retain its public sequence identity.')
    if len(np.unique(np.stack((sequence,ticks),-1),axis=0))!=len(sequence):
        raise ValueError('Public continuation sequences contain duplicate physical transitions.')
    arrays={}
    for key in ('public_motion','public_boids','public_target','public_velocity',
                'next_public_target','next_public_velocity'):
        observations=np.concatenate(data[key])
        if observations.dtype!=np.float32 or not np.isfinite(observations).all():
            raise ValueError('Public response sequences must retain finite native float32 measurements.')
        padded=np.zeros((count,horizon,*observations.shape[1:]),np.float32)
        padded[sequence,ticks]=observations
        arrays[key]=torch.as_tensor(padded,device=device)
    mask=np.zeros((count,horizon),bool);mask[sequence,ticks]=True
    if not np.array_equal(mask,np.arange(horizon)[None]<mask.sum(1)[:,None]):
        raise ValueError('Measured continuations must contain an uninterrupted physical prefix.')
    sequence_roots=np.asarray([np.unique(root_ids[sequence==i]).item() for i in unique_sequences])
    roots=np.unique(sequence_roots)
    if len(roots)<5:raise ValueError('Recursive prediction validation needs at least five factual roots.')
    held_roots=np.random.default_rng(seed).permutation(roots)[:max(1,len(roots)//5)]
    held=np.isin(sequence_roots,held_roots);training_ids=np.flatnonzero(~held);validation_ids=np.flatnonzero(held)
    def root_weights(ids):
        _,inverse,counts=np.unique(sequence_roots[ids],return_inverse=True,return_counts=True)
        weights=1./counts[inverse]
        return torch.as_tensor(weights/weights.sum(),device=device,dtype=torch.float32)
    probability=root_weights(training_ids);validation_weights=root_weights(validation_ids)
    training_ids=torch.as_tensor(training_ids,device=device);validation_ids=torch.as_tensor(validation_ids,device=device)
    mask=torch.as_tensor(mask,device=device)
    initial=None;initial_sha256=None
    if initial_model is not None:
        initial_model=Path(initial_model);initial=torch.load(initial_model,map_location='cpu',weights_only=False)
        initial_sha256=hashlib.sha256(initial_model.read_bytes()).hexdigest()
        fit=initial['target_feedback_fit'];recipe=initial['target_feedback']
        if (fit['acquisition']['predecessor_sha256']!=specification['predecessor_sha256']
                or set(fit['validation_roots'])!=set(held_roots.tolist())
                or recipe['width']!=width or len(recipe['states'])!=ensemble_size):
            raise ValueError('Warm target dynamics must preserve the predecessor, capacity and held-out root split.')
    def errors(predictor,indices):
        target=arrays['public_target'][indices,0];velocity=arrays['public_velocity'][indices,0]
        values=[]
        for tick in range(horizon):
            motion=arrays['public_motion'][indices,tick];boids=arrays['public_boids'][indices,tick]
            if predictor is None:
                following=target+.2*velocity;next_velocity=velocity
            elif isinstance(predictor,d.PublicTargetFeedback):
                following,next_velocity=predictor.predict_state(motion,boids,target,velocity)
            else:
                nodes,global_state,_,_,axis=d.PublicTargetMotion.encode_state(motion,boids,target,velocity,True)
                displacement,change=predictor(nodes,global_state,target.new_full((len(indices),),.2))
                def world(vector):
                    return torch.stack((axis[:,0]*vector[:,0]-axis[:,1]*vector[:,1],
                        axis[:,1]*vector[:,0]+axis[:,0]*vector[:,1]),-1)
                following=target+.2*velocity+world(displacement);next_velocity=velocity+world(change)
            values.append(torch.cat((following-arrays['next_public_target'][indices,tick],
                next_velocity-arrays['next_public_velocity'][indices,tick]),-1))
            active=mask[indices,tick,None]
            target=torch.where(active,following,target);velocity=torch.where(active,next_velocity,velocity)
        return torch.stack(values,1)
    scales=torch.ones((horizon,4),device=device,dtype=torch.float32)*.5
    scales[:,:2]=(.1+.05*torch.arange(1,horizon+1,device=device))[:,None]
    def costs(error,indices):
        return ((error/scales).square().mean(-1)*mask[indices]).sum(-1)/mask[indices].sum(-1)
    def validate(predictor,details=False):
        with torch.no_grad():
            residual=torch.cat([errors(predictor,ids) for ids in validation_ids.split(64)])
            loss=(costs(residual,validation_ids)*validation_weights).sum().item()
            mass=validation_weights[:,None]*mask[validation_ids]
            mse=(residual.square()*mass[:,:,None]).sum(0)/mass.sum(0).clamp_min(1e-8)[:,None]
        return (loss,mse.cpu().tolist()) if details else loss
    baseline,baseline_mse=validate(None,True)
    states=[];records=[];per_member=updates//ensemble_size
    mixture_record=None
    if mixture:
        # A single latent component persists throughout a continuation. Each
        # component sees the same public fleet inputs and recursively advances
        # its own target state, rather than averaging incompatible maneuvers.
        networks=torch.nn.ModuleList()
        for member in range(ensemble_size):
            torch.manual_seed(seed+1009*member)
            network=d.PublicTargetMotion(width,True).to(device)
            if initial is not None:network.load_state_dict(initial['target_feedback']['states'][member])
            networks.append(network)
        initial_losses=[validate(network) for network in networks]
        def mixture_terms(indices):
            residual=torch.stack([errors(network,indices) for network in networks])
            squared=(residual/scales).square().sum(-1)*mask[indices][None]
            log_density=-.5*squared.sum(-1)
            nll=-(torch.logsumexp(log_density,0)-np.log(ensemble_size))/(4*mask[indices].sum(-1))
            responsibility=log_density.softmax(0)
            return nll,responsibility
        def validate_mixture():
            with torch.no_grad():
                values=[mixture_terms(ids) for ids in validation_ids.split(32)]
                loss=(torch.cat([value[0] for value in values])*validation_weights).sum().item()
                mass=(torch.cat([value[1] for value in values],1)*validation_weights[None]).sum(-1)
            return loss,mass.cpu().tolist()
        initial_likelihood,_=validate_mixture();best_loss=initial_likelihood;best_step=0
        best_state=copy.deepcopy(networks.state_dict())
        optimizer=torch.optim.Adam(networks.parameters(),lr=3e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,per_member,eta_min=3e-5)
        with torch.enable_grad():
            for step in range(per_member):
                selected=training_ids[torch.multinomial(probability,32,replacement=True)]
                likelihood,responsibility=mixture_terms(selected)
                mass=responsibility.mean(-1).clamp_min(1e-8)
                loss=likelihood.mean()+.01*(mass*(mass*ensemble_size).log()).sum()
                if not torch.isfinite(loss):raise FloatingPointError('Target trajectory mixture produced a nonfinite loss.')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(networks.parameters(),5.);optimizer.step();schedule.step()
                if (step+1)%100==0 or step+1==per_member:
                    validation,_=validate_mixture()
                    if validation<best_loss:
                        best_loss=validation;best_step=step+1;best_state=copy.deepcopy(networks.state_dict())
                if (step+1)%500==0 or step+1==per_member:
                    print(dict(phase='public_target_mixture_fit',seed=seed,update=step+1,
                        member_updates=(step+1)*ensemble_size,loss=float(loss.detach()),
                        best_validation=best_loss,initial_validation=initial_likelihood),flush=True)
        networks.load_state_dict(best_state)
        _,component_mass=validate_mixture()
        mixture_record=dict(validation_normalized_nll=best_loss,initial_normalized_nll=initial_likelihood,
            selected_update=best_step,validation_component_responsibility=component_mass,
            likelihood='equal-prior mixture of complete recursive public target trajectories with fixed diagonal scales',
            position_scale_by_tick=scales[:,0].cpu().tolist(),velocity_scale=.5,
            component_balance_weight=.01,optimizer_steps=per_member,total_member_updates=updates)
        for member,network in enumerate(networks):
            states.append({key:value.detach().cpu().clone() for key,value in network.state_dict().items()})
            records.append(dict(member=member,updates=per_member,selected_update=best_step,
                initial_validation_normalized_mse=initial_losses[member],validation_normalized_mse=validate(network)))
        del networks,network,optimizer,schedule,best_state
    for member in range(0 if mixture else ensemble_size):
        torch.manual_seed(seed+1009*member)
        network=d.PublicTargetMotion(width,True).to(device)
        zero_state={key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
        if initial is not None:network.load_state_dict(initial['target_feedback']['states'][member])
        initial_loss=validate(network);best_loss=min(baseline,initial_loss);best_step=0
        best_state=({key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
                    if initial_loss<=baseline else zero_state)
        optimizer=torch.optim.Adam(network.parameters(),lr=3e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,per_member,eta_min=3e-5)
        with torch.enable_grad():
            for step in range(per_member):
                selected=training_ids[torch.multinomial(probability,32,replacement=True)]
                loss=costs(errors(network,selected),selected).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Recursive target dynamics produced a nonfinite loss.')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(),5.);optimizer.step();schedule.step()
                if (step+1)%100==0 or step+1==per_member:
                    validation=validate(network)
                    if validation<best_loss:
                        best_loss=validation;best_step=step+1
                        best_state={key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
                if (step+1)%500==0 or step+1==per_member:
                    print(dict(phase='public_target_rollout_fit',seed=seed,member=member,update=step+1,
                        loss=float(loss.detach()),best_validation=best_loss,constant_velocity=baseline),flush=True)
        states.append(best_state);records.append(dict(member=member,updates=per_member,
            selected_update=best_step,initial_validation_normalized_mse=initial_loss,
            validation_normalized_mse=best_loss))
        del network,optimizer,schedule
    predictor=d.PublicTargetFeedback(width,states,True,scenario_mode='members' if mixture else 'mean').to(device)
    normalized,coordinate_mse=validate(predictor,True)
    initial_error=None
    if initial is not None:
        before=d.PublicTargetFeedback(**initial['target_feedback']).to(device)
        initial_error=validate(before);del before
    model=copy.deepcopy(torch.load(predecessor,map_location='cpu',weights_only=False))
    model['target_feedback']=dict(width=width,states=states,absolute_frame=True,control_step=.2)
    if mixture:model['target_feedback']['scenario_mode']='members'
    model['target_feedback_fit']=dict(completed=True,seed=seed,updates=updates,ensemble_size=ensemble_size,
        width=width,parameters_per_member=sum(value.numel() for value in predictor.models[0].parameters()),
        observations=len(sequence),sequences=count,sequence_steps=horizon,training_roots=int(len(roots)-len(held_roots)),
        validation_roots=held_roots.tolist(),acquisition=specification,physical_steps=data['physical_steps'],members=records,
        initial_model=None if initial_model is None else str(initial_model),initial_model_sha256=initial_sha256,
        initial_training_updates=0 if initial is None else initial['target_feedback_fit']['updates'],
        validation_normalized_mse=normalized,constant_velocity_normalized_mse=baseline,
        initial_validation_normalized_mse=initial_error,validation_horizon_coordinate_mse=coordinate_mse,
        constant_velocity_horizon_coordinate_mse=baseline_mse,
        objective='recursive public target position and velocity errors across complete measured continuations',
        validation_scope='development prediction validation split by actual factual root; no final evaluation tasks',
        candidate_conditioning='recursive target state and public fleet response inside each complete candidate path')
    if mixture:
        model['target_feedback_fit'].update(mixture=mixture_record,
            objective='complete-trajectory mixture likelihood of recursive candidate-conditioned public target responses',
            forecast_scenarios=ensemble_size,
            initial_total_training_updates=(0 if initial is None else initial['target_feedback_fit']['updates']+
                initial['target_feedback_fit'].get('initial_total_training_updates',
                    initial['target_feedback_fit'].get('initial_training_updates',0))))
    atomic_save(model,output)
    del predictor
    return model


def source_successor_branch(task):
    """Measure an intervention through the deployed source episode decoder."""
    policy,job,scene,future,response_tick=task
    actor=d.joint_load(str(policy),Path(policy).stat().st_mtime_ns).cpu()
    env=copy.deepcopy(job['environment']);obs=job['observation'].copy()
    np.random.set_state(job['random_state'])
    if future is not None:np.random.seed(future)
    done=0;count=0;task_return=0.;source_return=0.;response=None
    def observe_response(current,observation,elapsed,terminal):
        nonlocal response
        physical_tick=count+elapsed
        if response is None and (terminal or physical_tick==response_tick):
            response=dict(packet=d.joint_packet(current,observation),terminal=int(terminal),steps=physical_tick)
    while not done:
        with torch.no_grad():
            action=(job['action'].copy() if count==0 else
                actor(d.mechanism_tensor(d.joint_packet(env,obs)),deterministic=True)[0][0].numpy())
        obs,reward,done,source_reward,steps=d.joint_advance(env,obs,actor,action,observe_step=observe_response)
        task_return+=actor.training_gamma**count*reward;source_return+=.99**count*source_reward;count+=steps
    seed,n,agility=scene
    row=dict(seed=int(seed),defenders=n,agility=agility,**d.outcome(env,done),
        terminal=int(done),physical_steps=count,task_return=task_return,task_discount=actor.training_gamma**count,
        discounted_return=source_return,environment_protocol=actor.environment_protocol,response=response)
    if future is not None:row['future_seed']=int(future)
    return row


def initialize_source_branches(source, opponent_checkpoint=None):
    torch.set_num_threads(1);d.initialize(str(source));d.numerical_source()
    configure_source_attacker(opponent_checkpoint)


def source_value_tuning_episode(task):
    scene,policy,revision,future,bandwidth=task
    actor=d.joint_load(str(policy),revision,(False,False)).cpu()
    value=actor.successor_task_value
    if value is None:raise ValueError('Value tuning requires an installed conditional value learner.')
    modules=[value]
    if hasattr(value.program_model,'bandwidth'):modules.append(value.program_model)
    original=[module.bandwidth for module in modules]
    try:
        for module in modules:module.bandwidth=bandwidth
        row=d.joint_episode((scene,str(policy),revision,False,0,0,0,0,future))
        row['value_bandwidth']=bandwidth
        return row
    finally:
        for module,previous in zip(modules,original):module.bandwidth=previous


def collect_successor_task_data(policy,scenes,output,*,repetitions=8,span=10,batch_size=64,
                                device='cuda:0',include_all_vessels=False,actuator_programs=False,
                                root_steps=0,root_future=None,joint_programs=None,future_start=0,
                                shared_futures=False,future_seeds=None,focus_codes=None,policy_iteration=False,
                                root_lookback=None,source_workers=1,future_block_edges=1,
                                factual_future_support=False):
    """Single-vessel paired task branches under the actual joint decoder.

    Edge groups use independent future blocks, sharing physical factual
    rollouts within a group. A branch records its
    physical successor after the intervention AND its true terminal task
    outcome under the frozen continuation. No safety surrogate supplies a
    value label. The same records also feed the unchanged ordinary learner.
    Source execution is the default so labels use the deployed decoder and
    arithmetic. Batched sampling remains an explicit source_workers=0 choice.
    """
    initialize();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if int(future_block_edges)!=future_block_edges or future_block_edges<1:
        raise ValueError('A future block must contain a positive integer number of intervention edges.')
    model=torch.load(policy,map_location='cpu',weights_only=False)
    actor=d.joint_load(str(policy),Path(policy).stat().st_mtime_ns)
    clean=copy.deepcopy(model)
    if policy_iteration:clean.pop('successor_task_value',None)
    if d.joint_value_objectives(clean):
        raise ValueError('Successor policy iteration cannot inherit another kind of value learner.')
    model.update(fixed_intervention_clock=True,physical_interventions=True,physical_intervention_span=span,
        physical_intervention_reversal=False,physical_span=span,physical_reversal=False,gate_period=1,
        branch_value_data=True,success_objective=True,initial_root=root_steps==0,observed_future=False,
        edge_futures=not shared_futures)
    program_options=([[-1.,-1.]]+[[left,right] for left in (0.,.5,1.) for right in (0.,.5,1.)]
        if actuator_programs else None)
    rollouts=[];total_steps=0
    for root_id,scene in enumerate(scenes):
        seed,n,agility=scene;np.random.seed(seed)
        env=d.numerical_source().TADEnv(n);obs,_=env.reset(agility=agility,noisy_agility=False)
        root_seed=root_future.get(int(seed)) if isinstance(root_future,dict) else root_future
        if root_seed is not None:np.random.seed(root_seed)
        root_target=root_steps.get(int(seed),0) if isinstance(root_steps,dict) else root_steps
        reached_steps=0
        if root_lookback is not None:
            history=[];done=0
            lookback=root_lookback.get(int(seed),20) if isinstance(root_lookback,dict) else root_lookback
            while not done:
                history.append((reached_steps,copy.deepcopy(env),obs.copy(),np.random.get_state()))
                with torch.no_grad():
                    decision=actor(d.mechanism_tensor(d.joint_packet(env,obs)),deterministic=True)[0][0].numpy()
                obs,_,done,_,elapsed=d.joint_advance(env,obs,actor,decision)
                reached_steps+=elapsed;total_steps+=elapsed
            cutoff=max(0,reached_steps-int(lookback))
            reached_steps,env,obs,random_state=max((item for item in history if item[0]<=cutoff),key=lambda item:item[0])
            np.random.set_state(random_state);root_target=reached_steps
        while reached_steps<root_target:
            with torch.no_grad():
                decision=actor(d.mechanism_tensor(d.joint_packet(env,obs)),deterministic=True)[0][0].numpy()
            # Observe only actual option boundaries. Truncating an option at
            # each physical tick silently discards its remaining decisions.
            obs,_,done,_,elapsed=d.joint_advance(env,obs,actor,decision)
            total_steps+=elapsed;reached_steps+=elapsed
            if done:raise ValueError('The predecessor terminated before the requested intervention state.')
        packet=d.joint_packet(env,obs)
        reference=actor(d.mechanism_tensor(packet),deterministic=True,gate_only=True)[0][0].detach().numpy()[:,:3]
        from conditional_fusion_value import successor_intervention_library
        requested_programs=list(joint_programs or [])
        if focus_codes is not None:
            radix=10 if actuator_programs else 4
            for code in focus_codes:
                if not 0<=int(code)<radix**n:
                    raise ValueError('A requested intervention code exceeds the fleet decision space.')
                requested_programs.append([(int(code)//radix**vessel)%radix for vessel in range(n)])
        programs,edges,subjects=successor_intervention_library(
            packet,reference,span,include_all_vessels,program_options,requested_programs)
        if focus_codes is not None:
            # Allocate additional paired futures to unresolved task-value
            # comparisons. Keep every intervened arm's single-vessel path
            # back to the factual program; no multi-vessel difference is
            # relabelled as a local effect.
            from conditional_fusion_value import public_return_context
            _,order=public_return_context(d.mechanism_tensor(packet),True,2)
            order=order[0].numpy();radix=10 if actuator_programs else 4
            def program_code(program):
                gates=program[order,9:11] if actuator_programs else program[order,3]
                digits=(np.where((gates<0.).any(-1),0,1+6*gates[:,0]+2*gates[:,1])
                    if actuator_programs else np.where(gates<0.,0,1+2*gates))
                return int((digits.astype(np.int64)*radix**np.arange(n)).sum())
            lookup={program_code(program):i for i,program in enumerate(programs)}
            selected={lookup[int(code)] for code in focus_codes}|{0}
            while True:
                parents={left for left,right in edges if right in selected}
                if parents<=selected:break
                selected|=parents
            edges=[(left,right) for left,right in edges if right in selected]
        jobs=[];tasks=[];sampled={};references={}
        root_futures=future_seeds.get(int(seed)) if isinstance(future_seeds,dict) else future_seeds
        for edge_id,(left,right) in enumerate(edges):
            for repeat in range(repetitions):
                stream=0 if shared_futures else edge_id//int(future_block_edges)
                future=(root_futures[(stream+repeat+future_start)%len(root_futures)]
                    if root_futures is not None else
                    int(np.random.SeedSequence([seed,root_id,stream,repeat+future_start,19081987]).generate_state(1)[0]))
                arms=tuple(dict.fromkeys((left,right,0))) if factual_future_support else (left,right)
                for arm in arms:
                    # A complete program and a future identify one physical
                    # continuation at this fixed root. Reusing that outcome
                    # keeps every local comparison incident to the same node
                    # consistent, including edges with overlapping futures.
                    key=(arm,future)
                    if key not in sampled:
                        sampled[key]=len(jobs)
                        jobs.append(dict(environment=env,observation=obs,random_state=np.random.get_state(),action=programs[arm]))
                        tasks.append((scene,future))
                    references[stream,repeat,arm]=sampled[key]
        if source_workers:
            branch_tasks=[(policy,job,scene,future,span) for job,(scene,future) in zip(jobs,tasks)]
            if source_workers==1:rows=[source_successor_branch(task) for task in branch_tasks]
            else:
                with ProcessPoolExecutor(max_workers=source_workers,initializer=initialize_source_branches,
                        initargs=(SOURCE,)) as executor:
                    rows=list(executor.map(source_successor_branch,branch_tasks,chunksize=1))
        else:
            rows=batched_credit_rows(policy,tasks,batch_size=batch_size,device=device,initial_jobs=jobs,response_tick=span)
        branches=[];outcomes={key:rows[index] for key,index in references.items()}
        total_steps+=sum(row['physical_steps'] for row in rows)
        for edge_id,(left,right) in enumerate(edges):
            for repeat in range(repetitions):
                stream=0 if shared_futures else edge_id//int(future_block_edges)
                before=outcomes[stream,repeat,left];after=outcomes[stream,repeat,right]
                def target(row):
                    return dict(success=row['success'],terminal=row['terminal'],reward=row['task_return'],
                        discount=row['task_discount'],next=None,first_step=dict(next=row['response']['packet'],
                        terminal=row['response']['terminal'],steps=row['response']['steps']))
                factual_row=(outcomes[stream,repeat,0] if factual_future_support or shared_futures
                    else before if left==0 else None)
                factual_target=target(factual_row) if factual_row is not None else None
                branches.append(dict(packet=packet,left=programs[left],right=programs[right],
                    before=target(before),after=target(after),reference_action=programs[0],
                    reference_outcome=factual_target,factual_outcome=factual_target,observed_future=False,
                    future_repetition=repeat,continuation_seed=before.get('future_seed'),root=(int(seed),reached_steps,left,right)))
        # Root factual returns use only actual q=0 continuations. Other edges
        # have independent futures and cannot borrow their factual label.
        factual=[rows[index] for (arm,_),index in sampled.items() if arm==0]
        rollouts.append(dict(row={k:v for k,v in factual[0].items() if k!='response'},steps=[],branches=branches,
            factual_success=float(np.mean([row['success'] for row in factual])),subjects=subjects))
        torch.save(dict(model=model,rollouts=rollouts,source_sha256=model['source_sha256'],
            simulation_steps=total_steps,use='development paired successor and terminal task learning',
            repetitions=repetitions,span=span,scenes=scenes,include_all_vessels=include_all_vessels,
            program_options=program_options,
            joint_programs=joint_programs,future_start=future_start,shared_futures=shared_futures,future_seeds=future_seeds,
            future_block_edges=int(future_block_edges),factual_future_support=bool(factual_future_support),
            focus_codes=focus_codes,policy_iteration=policy_iteration,
            root_steps=root_steps,root_future=root_future,root_lookback=root_lookback,source_workers=source_workers,
            initial_only=(root_lookback is None and
                (all(step==0 for step in root_steps.values()) if isinstance(root_steps,dict) else root_steps==0)),
            completed=root_id+1==len(scenes)),output/'training-data.pth')
        print(dict(phase='successor_task_collection',roots=root_id+1,total_roots=len(scenes),
            terminal_branches=len(rows),factual_success=rollouts[-1]['factual_success']),flush=True)
    return output/'training-data.pth'


def collect_joint_credit_batched(policy,output,tasks,batch_size=32,repetitions=2,device='cuda:0',factual=None):
    """Evaluate fixed-window program pairs against their frozen predecessor.

    Factual decisions retain their option boundaries. Every intervention and
    its factual reference share the copied state and continuation randomness;
    all labels come from complete closed-loop task outcomes.
    """
    from envs.TADgame import TADEnv
    model=torch.load(policy,map_location='cpu',weights_only=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    evidence=output/'training-data.pth'
    identity=(hashlib.sha256(Path(policy).read_bytes()).hexdigest(),tasks,repetitions)
    previous=torch.load(evidence,map_location='cpu',weights_only=False) if evidence.exists() else None
    if previous is not None:
        if previous.get('sampler_identity')!=identity:
            raise ValueError('Existing continuations belong to a different predecessor or sampling plan.')
        if previous.get('completed',False):return previous
    if model.get('context_prediction') or model.get('context_interceptors'):
        raise ValueError('This sampler uses the declared local and random joint program library.')
    span=int(model.get('physical_intervention_span',10))
    lookbacks=model.get('credit_lookbacks') or (model.get('failure_lookback',4.),)
    context_count=int(model.get('joint_contexts',24))
    paired=bool(model.get('paired_advantage',False))
    observed=bool(model.get('observed_future',False))
    fixed_clock=bool(model.get('fixed_intervention_clock',False))
    reversal=bool(model.get('physical_intervention_reversal',False))
    exhaustive=bool(model.get('exhaustive_contexts',False))
    max_held=int(model.get('max_held',0))
    edge_futures=bool(model.get('edge_futures',False))
    from conditional_fusion_value import JointInterventionValue
    program_proposer=[JointInterventionValue(**record).eval()
        for record in model.get('program_proposer',{}).get('values',[])]
    factual=(previous['factual'] if previous is not None else copy.deepcopy(factual) if factual is not None else
        batched_credit_rows(policy,tasks,batch_size=batch_size,device=device,keep_steps=True))
    balanced_subjects={}
    if model.get('balanced_futures',False):
        # Select a difficult public initial context once, then measure the
        # SAME program library under every factual continuation at that
        # context. Selecting only failed futures would confound rescue value
        # with the outcome-dependent sampling distribution.
        ranked={}
        for (scene,_),rollout in zip(tasks,factual):
            row=rollout['row'];seed,n,_=scene
            if row['success'] or (model.get('failure_duration',0.) and
                    row['duration']>model['failure_duration']):continue
            if seed in ranked and ranked[seed]<=row['duration']:continue
            packet=rollout['steps'][-1]['packet']
            if row['collision']:
                positions=packet['motion'][:,:2]
                distance=np.linalg.norm(positions[:,None]-positions[None],axis=-1)+1e6*np.eye(n)
                subjects=np.unravel_index(distance.argmin(),distance.shape)
            else:subjects=tuple(np.argsort(np.linalg.norm(packet['node'][:,2:4],axis=-1))[:2])
            ranked[seed]=row['duration'];balanced_subjects[seed]=subjects
    groups=[];jobs=[];job_tasks=[];locations=[]
    pending_roots=enumerate(zip(tasks,factual)) if previous is None else ()
    for episode,((scene,factual_future),rollout) in pending_roots:
        seed,n,agility=scene
        records=list(rollout['decisions']);indices=[record['step'] for record in records]
        midpoint=None
        if model.get('midpoint_refinement',False) and records:
            fraction=model.get('option_refinement_fraction',.5)
            delay=model.get('option_refinement_steps',0)
            eligible=[record for record in records if record['action'].shape[-1] in (6,9)
                and (record['action'][:,5]>=0.).any() and int(record['action'][0,4])>1
                and int(record['action'][0,4])-d.joint_refinement_step(record['action'][0,4],fraction,delay)>=2
                and (model.get('phase_preserving_refinement',False) or fraction==.5
                     or np.array_equal(record['action'][:,3],record['action'][:,5]))]
            if eligible:
                initial=eligible[0]['action'];duration=int(initial[0,4])
                elapsed=d.joint_refinement_step(duration,fraction,delay);candidate=eligible[0]['step']+elapsed
                if duration-elapsed==span and candidate<len(rollout['steps']):
                    midpoint=candidate
                    state=rollout['steps'][midpoint].get('controller_state')
                    if state is not None:
                        initial=state['program'];elapsed=state['ticks']
                    reference=d.joint_remaining_program(rollout['steps'][midpoint]['action'],initial,elapsed,
                        model.get('phase_preserving_refinement',False))
                    records.append(dict(step=midpoint,action=reference))
                    records.sort(key=lambda record:record['step'])
                    indices=[record['step'] for record in records]
        chosen=[]
        if model.get('midpoint_refinement',False):
            if midpoint is not None and (seed in balanced_subjects if model.get('balanced_futures',False)
                    else not rollout['row']['success']):chosen=[midpoint]
        elif model.get('balanced_futures',False):
            if seed in balanced_subjects:chosen=[indices[0]]
        elif not rollout['row']['success']:
            for lag in lookbacks:
                target=rollout['row']['duration']-lag
                eligible=[step for step in indices if .2*step<=target+1e-6]
                chosen.append(max(eligible) if eligible else indices[0])
            chosen=list(dict.fromkeys(chosen))
            if model.get('initial_root',False):chosen=[indices[0]]
            if model.get('failure_duration',0.) and rollout['row']['duration']>model['failure_duration']:
                chosen=[]
        roots={}
        if chosen:
            np.random.seed(seed);env=TADEnv(n,protocol=model.get('environment_protocol','source'))
            obs,_=env.reset(agility=agility,noisy_agility=False)
            if factual_future is not None:np.random.seed(factual_future)
            for step,record in enumerate(rollout['steps']):
                if step in chosen:
                    roots[step]=dict(environment=copy.deepcopy(env),observation=obs.copy(),
                        random_state=np.random.get_state(),packet=d.joint_packet(env,obs),
                        controller_state=record.get('controller_state') if model.get('prefix_refinement',False) else None)
                obs,_,done,_=d.joint_task_step(env,record['action'],1.,0.)
            if done!=rollout['row']['terminal']:
                raise ValueError('The factual replay changed the task outcome.')
            if rollout['row']['collision']:
                positions=np.asarray([v.pos for v in env.defender_list])
                distance=np.linalg.norm(positions[:,None]-positions[None],axis=-1)+1e6*np.eye(n)
                subjects=np.unravel_index(distance.argmin(),distance.shape)
            else:
                distance=np.asarray([np.linalg.norm(v.pos-env.attacker.pos) for v in env.defender_list])
                subjects=tuple(np.argsort(distance)[:2])
            if model.get('balanced_futures',False):subjects=balanced_subjects[seed]
            rng=np.random.default_rng(seed^917803)
            references={record['step']:record['action'] for record in records}
            for step in chosen:
                root=roots[step];reference=references[step].copy()
                future_rng=(np.random.default_rng(np.random.SeedSequence([
                    int(seed),0 if factual_future is None else int(factual_future),int(step),4165721]))
                    if model.get('independent_root_futures',False) else None)
                if model.get('prefix_refinement',False) and (reference.shape[-1]<6
                        or int(reference[0,4])!=span or not (reference[:,[3,5]]>=0.).any()):continue
                if reference.shape[-1]==3:
                    free=np.full((n,1),-1.,np.float32)
                    reference=np.concatenate((reference,free,np.full((n,1),span,np.float32),free),-1)
                arms=[];edges=[];grid=(-1.,0.,.5,1.)
                for x in grid:
                    for y in grid:
                        action=reference.copy();action[:,4]=span
                        action[subjects[0],3]=x;action[subjects[1],3]=y
                        action[:,5]=np.where((action[:,3]>=0.)&reversal,1.-action[:,3],action[:,3])
                        arms.append(action)
                for held in range(4):
                    for left in range(4):
                        for right in range(left+1,4):
                            edges.extend(((left*4+held,right*4+held),(held*4+left,held*4+right)))
                reference_index=grid.index(float(reference[subjects[0],3]))*4+grid.index(float(reference[subjects[1],3]))
                known={tuple(action[:,3]):j for j,action in enumerate(arms)}
                def add(action):
                    key=tuple(action[:,3])
                    if key not in known:known[key]=len(arms);arms.append(action)
                    return known[key]
                programs=None
                if max_held:
                    codes=np.arange(4**n);choices=(codes[:,None]//(4**np.arange(n)))%4
                    choices=choices[(choices>0).sum(-1)<=max_held]
                    programs=np.where(choices>0,(choices-1)*.5,-1.)
                elif exhaustive:
                    codes=np.arange(3**n);programs=((codes[:,None]//(3**np.arange(n)))%3)*.5
                count=0 if model.get('temporal_programs',False) else context_count if programs is None else len(programs)
                for context_id in range(count):
                    action=reference.copy();action[:,4]=span
                    if programs is not None:action[:,3]=programs[context_id]
                    elif context_id<2:action[:,3]=float(context_id)
                    elif context_id%2:action[:,3]=rng.integers(0,2,size=n)
                    else:action[:,3]=rng.choice(3,size=n,p=(.25,.5,.25))*.5
                    action[:,5]=np.where((action[:,3]>=0.)&reversal,1.-action[:,3],action[:,3])
                    changed=np.flatnonzero(action[:,3]!=reference[:,3])
                    if not len(changed):continue
                    subject=int(rng.choice(changed));deletion=action.copy()
                    deletion[subject,3]=reference[subject,3];deletion[subject,5]=reference[subject,5]
                    edge=(add(action),add(deletion))
                    if edge not in edges and edge[::-1] not in edges:edges.append(edge)
                if programs is not None:
                    existing={tuple(sorted(edge)) for edge in edges}
                    for gates in programs:
                        left=known[tuple(gates)]
                        for subject in range(n):
                            for gate in (grid if max_held else grid[1:]):
                                if gate<=gates[subject]:continue
                                alternate=gates.copy();alternate[subject]=gate
                                if tuple(alternate) not in known:continue
                                edge=tuple(sorted((left,known[tuple(alternate)])))
                                if edge not in existing:edges.append(edge);existing.add(edge)
                if model.get('temporal_programs',False):
                    from conditional_fusion_value import temporal_program_graph
                    with torch.no_grad():
                        seed_actions=[value(d.mechanism_tensor(root['packet']),torch.as_tensor(reference[None]))[0].numpy()
                            for value in program_proposer]
                    arms,edges,reference_index=temporal_program_graph(reference,subjects,span,context_count,rng,seed_actions,
                        model.get('temporal_blending_levels',(0.,.5,1.)))
                if model.get('prefix_refinement',False):
                    from conditional_fusion_value import prefix_intervention_graph
                    with torch.no_grad():
                        seed_actions=[value(d.mechanism_tensor(root['packet']),torch.as_tensor(reference[None]))[0].numpy()
                            for value in program_proposer]
                    arms,edges,reference_index=prefix_intervention_graph(reference,subjects,context_count,rng,seed_actions,
                        whole_remainder=model.get('remainder_refinement',False))
                if fixed_clock and not model.get('prefix_refinement',False):
                    for action in arms:
                        if (action[:,3]<0.).all():action[:,5]=-2.
                factual_index=len(arms)
                if paired:arms.append(reference.copy())
                # Retain common randomness WITHIN a local intervention pair,
                # but optionally allocate a fresh future to each comparison.
                # This measures more independent conditional differences than
                # repeating one small set of futures across the entire graph.
                schedules=([(list(range(len(arms))),edges,list(range(repetitions)))]
                    if not edge_futures else
                    ([(list(range(len(arms))),edges,[0])] if observed else [])+
                    [(sorted({left,right,reference_index,*([factual_index] if paired else [])}),
                      [(left,right)],[repetition])
                     for repetition in range(int(observed),repetitions) for left,right in edges])
                for selected,comparisons,ordinals in schedules:
                    mapping={arm:index for index,arm in enumerate(selected)}
                    group_arms=[arms[arm] for arm in selected]
                    group=dict(episode=episode,step=step,root=root,arms=group_arms,
                        edges=[(mapping[left],mapping[right]) for left,right in comparisons],
                        edge_identities=comparisons,
                        reference_index=mapping[reference_index],
                        factual_index=mapping[factual_index] if paired else len(group_arms),
                        reference_action=reference,
                        results=[[None]*len(group_arms) for _ in ordinals],futures=[],
                        observed_futures=[observed and ordinal==0 for ordinal in ordinals],
                        future_ordinals=ordinals)
                    group_id=len(groups);groups.append(group)
                    for repetition,ordinal in enumerate(ordinals):
                        future=int(rng.integers(1,2**31-1))
                        if future_rng is not None:future=int(future_rng.integers(1,2**31-1))
                        group['futures'].append(future)
                        for arm,action in enumerate(group_arms):
                            controller_state=root.get('controller_state')
                            if (controller_state is not None and model.get('prefix_refinement',False)
                                    and not np.array_equal(action[:,3:],reference[:,3:])):
                                controller_state=dict(controller_state,
                                    midpoint_checked=True)
                            jobs.append(dict(environment=root['environment'],observation=root['observation'],
                                random_state=root['random_state'],action=action,
                                controller_state=controller_state))
                            job_tasks.append((scene,None if observed and ordinal==0 else future))
                            locations.append((group_id,repetition,arm))
        if model.get('prefix_refinement',False):
            for decision in records:rollout['steps'][decision['step']]['action']=decision['action'].copy()
        rollout['steps']=[rollout['steps'][step] for step in indices]
    if previous is not None:
        groups=previous['groups'];jobs=previous['jobs']
        job_tasks=previous['job_tasks'];locations=previous['locations']
    root_count=len({(group['episode'],group['step']) for group in groups})
    print(dict(phase='paired_program_roots',episodes=len(factual),roots=root_count,continuations=len(jobs)),flush=True)
    branch_steps=previous['branch_steps'] if previous is not None else 0
    completed=previous['completed_continuations'] if previous is not None else 0
    def save_continuations(count):
        atomic_save(dict(model=model,sampler_identity=identity,completed=False,factual=factual,
            groups=groups,jobs=jobs,job_tasks=job_tasks,locations=locations,
            branch_steps=branch_steps,completed_continuations=count),evidence)
    save_continuations(completed)
    for begin in range(completed,len(jobs),128):
        selected=jobs[begin:begin+128]
        rows=batched_credit_rows(policy,job_tasks[begin:begin+128],batch_size=batch_size,
            device=device,initial_jobs=selected)
        for index,row in enumerate(rows,begin):
            group,repetition,arm=locations[index]
            groups[group]['results'][repetition][arm]=dict(reward=row['task_return'],discount=row['task_discount'],next=None,
                terminal=row['terminal'],success=row['success'],first_step=None,physical_steps=row['physical_steps'])
            branch_steps+=row['physical_steps']
        save_continuations(begin+len(rows))
        print(dict(phase='paired_program_continuations',completed=begin+len(rows),total=len(jobs),
            simulation_steps=branch_steps),flush=True)
    for group in groups:
        rollout=factual[group['episode']];seed=rollout['row']['seed'];arms=group['arms']
        for repetition,outcomes in enumerate(group['results']):
            is_observed=group.get('observed_futures',[observed and r==0 for r in range(repetitions)])[repetition]
            ordinal=group.get('future_ordinals',list(range(repetitions)))[repetition]
            identities=group.get('edge_identities',group['edges'])
            for (left,right),identity_edge in zip(group['edges'],identities):
                rollout['branches'].append(dict(packet=group['root']['packet'],left=arms[left],right=arms[right],
                    before=outcomes[left],after=outcomes[right],
                    reference_action=group.get('reference_action',arms[group['reference_index']]),
                    reference_outcome=outcomes[group['reference_index']],
                    factual_outcome=outcomes[group['factual_index']] if paired else None,
                    observed_future=is_observed,future_repetition=ordinal,
                    continuation_seed=None if is_observed else group['futures'][repetition],
                    root=(seed,group['step'],*identity_edge)))
            rollout['simulation_steps']+=sum(outcome['physical_steps'] for outcome in outcomes)
    data=dict(model=model,rollouts=factual,simulation_steps=sum(row['simulation_steps'] for row in factual),
        source_sha256=model.get('source_sha256'),completed=True,inference_device=device,
        sampler_identity=identity,
        sampling='fixed-predecessor paired programs; unchanged source-equation task; batched policy inference')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_save(data,output/'training-data.pth')
    d.save_json(output/'development.json',dict(collection=d.summarize([row['row'] for row in factual]),
        simulation_steps=data['simulation_steps'],roots=root_count,continuations=len(jobs)))
    return data


def evaluate(policy, output, workers=8, references=()):
    record=torch.load(policy,map_location='cpu',weights_only=False)
    if record.get('cooperative_role_credit') and not references:
        tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
            [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
        rows=batched_credit_rows(policy,tasks)
        variant=dict(model=str(policy),label=Path(policy).parent.name,mechanism_removal=record.get('mechanism_ablation',(False,False,False)),
            summary=d.summarize(rows),original_trajectories=d.summarize(rows[:64]),additional_futures=d.summarize(rows[64:]),rows=rows)
        d.save_json(output,dict(protocol='same development scenes with paired noise repeats',bank=4151000000,repetitions=4,variants=[variant]))
        return
    arguments = ['joint-evaluate', '--start', str(policy), '--output', str(output), '--teams', '6',
                 '--agilities', '2', '2.5', '--bank', '4151000000', '--per-cell', '32',
                 '--repetitions', '4', '--workers', str(workers)]
    if references:
        arguments += ['--references', *map(str, references)]
    command(arguments)


def run_arm(without_m1):
    initialize()
    name = 'six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195'
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    start = out / 'current.pth'
    if not start.exists():
        model = torch.load(RUNS / 'six-role-discovery-192/current.pth', map_location='cpu', weights_only=False)
        model.pop('mechanism_ablation', None)
        if without_m1:
            model.update(peer=False, risk_prior=0., risk_feedback=False, coupled_shortlist=0,
                         pair_scale=0., mechanism_ablation=(True, False, True))
        atomic_save(model, start)
    device = 'cpu' if without_m1 else 'cuda:0'
    explored = collect(start, SCENES, out / 'discovery', 3, 1, None, device)
    scenes, programs, selection = shortlist(explored)
    confirmed = collect(start, scenes, out / 'confirmation', 8, 4, programs, device)
    if 'confirmation_simulation_steps' not in confirmed:
        confirmed['confirmation_simulation_steps'] = confirmed['simulation_steps']
        confirmed['discovery_simulation_steps'] = explored['simulation_steps']
        confirmed['simulation_steps'] += explored['simulation_steps']
    confirmed.update(selection=selection, selection_from=str(out / 'discovery/training-data.pth'))
    atomic_save(confirmed, out / 'confirmation/training-data.pth')
    fit = ['joint-local-value', '--start', str(start), '--data', str(out / 'confirmation/training-data.pth'),
           '--output', str(out), '--joint-local-joint', '--joint-graph', '--joint-levels', '3',
           '--joint-paired-advantage', '--joint-independent-futures', '--joint-credit-context', '2',
           '--joint-role-transport', '--joint-credit-locality', '--joint-credit-confidence', '.5',
           '--joint-credit-margin', '0', '--joint-support-uncertainty', '--joint-initial-root',
           '--joint-physical-span', '80', '--difference-weight', '4', '--critic-only', '--workers', '1']
    if without_m1:
        fit.append('--joint-no-interaction')
    command(fit)
    evaluate(out / 'policy.pth', out / 'paired-futures.json')
    result = json.loads((out / 'paired-futures.json').read_text())['variants'][0]
    deadline = time.monotonic() + 3600
    while True:
        try:
            variants = json.loads(REFERENCE.read_text())['variants']
            if len(variants) == 2:
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('Remote reference evaluation has not completed.')
        time.sleep(5)
    reference = variants[1 if without_m1 else 0]
    comparison = dict(result=result['summary'], without_M2=reference['summary'],
        paired=compare(result['rows'], reference['rows']), counterfactual_training_steps=confirmed['simulation_steps'],
        neural_training_steps=275000, total_training_steps=275000 + confirmed['simulation_steps'],
        use='Development evidence on one six-defender base; five-training-seed paper evidence remains separate.')
    d.save_json(out / 'comparison.json', comparison)
    print(dict(phase='confirmed_role_arm_complete', arm=name, comparison=comparison), flush=True)


def run_baseline():
    from train import main
    from utils.config import load_config
    from utils.manager import ExperimentManager, set_seed
    torch.set_num_threads(1)
    cfg = load_config(str(REPO / 'train/experiments/six-defender-paper-arboids-seed42-20261014/config.yaml'))
    set_seed(42)
    exp = ExperimentManager(cfg, base_dir=str(REPO / 'train/experiments'),
        run_id='six-defender-paper-arboids-seed42-remote-20261015', repeat_idx=1)
    main(cfg, exp, torch.device('cuda:0'))


def run_source_baseline_suite(family,seeds=range(42,47),defenders=6):
    """Original source algorithms with the same training schedule at each fleet size."""
    import csv
    from train import main
    from utils.config import load_config
    from utils.manager import ExperimentManager, set_seed

    class DevelopmentManager(ExperimentManager):
        def record_metrics(self,**metrics):
            super().record_metrics(**metrics)
            self.latest=metrics

        def save_model(self,model,filename=None,step=None):
            super().save_model(model,filename,step)
            if defenders!=6 and int(self.latest['step'])==275000:
                model.save(str(Path(self.exp_dir)/'budget-275000.pth'))
            score=(self.latest['def_sr'],self.latest['reward'])
            if score>getattr(self,'best',(-1.,-float('inf'))):
                self.best=score
                model.save(str(Path(self.exp_dir)/'best-development.pth'))
                d.save_json(Path(self.exp_dir)/'selection.json',dict(family=family,seed=seed,training_defenders=defenders,
                    selection=self.latest,criterion='Maximum development success, then development return.',
                    use='Development checkpoint selection; final paired evaluation remains separate.'))

    torch.set_num_threads(1)
    for seed in seeds:
        prefix='six-paper' if defenders==6 else f'fleet{defenders}-paper'
        name=f'{prefix}-source-{family}-seed{seed}-20261015'
        directory=REPO/'train/experiments'/name
        metrics=directory/'metrics1.csv'
        if metrics.exists():
            rows=list(csv.DictReader(metrics.open()))
            if rows and int(rows[-1]['step'])>=1000000:
                continue
            raise RuntimeError('Incomplete source training requires explicit continuation handling: '+str(directory))
        cfg=load_config(str(REPO/'train/configs/paper-parameters.yaml'))
        cfg.agent.defender_num=defenders
        cfg.agent.adaptive=family=='arboids'
        cfg.agent.residual=family!='sac'
        cfg.training.seed=seed;cfg.training.model_name=family+'.pth'
        set_seed(seed)
        exp=DevelopmentManager(cfg,base_dir=str(REPO/'train/experiments'),run_id=name,repeat_idx=1)
        main(cfg,exp,torch.device('cuda:0'))
        print(dict(phase='source_baseline_seed_complete',family=family,seed=seed,defenders=defenders),flush=True)


def boids_reference_episode(scene):
    from envs.TADgame import TADEnv
    seed,n,agility=scene;np.random.seed(seed)
    env=TADEnv(n,protocol='paper-parameters-v1')
    env.reset(agility=agility,noisy_agility=False)
    done=0;returns=0.;steps=0
    while not done:
        _,reward,done,_=env.step(np.zeros((n,2)),'Boids')
        returns+=(.99**steps)*float(np.mean(reward));steps+=1
    return dict(seed=seed,defenders=n,agility=agility,**d.outcome(env,done),
        discounted_return=returns,environment_protocol='paper-parameters-v1')


@torch.no_grad()
def source_marl_evaluate(agent,scenes,batch_size=100,controller='RL'):
    from envs.TADgame import TADEnv
    saved=np.random.get_state();rows=[]
    try:
        for start in range(0,len(scenes),batch_size):
            active=[]
            for scene in scenes[start:start+batch_size]:
                seed,n,agility=scene;np.random.seed(seed)
                env=TADEnv(n,protocol='paper-parameters-v1');obs,_=env.reset(agility,noisy_agility=False)
                if getattr(agent,'public_state',False):obs=source_public_observation(env,obs)
                active.append(dict(scene=scene,env=env,obs=obs,rng=np.random.get_state(),returns=0.,steps=0))
            while active:
                obs=np.stack([item['obs'] for item in active]);chosen=agent.act(obs,True)
                actions=chosen[0] if isinstance(chosen,list) else chosen
                remaining=[]
                for item,action in zip(active,actions):
                    np.random.set_state(item['rng']);env=item['env'];obs,reward,done,_=env.step(action,controller)
                    item['returns']+=(.99**item['steps'])*float(np.mean(reward));item['steps']+=1
                    if done:
                        seed,n,agility=item['scene']
                        rows.append(dict(seed=seed,defenders=n,agility=agility,**d.outcome(env,done),
                            discounted_return=item['returns'],mean_reward=float(env.Rewards.mean())))
                    else:
                        if getattr(agent,'public_state',False):obs=source_public_observation(env,obs)
                        item.update(obs=obs,rng=np.random.get_state());remaining.append(item)
                active=remaining
        order={scene[0]:i for i,scene in enumerate(scenes)};rows.sort(key=lambda row:order[row['seed']])
        return rows
    finally:np.random.set_state(saved)


def source_public_observation(env,observation):
    """Give the generic MARL baselines the same measured fleet information."""
    velocity=np.asarray([boat.velocity for boat in env.defender_list]);n=len(velocity)
    heading=np.array([boat.theta for boat in env.defender_list]);c,s=np.cos(heading),np.sin(heading)
    own=np.stack((c*velocity[:,0]+s*velocity[:,1],-s*velocity[:,0]+c*velocity[:,1],velocity[:,2]),-1)/5.
    peers=[]
    for i in range(n):
        indices=[j for j in range(n) if j!=i];relative=velocity[indices,:2]-velocity[i,:2]
        drift=np.stack((c[i]*relative[:,0]+s[i]*relative[:,1],-s[i]*relative[:,0]+c[i]*relative[:,1]),-1)/5.
        angle=heading[indices]-heading[i]
        peers.append(np.concatenate((observation[i,14:].reshape(n-1,2),drift,
            np.sin(angle)[:,None],np.cos(angle)[:,None],velocity[indices,2:3]/5.),-1).reshape(-1))
    return np.concatenate((observation[:,:14],own,np.stack(peers)),-1).astype(np.float32)


def run_source_marl_suite(family,seeds=range(42,47)):
    from envs.TADgame import TADEnv
    from policy.mappo import SourceMAPPO,SourceMAAC
    from utils.manager import set_seed
    torch.set_num_threads(1)
    for seed in seeds:
        out=REPO/'train/experiments'/f'six-paper-public-{family}-seed{seed}-20261015'
        out.mkdir(parents=True,exist_ok=True);metrics_path=out/'metrics.json';state_path=out/'learning-state.pth'
        if (out/'benchmark.json').exists():continue
        set_seed(seed)
        if state_path.exists():
            state=torch.load(state_path,map_location='cuda:0',weights_only=False)
            agent=state['agent'];history=state['history'];steps=state['steps']
            # The saved state is a completed rollout boundary. New episodes
            # and off-policy replay are explicitly regenerated on resumption.
        else:
            agent=(SourceMAPPO if family=='mappo' else SourceMAAC)(device='cuda:0',seed=seed)
            history=[];steps=0
        environments=[TADEnv(6,protocol='paper-parameters-v1') for _ in range(10)]
        observations=np.stack([source_public_observation(env,env.reset(2.+.25*(steps//250000),noisy_agility=True)[0]) for env in environments])
        best=max((item['summary']['success'],item['mean_reward']) for item in history) if history else (-1.,-float('inf'))
        if family=='maac':
            capacity=200000;size=0;position=0;rng=np.random.default_rng(seed+steps)
            obs_dim=observations.shape[-1]
            replay=(np.empty((capacity,6,obs_dim),np.float32),np.empty((capacity,6,2),np.float32),
                np.empty((capacity,6),np.float32),np.empty((capacity,6,obs_dim),np.float32),np.empty(capacity,np.float32))
        while steps<1000000:
            rollout=[];losses={}
            for tick in range(500):
                before=observations.copy()
                if family=='mappo':actions,latent,logp,values=agent.act(before)
                else:actions=rng.uniform(-1.,1.,(10,6,2)) if size<50000 else agent.act(before)
                following=[];rewards=[];ends=[];physical_next=[]
                for env,action in zip(environments,actions):
                    obs,reward,done,_=env.step(action,'RL');physical_next.append(source_public_observation(env,obs))
                    rewards.append(reward);ends.append(bool(done))
                    if done:obs,_=env.reset(2.+.25*(steps//250000),noisy_agility=True)
                    following.append(source_public_observation(env,obs))
                rewards=np.asarray(rewards);ends=np.asarray(ends,dtype=np.float32);observations=np.stack(following)
                if family=='mappo':rollout.append((before,latent,logp,values,rewards,ends))
                else:
                    indices=(position+np.arange(10))%capacity
                    for buffer,values in zip(replay,(before,actions,rewards,np.stack(physical_next),ends)):buffer[indices]=values
                    position=(position+10)%capacity;size=min(capacity,size+10)
                    if size>=50000 and (steps+10)%100==0:
                        for update in range(10):
                            indices=rng.integers(size,size=512)
                            losses=agent.update([buffer[indices] for buffer in replay])
                steps+=10
            if family=='mappo':
                batch=[np.stack([row[k] for row in rollout]) for k in range(6)]
                batch.append(agent.act(observations,True)[-1]);losses=agent.update(batch)
            rows=source_marl_evaluate(agent,d.bank(4211000000,100,((6,2.),)))
            summary=d.summarize(rows);mean_reward=float(np.mean([row['mean_reward'] for row in rows]))
            result=dict(step=steps,summary=summary,mean_reward=mean_reward,losses=losses,rows=rows)
            history.append(result)
            checkpoint=dict(family=family,agent=agent,steps=steps,seed=seed,history=history,
                training_defenders=6,environment_protocol='paper-parameters-v1',
                method='MAPPO, feedforward shared policy and centralized value' if family=='mappo' else
                    'MAAC, continuous-action adaptation with shared policy and attention critic')
            atomic_save(checkpoint,state_path)
            score=(summary['success'],mean_reward)
            if score>best:
                best=score;atomic_save(checkpoint,out/'best-development.pth')
            d.save_json(metrics_path,dict(family=family,seed=seed,history=history))
            print(dict(phase='source_marl_training',family=family,seed=seed,step=steps,
                success=summary['success'],best=best[0],losses=losses),flush=True)
        selected=torch.load(out/'best-development.pth',map_location='cuda:0',weights_only=False)
        agent=selected['agent'];cells=[(6,float(a)) for a in np.arange(1.5,3.01,.25)]
        cells.extend((n,a) for n in (3,4,5,7,8) for a in (2.,2.5));rows=[]
        for cell_index,cell in enumerate(cells):
            rows.extend(source_marl_evaluate(agent,d.bank(4201000000+cell_index*10000,100,(cell,))))
        d.save_json(out/'benchmark.json',dict(family=family,seed=seed,training_defenders=6,
            selected_step=selected['steps'],summary=d.summarize(rows),rows=rows,completed=True))
        print(dict(phase='source_marl_seed_complete',family=family,seed=seed),flush=True)


def run_source_frozen_benchmarks():
    """Frozen six-vessel checkpoints, matched agility and unseen-fleet cells."""
    import csv
    from policy.networks import ActorAdap,ActorSAC
    torch.set_num_threads(1)
    class FrozenSource:
        def __init__(self,path,adaptive):
            self.actor=(ActorAdap(6,8,3,512) if adaptive else ActorSAC(6,8,2,512)).cuda().eval()
            self.actor.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True))
        @torch.no_grad()
        def act(self,obs,deterministic=True):
            b,n,dim=obs.shape
            return self.actor(torch.as_tensor(obs.reshape(b*n,dim),dtype=torch.float32,device='cuda'),True,False)[0].reshape(b,n,-1).cpu().numpy()
    pending={(family,seed) for family in ('sac','rp','arboids') for seed in range(42,47)}
    deadline=time.monotonic()+72*3600
    while pending:
        progressed=False
        for family,seed in sorted(pending):
            out=REPO/'train/experiments'/f'six-paper-source-{family}-seed{seed}-20261015'
            target=out/'benchmark.json'
            if target.exists() and json.loads(target.read_text()).get('completed'):
                pending.remove((family,seed));continue
            metrics=out/'metrics1.csv'
            if not metrics.exists():continue
            history=list(csv.DictReader(metrics.open()))
            if not history or int(history[-1]['step'])<1000000:continue
            if time.time()-metrics.stat().st_mtime<30.:continue
            agent=FrozenSource(out/'best-development.pth',family=='arboids')
            cells=[(6,float(a)) for a in np.arange(1.5,3.01,.25)]
            cells.extend((n,a) for n in (3,4,5,7,8) for a in (2.,2.5))
            rows=[];controller={'sac':'RL','rp':'Res','arboids':'AdaRes'}[family]
            for i,cell in enumerate(cells):
                rows.extend(source_marl_evaluate(agent,d.bank(4201000000+i*10000,100,(cell,)),controller=controller))
            selection=json.loads((out/'selection.json').read_text())
            d.save_json(target,dict(family=family,seed=seed,training_defenders=6,selection=selection,
                summary=d.summarize(rows),rows=rows,completed=True))
            pending.remove((family,seed));progressed=True
            print(dict(phase='source_frozen_benchmark',family=family,seed=seed,episodes=len(rows)),flush=True)
        if time.monotonic()>deadline and pending:raise TimeoutError('Source training was not completed within the bounded benchmark batch.')
        if pending and not progressed:time.sleep(30)


def run_boids_reference():
    out=RUNS/'six-paper-boids-reference';out.mkdir(parents=True,exist_ok=True)
    target=out/'benchmark.json'
    rows=json.loads(target.read_text())['rows'] if target.exists() else []
    cells=[(6,float(a)) for a in np.arange(1.5,3.01,.25)]
    cells.extend((n,a) for n in (3,4,5,7,8) for a in (2.,2.5))
    scenes=d.bank(4201000000,100,cells);finished={row['seed'] for row in rows}
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(boids_reference_episode,[scene for scene in scenes if scene[0] not in finished],chunksize=4):
            rows.append(row)
            if len(rows)%100==0 or len(rows)==len(scenes):
                d.save_json(target,dict(method='Unchanged source Boids',bank=4201000000,per_cell=100,
                    cells=cells,summary=d.summarize(rows),rows=rows,completed=len(rows)==len(scenes)))
                print(dict(phase='boids_reference',episodes=len(rows),total=len(scenes)),flush=True)


def run_fleet_profile(labels,counts=(2,3,4,5,6,7,8),per_cell=64):
    """Select fleet-size candidates on a common development bank, without refitting."""
    import hashlib
    candidates={
        'full': 'six-local-temporal-full-251',
        'full-invariant': 'fleet-invariant-credit-293',
        'full-clean': 'six-continued-credit-full-261',
        'm1': 'six-trained-interaction-185',
        'm2': 'six-continued-credit-without-m1-262',
        'neither': 'six-trained-baseline-184',
        'mission-m1': 'six-finite-hazard-full-291',
        'mission-full': 'six-finite-hazard-full-291',
    }
    for fleet in (3,4):
        for arm in ('full','m1','m2','neither'):
            source_arm='without-m1' if arm in ('m2','neither') else 'full'
            candidates[f'native-{fleet}-{arm}']=f'fleet{fleet}-finite-hazard-{source_arm}-seed42'
    directory=RUNS/'fleet-selection-20261015';directory.mkdir(parents=True,exist_ok=True)
    for label in labels:
        if label=='full-invariant':
            target=RUNS/candidates[label]/'policy.pth'
            if not target.exists():
                source=RUNS/'six-local-temporal-full-251/policy.pth'
                record=torch.load(source,map_location='cpu',weights_only=False)
                credit=dict(record['cooperative_role_credit'])
                contexts=torch.as_tensor(credit['contexts'])
                credit['support_weights']=(contexts.std(0,unbiased=False)>1e-6).float()
                credit['fit']=dict(credit['fit'],support_metric='observed-variation-only',
                    active_dimensions=int(credit['support_weights'].sum()))
                target.parent.mkdir(parents=True,exist_ok=True)
                atomic_save(dict(record,cooperative_role_credit=credit),target)
        if label=='arboids-three':
            from policy.networks import ActorAdap
            torch.set_num_threads(1)
            policy=SOURCE
            class FrozenARBoids:
                def __init__(self):
                    self.actor=ActorAdap(6,8,3,512).eval()
                    self.actor.load_state_dict(torch.load(policy,map_location='cpu',weights_only=True))
                @torch.no_grad()
                def act(self,obs,deterministic=True):
                    b,n,dim=obs.shape
                    return self.actor(torch.as_tensor(obs.reshape(b*n,dim),dtype=torch.float32),True,False)[0].reshape(b,n,-1).numpy()
            source_agent=FrozenARBoids()
            model=dict(source_training_defenders=3)
        else:
            teacher=(label=='mission-m1' or label.startswith('native-') and label.rsplit('-',1)[-1] in ('m1','neither'))
            policy=RUNS/candidates[label]/('teacher.pth' if teacher else 'policy.pth')
            if label=='mission-full':
                deadline=time.monotonic()+3*3600
                while not policy.exists():
                    if time.monotonic()>deadline:raise TimeoutError('The finite-horizon credit fit did not finish within this experiment batch.')
                    time.sleep(30)
            model=torch.load(policy,map_location='cpu',weights_only=False)
        fingerprint=hashlib.sha256(policy.read_bytes()).hexdigest()
        destination=directory/(label+'.json')
        previous=json.loads(destination.read_text()) if destination.exists() else {}
        if previous and previous['policy_sha256']!=fingerprint:
            raise ValueError('Fleet profiling must retain the same frozen policy: '+label)
        rows=previous.get('rows',[])
        for count in counts:
            for agility in (2.,2.5):
                # Cell identifiers remain fixed when extending a profile's sample size.
                start=4240000000+count*100000+int(10*agility)*1000
                scenes=d.bank(start,per_cell,((count,agility),))
                finished={row['seed'] for row in rows}
                pending=[(scene,None) for scene in scenes if scene[0] not in finished]
                if pending:
                    rows.extend(source_marl_evaluate(source_agent,[scene for scene,_ in pending],controller='AdaRes')
                        if label=='arboids-three' else batched_credit_rows(policy,pending,batch_size=32))
                included=[row for row in rows if row['defenders'] in counts and
                    row['seed']<4240000000+row['defenders']*100000+int(10*row['agility'])*1000+per_cell]
                rankings=[]
                for fleet in counts:
                    group=[row for row in included if row['defenders']==fleet]
                    if len(group)!=2*per_cell:continue
                    captured=[row['duration'] for row in group if row['capture']]
                    rankings.append(dict(defenders=fleet,summary=d.summarize(group),
                        capture_time=float(np.mean(captured)) if captured else None))
                rankings.sort(key=lambda item:(item['summary']['success'],item['summary']['capture'],
                    -item['summary']['collision']),reverse=True)
                result=dict(label=label,model=str(policy),policy_sha256=fingerprint,
                    source_training_defenders=model.get('source_training_defenders'),
                    mechanism_training_defenders=None if label=='arboids-three' else model.get('source_training_defenders',6),training_seed=42,
                    use=('Historical three-vessel ARBoids reproduction evaluated under the common paper task. Fresh matched-protocol training is separate.'
                        if label=='arboids-three' else 'Development fleet-size selection. This evaluates a frozen model; its training fleet size is recorded separately from the evaluation fleet size.'),
                    counts=list(counts),per_cell=per_cell,summary=d.summarize(included),
                    ranking=rankings,rows=rows,completed=len(included)==len(counts)*2*per_cell)
                d.save_json(destination,result)
                cell=[row for row in included if row['defenders']==count and row['agility']==agility]
                print(dict(phase='fleet_profile',label=label,defenders=count,agility=agility,
                    episodes=len(cell),success=float(np.mean([row['success'] for row in cell]))),flush=True)


def collect_feedback_shard(task):
    start,records,output,device,*settings=task
    programs,repetitions,future_start=settings[:3] if settings else (None,4,1)
    offsets=settings[3] if len(settings)>3 else None
    d.collect_initial_roles_batched(start,[record['scene'] for record in records],output,
        repetitions=repetitions,future_start=future_start,programs=programs,
        batch_size=32,device=device,root_records=records,proposal_offsets=offsets)
    return str(Path(output)/'training-data.pth')


def collect_root_programs(start,records,programs,out,repetitions,future_start,workers=2,device='cuda:0',offsets=None):
    out.mkdir(parents=True,exist_ok=True);master=out/'training-data.pth'
    if master.exists():return torch.load(master,map_location='cpu',weights_only=False)
    tasks=[]
    for rank in range(workers):
        part=out/'shards'/str(rank);part.mkdir(parents=True,exist_ok=True)
        tasks.append((str(start),records[rank::workers],str(part),device,
            programs[rank::workers],repetitions,future_start,
            None if offsets is None else offsets[rank::workers]))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(collect_feedback_shard,tasks))
    rolls=[]
    for path in paths:
        shard=torch.load(path,map_location='cpu',weights_only=False)
        if not shard['completed']:raise RuntimeError('Joint-role continuation is incomplete.')
        rolls.extend(shard['rollouts'])
    rolls.sort(key=lambda roll:roll['row']['root_id'])
    data=dict(model=torch.load(start,map_location='cpu',weights_only=False),rollouts=rolls,
        completed=True,simulation_steps=sum(roll['simulation_steps'] for roll in rolls))
    atomic_save(data,master);return data


def role_samples(roll):
    samples={}
    for branch in roll['branches']:
        for side,name in (('left','before'),('right','after')):
            gates=tuple(float(g) for g in branch[side][:,3])
            samples.setdefault(gates,{})[branch['continuation_seed']]=branch[name]['success']
    return samples


def run_joint_roles(without_m1=False):
    initialize()
    parent=RUNS/('six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195')
    out=RUNS/('six-joint-role-probe-without-m1-204' if without_m1 else 'six-joint-role-probe-199')
    execution=dict(workers=4,device='cpu') if without_m1 else {}
    out.mkdir(parents=True,exist_ok=True);start=out/'current.pth'
    model=torch.load(parent/'current.pth',map_location='cpu',weights_only=False)
    model.update(physical_span=40,physical_intervention_span=40);atomic_save(model,start)
    factual=torch.load(parent/'discovery/training-data.pth',map_location='cpu',weights_only=False)
    selected={scene[0] for scene in [*SCENES[:8],*SCENES[32:40]]}
    records=d.feedback_role_roots([factual],lookbacks=(12.,),scene_seeds=selected)
    programs=[]
    for record in records:
        n=record['scene'][1];choices=[[-1.]*n]
        for i in range(n):
            for gate in (0.,.5,1.):
                choice=[-1.]*n;choice[i]=gate;choices.append(choice)
        for i in range(n):
            for j in range(n):
                if i==j:continue
                choice=[-1.]*n;choice[i]=0.;choice[j]=1.;choices.append(choice)
        programs.append(choices)
    print(dict(phase='joint_role_roots',roots=len(records),programs=len(programs[0])),flush=True)
    explored=collect_root_programs(start,records,programs,out/'discovery',4,1,**execution)
    selected=[];programs=[]
    for roll in explored['rollouts']:
        samples=role_samples(roll);rank=lambda key:np.mean(list(samples[key].values()))
        single=max((key for key in samples if sum(g>=0. for g in key)==1),key=rank)
        pair=max((key for key in samples if sum(g>=0. for g in key)==2),key=rank)
        free=tuple([-1.]*len(single));choices=[free,single,pair]
        for i,g in enumerate(pair):
            if g>=0.:
                bridge=list(free);bridge[i]=g
                if tuple(bridge) not in choices:choices.append(tuple(bridge))
        selected.append(dict(single=single,pair=pair));programs.append(choices)
    confirmed=collect_root_programs(start,records,programs,out/'confirmation',8,5,**execution)
    rows=[]
    for roll,choice in zip(confirmed['rollouts'],selected):
        samples=role_samples(roll);free=tuple([-1.]*roll['row']['defenders'])
        rows.append(dict(seed=roll['row']['seed'],free=float(np.mean(list(samples[free].values()))),
            single=float(np.mean(list(samples[choice['single']].values()))),
            pair=float(np.mean(list(samples[choice['pair']].values()))),
            single_program=choice['single'],pair_program=choice['pair']))
    # The selected programs and their independently measured returns are kept separate.
    summary={key:float(np.mean([row[key] for row in rows])) for key in ('free','single','pair')}
    d.save_json(out/'comparison.json',dict(summary=summary,rows=rows,
        simulation_steps=explored['simulation_steps']+confirmed['simulation_steps']))
    print(dict(phase='joint_role_probe_complete',summary=summary),flush=True)


def proposal_samples(roll):
    samples={}
    for branch in roll['branches']:
        for side,name in (('left','before'),('right','after')):
            code=tuple(float(x) for x in branch[side][:,[3,6,7]].flatten())
            samples.setdefault(code,{})[branch['continuation_seed']]=branch[name]['success']
    return samples


def run_proposal_probe(without_m1):
    initialize()
    parent=RUNS/('six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195')
    out=RUNS/('six-proposal-credit-probe-without-m1-206' if without_m1 else 'six-proposal-credit-probe-full-205')
    out.mkdir(parents=True,exist_ok=True);start=out/'current.pth'
    model=torch.load(parent/'current.pth',map_location='cpu',weights_only=False)
    model.update(physical_span=40,physical_intervention_span=40,proposal_interventions=True)
    atomic_save(model,start)
    factual=torch.load(parent/'discovery/training-data.pth',map_location='cpu',weights_only=False)
    selected={scene[0] for scene in [*SCENES[:8],*SCENES[32:40]]}
    records=d.feedback_role_roots([factual],lookbacks=(12.,),scene_seeds=selected)
    programs=[];offsets=[]
    for record in records:
        n=record['scene'][1];choices=[[-1.]*n];changes=[np.zeros((n,2),np.float32)]
        for subject in range(n):
            for correction in ((0.,0.),(.5,.5),(-.5,-.5),(.5,-.5),(-.5,.5)):
                gate=[-1.]*n;gate[subject]=1.;change=np.zeros((n,2),np.float32)
                change[subject]=correction;choices.append(gate);changes.append(change)
        programs.append(choices);offsets.append(changes)
    execution=dict(workers=4,device='cpu' if without_m1 else 'cuda:0')
    print(dict(phase='proposal_credit_roots',roots=len(records),programs=len(programs[0])),flush=True)
    explored=collect_root_programs(start,records,programs,out/'discovery',4,1,offsets=offsets,**execution)
    programs=[];offsets=[];selection=[]
    for roll in explored['rollouts']:
        samples=proposal_samples(roll);n=roll['row']['defenders']
        free=next(key for key in samples if np.all(np.array(key).reshape(n,3)[:,0]<0.))
        rank=lambda key:(np.mean(list(samples[key].values())),-np.count_nonzero(np.array(key).reshape(n,3)[:,0]>=0.))
        old=max((key for key in samples if not np.any(np.array(key).reshape(n,3)[:,1:])),key=rank)
        new=max(samples,key=rank)
        matched=np.array(new).reshape(n,3).copy();matched[:,1:]=0.;matched=tuple(matched.flatten())
        choices=list(dict.fromkeys((free,old,new,matched)))
        if len(choices)==1:
            choices.append(next(key for key in samples if key!=free and
                not np.any(np.array(key).reshape(n,3)[:,1:])))
        programs.append([np.array(key).reshape(n,3)[:,0].tolist() for key in choices])
        offsets.append([np.array(key).reshape(n,3)[:,1:].tolist() for key in choices])
        selection.append(dict(free=free,old=old,new=new,matched=matched))
    confirmed=collect_root_programs(start,records,programs,out/'confirmation',8,5,offsets=offsets,**execution)
    rows=[]
    for roll,selected in zip(confirmed['rollouts'],selection):
        samples=proposal_samples(roll)
        # A root selecting only the free program has no contrast edges.
        if not samples:
            row=roll['row'];raise ValueError('A free-only confirmation needs an explicit reference contrast.')
        rows.append(dict(seed=roll['row']['seed'],**{key:float(np.mean(list(samples[code].values())))
            for key,code in selected.items()},selection=selected))
    summary={key:float(np.mean([row[key] for row in rows])) for key in ('free','old','new','matched')}
    d.save_json(out/'comparison.json',dict(summary=summary,rows=rows,
        simulation_steps=explored['simulation_steps']+confirmed['simulation_steps']))
    print(dict(phase='proposal_credit_probe_complete',summary=summary),flush=True)


def run_feedback(without_m1):
    from conditional_fusion_value import fit_feedback_role_credit
    initialize()
    parent=RUNS/('six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195')
    out=RUNS/('six-feedback-without-m1-198' if without_m1 else 'six-feedback-full-197')
    out.mkdir(parents=True,exist_ok=True)
    start=out/'current.pth';master=out/'training-data.pth'
    model=torch.load(parent/'current.pth',map_location='cpu',weights_only=False)
    model.update(physical_span=20,physical_intervention_span=20)
    atomic_save(model,start)
    if master.exists():
        data=torch.load(master,map_location='cpu',weights_only=False)
    else:
        factual=[torch.load(parent/name/'training-data.pth',map_location='cpu',weights_only=False)
            for name in ('discovery','confirmation')]
        records=d.feedback_role_roots(factual)
        del factual
        print(dict(phase='feedback_roots',arm=out.name,roots=len(records),layouts=len(SCENES)),flush=True)
        tasks=[]
        for rank in range(8):
            part=out/'shards'/str(rank);part.mkdir(parents=True,exist_ok=True)
            tasks.append((str(start),records[rank::8],str(part),'cpu' if without_m1 else 'cuda:0'))
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            paths=list(pool.map(collect_feedback_shard,tasks))
        rollouts=[]
        for path in paths:
            shard=torch.load(path,map_location='cpu',weights_only=False)
            if not shard['completed']:raise RuntimeError('Feedback continuations did not finish.')
            rollouts.extend(shard['rollouts'])
        rollouts.sort(key=lambda roll:roll['row']['root_id'])
        if [roll['row']['root_id'] for roll in rollouts]!=list(range(len(records))):
            raise RuntimeError('Feedback root coverage differs from the sampling plan.')
        data=dict(model=model,rollouts=rollouts,completed=True,
            simulation_steps=sum(roll['simulation_steps'] for roll in rollouts),
            use='Public feedback states; all single-vessel short roles under paired independent futures.')
        atomic_save(data,master)
    credit,fit=fit_feedback_role_credit(data,device='cuda:0')
    print(dict(phase='feedback_fit',arm=out.name,fit=fit,simulation_steps=data['simulation_steps']),flush=True)
    d.save_json(out/'fit.json',dict(fit=fit,simulation_steps=data['simulation_steps']))
    variants=[]
    for name,confidence,margin in (('conservative',1.,.05),('bootstrap',.5,.02),('mean',0.,.02)):
        value=dict(credit,confidence=confidence,margin=margin)
        candidate=dict(model,feedback_role_credit=value,
            simulation_steps=data['simulation_steps'],feedback_training=fit)
        policy=out/(name+'.pth');atomic_save(candidate,policy)
        result_path=out/(name+'-paired-futures.json')
        if not result_path.exists():evaluate(policy,result_path,workers=8)
        result=json.loads(result_path.read_text())['variants'][0]
        reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
        result.update(name=name,paired=compare(result['rows'],reference['rows']))
        variants.append(result)
        d.save_json(out/'comparison.json',dict(variants=variants,without_M2=reference['summary'],
            counterfactual_training_steps=data['simulation_steps'],neural_training_steps=275000))
        print(dict(phase='feedback_evaluated',arm=out.name,name=name,
            summary=result['summary'],paired=result['paired']),flush=True)


def run_local_feedback(without_m1):
    parent=RUNS/('six-feedback-without-m1-198' if without_m1 else 'six-feedback-full-197')
    out=RUNS/('six-feedback-local-without-m1-200' if without_m1 else 'six-feedback-local-full-201')
    out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+10800
    while not (parent/'training-data.pth').exists():
        if time.monotonic()>deadline:raise TimeoutError('Feedback labels did not finish.')
        time.sleep(5)
    fit=['joint-local-value','--start',str(parent/'current.pth'),'--data',str(parent/'training-data.pth'),
        '--output',str(out),'--joint-local-joint','--joint-graph','--joint-levels','3',
        '--joint-paired-advantage','--joint-independent-futures','--joint-credit-context','2',
        '--joint-role-transport','--joint-credit-locality','--joint-credit-confidence','.5',
        '--joint-credit-margin','0','--joint-support-uncertainty','--joint-physical-span','20',
        '--difference-weight','4','--critic-only','--workers','1']
    if without_m1:fit.append('--joint-no-interaction')
    command(fit)
    model=torch.load(out/'policy.pth',map_location='cpu',weights_only=False);variants=[]
    for bandwidth in (.15,.3):
        model['joint_intervention_value']['bandwidth']=bandwidth
        policy=out/('bandwidth-'+str(bandwidth)+'.pth');atomic_save(model,policy)
        target=out/('bandwidth-'+str(bandwidth)+'-paired-futures.json')
        evaluate(policy,target,workers=6)
        result=json.loads(target.read_text())['variants'][0]
        reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
        result.update(bandwidth=bandwidth,paired=compare(result['rows'],reference['rows']))
        variants.append(result)
        d.save_json(out/'comparison.json',dict(variants=variants,without_M2=reference['summary']))
        print(dict(phase='feedback_local_evaluated',arm=out.name,bandwidth=bandwidth,
            summary=result['summary'],paired=result['paired']),flush=True)


def run_supported_feedback(without_m1):
    from conditional_fusion_value import public_return_context
    initialize()
    parent=RUNS/('six-feedback-without-m1-198' if without_m1 else 'six-feedback-full-197')
    out=RUNS/('six-feedback-supported-without-m1-202' if without_m1 else 'six-feedback-supported-full-203')
    out.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+10800
    while not (parent/'mean.pth').exists():
        if time.monotonic()>deadline:raise TimeoutError('Feedback network fitting did not finish.')
        time.sleep(5)
    model=torch.load(parent/'mean.pth',map_location='cpu',weights_only=False)
    data=torch.load(parent/'training-data.pth',map_location='cpu',weights_only=False)
    contexts=torch.cat([public_return_context(d.mechanism_tensor(roll['branches'][0]['packet']),version=2)
        for roll in data['rollouts']])
    model['feedback_role_credit'].update(support_contexts=contexts,support_bandwidth=.3)
    policy=out/'policy.pth';atomic_save(model,policy)
    evaluate(policy,out/'paired-futures.json',workers=4)
    result=json.loads((out/'paired-futures.json').read_text())['variants'][0]
    reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
    comparison=dict(summary=result['summary'],paired=compare(result['rows'],reference['rows']))
    d.save_json(out/'comparison.json',comparison)
    print(dict(phase='supported_feedback_evaluated',arm=out.name,comparison=comparison),flush=True)


def run_credit_learning(without_m1,grounded=False,response=False):
    from conditional_fusion_value import fit_cooperative_role_credit
    initialize()
    feedback=RUNS/('six-feedback-without-m1-198' if without_m1 else 'six-feedback-full-197')
    joint=RUNS/('six-joint-role-probe-without-m1-204' if without_m1 else 'six-joint-role-probe-199')
    proposal=RUNS/('six-proposal-credit-probe-without-m1-206' if without_m1 else 'six-proposal-credit-probe-full-205')
    out=RUNS/(('six-grounded-credit-without-m1-210' if without_m1 else 'six-grounded-credit-full-209') if grounded
        else ('six-conditional-action-without-m1-208' if without_m1 else 'six-conditional-action-full-207'))
    if response:
        out=RUNS/('six-response-credit-without-m1-214' if without_m1 else 'six-response-credit-full-213')
    out.mkdir(parents=True,exist_ok=True)
    paths=[feedback/'training-data.pth',joint/'discovery/training-data.pth',joint/'confirmation/training-data.pth',
        proposal/'discovery/training-data.pth',proposal/'confirmation/training-data.pth']
    deadline=time.monotonic()+10800
    while not all(path.exists() for path in paths):
        if time.monotonic()>deadline:raise TimeoutError('Conditional action data did not complete.')
        time.sleep(5)
    datasets=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    base=torch.load(feedback/'current.pth',map_location='cpu',weights_only=False)
    simulation_steps=sum(data['simulation_steps'] for data in datasets)
    variants=[]
    for conditional in (True,False):
        family='conditional' if conditional else 'ordinary'
        existing=out/(family+'-0.5.pth')
        if existing.exists():
            previous=torch.load(existing,map_location='cpu',weights_only=False)
            credit,fit=previous['cooperative_role_credit'],previous['conditional_action_fit']
        else:
            credit,fit=fit_cooperative_role_credit(datasets,device='cuda:0',proposal=True,conditional=conditional,
                auxiliary_weight=.2 if grounded else 0.,probability_difference=grounded,response_features=response)
        if response:credit['bandwidth']=0.
        print(dict(phase='conditional_action_fit',arm=out.name,family=family,fit=fit),flush=True)
        for confidence in (.5,0.):
            model=dict(base,cooperative_role_credit=dict(credit,confidence=confidence),
                conditional_action_fit=fit,simulation_steps=simulation_steps)
            name=family+'-'+str(confidence);policy=out/(name+'.pth');atomic_save(model,policy)
            target=out/(name+'-paired-futures.json')
            if not target.exists():evaluate(policy,target,workers=4)
            result=json.loads(target.read_text())['variants'][0]
            reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
            result.update(name=name,paired=compare(result['rows'],reference['rows']))
            variants.append(result)
            d.save_json(out/'comparison.json',dict(variants=variants,without_M2=reference['summary'],
                simulation_steps=simulation_steps,neural_training_steps=275000,
                control='Ordinary terminal-return model has the same candidate actions, architecture, data and tuning budget.'))
            print(dict(phase='conditional_action_evaluated',arm=out.name,name=name,
                summary=result['summary'],paired=result['paired']),flush=True)


def run_generalized_credit(without_m1):
    parent=RUNS/('six-grounded-credit-without-m1-210' if without_m1 else 'six-grounded-credit-full-209')
    out=RUNS/('six-generalized-credit-without-m1-212' if without_m1 else 'six-generalized-credit-full-211')
    out.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+10800;variants=[]
    for family in ('conditional','ordinary'):
        source=parent/(family+'-0.5.pth')
        while not source.exists():
            if time.monotonic()>deadline:raise TimeoutError('Grounded conditional model did not finish.')
            time.sleep(5)
        model=torch.load(source,map_location='cpu',weights_only=False)
        for confidence in (.5,0.):
            model['cooperative_role_credit'].update(bandwidth=0.,confidence=confidence)
            name=family+'-'+str(confidence);policy=out/(name+'.pth');atomic_save(model,policy)
            target=out/(name+'-paired-futures.json');evaluate(policy,target,workers=3)
            result=json.loads(target.read_text())['variants'][0]
            reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
            result.update(name=name,paired=compare(result['rows'],reference['rows']))
            variants.append(result)
            d.save_json(out/'comparison.json',dict(variants=variants,without_M2=reference['summary'],
                use='Public-state terminal probability and bootstrap uncertainty; no stored-layout similarity veto.'))
            print(dict(phase='generalized_credit_evaluated',arm=out.name,name=name,
                summary=result['summary'],paired=result['paired']),flush=True)


def run_balanced_roles(without_m1):
    initialize()
    parent=RUNS/('six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195')
    out=RUNS/('six-balanced-role-without-m1-216' if without_m1 else 'six-balanced-role-full-215')
    out.mkdir(parents=True,exist_ok=True)
    start=out/'current.pth'
    atomic_save(torch.load(parent/'current.pth',map_location='cpu',weights_only=False),start)
    data=collect(start,SCENES,out/'continuations',8,20,None,'cpu' if without_m1 else 'cuda:0',workers=4 if without_m1 else 8)
    rows=[]
    for roll in data['rollouts']:
        samples=role_samples(roll);free=tuple([-1.]*roll['row']['defenders'])
        choices=list(samples);futures=sorted(samples[free]);values=np.array([[samples[k][f] for f in futures] for k in choices])
        cross=[]
        for held in range(len(futures)):
            mean=np.delete(values,held,axis=1).mean(1)
            # Choosing no intervention on a tie prevents arbitrary changes.
            best=choices.index(free) if mean[choices.index(free)]>=mean.max() else int(mean.argmax())
            cross.append(values[best,held])
        rows.append(dict(seed=roll['row']['seed'],free=float(values[choices.index(free)].mean()),
            oracle=float(values.mean(1).max()),leave_one_future_out=float(np.mean(cross))))
    result=dict(rows=rows,means={key:float(np.mean([r[key] for r in rows])) for key in ('free','oracle','leave_one_future_out')},
        simulation_steps=data['simulation_steps'],use='Balanced terminal-return labels; oracle is diagnostic, not deployment performance.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='balanced_role_credit_collected',arm=out.name,result=result['means']),flush=True)


def run_balanced_credit(without_m1,contexts=False,task_frame=False):
    from conditional_fusion_value import fit_cooperative_role_credit
    initialize()
    parent=RUNS/('six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195')
    feedback=RUNS/('six-feedback-without-m1-198' if without_m1 else 'six-feedback-full-197')
    balanced=RUNS/('six-balanced-role-without-m1-216' if without_m1 else 'six-balanced-role-full-215')
    out=RUNS/('six-balanced-credit-without-m1-218' if without_m1 else 'six-balanced-credit-full-217')
    if contexts:out=RUNS/('six-context-credit-without-m1-226' if without_m1 else 'six-context-credit-full-225')
    if task_frame:out=RUNS/('six-geometric-credit-without-m1-228' if without_m1 else 'six-geometric-credit-full-227')
    out.mkdir(parents=True,exist_ok=True)
    paths=[parent/'discovery/training-data.pth',parent/'confirmation/training-data.pth',
        feedback/'training-data.pth',balanced/'continuations/training-data.pth']
    if contexts:
        additional=RUNS/('six-conditional-context-without-m1-222' if without_m1 else 'six-conditional-context-full-221')
        paths.append(additional/'continuations/training-data.pth')
    deadline=time.monotonic()+10800
    while not all(path.exists() for path in paths):
        if time.monotonic()>deadline:raise TimeoutError('Balanced credit data did not finish.')
        time.sleep(5)
    datasets=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    base=torch.load(parent/'current.pth',map_location='cpu',weights_only=False)
    simulation_steps=sum(data.get('confirmation_simulation_steps',data['simulation_steps']) for data in datasets)
    variants=[]
    for conditional in (True,False):
        family='conditional' if conditional else 'ordinary'
        existing=out/(family+'-0.5.pth')
        if existing.exists():
            previous=torch.load(existing,map_location='cpu',weights_only=False)
            credit,fit=previous['cooperative_role_credit'],previous['conditional_action_fit']
        else:
            credit,fit=fit_cooperative_role_credit(datasets,device='cuda:0',proposal=False,conditional=conditional,
                auxiliary_weight=.2,probability_difference=True,response_features=True,response_standardized=True,
                max_changed=0 if contexts else 1,future_bootstrap=True,
                all_pairs=contexts,coordinate_steps=2 if contexts else 0,task_frame=task_frame)
        credit['bandwidth']=0.
        if contexts:credit['wait_span']=True
        print(dict(phase='balanced_credit_fit',arm=out.name,family=family,fit=fit),flush=True)
        for confidence in (.5,0.):
            model=dict(base,cooperative_role_credit=dict(credit,confidence=confidence),
                conditional_action_fit=fit,simulation_steps=simulation_steps)
            name=family+'-'+str(confidence);policy=out/(name+'.pth');atomic_save(model,policy)
            target=out/(name+'-paired-futures.json')
            if not target.exists():evaluate(policy,target,workers=4)
            result=json.loads(target.read_text())['variants'][0]
            reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
            result.update(name=name,paired=compare(result['rows'],reference['rows']))
            variants.append(result)
            d.save_json(out/'comparison.json',dict(variants=variants,without_M2=reference['summary'],
                simulation_steps=simulation_steps,neural_training_steps=275000,
                control='Matched architecture and terminal continuations; conditional coordinate credit versus ordinary returns.'))
            print(dict(phase='balanced_credit_evaluated',arm=out.name,name=name,
                summary=result['summary'],paired=result['paired']),flush=True)


def factual_credit_shard(task):
    start,tasks,output=task
    output=Path(output)
    if output.exists():return str(output)
    rollouts=batched_credit_rows(start,tasks,keep_steps=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(dict(model=torch.load(start,map_location='cpu',weights_only=False),rollouts=rollouts,
        simulation_steps=sum(row['simulation_steps'] for row in rollouts)),output)
    return str(output)


def fusion_policy_factual_shard(task):
    start,tasks,output,seed=task;output=Path(output)
    if output.exists():return str(output)
    torch.manual_seed(seed)
    rollouts=batched_credit_rows(start,tasks,keep_steps=True,sample_policy=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(dict(model=torch.load(start,map_location='cpu',weights_only=False),rollouts=rollouts,
        simulation_steps=sum(roll['simulation_steps'] for roll in rollouts)),output)
    return str(output)


def fusion_policy_pair_shard(task):
    start,roots,output=task
    d.collect_fusion_td_pairs(start,roots,output)
    return str(Path(output)/'training-data.pth')


def run_fusion_policy_learning(without_m1,conditional):
    from conditional_fusion_value import ConditionalFusionLearner
    initialize()
    name={ (False,True):'six-learned-fusion-full-233',(False,False):'six-learned-fusion-without-m2-234',
           (True,True):'six-learned-fusion-without-m1-235',(True,False):'six-learned-fusion-neither-236'}[without_m1,conditional]
    out=RUNS/name;out.mkdir(parents=True,exist_ok=True);current=out/'current.pth';state=out/'learning-state.pth'
    if state.exists():
        previous=torch.load(state,map_location='cuda:0',weights_only=False)
        learner=previous['learner'];history=previous['history'];total_steps=previous['simulation_steps'];begin=previous['iteration']
    else:
        parent=RUNS/('six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195')
        base=torch.load(parent/'current.pth',map_location='cpu',weights_only=False)
        base.update(preference_strength=0.,physical_interventions=False,physical_span=1,physical_intervention_span=1,
            fitted_iteration=False,initial_root=False,continuous_gate_learning=True,conditional_training=conditional,
            fusion_learning='paired-conditional-soft-policy-v1',mechanism_ablation=(without_m1,not conditional,without_m1))
        atomic_save(base,current)
        learner=ConditionalFusionLearner(current,conditional=conditional,device='cuda:0')
        factual=RUNS/('six-factual-credit-without-m1-232' if without_m1 else 'six-factual-credit-full-231')
        initial=torch.load(factual/'training-data.pth',map_location='cpu',weights_only=False)
        learner.remember(initial['rollouts']);learner.train(updates=800,warmup=True)
        history=[];total_steps=initial['simulation_steps'];begin=0
        atomic_save(dict(learner=learner,history=history,simulation_steps=total_steps,iteration=begin),state)
    if not any(row['iteration']==0 for row in history):
        reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
        rows=reference['rows'][:len(SCENES)]
        initial_result=dict(iteration=0,summary=d.summarize(rows),rows=rows,simulation_steps=0,
            conditional=conditional,without_M1=without_m1,use='Common unchanged fusion policy before internal learning.')
        history.insert(0,initial_result)
        if not (out/'policy.pth').exists() or (initial_result['summary']['success'],initial_result['summary']['capture'])>=max(
                (row['summary']['success'],row['summary']['capture']) for row in history):
            atomic_save(dict(learner.model,selection=initial_result),out/'policy.pth')
    best=max((row['summary']['success'],row['summary']['capture']) for row in history)
    selected_model=torch.load(out/'policy.pth',map_location='cpu',weights_only=False)
    learner.actor.load_state_dict(selected_model['actor'])
    for iteration in range(begin,20):
        atomic_save(learner.policy(),current)
        scenes=d.bank(4171000000+iteration*10000,32,((6,2.),(6,2.5)))
        tasks=[(scene,None) for scene in scenes];folder=out/('iteration-'+str(iteration))
        work=[(str(current),tasks[rank::3],str(folder/'factual'/str(rank)/'training-data.pth'),42+iteration*101+rank) for rank in range(3)]
        with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            paths=list(pool.map(fusion_policy_factual_shard,work))
        parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
        rollouts=[roll for part in parts for roll in part['rollouts']]
        dataset=dict(model=parts[0]['model'],rollouts=rollouts)
        roots=d.feedback_role_roots([dataset],lookbacks=(.4,2.,6.,12.))
        # Cover intervention timing and every vessel, instead of assigning a
        # noisy return difference to one randomly selected vessel per scene.
        selection=np.random.default_rng(4210000000+iteration).permutation(len(roots))[:64]
        roots=[roots[int(index)] for index in selection]
        work=[(str(current),roots[rank::3],str(folder/'paired-events'/str(rank))) for rank in range(3)]
        with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            paths=list(pool.map(fusion_policy_pair_shard,work))
        pair_parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
        pairs=[pair for part in pair_parts for pair in part['pairs']]
        used=sum(roll['simulation_steps'] for roll in rollouts)+sum(part['simulation_steps'] for part in pair_parts)
        total_steps+=used;learner.remember(rollouts,pairs)
        before_actor=copy.deepcopy(learner.actor.state_dict());before_optimizer=copy.deepcopy(learner.actor_optimizer.state_dict())
        metrics=learner.train(updates=1000)
        model=dict(learner.policy(),simulation_steps=total_steps);atomic_save(model,current)
        rows=batched_credit_rows(current,[(scene,None) for scene in SCENES])
        summary=d.summarize(rows)
        result=dict(iteration=iteration+1,summary=summary,rows=rows,simulation_steps=total_steps,
            iteration_simulation_steps=used,critic_updates=metrics,conditional=conditional,without_M1=without_m1,
            policy_improvement=model.get('fusion_improvement'))
        history.append(result)
        accepted=(summary['success'],summary['capture'])>=best
        result['accepted']=accepted
        if accepted:
            best=(summary['success'],summary['capture']);atomic_save(dict(model,selection=result),out/'policy.pth')
        if not accepted:
            learner.actor.load_state_dict(before_actor);learner.actor_optimizer.load_state_dict(before_optimizer)
            for group in learner.actor_optimizer.param_groups:group['lr']=max(3e-6,.7*group['lr'])
        d.save_json(out/'comparison.json',dict(history=history,
            use='Development policy learning. Both arms update the physical fusion policy with identical data types and critic capacity; only paired contrast supervision differs.'))
        atomic_save(dict(learner=learner,history=history,simulation_steps=total_steps,iteration=iteration+1),state)
        print(dict(phase='learned_fusion_iteration',arm=name,iteration=iteration+1,summary=summary,simulation_steps=total_steps),flush=True)
    policy=out/'policy.pth';rows=batched_credit_rows(policy,[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES])
    reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
    result=dict(summary=d.summarize(rows),rows=rows,paired=compare(rows,reference['rows']),simulation_steps=total_steps)
    d.save_json(out/'paired-futures.json',result)
    print(dict(phase='learned_fusion_complete',arm=name,summary=result['summary'],paired=result['paired']),flush=True)


def run_fusion_credit_confirmation(without_m1):
    initialize()
    parent=RUNS/('six-learned-fusion-without-m1-235' if without_m1 else 'six-learned-fusion-full-233')
    state=torch.load(parent/'learning-state.pth',map_location='cuda:0',weights_only=False)
    learner=state['learner'];iteration=state['iteration']-1
    records,before,after,variance=learner.credit_targets()
    effect=(after-before).cpu().numpy();order=np.argsort(-effect)
    selected=[];seen=set()
    for index in order:
        record=records[int(index)];key=(record['scene'][0],record['root_id'])
        if key in seen:continue
        seen.add(key);selected.append(int(index))
        if len(selected)==16:break
    parts=[torch.load(parent/f'iteration-{iteration}'/'factual'/str(rank)/'training-data.pth',
        map_location='cpu',weights_only=False) for rank in range(3)]
    dataset=dict(model=parts[0]['model'],rollouts=[roll for part in parts for roll in part['rollouts']])
    roots=d.feedback_role_roots([dataset],lookbacks=(.4,2.,6.,12.))
    lookup={(root['scene'][0],root['id']):root for root in roots};chosen=[]
    for index in selected:
        record=records[index];root=lookup[record['scene'][0],record['root_id']]
        root['credit_subject']=int(np.abs(record['right'][:,2]-record['left'][:,2]).argmax())
        chosen.append(root)
    out=parent/'credit-confirmation';out.mkdir(parents=True,exist_ok=True)
    policy=out/'policy.pth'
    model=dict(learner.model,actor={k:v.cpu() for k,v in learner.behavior.state_dict().items()})
    atomic_save(model,policy)
    results=[]
    for duration in (1,5,10):
        data=d.collect_fusion_td_pairs(policy,chosen,out/f'iteration-{iteration}-duration-{duration}',
            repetitions=8,horizon=300,intervention_steps=duration,future_start=100)
        pairs=data['pairs']
        if any(pair[side]['success'] is None for pair in pairs for side in ('before','after')):
            raise RuntimeError('A full credit confirmation did not reach the actual task terminal.')
        gains=np.array([pair['after']['success']-pair['before']['success'] for pair in pairs])
        result=dict(duration_steps=duration,interventions=len(pairs),
            baseline_success=float(np.mean([pair['before']['success'] for pair in pairs])),
            intervention_success=float(np.mean([pair['after']['success'] for pair in pairs])),
            gain=float(gains.mean()),rescues=int((gains>0).sum()),harms=int((gains<0).sum()),
            simulation_steps=data['simulation_steps'])
        results.append(result)
        d.save_json(out/'comparison.json',dict(training_iteration=iteration+1,estimated_one_step_gain=float(effect[selected].mean()),
            variants=results,roots=[dict(scene=root['scene'],root_id=root['id'],subject=root['credit_subject']) for root in chosen],
            use='Diagnostic on credit-selected states with disjoint full continuations; not deployed method success.'))
        print(dict(phase='fusion_credit_confirmed',without_M1=without_m1,**result),flush=True)


def run_committed_credit_confirmation():
    initialize()
    parent=RUNS/'six-learned-fusion-full-233';source=parent/'credit-confirmation'
    recorded=json.loads((source/'comparison.json').read_text());iteration=recorded['training_iteration']-1
    parts=[torch.load(parent/f'iteration-{iteration}'/'factual'/str(rank)/'training-data.pth',
        map_location='cpu',weights_only=False) for rank in range(3)]
    dataset=dict(model=parts[0]['model'],rollouts=[roll for part in parts for roll in part['rollouts']])
    roots=d.feedback_role_roots([dataset],lookbacks=(.4,2.,6.,12.))
    lookup={(root['scene'][0],root['id']):root for root in roots};chosen=[]
    for row in recorded['roots']:
        root=lookup[row['scene'][0],row['root_id']];root['credit_subject']=row['subject'];chosen.append(root)
    out=RUNS/'six-committed-credit-probe-245';out.mkdir(parents=True,exist_ok=True)
    model=torch.load(source/'policy.pth',map_location='cpu',weights_only=False)
    model['conditional_commitment']=True;policy=out/'policy.pth';atomic_save(model,policy);results=[]
    for duration in (1,5,10):
        data=d.collect_fusion_td_pairs(policy,chosen,out/f'duration-{duration}',
            repetitions=8,horizon=300,intervention_steps=duration,future_start=100)
        pairs=data['pairs'];gains=np.array([row['after']['success']-row['before']['success'] for row in pairs])
        row=dict(duration_steps=duration,interventions=len(pairs),
            baseline_success=float(np.mean([row['before']['success'] for row in pairs])),
            intervention_success=float(np.mean([row['after']['success'] for row in pairs])),
            gain=float(gains.mean()),rescues=int((gains>0).sum()),harms=int((gains<0).sum()),simulation_steps=data['simulation_steps'])
        results.append(row)
        d.save_json(out/'comparison.json',dict(variants=results,unconditioned_continuation=recorded['variants'],
            roots=recorded['roots'],use='Same selected states, physical interventions and future noise; future free-vessel decisions account for active commitments. Not deployed success.'))
        print(dict(phase='committed_credit_confirmed',**row),flush=True)


def role_terminal_samples(rollout):
    samples={}
    for branch in rollout['branches']:
        repetition=branch['future_repetition']
        for side,endpoint in (('left','before'),('right','after')):
            program=tuple(float(x) for x in branch[side][:,3])
            samples.setdefault(program,{})[repetition]=float(branch[endpoint]['success'])
    return {key:np.array([value[index] for index in sorted(value)]) for key,value in samples.items()}


def local_temporal_datasets(data,complete_contrasts=False):
    """Expose finite interventions as connected one-vessel role contrasts."""
    result=[]
    for block in data['paired_durations']:
        rolls={};span=block['duration']
        for pair in block['pairs']:
            scene=pair['scene'];key=(scene[0],pair['root_id'])
            if key not in rolls:rolls[key]=dict(row=dict(seed=scene[0]),branches=[])
            action=pair['left'];n=len(action);free=np.full((n,1),-1.,np.float32)
            before=np.concatenate((action,free,np.full((n,1),span,np.float32),free),-1)
            after=before.copy();subject=pair['subject']
            after[subject,3]=after[subject,5]=pair['right'][subject,2]
            rolls[key]['branches'].append(dict(pair,left=before,right=after,reference_action=before,
                observed_future=False))
        if complete_contrasts:
            # Every arm already shares its root, unchanged partners, and future
            # noise. Contrast the observed fixed-gate arms directly as well as
            # contrasting each one with free continuation. This supplies the
            # missing interior-versus-endpoint ranking without new outcomes.
            for roll in rolls.values():
                groups={}
                for branch in roll['branches']:
                    groups.setdefault((branch['subject'],branch['future_repetition']),[]).append(branch)
                extra=[]
                for branches in groups.values():
                    branches=sorted(branches,key=lambda item:float(item['right'][item['subject'],3]))
                    for i,left in enumerate(branches):
                        for right in branches[i+1:]:
                            if np.array_equal(left['right'][:,3],right['right'][:,3]):continue
                            extra.append(dict(left,left=left['right'].copy(),right=right['right'].copy(),
                                before=left['after'],after=right['after'],conditional_contrast=True))
                roll['branches'].extend(extra)
        result.append(dict(model=dict(data['model'],physical_intervention_span=span),rollouts=list(rolls.values())))
    return result


def run_local_effect_credit(without_m1,temporal=False):
    from conditional_fusion_value import fit_local_role_effects
    initialize()
    suffix='without-m1' if without_m1 else 'full'
    number=(252 if without_m1 else 251) if temporal else (254 if without_m1 else 253)
    out=RUNS/f'six-local-{("temporal" if temporal else "initial")}-{suffix}-{number}'
    out.mkdir(parents=True,exist_ok=True)
    source=RUNS/('six-temporal-terminal-without-m1-238' if without_m1 else 'six-temporal-terminal-full-237')
    if temporal:
        data=torch.load(source/'training-data.pth',map_location='cpu',weights_only=False)
        datasets=local_temporal_datasets(data);steps=data['simulation_steps'];base=dict(data['model'])
    else:
        early=RUNS/('six-early-roles-without-m1-249' if without_m1 else 'six-early-roles-full-248')
        paths=[early/f'duration-{span}/discovery/training-data.pth' for span in (10,25,50)]
        deadline=time.monotonic()+14400
        while not all(path.exists() for path in paths):
            if time.monotonic()>deadline:raise TimeoutError('Early role collection incomplete.')
            time.sleep(15)
        datasets=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
        if not all(data.get('completed') for data in datasets):raise ValueError('Early role collection incomplete.')
        base=dict(datasets[0]['model']);steps=sum(data['simulation_steps'] for data in datasets)
    selections=[]
    for leaf in (4,16):
        fitted=out/f'leaf-{leaf}.pth'
        record=torch.load(fitted,map_location='cpu',weights_only=False) if fitted.exists() else None
        if record is None or record['cooperative_role_credit'].get('fit',{}).get('support_geometry')!='bidirectional-conditional-edges':
            credit=fit_local_role_effects(datasets,leaf=leaf,heads=5,trees=96)
            credit.update(temporal=temporal,initial_only=not temporal,max_changed=1 if temporal else 2)
            record=dict(base,cooperative_role_credit=credit,simulation_steps=base.get('simulation_steps',0)+steps)
            atomic_save(record,fitted)
            print(dict(phase='local_effect_fitted',without_M1=without_m1,temporal=temporal,**credit['fit']),flush=True)
        for confidence in (0.,.5,1.):
            policy=out/f'leaf-{leaf}-confidence-{confidence:g}.pth';result=policy.with_suffix('.json')
            model=dict(record,cooperative_role_credit=dict(record['cooperative_role_credit'],confidence=confidence))
            atomic_save(model,policy)
            evaluated=json.loads(result.read_text()) if result.exists() else None
            if (evaluated is None or evaluated.get('eligible_credit_executed') is False or
                    evaluated.get('support_geometry')!='bidirectional-conditional-edges'):
                rows=batched_credit_rows(policy,[(scene,None) for scene in SCENES])
                evaluated=dict(summary=d.summarize(rows),rows=rows,leaf=leaf,confidence=confidence,
                    eligible_credit_executed=True,support_geometry='bidirectional-conditional-edges')
                d.save_json(result,evaluated)
            selections.append(dict(path=str(policy),**evaluated))
            print(dict(phase='local_effect_development',without_M1=without_m1,temporal=temporal,
                leaf=leaf,confidence=confidence,success=evaluated['summary']['success']),flush=True)
    selected=max(selections,key=lambda x:(x['summary']['success'],x['summary']['capture']))
    model=torch.load(selected['path'],map_location='cpu',weights_only=False)
    atomic_save(model,out/'policy.pth')
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
    rows=batched_credit_rows(out/'policy.pth',tasks)
    reference=json.loads((source/'reference-paired-futures.json').read_text())
    result=dict(summary=d.summarize(rows),rows=rows,paired=compare(rows,reference['rows']),reference=reference['summary'],
        selection=[{key:value for key,value in candidate.items() if key!='rows'} for candidate in selections],
        intervention_rate=float(np.mean([bool(row.get('interventions')) for row in rows])),
        use='Internal M2 estimator comparison. Inherited backbone is not a complete M2-removal ablation.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='local_effect_complete',without_M1=without_m1,temporal=temporal,
        success=result['summary']['success'],paired=result['paired']),flush=True)


def run_early_role_probe(without_m1):
    """Check causal potential of early, sustained cooperative fusion choices."""
    from envs.TADgame import TADEnv
    initialize()
    source=RUNS/('six-temporal-terminal-without-m1-238' if without_m1 else 'six-temporal-terminal-full-237')
    out=RUNS/('six-early-roles-without-m1-249' if without_m1 else 'six-early-roles-full-248')
    out.mkdir(parents=True,exist_ok=True);scenes=d.bank(4193000000,6,((6,2.),(6,2.5)))
    programs=[];free=(-1.,)*6
    for scene in scenes:
        np.random.seed(scene[0]);env=TADEnv(6,protocol='paper-parameters-v1');env.reset(scene[2],noisy_agility=False)
        distances=np.linalg.norm(np.array([boat.pos for boat in env.defender_list])-env.attacker.pos,axis=-1)
        leaders=np.argsort(distances)[:2]
        choices=[free]
        for subject in range(6):
            for gate in (0.,.5,1.):
                proposal=list(free);proposal[subject]=gate;choices.append(tuple(proposal))
        for first in (0.,.5,1.):
            for second in (0.,.5,1.):
                proposal=list(free);proposal[int(leaders[0])]=first;proposal[int(leaders[1])]=second;choices.append(tuple(proposal))
        programs.append(choices)
    results=[]
    for duration in (10,25,50):
        folder=out/f'duration-{duration}';folder.mkdir(parents=True,exist_ok=True)
        model=torch.load(source/'reference.pth',map_location='cpu',weights_only=False)
        model.update(physical_intervention_span=duration,physical_intervention_reversal=False,
            conditional_commitment=True,preference_strength=0.)
        policy=folder/'reference.pth';atomic_save(model,policy)
        discovery=collect(policy,scenes,folder/'discovery',8,500,programs,'cuda:0',workers=3)
        selected=[];confirmation_programs=[]
        for roll in discovery['rollouts']:
            samples=role_terminal_samples(roll);base=samples[free]
            def rank(program):
                delta=samples[program]-base
                return (float(delta.mean()-.5*delta.std(ddof=1)/np.sqrt(len(delta))),float(delta.mean()),-sum(g>=0. for g in program))
            single=max((program for program in samples if sum(g>=0. for g in program)<=1),key=rank)
            coordinated=max(samples,key=rank)
            candidates=[free,single,coordinated]
            for subject,gate in enumerate(coordinated):
                if gate>=0.:
                    intermediate=list(free);intermediate[subject]=gate;candidates.append(tuple(intermediate))
            candidates=list(dict.fromkeys(candidates))
            # Even a selected no-intervention result has a connected physical
            # alternative so all endpoints remain represented in paired data.
            if len(candidates)==1:
                fallback=list(free);fallback[0]=.5;candidates.append(tuple(fallback))
            selected.append(dict(scene=roll['row']['seed'],single=single,coordinated=coordinated))
            confirmation_programs.append(candidates)
        confirmed=collect(policy,scenes,folder/'confirmation',8,508,confirmation_programs,'cuda:0',workers=3)
        rows=[]
        for roll,choice in zip(confirmed['rollouts'],selected):
            samples=role_terminal_samples(roll)
            rows.append(dict(seed=choice['scene'],reference=samples[free].tolist(),
                single=samples[tuple(choice['single'])].tolist(),coordinated=samples[tuple(choice['coordinated'])].tolist(),
                selected_single=choice['single'],selected_coordinated=choice['coordinated']))
        result=dict(duration_steps=duration,rows=rows,simulation_steps=discovery['simulation_steps']+confirmed['simulation_steps'])
        for method in ('single','coordinated'):
            delta=np.array([np.array(row[method])-np.array(row['reference']) for row in rows])
            result[method]=dict(success=float(np.mean([row[method] for row in rows])),gain=float(delta.mean()),
                rescues=int((delta>0).sum()),harms=int((delta<0).sum()))
        result['reference']=float(np.mean([row['reference'] for row in rows]));results.append(result)
        d.save_json(out/'comparison.json',dict(variants=results,
            use='Causal development probe: twelve fresh layouts, eight selection futures and eight disjoint confirmation futures. Selection is not a deployable policy.'))
        print(dict(phase='early_role_probe',without_M1=without_m1,duration_steps=duration,reference=result['reference'],
            single=result['single'],coordinated=result['coordinated']),flush=True)


def continued_credit_shard(task):
    policy,roots,output,future_start=task
    data=d.collect_fusion_td_pairs(policy,roots,output,repetitions=8,horizon=400,batch_size=128,
        intervention_steps=10,future_start=future_start,resume_conditional=True)
    if any(pair[side]['success'] is None for pair in data['pairs'] for side in ('before','after')):
        raise RuntimeError('Conditional policy iteration requires actual task terminals.')
    return str(Path(output)/'training-data.pth')


def grid_credit_shard(task):
    policy,roots,output=task
    data=d.collect_fusion_td_pairs(policy,roots,output,repetitions=4,horizon=400,batch_size=128,
        intervention_steps=10,future_start=1900,intervention_mode='grid')
    if any(pair[side]['success'] is None for pair in data['pairs'] for side in ('before','after')):
        raise RuntimeError('The conditional gate grid must be labelled by actual task terminals.')
    return str(Path(output)/'training-data.pth')


def run_grid_credit_learning(without_m1):
    """Learn the complete scalar fusion intervention, including interior gates."""
    from conditional_fusion_value import fit_local_role_effects
    initialize()
    out=RUNS/('six-grid-credit-without-m1-275' if without_m1 else 'six-grid-credit-full-274')
    out.mkdir(parents=True,exist_ok=True)
    base=torch.load(RUNS/'six-trained-interaction-185/policy.pth',map_location='cpu',weights_only=False)
    base.update(mechanism_ablation=(without_m1,False,without_m1),continuous_gate_learning=True,
        conditional_commitment=True,preference_strength=0.,interception_weight=.5)
    teacher=out/'teacher.pth';atomic_save(base,teacher)
    scenes=d.bank(4221000000,48,((6,2.),(6,2.5)))
    tasks=[(scene,None) for scene in scenes]
    work=[(str(teacher),tasks[rank::4],str(out/'factual'/str(rank)/'training-data.pth')) for rank in range(4)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(factual_credit_shard,work))
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    data=dict(model=base,rollouts=[roll for part in parts for roll in part['rollouts']])
    roots=d.feedback_role_roots([data],lookbacks=(2.,4.,8.,24.),decision_period=10)
    work=[(str(teacher),roots[rank::4],str(out/'paired'/str(rank))) for rank in range(4)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(grid_credit_shard,work))
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    data['paired_durations']=[dict(duration=10,pairs=[pair for part in parts for pair in part['pairs']])]
    total=sum(part['simulation_steps'] for part in parts)+sum(roll['simulation_steps'] for roll in data['rollouts'])
    data['simulation_steps']=total;atomic_save(data,out/'training-data.pth')
    print(dict(phase='grid_credit_collected',without_M1=without_m1,roots=len(roots),simulation_steps=total),flush=True)
    datasets=local_temporal_datasets(data);fitted=[]
    for horizon in (0.,20.):
        credit=fit_local_role_effects(datasets,heads=5,trees=160,leaf=8,physical_features=True,
            threat_features=True,uniform_effect_weight=True,return_horizon=horizon)
        credit.update(temporal=True,full_grid=True,initial_only=False,max_changed=1,max_interventions=0,wait_span=True)
        model=dict(base,cooperative_role_credit=credit,simulation_steps=total,counterfactual_training_steps=total)
        path=out/f'value-{horizon:g}.pth';atomic_save(model,path)
        fitted.append(dict(path=str(path),horizon=horizon,fit=credit['fit']))
    d.save_json(out/'fits.json',fitted)
    selected=max(fitted,key=lambda record:np.mean([fit['held_layout_selected_gain'] for fit in record['fit']['fits']]))
    template=torch.load(selected['path'],map_location='cpu',weights_only=False);evaluated=[]
    for confidence,tolerance in ((0.,None),(.5,None),*([] if without_m1 else [(0.,0.),(.5,0.)])):
        label=f'confidence-{confidence:g}-risk-{tolerance}'
        model=dict(template,cooperative_role_credit=dict(template['cooperative_role_credit'],
            confidence=confidence,risk_tolerance=tolerance))
        path=out/(label+'.pth');atomic_save(model,path)
        rows=batched_credit_rows(path,[(scene,None) for scene in SCENES])
        result=dict(label=label,path=str(path),summary=d.summarize(rows),rows=rows)
        d.save_json(path.with_suffix('.json'),result);evaluated.append(result)
        print(dict(phase='grid_credit_selected',without_M1=without_m1,label=label,success=result['summary']['success']),flush=True)
    best=max(evaluated,key=lambda row:(row['summary']['success'],row['summary']['capture']))
    atomic_save(torch.load(best['path'],map_location='cpu',weights_only=False),out/'policy.pth')
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
    reference_rows=batched_credit_rows(teacher,tasks)
    rows=batched_credit_rows(out/'policy.pth',tasks)
    result=dict(summary=d.summarize(rows),rows=rows,reference=d.summarize(reference_rows),reference_rows=reference_rows,
        paired=compare(rows,reference_rows),simulation_steps=total,training_defenders=6,
        selection={key:value for key,value in best.items() if key!='rows'},
        use='Complete physical fusion grid, same task and six-vessel training; development comparison.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='grid_credit_complete',without_M1=without_m1,summary=result['summary'],paired=result['paired']),flush=True)


def run_decoupled_forecast_learning():
    """Separate target extrapolation from vessel encounter prediction in M1."""
    initialize();out=RUNS/'six-decoupled-forecast-297';out.mkdir(parents=True,exist_ok=True)
    base=torch.load(RUNS/'six-trained-interaction-185/policy.pth',map_location='cpu',weights_only=False)
    base.update(mechanism_ablation=(False,True,False),interception_weight=.5,risk_horizon=3.2)
    variants=[]
    settings=(('long-safety-short-reach',1.6,0.),('soft-tail-short-reach',1.6,.8),
        ('slow-tail-short-reach',1.6,1.6),('soft-tail-long-reach',0.,.8))
    for label,reach,discount in settings:
        path=out/(label+'.pth');target=path.with_suffix('.json')
        atomic_save(dict(base,interception_horizon=reach,prediction_discount=discount),path)
        if target.exists():result=json.loads(target.read_text())
        else:
            rows=batched_credit_rows(path,[(scene,None) for scene in SCENES])
            result=dict(label=label,path=str(path),summary=d.summarize(rows),rows=rows)
            d.save_json(target,result)
        variants.append(result)
        print(dict(phase='decoupled_forecast_development',label=label,summary=result['summary']),flush=True)
    d.save_json(out/'development.json',dict(variants=variants))
    tasks=[(scene,None if repeat==0 else int(np.random.SeedSequence(
        [scene[0],repeat,20261014]).generate_state(1)[0])) for repeat in range(4) for scene in SCENES]
    source=RUNS/'six-grid-credit-full-274';reference_path=source/'reference-paired-futures.json'
    if reference_path.exists():reference=json.loads(reference_path.read_text())
    elif (source/'comparison.json').exists():
        data=json.loads((source/'comparison.json').read_text())
        reference=dict(summary=data['reference'],rows=data['reference_rows'])
    else:
        rows=batched_credit_rows(source/'teacher.pth',tasks);reference=dict(summary=d.summarize(rows),rows=rows)
    selected=[]
    for variant in sorted(variants,key=lambda item:(item['summary']['success'],item['summary']['capture']),reverse=True)[:2]:
        target=out/(variant['label']+'-paired-futures.json')
        if target.exists():result=json.loads(target.read_text())
        else:
            rows=batched_credit_rows(variant['path'],tasks)
            result=dict(label=variant['label'],path=variant['path'],summary=d.summarize(rows),rows=rows,
                paired=compare(rows,reference['rows']),reference=reference['summary'])
            d.save_json(target,result)
        selected.append(result)
        print(dict(phase='decoupled_forecast_comparison',label=result['label'],
            summary=result['summary'],paired=result['paired']),flush=True)
    best=max(selected,key=lambda item:(item['summary']['success'],item['summary']['capture']))
    atomic_save(torch.load(best['path'],map_location='cpu',weights_only=False),out/'policy.pth')
    d.save_json(out/'comparison.json',dict(variants=selected,selected=best['label'],training_defenders=6,
        use='M1 internal prediction/scoring optimization; M2, its learned parameters and support functions are disabled.'))


def replicated_credit_shard(task):
    policy,roots,output=task
    d.collect_fusion_td_pairs(policy,roots,output,repetitions=16,horizon=400,batch_size=128,
        intervention_steps=10,future_start=3100,intervention_mode='grid')
    return str(Path(output)/'training-data.pth')


def replicate_uncertain_credit(data,teacher,out):
    """Spend fresh paired futures on uncertain within-vessel rankings."""
    destination=out/'training-data.pth'
    if destination.exists():return torch.load(destination,map_location='cpu',weights_only=False)
    groups={}
    for block in data['paired_durations']:
        for pair in block['pairs']:
            key=(pair['scene'][0],pair['root_id'],pair['subject'])
            gate=float(pair['right'][pair['subject'],2])
            groups.setdefault(key,{}).setdefault(gate,[]).append(pair['after']['success']-pair['before']['success'])
    rng=np.random.default_rng(42);selected={}
    for key,arms in groups.items():
        score=max(np.mean(values) for values in arms.values())+.25*np.mean(
            [np.mean(np.asarray(values)!=0) for values in arms.values()])
        order=(score,float(rng.random()))
        if key[0] not in selected or order>selected[key[0]][0]:selected[key[0]]=(order,key)
    roots=d.feedback_role_roots([data],lookbacks=(2.,4.,8.,24.),decision_period=10)
    chosen=[];keys=set()
    for root in roots:
        key=selected[root['scene'][0]][1]
        if root['id']==key[1]:
            chosen.append(dict(root,credit_subject=key[2]));keys.add(key)
    if len(chosen)!=len(selected):raise RuntimeError('Replication roots must match their observed conditional contrasts.')
    work=[(str(teacher),chosen[rank::4],str(out/'paired'/str(rank))) for rank in range(4)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(replicated_credit_shard,work))
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    fresh=[pair for part in parts for pair in part['pairs']]
    if any(pair[side]['success'] is None for pair in fresh for side in ('before','after')):
        raise RuntimeError('Replication must observe actual full-task terminals.')
    # Selection used only the earlier futures. Replace their selected-arm
    # labels with fresh outcomes, so optimistic selection noise is not reused.
    retained=[pair for block in data['paired_durations'] for pair in block['pairs']
        if (pair['scene'][0],pair['root_id'],pair['subject']) not in keys]
    refined=dict(data,paired_durations=[dict(duration=10,pairs=retained+fresh)],
        simulation_steps=data['simulation_steps']+sum(part['simulation_steps'] for part in parts),
        independent_replication=dict(roots=len(chosen),futures=16,future_start=3100,
            additional_simulation_steps=sum(part['simulation_steps'] for part in parts)))
    atomic_save(refined,destination)
    print(dict(phase='conditional_replication_complete',**refined['independent_replication']),flush=True)
    return refined


def run_contrast_credit_learning(without_m1,refined=False):
    """Learn matched, complete within-vessel contrasts without a second predictor."""
    from conditional_fusion_value import fit_local_role_effects
    initialize()
    source=RUNS/('six-grid-credit-without-m1-275' if without_m1 else 'six-grid-credit-full-274')
    out=RUNS/('six-closed-contrast-without-m1-294' if without_m1 else 'six-closed-contrast-full-293')
    if refined:out=RUNS/('six-replicated-contrast-without-m1-296' if without_m1 else 'six-replicated-contrast-full-295')
    out.mkdir(parents=True,exist_ok=True)
    data=torch.load(source/'training-data.pth',map_location='cpu',weights_only=False)
    if refined:data=replicate_uncertain_credit(data,source/'teacher.pth',out)
    base=data['model']
    # Forecast coordinates belong solely to M1. In its training ablation they
    # are zero, and M2 has no separate analytic actuator/response predictor.
    if not without_m1:
        teacher=source/'teacher.pth'
        actor=d.joint_load(str(teacher),teacher.stat().st_mtime_ns).to('cuda:0')
        packets={}
        for block in data['paired_durations']:
            for pair in block['pairs']:packets.setdefault((pair['scene'][0],pair['root_id']),pair['packet'])
        records=list(packets.values())
        with torch.no_grad():
            for begin in range(0,len(records),64):
                current=records[begin:begin+64]
                packet={key:torch.as_tensor(np.stack([row[key] for row in current]),device='cuda:0') for key in current[0]}
                proposal=actor.proposal_distribution(packet['obs'])[0].tanh()
                risk=actor.candidate_risk(packet,proposal).cpu().numpy()
                reach=packet['_candidate_interception'].cpu().numpy()
                for index,record in enumerate(current):
                    record['_joint_candidate_risk']=risk[index]
                    record['_candidate_interception']=reach[index]
        actor.cpu()
    results=[]
    for complete in ((True,) if refined else (False,True)):
        label='closed' if complete else 'star'
        path=out/(label+'-fitted.pth')
        if path.exists():model=torch.load(path,map_location='cpu',weights_only=False)
        else:
            credit=fit_local_role_effects(local_temporal_datasets(data,complete_contrasts=complete),
                heads=3,trees=160,leaf=8,physical_features=False,threat_features=True,
                prediction_features=True,uniform_effect_weight=True,failure_horizon=2. if without_m1 else 8.,
                future_bootstrap=refined)
            credit.update(temporal=True,full_grid=True,initial_only=False,max_changed=1,
                max_interventions=0,wait_span=True)
            credit['fit']['complete_action_contrasts']=complete
            model=dict(base,cooperative_role_credit=credit,simulation_steps=data['simulation_steps'],
                conditional_learning='complete-paired-one-vessel-contrasts' if complete else 'free-to-fixed-contrasts')
            atomic_save(model,path)
        print(dict(phase='contrast_credit_fit',without_M1=without_m1,label=label,
            fit=model['cooperative_role_credit']['fit']),flush=True)
        candidates=[]
        for confidence,margin in ((0.,.001),(0.,.005),(.5,.005)):
            trial=out/f'{label}-confidence-{confidence:g}-margin-{margin:g}.pth'
            candidate=dict(model,cooperative_role_credit=dict(model['cooperative_role_credit'],
                confidence=confidence,margin=margin))
            atomic_save(candidate,trial);target=trial.with_suffix('.json')
            if target.exists():record=json.loads(target.read_text())
            else:
                rows=batched_credit_rows(trial,[(scene,None) for scene in SCENES])
                record=dict(summary=d.summarize(rows),rows=rows,path=str(trial))
                d.save_json(target,record)
            candidates.append(record)
            print(dict(phase='contrast_credit_development',without_M1=without_m1,label=trial.stem,
                summary=record['summary']),flush=True)
        best=max(candidates,key=lambda item:(item['summary']['success'],item['summary']['capture']))
        path=out/(label+'-policy.pth');atomic_save(torch.load(best['path'],map_location='cpu',weights_only=False),path)
        tasks=[(scene,None if repeat==0 else int(np.random.SeedSequence(
            [scene[0],repeat,20261014]).generate_state(1)[0])) for repeat in range(4) for scene in SCENES]
        target=out/(label+'-paired-futures.json')
        if target.exists():result=json.loads(target.read_text())
        else:
            rows=batched_credit_rows(path,tasks)
            result=dict(label=label,summary=d.summarize(rows),rows=rows,
                selection={key:value for key,value in best.items() if key!='rows'})
            d.save_json(target,result)
        results.append(result)
    reference=json.loads((source/'comparison.json').read_text())
    result=dict(variants=results,closed_minus_star=compare(results[1]['rows'],results[0]['rows']) if len(results)>1 else None,
        closed_minus_reference=compare(results[-1]['rows'],reference['reference_rows']),
        reference=reference['reference'],training_defenders=6,training_seed=42,
        simulation_steps=data['simulation_steps'],independent_replication=data.get('independent_replication'),
        use='Same observed arms and future noise; additional contrasts introduce no new terminal samples. M2 contains no independent analytic response predictor.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='contrast_credit_complete',without_M1=without_m1,
        closed_minus_reference=result['closed_minus_reference'],closed_minus_star=result['closed_minus_star']),flush=True)


def finite_hazard_shard(task):
    policy,roots,output=task
    d.collect_fusion_td_pairs(policy,roots,output,repetitions=4,horizon=10,batch_size=128,
        intervention_steps=10,future_start=2300,intervention_mode='grid')
    return str(Path(output)/'training-data.pth')


def run_finite_hazard_learning(without_m1,defenders=6):
    """Learn failure credit over the actual intervention's two-second support."""
    from conditional_fusion_value import fit_local_role_effects
    initialize()
    out=RUNS/(('six-finite-hazard-without-m1-292' if without_m1 else 'six-finite-hazard-full-291') if defenders==6 else
        f'fleet{defenders}-finite-hazard-{"without-m1" if without_m1 else "full"}-seed42')
    out.mkdir(parents=True,exist_ok=True)
    base=torch.load(RUNS/'six-trained-interaction-185/policy.pth',map_location='cpu',weights_only=False)
    if defenders!=6:
        import hashlib
        source=REPO/'train/experiments'/f'fleet{defenders}-paper-source-arboids-seed42-20261015'/'budget-275000.pth'
        deadline=time.monotonic()+48*3600
        while not source.exists():
            if time.monotonic()>deadline:raise TimeoutError('The matched native-fleet proposal training did not finish.')
            time.sleep(30)
        time.sleep(2)
        proposal=torch.load(source,map_location='cpu',weights_only=True)
        base['actor']=dict(base['actor'],**{'base.'+key:value for key,value in proposal.items()})
        base.update(source=str(source),source_training_defenders=defenders,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            proposal_initialization_sha256=d.proposal_parameter_digest(proposal),
            neural_training_simulation_steps=275000,simulation_steps=0,
            initialization=f'{defenders}-vessel source proposal; fresh interaction and conditional-credit modules')
    base.update(mechanism_ablation=(without_m1,False,without_m1),continuous_gate_learning=True,
        conditional_commitment=True,preference_strength=0.,interception_weight=.5,risk_horizon=1.6)
    teacher=out/'teacher.pth';atomic_save(base,teacher)
    development=(SCENES if defenders==6 else d.bank(4250000000+defenders*100000,32,((defenders,2.),(defenders,2.5))))
    scenes=d.bank(4235000000+(0 if defenders==6 else defenders*100000),48,((defenders,2.),(defenders,2.5)))
    tasks=[(scene,None) for scene in scenes]
    work=[(str(teacher),tasks[rank::4],str(out/'factual'/str(rank)/'training-data.pth')) for rank in range(4)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(factual_credit_shard,work))
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    data=dict(model=base,rollouts=[roll for part in parts for roll in part['rollouts']])
    available=d.feedback_role_roots([data],lookbacks=tuple(np.arange(2.,62.,2.)),decision_period=10)
    rng=np.random.default_rng(42);roots=[]
    for scene in scenes:
        candidates=sorted([root for root in available if root['scene'][0]==scene[0]],
            key=lambda root:root['environment'].Current_T)
        if not candidates:continue
        roots.append(candidates[-1])
        if len(candidates)>1:
            chosen=rng.choice(len(candidates)-1,min(3,len(candidates)-1),replace=False)
            roots.extend(candidates[int(index)] for index in chosen)
    work=[(str(teacher),roots[rank::4],str(out/'paired'/str(rank))) for rank in range(4)]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(finite_hazard_shard,work))
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    data['paired_durations']=[dict(duration=10,pairs=[pair for part in parts for pair in part['pairs']])]
    total=sum(part['simulation_steps'] for part in parts)+sum(roll['simulation_steps'] for roll in data['rollouts'])
    data.update(simulation_steps=total,failure_deadline=2.,
        use='Two-second observed failure costs. Surviving branch endpoints remain censored, not successful terminals.')
    atomic_save(data,out/'training-data.pth')
    credit=fit_local_role_effects(local_temporal_datasets(data),heads=5,trees=160,leaf=8,
        physical_features=True,threat_features=True,uniform_effect_weight=True,failure_deadline=2.)
    credit.update(temporal=True,full_grid=True,initial_only=False,max_changed=1,max_interventions=0,wait_span=True)
    if defenders!=6:
        contexts=torch.as_tensor(credit['contexts'])
        credit['support_weights']=(contexts.std(0,unbiased=False)>1e-6).float()
        credit['fit']=dict(credit['fit'],support_metric='observed-variation-only')
    d.save_json(out/'fit.json',credit['fit']);variants=[]
    for confidence,margin in ((0.,.005),(.5,.01),(1.,.02)):
        path=out/f'confidence-{confidence:g}-margin-{margin:g}.pth'
        atomic_save(dict(base,cooperative_role_credit=dict(credit,confidence=confidence,margin=margin),
            simulation_steps=total,conditional_learning='finite-absorbing-failure-credit',failure_deadline=2.),path)
        rows=batched_credit_rows(path,[(scene,None) for scene in development])
        record=dict(path=str(path),summary=d.summarize(rows),rows=rows,confidence=confidence,margin=margin)
        d.save_json(path.with_suffix('.json'),record);variants.append(record)
        print(dict(phase='finite_hazard_development',without_M1=without_m1,summary=record['summary'],
            confidence=confidence,margin=margin),flush=True)
    best=max(variants,key=lambda row:(row['summary']['success'],row['summary']['capture']))
    atomic_save(torch.load(best['path'],map_location='cpu',weights_only=False),out/'policy.pth')
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in development]
    reference=batched_credit_rows(teacher,tasks);rows=batched_credit_rows(out/'policy.pth',tasks)
    result=dict(summary=d.summarize(rows),rows=rows,reference=d.summarize(reference),reference_rows=reference,
        paired=compare(rows,reference),fit=credit['fit'],simulation_steps=total,training_defenders=defenders,
        selection={key:value for key,value in best.items() if key!='rows'},
        use='Finite intervention credit, evaluated on the full unchanged sixty-second task.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='finite_hazard_complete',without_M1=without_m1,summary=result['summary'],paired=result['paired']),flush=True)


def run_hazard_credit_learning(without_m1,prediction_features=False):
    """Resolve action credit at the time scale of actual absorbing failures."""
    from conditional_fusion_value import fit_local_role_effects
    initialize()
    source=RUNS/('six-grid-credit-without-m1-275' if without_m1 else 'six-grid-credit-full-274')
    out=RUNS/('six-hazard-credit-without-m1-285' if without_m1 else 'six-hazard-credit-full-284')
    if prediction_features:out=RUNS/('six-forecast-credit-without-m1-287' if without_m1 else 'six-forecast-credit-full-286')
    out.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+10800
    while not (source/'training-data.pth').exists():
        if time.monotonic()>deadline:raise TimeoutError('Complete fusion intervention collection has not finished.')
        time.sleep(15)
    data=torch.load(source/'training-data.pth',map_location='cpu',weights_only=False)
    if prediction_features and not without_m1:
        teacher=source/'teacher.pth';actor=d.joint_load(str(teacher),teacher.stat().st_mtime_ns).to('cuda:0')
        packets={}
        for block in data['paired_durations']:
            for pair in block['pairs']:packets.setdefault((pair['scene'][0],pair['root_id']),pair['packet'])
        records=list(packets.values())
        with torch.no_grad():
            for begin in range(0,len(records),64):
                current=records[begin:begin+64]
                packet={key:torch.as_tensor(np.stack([row[key] for row in current]),device='cuda:0') for key in current[0]}
                proposal=actor.proposal_distribution(packet['obs'])[0].tanh()
                risk=actor.candidate_risk(packet,proposal).cpu().numpy()
                interception=packet.get('_candidate_interception')
                interception=interception.cpu().numpy() if interception is not None else None
                for index,record in enumerate(current):
                    record['_joint_candidate_risk']=risk[index]
                    if interception is not None:record['_candidate_interception']=interception[index]
        actor.cpu()
    datasets=local_temporal_datasets(data);base=data['model'];fitted=[]
    for horizon in ((0.,4.,8.) if prediction_features else (2.,4.,8.)):
        path=out/f'failure-horizon-{horizon:g}.pth'
        if path.exists():model=torch.load(path,map_location='cpu',weights_only=False)
        else:
            credit=fit_local_role_effects(datasets,heads=3,trees=160,leaf=8,physical_features=True,
                threat_features=True,uniform_effect_weight=True,failure_horizon=horizon,prediction_features=prediction_features)
            credit.update(temporal=True,full_grid=True,initial_only=False,max_changed=1,
                max_interventions=0,wait_span=True)
            model=dict(base,cooperative_role_credit=credit,simulation_steps=data['simulation_steps'],
                conditional_learning='paired-discounted-absorbing-failure-cost')
            atomic_save(model,path)
        fitted.append(dict(horizon=horizon,path=str(path),fit=model['cooperative_role_credit']['fit']))
        d.save_json(out/'fits.json',fitted)
        print(dict(phase='hazard_credit_fit',without_M1=without_m1,**fitted[-1]),flush=True)
    selected=max(fitted,key=lambda row:np.mean([fit['held_layout_selected_gain'] for fit in row['fit']['fits']]))
    template=torch.load(selected['path'],map_location='cpu',weights_only=False);candidates=[]
    for confidence,margin in ((0.,.005),(.5,.01),(1.,.02)):
        label=f'confidence-{confidence:g}-margin-{margin:g}'
        path=out/(label+'.pth');target=path.with_suffix('.json')
        model=dict(template,cooperative_role_credit=dict(template['cooperative_role_credit'],confidence=confidence,margin=margin))
        atomic_save(model,path)
        if target.exists():result=json.loads(target.read_text())
        else:
            rows=batched_credit_rows(path,[(scene,None) for scene in SCENES])
            result=dict(label=label,path=str(path),summary=d.summarize(rows),rows=rows)
            d.save_json(target,result)
        candidates.append(result)
        print(dict(phase='hazard_credit_development',without_M1=without_m1,label=label,success=result['summary']['success']),flush=True)
    best=max(candidates,key=lambda row:(row['summary']['success'],row['summary']['capture']))
    path=out/'policy.pth';atomic_save(torch.load(best['path'],map_location='cpu',weights_only=False),path)
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
    rows=batched_credit_rows(path,tasks)
    reference_path=source/'reference-paired-futures.json'
    if reference_path.exists():reference=json.loads(reference_path.read_text())
    elif (source/'comparison.json').exists():
        compared=json.loads((source/'comparison.json').read_text())
        reference=dict(summary=compared['reference'],rows=compared['reference_rows'])
    else:
        reference_rows=batched_credit_rows(source/'teacher.pth',tasks)
        reference=dict(summary=d.summarize(reference_rows),rows=reference_rows)
        d.save_json(reference_path,reference)
    result=dict(summary=d.summarize(rows),rows=rows,reference=reference['summary'],paired=compare(rows,reference['rows']),
        fits=fitted,selection={key:value for key,value in best.items() if key!='rows'},
        training_defenders=6,training_seed=42,simulation_steps=data['simulation_steps'],
        use='Conditional failure timing from the same complete task terminals; environment rewards and success definition unchanged.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='hazard_credit_complete',without_M1=without_m1,summary=result['summary'],paired=result['paired']),flush=True)


def run_integrable_credit_learning(without_m1):
    """Compare paired outcome differences with matched ordinary-return fitting."""
    from conditional_fusion_value import fit_local_role_effects
    initialize()
    source=RUNS/('six-grid-credit-without-m1-275' if without_m1 else 'six-grid-credit-full-274')
    out=RUNS/('six-integrable-credit-without-m1-279' if without_m1 else 'six-integrable-credit-full-278')
    out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+10800
    while not (source/'training-data.pth').exists():
        if time.monotonic()>deadline:raise TimeoutError('Complete fusion intervention collection has not finished.')
        time.sleep(15)
    data=torch.load(source/'training-data.pth',map_location='cpu',weights_only=False)
    datasets=local_temporal_datasets(data);base=data['model'];fitted=[]
    for weight in (0.,4.,16.):
        path=out/f'outcome-weight-{weight:g}.pth'
        if path.exists():model=torch.load(path,map_location='cpu',weights_only=False)
        else:
            credit=fit_local_role_effects(datasets,heads=3,physical_features=True,threat_features=True,
                uniform_effect_weight=True,conditional_outcome_weight=weight)
            credit.update(temporal=True,full_grid=True,initial_only=False,max_changed=1,
                max_interventions=0,wait_span=True)
            model=dict(base,cooperative_role_credit=credit,simulation_steps=data['simulation_steps'],
                conditional_learning='anchored-paired-outcome-differences' if weight else 'ordinary-terminal-outcomes')
            atomic_save(model,path)
        fitted.append(dict(weight=weight,path=str(path),fit=model['cooperative_role_credit']['fit']))
        d.save_json(out/'fits.json',fitted)
        print(dict(phase='integrable_credit_fit',without_M1=without_m1,**fitted[-1]),flush=True)
    paired=max(fitted[1:],key=lambda row:np.mean([fit['held_layout_selected_gain'] for fit in row['fit']['fits']]))
    selected=[]
    for family,record in [('ordinary',fitted[0]),('paired',paired)]:
        template=torch.load(record['path'],map_location='cpu',weights_only=False);candidates=[]
        for confidence,margin in ((.5,.01),(1.,.05)):
            label=f'{family}-confidence-{confidence:g}-margin-{margin:g}'
            path=out/(label+'.pth');target=path.with_suffix('.json')
            model=dict(template,cooperative_role_credit=dict(template['cooperative_role_credit'],
                confidence=confidence,margin=margin))
            atomic_save(model,path)
            if target.exists():result=json.loads(target.read_text())
            else:
                rows=batched_credit_rows(path,[(scene,None) for scene in SCENES])
                result=dict(label=label,path=str(path),summary=d.summarize(rows),rows=rows)
                d.save_json(target,result)
            candidates.append(result)
            print(dict(phase='integrable_credit_development',without_M1=without_m1,label=label,
                success=result['summary']['success']),flush=True)
        best=max(candidates,key=lambda row:(row['summary']['success'],row['summary']['capture']))
        path=out/(family+'-policy.pth');atomic_save(torch.load(best['path'],map_location='cpu',weights_only=False),path)
        tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
            [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
        rows=batched_credit_rows(path,tasks)
        selected.append(dict(family=family,summary=d.summarize(rows),rows=rows,
            selection={key:value for key,value in best.items() if key!='rows'}))
        d.save_json(out/(family+'-paired-futures.json'),selected[-1])
    result=dict(variants=selected,paired_minus_ordinary=compare(selected[1]['rows'],selected[0]['rows']),
        fits=fitted,training_defenders=6,training_seed=42,simulation_steps=data['simulation_steps'],
        use='Same architecture, paired continuations and optimization steps; only the conditional difference loss changes.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='integrable_credit_complete',without_M1=without_m1,
        paired_minus_ordinary=result['paired_minus_ordinary']),flush=True)


def run_threat_credit_learning(without_m1,temporal_return=False,kernel=False):
    """Fit conditional credit with equal root weight and encounter context."""
    from conditional_fusion_value import fit_local_role_effects
    initialize()
    source=RUNS/('six-continued-credit-without-m1-262' if without_m1 else 'six-continued-credit-full-261')
    out=RUNS/(('six-survival-credit-without-m1-269' if without_m1 else 'six-survival-credit-full-268')
        if temporal_return else ('six-threat-credit-without-m1-267' if without_m1 else 'six-threat-credit-full-266'))
    if kernel:out=RUNS/('six-kernel-credit-without-m1-271' if without_m1 else 'six-kernel-credit-full-270')
    out.mkdir(parents=True,exist_ok=True)
    observed=[torch.load(source/f'iteration-{iteration}/training-data.pth',map_location='cpu',weights_only=False)
        for iteration in (0,1)]
    datasets=[dataset for record in observed for dataset in local_temporal_datasets(record)]
    base=torch.load(source/'iteration-0/teacher.pth',map_location='cpu',weights_only=False)
    total=sum(record['simulation_steps'] for record in observed)
    fitted=[]
    settings=(('terminal-20',True,False,8,20.,0.),('terminal-60',True,False,8,60.,0.)) if temporal_return else (
        ('equal-root',False,False,8,0.,0.),('encounter',True,False,8,0.,0.),('encounter-boosted',True,True,16,0.,0.))
    if kernel:settings=tuple((f'kernel-{bandwidth:g}',True,False,8,0.,bandwidth) for bandwidth in (.5,1.,2.))
    for label,threat,boosted,leaf,horizon,bandwidth in settings:
        path=out/(label+'.pth')
        if path.exists():model=torch.load(path,map_location='cpu',weights_only=False)
        else:
            credit=fit_local_role_effects(datasets,heads=3,trees=160,leaf=leaf,physical_features=True,
                threat_features=threat,uniform_effect_weight=True,boosted=boosted,return_horizon=horizon,
                kernel_bandwidth=bandwidth)
            credit.update(temporal=True,initial_only=False,max_changed=1,max_interventions=0,wait_span=True,margin=0.)
            model=dict(base,cooperative_role_credit=credit,counterfactual_training_steps=total,
                simulation_steps=total,conditional_learning='paired-root-balanced-encounter-value')
            atomic_save(model,path)
        fitted.append(dict(label=label,path=str(path),fit=model['cooperative_role_credit']['fit']))
        d.save_json(out/'fits.json',fitted)
        print(dict(phase='threat_credit_fit',without_M1=without_m1,**fitted[-1]),flush=True)
    selected=max(fitted,key=lambda record:np.mean([fit['held_layout_selected_gain'] for fit in record['fit']['fits']]))
    model=torch.load(selected['path'],map_location='cpu',weights_only=False)
    candidates=[]
    for confidence in (0.,.5):
        path=out/f'confidence-{confidence:g}.pth';target=path.with_suffix('.json')
        candidate=dict(model,cooperative_role_credit=dict(model['cooperative_role_credit'],confidence=confidence))
        atomic_save(candidate,path)
        if target.exists():result=json.loads(target.read_text())
        else:
            rows=batched_credit_rows(path,[(scene,None) for scene in SCENES])
            result=dict(summary=d.summarize(rows),rows=rows,confidence=confidence,path=str(path))
            d.save_json(target,result)
        candidates.append(result)
        print(dict(phase='threat_credit_selection',without_M1=without_m1,confidence=confidence,
            success=result['summary']['success']),flush=True)
    best=max(candidates,key=lambda row:(row['summary']['success'],row['summary']['capture']))
    atomic_save(torch.load(best['path'],map_location='cpu',weights_only=False),out/'policy.pth')
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
    rows=batched_credit_rows(out/'policy.pth',tasks)
    reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
    result=dict(summary=d.summarize(rows),rows=rows,paired=compare(rows,reference['rows']),reference=reference['summary'],
        selection={key:value for key,value in best.items() if key!='rows'},fits=fitted,
        simulation_steps=total,training_defenders=6,training_seed=42,
        use='Development comparison using the same on-policy paired samples; no new layouts or future labels were added for this representation change.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='threat_credit_complete',without_M1=without_m1,success=result['summary']['success'],paired=result['paired']),flush=True)


def run_role_persistence_probe(without_m1):
    """Test whether persistence, reversal, or an interior mixture causes gain."""
    initialize()
    source=RUNS/('six-continued-credit-without-m1-262' if without_m1 else 'six-continued-credit-full-261')
    out=RUNS/('six-role-persistence-without-m1-264' if without_m1 else 'six-role-persistence-full-263')
    out.mkdir(parents=True,exist_ok=True)
    parts=[torch.load(source/f'iteration-0/factual/{rank}/training-data.pth',map_location='cpu',weights_only=False)
        for rank in range(3)]
    data=dict(model=parts[0]['model'],rollouts=[roll for part in parts for roll in part['rollouts']])
    roots=d.feedback_role_roots([data],lookbacks=(6.,12.,24.),decision_period=10)
    selected=[]
    for agility in (2.,2.5):
        scenes=sorted({root['scene'][0] for root in roots if root['scene'][2]==agility})[:6]
        for scene in scenes:
            candidates=[root for root in roots if root['scene'][0]==scene]
            selected.append(candidates[len(candidates)//2])
    policy=source/'iteration-0/teacher.pth';results=[]
    for mode in ('reverse','maintain','blend'):
        observed=d.collect_fusion_td_pairs(str(policy),selected,out/mode,repetitions=8,horizon=400,
            batch_size=128,intervention_steps=10,future_start=1600,intervention_mode=mode)
        grouped={}
        for pair in observed['pairs']:
            key=(pair['scene'][0],pair['root_id'])
            grouped.setdefault(key,{}).setdefault(pair['subject'],[]).append(pair)
        rows=[]
        for (scene,root_id),alternatives in grouped.items():
            ordered={subject:sorted(pairs,key=lambda pair:pair['future_repetition'])
                for subject,pairs in alternatives.items()}
            def gain(subject):
                return float(np.mean([p['after']['success']-p['before']['success'] for p in ordered[subject][:4]]))
            subject=max(ordered,key=gain);use=gain(subject)>0.
            for pair in ordered[subject][4:]:
                rows.append(dict(seed=scene,root_id=root_id,subject=subject if use else None,
                    repetition=pair['future_repetition'],reference=pair['before']['success'],
                    selected=pair['after']['success'] if use else pair['before']['success']))
        delta=np.array([row['selected']-row['reference'] for row in rows])
        result=dict(mode=mode,roots=len(selected),selection_futures=4,confirmation_futures=4,
            reference=float(np.mean([row['reference'] for row in rows])),
            selected=float(np.mean([row['selected'] for row in rows])),gain=float(delta.mean()),
            rescues=int((delta>0).sum()),harms=int((delta<0).sum()),rows=rows,
            simulation_steps=observed['simulation_steps'])
        results.append(result)
        d.save_json(out/'comparison.json',dict(variants=results,
            use='Causal action-basis probe at twelve training states; selection and confirmation futures are disjoint. This is not a deployed-policy success rate.'))
        print(dict(phase='role_persistence_probe',without_M1=without_m1,
            **{key:value for key,value in result.items() if key!='rows'}),flush=True)


def run_continued_credit_learning(without_m1):
    """Policy iteration with a common, actually executed conditional tail."""
    from conditional_fusion_value import fit_local_role_effects
    initialize()
    out=RUNS/('six-continued-credit-without-m1-262' if without_m1 else 'six-continued-credit-full-261')
    out.mkdir(parents=True,exist_ok=True)
    base=torch.load(RUNS/('six-trained-baseline-184/policy.pth' if without_m1 else
        'six-trained-interaction-185/policy.pth'),map_location='cpu',weights_only=False)
    base.update(mechanism_ablation=(without_m1,False,without_m1),continuous_gate_learning=True,
        conditional_commitment=True,preference_strength=0.,prediction_relaxation=0.)
    current=out/'policy.pth';state=out/'learning-state.pth'
    if state.exists():
        record=torch.load(state,map_location='cpu',weights_only=False)
        history=record['history'];total=record['simulation_steps'];begin=record['iteration'];best=tuple(record['best'])
    else:
        atomic_save(base,current);rows=batched_credit_rows(current,[(scene,None) for scene in SCENES])
        initial=dict(iteration=0,summary=d.summarize(rows),rows=rows,simulation_steps=0)
        history=[initial];total=0;begin=0;best=(initial['summary']['success'],initial['summary']['capture'])
    for iteration in range(begin,4):
        folder=out/f'iteration-{iteration}';folder.mkdir(parents=True,exist_ok=True)
        teacher=folder/'teacher.pth';atomic_save(torch.load(current,map_location='cpu',weights_only=False),teacher)
        scenes=d.bank(4197000000+iteration*100000,32,((6,2.),(6,2.5)))
        tasks=[(scene,None) for scene in scenes]
        work=[(str(teacher),tasks[rank::3],str(folder/'factual'/str(rank)/'training-data.pth')) for rank in range(3)]
        with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            paths=list(pool.map(factual_credit_shard,work))
        parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
        data=dict(model=parts[0]['model'],rollouts=[roll for part in parts for roll in part['rollouts']])
        roots=d.feedback_role_roots([data],lookbacks=(2.,6.,12.,24.),decision_period=10)
        selected=[]
        for scene in scenes:
            candidates=[root for root in roots if root['scene'][0]==scene[0]]
            selected.append(candidates[0])
            if len(candidates)>1:selected.append(candidates[1+(iteration%(len(candidates)-1))])
        work=[(str(teacher),selected[rank::3],str(folder/'paired'/str(rank)),700+16*iteration) for rank in range(3)]
        with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            paths=list(pool.map(continued_credit_shard,work))
        parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
        data['paired_durations']=[dict(duration=10,pairs=[pair for part in parts for pair in part['pairs']])]
        used=sum(part['simulation_steps'] for part in parts)+sum(roll['simulation_steps'] for roll in data['rollouts'])
        total+=used;data['simulation_steps']=used;atomic_save(data,folder/'training-data.pth')
        # Reuse observations while their actual continuation policy is still
        # the current teacher. Reset this on-policy replay after an accepted
        # improvement, so conditional effects never mix incompatible tails.
        teacher_begin=max((row['iteration'] for row in history if row.get('accepted')),default=0)
        datasets=[]
        for previous in range(teacher_begin,iteration+1):
            observed=(data if previous==iteration else torch.load(
                out/f'iteration-{previous}'/'training-data.pth',map_location='cpu',weights_only=False))
            datasets.extend(local_temporal_datasets(observed))
        credit=fit_local_role_effects(datasets,heads=3,trees=96,leaf=8,physical_features=True)
        credit.update(temporal=True,initial_only=False,max_changed=1,max_interventions=0,wait_span=True,margin=0.)
        candidates=[]
        for confidence in (.5,0.):
            model=dict(base,cooperative_role_credit=dict(credit,confidence=confidence),
                counterfactual_training_steps=total,simulation_steps=total,
                conditional_tail='Current complete policy resumes every ten physical steps in both counterfactual arms.')
            path=folder/f'confidence-{confidence:g}.pth';atomic_save(model,path)
            rows=batched_credit_rows(path,[(scene,None) for scene in SCENES])
            evaluation=dict(path=str(path),summary=d.summarize(rows),rows=rows,confidence=confidence)
            candidates.append(evaluation);d.save_json(path.with_suffix('.json'),evaluation)
        chosen=max(candidates,key=lambda item:(item['summary']['success'],item['summary']['capture']))
        score=(chosen['summary']['success'],chosen['summary']['capture']);accepted=score>best
        if accepted:
            best=score;atomic_save(torch.load(chosen['path'],map_location='cpu',weights_only=False),current)
        result=dict(iteration=iteration+1,simulation_steps=total,accepted=accepted,best_success=best[0],
            fit=credit['fit'],**chosen);history.append(result)
        atomic_save(dict(iteration=iteration+1,history=history,simulation_steps=total,best=best),state)
        d.save_json(out/'learning.json',dict(history=history,
            use='Development only: one common six-vessel source, actual conditional continuation, complete removal of M1 forecasts in the no-M1 arm.'))
        print(dict(phase='continued_credit_iteration',without_M1=without_m1,iteration=iteration+1,
            success=chosen['summary']['success'],accepted=accepted,best=best[0],simulation_steps=total),flush=True)
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
    rows=batched_credit_rows(current,tasks)
    reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
    result=dict(summary=d.summarize(rows),rows=rows,paired=compare(rows,reference['rows']),reference=reference['summary'],
        simulation_steps=total,training_seed=42,use='Development policy iteration; multiple independent training seeds and matched ordinary-learning controls remain required.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='continued_credit_complete',without_M1=without_m1,success=result['summary']['success'],paired=result['paired']),flush=True)


def run_relaxed_prediction_control():
    """Fit M1's forecast continuation to the receding fusion execution."""
    initialize();out=RUNS/'six-feedback-prediction-259';out.mkdir(parents=True,exist_ok=True)
    base=torch.load(RUNS/'six-trained-interaction-185/policy.pth',map_location='cpu',weights_only=False)
    if base.get('coupled_shortlist',0):raise ValueError('This study controls the marginal candidate predictor.')
    selections=[]
    for relaxation in (.4,1.2):
        for risk in (8.,32.):
            model=dict(base,prediction_relaxation=relaxation,risk_prior=risk)
            path=out/f'relaxation-{relaxation:g}-risk-{risk:g}.pth';atomic_save(model,path)
            result=path.with_suffix('.json')
            if result.exists():data=json.loads(result.read_text())
            else:
                rows=batched_credit_rows(path,[(scene,None) for scene in SCENES])
                data=dict(summary=d.summarize(rows),rows=rows,relaxation=relaxation,risk=risk)
                d.save_json(result,data)
            selections.append(dict(path=str(path),**data))
            print(dict(phase='feedback_prediction_development',relaxation=relaxation,risk=risk,
                success=data['summary']['success']),flush=True)
    selected=max(selections,key=lambda row:(row['summary']['success'],row['summary']['capture']))
    model=torch.load(selected['path'],map_location='cpu',weights_only=False);atomic_save(model,out/'policy.pth')
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
    rows=batched_credit_rows(out/'policy.pth',tasks)
    reference=json.loads(REFERENCE.read_text())['variants'][0]
    result=dict(summary=d.summarize(rows),rows=rows,reference=reference['summary'],paired=compare(rows,reference['rows']),
        selection=[{key:value for key,value in row.items() if key!='rows'} for row in selections],
        use='M1 internal prediction study at a fixed six-vessel source policy, with M2 entirely disabled.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='feedback_prediction_complete',success=result['summary']['success'],paired=result['paired']),flush=True)


def run_independent_prediction_control():
    """Same source controls and dynamics, with separable candidate forecasts."""
    initialize();out=RUNS/'six-independent-prediction-258';out.mkdir(parents=True,exist_ok=True)
    base=torch.load(RUNS/'six-trained-interaction-185/policy.pth',map_location='cpu',weights_only=False)
    for name,value in base['actor'].items():
        if name.startswith(('adapter.gate.4.','gate_pairs.4.')) and torch.count_nonzero(value):
            raise ValueError('Fixed-predictor control must start from the unchanged shared Adapter.')
    selections=[]
    for horizon in (1.6,3.2):
        for risk in (8.,32.):
            model=dict(base,peer=False,mechanism_ablation=(True,True,False),risk_horizon=horizon,risk_prior=risk,
                prediction_control='Own candidate feedback with observed constant-velocity peers; separable decisions.')
            path=out/f'horizon-{horizon:g}-risk-{risk:g}.pth';atomic_save(model,path)
            result=path.with_suffix('.json')
            if result.exists():data=json.loads(result.read_text())
            else:
                rows=batched_credit_rows(path,[(scene,None) for scene in SCENES])
                data=dict(summary=d.summarize(rows),rows=rows,horizon=horizon,risk=risk)
                d.save_json(result,data)
            selections.append(dict(path=str(path),**data))
            print(dict(phase='independent_prediction_development',horizon=horizon,risk=risk,
                success=data['summary']['success']),flush=True)
    selected=max(selections,key=lambda row:(row['summary']['success'],row['summary']['capture']))
    model=torch.load(selected['path'],map_location='cpu',weights_only=False);atomic_save(model,out/'policy.pth')
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
    rows=batched_credit_rows(out/'policy.pth',tasks)
    joint=json.loads(REFERENCE.read_text())['variants'][0]
    result=dict(summary=d.summarize(rows),rows=rows,joint=joint['summary'],joint_minus_independent=compare(joint['rows'],rows),
        selection=[{key:value for key,value in row.items() if key!='rows'} for row in selections],
        use='M1 predictive-selection replacement at a common six-vessel source checkpoint; one training seed, M2 disabled.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='independent_prediction_complete',success=result['summary']['success'],
        joint_minus_independent=result['joint_minus_independent']),flush=True)


def run_conditional_role_search(without_m1):
    """Search connected multi-boat roles using true conditional continuations."""
    initialize()
    early=RUNS/('six-early-roles-without-m1-249' if without_m1 else 'six-early-roles-full-248')
    out=RUNS/('six-conditional-role-search-without-m1-257' if without_m1 else 'six-conditional-role-search-full-256')
    out.mkdir(parents=True,exist_ok=True)
    scenes=d.bank(4193000000,6,((6,2.),(6,2.5)));free=(-1.,)*6;results=[]
    def rank(samples,program):
        change=samples[program]-samples[free]
        return (float(change.mean()-.5*change.std(ddof=1)/np.sqrt(len(change))),
            float(change.mean()),-sum(g>=0. for g in program))
    def connect(candidates,program):
        prefix=list(free)
        for subject,gate in enumerate(program):
            if gate>=0.:
                prefix[subject]=gate;candidates.append(tuple(prefix))
    for duration in (10,25):
        source=early/f'duration-{duration}/discovery/training-data.pth'
        deadline=time.monotonic()+14400
        while not source.exists():
            if time.monotonic()>deadline:raise TimeoutError('First-stage role collection incomplete.')
            time.sleep(15)
        initial=torch.load(source,map_location='cpu',weights_only=False)
        if not initial.get('completed'):raise ValueError('First-stage role collection incomplete.')
        folder=out/f'duration-{duration}';folder.mkdir(parents=True,exist_ok=True)
        policy=folder/'reference.pth';atomic_save(initial['model'],policy)
        samples=[{key:value for key,value in role_terminal_samples(roll).items()
            if sum(g>=0. for g in key)<=1} for roll in initial['rollouts']]
        costs=initial['simulation_steps']
        for depth in (2,3):
            programs=[]
            for observed in samples:
                beam=sorted((program for program in observed if program!=free),
                    key=lambda program:rank(observed,program),reverse=True)[:2]
                candidates=[free]
                for parent in beam:
                    candidates.append(parent);connect(candidates,parent)
                    for subject in range(6):
                        for gate in (-1.,0.,.5,1.):
                            child=list(parent);child[subject]=gate;candidates.append(tuple(child))
                programs.append(list(dict.fromkeys(candidates)))
            data=collect(policy,scenes,folder/f'depth-{depth}',8,500,programs,'cuda:0',workers=3)
            costs+=data['simulation_steps']
            for observed,roll in zip(samples,data['rollouts']):observed.update(role_terminal_samples(roll))
            print(dict(phase='conditional_role_search',without_M1=without_m1,duration_steps=duration,
                depth=depth,simulation_steps=costs),flush=True)
        choices=[max(observed,key=lambda program:rank(observed,program)) for observed in samples]
        programs=[]
        for choice in choices:
            candidates=[free,choice];connect(candidates,choice)
            if len(set(candidates))==1:
                fallback=list(free);fallback[0]=.5;candidates.append(tuple(fallback))
            programs.append(list(dict.fromkeys(candidates)))
        confirmed=collect(policy,scenes,folder/'confirmation',8,516,programs,'cuda:0',workers=3)
        costs+=confirmed['simulation_steps'];rows=[]
        for roll,choice in zip(confirmed['rollouts'],choices):
            observed=role_terminal_samples(roll)
            rows.append(dict(seed=roll['row']['seed'],program=choice,reference=observed[free].tolist(),
                intervention=observed[choice].tolist()))
        delta=np.array([np.array(row['intervention'])-np.array(row['reference']) for row in rows])
        result=dict(duration_steps=duration,reference=float(np.mean([row['reference'] for row in rows])),
            success=float(np.mean([row['intervention'] for row in rows])),gain=float(delta.mean()),
            rescues=int((delta>0).sum()),harms=int((delta<0).sum()),rows=rows,simulation_steps=costs)
        results.append(result);d.save_json(out/'comparison.json',dict(variants=results,
            use='Connected conditional role-search potential. Eight search futures and eight disjoint confirmation futures; not a deployable policy.'))
        print(dict(phase='conditional_role_search_confirmation',without_M1=without_m1,
            **{key:value for key,value in result.items() if key!='rows'}),flush=True)


def temporal_terminal_shard(task):
    policy,roots,out,duration=task
    data=d.collect_fusion_td_pairs(policy,roots,out,repetitions=8,horizon=300,
        batch_size=512,intervention_steps=duration,future_start=200)
    if any(pair[side]['success'] is None for pair in data['pairs'] for side in ('before','after')):
        raise RuntimeError('Temporal credit requires actual terminal outcomes.')
    return str(Path(out)/'training-data.pth')


def run_temporal_terminal_credit(without_m1):
    initialize()
    parent=RUNS/('six-learned-fusion-without-m1-235' if without_m1 else 'six-learned-fusion-full-233')
    out=RUNS/('six-temporal-terminal-without-m1-238' if without_m1 else 'six-temporal-terminal-full-237')
    out.mkdir(parents=True,exist_ok=True);policy=out/'reference.pth'
    if not policy.exists():atomic_save(torch.load(parent/'policy.pth',map_location='cpu',weights_only=False),policy)
    scenes=d.bank(4181000000,32,((6,2.),(6,2.5)))
    tasks=[(scene,None) for scene in scenes]
    work=[(str(policy),tasks[rank::3],str(out/'factual'/str(rank)/'training-data.pth')) for rank in range(3)]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(factual_credit_shard,work))
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    data=dict(model=parts[0]['model'],rollouts=[roll for part in parts for roll in part['rollouts']])
    roots=d.feedback_role_roots([data],lookbacks=(.4,2.,6.,12.))
    total_steps=sum(part['simulation_steps'] for part in parts);duration_data=[]
    for duration in (1,5,10):
        work=[(str(policy),roots[rank::6],str(out/f'duration-{duration}'/str(rank)),duration) for rank in range(6)]
        with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            paths=list(pool.map(temporal_terminal_shard,work))
        blocks=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
        paired=dict(duration=duration,pairs=[pair for block in blocks for pair in block['pairs']],
            simulation_steps=sum(block['simulation_steps'] for block in blocks))
        total_steps+=paired['simulation_steps'];duration_data.append(paired)
        atomic_save(dict(data,paired_durations=duration_data,simulation_steps=total_steps,
            completed=len(duration_data)==3,target='Actual terminal success under a finite single-vessel intervention.'),out/'training-data.pth')
        print(dict(phase='temporal_terminal_collected',without_M1=without_m1,duration=duration,
            roots=len(roots),pairs=len(paired['pairs']),simulation_steps=total_steps),flush=True)


def run_temporal_credit_fit(without_m1,paired_outcomes=False,rank_credit=False):
    from conditional_fusion_value import fit_finite_temporal_credit,fit_finite_wait_value,fit_paired_terminal_outcomes
    initialize()
    source=RUNS/('six-temporal-terminal-without-m1-238' if without_m1 else 'six-temporal-terminal-full-237')
    out=RUNS/('six-public-outcome-without-m1-242' if without_m1 else 'six-public-outcome-full-241') if paired_outcomes else source
    if rank_credit:out=RUNS/('six-ranked-credit-without-m1-244' if without_m1 else 'six-ranked-credit-full-243')
    out.mkdir(parents=True,exist_ok=True)
    path=source/'training-data.pth';deadline=time.monotonic()+86400
    while True:
        if path.exists():
            data=torch.load(path,map_location='cpu',weights_only=False)
            if data.get('completed'):break
        if time.monotonic()>deadline:raise TimeoutError('Finite terminal credit collection did not finish.')
        time.sleep(15)
    parent=RUNS/('six-learned-fusion-without-m1-235' if without_m1 else 'six-learned-fusion-full-233')
    previous=torch.load(parent/'learning-state.pth',map_location='cpu',weights_only=False)
    initial=[net.state_dict() for net in previous['learner'].factual]
    del previous
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
    reference_path=source/'reference-paired-futures.json'
    if reference_path.exists():reference=json.loads(reference_path.read_text())
    else:
        rows=batched_credit_rows(source/'reference.pth',tasks)
        reference=dict(summary=d.summarize(rows),rows=rows);d.save_json(reference_path,reference)
    results=[]
    for conditional in (True,False):
        family='conditional' if conditional else 'ordinary'
        policy=out/(family+'.pth')
        if policy.exists():model=torch.load(policy,map_location='cpu',weights_only=False)
        else:
            credit=(fit_paired_terminal_outcomes(data,conditional=conditional,device='cuda:0',rank_credit=rank_credit) if paired_outcomes else
                fit_finite_temporal_credit(data,conditional=conditional,device='cuda:0',initial_states=initial))
            model=dict(data['model'],finite_temporal_credit=credit,
                simulation_steps=data['model'].get('simulation_steps',0)+data['simulation_steps'],
                temporal_credit_target='Actual terminal success; eight independent futures per single-vessel intervention.')
            atomic_save(model,policy)
        variants=[('full',model)]
        if conditional:
            credit=model['finite_temporal_credit']
            no_time_path=out/'conditional-no-time.pth'
            if no_time_path.exists():no_time=torch.load(no_time_path,map_location='cpu',weights_only=False)
            elif paired_outcomes:
                one_step=dict(data,paired_durations=data['paired_durations'][:1])
                time_removed=fit_paired_terminal_outcomes(one_step,device='cuda:0',use_wait=False,rank_credit=rank_credit)
                no_time=dict(model,finite_temporal_credit=time_removed);atomic_save(no_time,no_time_path)
            else:
                no_time=dict(model,finite_temporal_credit=dict(credit,
                    states=credit['states'][:1],spans=credit['spans'][:1],wait_states=[],use_wait=False))
            no_support_path=out/'conditional-no-support.pth'
            if no_support_path.exists():no_support=torch.load(no_support_path,map_location='cpu',weights_only=False)
            else:
                unsupported=dict(credit,bandwidth=0.,wait_states=[])
                unsupported['wait_states']=fit_finite_wait_value(data,unsupported,device='cuda:0')
                no_support=dict(model,finite_temporal_credit=unsupported)
                atomic_save(no_support,no_support_path)
            variants.extend((('no-time',no_time),('no-support',no_support)))
        for name,candidate in variants:
            label=family+'-'+name;destination=out/(label+'.pth');atomic_save(candidate,destination)
            evaluated=out/(label+'-paired-futures.json')
            if evaluated.exists():result=json.loads(evaluated.read_text())
            else:
                rows=batched_credit_rows(destination,tasks)
                result=dict(label=label,summary=d.summarize(rows),rows=rows,paired=compare(rows,reference['rows']),
                    intervention_rate=float(np.mean([row['conditional_interventions']>0 for row in rows])),
                    simulation_steps=candidate['simulation_steps'])
                d.save_json(evaluated,result)
            results.append(result)
            d.save_json(out/'comparison.json',dict(reference=reference['summary'],variants=results,
                use='Internal M2 development comparison on a common frozen backbone; this does not replace the independently retrained whole-mechanism 2x2 ablation.',
                representation='Subject-centered paired terminal probabilities' if paired_outcomes else 'Difference of joint terminal-return values'))
            print(dict(phase='temporal_credit_evaluated',label=label,without_M1=without_m1,
                summary=result['summary'],paired=result['paired'],intervention_rate=result['intervention_rate']),flush=True)


def run_bellman_credit(without_m1):
    from conditional_fusion_value import fit_finite_wait_value
    initialize()
    parent=RUNS/('six-ranked-credit-without-m1-244' if without_m1 else 'six-ranked-credit-full-243')
    source=RUNS/('six-temporal-terminal-without-m1-238' if without_m1 else 'six-temporal-terminal-full-237')
    out=RUNS/('six-bellman-credit-without-m1-247' if without_m1 else 'six-bellman-credit-full-246')
    out.mkdir(parents=True,exist_ok=True)
    data=torch.load(source/'training-data.pth',map_location='cpu',weights_only=False)
    reference=json.loads((source/'reference-paired-futures.json').read_text());results=[]
    tasks=[(scene,None if repetition==0 else int(np.random.SeedSequence(
        [scene[0],repetition,20261014]).generate_state(1)[0])) for repetition in range(4) for scene in SCENES]
    for label in ('conditional-full','conditional-no-time','conditional-no-support','ordinary-full'):
        path=parent/('conditional-no-time.pth' if label=='conditional-no-time' else
            'ordinary.pth' if label=='ordinary-full' else 'conditional.pth')
        deadline=time.monotonic()+14400
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError('Paired terminal ranking did not finish.')
            time.sleep(10)
        template=torch.load(path,map_location='cpu',weights_only=False);trials=[];best=(-1.,-1.)
        for confidence in (1.,.5,0.):
            policy=out/(label+'-confidence-'+str(confidence)+'.pth')
            if policy.exists():model=torch.load(policy,map_location='cpu',weights_only=False)
            else:
                credit=dict(template['finite_temporal_credit'],confidence=confidence,wait_states=[])
                if label=='conditional-no-support':credit['bandwidth']=0.
                if credit['use_wait']:credit['wait_states']=fit_finite_wait_value(data,credit,device='cuda:0')
                model=dict(template,finite_temporal_credit=credit,waiting_cost_consistent=True)
                atomic_save(model,policy)
            selection_path=policy.with_suffix('.json')
            if selection_path.exists():selection=json.loads(selection_path.read_text())
            else:
                rows=batched_credit_rows(policy,tasks[:64])
                selection=dict(confidence=confidence,summary=d.summarize(rows),rows=rows)
                d.save_json(selection_path,selection)
            trials.append(selection);score=(selection['summary']['success'],selection['summary']['capture'])
            if score>best:
                best=score;selected=model;chosen_confidence=confidence
            print(dict(phase='bellman_credit_selection',without_M1=without_m1,label=label,confidence=confidence,
                success=score[0]),flush=True)
        policy=out/(label+'.pth');atomic_save(selected,policy)
        target=out/(label+'-paired-futures.json')
        if target.exists():result=json.loads(target.read_text())
        else:
            rows=batched_credit_rows(policy,tasks)
            result=dict(label=label,summary=d.summarize(rows),rows=rows,paired=compare(rows,reference['rows']),
                confidence=chosen_confidence,intervention_rate=float(np.mean([row['conditional_interventions']>0 for row in rows])),
                development_selection=trials)
            d.save_json(target,result)
        results.append(result)
        d.save_json(out/'comparison.json',dict(reference=reference['summary'],variants=results,
            use='Development comparison with consistent net intervention value in the stopping Bellman equation; all alternatives receive the same confidence search.'))
        print(dict(phase='bellman_credit_evaluated',without_M1=without_m1,label=label,summary=result['summary'],paired=result['paired']),flush=True)


def run_factual_credit(without_m1):
    parent=RUNS/('six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195')
    out=RUNS/('six-factual-credit-without-m1-232' if without_m1 else 'six-factual-credit-full-231')
    out.mkdir(parents=True,exist_ok=True)
    tasks=[(scene,None) for scene in d.bank(4161000000,96,((6,2.),(6,2.5)))]
    work=[(str(parent/'current.pth'),tasks[rank::3],str(out/'shards'/str(rank)/'training-data.pth')) for rank in range(3)]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(factual_credit_shard,work))
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    data=dict(model=parts[0]['model'],rollouts=[roll for part in parts for roll in part['rollouts']],
        simulation_steps=sum(part['simulation_steps'] for part in parts))
    atomic_save(data,out/'training-data.pth')
    d.save_json(out/'comparison.json',dict(factual=d.summarize([r['row'] for r in data['rollouts']]),
        simulation_steps=data['simulation_steps'],use='Fresh-layout factual experience for continuation learning; not a final method evaluation.'))
    print(dict(phase='factual_credit_complete',arm=out.name,simulation_steps=data['simulation_steps']),flush=True)


def recurrent_credit_shard(task):
    start,roots,value,output,device=task
    d.collect_bootstrapped_roles(start,roots,value,output,span=10,repetitions=4,device=device)
    return str(Path(output)/'training-data.pth')


def recurrent_confirmation_shard(task):
    start,roots,programs,output,device=task
    d.collect_initial_roles_batched(start,[root['scene'] for root in roots],output,
        repetitions=8,future_start=50,programs=programs,root_records=roots,batch_size=128,device=device)
    return str(Path(output)/'training-data.pth')


def run_recurrent_confirmation(without_m1):
    from conditional_fusion_value import conditional_action_programs
    initialize()
    parent=RUNS/('six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195')
    balanced=RUNS/('six-balanced-role-without-m1-216' if without_m1 else 'six-balanced-role-full-215')
    out=RUNS/('six-recurrent-credit-without-m1-230' if without_m1 else 'six-recurrent-credit-full-229')
    policy=out/'conditional-0.5.pth';deadline=time.monotonic()+10800
    while not policy.exists():
        if time.monotonic()>deadline:raise TimeoutError('Short conditional credit did not finish fitting.')
        time.sleep(5)
    dataset=torch.load(balanced/'continuations/training-data.pth',map_location='cpu',weights_only=False)
    roots=d.feedback_role_roots([dataset],lookbacks=(2.,))
    model=torch.load(parent/'current.pth',map_location='cpu',weights_only=False)
    model['physical_intervention_span']=10;start=out/'confirmation-current.pth';atomic_save(model,start)
    actor=d.joint_load(str(policy),policy.stat().st_mtime_ns).to('cuda:0');programs=[];predictions=[]
    with torch.no_grad():
        for root in roots:
            record=d.joint_packet(root['environment'],root['observation'])
            packet={k:torch.as_tensor(v[None],device='cuda:0') for k,v in record.items()}
            action=actor(packet,deterministic=True,gate_only=True)[0]
            head=actor.cooperative_role_credit
            features,gates=actor.conditional_response_features(packet,action,proposal=False,return_decisions=True)
            packet['_response_features']=features;packet['_response_gates']=gates
            values=torch.stack([net(packet,action,10) for net in head.models])[:,0]
            candidates,_=conditional_action_programs(action.shape[1],action.device)
            lower=values.mean(0)-.5*values.std(0,unbiased=False)
            lower=lower.masked_fill((candidates>=0.).sum(-1)!=1,-1e6)
            index=int(lower.argmax());programs.append([[-1.]*len(candidates[index]),candidates[index].cpu().tolist()])
            predictions.append(dict(seed=root['scene'][0],root_id=root['id'],estimated_gain=float(values[:,index].mean()),
                lower=float(lower[index]),selected=bool(lower[index]>.02)))
    actor.cpu();workers=2 if without_m1 else 3
    tasks=[(str(start),roots[rank::workers],programs[rank::workers],str(out/'confirmation'/str(rank)),
        'cpu' if without_m1 else 'cuda:0') for rank in range(workers)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(recurrent_confirmation_shard,tasks))
    rolls=[roll for path in paths for roll in torch.load(path,map_location='cpu',weights_only=False)['rollouts']]
    by_seed={roll['row']['seed']:roll for roll in rolls};rows=[]
    for prediction,program in zip(predictions,programs):
        samples=role_samples(by_seed[prediction['seed']]);free=tuple(program[0]);chosen=tuple(program[1])
        futures=sorted(samples[free]);delta=np.asarray([samples[chosen][f]-samples[free][f] for f in futures])
        rows.append(dict(prediction,realized_gain=float(delta.mean()),actual_selected_gain=float(delta.mean()) if prediction['selected'] else 0.,
            rescues=int((delta==1).sum()),harms=int((delta==-1).sum())))
    result=dict(rows=rows,selected_roots=sum(row['selected'] for row in rows),
        estimated_gain=float(np.mean([row['estimated_gain'] for row in rows])),
        realized_gain=float(np.mean([row['realized_gain'] for row in rows])),
        actual_selected_gain=float(np.mean([row['actual_selected_gain'] for row in rows])),
        simulation_steps=sum(roll['simulation_steps'] for roll in rolls),
        use='Eight disjoint full terminal continuations per training root; estimated short targets are not treated as task outcomes.')
    d.save_json(out/'confirmation.json',result)
    print(dict(phase='recurrent_credit_confirmed',arm=out.name,**{k:v for k,v in result.items() if k!='rows'}),flush=True)


def run_recurrent_credit(without_m1):
    from conditional_fusion_value import fit_factual_continuation_value,fit_cooperative_role_credit
    initialize()
    parent=RUNS/('six-confirmed-without-m1-196' if without_m1 else 'six-confirmed-full-195')
    feedback=RUNS/('six-feedback-without-m1-198' if without_m1 else 'six-feedback-full-197')
    balanced=RUNS/('six-balanced-role-without-m1-216' if without_m1 else 'six-balanced-role-full-215')
    out=RUNS/('six-recurrent-credit-without-m1-230' if without_m1 else 'six-recurrent-credit-full-229')
    out.mkdir(parents=True,exist_ok=True)
    paths=[parent/'discovery/training-data.pth',parent/'confirmation/training-data.pth',
        feedback/'training-data.pth',balanced/'continuations/training-data.pth']
    datasets=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    value_path=out/'continuation-value.pth'
    if value_path.exists():value=torch.load(value_path,map_location='cpu',weights_only=False)
    else:
        value=fit_factual_continuation_value(datasets,device='cuda:0')
        atomic_save(value,value_path)
    root_records=d.feedback_role_roots([datasets[-1]],lookbacks=(2.,4.,8.,12.))
    workers=4 if without_m1 else 6;tasks=[]
    for rank in range(workers):
        output=out/'continuations'/str(rank)
        tasks.append((str(parent/'current.pth'),root_records[rank::workers],value,str(output),'cpu' if without_m1 else 'cuda:0'))
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        paths=list(pool.map(recurrent_credit_shard,tasks))
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    data=dict(parts[0],rollouts=[roll for part in parts for roll in part['rollouts']],
        simulation_steps=sum(part['simulation_steps'] for part in parts),
        terminal_branches=sum(part['terminal_branches'] for part in parts),branches=sum(part['branches'] for part in parts))
    atomic_save(data,out/'training-data.pth')
    print(dict(phase='recurrent_credit_targets',roots=len(root_records),simulation_steps=data['simulation_steps'],
        terminal_branches=data['terminal_branches'],branches=data['branches'],value_fits=value['fits']),flush=True)
    base=torch.load(parent/'current.pth',map_location='cpu',weights_only=False);variants=[]
    for conditional in (True,False):
        family='conditional' if conditional else 'ordinary';existing=out/(family+'-0.5.pth')
        if existing.exists():
            previous=torch.load(existing,map_location='cpu',weights_only=False)
            credit,fit=previous['cooperative_role_credit'],previous['conditional_action_fit']
        else:
            credit,fit=fit_cooperative_role_credit([data],device='cuda:0',conditional=conditional,
                probability_difference=True,response_features=True,response_standardized=True,
                max_changed=1,future_bootstrap=True,task_frame=True)
        credit.update(bandwidth=0.,wait_span=True)
        for confidence in (.5,0.):
            name=family+'-'+str(confidence);policy=out/(name+'.pth')
            model=dict(base,cooperative_role_credit=dict(credit,confidence=confidence),conditional_action_fit=fit,
                factual_value_fit=value['fits'],simulation_steps=data['simulation_steps'],
                learning_target='Short paired intervention and cross-fitted factual continuation, estimated targets kept separate from terminal success.')
            atomic_save(model,policy);target=out/(name+'-paired-futures.json')
            if not target.exists():evaluate(policy,target)
            result=json.loads(target.read_text())['variants'][0]
            reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
            result.update(name=name,paired=compare(result['rows'],reference['rows']));variants.append(result)
            d.save_json(out/'comparison.json',dict(variants=variants,without_M2=reference['summary'],value_fits=value['fits'],
                simulation_steps=data['simulation_steps'],terminal_branches=data['terminal_branches'],branches=data['branches']))
            print(dict(phase='recurrent_credit_evaluated',arm=out.name,name=name,summary=result['summary'],paired=result['paired']),flush=True)


def run_credit_continuation(without_m1):
    parent=RUNS/('six-response-credit-without-m1-214' if without_m1 else 'six-response-credit-full-213')
    out=RUNS/('six-credit-continuation-without-m1-220' if without_m1 else 'six-credit-continuation-full-219')
    out.mkdir(parents=True,exist_ok=True);variants=[]
    base=torch.load(parent/'conditional-0.5.pth',map_location='cpu',weights_only=False)
    for initial_only in (False,True):
        model=dict(base,cooperative_role_credit=dict(base['cooperative_role_credit'],
            max_interventions=1,initial_only=initial_only))
        name='initial-only' if initial_only else 'once'
        policy=out/(name+'.pth');atomic_save(model,policy)
        target=out/(name+'-paired-futures.json')
        if not target.exists():evaluate(policy,target,workers=2)
        result=json.loads(target.read_text())['variants'][0]
        reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
        result.update(name=name,paired=compare(result['rows'],reference['rows']))
        variants.append(result)
        d.save_json(out/'comparison.json',dict(variants=variants,without_M2=reference['summary'],
            use='Same learned credit weights; one intervention followed by the training continuation, versus repeated deployment.'))
        print(dict(phase='credit_continuation_evaluated',arm=out.name,name=name,
            summary=result['summary'],paired=result['paired']),flush=True)


def run_conditional_contexts(without_m1):
    initialize()
    parent=RUNS/('six-balanced-role-without-m1-216' if without_m1 else 'six-balanced-role-full-215')
    out=RUNS/('six-conditional-context-without-m1-222' if without_m1 else 'six-conditional-context-full-221')
    out.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+10800
    source=parent/'continuations/training-data.pth'
    while not source.exists():
        if time.monotonic()>deadline:raise TimeoutError('Balanced first-coordinate labels did not finish.')
        time.sleep(5)
    data=torch.load(source,map_location='cpu',weights_only=False)
    scenes=[*SCENES[:16],*SCENES[32:48]];selected={scene[0] for scene in scenes}
    programs=[];anchors={}
    for roll in data['rollouts']:
        if roll['row']['seed'] not in selected:continue
        samples=role_samples(roll);free=tuple([-1.]*roll['row']['defenders'])
        score={key:np.mean(list(values.values())) for key,values in samples.items()}
        # Force the best single context when tied with free: the independent
        # next batch measures its value and can reject it or undo that change.
        anchor=max((key for key in score if key!=free),key=lambda key:score[key])
        choices=[free,anchor]
        for i in range(len(anchor)):
            for gate in (-1.,0.,.5,1.):
                candidate=list(anchor);candidate[i]=gate;candidate=tuple(candidate)
                if candidate not in choices:choices.append(candidate)
        programs.append(choices);anchors[roll['row']['seed']]=anchor
    start=out/'current.pth';atomic_save(data['model'],start)
    observed=collect(start,scenes,out/'continuations',8,28,programs,'cpu' if without_m1 else 'cuda:0',workers=4 if without_m1 else 8)
    rows=[]
    for roll in observed['rollouts']:
        samples=role_samples(roll);free=tuple([-1.]*roll['row']['defenders']);anchor=anchors[roll['row']['seed']]
        keys=list(samples);futures=sorted(samples[free]);values=np.array([[samples[key][future] for future in futures] for key in keys])
        cross=[]
        for held in range(len(futures)):
            mean=np.delete(values,held,axis=1).mean(1)
            best=keys.index(free) if mean[keys.index(free)]>=mean.max() else int(mean.argmax())
            cross.append(values[best,held])
        rows.append(dict(seed=roll['row']['seed'],anchor=list(anchor),free=float(values[keys.index(free)].mean()),
            first_coordinate=float(values[keys.index(anchor)].mean()),oracle=float(values.mean(1).max()),
            leave_one_future_out=float(np.mean(cross))))
    result=dict(rows=rows,means={key:float(np.mean([row[key] for row in rows]))
        for key in ('free','first_coordinate','oracle','leave_one_future_out')},simulation_steps=observed['simulation_steps'],
        use='Single-vessel contrasts conditioned on changed teammate decisions, tested on independent future noise.')
    d.save_json(out/'comparison.json',result)
    print(dict(phase='conditional_contexts_collected',arm=out.name,result=result['means']),flush=True)


def run_wait_credit(without_m1):
    parent=RUNS/('six-balanced-credit-without-m1-218' if without_m1 else 'six-balanced-credit-full-217')
    out=RUNS/('six-wait-credit-without-m1-224' if without_m1 else 'six-wait-credit-full-223')
    out.mkdir(parents=True,exist_ok=True);variants=[]
    for family in ('conditional','ordinary'):
        source=parent/(family+'-0.5.pth');deadline=time.monotonic()+10800
        while not source.exists():
            if time.monotonic()>deadline:raise TimeoutError('Balanced value fit did not finish.')
            time.sleep(5)
        base=torch.load(source,map_location='cpu',weights_only=False)
        for confidence in (.5,0.):
            model=dict(base,cooperative_role_credit=dict(base['cooperative_role_credit'],wait_span=True,confidence=confidence))
            name=family+'-'+str(confidence);policy=out/(name+'.pth');atomic_save(model,policy)
            target=out/(name+'-paired-futures.json');evaluate(policy,target)
            result=json.loads(target.read_text())['variants'][0]
            reference=json.loads(REFERENCE.read_text())['variants'][1 if without_m1 else 0]
            result.update(name=name,paired=compare(result['rows'],reference['rows']))
            variants.append(result)
            d.save_json(out/'comparison.json',dict(variants=variants,without_M2=reference['summary'],
                use='Waiting is a duration-bearing M2 action; M1 and the original controls still update every 0.2 seconds.'))
            print(dict(phase='wait_credit_evaluated',arm=out.name,name=name,summary=result['summary'],paired=result['paired']),flush=True)


def run_legacy():
    out = RUNS / 'six-initial-full-179'
    if (out / 'comparison.json').exists():
        return
    workers = 8 if (RUNS / 'six-confirmed-without-m1-196/comparison.json').exists() else 2
    evaluate(out / 'policy.pth', out / 'paired-futures.json', workers=workers)
    result = json.loads((out / 'paired-futures.json').read_text())['variants'][0]
    reference = json.loads((RUNS / 'six-contact-140/paired-futures.json').read_text())
    d.save_json(out / 'comparison.json', dict(full=result['summary'], without_M2=reference['summary'],
        paired=compare(result['rows'], reference['rows']),
        use='Legacy 75000-step-base development arm, separate from the 275000-step factorial comparison.'))


def factorial():
    full = json.loads((RUNS / 'six-confirmed-full-195/paired-futures.json').read_text())['variants'][0]
    only_m2 = json.loads((RUNS / 'six-confirmed-without-m1-196/paired-futures.json').read_text())['variants'][0]
    only_m1, neither = json.loads(REFERENCE.read_text())['variants']
    d.save_json(RUNS / 'six-confirmed-full-195/factorial.json', dict(
        cells={'11': full['summary'], '10': only_m1['summary'], '01': only_m2['summary'], '00': neither['summary']},
        M1_with_M2=compare(full['rows'], only_m2['rows']), M1_without_M2=compare(only_m1['rows'], neither['rows']),
        M2_with_M1=compare(full['rows'], only_m1['rows']), M2_without_M1=compare(only_m2['rows'], neither['rows']),
        full_above_90=full['summary']['success'] > .9,
        use='Same remote host, common six-defender base, complete 2x2 development ablation.'))
    print(dict(phase='factorial_complete', success=full['summary']['success']), flush=True)


def supervise(include_baseline=True):
    processes = {}
    handles = []
    def launch(job):
        output = open('/tmp/arboids-six-paper-' + job + '.stdout', 'w', buffering=1)
        handles.append(output)
        process = subprocess.Popen([sys.executable, '-X', 'utf8', '-u', __file__, job],
            cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
        processes[job] = process
        print(dict(phase='started', job=job, pid=process.pid), flush=True)
    for job in (*(('baseline',) if include_baseline else ()), 'reference', 'full', 'without-m1'):
        launch(job)
    finished = {}
    factorial_done = False
    while processes:
        for job, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            finished[job] = code
            del processes[job]
            print(dict(phase='finished', job=job, exit_code=code), flush=True)
            if job == 'reference' and code == 0:
                launch('legacy')
        if not factorial_done and all(finished.get(job) == 0 for job in ('full', 'without-m1', 'reference')):
            factorial()
            factorial_done = True
        if processes:
            time.sleep(5)
    for handle in handles:
        handle.close()
    if any(finished.values()):
        raise RuntimeError('One or more remote experiments failed: ' + str(finished))


def public_attacker_observation(env):
    """Published eight-dimensional attacker input, using the nearest three boats."""
    observation = np.zeros(8, np.float32)
    def polar(vector):
        distance = float(np.linalg.norm(vector))
        angle = float(np.arctan2(vector[1], vector[0])-env.attacker.theta)
        if angle > np.pi:
            angle -= 2*np.pi
        elif angle < -np.pi:
            angle += 2*np.pi
        return distance, angle
    observation[:2] = polar(-env.attacker.pos)
    relative = np.asarray([boat.pos for boat in env.defender_list])-env.attacker.pos
    distances = np.linalg.norm(relative, axis=1)
    for rank, index in enumerate(np.argsort(distances, kind='stable')[:3]):
        if distances[index] < env.Att_Sensing_R:
            observation[2+2*rank:4+2*rank] = polar(relative[index])
    return observation


def source_attacker_reward(env, outcome):
    side = env.LearningSide
    try:
        env.LearningSide = 'Att'
        return float(env._get_rewards(outcome))
    finally:
        env.LearningSide = side


def attacker_task_episode(defender, attacker, scene):
    """A common learned opponent with the unchanged defender option executor."""
    seed, defenders, agility = scene
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = d.numerical_source().TADEnv(defenders)
    obs, _ = env.reset(agility=agility, noisy_agility=False)
    done = 0
    while not done:
        with torch.no_grad():
            action = defender(d.mechanism_tensor(d.joint_packet(env, obs)), deterministic=True)[0][0].numpy()
        commands = d.joint_commands(env, obs, defender, action)
        try:
            physical = next(commands)
            while True:
                attack = attacker(public_attacker_observation(env))
                record = d.joint_task_step(env, physical, defender.training_gamma,
                                           defender.training_step_cost, attacker_action=attack)
                obs, _, done, _ = record
                physical = commands.send(record)
        except StopIteration:
            pass
        finally:
            commands.close()
    return dict(seed=int(seed), defenders=defenders, agility=agility, **d.outcome(env, done),
                conditional_interventions=int(getattr(env, '_conditional_option_count', 0)),
                environment_protocol='source', opponent='learned source-SAC attacker')


_SOURCE_ATTACKER = None
_SOURCE_ATTACKER_SHA256 = None


class FrozenSourceAttacker:
    """Execute the original deterministic attacker from a frozen learning result."""
    def __init__(self, record):
        source = d.numerical_source()
        if record['specification']['source_commit'] != d.SOURCE_COMMIT:
            raise ValueError('The opponent must use the same original source dynamics.')
        network = sys.modules[source.SAC.__module__].ActorAtt
        width = record['specification']['config']['rl']['hidden_dim']
        # Loading a frozen adversary must not advance paired rollout randomness.
        with torch.random.fork_rng(devices=[]):
            self.actor = network(2, 6, 2, width).eval()
        self.actor.load_state_dict(record['actor'], strict=True)
        self.actor.requires_grad_(False)
        self.device = torch.device('cpu')
        self.choose_action = source.SAC.choose_action.__get__(self)

    def __call__(self, observation):
        return self.choose_action(observation[None], deterministic=True)


def _source_opponent_step(env, rl_action, controller='Boids', att_action=None):
    if att_action is None and _SOURCE_ATTACKER is not None:
        att_action = _SOURCE_ATTACKER(public_attacker_observation(env))
    return type(env)._arboids_original_step(env, rl_action, controller, att_action)


def configure_source_attacker(checkpoint=None):
    """Bind the same opponent to prefixes, paired branches, and evaluation.

    Original source files are untouched. Passing None restores the exact APF
    executor. Each sampling worker receives the checkpoint explicitly.
    """
    import io
    global _SOURCE_ATTACKER, _SOURCE_ATTACKER_SHA256
    environment = d.numerical_source().TADEnv
    if not hasattr(environment, '_arboids_original_step'):
        environment._arboids_original_step = environment.step
    if checkpoint is None:
        environment.step = environment._arboids_original_step
        _SOURCE_ATTACKER = _SOURCE_ATTACKER_SHA256 = None
        return None
    payload = Path(checkpoint).read_bytes()
    identity = hashlib.sha256(payload).hexdigest()
    if identity != _SOURCE_ATTACKER_SHA256:
        record = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        if not record.get('completed'):
            raise ValueError('Freeze a completed attacker run before collecting adaptation labels.')
        _SOURCE_ATTACKER = FrozenSourceAttacker(record)
        _SOURCE_ATTACKER_SHA256 = identity
    environment.step = _source_opponent_step
    return identity


def train_source_attacker(defender_checkpoint, output, *, seed=42, defenders=6, steps=500000,
                         warm_steps=50000, eval_interval=25000, evaluation_per_cell=32,
                         batch_size=4096, device='cuda:0', runtime_sha256=None):
    """Train a common adversary against one frozen defender under source dynamics.

    Checkpoints occur at episode boundaries, so a resumed run never loses an
    active defender commitment. The final episode is finished and its actual
    number of physical training steps is reported, including any budget excess.
    """
    import yaml
    from utils.config import _dict_to_namespace
    if min(steps, eval_interval, evaluation_per_cell, batch_size) <= 0 or warm_steps < 0 or defenders < 2:
        raise ValueError('The adversarial experiment requires positive budgets and at least two defenders.')
    initialize()
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/'attacker-training.pth'
    defender_checkpoint = Path(defender_checkpoint)
    identity = hashlib.sha256(defender_checkpoint.read_bytes()).hexdigest()
    defender = d.joint_load(str(defender_checkpoint), defender_checkpoint.stat().st_mtime_ns)
    if defender.environment_protocol != 'source':
        raise ValueError('The current attacker experiment uses the exact source environment.')
    source = d.numerical_source()
    config = yaml.safe_load((REPO/'train/configs/adversarial.yaml').read_text(encoding='utf-8'))
    config['rl']['batch_size'] = batch_size
    config['agent'].update(learning_side='Att', defender_num=defenders)
    config['training'].update(warm_steps=warm_steps, total_steps=steps, eval_interval=eval_interval)
    agent = source.SAC(_dict_to_namespace(config), 2, 6, 2, attacker=True, device=torch.device(device))
    replay = sys.modules[source.SAC.__module__].ReplayBuffer(8, 2)
    specification = dict(seed=seed, defenders=defenders, requested_training_steps=steps,
        config=config, defender_sha256=identity, source_sha256=d.BASELINE_SHA256,
        source_commit=d.SOURCE_COMMIT, runtime_sha256=runtime_sha256,
        attacker_input='target plus three nearest sensed defenders; same eight source features',
        defender_execution='unchanged option generator, including complete intervention windows',
        result_role='development learned-opponent training', training_agility=[2., 2.5])
    step, episodes, elapsed, history, best = 0, 0, 0., [], None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['specification'] != specification:
            raise ValueError('The resumed attacker must retain its frozen defender and complete training specification.')
        for key in ('actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
            getattr(agent, key).load_state_dict(saved['agent'][key])
        with torch.no_grad():
            agent.log_alpha.copy_(saved['agent']['log_alpha'])
        agent.alpha = agent.log_alpha.exp().detach().to(agent.device)
        for key in ('size', 'count', 'max_size'):
            setattr(replay, key, saved['replay'][key])
        for key in ('s', 'a', 'r', 's_', 'dw'):
            value = saved['replay'][key]
            getattr(replay, key)[:len(value)] = value
        step, episodes, elapsed, history, best = (saved[key] for key in ('step', 'episodes', 'elapsed', 'history', 'best'))
        np.random.set_state(saved['numpy_random_state'])
        torch.set_rng_state(saved['torch_random_state'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_random_state'])
        del saved
        if step >= steps:
            return torch.load(output/'attacker.pth', map_location='cpu', weights_only=False)
    started = time.monotonic()
    next_evaluation = ((step//eval_interval)+1)*eval_interval
    evaluation_scenes = [(432610000+int(agility*10)*1000+i, defenders, agility)
                         for agility in (2., 2.5) for i in range(evaluation_per_cell)]

    def attack(observation):
        return agent.choose_action(observation[None], deterministic=True)

    def evaluate_and_save():
        nonlocal best
        random = (np.random.get_state(), torch.get_rng_state(),
                  torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
        try:
            rows = [attacker_task_episode(defender, attack, scene) for scene in evaluation_scenes]
        finally:
            np.random.set_state(random[0])
            torch.set_rng_state(random[1])
            if device.startswith('cuda'):
                torch.cuda.set_rng_state_all(random[2])
        score = (sum(not row['success'] for row in rows), sum(row['source_failure'] for row in rows))
        history.append(dict(step=step, episodes=episodes, defender_success=sum(row['success'] for row in rows),
                            tasks=evaluation_scenes, rows=rows))
        policy = dict(actor={k:v.detach().cpu().clone() for k,v in agent.actor.state_dict().items()},
                      specification=specification, step=step, episodes=episodes, completed=step>=steps,
                      history=history, evaluation_score=score)
        atomic_save(policy, output/'attacker.pth')
        if best is None or score > tuple(best):
            best = score
            atomic_save(policy, output/'best-development-attacker.pth')
        agent_state = {key:getattr(agent, key).state_dict() for key in
                       ('actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer')}
        agent_state['log_alpha'] = agent.log_alpha.detach().cpu().clone()
        atomic_save(dict(specification=specification, step=step, episodes=episodes,
            elapsed=elapsed+time.monotonic()-started, completed=step>=steps, history=history, best=best,
            agent=agent_state, numpy_random_state=random[0], torch_random_state=random[1], cuda_random_state=random[2],
            replay={**{key:getattr(replay, key)[:replay.size] for key in ('s', 'a', 'r', 's_', 'dw')},
                    **{key:getattr(replay, key) for key in ('size', 'count', 'max_size')}}), checkpoint)
        print(dict(phase='source_attacker_training', step=step, requested=steps,
                   defender_success=history[-1]['defender_success'], evaluated=len(rows), best=best), flush=True)

    while step < steps:
        env = source.TADEnv(defenders)
        agility = float(np.random.choice((2., 2.5)))
        obs, _ = env.reset(agility=agility, noisy_agility=False)
        done = 0
        while not done:
            with torch.no_grad():
                action = defender(d.mechanism_tensor(d.joint_packet(env, obs)), deterministic=True)[0][0].numpy()
            commands = d.joint_commands(env, obs, defender, action)
            try:
                physical = next(commands)
                while True:
                    before = public_attacker_observation(env)
                    attack_action = (np.random.uniform(-1., 1., 2) if step < warm_steps else
                                     agent.choose_action(before[None]))
                    record = d.joint_task_step(env, physical, defender.training_gamma,
                                               defender.training_step_cost, attacker_action=attack_action)
                    obs, _, done, _ = record
                    following = public_attacker_observation(env)
                    replay.store(before, attack_action, source_attacker_reward(env, done), following, bool(done))
                    step += 1
                    if step >= warm_steps:
                        with torch.enable_grad():
                            agent.learn(replay)
                    physical = commands.send(record)
            except StopIteration:
                pass
            finally:
                commands.close()
        episodes += 1
        if step >= next_evaluation or step >= steps:
            evaluate_and_save()
            next_evaluation = ((step//eval_interval)+1)*eval_interval
    return torch.load(output/'attacker.pth', map_location='cpu', weights_only=False)


def continue_current_mechanisms(job, workers=6, fit_only=False, arms=('F11', 'F01'),
                                control_modes=('independent', 'single_reference'),
                                execution_runtime_sha256=None):
    """Resume the current development experiments without replacing their anchors."""
    import gc
    import mechanism_controls as controls
    import local_intervention_value as local
    from plain_return import fit_plain_return

    if execution_runtime_sha256 is not None and not fit_only:
        raise ValueError('In-memory source fitting cannot spawn filesystem evaluation workers.')
    mp.set_start_method('spawn', force=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    initialize()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    directory = d.ROOT / 'local-difference-joint-development'
    anchors = {
        'F11': ('F11-local-difference-r20-population-rank',
                '122b854a1c528b7450403f82ad14ec5b7e4e8f722e0ee17bc418b91f63e2dc6e', 'r18'),
        'F01': ('F01-local-difference-r24-population-rank',
                '3ea2a8ee9beea9df88fd31fc246e4e88f9aa4809184ece8813b0abbde6732ca2', 'r22')}
    protected = {directory / (stem+'.pth'): identity for stem, identity, _ in anchors.values()}
    protected.update({
        d.ROOT / 'joint-supported-credit-128/policy.pth':
            '12a66ac257748331db94975d9a3357cc5531b03cf5c20005feaf7577abb6b8ce',
        d.ROOT / 'joint-complete-task-development/task-ordered-source-consistent.pth':
            '2e148ac6f19e516aad29b80dd83a0d6cc4783caba8ec82a83c3c199f3af7d482'})
    for path, identity in protected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity:
            raise ValueError('The protected checkpoint changed: '+str(path))
    suites = {}
    for suffix in ('source-development', 'new4310-200', 'new-development', 'fresh-200'):
        result = torch.load(directory / (anchors['F11'][0]+'-'+suffix+'.pth'),
                            map_location='cpu', weights_only=False)
        if not result['completed'] or result['evaluated'] != len(result['tasks']):
            raise ValueError('A continuation suite must have complete reference evidence.')
        suites[suffix] = result['tasks']
    reserved_layouts = {tuple(scene) for scene, _ in suites['new4310-200']}

    def evaluate(path, remote_reference=False, evaluation_context=None):
        if fit_only:
            print(dict(phase='candidate_fit_ready', policy=path.name, evaluated=False), flush=True)
            return
        for suffix, tasks in suites.items():
            controls.evaluate_models([path], tasks, workers=workers,
                result_suffix=('remote-linux-'+suffix if remote_reference else suffix),
                evaluation_context=evaluation_context)
        print(dict(phase='complete_current_candidate', policy=path.name), flush=True)

    def exposed_development():
        diagnostics = {}
        for stem, suffix, count in (
                ('F11-local-difference-r49-factual-transport', '4320-256', 256),
                ('F11-local-difference-r65-outcome-confidence', '4321-512', 512)):
            record = torch.load(directory / (stem+'-confirmation-'+suffix+'.pth'),
                                map_location='cpu', weights_only=False)
            if not record['completed'] or record['evaluated'] != count:
                raise ValueError('The exposed diagnostic population must be complete.')
            diagnostics['diagnostic-'+suffix] = record['tasks']
        scenes = [(start+i, 6, aggression) for start, aggression in
                  ((432219000, 2.), (432220000, 2.5)) for i in range(128)]
        tasks = [(scene, future) for scene in scenes for future in
                 (None, int(np.random.SeedSequence(
                     [20261021, scene[0], 6, int(scene[2]*10), 917805]).generate_state(1)[0]))]
        exposed_layouts = {tuple(scene) for population in (*suites.values(), *diagnostics.values())
                           for scene, _ in population}
        if exposed_layouts.intersection(scenes):
            raise ValueError('The next confirmation plan overlaps an exposed layout.')
        context = dict(development_phase='post-4321-feedback-risk',
            result_role='development selection after exposed confirmations',
            prior_confirmation_exposed=True, exposed_confirmations=['4320-256', '4321-512'],
            selection_suites=list(suites)+list(diagnostics),
            selection_rule=('full and M1 must not underperform exact frozen F10 in any of six suites; '
                'M2 must not underperform fixed r24 in any suite; among eligible versions maximize '
                'total successes, then minimum per-suite gain rate, then fewer interventions'),
            next_confirmation_plan=dict(result_suffix='confirmation-4322-512',
                tasks=tasks, tasks_sha256=hashlib.sha256(json.dumps(tasks, separators=(',', ':')).encode()).hexdigest(),
                role='unused until candidate and comparator lock after development selection'),
            runtime_sha256={name: (execution_runtime_sha256[name] if execution_runtime_sha256 is not None
                else hashlib.sha256((REPO / 'train' / name).read_bytes()).hexdigest())
                for name in ('conditional_fusion_value.py', 'develop_source_fusion.py',
                             'local_intervention_value.py', 'mechanism_controls.py',
                             'plain_return.py', 'run_remote_evidence.py')},
            fitting_source_mode='in-memory source' if execution_runtime_sha256 is not None else 'filesystem')
        return diagnostics, context

    def evaluate_exposed(path, diagnostics, context):
        if fit_only:
            print(dict(phase='candidate_fit_ready', policy=path.name, evaluated=False), flush=True)
            return
        diagnostic_context = dict(context, result_role='post-unblinding diagnostic development')
        for suffix, tasks in diagnostics.items():
            controls.evaluate_models([path], tasks, workers=workers, result_suffix=suffix,
                                     evaluation_context=diagnostic_context)
        evaluate(path, evaluation_context=context)

    if job == 'continue-fixed-reference':
        for base in (d.ROOT / 'joint-128-factorial-development/F10-mapped-actuator-joint.pth',
                     directory / (anchors['F11'][0]+'.pth'),
                     directory / (anchors['F01'][0]+'.pth'),
                     d.ROOT / 'joint-complete-task-development/F00-source-clean.pth'):
            evaluate(base, remote_reference=True)
    elif job in ('continue-m2-single-intervention', 'continue-m2-strong-confidence',
                 'continue-m1-single-intervention', 'continue-m1-strong-confidence',
                 'continue-m2-outcome-confidence', 'continue-m2-factual-risk', 'continue-m1-factual-risk'):
        single = job.endswith('-single-intervention')
        current_m1 = job.startswith('continue-m1-')
        outcome_scaled = job == 'continue-m2-outcome-confidence'
        factual_risk = job.endswith('-factual-risk')
        revision = ((68 if current_m1 else 67) if factual_risk else
                    65 if outcome_scaled else (63 if single else 64) if current_m1 else (61 if single else 62))
        variant = ('factual-risk' if factual_risk else 'outcome-confidence' if outcome_scaled else
                   'single-intervention' if single else 'strong-confidence')
        diagnostics, phase_context = exposed_development() if factual_risk else (None, None)
        diagnostic = torch.load(directory / 'F11-local-difference-r49-factual-transport-confirmation-4320-256.pth',
                                map_location='cpu', weights_only=False)
        if not diagnostic['completed'] or diagnostic['evaluated'] != 256:
            raise ValueError('Use the complete, already exposed diagnostic population.')
        for arm in (('F11',) if current_m1 else arms):
            source = directory / (
                'F11-local-difference-r59-current-coverage.pth' if current_m1 else
                'F11-local-difference-r49-factual-transport.pth' if arm == 'F11' else
                'F01-local-difference-r47-uniform-population.pth')
            identity = hashlib.sha256(source.read_bytes()).hexdigest()
            output = directory / f'{arm}-local-difference-r{revision}-{variant}.pth'
            if not output.exists():
                model = torch.load(source, map_location='cpu', weights_only=False)
                value = model['successor_task_value']
                if value['kind'] != 'local_difference_joint' or value['intervention_budget'] != 0:
                    raise ValueError('The horizon comparison requires the recorded rolling predecessor.')
                if single:
                    value['intervention_budget'] = 1
                else:
                    value['confidence'] = 1.
                    if outcome_scaled:
                        value['outcome_scaled_confidence'] = True
                    if factual_risk:
                        value['factual_risk_confidence'] = True
                model['successor_task_fit'].update(execution_revision=revision,
                    decision_predecessor_sha256=identity, additional_parameter_updates=0,
                    additional_simulation_steps=0, receding_option_improvement=not single,
                    intervention_budget=value['intervention_budget'],
                    confidence=value['confidence'],
                    outcome_scaled_confidence=outcome_scaled,
                    factual_risk_confidence=factual_risk,
                    deployment_continuation=('frozen predecessor after the first accepted intervention'
                        if single else 'rolling decisions with uncertainty increasing with factual success'
                        if factual_risk else 'rolling decisions with outcome-scaled uncertainty'
                        if outcome_scaled else 'rolling decisions with a full uncertainty penalty'),
                    prior_confirmation_exposed=True,
                    selection_suites=list(suites)+['diagnostic-4320-256'])
                if factual_risk:
                    model['successor_task_fit'].update(phase_context)
                atomic_save(model, output)
                del model
                gc.collect()
            saved = torch.load(output, map_location='cpu', weights_only=False)
            if (saved['successor_task_fit'].get('execution_revision') != revision
                    or saved['successor_task_fit'].get('decision_predecessor_sha256') != identity):
                raise ValueError('The decision comparison must retain its exact predecessor.')
            del saved
            if factual_risk:
                evaluate_exposed(output, diagnostics, phase_context)
                continue
            if not fit_only:
                controls.evaluate_models([output], diagnostic['tasks'], workers=workers,
                    result_suffix='diagnostic-4320-256')
                record_path = output.with_name(output.stem+'-diagnostic-4320-256.pth')
                record = torch.load(record_path, map_location='cpu', weights_only=False)
                record.update(result_role='post-unblinding diagnostic development',
                              prior_confirmation_exposed=True)
                atomic_save(record, record_path)
            evaluate(output)
            if single and not fit_only:
                for suffix in (*suites, 'diagnostic-4320-256'):
                    record = torch.load(output.with_name(output.stem+'-'+suffix+'.pth'),
                                        map_location='cpu', weights_only=False)
                    if any(row['conditional_interventions'] > 1 for row in record['rows']):
                        raise ValueError('A single-intervention comparison executed another intervention.')
    elif job == 'continue-m1-controls':
        bank = torch.load(directory / 'M1-independent-r33-with-M2-local-retrained-factual-data.pth',
                          map_location='cpu', weights_only=False)
        if not bank['completed'] or bank['trajectories'] != 756:
            raise ValueError('Resume from the complete, previously fixed factual sampling design.')
        tasks = [(tuple(record['scene']), record['future']) for record in bank['records']]
        if any(tuple(scene) in reserved_layouts for scene, _ in tasks):
            raise ValueError('The 4310 evaluation layout cannot enter this fixed training design.')
        del bank
        for mode in control_modes:
            base = directory / f'M1-{mode}-r33-base.pth'
            # Recreate Linux trajectories instead of combining the interrupted
            # Windows comparison batch with a numerically different prefix.
            output = directory / f'M1-{mode}-r41-remote-retrained.pth'
            references = [dict(policy_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                               tasks=tasks, rows=[dict(success=None) for _ in tasks])]
            controls.retrain_local_fusion_control(base, directory / (anchors['F11'][0]+'.pth'),
                directory / 'F11-local-joint-r18-onpolicy-training-data.pth',
                references, output, workers=workers, device='cuda:0',
                inference_batch=32, sampling_device='cpu')
            gc.collect()
            torch.cuda.empty_cache()
            evaluate(output)
    elif job in ('continue-m2-bounds', 'continue-m2-local-bounds'):
        for arm in arms:
            localized = job == 'continue-m2-local-bounds'
            revision = 42 if localized else 39
            variant = 'supported-consequences' if localized else 'bounded-consequences'
            output = directory / f'{arm}-local-difference-r{revision}-{variant}.pth'
            if not output.exists():
                stem, _, data_revision = anchors[arm]
                model = torch.load(directory / (stem+'.pth'), map_location='cpu', weights_only=False)
                data = torch.load(directory / f'{arm}-local-joint-{data_revision}-onpolicy-training-data.pth',
                                  map_location='cpu', weights_only=False)
                model = local.calibrate_conditional_differences(model, data)
                model['successor_task_value'].update(stratum_value_bounds=True,
                    stratum_localized_bounds=localized,
                    factual_bound_pruning=True, candidate_bound_pruning=True, inference_device='cuda:0')
                model['successor_task_fit'].update(execution_revision=revision,
                    conditional_consequence_bounds=True, supported_conditional_bounds=localized,
                    reference_sha256=anchors[arm][1])
                atomic_save(model, output)
                del model, data
                gc.collect()
                torch.cuda.empty_cache()
            evaluate(output)
    elif job == 'continue-m2-evidence-confidence':
        for arm in arms:
            stem, identity, revision = anchors[arm]
            source = directory / f'{arm}-local-difference-r49-factual-transport.pth'
            output = directory / f'{arm}-local-difference-r56-effective-confidence.pth'
            refresh_required = not output.exists()
            if output.exists():
                saved = torch.load(output, map_location='cpu', weights_only=False)
                refresh_required = not saved['successor_task_fit'].get('confidence_only_calibration', False)
                del saved
                if refresh_required and any(output.parent.glob(output.stem+'-*.pth')):
                    raise ValueError('A measured confidence candidate cannot be replaced in place.')
            if refresh_required:
                reference = torch.load(source, map_location='cpu', weights_only=False)
                if reference['successor_task_fit'].get('execution_revision') != 49:
                    raise ValueError('Effective-future confidence requires the completed transported value.')
                data = torch.load(directory / f'{arm}-local-joint-{revision}-onpolicy-training-data.pth',
                                  map_location='cpu', weights_only=False)
                model = local.refresh_local_calibration(reference, data,
                    effective_future_confidence=True, confidence_only=True)
                model['successor_task_fit'].update(execution_revision=56,
                    reference_sha256=identity,
                    confidence_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    additional_parameter_updates=0, additional_terminal_observations=0,
                    confidence_estimator='paired outcome bounds with effective weighted future counts')
                atomic_save(model, output)
                del model, data, reference
                gc.collect()
                torch.cuda.empty_cache()
            evaluate(output)
    elif job in ('continue-m2-factual-transport', 'continue-m2-population-transport'):
        for arm in arms:
            population = job == 'continue-m2-population-transport'
            execution_revision = 52 if population else 49
            variant = 'population-transport' if population else 'factual-transport'
            output = directory / f'{arm}-local-difference-r{execution_revision}-{variant}.pth'
            if not output.exists():
                stem, identity, revision = anchors[arm]
                source = directory / (f'{arm}-local-difference-r47-uniform-population.pth'
                                      if population else stem+'.pth')
                reference = torch.load(source, map_location='cpu', weights_only=False)
                if population and (reference['successor_task_fit'].get('execution_revision') != 47
                        or reference['successor_task_fit'].get('reference_sha256') != identity):
                    raise ValueError('Population transport requires its completed uniform-population fit.')
                data = torch.load(directory / f'{arm}-local-joint-{revision}-onpolicy-training-data.pth',
                                  map_location='cpu', weights_only=False)
                if population:
                    for root in data['roots']:
                        root['future_weights'] = np.ones(len(root['futures']), np.float32)
                model = local.refresh_local_calibration(
                    reference, data, factual_difference_transport=True)
                model['successor_task_value'].update(
                    inference_device='cuda:0', factual_bound_pruning=True, candidate_bound_pruning=True)
                model['successor_task_fit'].update(execution_revision=execution_revision,
                    reference_sha256=identity, transport_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    factual_calibration_transport='preserve paired neural differences within task-value bounds')
                atomic_save(model, output)
                del model, data, reference
                gc.collect()
                torch.cuda.empty_cache()
            evaluate(output)
    elif job == 'continue-m1-adapt':
        base = directory / 'M1-joint-r40-refined-beam.pth'
        behavior = directory / 'F11-local-difference-r43-refined-fusion.pth'
        output = directory / 'F11-local-difference-r45-adapted-fusion.pth'
        if not output.exists():
            references = []; behavior_references = []
            for suffix in ('source-development', 'new-development', 'fresh-200'):
                for policy, target in ((base, references), (behavior, behavior_references)):
                    record = torch.load(policy.with_name(policy.stem+'-'+suffix+'.pth'),
                                        map_location='cpu', weights_only=False)
                    if not record['completed'] or record['policy_sha256'] != hashlib.sha256(policy.read_bytes()).hexdigest():
                        raise ValueError('Adaptation needs complete results from its exact controllers.')
                    target.append(record)
            selected = local.development_roots(behavior_references, failures=4, successes=2, late=True)
            outcomes = {(tuple(scene), future): row for record in behavior_references
                        for (scene, future), row in zip(record['tasks'], record['rows'])}
            root_tasks = [(scene, future, 'last_intervention' if lookback is not None
                          and outcomes[tuple(scene), future]['conditional_interventions'] else lookback)
                          for scene, future, lookback in selected]
            if any(tuple(scene) in reserved_layouts for scene, _, _ in root_tasks):
                raise ValueError('The 4310 evaluation layouts cannot enter adaptation.')
            bank_path = output.with_name(output.stem+'-factual-data.pth')
            bank = local.collect_factual_trajectories(base, references, bank_path,
                workers=workers, inference_batch=32, inference_device='cpu')
            data_path = output.with_name(output.stem+'-training-data.pth')
            previous = torch.load(data_path, map_location='cpu', weights_only=False) if data_path.exists() else None
            sampled = local.collect_local_tasks(base, root_tasks, data_path,
                workers=workers, repetitions=2, confirmation=6, screen_limit=48,
                context_policy=behavior, neighborhood_width=2, record_continuations=True,
                existing_data=previous)
            factual = local.factual_reference_data(base, references)
            factual['simulation_steps'] = 0
            data = local.focus_development_futures(local.merge_paired_data([sampled, factual]),
                references, weight=128.)
            data['training_data'] = [str(data_path), str(bank_path)]
            continuation = local.paired_continuation_bank(sampled)
            bank['records'] += continuation['records']
            bank['paired_continuation_state_labels'] = continuation['state_labels']
            bank['training_data'] = [str(bank_path), str(data_path)]
            reference = torch.load(directory / (anchors['F11'][0]+'.pth'), map_location='cpu', weights_only=False)
            fit = reference['successor_task_fit']; value = reference['successor_task_value']
            model = local.fit_joint_potential(data, output, factual_bank=bank, device='cuda:0',
                updates=6000, width=fit['width'], seed=fit['seed'], initial_states=value['states'],
                learning_rate=2e-4, edge_weight=fit['difference_weight'],
                anchor_weight=fit['factual_contrast_weight'], rank_weight=fit['local_rank_weight'],
                factual_weight=fit['factual_weight'],
                **{key:value[key] for key in ('bandwidth','margin','confidence','global_penalty',
                    'context_time_scale','interpolation_power','context_conditioned','motion_basis',
                    'local_contrast_head','joint_factual_calibration','residual_potential')})
            model['successor_task_value'].update(inference_device='cuda:0',
                factual_bound_pruning=True, candidate_bound_pruning=True)
            model['successor_task_fit'].update(execution_revision=45,
                initialization_checkpoint_sha256=anchors['F11'][1],
                initialization_updates=fit['updates'], total_parameter_updates=fit['updates']+6000,
                initialization_simulation_steps=fit['simulation_steps'],
                behavior_checkpoint_sha256=hashlib.sha256(behavior.read_bytes()).hexdigest(),
                actual_replacement_labels=True, adaptation_contexts=len(root_tasks))
            atomic_save(model, output)
            del reference, data, bank, sampled, model
            gc.collect()
            torch.cuda.empty_cache()
        saved = torch.load(output, map_location='cpu', weights_only=False)
        if saved['successor_task_fit'].get('execution_revision') != 45:
            raise ValueError('The replacement adaptation has not completed its provenance check.')
        del saved
        evaluate(output)
    elif job in ('continue-m2-conditional-fit', 'continue-m2-population-fit'):
        for arm in arms:
            population = job == 'continue-m2-population-fit'
            execution_revision = 47 if population else 44
            variant = 'uniform-population' if population else 'conditional-outcomes'
            output = directory / f'{arm}-local-difference-r{execution_revision}-{variant}.pth'
            if not output.exists():
                stem, _, revision = anchors[arm]
                reference = torch.load(directory / (stem+'.pth'), map_location='cpu', weights_only=False)
                data = torch.load(directory / f'{arm}-local-joint-{revision}-onpolicy-training-data.pth',
                                  map_location='cpu', weights_only=False)
                bank = local.factual_bank_from_model(reference, data)
                if population:
                    # The execution noise seed is not an observation. Give
                    # each measured future equal mass instead of concentrating
                    # the task-value target on a particular development draw.
                    for root in data['roots']:
                        root['future_weights'] = np.ones(len(root['futures']), np.float32)
                fit = reference['successor_task_fit']
                value = reference['successor_task_value']
                model = local.fit_joint_potential(data, output, factual_bank=bank, device='cuda:0',
                    updates=fit['updates'], width=fit['width'], seed=fit['seed'],
                    edge_weight=fit['difference_weight'], anchor_weight=fit['factual_contrast_weight'],
                    rank_weight=fit['local_rank_weight'], factual_weight=fit['factual_weight'],
                    conditional_strata=not population, stratified_local_calibration=not population,
                    **{key:value[key] for key in ('bandwidth','margin','confidence','global_penalty',
                        'context_time_scale','interpolation_power','context_conditioned','motion_basis',
                        'local_contrast_head','joint_factual_calibration','residual_potential')})
                if not population and model['successor_task_fit']['training_observation_sha256'] != fit['training_observation_sha256']:
                    raise ValueError('Conditional outcome heads must use the fixed common observations.')
                model['successor_task_value'].update(inference_device='cuda:0',
                    factual_bound_pruning=True, candidate_bound_pruning=True)
                model['successor_task_fit'].update(execution_revision=execution_revision,
                    reference_sha256=anchors[arm][1],
                    source_observation_sha256=fit['training_observation_sha256'],
                    future_population_weight=1. if population else 'inherited',
                    conditional_outcome_heads=('single joint potential with paired local differences' if population else
                        'factual probability, rescue given failure, harm given success'))
                atomic_save(model, output)
                del reference, data, bank, model
                gc.collect()
                torch.cuda.empty_cache()
            saved = torch.load(output, map_location='cpu', weights_only=False)
            saved_fit = saved['successor_task_fit']
            if (saved_fit.get('execution_revision') != execution_revision
                    or saved_fit.get('reference_sha256') != anchors[arm][1]):
                raise ValueError('The conditional outcome fit has not completed its provenance check.')
            del saved
            evaluate(output)
    elif job in ('continue-m2-dual-motion', 'continue-m2-dual-population',
                 'continue-m2-response', 'continue-m2-dual-response'):
        uniform = job == 'continue-m2-dual-population'
        response_supervision = job in ('continue-m2-response', 'continue-m2-dual-response')
        dual = job != 'continue-m2-response'
        execution_revision = {'continue-m2-dual-motion':85, 'continue-m2-dual-population':86,
            'continue-m2-response':90, 'continue-m2-dual-response':91}[job]
        method = 'response-supervision' if response_supervision else 'dual-motion'
        diagnostics, phase_context = exposed_development()
        phase_context['development_phase'] = ('measured-response-conditional-intervention' if response_supervision
                                               else 'dual-timescale-conditional-intervention')
        for arm in arms:
            stem, identity, revision = anchors[arm]
            output = directory / f'{arm}-local-difference-r{execution_revision}-{method}.pth'
            if not output.exists():
                reference = torch.load(directory / (stem+'.pth'), map_location='cpu', weights_only=False)
                data_path = directory / f'{arm}-local-joint-{revision}-onpolicy-training-data.pth'
                data = torch.load(data_path, map_location='cpu', weights_only=False)
                bank = local.factual_bank_from_model(reference, data)
                if uniform:
                    for root in data['roots']:
                        root['future_weights'] = np.ones(len(root['futures']), np.float32)
                fit, value = reference['successor_task_fit'], reference['successor_task_value']
                model = local.fit_joint_potential(data, output, factual_bank=bank, device='cuda:0',
                    updates=24000, ensemble_size=3, width=61 if dual else 64, seed=42,
                    finite_motion_basis=dual, dual_motion_basis=dual,
                    response_supervision_weight=4. if response_supervision else 0.,
                    factual_difference_transport=True, effective_future_confidence=True,
                    edge_weight=fit['difference_weight'], anchor_weight=fit['factual_contrast_weight'],
                    rank_weight=fit['local_rank_weight'], factual_weight=fit['factual_weight'],
                    **{key:value[key] for key in ('bandwidth','margin','confidence','global_penalty',
                        'context_time_scale','interpolation_power','context_conditioned','motion_basis',
                        'local_contrast_head','joint_factual_calibration','residual_potential')})
                if not uniform and model['successor_task_fit']['training_observation_sha256'] != fit['training_observation_sha256']:
                    raise ValueError('The dual-time representation changed the common observations.')
                model['successor_task_value'].update(inference_device='cuda:0',
                    factual_bound_pruning=True, candidate_bound_pruning=True)
                model['successor_task_fit'].update(phase_context, execution_revision=execution_revision,
                    reference_sha256=identity, paired_training_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    optimization_condition=('fresh joint potential with measured physical-response auxiliary supervision'
                        if response_supervision else 'fresh joint-potential fit with sustained and finite-window motion features'),
                    future_population_weight=1. if uniform else 'inherited',
                    total_parameter_updates=24000, total_optimizer_steps=24000,
                    additional_simulation_steps=0, inherited_calibration=False, warm_start=False)
                atomic_save(model, output)
                del reference, data, bank, model
                gc.collect()
                torch.cuda.empty_cache()
            saved = torch.load(output, map_location='cpu', weights_only=False)
            if (saved['successor_task_fit'].get('execution_revision') != execution_revision
                    or bool(saved['successor_task_value'].get('dual_motion_basis')) != dual
                    or bool(saved['successor_task_fit'].get('response_supervision_weight')) != response_supervision):
                raise ValueError('The conditional candidate must have its complete recorded fit.')
            del saved
            evaluate_exposed(output, diagnostics, phase_context)
    elif job == 'continue-m2-short-response':
        diagnostics, phase_context = exposed_development()
        phase_context['development_phase'] = 'measured-short-intervention-continuation'
        for arm in arms:
            _, _, revision = anchors[arm]
            previous_path = directory / f'{arm}-local-joint-{revision}-onpolicy-training-data.pth'
            fresh_path = directory / f'{arm}-local-joint-r95-short-response-training-data.pth'
            source = directory / (f'{arm}-local-difference-r61-single-intervention.pth' if arm == 'F11'
                                  else f'{arm}-local-difference-r65-outcome-confidence.pth')
            plain_source = directory / f'{arm}-ordinary-r40-uniform-mlp.pth'
            output = directory / f'{arm}-local-difference-r97-short-response-update.pth'
            plain_output = directory / f'{arm}-ordinary-r97-short-response-mlp.pth'
            previous = torch.load(previous_path, map_location='cpu', weights_only=False)
            fresh = torch.load(fresh_path, map_location='cpu', weights_only=False)
            if not fresh['completed'] or not fresh['roots']:
                raise ValueError('Short-response continuation requires the complete new collection.')
            initial = torch.load(source, map_location='cpu', weights_only=False)
            value, fit = initial['successor_task_value'], initial['successor_task_fit']
            if previous['predecessor_sha256'] != fit['predecessor_sha256']:
                raise ValueError('Short interventions must retain the exact frozen predecessor.')
            if arm == 'F01':
                for root in previous['roots']:
                    root['future_weights'] = np.ones(len(root['futures']), np.float32)
            # Keep the inherited objective population for repeated futures.
            # New futures retain unit weight, including newly exposed layouts.
            def root_key(root):
                return (tuple(root['scene']), None if root['root_steps'] == 0 else root['prefix_future'],
                        root['root_steps'])
            def future_keys(root):
                return [root['prefix_future'] if f is None and root['root_steps'] == 0 else f
                        for f in root['futures']]
            old_roots = {}
            for root in previous['roots']:
                old_roots.setdefault(root_key(root), []).append(root)
            for root in fresh['roots']:
                inherited_weights = {}
                for old in old_roots.get(root_key(root), ()):
                    if (not np.allclose(old['context'], root['context'], atol=1e-6)
                            or not np.allclose(old['effect_controls'], root['effect_controls'], atol=1e-6)):
                        continue
                    for future, weight in zip(future_keys(old), old.get('future_weights', np.ones(len(old['futures'])))):
                        if future in inherited_weights and not np.isclose(inherited_weights[future], weight):
                            raise ValueError('The inherited population contains conflicting future weights.')
                        inherited_weights[future] = float(weight)
                root['future_weights'] = np.asarray([inherited_weights.get(future, 1.)
                    for future in future_keys(root)], np.float32)
            data = local.merge_paired_data([previous, fresh], allow_distinct_contexts=True)
            data['training_data'] = [str(previous_path), str(fresh_path)]
            bank = local.factual_bank_from_model(initial, previous)
            continuation = local.paired_continuation_bank(fresh)
            bank['records'] += continuation['records']
            bank['state_labels'] += continuation['state_labels']
            bank['paired_continuation_state_labels'] += continuation['state_labels']
            bank['training_data'] = [bank.get('training_data'), str(fresh_path)]
            provenance = dict(phase_context, execution_revision=97,
                retained_training_data_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                paired_training_data_sha256=hashlib.sha256(fresh_path.read_bytes()).hexdigest(),
                retained_collection_completed=bool(previous['completed']), new_collection_completed=True,
                retained_paired_simulation_steps=previous['simulation_steps'],
                fresh_paired_simulation_steps=fresh['simulation_steps'],
                new_measured_roots=len(fresh['roots']), actual_replacement_labels=True,
                new_collection_recipe=fresh.get('recipe'), inherited_calibration=False,
                future_population_rule='inherited weights for repeated physical roots and futures; unit weight for new futures')
            if not output.exists():
                model = local.fit_joint_potential(data, output, factual_bank=bank, device='cuda:0',
                    updates=6000, ensemble_size=len(value['states']), width=fit['width'], seed=fit['seed'],
                    initial_states=value['states'], additional_candidate_options=list(value['options']),
                    factual_difference_transport=True, effective_future_confidence=True, learning_rate=2e-4,
                    edge_weight=fit['difference_weight'], anchor_weight=fit['factual_contrast_weight'],
                    rank_weight=fit['local_rank_weight'], factual_weight=fit['factual_weight'],
                    **{key:value[key] for key in ('bandwidth','margin','confidence','global_penalty',
                        'context_time_scale','interpolation_power','context_conditioned','motion_basis',
                        'local_contrast_head','joint_factual_calibration','residual_potential')})
                model['successor_task_value'].update(inference_device='cuda:0',
                    factual_bound_pruning=True, candidate_bound_pruning=True,
                    intervention_budget=int(value.get('intervention_budget', 0)),
                    outcome_scaled_confidence=bool(value.get('outcome_scaled_confidence', False)))
                initialization_updates = fit.get('total_parameter_updates', fit['updates'])
                model['successor_task_fit'].update(provenance,
                    initialization_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    initialization_updates=initialization_updates, total_parameter_updates=initialization_updates+6000,
                    additional_simulation_steps=fresh['simulation_steps'])
                atomic_save(model, output)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            matched = torch.load(output, map_location='cpu', weights_only=False)
            if matched['successor_task_fit'].get('execution_revision') != 97:
                raise ValueError('The conditional short-response fit has incomplete provenance.')
            if not plain_output.exists():
                plain_initial = torch.load(plain_source, map_location='cpu', weights_only=False)
                model = fit_plain_return(data, plain_output, factual_bank=bank, updates=6000,
                    parameter_budget=145731, seed=42, uniform_sampling=True,
                    initial_model=plain_initial, learning_rate=2e-4,
                    additional_candidate_options=matched['successor_task_value']['options'],
                    expected_observation_sha256=matched['successor_task_fit']['training_observation_sha256'])
                model['successor_task_value']['intervention_budget'] = int(value.get('intervention_budget', 0))
                model['successor_task_fit'].update(provenance,
                    initialization_checkpoint_sha256=hashlib.sha256(plain_source.read_bytes()).hexdigest(),
                    matched_conditional_checkpoint_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    additional_simulation_steps=0)
                atomic_save(model, plain_output)
                del model, plain_initial
            saved = torch.load(plain_output, map_location='cpu', weights_only=False)
            if (saved['successor_task_fit'].get('execution_revision') != 97
                    or saved['successor_task_fit']['training_observation_sha256'] !=
                    matched['successor_task_fit']['training_observation_sha256']):
                raise ValueError('The ordinary comparison must use every identical observation.')
            del saved, matched, initial, previous, fresh, data, bank, continuation
            gc.collect()
            torch.cuda.empty_cache()
            evaluate_exposed(output, diagnostics, phase_context)
            evaluate_exposed(plain_output, diagnostics, phase_context)
    elif job == 'continue-plain':
        for arm in arms:
            output = directory / f'{arm}-ordinary-r40-uniform-mlp.pth'
            if not output.exists():
                stem, _, revision = anchors[arm]
                model = torch.load(directory / (stem+'.pth'), map_location='cpu', weights_only=False)
                data = torch.load(directory / f'{arm}-local-joint-{revision}-onpolicy-training-data.pth',
                                  map_location='cpu', weights_only=False)
                bank = local.factual_bank_from_model(model, data)
                fit_plain_return(data, output, factual_bank=bank,
                    updates=model['successor_task_fit']['updates'], parameter_budget=145731,
                    seed=42, uniform_sampling=True,
                    expected_observation_sha256=model['successor_task_fit']['training_observation_sha256'])
                del model, data, bank
                gc.collect()
                torch.cuda.empty_cache()
            evaluate(output)
    elif job == 'continue-current-finite-motion':
        diagnostics, phase_context = exposed_development()
        phase_context['development_phase'] = 'post-4321-finite-window-motion'
        base = directory / 'M1-joint-r50-pareto-refinement.pth'
        old = directory / 'F11-local-difference-r55-current-local-training-data.pth'
        fresh = directory / 'F11-local-difference-r59-current-coverage-training-data.pth'
        reference = directory / 'F11-local-difference-r59-current-coverage.pth'
        original = directory / 'F11-local-difference-r70-current-scratch.pth'
        plain = directory / 'F11-ordinary-r71-current-scratch-mlp.pth'
        output = directory / 'F11-local-difference-r73-finite-window-motion.pth'
        expected = {
            base: '835932398d2703dae65aa512b417f85589db4048ea8f6c2917939a58ffb74cbb',
            old: '810a40f9696e01bc093e1da5cec2f53b7fc0d28876b8a78047a669372088f7c9',
            fresh: '7b4adb761cd13ee11a7e6d8b30aadb9c9733ecb9403b3fa44ef0aa8f73ae0f23',
            original: '7b3cfa6785e549e11feadc3293cd3079e806a7ac0d5d00b4c8ee72be09393a96',
            plain: '74c66eb6496a32f2036feea7a8ca91db2d5ebba9b4630e8bf773c4b69aac8d9c'}
        for path, identity in expected.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != identity:
                raise ValueError('The fixed finite-motion comparison input changed: '+str(path))
        observation_sha = '389dcec396b8dd5b38d1ede98bf839256421e18ebf92141120017904607b27f9'
        if not output.exists():
            before = torch.load(old, map_location='cpu', weights_only=False)
            after = torch.load(fresh, map_location='cpu', weights_only=False)
            if not before['completed'] or not after['completed']:
                raise ValueError('The finite-motion fit requires complete common observations.')
            paired = local.merge_paired_data([before, after], allow_distinct_contexts=True)
            references = [torch.load(base.with_name(base.stem+'-'+suffix+'.pth'),
                map_location='cpu', weights_only=False)
                for suffix in ('source-development', 'new-development', 'fresh-200')]
            if any(not record['completed'] or record['policy_sha256'] != expected[base] for record in references):
                raise ValueError('The finite-motion fit requires the exact current M1 factual outcomes.')
            factual = local.factual_reference_data(base, references)
            factual['simulation_steps'] = 0
            data = local.merge_paired_data([paired, factual], allow_distinct_contexts=True)
            data['training_data'] = [str(old), str(fresh), str(reference)]
            initial = torch.load(reference, map_location='cpu', weights_only=False)
            fit, value = initial['successor_task_fit'], initial['successor_task_value']
            if fit['training_observation_sha256'] != observation_sha:
                raise ValueError('The finite-motion experiment must retain its common observation population.')
            if any(not np.all(np.asarray(root.get('future_weights', np.ones(len(root['futures'])))) == 1)
                   for root in data['roots']):
                raise ValueError('The common observations must retain uniform future weights.')
            reserved = {tuple(scene) for scene, _ in phase_context['next_confirmation_plan']['tasks']}
            if reserved.intersection(tuple(root['scene']) for root in data['roots']):
                raise ValueError('The finite-motion fit overlaps the unused confirmation population.')
            bank = local.factual_bank_from_model(initial, data)
            print(dict(phase='finite_motion_inputs_checked', roots=len(data['roots']),
                observation_sha256=observation_sha, optimizer_steps=24000), flush=True)
            model = local.fit_joint_potential(data, output, factual_bank=bank, device='cuda:0',
                updates=24000, ensemble_size=len(value['states']), width=fit['width'], seed=fit['seed'],
                initial_states=None, additional_candidate_options=list(value['options']),
                factual_difference_transport=True, effective_future_confidence=True, learning_rate=8e-4,
                finite_motion_basis=True, edge_weight=fit['difference_weight'],
                anchor_weight=fit['factual_contrast_weight'], rank_weight=fit['local_rank_weight'],
                factual_weight=fit['factual_weight'],
                **{key:value[key] for key in ('bandwidth','margin','confidence','global_penalty',
                    'context_time_scale','interpolation_power','context_conditioned','motion_basis',
                    'local_contrast_head','joint_factual_calibration','residual_potential')})
            if model['successor_task_fit']['training_observation_sha256'] != observation_sha:
                raise ValueError('Finite-motion fitting changed the shared observations.')
            model['successor_task_value'].update(inference_device='cuda:0',
                factual_bound_pruning=True, candidate_bound_pruning=True)
            model['successor_task_fit'].update(phase_context, execution_revision=73,
                optimization_condition='fresh initialization with public finite-window motion features',
                initialization_updates=0, total_parameter_updates=24000, total_optimizer_steps=24000,
                additional_optimizer_steps=24000, additional_simulation_steps=0,
                actual_replacement_labels=True, inherited_calibration=False, warm_start=False,
                paired_training_data_sha256=expected[fresh], retained_training_data_sha256=expected[old],
                source_observation_checkpoint_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                matched_persistent_feature_checkpoint_sha256=expected[original],
                matched_plain_checkpoint_sha256=expected[plain])
            atomic_save(model, output)
        saved = torch.load(output, map_location='cpu', weights_only=False)
        if (saved['successor_task_fit'].get('execution_revision') != 73
                or saved['successor_task_fit']['training_observation_sha256'] != observation_sha
                or not saved['successor_task_value'].get('finite_motion_basis')):
            raise ValueError('A finite-motion comparison requires its complete recorded fit.')
        del saved
        evaluate_exposed(output, diagnostics, phase_context)
    elif job in ('continue-m1-coverage', 'continue-m1-coverage-plain'):
        ordinary = job == 'continue-m1-coverage-plain'
        revision = 60 if ordinary else 59
        base = directory / 'M1-joint-r50-pareto-refinement.pth'
        behavior = directory / 'F11-local-difference-r57-current-learned.pth'
        old_base = d.ROOT / 'joint-128-factorial-development/F10-mapped-actuator-joint.pth'
        conditional = directory / 'F11-local-difference-r59-current-coverage.pth'
        source = (directory / 'F11-ordinary-r58-current-uniform-mlp.pth' if ordinary else behavior)
        output = directory / 'F11-ordinary-r60-current-coverage-mlp.pth' if ordinary else conditional
        old_data_path = directory / 'F11-local-difference-r55-current-local-training-data.pth'
        data_path = conditional.with_name(conditional.stem+'-training-data.pth')
        bank_path = directory / 'F11-local-difference-r54-current-factual-factual-data.pth'
        if ordinary and not conditional.exists():
            raise ValueError('Complete the shared coverage fit before its plain-return comparison.')
        if not output.exists():
            references = []; root_tasks = []; selected_counts = []
            for suffix in ('source-development', 'new-development', 'fresh-200'):
                records = []
                for path, remote in ((old_base, True), (base, False), (behavior, False)):
                    result = path.with_name(path.stem+('-remote-linux-' if remote else '-')+suffix+'.pth')
                    record = torch.load(result, map_location='cpu', weights_only=False)
                    if not record['completed'] or record['policy_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise ValueError('Coverage selection requires complete results from the exact controllers.')
                    records.append(record)
                original, factual, learned = records
                if original['tasks'] != factual['tasks'] or factual['tasks'] != learned['tasks']:
                    raise ValueError('Coverage selection requires paired layout and future identities.')
                references.append(factual)
                counts = dict(harm=0, m1_regression=0, residual_failure=0)
                for (scene, future), old, before, after in zip(
                        factual['tasks'], original['rows'], factual['rows'], learned['rows']):
                    if before['success'] and not after['success']:
                        kind = 'harm'
                    elif old['success'] and not before['success']:
                        kind = 'm1_regression'
                    elif not after['success']:
                        kind = 'residual_failure'
                    else:
                        continue
                    if after['conditional_interventions']:
                        root = 'first_intervention' if kind == 'harm' else 'last_intervention'
                    else:
                        root = 40 if after['outcome'] == 2 else 120
                    root_tasks.append((scene, future, root)); counts[kind] += 1
                selected_counts.append(dict(suite=suffix, **counts))
            if not root_tasks or any(tuple(scene) in reserved_layouts for scene, _, _ in root_tasks):
                raise ValueError('Coverage sampling must use the three development populations only.')
            if len(root_tasks) != len(set(root_tasks)):
                raise ValueError('Coverage sampling must preserve distinct layout and future identities.')
            previous = torch.load(old_data_path, map_location='cpu', weights_only=False)
            if not previous['completed']:
                raise ValueError('Retain the complete earlier observations, including successful rescues.')
            fresh = torch.load(data_path, map_location='cpu', weights_only=False) if data_path.exists() else None
            if ordinary:
                if fresh is None or not fresh['completed']:
                    raise ValueError('The plain comparison cannot trigger additional coverage sampling.')
            else:
                if fresh is not None and fresh['recipe']['context_policy_sha256'] != hashlib.sha256(behavior.read_bytes()).hexdigest():
                    raise ValueError('Resumed coverage sampling requires the exact behavior policy.')
                fresh = local.collect_local_tasks(base, root_tasks, data_path,
                    workers=workers, repetitions=2, confirmation=8, screen_limit=24,
                    context_policy=behavior, neighborhood_width=3, record_continuations=True,
                    existing_data=fresh, inference_batch=32, inference_device='cpu')
            if not fresh['completed']:
                raise ValueError('Coverage fitting cannot use a partial collection.')
            measured = {(tuple(root['scene']), root['prefix_future'], root['lookback'])
                        for root in fresh['roots']}
            if set(root_tasks) != measured:
                raise ValueError('The completed coverage observations must match the fixed selection.')
            paired = local.merge_paired_data([previous, fresh], allow_distinct_contexts=True)
            factual = local.factual_reference_data(base, references)
            factual['simulation_steps'] = 0
            data = local.merge_paired_data([paired, factual], allow_distinct_contexts=True)
            data['training_data'] = [str(old_data_path), str(data_path), str(bank_path)]
            initial = torch.load(source, map_location='cpu', weights_only=False)
            if ordinary:
                matched = torch.load(conditional, map_location='cpu', weights_only=False)
                if matched['successor_task_fit'].get('execution_revision') != 59:
                    raise ValueError('The coverage fit has not completed its provenance check.')
                bank = local.factual_bank_from_model(matched, data)
                model = fit_plain_return(data, output, factual_bank=bank, updates=6000,
                    parameter_budget=145731, seed=42, uniform_sampling=True,
                    initial_model=initial, learning_rate=2e-4,
                    additional_candidate_options=matched['successor_task_value']['options'],
                    expected_observation_sha256=matched['successor_task_fit']['training_observation_sha256'])
                del matched
            else:
                bank = torch.load(bank_path, map_location='cpu', weights_only=False)
                if not bank['completed'] or bank['predecessor_sha256'] != data['predecessor_sha256']:
                    raise ValueError('Coverage fitting requires the exact current-controller factual bank.')
                continuation = local.paired_continuation_bank(paired)
                bank['records'] += continuation['records']
                bank['paired_continuation_state_labels'] = continuation['state_labels']
                bank['training_data'] = [str(bank_path), str(old_data_path), str(data_path)]
                fit = initial['successor_task_fit']; value = initial['successor_task_value']
                model = local.fit_joint_potential(data, output, factual_bank=bank, device='cuda:0',
                    updates=6000, ensemble_size=len(value['states']), width=fit['width'], seed=fit['seed'],
                    initial_states=value['states'], additional_candidate_options=list(value['options']),
                    factual_difference_transport=True, effective_future_confidence=True, learning_rate=2e-4,
                    edge_weight=fit['difference_weight'], anchor_weight=fit['factual_contrast_weight'],
                    rank_weight=fit['local_rank_weight'], factual_weight=fit['factual_weight'],
                    **{key:value[key] for key in ('bandwidth','margin','confidence','global_penalty',
                        'context_time_scale','interpolation_power','context_conditioned','motion_basis',
                        'local_contrast_head','joint_factual_calibration','residual_potential')})
                model['successor_task_value'].update(inference_device='cuda:0',
                    factual_bound_pruning=True, candidate_bound_pruning=True)
            initial_fit = initial['successor_task_fit']
            initialization_updates = initial_fit.get('total_parameter_updates', initial_fit['updates'])
            model['successor_task_fit'].update(execution_revision=revision,
                initialization_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                initialization_updates=initialization_updates, total_parameter_updates=initialization_updates+6000,
                initialization_simulation_steps=initial_fit['simulation_steps'],
                behavior_checkpoint_sha256=hashlib.sha256(behavior.read_bytes()).hexdigest(),
                fresh_paired_simulation_steps=fresh['simulation_steps'],
                retained_paired_simulation_steps=previous['simulation_steps'],
                additional_simulation_steps=0 if ordinary else fresh['simulation_steps'],
                paired_training_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                retained_training_data_sha256=hashlib.sha256(old_data_path.read_bytes()).hexdigest(),
                actual_replacement_labels=True, inherited_calibration=False,
                adaptation_contexts=len(root_tasks), adaptation_selection=selected_counts)
            atomic_save(model, output)
            del model, initial, data, bank, previous, fresh, paired
            gc.collect()
            torch.cuda.empty_cache()
        saved = torch.load(output, map_location='cpu', weights_only=False)
        if saved['successor_task_fit'].get('execution_revision') != revision:
            raise ValueError('Coverage continuation has not completed its provenance check.')
        del saved
        evaluate(output)
    elif job == 'continue-m1-plain-current':
        base = directory / 'M1-joint-r50-pareto-refinement.pth'
        calibration = directory / 'F11-local-difference-r55-current-local.pth'
        source = directory / 'F11-ordinary-r40-uniform-mlp.pth'
        output = directory / 'F11-ordinary-r58-current-uniform-mlp.pth'
        if not output.exists():
            calibrated = torch.load(calibration, map_location='cpu', weights_only=False)
            if calibrated['successor_task_fit'].get('execution_revision') != 55:
                raise ValueError('The plain continuation requires the completed matched calibration.')
            data_path = calibration.with_name(calibration.stem+'-training-data.pth')
            sampled = torch.load(data_path, map_location='cpu', weights_only=False)
            if not sampled['completed']:
                raise ValueError('The plain continuation cannot use partial paired observations.')
            references = [torch.load(base.with_name(base.stem+'-'+suffix+'.pth'),
                          map_location='cpu', weights_only=False)
                          for suffix in ('source-development', 'new-development', 'fresh-200')]
            factual = local.factual_reference_data(base, references)
            factual['simulation_steps'] = 0
            data = local.merge_paired_data([sampled, factual])
            bank = local.factual_bank_from_model(calibrated, data)
            data['training_data'] = calibrated['successor_task_fit']['training_data']
            initial = torch.load(source, map_location='cpu', weights_only=False)
            model = fit_plain_return(data, output, factual_bank=bank, updates=6000,
                parameter_budget=145731, seed=42, uniform_sampling=True,
                initial_model=initial, learning_rate=2e-4,
                additional_candidate_options=calibrated['successor_task_value']['options'],
                expected_observation_sha256=calibrated['successor_task_fit']['training_observation_sha256'])
            model['successor_task_fit'].update(execution_revision=58,
                initialization_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                initialization_simulation_steps=initial['successor_task_fit']['simulation_steps'],
                paired_calibration_checkpoint_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
                paired_training_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                additional_simulation_steps=0, actual_replacement_labels=True)
            atomic_save(model, output)
            del model, initial, calibrated, data, sampled, bank
            gc.collect()
            torch.cuda.empty_cache()
        saved = torch.load(output, map_location='cpu', weights_only=False)
        if saved['successor_task_fit'].get('execution_revision') != 58:
            raise ValueError('Plain return continuation has not completed its provenance check.')
        del saved
        evaluate(output)
    elif job in ('continue-m1-local-rebase', 'continue-m1-local-learned'):
        learn_parameters = job == 'continue-m1-local-learned'
        revision = 57 if learn_parameters else 55
        base = directory / 'M1-joint-r50-pareto-refinement.pth'
        behavior = directory / 'F11-local-difference-r54-current-factual.pth'
        source = directory / 'F11-local-difference-r49-factual-transport.pth'
        calibration = directory / 'F11-local-difference-r55-current-local.pth'
        output = (directory / 'F11-local-difference-r57-current-learned.pth'
                  if learn_parameters else calibration)
        if learn_parameters and not calibration.exists():
            raise ValueError('Fit the current-controller calibration before learning its paired effects.')
        if not output.exists():
            references = []; root_tasks = []; selected_counts = []
            for suffix in ('source-development', 'new-development', 'fresh-200'):
                records = []
                for path in (base, behavior):
                    record = torch.load(path.with_name(path.stem+'-'+suffix+'.pth'),
                                        map_location='cpu', weights_only=False)
                    if not record['completed'] or record['policy_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise ValueError('Local adaptation requires complete results from its exact controllers.')
                    records.append(record)
                factual, behavioral = records
                if factual['tasks'] != behavioral['tasks']:
                    raise ValueError('Local adaptation requires paired scene and future identities.')
                references.append(factual)
                used_layouts = set(); counts = {}
                # Measure both harmful decisions and residual failures, with
                # successful rescues retained to check the useful decisions.
                for kind, quota in (('harm', 4), ('residual_failure', 6), ('rescue', 2)):
                    count = 0
                    for (scene, future), before, after in zip(
                            factual['tasks'], factual['rows'], behavioral['rows']):
                        category = ('harm' if before['success'] and not after['success'] else
                                    'residual_failure' if not after['success'] else
                                    'rescue' if not before['success'] else 'unchanged_success')
                        if category != kind or scene[0] in used_layouts:
                            continue
                        used_layouts.add(scene[0])
                        if after['conditional_interventions']:
                            root = 'first_intervention' if kind == 'harm' else 'last_intervention'
                        else:
                            root = 40 if after['outcome'] == 2 else 120
                        root_tasks.append((scene, future, root)); count += 1
                        if count == quota:
                            break
                    counts[kind] = count
                selected_counts.append(dict(suite=suffix, **counts))
            if not root_tasks or any(tuple(scene) in reserved_layouts for scene, _, _ in root_tasks):
                raise ValueError('Fresh local calibration needs nonempty development-only sampling.')
            data_path = calibration.with_name(calibration.stem+'-training-data.pth')
            previous = torch.load(data_path, map_location='cpu', weights_only=False) if data_path.exists() else None
            if learn_parameters and (previous is None or not previous.get('completed')):
                raise ValueError('Current-controller learning reuses the complete, fixed paired observations.')
            if learn_parameters:
                measured = {(tuple(root['scene']), root['prefix_future'], root['lookback'])
                            for root in previous['roots']}
                if set(root_tasks) != measured:
                    raise ValueError('Parameter adaptation must not change the completed calibration design.')
            sampled = local.collect_local_tasks(base, root_tasks, data_path,
                workers=workers, repetitions=2, confirmation=6, screen_limit=12,
                context_policy=behavior, neighborhood_width=2, record_continuations=True,
                existing_data=previous, inference_batch=32, inference_device='cpu')
            if not sampled['completed']:
                raise ValueError('Fresh local calibration cannot use a partial collection.')
            factual = local.factual_reference_data(base, references)
            factual['simulation_steps'] = 0
            data = local.merge_paired_data([sampled, factual])
            bank_path = directory / 'F11-local-difference-r54-current-factual-factual-data.pth'
            bank = torch.load(bank_path, map_location='cpu', weights_only=False)
            if not bank['completed'] or bank['predecessor_sha256'] != data['predecessor_sha256']:
                raise ValueError('Use the completed factual trajectories of this exact M1.')
            continuation = local.paired_continuation_bank(sampled)
            bank['records'] += continuation['records']
            bank['paired_continuation_state_labels'] = continuation['state_labels']
            bank['training_data'] = [str(bank_path), str(data_path)]
            data['training_data'] = [str(data_path), str(bank_path)]
            reference = torch.load(source, map_location='cpu', weights_only=False)
            fit = reference['successor_task_fit']; value = reference['successor_task_value']
            updates = 6000 if learn_parameters else 0
            model = local.fit_joint_potential(data, output, factual_bank=bank, device='cuda:0',
                updates=updates, ensemble_size=len(value['states']), width=fit['width'], seed=fit['seed'],
                initial_states=value['states'], additional_candidate_options=list(value['options']),
                factual_difference_transport=True, effective_future_confidence=True,
                learning_rate=2e-4 if learn_parameters else 8e-4,
                edge_weight=fit['difference_weight'], anchor_weight=fit['factual_contrast_weight'],
                rank_weight=fit['local_rank_weight'], factual_weight=fit['factual_weight'],
                **{key:value[key] for key in ('bandwidth','margin','confidence','global_penalty',
                    'context_time_scale','interpolation_power','context_conditioned','motion_basis',
                    'local_contrast_head','joint_factual_calibration','residual_potential')})
            model['successor_task_value'].update(inference_device='cuda:0',
                factual_bound_pruning=True, candidate_bound_pruning=True)
            model['successor_task_fit'].update(execution_revision=revision,
                initialization_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                initialization_updates=fit['updates'], total_parameter_updates=fit['updates']+updates,
                initialization_simulation_steps=fit['simulation_steps'],
                behavior_checkpoint_sha256=hashlib.sha256(behavior.read_bytes()).hexdigest(),
                fresh_paired_simulation_steps=sampled['simulation_steps'],
                actual_replacement_labels=True, inherited_calibration=False,
                adaptation_contexts=len(root_tasks), adaptation_selection=selected_counts)
            if learn_parameters:
                model['successor_task_fit'].update(
                    paired_calibration_checkpoint_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
                    paired_training_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    additional_simulation_steps=0)
            atomic_save(model, output)
            del reference, model, data, bank, sampled
            gc.collect()
            torch.cuda.empty_cache()
        saved = torch.load(output, map_location='cpu', weights_only=False)
        if saved['successor_task_fit'].get('execution_revision') != revision:
            raise ValueError('Fresh local calibration has not completed its provenance check.')
        del saved
        evaluate(output)
    elif job == 'continue-m1-factual-rebase':
        base = directory / 'M1-joint-r50-pareto-refinement.pth'
        source = directory / 'F11-local-difference-r49-factual-transport.pth'
        output = directory / 'F11-local-difference-r54-current-factual.pth'
        if not output.exists():
            references = []
            for suffix in ('source-development', 'new-development', 'fresh-200'):
                record = torch.load(base.with_name(base.stem+'-'+suffix+'.pth'),
                                    map_location='cpu', weights_only=False)
                if not record['completed'] or record['policy_sha256'] != hashlib.sha256(base.read_bytes()).hexdigest():
                    raise ValueError('Current factual calibration needs the exact replacement controller.')
                references.append(record)
            bank_path = output.with_name(output.stem+'-factual-data.pth')
            bank = local.collect_factual_trajectories(
                base, references, bank_path, workers=workers, inference_batch=32, inference_device='cpu')
            model = torch.load(source, map_location='cpu', weights_only=False)
            replacement = torch.load(base, map_location='cpu', weights_only=False)
            model['actuator_fusion_refinement'] = copy.deepcopy(replacement['actuator_fusion_refinement'])
            value = model['successor_task_value']
            for field, key in (('continuation_contexts', 'contexts'), ('continuation_controls', 'controls'),
                               ('continuation_values', 'values'), ('continuation_remaining', 'remaining')):
                value[field] = torch.as_tensor(np.concatenate(
                    [record[key] for record in bank['records']]).astype(np.float32))
            value.update(factual_values=None, continuation_factual_only=True,
                         factual_difference_transport=True)
            model['successor_task_fit'].update(execution_revision=54,
                transferred_difference_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                factual_calibration_predecessor_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                factual_calibration_data=str(bank_path),
                factual_calibration_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                additional_factual_simulation_steps=bank['simulation_steps'],
                additional_factual_trajectory_labels=bank['trajectories'],
                inherited_difference_prior=True, inherited_factual_calibration=False,
                additional_parameter_updates=0)
            atomic_save(model, output)
        evaluate(output)
    elif job == 'continue-m1-pareto-transport':
        source = directory / 'F11-local-difference-r49-factual-transport.pth'
        fusion = directory / 'M1-joint-r50-pareto-refinement.pth'
        output = directory / 'F11-local-difference-r53-pareto-transport.pth'
        if not output.exists():
            model = torch.load(source, map_location='cpu', weights_only=False)
            replacement = torch.load(fusion, map_location='cpu', weights_only=False)
            if (model['successor_task_fit'].get('execution_revision') != 49
                    or not replacement['actuator_fusion_refinement'].get('pareto_refinement')):
                raise ValueError('Composition requires both completed mechanism candidates.')
            model['actuator_fusion_refinement'] = copy.deepcopy(replacement['actuator_fusion_refinement'])
            model['fusion_optimization_revision'] = dict(
                reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                fusion_sha256=hashlib.sha256(fusion.read_bytes()).hexdigest(),
                method='Pareto fleet refinement with factual-transported local task differences',
                retained_value_fit=True, additional_value_training=False)
            atomic_save(model, output)
        evaluate(output)
    elif job in ('continue-m1-resolution', 'continue-m1-interception'):
        base = d.ROOT / 'joint-128-factorial-development/F10-mapped-actuator-joint.pth'
        identity = hashlib.sha256(base.read_bytes()).hexdigest()
        if identity != 'c6a85079455fdb5ac00f8b3ca2360fbefcf72ff8be80a8f6217734886927f32d':
            raise ValueError('The source joint-fusion initialization changed.')
        diagnostics, phase_context = exposed_development()
        interception = job == 'continue-m1-interception'
        phase_context['development_phase'] = ('fleet-interception-candidate-objective' if interception
                                               else 'source-preserving-candidate-resolution')
        configurations = ([(92, 1, True, .5), (93, 1, True, 1.5), (94, 2, True, .5)] if interception else
                          [(87, 2, False, 0.), (88, 2, True, 0.), (89, 3, False, 0.)])
        method = 'fleet-interception' if interception else 'resolved-candidates'
        for revision, subdivisions, refine_all, capture_weight in configurations:
            output = directory / f'M1-joint-r{revision}-{method}.pth'
            if not output.exists():
                model = torch.load(base, map_location='cpu', weights_only=False)
                model['actuator_fusion_refinement'].update(
                    candidate_subdivisions=subdivisions, refine_all_plans=refine_all)
                if capture_weight:
                    model['actuator_fusion_refinement']['capture_weight'] = capture_weight
                model['fusion_optimization_revision'] = dict(phase_context,
                    execution_revision=revision, reference_sha256=identity,
                    method=('joint fleet minimum interception distance and pair risk' if interception
                        else 'joint selection over a refined source-preference candidate grid'),
                    actuator_candidates_per_vessel=(2*subdivisions+1)**2,
                    capture_weight=capture_weight,
                    additional_parameter_updates=0, additional_simulation_steps=0,
                    matched_control='independent choice with the identical candidate grid and prediction')
                atomic_save(model, output)
            evaluate_exposed(output, diagnostics, phase_context)
    elif job in ('continue-m1-feedback-guard', 'continue-m1-componentwise'):
        base = d.ROOT / 'joint-128-factorial-development/F10-mapped-actuator-joint.pth'
        identity = hashlib.sha256(base.read_bytes()).hexdigest()
        if identity != 'c6a85079455fdb5ac00f8b3ca2360fbefcf72ff8be80a8f6217734886927f32d':
            raise ValueError('The frozen joint-fusion predecessor changed.')
        diagnostics, phase_context = exposed_development()
        componentwise = job == 'continue-m1-componentwise'
        revision = 74 if componentwise else 69
        output = directory / ('M1-joint-r74-componentwise.pth' if componentwise
                              else 'M1-joint-r69-feedback-guard.pth')
        if not output.exists():
            model = torch.load(base, map_location='cpu', weights_only=False)
            model['actuator_fusion_refinement'].update(refine_all_plans=True, pair_refinements=1,
                pareto_refinement=True)
            if componentwise:
                model['actuator_fusion_refinement']['pareto_componentwise'] = True
            else:
                model['actuator_fusion_refinement']['pareto_feedback_horizons'] = (.8, 1.6, 3.2)
            model['fusion_optimization_revision'] = dict(phase_context,
                execution_revision=revision, reference_sha256=identity,
                method=('Pareto fleet proposal preserving every vessel deviation and pair risk'
                        if componentwise else 'Pareto fleet proposal guarded by shared source feedback at three horizons'),
                additional_parameter_updates=0, additional_simulation_steps=0,
                additional_value_training=False, retained_value_fit=False)
            atomic_save(model, output)
        saved = torch.load(output, map_location='cpu', weights_only=False)
        if (saved['fusion_optimization_revision'].get('execution_revision') != revision
                or saved['fusion_optimization_revision'].get('reference_sha256') != identity):
            raise ValueError('The feedback guard must retain its exact frozen predecessor.')
        del saved
        evaluate_exposed(output, diagnostics, phase_context)
    elif job in ('continue-m1-search', 'continue-m1-transfer', 'continue-m1-pair-block',
                 'continue-m1-feedback-shortlist', 'continue-m1-pareto', 'continue-m1-pareto-transfer'):
        base = d.ROOT / 'joint-128-factorial-development/F10-mapped-actuator-joint.pth'
        if hashlib.sha256(base.read_bytes()).hexdigest() != 'c6a85079455fdb5ac00f8b3ca2360fbefcf72ff8be80a8f6217734886927f32d':
            raise ValueError('The frozen joint-fusion predecessor changed.')
        pareto = job in ('continue-m1-pareto', 'continue-m1-pareto-transfer')
        transfer = job in ('continue-m1-transfer', 'continue-m1-pareto-transfer')
        output = directory / ('F11-local-difference-r51-pareto-fusion.pth' if pareto and transfer
                              else 'M1-joint-r50-pareto-refinement.pth' if pareto
                              else 'F11-local-difference-r43-refined-fusion.pth' if transfer
                              else 'M1-joint-r48-feedback-shortlist.pth' if job == 'continue-m1-feedback-shortlist'
                              else 'M1-joint-r46-pair-block.pth' if job == 'continue-m1-pair-block'
                              else 'M1-joint-r40-refined-beam.pth')
        if not output.exists():
            source = directory / (anchors['F11'][0]+'.pth') if transfer else base
            model = torch.load(source, map_location='cpu', weights_only=False)
            model['actuator_fusion_refinement']['refine_all_plans'] = True
            if job == 'continue-m1-pair-block' or pareto:
                model['actuator_fusion_refinement']['pair_refinements'] = 1
            if pareto:
                model['actuator_fusion_refinement']['pareto_refinement'] = True
            if job == 'continue-m1-feedback-shortlist':
                model['actuator_fusion_refinement'].update(
                    joint_feedback_topk=8, joint_feedback_horizon=3.2)
            model['fusion_optimization_revision'] = dict(
                reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                method=('Pareto improvement of fleet interaction risk and source-control deviation'
                        if pareto else 'joint two-vessel block refinement of the retained fleet plans'
                        if job == 'continue-m1-pair-block' else
                        'coupled source-feedback prediction of eight distinct proposed fleet maneuvers'
                        if job == 'continue-m1-feedback-shortlist' else
                        'refine all retained joint plans under the unchanged interaction objective'),
                retained_value_fit=transfer, additional_value_training=False)
            atomic_save(model, output)
        evaluate(output)
    else:
        raise ValueError('Unknown current-mechanism continuation: '+job)
    for path, identity in protected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity
    print(dict(phase='continuation_finished', job=job, fixed_anchors_unchanged=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=('all', 'mechanisms', 'full', 'without-m1', 'reference', 'legacy', 'baseline',
                                      'feedback-full','feedback-without-m1','joint-roles',
                                      'local-feedback-full','local-feedback-without-m1',
                                      'supported-feedback-full','supported-feedback-without-m1','joint-roles-without-m1',
                                      'proposal-full','proposal-without-m1','credit-full','credit-without-m1',
                                      'grounded-full','grounded-without-m1','generalized-full','generalized-without-m1',
                                      'response-full','response-without-m1','balanced-full','balanced-without-m1',
                                      'balanced-credit-full','balanced-credit-without-m1',
                                      'continuation-full','continuation-without-m1','contexts-full','contexts-without-m1',
                                      'wait-credit-full','wait-credit-without-m1','context-credit-full','context-credit-without-m1',
                                      'geometric-credit-full','geometric-credit-without-m1',
                                      'recurrent-credit-full','recurrent-credit-without-m1',
                                      'factual-credit-full','factual-credit-without-m1',
                                      'recurrent-confirm-full','recurrent-confirm-without-m1',
                                      'fusion-policy-full','fusion-policy-without-m2','fusion-policy-without-m1','fusion-policy-neither',
                                      'source-suite-sac','source-suite-rp','source-suite-arboids','boids-reference',
                                      'source-marl-mappo','source-marl-maac',
                                      'fusion-confirm-full','fusion-confirm-without-m1',
                                      'temporal-terminal-full','temporal-terminal-without-m1',
                                      'temporal-fit-full','temporal-fit-without-m1','outcome-fit-full','outcome-fit-without-m1',
                                      'rank-fit-full','rank-fit-without-m1','commitment-confirm-full',
                                      'bellman-fit-full','bellman-fit-without-m1','early-role-full','early-role-without-m1',
                                      'local-temporal-full','local-temporal-without-m1','local-initial-full','local-initial-without-m1',
                                      'conditional-search-full','conditional-search-without-m1','independent-prediction','feedback-prediction',
                                      'source-frozen-benchmarks','continued-credit-full','continued-credit-without-m1',
                                      'role-persistence-full','role-persistence-without-m1',
                                      'threat-credit-full','threat-credit-without-m1',
                                      'survival-credit-full','survival-credit-without-m1',
                                      'kernel-credit-full','kernel-credit-without-m1',
                                      'grid-credit-full','grid-credit-without-m1',
                                      'integrable-credit-full','integrable-credit-without-m1',
                                      'hazard-credit-full','hazard-credit-without-m1',
                                      'forecast-credit-full','forecast-credit-without-m1',
                                      'finite-hazard-full','finite-hazard-without-m1','fleet-profile','native-fleet',
                                      'contrast-credit-full','contrast-credit-without-m1',
                                      'replicated-credit-full','replicated-credit-without-m1','decoupled-forecast',
                                      'continue-m1-controls','continue-m2-bounds','continue-m2-local-bounds',
                                      'continue-m2-conditional-fit','continue-m2-population-fit',
                                      'continue-m2-dual-motion','continue-m2-dual-population',
                                      'continue-m2-response','continue-m2-dual-response',
                                      'continue-m2-short-response',
                                      'continue-m2-factual-transport',
                                      'continue-m2-population-transport',
                                      'continue-m2-evidence-confidence',
                                      'continue-plain','continue-m1-search','continue-m1-transfer',
                                      'continue-m1-pair-block',
                                      'continue-m1-pareto', 'continue-m1-pareto-transfer',
                                      'continue-m1-pareto-transport',
                                      'continue-m1-factual-rebase',
                                      'continue-m1-local-rebase',
                                      'continue-m1-local-learned',
                                      'continue-m1-plain-current',
                                      'continue-m1-coverage', 'continue-m1-coverage-plain',
                                      'continue-m2-single-intervention', 'continue-m2-strong-confidence',
                                      'continue-m1-single-intervention', 'continue-m1-strong-confidence',
                                      'continue-m2-outcome-confidence',
                                      'continue-m2-factual-risk', 'continue-m1-factual-risk',
                                      'continue-m1-feedback-guard',
                                      'continue-m1-componentwise', 'continue-m1-resolution', 'continue-m1-interception',
                                      'continue-current-finite-motion',
                                      'continue-m1-feedback-shortlist',
                                      'continue-fixed-reference','continue-m1-adapt','source-attacker'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=('F11', 'F01'), default=('F11', 'F01'))
    parser.add_argument('--control-modes', nargs='+', choices=('independent', 'single_reference'),
                        default=('independent', 'single_reference'))
    parser.add_argument('--training-seeds',type=int,nargs='+')
    parser.add_argument('--training-defenders',type=int,default=6)
    parser.add_argument('--attacker-steps', type=int, default=500000)
    parser.add_argument('--attacker-defender', type=Path,
                        default=d.ROOT/'joint-complete-task-development/F00-source-clean.pth')
    parser.add_argument('--fleet-models',nargs='+',choices=('full','full-invariant','full-clean','m1','m2','neither','arboids-three','mission-m1','mission-full',
                        *(f'native-{n}-{arm}' for n in (3,4) for arm in ('full','m1','m2','neither'))),
                        default=('full-clean','m1','m2','neither','full'))
    parser.add_argument('--fleet-counts',type=int,nargs='+',default=(2,3,4,5,6,7,8))
    parser.add_argument('--fleet-per-cell',type=int,default=64)
    args = parser.parse_args()
    os.chdir(REPO)
    if args.job == 'source-attacker':
        runtime = {name:hashlib.sha256((REPO/'train'/name).read_bytes()).hexdigest() for name in
                   ('develop_source_fusion.py','conditional_fusion_value.py','local_intervention_value.py',
                    'plain_return.py','mechanism_controls.py','run_remote_evidence.py')}
        for seed in args.training_seeds or (42,):
            train_source_attacker(args.attacker_defender,
                d.ROOT/'adversarial-learning'/f'{args.attacker_defender.stem}-seed{seed}',
                seed=seed, defenders=args.training_defenders, steps=args.attacker_steps, runtime_sha256=runtime)
    elif args.job.startswith('continue-'):
        continue_current_mechanisms(args.job, workers=args.workers, fit_only=args.fit_only,
                                   arms=args.arms, control_modes=args.control_modes)
    elif args.job in ('all', 'mechanisms'):
        supervise(include_baseline=args.job == 'all')
    elif args.job == 'full':
        run_arm(False)
    elif args.job == 'without-m1':
        run_arm(True)
    elif args.job == 'reference':
        evaluate(RUNS / 'six-trained-interaction-185/policy.pth', REFERENCE, workers=3,
                 references=(RUNS / 'six-trained-baseline-184/policy.pth',))
    elif args.job == 'legacy':
        run_legacy()
    elif args.job in ('feedback-full','feedback-without-m1'):
        run_feedback(args.job=='feedback-without-m1')
    elif args.job in ('joint-roles','joint-roles-without-m1'):
        run_joint_roles(args.job=='joint-roles-without-m1')
    elif args.job in ('local-feedback-full','local-feedback-without-m1'):
        run_local_feedback(args.job=='local-feedback-without-m1')
    elif args.job in ('supported-feedback-full','supported-feedback-without-m1'):
        run_supported_feedback(args.job=='supported-feedback-without-m1')
    elif args.job in ('proposal-full','proposal-without-m1'):
        run_proposal_probe(args.job=='proposal-without-m1')
    elif args.job in ('credit-full','credit-without-m1'):
        run_credit_learning(args.job=='credit-without-m1')
    elif args.job in ('grounded-full','grounded-without-m1'):
        run_credit_learning(args.job=='grounded-without-m1',grounded=True)
    elif args.job in ('generalized-full','generalized-without-m1'):
        run_generalized_credit(args.job=='generalized-without-m1')
    elif args.job in ('response-full','response-without-m1'):
        run_credit_learning(args.job=='response-without-m1',grounded=True,response=True)
    elif args.job in ('balanced-full','balanced-without-m1'):
        run_balanced_roles(args.job=='balanced-without-m1')
    elif args.job in ('balanced-credit-full','balanced-credit-without-m1'):
        run_balanced_credit(args.job=='balanced-credit-without-m1')
    elif args.job in ('continuation-full','continuation-without-m1'):
        run_credit_continuation(args.job=='continuation-without-m1')
    elif args.job in ('contexts-full','contexts-without-m1'):
        run_conditional_contexts(args.job=='contexts-without-m1')
    elif args.job in ('wait-credit-full','wait-credit-without-m1'):
        run_wait_credit(args.job=='wait-credit-without-m1')
    elif args.job in ('context-credit-full','context-credit-without-m1'):
        run_balanced_credit(args.job=='context-credit-without-m1',contexts=True)
    elif args.job in ('geometric-credit-full','geometric-credit-without-m1'):
        run_balanced_credit(args.job=='geometric-credit-without-m1',contexts=True,task_frame=True)
    elif args.job in ('recurrent-credit-full','recurrent-credit-without-m1'):
        run_recurrent_credit(args.job=='recurrent-credit-without-m1')
    elif args.job in ('factual-credit-full','factual-credit-without-m1'):
        run_factual_credit(args.job=='factual-credit-without-m1')
    elif args.job in ('recurrent-confirm-full','recurrent-confirm-without-m1'):
        run_recurrent_confirmation(args.job=='recurrent-confirm-without-m1')
    elif args.job.startswith('fusion-policy-'):
        run_fusion_policy_learning(args.job in ('fusion-policy-without-m1','fusion-policy-neither'),
            args.job in ('fusion-policy-full','fusion-policy-without-m1'))
    elif args.job.startswith('source-suite-'):
        run_source_baseline_suite(args.job.removeprefix('source-suite-'),args.training_seeds or range(42,47),args.training_defenders)
    elif args.job.startswith('source-marl-'):
        run_source_marl_suite(args.job.removeprefix('source-marl-'),args.training_seeds or range(42,47))
    elif args.job=='boids-reference':
        run_boids_reference()
    elif args.job=='fleet-profile':
        run_fleet_profile(args.fleet_models,args.fleet_counts,args.fleet_per_cell)
    elif args.job=='native-fleet':
        if args.training_defenders not in (3,4):raise ValueError('Native-fleet candidates currently have matched proposal training at three and four vessels.')
        for without_m1 in (False,True):run_finite_hazard_learning(without_m1,args.training_defenders)
        run_fleet_profile([f'native-{args.training_defenders}-{arm}' for arm in ('full','m1','m2','neither')],per_cell=64)
    elif args.job in ('fusion-confirm-full','fusion-confirm-without-m1'):
        run_fusion_credit_confirmation(args.job=='fusion-confirm-without-m1')
    elif args.job in ('temporal-terminal-full','temporal-terminal-without-m1'):
        run_temporal_terminal_credit(args.job=='temporal-terminal-without-m1')
    elif args.job in ('temporal-fit-full','temporal-fit-without-m1'):
        run_temporal_credit_fit(args.job=='temporal-fit-without-m1')
    elif args.job in ('outcome-fit-full','outcome-fit-without-m1'):
        run_temporal_credit_fit(args.job=='outcome-fit-without-m1',paired_outcomes=True)
    elif args.job in ('rank-fit-full','rank-fit-without-m1'):
        run_temporal_credit_fit(args.job=='rank-fit-without-m1',paired_outcomes=True,rank_credit=True)
    elif args.job=='commitment-confirm-full':
        run_committed_credit_confirmation()
    elif args.job in ('bellman-fit-full','bellman-fit-without-m1'):
        run_bellman_credit(args.job=='bellman-fit-without-m1')
    elif args.job in ('early-role-full','early-role-without-m1'):
        run_early_role_probe(args.job=='early-role-without-m1')
    elif args.job in ('local-temporal-full','local-temporal-without-m1','local-initial-full','local-initial-without-m1'):
        run_local_effect_credit(args.job.endswith('without-m1'),temporal=args.job.startswith('local-temporal'))
    elif args.job in ('conditional-search-full','conditional-search-without-m1'):
        run_conditional_role_search(args.job.endswith('without-m1'))
    elif args.job=='independent-prediction':
        run_independent_prediction_control()
    elif args.job=='feedback-prediction':
        run_relaxed_prediction_control()
    elif args.job=='source-frozen-benchmarks':
        run_source_frozen_benchmarks()
    elif args.job in ('continued-credit-full','continued-credit-without-m1'):
        run_continued_credit_learning(args.job.endswith('without-m1'))
    elif args.job in ('contrast-credit-full','contrast-credit-without-m1'):
        run_contrast_credit_learning(args.job.endswith('without-m1'))
    elif args.job in ('replicated-credit-full','replicated-credit-without-m1'):
        run_contrast_credit_learning(args.job.endswith('without-m1'),refined=True)
    elif args.job=='decoupled-forecast':
        run_decoupled_forecast_learning()
    elif args.job in ('role-persistence-full','role-persistence-without-m1'):
        run_role_persistence_probe(args.job.endswith('without-m1'))
    elif args.job in ('threat-credit-full','threat-credit-without-m1'):
        run_threat_credit_learning(args.job.endswith('without-m1'))
    elif args.job in ('survival-credit-full','survival-credit-without-m1'):
        run_threat_credit_learning(args.job.endswith('without-m1'),temporal_return=True)
    elif args.job in ('kernel-credit-full','kernel-credit-without-m1'):
        run_threat_credit_learning(args.job.endswith('without-m1'),kernel=True)
    elif args.job in ('grid-credit-full','grid-credit-without-m1'):
        run_grid_credit_learning(args.job.endswith('without-m1'))
    elif args.job in ('integrable-credit-full','integrable-credit-without-m1'):
        run_integrable_credit_learning(args.job.endswith('without-m1'))
    elif args.job in ('hazard-credit-full','hazard-credit-without-m1'):
        run_hazard_credit_learning(args.job.endswith('without-m1'))
    elif args.job in ('forecast-credit-full','forecast-credit-without-m1'):
        run_hazard_credit_learning(args.job.endswith('without-m1'),prediction_features=True)
    elif args.job in ('finite-hazard-full','finite-hazard-without-m1'):
        run_finite_hazard_learning(args.job.endswith('without-m1'),args.training_defenders)
    else:
        run_baseline()
