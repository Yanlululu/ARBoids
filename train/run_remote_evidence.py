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
def batched_credit_rows(policy,tasks,batch_size=32,device='cuda:0',keep_steps=False,sample_policy=False,initial_jobs=None):
    """Batch deterministic policy inference, retaining each episode's RNG."""
    from envs.TADgame import TADEnv
    initialize();torch.backends.cuda.matmul.allow_tf32=False
    actor=d.joint_load(str(policy),Path(policy).stat().st_mtime_ns).to(device)
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
        returns=np.zeros(len(current));task_returns=np.zeros(len(current));rows=[None]*len(current)
        intervention_events=[[] for _ in current]
        decision_records=[[] for _ in current] if keep_steps else None
        trajectories=[[] for _ in current] if keep_steps else None
        def packets(indices):
            records=[d.joint_packet(environments[i],observations[i]) for i in indices]
            for record in records:record.setdefault('_conditional_count',np.zeros(1,np.float32))
            return {key:torch.as_tensor(np.stack([record[key] for record in records]),device=device) for key in records[0]}
        for step in range(400):
            if not active:break
            controls={};fresh=[i for i in active if remaining[i]==0]
            if fresh:
                decisions=(np.stack([initial_jobs[begin+i]['action'] for i in fresh])
                    if initial_jobs is not None and step==0 else
                    actor(packets(fresh),deterministic=not sample_policy)[0].cpu().numpy())
                for i,action in zip(fresh,decisions):
                    if keep_steps:decision_records[i].append(dict(step=step,action=action.copy()))
                    controls[i]=action[:,:3]
                    intervene=(action.shape[-1]>3 and ((action[:,3]>=0.).any()
                        or (action.shape[-1]>5 and (action[:,5]>=0.).any())))
                    waiting=(actor.cooperative_role_credit is not None and actor.cooperative_role_credit.wait_span)
                    waiting=waiting or (action.shape[-1]>5 and (action[:,3]<0.).all() and (action[:,5]==-2.).all())
                    if action.shape[-1]>3 and (intervene or waiting):
                        options[i]=action;remaining[i]=int(action[0,4]);ticks[i]=0
                        midpoint_checked[i]=False
                        if intervene:
                            environments[i]._conditional_option_count=getattr(environments[i],'_conditional_option_count',0)+1
                            intervention_events[i].append(dict(time=float(environments[i].Current_T),
                                duration_steps=int(action[0,4]),subjects=np.flatnonzero(action[:,3]>=0.).tolist(),
                                coefficients=action[action[:,3]>=0.,3].tolist()))
            committed=[i for i in active if remaining[i]>0]
            if committed:
                batch=packets(committed)
                fixed=torch.as_tensor(np.stack([options[i][:,5 if ticks[i]>=d.joint_phase_boundary(options[i]) else 3]
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
            for i in active:
                env=environments[i];np.random.set_state(random_states[i])
                before=d.joint_packet(env,observations[i]) if keep_steps else None
                obs,reward,done,sr=d.joint_task_step(env,controls[i],actor.training_gamma,actor.training_step_cost)
                if keep_steps:
                    trajectories[i].append(dict(packet=before,action=controls[i].copy(),done=done,physical_steps=1,
                        reward=reward,source_reward=sr,discount=actor.training_gamma))
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
                    rows[i]=row
                else:following.append(i)
            active=following
        if active:raise RuntimeError('Evaluation did not reach the unchanged task terminal.')
        if keep_steps:
            for row,steps,decisions in zip(rows,trajectories,decision_records):
                continuation=0.
                for record in reversed(steps):
                    continuation=record['source_reward']+.99*continuation
                    record['source_return']=continuation
                results.append(dict(row=row,steps=steps,decisions=decisions,branches=[],value_steps=[],simulation_steps=len(steps)))
        else:results.extend(rows)
        print(dict(phase='batched_credit_evaluation',model=str(policy),episodes=len(results),total=len(tasks)),flush=True)
    actor.cpu()
    return results


def collect_joint_credit_batched(policy,output,tasks,batch_size=32,repetitions=2,device='cuda:0'):
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
    factual=(previous['factual'] if previous is not None else
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
                        random_state=np.random.get_state(),packet=d.joint_packet(env,obs))
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
                            jobs.append(dict(environment=root['environment'],observation=root['observation'],
                                random_state=root['random_state'],action=action))
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
                                      'replicated-credit-full','replicated-credit-without-m1','decoupled-forecast'))
    parser.add_argument('--training-seeds',type=int,nargs='+')
    parser.add_argument('--training-defenders',type=int,default=6)
    parser.add_argument('--fleet-models',nargs='+',choices=('full','full-invariant','full-clean','m1','m2','neither','arboids-three','mission-m1','mission-full',
                        *(f'native-{n}-{arm}' for n in (3,4) for arm in ('full','m1','m2','neither'))),
                        default=('full-clean','m1','m2','neither','full'))
    parser.add_argument('--fleet-counts',type=int,nargs='+',default=(2,3,4,5,6,7,8))
    parser.add_argument('--fleet-per-cell',type=int,default=64)
    args = parser.parse_args()
    os.chdir(REPO)
    if args.job in ('all', 'mechanisms'):
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
