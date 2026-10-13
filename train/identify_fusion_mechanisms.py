"""Frozen-policy, paired-environment identification of candidate fusion.

All outcomes use the real TADEnv and recomputed feedback candidates. Learned arms
share the CBF execution layer; the original ARBoids reference bypasses it.
Scenario splits and future innovations are separate. Learned gate actors use
frozen critics during fitting and public current information during deployment.
"""
import study_runtime
import argparse
import copy
import hashlib
import json
import sys
import time
from types import MethodType
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from envs.TADgame import TADEnv
from envs.snapshot import seed_random, preserved_random_state
from interaction_rollout import public_packet, execute, compact_snapshot, safety_controller
from train_interaction import outcome_row, atomic_json, atomic_torch
from train_response_fusion import ResponsePolicy, public_intent_history, DifferentiableInterventionDynamics, safety_derivatives

SOURCE = Path(r'D:\ARBoids\train\experiments\paper-parameters-seed42-20260921\adares1.pth')
CHECKPOINT = Path(__file__).parent/'experiments/ia-crrl-paired-factors/candidate-8042.pth'
OUTPUT = Path(__file__).parent/'experiments/ia-crrl-paired-factors/identification'
POLICIES = None
GATE_POLICIES = None
VALUE_ARMS=('td','absolute','difference','difference_raw','difference_no_peer')
EVENT_TIMES=(0.,1.,2.,3.,4.,6.,8.,12.,16.,24.,32.,48.,60.)


def initialize(source=str(SOURCE), checkpoint=str(CHECKPOINT)):
    global POLICIES
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    POLICIES = {}
    for probe in ('aligned', 'permuted', 'state-only'):
        policy = ResponsePolicy(source, checkpoint)
        policy.fusion_scale = .5
        policy.horizon = 10
        policy.model.candidate_probe = probe
        POLICIES[probe] = policy
    safety_controller(5)


def reset_policy(policy, memory=None):
    policy.remaining, policy.current = 0, None
    policy.history, policy.last_time = [], None
    if memory:
        policy.history = copy.deepcopy(memory['history'])
        policy.last_time = memory['last_time']


def policy_memory(policy):
    return dict(history=copy.deepcopy(policy.history), last_time=policy.last_time)


def effective_weights(policy, packet):
    base = policy.source(packet['obs'])
    return np.clip(base[:,2,None] + policy.current, 0., 1.)


def filtered(packet, action, boids):
    return safety_controller(len(action)).control(packet['motion'], action, boids)[1]


def batch_filtered(packets,actions,boids):
    """Vectorize the unchanged upstream CBF QP; no model dynamics are used."""
    # A large vmap waits for the slowest elastic QP at every Newton iteration.
    # Small fixed groups avoid multiplying rare difficult solves across the
    # entire counterfactual bank, and keep compilation/memory bounded.
    if len(actions)>32:
        return np.concatenate([batch_filtered(packets[i:i+32],actions[i:i+32],boids[i:i+32])
            for i in range(0,len(actions),32)])
    from source_arboids import source_mixture
    motion=np.stack([packet['motion'] for packet in packets]).astype(np.float64)
    nominal=np.stack([source_mixture(action,rule) for action,rule in zip(actions,boids)]).astype(np.float64)
    count=len(actions);capacity=max(8,1<<(count-1).bit_length())
    if capacity>count:
        motion=np.concatenate((motion,np.repeat(motion[-1:],capacity-count,0)))
        nominal=np.concatenate((nominal,np.repeat(nominal[-1:],capacity-count,0)))
    forward,_=safety_derivatives(motion.shape[1])
    applied=np.asarray(forward(motion,nominal))[:count].copy()
    if not np.isfinite(applied).all():raise FloatingPointError('Non-finite batched CBF controls.')
    return applied


def source_current_motion(boat):
    """Scalar form of the released WAMV matrix acceleration, including current.

    Pose integration, measured-velocity lag, currents, clipping, APF, Boids and
    termination remain in TADEnv/WAMV. In particular, the source's measured yaw
    rate in V_r is retained; this is not the zero-current prediction backend.
    """
    c,s=np.cos(boat.theta),np.sin(boat.theta)
    ur=c*boat.velocity_r[0]+s*boat.velocity_r[1]
    vr=-s*boat.velocity_r[0]+c*boat.velocity_r[1]
    u=c*boat.velocity[0]+s*boat.velocity[1]
    v=-s*boat.velocity[0]+c*boat.velocity[1];r=boat.velocity[2]
    left,right=boat.left_thrust,boat.right_thrust
    b0=400.*r*v-(100.+150.*abs(ur))*ur+(left+right)
    b1=-400.*r*u-((100.+150.*abs(vr))*vr-20.*ur*r)
    b2=-(20.*ur*vr+(980.+950.*abs(r))*r)+(left*1.25-right*1.25)
    inv=boat._identification_mass_inverse
    ur+=inv[0]*b0*boat.dt;vr+=inv[1]*b1*boat.dt;r+=inv[2]*b2*boat.dt
    boat.velocity_r=np.array([c*ur-s*vr,s*ur+c*vr,r])


def use_source_current_motion(env):
    for boat in [env.attacker,*env.defender_list]:
        expected=dict(m=400,Izz=450,width=2.5,xDotU=20,yDotV=0,yDotR=0,nDotV=0,nDotR=-980,
            xU=-100,xUU=-150,yV=-100,yVV=-150,yR=0,yRV=0,yVR=0,yRR=0,nR=-980,nRR=-950,
            nV=0,nVV=0,nRV=0,nVR=0,left_pos=0.,right_pos=0.)
        if any(getattr(boat,k)!=v for k,v in expected.items()):raise ValueError('Scalar integration requires the released vessel parameters.')
        boat._identification_mass_inverse=tuple(float(boat.mass_solver[j,j]) for j in range(3))
        boat.compute_motion=MethodType(source_current_motion,boat)


@torch.no_grad()
def sampled_candidates(source, observations, noise):
    """Sample the released SAC proposal distribution with explicit innovations.

    Keeping the innovation over an option gives feedback candidates, not held
    thrusts. A zero innovation exactly recovers the deterministic source.
    """
    actor=source.actor;x=torch.as_tensor(observations,dtype=torch.float32)
    hidden=torch.cat((actor.activation(actor.f1(x[...,:6])),actor.activation(actor.f2(x[...,6:14])),
        actor.activation(actor.t1(x[...,14:].reshape(*x.shape[:-1],-1,2))).mean(-2)),-1)
    hidden=actor.activation(actor.l2(actor.activation(actor.l1(hidden))))
    proposal=(actor.mean_layer(hidden)+actor.log_std_layer(hidden).clamp(-20,2).exp()*torch.as_tensor(noise,dtype=x.dtype)).tanh()
    own=actor.activation(actor.a1(torch.cat((proposal,x[...,12:14]),-1)))
    gate=.5+.5*actor.adap_layer(torch.cat((own,hidden),-1)).tanh()
    return torch.cat((proposal,gate),-1).numpy()


def summarize_pairs(rows, key, ref='aligned'):
    delta=np.asarray([r[key]['capture_time']-r[ref]['capture_time'] for r in rows])
    rng=np.random.default_rng(72481)
    means=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    return dict(time_difference=float(delta.mean()), ci=np.quantile(means,[.025,.975]).tolist(),
        **{k:float(np.mean([r[key][k] for r in rows])) for k in ('capture','collision','breach','timeout','capture_time')})


def audit_scene(scene):
    rows={}; diagnostics=[]
    for probe,policy in POLICIES.items():
        seed_random(scene); env=TADEnv(5,protocol='paper-parameters-v1')
        obs,_=env.reset(2.25,noisy_agility=False)
        np.random.seed(scene+1000000)
        packet=public_packet(env,obs); reset_policy(policy); done=0; step=0
        if probe=='aligned':
            for other in ('permuted','state-only'): reset_policy(POLICIES[other])
        while not done:
            action=policy.choose_action(packet)[0]
            if probe=='aligned':
                theta=effective_weights(policy,packet)
                thrust=filtered(packet,action,env.boids_actions)
                for other in ('permuted','state-only'):
                    alternative=POLICIES[other].choose_action(packet)[0]
                    dtheta=effective_weights(POLICIES[other],packet)-theta
                    du=filtered(packet,alternative,env.boids_actions)-thrust
                    diagnostics.append(dict(step=step,decision=step%10==0,probe=other,
                        gate_l1=float(np.abs(dtheta).mean()),gate_max=float(np.abs(dtheta).max()),
                        thrust_l2=float(np.linalg.norm(du)),thrust_max=float(np.abs(du).max())))
            packet,_,done,_,_,_=execute(env,packet,action)
            step+=1
        rows[probe]=outcome_row(env,done)
    return dict(scene=scene,**rows,diagnostics=diagnostics)


def collect_root(task):
    index,split,out=task[:3]
    influence=bool(task[3]) if len(task)>3 else False
    split_id={'train':0,'validation':1,'test':2}[split]
    scene=2740000000+split_id*100000+index+(8000000 if influence else 0)
    seed_random(scene);env=TADEnv(5,protocol='paper-parameters-v1')
    obs,_=env.reset(2.25,noisy_agility=False);np.random.seed(scene+1000000)
    packet=public_packet(env,obs);policy=POLICIES['aligned'];reset_policy(policy)
    candidates=[];all_roots=[];done=0
    while not done:
        if policy.remaining==0:
            root=dict(snapshot=compact_snapshot(env),memory=policy_memory(policy),packet=copy.deepcopy(packet))
            all_roots.append(root)
            distance=float(packet['obs'][:,2].min())
            if 6.<distance<28. and env.Current_T>=2.:candidates.append(root)
        action=policy.choose_action(packet)[0]
        packet,_,done,_,_,_=execute(env,packet,action)
    rng=np.random.default_rng(scene+2000000)
    root=(candidates or all_roots)[int(rng.integers(len(candidates or all_roots)))]
    best=-1.;selected_pair=None
    if influence:
        for trial in (candidates or all_roots):
            p=trial['packet'];reset_policy(policy,trial['memory']);action=policy.choose_action(p)[0]
            base=policy.source(p['obs']);current=policy.current
            reference=filtered(p,action,trial['snapshot'].environment.boids_actions)
            cross=np.zeros((5,5));authority=np.linalg.norm(base[:,:2]-p['obs'][:,12:14],axis=-1)
            for receiver in range(5):
                for shift in (-.4,.4):
                    changed=current.copy();changed[receiver]+=shift
                    weights=np.clip(base[:,2,None]+changed,0.,1.)
                    nominal=policy.model.modal_nominal(torch.as_tensor(base[:,:2]),torch.as_tensor(p['obs'][:,12:14]),torch.as_tensor(weights)).numpy()
                    applied=filtered(p,np.column_stack((nominal,np.ones(5,dtype=np.float32))),trial['snapshot'].environment.boids_actions)
                    cross[receiver]=np.maximum(cross[receiver],np.linalg.norm(applied-reference,axis=-1))
            np.fill_diagonal(cross,0.)
            distance=p['obs'][:,2]
            score=(cross+cross.T)*np.sqrt(authority[:,None]*authority[None,:])/(1.+(np.minimum(distance[:,None],distance[None,:])/15.)**2)
            np.fill_diagonal(score,-np.inf)
            i,j=np.unravel_index(score.argmax(),score.shape)
            if score[i,j]>best:
                root=trial;best=float(score[i,j]);selected_pair=np.array([i,j]);root['cross_response']=cross
    root.update(index=index,split=split,scene=scene,eligible=bool(candidates),checkpoint=str(CHECKPOINT),
        fusion_scale=.5,option_steps=10,influence_sampling=influence,influence_score=best)
    packet=root['packet'];base=policy.source(packet['obs'])
    # The tested boats have the largest feasible control influence among the
    # three closest to the attacker. This depends on present inputs only.
    near=np.argsort(packet['obs'][:,2])[:3]
    authority=np.linalg.norm(base[:,:2]-packet['obs'][:,12:14],axis=-1)
    focal=selected_pair if selected_pair is not None else near[np.argsort(-authority[near])[:2]]
    changes=[np.zeros(5,dtype=np.float32)]
    for a,b in ((-.4,0.),(.4,0.),(0.,-.4),(0.,.4),(-.4,-.4),(-.4,.4),(.4,-.4),(.4,.4)):
        change=np.zeros(5,dtype=np.float32);change[focal]=a,b;changes.append(change)
    root['focal']=focal;root['changes']=np.stack(changes)
    path=Path(out)/split/f'root-{index}.pth';path.parent.mkdir(parents=True,exist_ok=True);atomic_torch(path,root)
    return dict(index=index,split=split,eligible=root['eligible'],time=float(root['snapshot'].environment.Current_T),scene=scene)


def public_features(packet,base,current,change,boids):
    """Present candidate effects in a target-centred physical reference frame.

    No latent simulator state, future random number or rollout is a feature.
    The joint CBF response is the actually executable present control.
    """
    m=packet['motion'].astype(np.float64);obs=packet['obs'];n=len(base)
    theta=np.clip(base[:,2]+current[:,0]+change,0.,1.)
    mixed=theta[:,None]*base[:,:2]+(1.-theta[:,None])*obs[:,12:14]
    action=np.column_stack((mixed,np.ones(n,dtype=np.float32)))
    thrust=filtered(packet,action,boids)
    bearing=m[:,2]+obs[:,3]
    attacker=(m[:,:2]+obs[:,2,None]*np.column_stack((np.cos(bearing),np.sin(bearing)))).mean(0)
    phi=np.arctan2(attacker[1],attacker[0]);c,s=np.cos(phi),np.sin(phi)
    rotation=np.array([[c,-s],[s,c]])
    pos=(m[:,:2]-attacker)@rotation/20.;velocity=m[:,3:5]@rotation/5.
    angle=m[:,2]-phi
    own=np.column_stack((pos,velocity,np.sin(angle),np.cos(angle),m[:,5],
        base[:,:2],obs[:,12:14],base[:,2],current[:,0],theta,change,thrust/1000.,
        obs[:,4]/5.,np.sin(obs[:,5]),np.cos(obs[:,5])))
    # Pair effects retain which candidate belongs to which moving vessel.
    p=m[:,:2];v=m[:,3:5]
    relative=(p[None]-p[:,None])@rotation/15.
    relative_velocity=(v[None]-v[:,None])@rotation/5.
    c_h,s_h=np.cos(m[:,2]),np.sin(m[:,2])
    u=c_h*v[:,0]+s_h*v[:,1];lateral=-s_h*v[:,0]+c_h*v[:,1];yaw=m[:,5]
    # Local 0.4 s Taylor response using the released 3-DoF mass and drag.
    # Measured velocity is used with a zero-current local approximation;
    # full simulator states and future currents are deliberately unavailable.
    du=(400.*yaw*lateral-(100.+150.*np.abs(u))*u+thrust.sum(-1))/380.
    dv=(-400.*yaw*u+20.*u*yaw-(100.+150.*np.abs(lateral))*lateral)/400.
    acceleration=np.column_stack((c_h*du-s_h*dv-yaw*v[:,1],s_h*du+c_h*dv+yaw*v[:,0]))
    future=p+.4*v+.08*acceleration
    projected=(future[None]-future[:,None])@rotation/15.
    candidates=np.column_stack((base[:,:2],obs[:,12:14]))
    pairs=np.concatenate((relative,relative_velocity,projected,
        np.broadcast_to(candidates[:,None],(n,n,4)),np.broadcast_to(candidates[None],(n,n,4)),
        np.broadcast_to(theta[:,None,None],(n,n,1)),np.broadcast_to(theta[None,:,None],(n,n,1))),-1)
    global_state=np.array([np.linalg.norm(attacker)/60.,packet['central'][-1],n/5.],dtype=np.float32)
    return dict(node=own.astype(np.float32),pair=pairs.astype(np.float32),global_state=global_state)


def evaluate_root(task):
    split,index,repeats,out=task
    out=Path(out);root=torch.load(out/split/f'root-{index}.pth',weights_only=False)
    policy=POLICIES['aligned'];returns=[];immediate=[];done_option=[];features=[];next_features=[]
    continuation=[];root_time=float(root['snapshot'].environment.Current_T)
    split_id={'train':0,'validation':1,'test':2}[split]
    for repeat in range(repeats):
        costs=[];local_immediate=[];local_done=[];local_next=[]
        future=2840000000+split_id*1000000+index*1000+repeat+(10000000 if root.get('influence_sampling') else 0)
        for option,change in enumerate(root['changes']):
            env=root['snapshot'].restore(future_seed=future)
            packet=public_packet(env);reset_policy(policy,root['memory']);done=0;step=0;records=[]
            while not done:
                decision=policy.remaining==0
                action=policy.choose_action(packet)[0]
                if step==0:
                    policy.current=policy.current+change[:,None]
                    base=policy.source(packet['obs'])
                    weights=np.clip(base[:,2,None]+policy.current,0.,1.)
                    nominal=policy.model.modal_nominal(torch.as_tensor(base[:,:2]),torch.as_tensor(packet['obs'][:,12:14]),torch.as_tensor(weights)).numpy()
                    action=np.column_stack((nominal,np.ones(len(base),dtype=np.float32)))
                if decision:
                    base=policy.source(packet['obs'])
                    feature=public_features(packet,base,policy.current-change[:,None] if step==0 else policy.current,
                        change if step==0 else np.zeros(5),env.boids_actions)
                    records.append(dict(feature=feature,time=float(env.Current_T)))
                    if repeat==0 and step==0:features.append(feature)
                packet,_,done,_,_,_=execute(env,packet,action)
                step+=1
                if step==10 or done:
                    if len(local_immediate)==option:
                        local_immediate.append(float(env.Current_T-root_time if not done or done==3 else 60.-root_time))
                        local_done.append(bool(done))
                        if not done:
                            memory=policy_memory(policy)
                            policy.choose_action(packet)
                            nxt=public_features(packet,policy.source(packet['obs']),policy.current,np.zeros(5),env.boids_actions)
                            reset_policy(policy,memory)
                        else:nxt={k:np.zeros_like(v) for k,v in records[0]['feature'].items()}
                        local_next.append(nxt)
            cost=float(env.Current_T-root_time if done==3 else 60.-root_time);costs.append(cost)
            for j,record in enumerate(records):
                terminal=j+1==len(records)
                elapsed=float((env.Current_T if terminal else records[j+1]['time'])-record['time'])
                step_cost=(60.-record['time']) if terminal and done!=3 else elapsed
                continuation.append(dict(feature=record['feature'],next_feature=records[j+1]['feature'] if not terminal else {k:np.zeros_like(v) for k,v in record['feature'].items()},
                    cost=step_cost,done=terminal,total_cost=(env.Current_T if done==3 else 60.)-record['time']))
        returns.append(costs);immediate.append(local_immediate);done_option.append(local_done);next_features.append(local_next)
    data=dict(index=index,split=split,scene=root['scene'],root_time=root_time,eligible=root['eligible'],
        changes=root['changes'],focal=root['focal'],features=features,costs=np.asarray(returns,dtype=np.float32),
        immediate=np.asarray(immediate,dtype=np.float32),done=np.asarray(done_option),next_features=next_features,
        continuation=continuation,futures=[2840000000+split_id*1000000+index*1000+r+(10000000 if root.get('influence_sampling') else 0) for r in range(repeats)])
    atomic_torch(out/split/f'outcomes-{index}.pth',data)
    return dict(index=index,split=split,spread=float(np.ptp(data['costs'].mean(0))),
        opportunity=float(data['costs'][:,0].mean()-data['costs'].mean(0).min()),transitions=len(continuation))


def evaluate_root_batched(task, *, save=True):
    """Batch only neural inference; every branch still steps the real TADEnv."""
    split,index,repeats,out=task[:4];out=Path(out)
    settings=task[4] if len(task)>4 else {}
    root_source=Path(settings.get('root_source') or out)
    root=torch.load(root_source/split/f'root-{index}.pth',weights_only=False)
    contexts=int(settings.get('candidate_contexts',1))
    intervention_steps=int(settings.get('intervention_steps',10))
    if intervention_steps<10 or intervention_steps%10:raise ValueError('An intervention must contain whole two-second feedback intervals.')
    batched_safety=bool(settings.get('batch_safety',False))
    if settings.get('resume') and (out/split/f'outcomes-{index}.pth').exists():
        previous=torch.load(out/split/f'outcomes-{index}.pth',weights_only=False)
        if (previous['scene']==root['scene'] and previous.get('candidate_contexts',1)==contexts and
                previous['costs'].shape[-2]==repeats and previous.get('intervention_steps',10)==intervention_steps):
            cost=previous['costs'] if contexts>1 else previous['costs'][None]
            mean=cost.mean(1)
            return dict(index=index,split=split,resumed=True,
                opportunity=float((mean[:,0]-mean.min(-1)).mean()),transitions=len(previous['continuation']))
    policy=POLICIES['aligned'];changes=root['changes'] if contexts==1 else root['changes'][:3]
    K=len(changes);count=K*repeats*contexts;n=len(changes[0])
    noise=np.zeros((contexts,n,2),dtype=np.float32)
    if contexts>1:
        rng=np.random.default_rng(root['scene']+314159)
        for context in range(1,contexts,2):
            noise[context]=rng.normal(size=(n,2))
            if context+1<contexts:noise[context+1]=-noise[context]
        # Across contexts the receiver's observation, candidate and reference
        # gate are identical. Only the received peer proposals can disambiguate
        # the conditional effect of changing that receiver's composition.
        noise[:,root['focal'][0]]=0.
    split_id={'train':0,'validation':1,'test':2}[split]
    futures=[2840000000+split_id*1000000+index*1000+r+(10000000 if root.get('influence_sampling') else 0) for r in range(repeats)]
    envs=[];packets=[];histories=[];random_states=[];records=[[] for _ in range(count)]
    for context in range(contexts):
        for repeat in range(repeats):
            for option in range(K):
                env=root['snapshot'].restore(future_seed=futures[repeat]);envs.append(env)
                if settings.get('scalar_vessels'):use_source_current_motion(env)
                packets.append(public_packet(env));histories.append(copy.deepcopy(root['memory']['history']))
                random_states.append(np.random.get_state())
    active=list(range(count));currents={};outcomes=np.zeros(count,dtype=int)
    for step in range(300):
        if not active:break
        for branch in active:
            histories[branch].append(dict(obs=packets[branch]['obs'],motion=packets[branch]['motion']))
            histories[branch]=histories[branch][-8:]
        observations=np.concatenate([packets[branch]['obs'] for branch in active])
        bases=list(policy.source(observations).reshape(len(active),n,3))
        if step%10==0:
            batch={k:torch.as_tensor(np.stack([packets[i][k] for i in active]),dtype=torch.float32) for k in ('obs','motion','central')}
            batch['intent_history']=torch.as_tensor(np.concatenate([public_intent_history(histories[i])[-1:] for i in active]),dtype=torch.float32)
            base_tensor=torch.as_tensor(np.stack(bases))
            corrections=.5*policy.model.fusion(batch,base_tensor).numpy()
            for j,branch in enumerate(active):
                change=changes[branch%K] if step<intervention_steps else np.zeros(n,dtype=np.float32)
                currents[branch]=corrections[j]+change[:,None]
                if step<intervention_steps and contexts>1:
                    bases[j]=sampled_candidates(policy.source,packets[branch]['obs'],noise[branch//(K*repeats)])
                feature=public_features(packets[branch],bases[j],corrections[j],change,envs[branch].boids_actions)
                feature['option_remaining']=max(0,intervention_steps-step)/10.
                if step and np.any(change):
                    reference=public_features(packets[branch],bases[j],corrections[j],np.zeros(n,dtype=np.float32),envs[branch].boids_actions)
                    feature['reference_thrust']=reference['node'][:,15:17].copy()
                records[branch].append(dict(feature=feature,time=float(envs[branch].Current_T)))
        elif step<intervention_steps and contexts>1:
            innovations=np.concatenate([noise[branch//(K*repeats)] for branch in active])
            bases=list(sampled_candidates(policy.source,observations,innovations).reshape(len(active),n,3))
        following=[];actions=[]
        for j,branch in enumerate(active):
            base=bases[j];packet=packets[branch];current=currents[branch]
            weights=np.clip(base[:,2,None]+current,0.,1.)
            nominal=policy.model.modal_nominal(torch.as_tensor(base[:,:2]),torch.as_tensor(packet['obs'][:,12:14]),torch.as_tensor(weights)).numpy()
            action=np.column_stack((nominal,np.ones(5,dtype=np.float32))) if np.count_nonzero(current) else base
            actions.append(action)
        applied=batch_filtered([packets[i] for i in active],actions,[envs[i].boids_actions for i in active]) if batched_safety else None
        for j,branch in enumerate(active):
            action=actions[j];packet=packets[branch]
            np.random.set_state(random_states[branch])
            if applied is None:packets[branch],_,done,_,_,_=execute(envs[branch],packet,action)
            else:
                observations,_,done,_=envs[branch].step(action,'AdaRes',defender_thrust=applied[j])
                packets[branch]=public_packet(envs[branch],observations)
            random_states[branch]=np.random.get_state()
            if done:outcomes[branch]=done
            else:following.append(branch)
        active=following
    if active:raise RuntimeError('All identification futures must reach an actual terminal outcome.')
    root_time=float(root['snapshot'].environment.Current_T);costs=[];immediate=[];dones=[];next_features=[];continuation=[]
    for branch in range(count):
        history=records[branch];env=envs[branch];outcome=outcomes[branch]
        cost=float(env.Current_T-root_time if outcome==3 else 60.-root_time);costs.append(cost)
        for j,record in enumerate(history):
            record['feature']['reference_thrust']=(records[(branch//(K*repeats))*K*repeats][0]['feature']['node'][:,15:17].copy()
                if j==0 else record['feature'].get('reference_thrust',record['feature']['node'][:,15:17]).copy())
            terminal=j+1==len(history)
            elapsed=float((env.Current_T if terminal else history[j+1]['time'])-record['time'])
            step_cost=(60.-record['time']) if terminal and outcome!=3 else elapsed
            next_feature=history[j+1]['feature'] if not terminal else {k:np.zeros_like(v) for k,v in record['feature'].items()}
            continuation.append(dict(feature=record['feature'],next_feature=next_feature,cost=step_cost,done=terminal,
                total_cost=(env.Current_T if outcome==3 else 60.)-record['time']))
            if j==0:immediate.append(step_cost);dones.append(terminal);next_features.append(next_feature)
    data=dict(index=index,split=split,scene=root['scene'],root_time=root_time,eligible=root['eligible'],
        changes=changes,focal=root['focal'],features=[records[c*K*repeats+j][0]['feature'] for c in range(contexts) for j in range(K)],
        costs=np.asarray(costs,dtype=np.float32).reshape(contexts,repeats,K),
        immediate=np.asarray(immediate,dtype=np.float32).reshape(contexts,repeats,K),done=np.asarray(dones).reshape(contexts,repeats,K),
        next_features=[next_features[j*K:(j+1)*K] for j in range(contexts*repeats)],continuation=continuation,
        candidate_contexts=contexts,candidate_innovations=noise,root_source=str(root_source),
        intervention_steps=intervention_steps,
        terminations=outcomes.reshape(contexts,repeats,K),
        futures=futures,inference_backend='batched-source-and-fusion-real-environment-v2',batch_safety=batched_safety,
        scalar_vessels=bool(settings.get('scalar_vessels',False)))
    if contexts==1:
        for key in ('costs','immediate','done'):data[key]=data[key][0]
    if not save:return data
    (out/split).mkdir(parents=True,exist_ok=True)
    atomic_torch(out/split/f'outcomes-{index}.pth',data)
    means=np.asarray(costs).reshape(contexts,repeats,K).mean(1)
    return dict(index=index,split=split,spread=float(np.ptp(means,axis=-1).mean()),
        opportunity=float((means[:,0]-means.min(-1)).mean()),transitions=len(continuation))


class ConditionalValue(nn.Module):
    """Same architecture for TD, absolute and conditional-difference learning."""
    def __init__(self,node_dim=23,pair_dim=18,width=64,representation='response',bounded_link='sigmoid',separate_state=False,pair_interaction=False):
        super().__init__();self.representation=representation;self.bounded_link=bounded_link;self.separate_state=separate_state;self.pair_interaction=pair_interaction
        self.node=nn.Sequential(nn.Linear(node_dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.edge=nn.Sequential(nn.Linear(pair_dim,width),nn.SiLU(),nn.Linear(width,32),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(width+32,width),nn.SiLU())
        self.value=nn.Sequential(nn.Linear(2*width+3,width),nn.SiLU(),nn.Linear(width,1))
        self.composition=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,10))
        if bounded_link=='additive':
            nn.init.zeros_(self.composition[-1].weight);nn.init.zeros_(self.composition[-1].bias)
        self.register_buffer('node_mean',torch.zeros(node_dim));self.register_buffer('node_scale',torch.ones(node_dim))
        self.register_buffer('pair_mean',torch.zeros(pair_dim));self.register_buffer('pair_scale',torch.ones(pair_dim))
        if pair_interaction:
            self.pair_composition=nn.Sequential(nn.Linear(pair_dim,width),nn.SiLU(),nn.Linear(width,4))
            nn.init.zeros_(self.pair_composition[-1].weight);nn.init.zeros_(self.pair_composition[-1].bias)
        if separate_state:
            self.state_node=copy.deepcopy(self.node);self.state_edge=copy.deepcopy(self.edge);self.state_context=copy.deepcopy(self.context)

    def forward(self,features):
        features={k:v.to(self.node_mean.device) for k,v in features.items()}
        raw_node=features['node'];raw_pair=features['pair']
        if self.representation in ('raw','no_peer'):
            raw_node=raw_node.clone();raw_pair=raw_pair.clone()
            # Keep nominal candidate control information but remove the
            # additional executed-response representation, at equal capacity.
            raw_node[...,15:17]=features['nominal']
            raw_pair[...,4:6]=0.
            raw_node[...,21:23]=0.;raw_pair[...,16:18]=0.
            surge=features['nominal'].mean(-1)
            raw_pair[...,14]=surge[:,:,None];raw_pair[...,15]=surge[:,None,:]
            if self.representation=='no_peer':
                # Receiver-local ablation: teammate state remains, while
                # teammate candidates and every response derived from them
                # are removed. Own effects use a constant-velocity peer.
                raw_pair[...,4:6]=features['coasting_pair']
                raw_pair[...,10:14]=0.;raw_pair[...,15]=0.
        n=raw_node.shape[1];mask=1.-torch.eye(n,device=raw_node.device)[None,:,:,None]
        def encode(node,pair,state=False):
            node=(node-self.node_mean)/self.node_scale;pair=(pair-self.pair_mean)/self.pair_scale
            enc_node,enc_edge,context=(self.state_node,self.state_edge,self.state_context) if state and self.separate_state else (self.node,self.edge,self.context)
            message=(enc_edge(pair)*mask).sum(2)/max(1,n-1)
            return context(torch.cat((enc_node(node),message),-1))
        hidden=encode(raw_node,raw_pair)
        # The baseline head sees only fields invariant to the tested initial
        # composition. The residual is anchored by actual candidate authority.
        # Thus ordinary state-value error cannot masquerade as a gate effect.
        state_node=raw_node.clone();state_pair=raw_pair.clone()
        state_node[...,13:17]=0.;state_pair[...,4:6]=0.;state_pair[...,14:16]=0.
        state_node[...,20:23]=0.;state_pair[...,16:18]=0.
        state=encode(state_node,state_pair,state=True)
        pooled=torch.cat((state.mean(1),state.amax(1),features['global_state']),-1)
        remaining=60.*features['global_state'][:,1]
        baseline_logit=self.value(pooled).squeeze(-1)
        baseline=baseline_logit.sigmoid()*remaining
        # Persistent gate exchange is essential: even when present controls
        # coincide, feedback candidates may diverge later in the 2 s option.
        change=torch.cat((raw_node[...,20:21],raw_node[...,13:15],raw_node[...,21:23]),-1)
        basis=torch.cat((change,change.square()),-1)
        raw=(self.composition(hidden)*basis).sum((-1,-2))
        if self.pair_interaction:
            exchange=raw_node[...,20]/.4
            powers=torch.stack((exchange,exchange.square()),-1)
            product=(powers[:,:,None,:,None]*powers[:,None,:,None,:]).flatten(-2)
            coefficient=self.pair_composition((raw_pair-self.pair_mean)/self.pair_scale)
            raw=raw+.5*(coefficient*product*mask).sum((-1,-2,-3))
        if self.bounded_link=='additive':
            # A centered conditional loss annihilates the state baseline
            # exactly. A sigmoid/clamp would couple its calibration error back
            # into the composition derivative, or kill that derivative.
            return baseline+12.*torch.tanh(raw/12.)
        correction=40.*torch.tanh(raw)
        if self.bounded_link=='sigmoid':
            return remaining*torch.sigmoid(baseline_logit+4.*correction/remaining.clamp_min(1.))
        return torch.maximum(torch.minimum(baseline+correction,remaining),torch.zeros_like(remaining))


def event_labels(costs, remaining):
    """Linear event projection preserves every sampled capped-time return."""
    times=costs.new_tensor(EVENT_TIMES)
    support=torch.minimum(times,remaining[...,None])
    while support.ndim<costs.ndim+1:support=support.unsqueeze(-2)
    support=support.expand(*costs.shape,len(times))
    high=(support<costs[...,None]).sum(-1).clamp(max=len(times)-1)
    low=(high-1).clamp_min(0)
    left=support.gather(-1,low[...,None]).squeeze(-1)
    right=support.gather(-1,high[...,None]).squeeze(-1)
    weight=torch.where(right>left,(costs-left)/(right-left).clamp_min(1e-8),torch.zeros_like(costs)).clamp(0.,1.)
    label=costs.new_zeros(*costs.shape,len(times))
    label.scatter_add_(-1,low[...,None],(1.-weight)[...,None])
    label.scatter_add_(-1,high[...,None],weight[...,None])
    return label


class ConditionalEventValue(ConditionalValue):
    """Learn conditional event distributions before integrating task cost.

    The state and intervention use separate heads. The intervention head has
    no output at zero composition change, and receives the matching pair
    controls before pooling. All supervision arms have identical parameters.
    """
    def __init__(self,representation='response',width=64):
        super().__init__(representation=representation,width=width)
        count=len(EVENT_TIMES)
        self.value=nn.Sequential(nn.Linear(2*width+3,width),nn.SiLU(),nn.Linear(width,count))
        self.composition=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,10*count))
        nn.init.zeros_(self.composition[-1].weight);nn.init.zeros_(self.composition[-1].bias)

    def distribution(self,features):
        f={k:v.to(self.node_mean.device) for k,v in features.items()}
        node=f['node'];pair=f['pair']
        if self.representation in ('raw','no_peer'):
            node=node.clone();pair=pair.clone()
            node[...,15:17]=f['nominal'];node[...,21:23]=0.
            pair[...,4:6]=0.;pair[...,16:18]=0.
            surge=f['nominal'].mean(-1);pair[...,14]=surge[:,:,None];pair[...,15]=surge[:,None,:]
            if self.representation=='no_peer':
                pair[...,4:6]=f['coasting_pair'];pair[...,10:14]=0.;pair[...,15]=0.
        n=node.shape[1];mask=1.-torch.eye(n,device=node.device)[None,:,:,None]
        def encode(x,e):
            z=self.node((x-self.node_mean)/self.node_scale)
            msg=(self.edge((e-self.pair_mean)/self.pair_scale)*mask).sum(2)/max(1,n-1)
            return self.context(torch.cat((z,msg),-1))
        hidden=encode(node,pair)
        state_node=node.clone();state_pair=pair.clone()
        state_node[...,13:17]=0.;state_node[...,20:23]=0.
        state_pair[...,4:6]=0.;state_pair[...,14:18]=0.
        state=encode(state_node,state_pair)
        baseline=self.value(torch.cat((state.mean(1),state.amax(1),f['global_state']),-1))
        change=torch.cat((node[...,20:21],node[...,13:15],node[...,21:23]),-1)
        basis=torch.cat((change,change.square()),-1)
        effect=(self.composition(hidden).reshape(*hidden.shape[:2],10,-1)*basis[...,None]).sum((1,2))
        return (baseline+effect).softmax(-1)

    def forward(self,features):
        probability=self.distribution(features)
        remaining=60.*features['global_state'][:,1].to(probability.device)
        support=torch.minimum(probability.new_tensor(EVENT_TIMES),remaining[:,None])
        return (probability*support).sum(-1)


class CandidateGateAdapter(nn.Module):
    """A public, receiver-local actor optimized through a frozen learned Q.

    Peer candidates enter only the matching receiver's relation encoder. The
    no-peer arm preserves all relative kinematics, capacity and own candidates,
    while removing candidate-dependent executed/projected peer information.
    No critic, return label, simulator state or rollout is needed at deployment.
    """
    def __init__(self,peer_candidates=True,width=64,joint_actor=False):
        super().__init__();self.peer_candidates=peer_candidates;self.joint_actor=joint_actor
        self.node=nn.Sequential(nn.Linear(23,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.edge=nn.Sequential(nn.Linear(18,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.gate=nn.Sequential(nn.Linear(2*width+3,width),nn.SiLU(),nn.Linear(width,3))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        if joint_actor:
            self.joint_gate=nn.Sequential(nn.Linear(18,width),nn.SiLU(),nn.Linear(width,4))
            nn.init.zeros_(self.joint_gate[-1].weight);nn.init.zeros_(self.joint_gate[-1].bias)
        self.register_buffer('node_mean',torch.zeros(23));self.register_buffer('node_scale',torch.ones(23))
        self.register_buffer('pair_mean',torch.zeros(18));self.register_buffer('pair_scale',torch.ones(18))

    def receiver_inputs(self,features):
        f={k:v.to(self.node_mean.device) for k,v in features.items()}
        node=f['node'];pair=f['pair'];n=node.shape[1]
        if not self.peer_candidates:
            node=node.clone();pair=pair.clone()
            node[...,15:17]=f['nominal'];node[...,21:23]=0.
            pair[...,4:6]=f['coasting_pair'];pair[...,10:14]=0.;pair[...,15:18]=0.
            pair[...,14]=f['nominal'].mean(-1)[:,:,None]
        return f,node,pair

    def forward(self,features):
        f,node,pair=self.receiver_inputs(features);n=node.shape[1]
        mask=(1.-torch.eye(n,device=node.device))[None,:,:,None]
        # Distance weighting changes no information available to either arm.
        proximity=(-pair[...,:2].norm(dim=-1)).exp()[...,None]*mask
        message=(self.edge((pair-self.pair_mean)/self.pair_scale)*proximity).sum(2)/proximity.sum(2).clamp_min(1e-6)
        hidden=self.node((node-self.node_mean)/self.node_scale)
        return self.gate(torch.cat((hidden,message,f['global_state'][:,None].expand(-1,n,-1)),-1))

    def joint_logits(self,features,focal):
        logits=self(features);focal=torch.as_tensor(focal,device=logits.device)
        batch=torch.arange(len(logits),device=logits.device);i,j=focal.unbind(-1)
        coordinates=logits.new_tensor([[0,0],[1,0],[2,0],[0,1],[0,2],[1,1],[1,2],[2,1],[2,2]],dtype=torch.long)
        result=logits[batch,i][:,coordinates[:,0]]+logits[batch,j][:,coordinates[:,1]]
        if self.joint_actor:
            _,_,pair=self.receiver_inputs(features)
            edge=(pair-self.pair_mean)/self.pair_scale
            direct=self.joint_gate(edge[batch,i,j]).reshape(-1,2,2)
            reverse=self.joint_gate(edge[batch,j,i]).reshape(-1,2,2).transpose(-1,-2)
            # Explicit nonseparable 2x2 factor, anchored at the reference
            # action of either receiver and equivariant to swapping receivers.
            interaction=.5*(direct+reverse)
            result=torch.cat((result[:,:5],result[:,5:]+interaction.flatten(1)),-1)
        return result


def gate_probabilities(model,data):
    K=data['costs'].shape[1]
    inputs=slice_features(data['features'],torch.arange(len(data['costs']))*K)
    logits=model(inputs);focal=torch.as_tensor(data['focal'],device=logits.device)
    probabilities=logits.softmax(-1)
    receiver=probabilities[torch.arange(len(logits),device=logits.device),focal[:,0]]
    if K==3:return receiver
    if K!=9:raise ValueError('A gate uses three single-boat or nine paired options.')
    if model.joint_actor:return model.joint_logits(inputs,focal).softmax(-1)
    peer=probabilities[torch.arange(len(logits),device=logits.device),focal[:,1]]
    coordinates=logits.new_tensor([[0,0],[1,0],[2,0],[0,1],[0,2],[1,1],[1,2],[2,1],[2,2]],dtype=torch.long)
    return receiver[:,coordinates[:,0]]*peer[:,coordinates[:,1]]


def present_focal_pair(packet,base,current,boids):
    """Select interacting receivers from present feasible control influence."""
    n=len(base)
    from train_response_fusion import CandidateInterventionModel
    def command(shift):
        theta=np.clip(base[:,2,None]+shift,0.,1.)
        value=CandidateInterventionModel.modal_nominal(torch.as_tensor(base[:,:2]),torch.as_tensor(packet['obs'][:,12:14]),torch.as_tensor(theta)).numpy()
        return np.column_stack((value,np.ones(n,dtype=np.float32)))
    reference=filtered(packet,command(current),boids);cross=np.zeros((n,n))
    for receiver in range(n):
        for shift in (-.4,.4):
            changed=current.copy();changed[receiver]+=shift
            applied=filtered(packet,command(changed),boids)
            cross[receiver]=np.maximum(cross[receiver],np.linalg.norm(applied-reference,axis=-1))
    np.fill_diagonal(cross,0.)
    authority=np.linalg.norm(base[:,:2]-packet['obs'][:,12:14],axis=-1)
    distance=packet['obs'][:,2]
    score=(cross+cross.T)*np.sqrt(authority[:,None]*authority[None,:])/(1.+(np.minimum(distance[:,None],distance[None,:])/15.)**2)
    np.fill_diagonal(score,-np.inf)
    return np.asarray(np.unravel_index(score.argmax(),score.shape)),float(score.max())


class GateAdapterPolicy:
    """Executable actor refinement with the same feedback and safety semantics."""
    def __init__(self,path,source=SOURCE,checkpoint=CHECKPOINT,max_decisions=1):
        saved=torch.load(path,weights_only=True)
        if saved['format']!='conditional-value-gate-v1':raise ValueError('Unsupported gate policy.')
        self.actor=CandidateGateAdapter(saved['peer_candidates'],joint_actor=saved.get('joint_actor',False)).eval().requires_grad_(False)
        self.actor.load_state_dict(saved['model']);self.options=saved['options']
        self.initialize_reference(saved,source,checkpoint,max_decisions)

    def initialize_reference(self,saved,source,checkpoint,max_decisions):
        self.reference=ResponsePolicy(source,checkpoint)
        self.reference.fusion_scale=saved['fusion_scale'];self.reference.horizon=saved['option_steps']
        self.intervention_steps=saved.get('intervention_steps',10);self.extra_remaining=0;self.extra=None
        self.max_decisions=max_decisions;self.decisions=0;self.last_time=None;self.control_changes=[]

    def select_choices(self,packet,base,current,boids,focal):
        feature=public_features(packet,base,current,np.zeros(len(base)),boids)
        feature['option_remaining']=self.intervention_steps/10.
        inputs=stack_features([feature])
        choices=self.actor(inputs)[0].argmax(-1).numpy()
        if self.actor.joint_actor:
            selected=int(self.actor.joint_logits(inputs,focal[None])[0].argmax())
            coordinates=np.array([[0,0],[1,0],[2,0],[0,1],[0,2],[1,1],[1,2],[2,1],[2,2]])
            choices[focal]=coordinates[selected]
        return choices

    @torch.no_grad()
    def choose_action(self,packet,deterministic=True):
        remaining=float(packet['central'][-1])
        if self.last_time is None or remaining>=self.last_time:
            reset_policy(self.reference);self.decisions=0;self.control_changes=[];self.extra_remaining=0;self.extra=None
        self.last_time=remaining;decision=self.reference.remaining==0
        action=self.reference.choose_action(packet)[0]
        eligible=6.<float(packet['obs'][:,2].min())<28. and remaining<=1.-2./60.+1e-6
        choose=(decision and eligible and not self.extra_remaining and
            (not self.max_decisions or self.decisions<self.max_decisions))
        if not choose and not self.extra_remaining:return action,0.
        base=self.reference.source(packet['obs']);current=self.reference.current
        boids=750.*packet['obs'][:,12:14].astype(np.float64)+250.
        if choose:
            focal,_=present_focal_pair(packet,base,current,boids)
            choices=self.select_choices(packet,base,current,boids,focal)
            self.extra=np.zeros(len(base),dtype=np.float32)
            for receiver in focal[:1 if self.options==3 else 2]:self.extra[receiver]=(0.,-.4,.4)[choices[receiver]]
            self.extra_remaining=self.intervention_steps;self.decisions+=1
        weights=np.clip(base[:,2,None]+current+self.extra[:,None],0.,1.)
        nominal=self.reference.model.modal_nominal(torch.as_tensor(base[:,:2]),torch.as_tensor(packet['obs'][:,12:14]),torch.as_tensor(weights)).numpy()
        chosen=np.column_stack((nominal,np.ones(len(base),dtype=np.float32)))
        if choose:
            control_delta=filtered(packet,chosen,boids)-filtered(packet,action,boids)
            self.control_changes.append(dict(focal=focal.tolist(),choice=choices[focal].tolist(),
                thrust_l2=float(np.linalg.norm(control_delta))))
        self.extra_remaining-=1
        return chosen,0.


class ValueGateAdapterPolicy(GateAdapterPolicy):
    """A learned energy adapter: exact discrete policy improvement over Q.

    The finite candidate energies are evaluated from public present inputs.
    There is no environment rollout, online return label or extra actor fit.
    """
    def __init__(self,path,source=SOURCE,checkpoint=CHECKPOINT,max_decisions=1):
        self.value,saved=restore_value(path)
        physical_roots=saved['train_roots']+(saved.get('extra_train_count',0) if saved.get('extra_train_output') else 0)
        self.options=saved.get('options',3 if saved['train_contexts']>physical_roots else 9)
        self.initialize_reference(saved,source,checkpoint,max_decisions)

    def select_choices(self,packet,base,current,boids,focal):
        coordinates=np.array([[0,0],[1,0],[2,0],[0,1],[0,2],[1,1],[1,2],[2,1],[2,2]])[:self.options]
        features=[]
        for coordinate in coordinates:
            change=np.zeros(len(base),dtype=np.float32)
            change[focal]=np.array([0.,-.4,.4])[coordinate]
            feature=public_features(packet,base,current,change,boids)
            feature['option_remaining']=self.intervention_steps/10.
            features.append(feature)
        for feature in features:feature['reference_thrust']=features[0]['node'][:,15:17]
        energy=self.value(stack_features(features))
        selected=int(energy.argmin());choices=np.zeros(len(base),dtype=int)
        choices[focal]=coordinates[selected]
        return choices


def initialize_gate_policies(output,seed,max_decisions,policy_kind='actor'):
    global GATE_POLICIES
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    GATE_POLICIES={}
    for arm in ('td','absolute','difference','no_peer'):
        name='difference_no_peer' if policy_kind=='value' and arm=='no_peer' else arm
        cls,prefix=(ValueGateAdapterPolicy,'value') if policy_kind=='value' else (GateAdapterPolicy,'gate')
        GATE_POLICIES[arm]=cls(Path(output)/f'{prefix}-{name}-{seed}.pth',max_decisions=max_decisions)
    base=ResponsePolicy(SOURCE,CHECKPOINT);base.fusion_scale=.5;base.horizon=10
    GATE_POLICIES['reference']=base
    for arm,bias in (('arboids_cbf',0.),('constant_bias',-.05)):
        base=ResponsePolicy(SOURCE,CHECKPOINT);base.fusion_scale=0.;base.fusion_bias=bias
        GATE_POLICIES[arm]=base
    safety_controller(5)


def gate_episode(scene):
    rows={}
    for arm,policy in GATE_POLICIES.items():
        seed_random(scene);env=TADEnv(5,protocol='paper-parameters-v1')
        obs,_=env.reset(2.25,noisy_agility=False);np.random.seed(scene+1000000)
        packet=public_packet(env,obs);done=0
        if isinstance(policy,GateAdapterPolicy):policy.last_time=None
        else:reset_policy(policy)
        while not done:
            action=policy.choose_action(packet)[0]
            packet,_,done,_,_,_=execute(env,packet,action)
        rows[arm]=outcome_row(env,done)
        if isinstance(policy,GateAdapterPolicy):rows[arm]['control_decisions']=policy.control_changes
    return dict(scene=scene,**rows)


def evaluate_gate_episodes(args):
    start=2960000000+args.offset
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_gate_policies,
            initargs=(args.output,args.seed,args.max_decisions,args.policy_kind)) as pool:
        rows=[]
        for row in pool.map(gate_episode,range(start,start+args.count)):
            rows.append(row)
            if len(rows)%16==0:print(json.dumps(dict(completed=len(rows))),flush=True)
    summary={arm:summarize_pairs(rows,arm,ref='difference') for arm in rows[0] if arm!='scene'}
    for arm in ('td','absolute','difference','no_peer'):
        changes=[d['thrust_l2'] for row in rows for d in row[arm]['control_decisions']]
        summary[arm]['executed_control_l2']=float(np.mean(changes)) if changes else 0.
        summary[arm]['episodes_with_control_change']=sum(any(d['thrust_l2']>1. for d in row[arm]['control_decisions']) for row in rows)
    prefix='value' if args.policy_kind=='value' else 'gate'
    result=dict(seed=args.seed,episodes=args.count,scene_start=start,max_decisions=args.max_decisions,policy_kind=args.policy_kind,summary=summary,rows=rows,
        policy_hashes={arm:hashlib.sha256((args.output/f"{prefix}-{'difference_no_peer' if prefix=='value' and arm=='no_peer' else arm}-{args.seed}.pth").read_bytes()).hexdigest()
            for arm in ('td','absolute','difference','no_peer')},incumbent_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
    atomic_json(args.output/f'{prefix}-episodes-{args.seed}-{args.offset}-{args.max_decisions}.json',result)
    print(json.dumps(summary),flush=True)


def stack_features(rows,device='cpu'):
    result={k:torch.as_tensor(np.stack([r[k] for r in rows]),device=device,dtype=torch.float32)
            for k in ('node','pair','global_state')}
    raw=result['node'];theta=raw[...,13].clone();prior=(raw[...,11]+raw[...,12]).clamp(0.,1.)
    persistence=raw[...,14:15].clone()*torch.as_tensor([r.get('option_remaining',1.) for r in rows],device=device,dtype=raw.dtype)[:,None,None]
    gap=raw[...,7:9]-raw[...,9:11]
    change=(theta-prior)[...,None]*gap
    result['nominal']=.75*(theta[...,None]*raw[...,7:9]+(1.-theta[...,None])*raw[...,9:11])+.25
    # Separate persistent composition, immediate nominal change, and the
    # change actually admitted by the joint safety filter.
    raw[...,13]=change.mean(-1);raw[...,14]=.5*(change[...,0]-change[...,1])
    surge=raw[...,15:17].mean(-1)
    result['pair'][...,14]=surge[:,:,None];result['pair'][...,15]=surge[:,None,:]
    reference=torch.as_tensor(np.stack([r.get('reference_thrust',r['node'][:,15:17]) for r in rows]),device=device,dtype=torch.float32)
    applied=raw[...,15:17]-reference
    applied=torch.stack((applied.mean(-1),.5*(applied[...,0]-applied[...,1])),-1)
    result['node']=torch.cat((raw,persistence,applied),-1)
    # Deterministic candidate-effect representation using public measured
    # motion and released vessel dynamics. Outcomes never use this projection.
    state=torch.cat((raw[...,:2]*20.,torch.atan2(raw[...,4:5],raw[...,5:6]),raw[...,2:4]*5.,raw[...,6:7]),-1)
    future=state.clone();reference_future=state.clone();nominal_future=state.clone()
    for _ in range(10):
        future,_=DifferentiableInterventionDynamics.vessel(future,raw[...,15:17]*1000.,torch.zeros_like(future[...,3:6]))
        reference_future,_=DifferentiableInterventionDynamics.vessel(reference_future,reference*1000.,torch.zeros_like(reference_future[...,3:6]))
        nominal_future,_=DifferentiableInterventionDynamics.vessel(nominal_future,result['nominal']*1000.,torch.zeros_like(nominal_future[...,3:6]))
    relative=(future[:,None,:,:2]-future[:,:,None,:2])/15.
    reference_relative=(reference_future[:,None,:,:2]-reference_future[:,:,None,:2])/15.
    result['pair'][...,4:6]=relative
    result['pair']=torch.cat((result['pair'],relative-reference_relative),-1)
    coasting=state[...,:2]+2.*state[...,3:5]
    result['coasting_pair']=(coasting[:,None]-nominal_future[:,:,None,:2])/15.
    return result


def slice_features(features,index):return {k:v[index] for k,v in features.items()}


def paired_contrast_precision(costs,ridge=4.):
    """Remove common future shocks before estimating contrast uncertainty."""
    centered=costs-costs.mean(-1,keepdims=True)
    covariance=np.cov(centered,rowvar=False,ddof=1)/len(costs)
    count=costs.shape[-1]
    # Shrink unresolved directions at small repetition counts instead of
    # treating a low-rank sample covariance as certainty in its null space.
    average=np.trace(covariance)/max(1,count-1)
    covariance=.5*covariance+.5*average*np.eye(count)
    return ridge*np.linalg.inv(covariance+ridge*np.eye(count))


def load_value_split(out,split,count,device='cpu',transitions=False,offset=0):
    rows=[torch.load(Path(out)/split/f'outcomes-{i}.pth',weights_only=False) for i in range(offset,offset+count)]
    context_count=rows[0].get('candidate_contexts',1)
    if any(r.get('candidate_contexts',1)!=context_count for r in rows):raise ValueError('Mixed candidate protocols.')
    joint_precision=[]
    if context_count>1:
        for row in rows:
            value=row['costs'].transpose(1,0,2)
            value=value-value.mean(-1,keepdims=True)
            joint_precision.append(paired_contrast_precision(value.reshape(len(value),-1)))
    label_repetitions=[int(r['costs'].shape[-2]) for r in rows]
    expanded=[]
    for row in rows:
        contexts=row.get('candidate_contexts',1)
        if contexts==1:expanded.append(row);continue
        K=row['costs'].shape[-1]
        for context in range(contexts):
            local={**row,'costs':row['costs'][context],'features':row['features'][context*K:(context+1)*K]}
            # Every recorded transition is included exactly once, even though
            # a root supplies several controlled candidate contexts.
            local['continuation']=row['continuation'] if context==0 else []
            expanded.append(local)
    rows=expanded
    for row in rows:
        reference=row['features'][0]['node'][:,15:17]
        for f in row['features']:f['reference_thrust']=reference
        if transitions:
            for r in row['continuation']:
                for name in ('feature','next_feature'):
                    f=r[name]
                    if 'reference_thrust' not in f:
                        f['reference_thrust']=reference if np.any(f['node'][:,14]) else f['node'][:,15:17]
    feature=stack_features([f for r in rows for f in r['features']],device)
    costs=torch.as_tensor(np.stack([r['costs'].mean(0) for r in rows]),device=device)
    result=dict(features=feature,costs=costs)
    result['contexts_per_root']=context_count
    result['intervention_steps']=rows[0].get('intervention_steps',10)
    result['label_repetitions']=label_repetitions
    if joint_precision:result['joint_precision']=torch.as_tensor(np.stack(joint_precision),device=device,dtype=torch.float32)
    result['groups']=np.asarray([r['scene'] for r in rows])
    result['focal']=np.asarray([r['focal'] for r in rows])
    samples=torch.as_tensor(np.stack([r['costs'] for r in rows]),device=device)
    remaining=60.-torch.as_tensor([r['root_time'] for r in rows],device=device)
    result['event_labels']=event_labels(samples,remaining).mean(1)
    result['variance']=torch.as_tensor(np.stack([r['costs'].var(0,ddof=1)/len(r['costs']) for r in rows]),device=device)
    result['difference_variance']=torch.as_tensor(np.stack([(r['costs']-r['costs'][:,:1]).var(0,ddof=1)/len(r['costs']) for r in rows]),device=device)
    result['contrast_precision']=torch.as_tensor(np.stack([paired_contrast_precision(r['costs']) for r in rows]),device=device,dtype=torch.float32)
    if transitions:
        records=[record for r in rows for record in r['continuation']]
        result.update(transitions=stack_features([r['feature'] for r in records],device),
            next_transitions=stack_features([r['next_feature'] for r in records],device),
            transition_cost=torch.as_tensor([r['cost'] for r in records],dtype=torch.float32,device=device),
            transition_done=torch.as_tensor([r['done'] for r in records],device=device),
            transition_return=torch.as_tensor([r['total_cost'] for r in records],dtype=torch.float32,device=device))
    return result


def load_value_training(args,transitions=False):
    data=load_value_split(args.output,'train',args.train_count,transitions=transitions)
    if args.extra_train_output is None:return data
    extra=load_value_split(args.extra_train_output,'train',args.extra_train_count,transitions=transitions)
    if data['contexts_per_root']!=extra['contexts_per_root'] or data['costs'].shape[1]!=extra['costs'].shape[1]:
        raise ValueError('Training banks must have the same candidate protocol.')
    if set(data['groups']) & set(extra['groups']):raise ValueError('Training banks overlap.')
    def join(a,b):
        if isinstance(a,dict):return {k:join(a[k],b[k]) for k in a}
        if isinstance(a,torch.Tensor):return torch.cat((a,b))
        if isinstance(a,np.ndarray):return np.concatenate((a,b))
        if isinstance(a,list):return a+b
        if a!=b:raise ValueError('Incompatible training metadata.')
        return a
    return join(data,extra)


@torch.no_grad()
def value_metrics(model,data,retain=False):
    prediction=torch.cat([model(slice_features(data['features'],slice(i,i+256))).to(data['costs'].device) for i in range(0,len(data['features']['node']),256)]).reshape_as(data['costs'])
    truth=data['costs'];delta=truth-truth[:,:1];estimate=prediction-prediction[:,:1]
    chosen=prediction.argmin(-1);selected=truth[torch.arange(len(truth),device=truth.device),chosen]
    error=(estimate[:,1:]-delta[:,1:]).abs().mean(-1)
    left,right=torch.triu_indices(truth.shape[1],truth.shape[1],1,device=truth.device)
    gap=truth[:,left]-truth[:,right];pred=prediction[:,left]-prediction[:,right]
    meaningful=gap.abs()>=.5
    agreement=((gap*pred>0)&meaningful).float().sum(-1)/meaningful.sum(-1).clamp_min(1)
    result=dict(E_env=float(error.mean()),absolute_mae=float((prediction-truth).abs().mean()),
        decision_cost=float(selected.mean()),reference_cost=float(truth[:,0].mean()),
        improvement=float((truth[:,0]-selected).mean()),regret=float((selected-truth.min(-1).values).mean()),
        pair_accuracy=float(agreement[meaningful.any(-1)].mean()),selection_change=float((chosen!=0).float().mean()))
    n=data['features']['node'].shape[1]
    thrust=data['features']['node'][...,15:17].reshape(len(truth),truth.shape[1],n,2)*1000.
    controls=thrust[torch.arange(len(truth)),chosen.cpu()]-thrust[:,0]
    norm=controls.flatten(1).norm(dim=-1)
    result.update(executed_control_l2=float(norm.mean()),executed_control_changed=float((norm>1.).float().mean()))
    if 'groups' in data:
        selected_indices=chosen.cpu().numpy();groups=data['groups']
        result['candidate_context_adaptation']=float(np.mean([len(np.unique(selected_indices[groups==g]))>1 for g in np.unique(groups)]))
    if retain:
        result.update(predictions=prediction.cpu().tolist(),truth=truth.cpu().tolist(),chosen=chosen.cpu().tolist(),
            errors=error.cpu().tolist(),selected_costs=selected.cpu().tolist(),pair_agreement=agreement.cpu().tolist())
    return result


def fit_values(args):
    torch.set_num_threads(1);torch.set_grad_enabled(True)
    device=args.device
    training=load_value_training(args,transitions=True)
    validation=load_value_split(args.output,'validation',args.validation_count,'cpu')
    # No test outcomes are loaded during fitting or checkpoint selection.
    summaries={};K=training['costs'].shape[1];start=time.monotonic()
    for arm in args.arms.split(','):
        if arm not in VALUE_ARMS:raise ValueError('Unknown value supervision arm: '+arm)
        torch.manual_seed(args.seed)
        representation='raw' if arm.endswith('_raw') else 'no_peer' if arm.endswith('_no_peer') else 'response'
        model=(ConditionalEventValue(representation=representation) if args.event_value else
            ConditionalValue(representation=representation,bounded_link='additive' if args.additive_value else 'sigmoid',separate_state=args.separate_state,pair_interaction=args.pair_interaction)).to(device)
        for key in ('node','pair'):
            values=training['features'][key].flatten(0,-2)
            getattr(model,key+'_mean').copy_(values.mean(0));getattr(model,key+'_scale').copy_(values.std(0).clamp_min(.05))
        target=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best=float('inf');history=[];rng=np.random.default_rng(args.seed)
        for iteration in range(1,args.updates+1):
            ix=torch.as_tensor(rng.integers(len(training['transition_cost']),size=256))
            if args.joint_contrasts:
                contexts=training['contexts_per_root']
                groups=torch.as_tensor(rng.integers(len(training['costs'])//contexts,size=max(1,32//contexts)))
                roots=(groups[:,None]*contexts+torch.arange(contexts)).flatten()
            else:roots=torch.as_tensor(rng.integers(len(training['costs']),size=32))
            if args.event_value:
                transitions=slice_features(training['transitions'],ix)
                with torch.no_grad():
                    nxt=slice_features(training['next_transitions'],ix)
                    probability=target.distribution(nxt)
                    remaining=60.*transitions['global_state'][:,1].to(device)
                    duration=training['transition_cost'][ix].to(device)
                    support=torch.minimum(probability.new_tensor(EVENT_TIMES),60.*nxt['global_state'][:,1,None].to(device))
                    future=duration[:,None]+(~training['transition_done'][ix]).to(device)[:,None]*support
                    y=(event_labels(future,remaining)*probability[...,None]).sum(1)
                td=-(y*model.distribution(transitions).clamp_min(1e-8).log()).sum(-1).mean()
            else:
                with torch.no_grad():
                    y=training['transition_cost'][ix].to(device)+(~training['transition_done'][ix]).to(device)*target(slice_features(training['next_transitions'],ix))
                td=F.smooth_l1_loss(model(slice_features(training['transitions'],ix))/20.,y/20.,beta=.5)
            loss=td
            if arm!='td':
                indices=(roots[:,None]*K+torch.arange(K)).flatten()
                if args.event_value:
                    probability=model.distribution(slice_features(training['features'],indices)).reshape(-1,K,len(EVENT_TIMES))
                    truth=training['event_labels'][roots].to(device)
                    error=(probability-truth).cumsum(-1)[...,:-1]
                    widths=probability.new_tensor(EVENT_TIMES).diff()/60.
                    if arm.startswith('difference'):
                        error=error-error.mean(1,keepdim=True)
                    auxiliary=(error.square()*widths).sum(-1).mean()
                    loss=loss+args.aux_weight*auxiliary
                else:
                    loss=loss+args.aux_weight*scalar_auxiliary(model,training,indices,roots,K,arm,args,device)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            with torch.no_grad():
                for a,b in zip(target.parameters(),model.parameters()):a.lerp_(b,.02)
            if iteration%100==0:
                metrics=value_metrics(model,validation)
                # The identical validation criterion applies to every arm;
                # held-out outcomes never participate in selection.
                score=metrics['E_env']+metrics['regret']
                history.append(dict(iteration=iteration,loss=float(loss.detach()),**metrics))
                if score<best:
                    best=score
                    atomic_torch(args.output/f'value-{arm}-{args.seed}.pth',dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        arm=arm,seed=args.seed,iteration=iteration,representation=model.representation,validation=metrics,
                        train_roots=args.train_count,validation_roots=args.validation_count,option_steps=10,fusion_scale=.5,
                        options=K,
                        train_contexts=len(training['costs']),validation_contexts=len(validation['costs']),dataset=str(args.output),
                        extra_train_output=str(args.extra_train_output) if args.extra_train_output else None,extra_train_count=args.extra_train_count,
                        uncertainty_weighting=args.uncertainty_weighting,contrast_covariance=args.contrast_covariance,
                        event_value=args.event_value,aux_weight=args.aux_weight,
                        objective_version='paired-value-v3-weighted-scalar',
                        joint_contrasts=args.joint_contrasts,label_repetitions=training['label_repetitions'],
                        separate_state=model.separate_state,
                        pair_interaction=getattr(model,'pair_interaction',False),
                        intervention_steps=training['intervention_steps'],
                        bounded_link=model.bounded_link,feature_version='executed-response-option-v2'))
                if iteration%500==0:print(json.dumps(dict(arm=arm,seed=args.seed,iteration=iteration,elapsed=round(time.monotonic()-start,1),**metrics)),flush=True)
        summaries[arm]=history
    history_path=args.output/f'value-training-{args.seed}.json'
    existing=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else {}
    atomic_json(history_path,{**existing,**summaries})


def scalar_auxiliary(model,training,indices,roots,K,arm,args,device):
    prediction=model(slice_features(training['features'],indices)).reshape(-1,K)
    truth=training['costs'][roots].to(device)
    variance=training['variance'][roots].to(device)
    if arm.startswith('difference') and args.joint_contrasts and training['contexts_per_root']>1:
        contexts=training['contexts_per_root']
        error=(prediction-truth)/20.;error=error-error.mean(-1,keepdim=True)
        error=error.reshape(-1,contexts*K)
        groups=roots.reshape(-1,contexts)[:,0]//contexts
        precision=training['joint_precision'][groups].to(device)
        return torch.einsum('bi,bij,bj->b',error,precision,error).mean()/(contexts*K)
    if arm.startswith('difference') and args.contrast_covariance:
        error=(prediction-truth)/20.;error=error-error.mean(-1,keepdim=True)
        precision=training['contrast_precision'][roots].to(device)
        return torch.einsum('bi,bij,bj->b',error,precision,error).mean()/K
    if arm.startswith('difference'):
        prediction=prediction[:,1:]-prediction[:,:1];truth=truth[:,1:]-truth[:,:1]
        variance=training['difference_variance'][roots,1:].to(device)
    weight=4./(4.+variance) if args.uncertainty_weighting else torch.ones_like(variance)
    return (weight*((prediction-truth)/20.).square()).mean()


def restore_value(path):
    saved=torch.load(path,weights_only=True)
    model=(ConditionalEventValue(representation=saved['representation']) if saved.get('event_value') else
        ConditionalValue(representation=saved['representation'],bounded_link=saved.get('bounded_link','clip'),separate_state=saved.get('separate_state',False),pair_interaction=saved.get('pair_interaction',False)))
    model.load_state_dict(saved['model']);return model.eval().requires_grad_(False),saved


@torch.no_grad()
def gate_metrics(model,data,retain=False):
    probabilities=gate_probabilities(model,data).cpu();chosen=probabilities.argmax(-1)
    truth=data['costs'].cpu();selected=truth[torch.arange(len(truth)),chosen]
    result=dict(decision_cost=float(selected.mean()),reference_cost=float(truth[:,0].mean()),
        improvement=float((truth[:,0]-selected).mean()),regret=float((selected-truth.min(-1).values).mean()),
        selection_change=float((chosen!=0).float().mean()))
    controls=data['features']['node'][...,15:17].cpu().reshape(len(truth),truth.shape[1],-1,2)*1000.
    difference=controls[torch.arange(len(truth)),chosen]-controls[:,0]
    result['executed_control_l2']=float(difference.flatten(1).norm(dim=-1).mean())
    result['executed_control_changed']=float((difference.flatten(1).norm(dim=-1)>1.).float().mean())
    if 'groups' in data:
        groups=data['groups']
        result['candidate_context_adaptation']=float(np.mean([len(np.unique(chosen.numpy()[groups==g]))>1 for g in np.unique(groups)]))
    if retain:result.update(chosen=chosen.tolist(),selected_costs=selected.tolist(),probabilities=probabilities.tolist())
    return result


def fit_gates(args):
    """All actors see the same states; only information or Q supervision differs."""
    torch.set_num_threads(1);torch.set_grad_enabled(True)
    training=load_value_training(args)
    validation=load_value_split(args.output,'validation',args.validation_count)
    K=training['costs'].shape[1];history={}
    for arm in ('td','absolute','difference','no_peer'):
        teacher_arm='difference' if arm=='no_peer' else arm
        path=args.output/f'value-{teacher_arm}-{args.seed}.pth';critic,_=restore_value(path)
        with torch.no_grad():
            costs=torch.tensor(value_metrics(critic,training,True)['predictions'],device=args.device)
            costs=costs-costs.mean(-1,keepdim=True)
        torch.manual_seed(args.seed)
        actor=CandidateGateAdapter(peer_candidates=arm!='no_peer',joint_actor=args.joint_actor).to(args.device)
        prior=costs.new_tensor([args.gate_reference_prior,(1.-args.gate_reference_prior)/2.,(1.-args.gate_reference_prior)/2.])
        with torch.no_grad():actor.gate[-1].bias.copy_(prior.log())
        if K==3:joint_prior=prior
        else:
            coordinates=torch.tensor([[0,0],[1,0],[2,0],[0,1],[0,2],[1,1],[1,2],[2,1],[2,2]],device=prior.device)
            joint_prior=prior[coordinates[:,0]]*prior[coordinates[:,1]]
        for key in ('node','pair'):
            values=training['features'][key][::K].flatten(0,-2)
            getattr(actor,key+'_mean').copy_(values.mean(0));getattr(actor,key+'_scale').copy_(values.std(0).clamp_min(.05))
        optimizer=torch.optim.AdamW(actor.parameters(),lr=.0005,weight_decay=.001)
        rng=np.random.default_rng(args.seed);best=float('inf');history[arm]=[]
        for iteration in range(1,args.updates+1):
            index=torch.as_tensor(rng.integers(len(costs),size=64));flat=(index[:,None]*K+torch.arange(K)).flatten()
            batch=dict(features=slice_features(training['features'],flat),costs=training['costs'][index],focal=training['focal'][index])
            probability=gate_probabilities(actor,batch)
            loss=(probability*costs[index]).sum(-1).mean()/5.+args.gate_trust*(probability*(probability.clamp_min(1e-8).log()-joint_prior.log())).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),5.);optimizer.step()
            if iteration%100==0:
                metrics=gate_metrics(actor,validation);history[arm].append(dict(iteration=iteration,**metrics))
                if metrics['decision_cost']<best:
                    best=metrics['decision_cost']
                    atomic_torch(args.output/f'gate-{arm}-{args.seed}.pth',dict(format='conditional-value-gate-v1',model={k:v.detach().cpu() for k,v in actor.state_dict().items()},
                        peer_candidates=actor.peer_candidates,arm=arm,seed=args.seed,iteration=iteration,validation=metrics,
                        joint_actor=actor.joint_actor,
                        intervention_steps=training['intervention_steps'],
                        critic_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),critic=str(path),
                        gate_trust=args.gate_trust,gate_reference_prior=args.gate_reference_prior,
                        extra_train_output=str(args.extra_train_output) if args.extra_train_output else None,extra_train_count=args.extra_train_count,
                        train_roots=args.train_count,validation_roots=args.validation_count,options=K,option_steps=10,fusion_scale=.5))
                if iteration%500==0:print(json.dumps(dict(arm=arm,iteration=iteration,**metrics)),flush=True)
    atomic_json(args.output/f'gate-training-{args.seed}.json',history)


def evaluate_gates(args):
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    data=load_value_split(args.output,'test',args.count,offset=args.offset);results={}
    for arm in ('td','absolute','difference','no_peer'):
        saved=torch.load(args.output/f'gate-{arm}-{args.seed}.pth',weights_only=True)
        model=CandidateGateAdapter(peer_candidates=saved['peer_candidates'],joint_actor=saved.get('joint_actor',False)).eval();model.load_state_dict(saved['model'])
        results[arm]=gate_metrics(model,data,True)
    groups=data['groups'];unique=np.unique(groups);rng=np.random.default_rng(719031)
    bootstrap=rng.integers(len(unique),size=(10000,len(unique)));contrasts={}
    for reference in ('td','absolute','no_peer'):
        delta=np.asarray(results['difference']['selected_costs'])-np.asarray(results[reference]['selected_costs'])
        cluster=np.asarray([delta[groups==group].mean() for group in unique])
        contrasts[reference]=dict(difference=float(cluster.mean()),ci=np.quantile(cluster[bootstrap].mean(-1),[.025,.975]).tolist(),
            action_disagreement=float(np.mean(np.asarray(results['difference']['chosen'])!=np.asarray(results[reference]['chosen']))))
    result=dict(seed=args.seed,roots=len(unique),contexts=len(groups),results=results,contrasts=contrasts,
        groups=groups.tolist(),
        bootstrap_unit='physical root; paired candidate contexts remain grouped')
    suffix=f'-{args.offset}-{args.count}' if args.offset else ''
    atomic_json(args.output/f'gate-test-{args.seed}{suffix}.json',result)
    print(json.dumps(dict(results={k:{n:v for n,v in r.items() if not isinstance(v,list)} for k,r in results.items()},contrasts=contrasts)),flush=True)


def evaluate_values(args):
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    data=load_value_split(args.output,'test',args.count,offset=args.offset)
    results={}
    for arm in args.arms.split(','):
        model,saved=restore_value(args.output/f'value-{arm}-{args.seed}.pth')
        results[arm]=value_metrics(model,data,True)
        print(json.dumps(dict(arm=arm,seed=args.seed,**{k:v for k,v in results[arm].items() if not isinstance(v,list)})),flush=True)
    groups=data['groups'];unique=np.unique(groups)
    rng=np.random.default_rng(928147);ix=rng.integers(len(unique),size=(10000,len(unique)))
    contrasts={}
    for reference in ('td','absolute','difference_raw','difference_no_peer'):
        if reference not in results:continue
        contrasts[reference]={}
        for key in ('errors','selected_costs'):
            delta=np.asarray(results['difference'][key])-np.asarray(results[reference][key])
            cluster=np.asarray([delta[groups==group].mean() for group in unique])
            contrasts[reference][key]=dict(difference=float(cluster.mean()),ci=np.quantile(cluster[ix].mean(-1),[.025,.975]).tolist())
        contrasts[reference]['action_disagreement']=float(np.mean(np.asarray(results['difference']['chosen'])!=np.asarray(results[reference]['chosen'])))
    result=dict(seed=args.seed,roots=len(unique),contexts=len(groups),results=results,contrasts=contrasts,
        groups=groups.tolist(),policy_hashes={arm:hashlib.sha256((args.output/f'value-{arm}-{args.seed}.pth').read_bytes()).hexdigest() for arm in results},
        bootstrap_unit='physical root; all candidate contexts and paired futures stay together')
    suffix=f'-{args.offset}-{args.count}' if args.offset else ''
    atomic_json(args.output/f'value-test-{args.seed}{suffix}.json',result)
    print(json.dumps(contrasts),flush=True)


def summarize_identification(args):
    """Crossed resampling keeps shared scenes and training seeds paired."""
    seeds=[int(value) for value in args.seeds.split(',')]
    suffix=f'-{args.test_offset}-{args.count}' if args.test_offset else ''
    values=[json.loads((args.output/f'value-test-{seed}{suffix}.json').read_text(encoding='utf-8')) for seed in seeds]
    prefix='value' if args.policy_kind=='value' else 'gate'
    episodes=[json.loads((args.output/f'{prefix}-episodes-{seed}-{args.offset}-{args.max_decisions}.json').read_text(encoding='utf-8')) for seed in seeds]
    groups=np.asarray(values[0]['groups']);unique=np.unique(groups)
    scenes=[r['scene'] for r in episodes[0]['rows']]
    if any(v['groups']!=groups.tolist() for v in values):raise ValueError('Unpaired test roots.')
    if any([r['scene'] for r in e['rows']]!=scenes for e in episodes):raise ValueError('Unpaired episode scenes.')
    def interval(array):
        rng=np.random.default_rng(614251);s,n=array.shape
        seed_ix=rng.integers(s,size=(10000,s));scene_ix=rng.integers(n,size=(10000,n))
        bootstrap=array[seed_ix[:,:,None],scene_ix[:,None,:]].mean((1,2))
        return dict(difference=float(array.mean()),ci=np.quantile(bootstrap,[.025,.975]).tolist(),
            per_seed=array.mean(1).tolist())
    value_scores={arm:{key:float(np.mean([v['results'][arm][key] for v in values]))
        for key in ('E_env','decision_cost','improvement','regret','executed_control_l2','executed_control_changed','candidate_context_adaptation')}
        for arm in ('td','absolute','difference','difference_no_peer')}
    value_contrasts={}
    for arm in ('td','absolute','difference_no_peer'):
        value_contrasts[arm]={}
        for metric in ('errors','selected_costs'):
            delta=np.array([np.asarray(v['results']['difference'][metric])-np.asarray(v['results'][arm][metric]) for v in values])
            cluster=np.stack([delta[:,groups==group].mean(1) for group in unique],1)
            value_contrasts[arm][metric]=interval(cluster)
        value_contrasts[arm]['action_disagreement']=float(np.mean([np.asarray(v['results']['difference']['chosen'])!=np.asarray(v['results'][arm]['chosen']) for v in values]))
    episode_scores={arm:{metric:float(np.mean([[r[arm][metric] for r in e['rows']] for e in episodes]))
        for metric in ('capture_time','capture','collision','breach','timeout')}
        for arm in ('td','absolute','difference','no_peer','reference','arboids_cbf','constant_bias')}
    episode_contrasts={arm:interval(np.array([[r['difference']['capture_time']-r[arm]['capture_time'] for r in e['rows']] for e in episodes]))
        for arm in episode_scores if arm!='difference'}
    result=dict(seeds=seeds,physical_test_roots=len(unique),candidate_contexts=len(groups),episode_scenes=len(scenes),test_offset=args.test_offset,
        policy_kind=args.policy_kind,scene_start=scenes[0],max_decisions=args.max_decisions,
        bootstrap='10000 crossed resamples of training seeds and physical scenarios; all arms and candidate contexts stay paired',
        sign='full minus comparison; negative cost/error differences favor the full method',
        value_scores=value_scores,value_contrasts=value_contrasts,episode_scores=episode_scores,episode_contrasts=episode_contrasts)
    atomic_json(args.output/f'mechanism-summary-{args.seeds.replace(",","-")}-{args.offset}-{args.max_decisions}.json',result)
    print(json.dumps(result),flush=True)


def fleet_options(n, rng=None):
    """A reference plus a complete binary face of feedback compositions.

    A negative entry means the source's state-dependent coefficient. During
    collection a random three-vessel face supplies conditional single-vessel
    effects under four different peer configurations, using identical futures.
    """
    if rng is None:
        bits=((np.arange(2**n)[:,None] >> np.arange(n)) & 1).astype(np.float32)
    else:
        bits=np.tile(rng.integers(2,size=n),(8,1)).astype(np.float32)
        selected=rng.choice(n,size=min(3,n),replace=False)
        bits[:,selected]=((np.arange(8)[:,None] >> np.arange(len(selected))) & 1)
    return np.concatenate((np.full((1,n),-1.,dtype=np.float32),bits))


def fleet_lead_rule(observations):
    """Boids attraction to a causal, at most two-second target lead."""
    obs=np.asarray(observations)
    def vector(index):
        return obs[...,index,None]*np.stack((np.cos(obs[...,index+1]),np.sin(obs[...,index+1])),-1)
    seconds=np.clip((obs[...,2]-5.)/3.2,0.,2.)
    force=.5*(vector(2)+seconds[...,None]*vector(4))+10.*vector(6)+.1*vector(8)+.1*vector(10)
    control=force/np.maximum(np.linalg.norm(force,axis=-1,keepdims=True),1e-12)
    yaw=.75*np.arctan2(control[...,1],control[...,0])
    thrust=np.clip(np.stack((control[...,0]+yaw,control[...,0]-yaw),-1),-.5,1.)*1000.
    return ((thrust-250.)/750.).astype(np.float32)


def fleet_active_options(packet,active=0,rng=None):
    """A complete conditional face among the closest measured interceptors."""
    n=len(packet['obs'])
    if not active:return fleet_options(n,rng)
    if active!=3 or n<3:raise ValueError('Active feedback faces require exactly three interceptors.')
    selected=np.argsort(packet['obs'][:,2],kind='stable')[:active]
    bits=np.ones((2**active,n),dtype=np.float32)
    bits[:,selected]=((np.arange(2**active)[:,None] >> np.arange(active)) & 1)
    return np.concatenate((np.full((1,n),-1.,dtype=np.float32),bits))


def fleet_action(base, coefficients, observations=None, candidate_mode='source'):
    if candidate_mode=='feedback':
        if observations is None:raise ValueError('Feedback candidates require public observations.')
        learned=base[...,2,None]*base[...,:2]+(1.-base[...,2,None])*observations[...,12:14]
        rule=fleet_lead_rule(observations)
        theta=np.where(np.asarray(coefficients)<0.,1.,coefficients)[...,None]
        return np.concatenate((theta*learned+(1.-theta)*rule,np.ones_like(theta)),-1).astype(np.float32)
    action=base.copy()
    action[...,2]=np.where(np.asarray(coefficients)<0.,base[...,2],coefficients)
    return action


def fleet_features(packet, base, options):
    """Public, action-conditioned response graph in the target's radial frame.

    Prediction is a fixed feature map, never a task-return oracle. The no-peer
    representation replaces each recipient's peer paths with measured coasting
    paths and erases peer controls, including all their derived quantities.
    """
    from policy.interaction_prediction import InteractionPredictor, PredictionConfig
    m=np.asarray(packet['motion'],dtype=float);obs=packet['obs'];n=len(m);k=len(options)
    bearing=m[:,2]+obs[:,3]
    target=(m[:,:2]+obs[:,2,None]*np.stack((np.cos(bearing),np.sin(bearing)),-1)).mean(0)
    velocity=(obs[:,4,None]*np.stack((np.cos(m[:,2]+obs[:,5]),np.sin(m[:,2]+obs[:,5])),-1)).mean(0)
    phi=np.arctan2(target[1],target[0]);c,s=np.cos(phi),np.sin(phi);rot=np.array([[c,-s],[s,c]])
    theta=np.where(options<0.,base[None,:,2],options)
    nominal=theta[...,None]*base[None,:,:2]+(1.-theta[...,None])*obs[None,:,12:14]
    predictor=InteractionPredictor(PredictionConfig(horizon=2.,dt=.05))
    path=predictor.trajectories(np.broadcast_to(m,(k,n,6)).reshape(-1,6),
        (750.*nominal+250.).reshape(-1,2)).reshape(k,n,41,2)
    times=np.array([0.,.5,1.,2.]);path=path[:,:,[0,10,20,40]]
    target_path=target+times[:,None]*velocity
    future=(path-target_path[None,None])@rot/20.
    pos=(m[:,:2]-target)@rot/20.;v=m[:,3:5]@rot/5.;heading=m[:,2]-phi
    state=np.column_stack((pos,v,np.sin(heading),np.cos(heading),m[:,5],
        np.broadcast_to(velocity@rot/5.,(n,2)),np.full(n,np.linalg.norm(target)/60.),
        np.full(n,packet['central'][-1]),obs[:,2]/20.))
    candidates=np.concatenate((base[:,:2],obs[:,12:14],base[:,2,None]),-1)
    # 12 state, 5 own candidates, 1 coefficient, 2 nominal controls, 8 response.
    node=np.concatenate((np.broadcast_to(state,(k,n,12)),np.broadcast_to(candidates,(k,n,5)),
        theta[...,None],nominal,future.reshape(k,n,8)),-1)
    relative=(m[None,:,:2]-m[:,None,:2])@rot/15.
    relv=(m[None,:,3:5]-m[:,None,3:5])@rot/5.
    angle=m[None,:,2]-m[:,None,2]
    geometry=np.concatenate((relative,relv,np.sin(angle)[...,None],np.cos(angle)[...,None]),-1)
    own=np.concatenate((np.broadcast_to(candidates,(k,n,5)),theta[...,None],nominal),-1)
    coasting=m[None,:,:2]+times[:,None,None]*m[None,:,3:5]
    coasting=np.transpose(coasting,(1,0,2))
    def edges(peer_path, peer_controls):
        separation=(peer_path[:,None]-path[:,:,None])@rot/15.
        return np.concatenate((np.broadcast_to(geometry,(k,n,n,6)),
            np.broadcast_to(own[:,:,None],(k,n,n,8)),
            np.broadcast_to(peer_controls[:,None],(k,n,n,8)),separation.reshape(k,n,n,8)),axis=-1)
    full=edges(path,own)
    no_peer=edges(np.broadcast_to(coasting,(k,n,4,2)),np.zeros_like(own))
    weight=np.exp(-np.maximum(obs[:,2]-obs[:,2].min(),0.)/12.)
    weight=weight/weight.sum()
    return dict(node=node.astype(np.float32),edge=full.astype(np.float32),
        no_peer_edge=no_peer.astype(np.float32),weight=np.broadcast_to(weight,(k,n)).astype(np.float32))


class FleetConditionalValue(nn.Module):
    """Anchored conditional energy with candidate-to-candidate messages."""
    def __init__(self,width=64,no_peer=False):
        super().__init__();self.width=width;self.no_peer=no_peer
        self.node=nn.Sequential(nn.Linear(28,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.edge=nn.Sequential(nn.Linear(30,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.effect=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,1))
        self.state=nn.Sequential(nn.Linear(12,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))

    def forward(self,features):
        x=features['node'];e=features['no_peer_edge'] if self.no_peer else features['edge'];n=x.shape[-2]
        mask=1.-torch.eye(n,device=x.device,dtype=x.dtype)
        distance=e[...,:2].square().sum(-1)
        attention=mask/(1.+distance)
        message=(self.edge(e)*attention[...,None]).sum(-2)/attention.sum(-1,keepdim=True).clamp_min(1e-6)
        effect=(self.effect(torch.cat((self.node(x),message),-1)).squeeze(-1)*features['weight']).sum(-1)
        baseline=(self.state(x[...,:12]).squeeze(-1)*features['weight']).sum(-1)
        return baseline,effect


def initialize_fleet():
    global FLEET_SOURCE
    from interaction_evaluation import OriginalPolicy
    torch.set_num_threads(1);torch.set_grad_enabled(False)
    FLEET_SOURCE=OriginalPolicy(SOURCE)


def fleet_public_packet(env,observations=None,measured_target=False):
    packet=public_packet(env,observations)
    if measured_target:
        # Same public target pose, ground velocity and yaw-rate contract as
        # feedback_joint_control.observe, available equally to every ablation.
        # Water-relative velocity, realized currents and agility stay private.
        a=env.attacker
        packet['target_motion']=np.array([a.x,a.y,a.theta,*a.velocity],dtype=np.float64)
    return packet


def collect_fleet(task):
    index,split,out,n,horizon,repeats,resume,candidate_mode,*extra=task
    active=extra[0] if extra else 0
    if active and candidate_mode!='feedback':raise ValueError('Active faces retain source feedback on other vessels.')
    out=Path(out)/split;path=out/f'fleet-{index}.pth'
    out.mkdir(parents=True,exist_ok=True)
    if resume and path.exists():
        previous=torch.load(path,weights_only=False)
        if (previous['defenders'],previous['horizon'],previous['repeats'])!=(n,horizon,repeats):
            raise ValueError('Cannot resume a different fleet collection protocol.')
        if previous.get('candidate_mode','source')!=candidate_mode:raise ValueError('Different feedback candidates.')
        if previous.get('active_defenders',0)!=active:raise ValueError('Different active-interceptor rule.')
        return dict(index=index,resumed=True)
    split_id={'train':0,'validation':1,'test':2}[split]
    scene=3070000000+split_id*100000+index
    rng=np.random.default_rng(scene+1000000);seed_random(scene)
    env=TADEnv(n,protocol='paper-parameters-v1');obs,_=env.reset(2.25,noisy_agility=False)
    np.random.seed(scene+2000000);packet=public_packet(env,obs);use_source_current_motion(env)
    roots=[];done=0;step=0;coefficients=np.full(n,-1.,dtype=np.float32)
    while not done:
        if step%10==0:
            if env.Current_T>=2. and packet['obs'][:,2].min()<45. and (candidate_mode=='source' or env.Current_T<=20.):
                snapshot=compact_snapshot(env)
                for boat in [snapshot.environment.attacker,*snapshot.environment.defender_list]:
                    boat.__dict__.pop('compute_motion',None)
                roots.append(snapshot)
            # Coverage includes unmodified source trajectories and persistent
            # explorations. The sampling law does not inspect branch outcomes.
            if rng.random()<.35:
                coefficients=(fleet_active_options(packet,active)[1+int(rng.integers(8))] if active else rng.integers(2,size=n).astype(np.float32))
            else:coefficients=np.full(n,-1.,dtype=np.float32)
        action=fleet_action(FLEET_SOURCE(packet['obs']),coefficients,packet['obs'],candidate_mode)
        packet,_,done,_,_,_=execute(env,packet,action);step+=1
    if not roots:raise RuntimeError('No eligible fleet root.')
    root=roots[int(rng.integers(len(roots)))];env=root.restore();packet=public_packet(env)
    base=FLEET_SOURCE(packet['obs']);options=fleet_active_options(packet,active,rng)
    if candidate_mode=='feedback':
        feature_packet=copy.deepcopy(packet);feature_packet['obs'][:,12:14]=fleet_lead_rule(packet['obs'])
        feature_base=fleet_action(base,np.full(n,-1.),packet['obs'],candidate_mode)
        features=fleet_features(feature_packet,feature_base,options)
    else:features=fleet_features(packet,base,options)
    branches=[];packets=[];states=[];count=len(options)*repeats
    futures=[3170000000+split_id*1000000+index*32+r for r in range(repeats)]
    for r in range(repeats):
        for option in options:
            branch=root.restore(future_seed=futures[r]);use_source_current_motion(branch)
            branches.append(branch);packets.append(public_packet(branch));states.append(np.random.get_state())
    active=list(range(count));outcomes=np.zeros(count,dtype=int);continuations=[]
    for step in range(300):
        if not active:break
        bases=FLEET_SOURCE(np.concatenate([packets[i]['obs'] for i in active])).reshape(len(active),n,3)
        actions=[fleet_action(bases[j],options[i%len(options)],packets[i]['obs'],candidate_mode) if step<horizon else bases[j] for j,i in enumerate(active)]
        applied=batch_filtered([packets[i] for i in active],actions,[branches[i].boids_actions for i in active])
        following=[]
        for j,i in enumerate(active):
            np.random.set_state(states[i]);obs,_,done,_=branches[i].step(actions[j],'AdaRes',defender_thrust=applied[j])
            states[i]=np.random.get_state();packets[i]=public_packet(branches[i],obs)
            if done:outcomes[i]=int(done)
            else:following.append(i)
        active=following
    if active:raise RuntimeError('Unfinished fleet branch.')
    t=float(root.environment.Current_T)
    costs=np.array([b.Current_T-t if outcomes[i]==3 else 60.-t for i,b in enumerate(branches)],dtype=np.float32).reshape(repeats,-1)
    record=dict(scene=scene,index=index,split=split,defenders=n,horizon=horizon,repeats=repeats,
        packet=packet,base=base,options=options,features=features,costs=costs,futures=futures,
        root_time=t,terminations=outcomes.reshape(repeats,-1),protocol='paper-parameters-v1',
        snapshot=root,candidate_mode=candidate_mode,active_defenders=active,objective='actual capped capture time; fixed feedback option then source continuation')
    atomic_torch(path,record)
    return dict(index=index,spread=float(np.ptp(costs.mean(0))),oracle_gain=float(costs[:,0].mean()-costs.mean(0).min()))


def load_fleet(out,split,count,device):
    rows=[torch.load(Path(out)/split/f'fleet-{i}.pth',weights_only=False) for i in range(count)]
    features={k:torch.as_tensor(np.stack([row['features'][k] for row in rows]),device=device) for k in rows[0]['features']}
    costs=torch.as_tensor(np.stack([row['costs'].mean(0) for row in rows]),device=device)
    options=np.stack([row['options'] for row in rows]);edges=[]
    for opts in options:
        pair=np.abs(opts[:,None]-opts[None]).sum(-1)==1.
        pair[0]=False;pair[:,0]=False
        edges.append(pair)
    return dict(features=features,costs=costs,edges=torch.as_tensor(np.stack(edges),device=device),rows=rows)


def fleet_metrics(model,data):
    with torch.no_grad():
        b,q=model(data['features']);delta=q-q[:,:1];truth=data['costs']-data['costs'][:,:1]
        choice=q.argmin(-1);selected=data['costs'].gather(-1,choice[:,None]).squeeze(-1)
        return dict(contrast_mae=float((delta[:,1:]-truth[:,1:]).abs().mean()),
            gain=float((data['costs'][:,0]-selected).mean()),regret=float((selected-data['costs'].min(-1).values).mean()),
            changes=float((choice!=0).float().mean()))


def fit_fleet(args):
    torch.set_num_threads(1);torch.set_grad_enabled(True);device=args.device
    train=load_fleet(args.output,'train',args.train_count,device);valid=load_fleet(args.output,'validation',args.validation_count,device)
    for arm in args.arms.split(','):
        if arm not in ('difference','absolute','no_peer'):raise ValueError('Fleet arms: difference,absolute,no_peer.')
        torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
        model=FleetConditionalValue(args.width,arm=='no_peer').to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.001)
        best=None;best_score=float('inf');history=[]
        for step in range(1,args.updates+1):
            ix=torch.as_tensor(rng.integers(args.train_count,size=min(32,args.train_count)),device=device)
            features={k:v[ix] for k,v in train['features'].items()};cost=train['costs'][ix]
            b,q=model(features);delta=q-q[:,:1];true=cost-cost[:,:1]
            if arm=='absolute':
                loss=F.smooth_l1_loss((b+delta)/10.,cost/10.)
            else:
                mask=train['edges'][ix]
                error=(q[:,:,None]-q[:,None,:]-(cost[:,:,None]-cost[:,None,:]))/5.
                loss=(F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')*mask).sum()/mask.sum().clamp_min(1)
                loss+=.25*F.smooth_l1_loss(delta/10.,true/10.)
                # Conditional supervision allocates its budget to ranking
                # feasible compositions, with the same actual returns as the
                # absolute-return control arm.
                target=torch.softmax(-cost/3.,-1)
                loss+=.2*-(target*torch.log_softmax(-q/3.,-1)).sum(-1).mean()
                loss+=F.smooth_l1_loss(b[:,0]/10.,cost[:,0]/10.)
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if step%100==0 or step==args.updates:
                metrics=fleet_metrics(model,valid);history.append(dict(step=step,**metrics))
                # Selection uses validation control regret, not merely Q error.
                score=metrics['regret']+.05*metrics['contrast_mae']
                if score<best_score:
                    best_score=score;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_step=step
                if step%500==0 or step==args.updates:print(json.dumps(dict(arm=arm,step=step,**metrics)),flush=True)
        saved=dict(model=best,seed=args.seed,arm=arm,width=args.width,train_count=args.train_count,
            validation_count=args.validation_count,horizon=train['rows'][0]['horizon'],defenders=train['rows'][0]['defenders'],
            objective_version='fleet-conditional-feedback-v1',selection_step=best_step,history=history,
            source=str(SOURCE),learning_rate=args.learning_rate,updates=args.updates)
        atomic_torch(args.output/f'fleet-{arm}-{args.seed}.pth',saved)


class FleetValuePolicy:
    def __init__(self,path,period=None):
        from interaction_evaluation import OriginalPolicy
        saved=torch.load(path,map_location='cpu',weights_only=False);self.saved=saved
        self.model=FleetConditionalValue(saved['width'],saved['arm']=='no_peer').eval().requires_grad_(False)
        self.model.load_state_dict(saved['model']);self.source=OriginalPolicy(saved['source'])
        self.period=saved['horizon'] if period is None else period;self.reset()

    def reset(self):self.remaining=0;self.coefficients=None;self.decisions=[]

    @torch.no_grad()
    def choose_action(self,packet,deterministic=True):
        base=self.source(packet['obs'])
        if 60.*(1.-float(packet['central'][-1]))+1e-5<getattr(self,'start_time',0.):
            return base,0.
        if self.remaining==0:
            options=fleet_options(len(base));features={k:torch.as_tensor(v) for k,v in fleet_features(packet,base,options).items()}
            _,q=self.model(features);choice=int(q.argmin());self.coefficients=options[choice].copy()
            self.remaining=self.period;self.decisions.append(dict(choice=choice,gain=float(q[0]-q[choice])))
        self.remaining-=1
        return fleet_action(base,self.coefficients),0.


@torch.no_grad()
def fleet_rollout_features(packet,source,options,steps=30,candidate_mode='source',repeats=1):
    """Finite public-model response followed by a learned continuation value.

    All choices share one causal state estimate. The predictor never receives
    central dynamics, actual future currents or the simulator snapshot.
    """
    dynamics=DifferentiableInterventionDynamics
    m=np.asarray(packet['motion']);obs=packet['obs'];n=len(m);k=len(options)
    bearing=m[:,2]+obs[:,3]
    target=(m[:,:2]+obs[:,2,None]*np.stack((np.cos(bearing),np.sin(bearing)),-1)).mean(0)
    direction=m[:,2]+obs[:,5]
    vel=(obs[:,4,None]*np.stack((np.cos(direction),np.sin(direction)),-1)).mean(0)
    heading=float(np.arctan2(vel[1],vel[0])) if np.linalg.norm(vel)>.2 else float(np.arctan2(-target[1],-target[0]))
    target_state=np.asarray(packet.get('target_motion',[*target,heading,*vel,0.]),dtype=float)
    initial=np.concatenate((m,target_state[None]),0)
    # Extra unary worlds evaluate each receiver with reference peer controls.
    singles=[];lookup={}
    for i in range(n):
        for value in np.unique(options[:,i]):
            if value<0.:continue
            single=np.full(n,-1.,dtype=np.float32);single[i]=value
            lookup[i,float(value)]=k+len(singles);singles.append(single)
    programs=np.concatenate((options,np.asarray(singles)),axis=0);program_count=len(programs)
    if repeats<1:raise ValueError('At least one public-model response is required.')
    programs=np.tile(programs,(repeats,1))
    # These are fixed synthetic futures, independent of the real simulator's
    # random stream and scenario identity, shared across every candidate.
    innovations=(np.random.default_rng(34911803).normal(size=(repeats,steps,2+3*(n+1))).astype(np.float32)
        if repeats>1 else None)
    count=len(programs);state=torch.as_tensor(np.broadcast_to(initial,(count,n+1,6)).copy(),dtype=torch.float32)
    measured=state[...,3:6].clone();agility=torch.full((count,),2.25);remaining=torch.full((count,),float(packet['central'][-1]))
    coefficients=torch.as_tensor(programs);alive=torch.ones(count,dtype=torch.bool);captured=torch.zeros(count,dtype=torch.bool)
    failed=torch.zeros_like(captured);elapsed=torch.zeros(count);minimum=torch.full((count,),100.)
    for step in range(steps):
        public=dynamics.packet(state,measured,agility,remaining-step*.2/60.)
        if step==0:
            observations=np.broadcast_to(obs,(count,*obs.shape)).reshape(count*n,-1)
        else:observations=public['obs'].reshape(count*n,-1).numpy()
        raw=source(observations).reshape(count,n,3)
        if candidate_mode=='feedback':
            composed=fleet_action(raw,programs,observations.reshape(count,n,-1),candidate_mode)
            nominal=torch.as_tensor(750.*composed[...,:2]+250.)
        else:
            base=torch.as_tensor(raw);base[...,2]=torch.where(coefficients<0.,base[...,2],coefficients)
            nominal=base[...,2,None]*(750.*base[...,:2]+250.)+(1.-base[...,2,None])*(750.*public['obs'][...,12:14]+250.)
            if step==0:nominal=base[...,2,None]*(750.*base[...,:2]+250.)+(1.-base[...,2,None])*torch.as_tensor(750.*obs[None,:,12:14]+250.)
        motion=torch.cat((state[:,:-1,:3],measured[:,:-1]),-1)
        forward,_=safety_derivatives(n)
        # Small groups retain the original CBF and bound the QP batch size.
        force=[]
        for start in range(0,count,16):
            x=motion[start:start+16].double().numpy();u=nominal[start:start+16].double().numpy();used=len(x)
            if used<16:
                x=np.concatenate((x,np.repeat(x[-1:],16-used,0)));u=np.concatenate((u,np.repeat(u[-1:],16-used,0)))
            force.append(np.asarray(forward(x,u))[:used].copy())
        defender=torch.as_tensor(np.concatenate(force),dtype=torch.float32)
        apf=dynamics.apf(state);current=torch.zeros_like(measured)
        if innovations is not None:
            noise=torch.as_tensor(innovations[:,step]).repeat_interleave(program_count,dim=0)
            apf=apf+.3*noise[:,:2]
            currents=noise[:,2:].reshape(count,n+1,3)
            current=torch.cat((currents[:,1:],currents[:,:1]),1)*state.new_tensor([.2,.2,.02])
        attacker=dynamics.thrust(apf,state[:,-1,2],agility[:,None])
        following,observed=dynamics.vessel(state,torch.cat((defender,attacker[:,None]),1),current)
        state=torch.where(alive[:,None,None],following,state);measured=torch.where(alive[:,None,None],observed,measured)
        elapsed+=alive*.2
        distance=(state[:,:-1,:2]-state[:,-1:,:2]).norm(dim=-1).min(-1).values
        separation=(state[:,:-1,None,:2]-state[:,None,:-1,:2]).norm(dim=-1)+100.*torch.eye(n)[None]
        collision=separation.flatten(1).min(-1).values<5.
        breach=state[:,-1,:2].norm(dim=-1)<15.
        minimum=torch.minimum(minimum,distance)
        success=(distance<4.5)&~collision&~breach&alive
        captured|=success;failed|=(collision|breach)&alive
        alive&=~(success|collision|breach)
        if not alive.any():break
    position=state[:,:-1,:2]-state[:,-1:,:2];distance=position.norm(dim=-1)
    radial=state[:,-1,:2].norm(dim=-1)
    relative=measured[:,:-1,:2]-measured[:,-1:,:2]
    closing=-(position*relative).sum(-1)/distance.clamp_min(.1)
    toward=torch.atan2(-position[...,1],-position[...,0])-state[:,:-1,2]
    angle=torch.atan2(toward.sin(),toward.cos()).abs()
    guard=radial-state[:,:-1,:2].norm(dim=-1).min(-1).values
    sorted_distance=distance.sort(-1).values
    radial_unit=state[:,-1,:2]/radial[:,None].clamp_min(.01)
    longitudinal=(position*radial_unit[:,None]).sum(-1)
    lateral=position[...,0]*radial_unit[:,None,1]-position[...,1]*radial_unit[:,None,0]
    # A compact continuation representation limits variance from noisy returns.
    alive_float=alive.float();cap=captured.float();bad=failed.float()
    ETA=((distance-4.5).clamp_min(0.)/3.2+.4*angle/.85).min(-1).values
    if candidate_mode=='feedback':
        from predictive_interception_v2 import reachable_capture_time
        public_end=torch.cat((state[...,:3],measured),-1).numpy()
        ETA=torch.as_tensor(np.array([reachable_capture_time(s[:-1],s[-1],radius=4.5).min() for s in public_end]),dtype=torch.float32)
    features=torch.stack((torch.ones(count),elapsed,cap,bad,alive_float,
        alive_float*ETA,alive_float*ETA.square()/20.,alive_float*sorted_distance[:,0]/5.,
        alive_float*sorted_distance[:,1]/5.,alive_float*sorted_distance[:,:3].mean(-1)/5.,
        alive_float*distance.mean(-1)/10.,alive_float*closing.max(-1).values,alive_float*closing.mean(-1),
        alive_float*angle.mean(-1),alive_float*radial/20.,alive_float*torch.relu(5.-guard),
        alive_float*torch.relu(-longitudinal).mean(-1)/10.,alive_float*lateral.std(-1)/10.,
        alive_float*minimum/5.,alive_float*measured[:,-1,:2].norm(dim=-1),
        bad*torch.full((count,),float(packet['central'][-1])*60.)),dim=-1).numpy().astype(np.float32)
    features=features.reshape(repeats,program_count,-1).mean(0)
    physical_cost=torch.where(failed,torch.full_like(elapsed,float(packet['central'][-1])*60.),elapsed+alive_float*ETA)
    physical_cost=physical_cost.numpy().reshape(repeats,program_count).mean(0)[:k]
    full=features[:k];unary=np.tile(features[0],(k,1))
    for j,program in enumerate(options):
        for i,value in enumerate(program):
            if value>=0.:unary[j]+=features[lookup[i,float(value)]]-features[0]
    # Public root basis is shared by all alternatives and cancels exactly
    # from conditional comparisons; absolute supervision must estimate it.
    root=np.array([1.,np.linalg.norm(target)/60.,float(packet['central'][-1]),
        obs[:,2].min()/20.,obs[:,2].mean()/20.,obs[:,2].std()/20.,np.linalg.norm(vel)/5.,
        np.linalg.norm(m[:,3:5],axis=-1).mean()/5.],dtype=np.float32)
    return dict(full=full,no_peer=unary,root=root,model_cost=physical_cost)


def fleet_projection_keys(measured_target,repeats=1):
    key='rollout_measured' if measured_target else 'rollout_features'
    if repeats>1:key+=f'_mc{repeats}'
    return key,('rollout_steps' if not measured_target and repeats==1 else key+'_steps')


def project_fleet(task):
    split,index,output,steps,measured_target,repeats=task;path=Path(output)/split/f'fleet-{index}.pth'
    row=torch.load(path,weights_only=False)
    feature_key,step_key=fleet_projection_keys(measured_target,repeats)
    if row.get(step_key)==steps:return dict(index=index,resumed=True)
    packet=(fleet_public_packet(row['snapshot'].environment,row['packet']['obs'],True) if measured_target else row['packet'])
    row[feature_key]=fleet_rollout_features(packet,FLEET_SOURCE,row['options'],steps,row.get('candidate_mode','source'),repeats)
    row[step_key]=steps;atomic_torch(path,row)
    return dict(index=index)


def fleet_contrast_precision(costs,floor=4.,local_weight=.5):
    """Shrunk covariance of paired mean contrasts, in an independent basis."""
    covariances=[]
    for outcomes in costs:
        outcomes=np.asarray(outcomes,dtype=float)
        if outcomes.ndim!=2 or len(outcomes)<2:
            raise ValueError('Conditional covariance requires repeated paired futures.')
        differences=outcomes[:,1:]-outcomes[:,:1]
        covariances.append(np.cov(differences,rowvar=False,ddof=1)/len(outcomes))
    covariances=np.stack(covariances);pooled=covariances.mean(0)
    if floor<=0. or not 0.<=local_weight<=1.:raise ValueError('Invalid covariance regularization.')
    covariance=local_weight*covariances+(1.-local_weight)*pooled[None]+floor*np.eye(covariances.shape[-1])[None]
    precision=np.linalg.inv(covariance)
    return precision/(np.trace(precision,axis1=1,axis2=2).mean()/covariances.shape[-1])


def fleet_context_features(features,root,normalization):
    """State-conditioned coefficients preserve unary additivity exactly."""
    if normalization is None:return features
    state=np.tanh((np.asarray(root)[...,1:]-np.asarray(normalization['mean']))/np.asarray(normalization['scale']))
    context=np.concatenate((np.ones((*state.shape[:-1],1)),state),axis=-1)
    return (features[...,None]*context[...,None,None,:]).reshape(*features.shape[:-1],-1)


def fleet_anchor_equations(gram,rhs,dx,roots,values,weights,equations):
    """Total-value anchor plus a contrast block with zero baseline gradient."""
    design=np.concatenate((np.broadcast_to(roots[:,None],(*dx.shape[:2],roots.shape[-1])),dx),-1)
    design=design.reshape(-1,design.shape[-1]);weight=np.repeat(weights,dx.shape[1])
    absolute_gram=design.T@(weight[:,None]*design);absolute_rhs=design.T@(weight*values.ravel())
    conditional_gram=np.zeros_like(absolute_gram);conditional_rhs=np.zeros_like(absolute_rhs)
    factor=dx.shape[1]/equations
    conditional_gram[-dx.shape[-1]:,-dx.shape[-1]:]=factor*gram
    conditional_rhs[-dx.shape[-1]:]=factor*rhs
    return absolute_gram,absolute_rhs,conditional_gram,conditional_rhs


def fleet_response_basis(features,options,basis):
    """Nonlinear continuation before the additive/interaction decomposition.

    Every unary value is evaluated with all peers at source feedback. Applying
    nonlinearity to the summed unary features instead would leak interactions
    into the no-peer control and invalidate that ablation.
    """
    if basis is None:return features
    full=np.asarray(features['full'],dtype=float)
    indices=basis['indices'];z=np.clip((full[:,indices]-basis['mean'])/basis['scale'],-5.,5.)
    i,j=np.triu_indices(len(indices))
    transformed=np.concatenate((full,z[:,i]*z[:,j]),axis=-1)
    lookup={tuple(np.where(row<0.,1.,row)):k for k,row in enumerate(options)}
    reference=transformed[0];unary=np.tile(reference,(len(options),1))
    for vessel in range(options.shape[1]):
        mask=options[:,vessel]==0.
        if not mask.any():continue
        single=np.ones(options.shape[1]);single[vessel]=0.
        if tuple(single) not in lookup:
            raise ValueError('Nonlinear continuation requires a complete reference-centered candidate face.')
        unary[mask]+=transformed[lookup[tuple(single)]]-reference
    return dict(features,full=transformed,no_peer=unary)


def fleet_ridge_solution(gram,rhs,penalty,monotone=False):
    matrix=gram+np.diag(penalty)
    solution=np.linalg.solve(matrix,rhs)
    if not monotone:return solution
    from scipy.optimize import minimize
    # Shared physical continuation: more remaining distance/time/risk cannot
    # lower the predicted cost, and capture/closing cannot increase it.
    bounds=[(None,None)]*len(solution);offset=len(solution)-21
    for i in (1,3,5,6,7,8,9,10,13,15,18,20):bounds[offset+i]=(0.,None)
    for i in (2,11,12):bounds[offset+i]=(None,0.)
    scale=np.sqrt(np.diag(matrix));normalized=matrix/scale[:,None]/scale[None,:];target=rhs/scale
    initial=solution*scale
    for i,(lo,hi) in enumerate(bounds):
        if lo is not None:initial[i]=max(initial[i],lo)
        if hi is not None:initial[i]=min(initial[i],hi)
    def objective(z):return .5*z@normalized@z-target@z,normalized@z-target
    solved=minimize(objective,initial,jac=True,bounds=bounds,method='L-BFGS-B',options=dict(maxiter=2000,ftol=1e-13,gtol=1e-8))
    if not solved.success:raise RuntimeError(f'Constrained value fitting did not converge: {solved.message}')
    return solved.x/scale


def fit_fleet_ridge(args):
    torch.set_num_threads(1)
    if getattr(args,'mixed_only',False) and (not args.conditional_edges or not args.interaction_residual or args.contrast_covariance):
        raise ValueError('Mixed-only supervision requires explicit conditional edges and a separate interaction residual, without GLS override.')
    if getattr(args,'monotone_readout',False) and (args.interaction_residual or getattr(args,'nonlinear_continuation',False) or getattr(args,'context_modulation',False)):
        raise ValueError('Monotone readout requires the shared linear physical continuation basis.')
    data={split:[torch.load(args.output/split/f'fleet-{i}.pth',weights_only=False) for i in range(count)]
        for split,count in [('train',args.train_count),('validation',args.validation_count)]}
    rows=data['train'];valid=data['validation'];rng=np.random.default_rng(args.seed if args.bootstrap_seed is None else args.bootstrap_seed)
    for key,default in [('defenders',None),('horizon',None),('candidate_mode','source'),('active_defenders',0)]:
        if any(r.get(key,default)!=rows[0].get(key,default) for r in rows+valid):
            raise ValueError(f'Mixed fleet data contracts for {key}.')
    # Repeated fits use paired scene bootstrap weights, shared by every arm.
    weight=(np.ones(len(rows)) if getattr(args,'no_bootstrap',False)
        else rng.multinomial(len(rows),np.full(len(rows),1./len(rows))).astype(float))
    feature_key,step_key=fleet_projection_keys(args.measured_target,getattr(args,'projection_repeats',1))
    response_basis=None
    if getattr(args,'nonlinear_continuation',False):
        if rows[0].get('candidate_mode')!='feedback':raise ValueError('Nonlinear continuation currently uses feedback faces.')
        indices=[5,7,8,10,11,13,14,15,17,18,19]
        physical=np.concatenate([r[feature_key]['full'] for r in rows])[:,indices].astype(float)
        response_basis=dict(indices=indices,mean=physical.mean(0).tolist(),scale=np.maximum(physical.std(0),.25).tolist())
    train_context=np.stack([r[feature_key]['root'][1:] for r in rows]).astype(float)
    normalization=(dict(mean=train_context.mean(0).tolist(),scale=np.maximum(train_context.std(0),.1).tolist())
        if getattr(args,'context_modulation',False) else None)
    for arm in args.arms.split(','):
        if arm not in ('difference','absolute','no_peer','absolute_no_peer'):
            raise ValueError(f'Unknown fleet learning arm: {arm}')
        no_peer=arm in ('no_peer','absolute_no_peer');absolute=arm.startswith('absolute')
        name='no_peer' if no_peer else 'full'
        def representation(records):
            responses=[fleet_response_basis(r[feature_key],r['options'],response_basis) for r in records]
            feature=np.stack([r[name] for r in responses]).astype(float)
            if args.interaction_residual:
                unary=np.stack([r['no_peer'] for r in responses]).astype(float)
                feature=np.concatenate((unary,feature-unary),-1)
            return fleet_context_features(feature,np.stack([r[feature_key]['root'] for r in records]),normalization)
        x=representation(rows);v=representation(valid)
        y=np.stack([r['costs'].mean(0) for r in rows]).astype(float)
        vy=np.stack([r['costs'].mean(0) for r in valid]).astype(float)
        roots=np.stack([r[feature_key]['root'] for r in rows]).astype(float)
        vroots=np.stack([r[feature_key]['root'] for r in valid]).astype(float)
        scales=np.maximum(x.std((0,1)),.25);dx=(x-x[:,:1])/scales
        if absolute:
            a=np.concatenate((np.broadcast_to(roots[:,None],(*x.shape[:2],roots.shape[-1])),dx),-1);target=y
        else:
            if args.conditional_edges:
                # A face edge changes precisely one vessel while retaining the
                # other candidates. Four-corner differences isolate changes of
                # that conditional effect, rather than assigning common return
                # variation to a candidate interaction.
                design=[];targets=[]
                if not getattr(args,'mixed_only',False):
                    for bit in range(3):
                        for corner in range(8):
                            if corner & (1<<bit):continue
                            lo=1+corner;hi=1+(corner | (1<<bit))
                            design.append((x[:,hi]-x[:,lo])/scales)
                            targets.append(y[:,hi]-y[:,lo])
                if args.interaction_residual:
                    for first in range(3):
                        for second in range(first+1,3):
                            for corner in range(8):
                                if corner & ((1<<first)|(1<<second)):continue
                                z=1+corner;i=1+(corner|(1<<first));j=1+(corner|(1<<second));ij=1+(corner|(1<<first)|(1<<second))
                                design.append((x[:,ij]-x[:,i]-x[:,j]+x[:,z])/scales)
                                targets.append(y[:,ij]-y[:,i]-y[:,j]+y[:,z])
                # The source alternative anchors policy improvement; it has a
                # smaller weight than identifiable conditional face effects.
                if not getattr(args,'mixed_only',False):
                    design.extend([.25*dx[:,j] for j in range(1,len(x[0]))])
                    targets.extend([.25*(y[:,j]-y[:,0]) for j in range(1,len(x[0]))])
                elif no_peer:
                    # The exact mixed derivative of an additive model is zero.
                    # Avoid numerical cancellation producing spurious updates.
                    design=[np.zeros_like(term) for term in design]
                a=np.stack(design,1);target=np.stack(targets,1)
            else:
                a=(x-x.mean(1,keepdims=True))/scales;target=y-y.mean(1,keepdims=True)
        equations=a.shape[1]
        a=a.reshape(-1,a.shape[-1]);target=target.ravel();w=np.repeat(weight,equations)
        best=None;history=[];best_score=float('inf')
        gram=a.T@(w[:,None]*a);rhs=a.T@(w*target)
        if not absolute and args.contrast_covariance:
            # Common futures correlate the reference contrasts. A complete
            # independent contrast basis avoids counting redundant edges as
            # independent observations; every single-vessel face contrast is
            # a linear combination of this basis.
            precision=fleet_contrast_precision([row['costs'] for row in rows],getattr(args,'covariance_floor',4.),getattr(args,'covariance_local_weight',.5))
            design=dx[:,1:];target=y[:,1:]-y[:,:1]
            gram=np.einsum('bki,bkl,blj,b->ij',design,precision,design,weight,optimize=True)
            rhs=np.einsum('bki,bkl,bl,b->i',design,precision,target,weight,optimize=True)
        anchored=getattr(args,'absolute_anchor',False) and not absolute
        if anchored:
            ag,ar,cg,cr=fleet_anchor_equations(gram,rhs,dx,roots,y,weight,x.shape[1]-1 if args.contrast_covariance else equations)
        for auxiliary in ((0.,.1,.3,1.,3.,10.,30.) if anchored else (0.,)):
            fit_gram=ag+auxiliary*cg if anchored else gram
            fit_rhs=ar+auxiliary*cr if anchored else rhs
            for ridge in (.001,.01,.1,1.,10.,100.,1000.,10000.):
                for interaction_ridge in ((1.,10.,100.,1000.,10000.,1e8) if args.interaction_residual and not no_peer else (ridge,)):
                    penalty=np.full(fit_gram.shape[0],ridge)
                    if args.interaction_residual:penalty[-x.shape[-1]//2:]=interaction_ridge
                    if normalization is not None:
                        controls=penalty[-x.shape[-1]:].reshape(-1,8);controls[:,1:]*=10.
                    solution=fleet_ridge_solution(fit_gram,fit_rhs,penalty,getattr(args,'monotone_readout',False))
                    coefficient=solution[-x.shape[-1]:]/scales
                    pred=v@coefficient
                    for margin in ((0.,.25,.5,1.,2.,4.,8.,16.) if args.calibrate_margin else (0.,)):
                        score_value=pred.copy();score_value[:,1:]+=margin
                        choice=score_value.argmin(-1);selected=vy[np.arange(len(v)),choice]
                        error=(pred-pred[:,:1]-vy+vy[:,:1])[:,1:]
                        actual_change=np.array([np.any(r['options'][c]==0.) if r.get('candidate_mode')=='feedback' else c!=0 for r,c in zip(valid,choice)])
                        row=dict(ridge=ridge,interaction_ridge=interaction_ridge,trust_margin=margin,auxiliary_strength=auxiliary,
                            gain=float((vy[:,0]-selected).mean()),contrast_mae=float(np.abs(error).mean()),contrast_mse=float(np.square(error).mean()),changes=float(actual_change.mean()))
                        history.append(row)
                        criterion=getattr(args,'validation_criterion','gain')
                        score=(-row['gain']+.01*row['contrast_mae'] if criterion=='gain' else row['contrast_'+criterion])
                        if score<best_score:best_score=score;best=(coefficient.copy(),ridge,interaction_ridge,margin,auxiliary)
        saved=dict(kind='rollout-ridge',coefficient=best[0],ridge=best[1],seed=args.seed,arm=arm,
            source=str(SOURCE),horizon=rows[0]['horizon'],rollout_steps=rows[0][step_key],defenders=rows[0]['defenders'],projection_repeats=getattr(args,'projection_repeats',1),
            train_count=len(rows),validation_count=len(valid),history=history,bootstrap_weights=weight,
            candidate_mode=rows[0].get('candidate_mode','source'),active_defenders=rows[0].get('active_defenders',0),interaction_residual=args.interaction_residual,interaction_ridge=best[2],conditional_edges=args.conditional_edges,
            measured_target=args.measured_target,trust_margin=best[3],bootstrap_seed=args.seed if args.bootstrap_seed is None else args.bootstrap_seed,
            conditional_covariance=args.contrast_covariance,
            data_hashes={split:hashlib.sha256(b''.join(hashlib.sha256((args.output/split/f'fleet-{i}.pth').read_bytes()).digest()
                for i in range(len(records)))).hexdigest() for split,records in data.items()},
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),context_normalization=normalization,
            covariance_floor=getattr(args,'covariance_floor',4.),covariance_local_weight=getattr(args,'covariance_local_weight',.5))
        saved['bootstrap_strategy']='uniform' if getattr(args,'no_bootstrap',False) else 'paired-multinomial'
        saved.update(absolute_anchor=anchored,conditional_auxiliary_strength=best[4] if anchored else None,
            mixed_only=getattr(args,'mixed_only',False),validation_criterion=getattr(args,'validation_criterion','gain'),response_basis=response_basis,
            monotone_readout=getattr(args,'monotone_readout',False))
        atomic_torch(args.output/f'fleet-{arm}-{args.seed}.pth',saved)
        print(json.dumps(dict(arm=arm,selected=next(r for r in history if r['ridge']==best[1] and r['interaction_ridge']==best[2] and r['trust_margin']==best[3] and r['auxiliary_strength']==best[4]))),flush=True)


def rank_fleet(args):
    """Decision-focused conditional supervision, with the paired GLS anchor."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    if args.initial_seed is None or args.initial_seed==args.seed:
        raise ValueError('Specify a distinct frozen --initial-seed for conditional refinement.')
    torch.set_num_threads(1)
    records={split:[torch.load(args.output/split/f'fleet-{i}.pth',weights_only=False) for i in range(count)]
        for split,count in [('train',args.train_count),('validation',args.validation_count)]}
    for arm in args.arms.split(','):
        path=args.output/f'fleet-{arm}-{args.initial_seed}.pth';saved=torch.load(path,weights_only=False)
        if (saved['train_count'],saved['validation_count'])!=(args.train_count,args.validation_count):
            raise ValueError('Refinement must use the same training and validation sets as its parent.')
        if not saved.get('interaction_residual') or not saved.get('conditional_covariance'):
            raise ValueError('Refinement requires a paired GLS interaction readout.')
        if saved.get('absolute_anchor'):
            raise ValueError('This ranking refiner does not preserve the total-value anchor.')
        saved=dict(saved,seed=args.seed,parent_seed=args.initial_seed,parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if arm.startswith('absolute'):
            # The absolute-value comparator is already globally optimized for
            # its squared-error objective; it receives no conditional labels.
            atomic_torch(args.output/f'fleet-{arm}-{args.seed}.pth',saved);continue
        feature_key,_=fleet_projection_keys(saved.get('measured_target',False),saved.get('projection_repeats',1))
        def arrays(rows):
            unary=np.stack([r[feature_key]['no_peer'] for r in rows]).astype(float)
            joint=np.stack([r[feature_key]['full'] for r in rows]).astype(float)
            feature=np.concatenate((unary,np.zeros_like(unary) if arm=='no_peer' else joint-unary),axis=-1)
            feature=fleet_context_features(feature,np.stack([r[feature_key]['root'] for r in rows]),saved.get('context_normalization'))
            values=np.stack([r['costs'].mean(0) for r in rows]).astype(float)
            return feature,values
        x,y=arrays(records['train']);v,vy=arrays(records['validation'])
        scales=np.maximum(x.std((0,1)),.25);design=(x-x[:,:1])/scales;target=y[:,1:]-y[:,:1]
        weight=saved['bootstrap_weights'];denominator=weight.sum()*(x.shape[1]-1)
        precision=fleet_contrast_precision([r['costs'] for r in records['train']],saved.get('covariance_floor',4.),saved.get('covariance_local_weight',.5))
        a=design[:,1:]
        gram=np.einsum('bki,bkl,blj,b->ij',a,precision,a,weight,optimize=True)/denominator
        rhs=np.einsum('bki,bkl,bl,b->i',a,precision,target,weight,optimize=True)/denominator
        penalty=np.full(x.shape[-1],saved['ridge']);penalty[-x.shape[-1]//2:]=saved['interaction_ridge']
        if saved.get('context_normalization') is not None:penalty.reshape(-1,8)[:,1:]*=10.
        gram+=np.diag(penalty/denominator)
        initial=saved['coefficient']*scales;best_score=float('inf');best=None;history=[]
        for temperature in (1.,3.,6.):
            log_teacher=-y/temperature;log_teacher-=logsumexp(log_teacher,axis=1,keepdims=True)
            teacher=np.exp(log_teacher)
            for strength in (0.,.3,1.,3.,10.):
                def objective(coefficient):
                    logits=-(design@coefficient)/temperature
                    log_probability=logits-logsumexp(logits,axis=1,keepdims=True)
                    probability=np.exp(log_probability)
                    loss=float(coefficient@gram@coefficient-2.*rhs@coefficient)
                    gradient=2.*(gram@coefficient-rhs)
                    loss+=strength*temperature**2*np.sum(weight[:,None]*teacher*(log_teacher-log_probability))/weight.sum()
                    gradient+=strength*temperature*np.einsum('bki,bk,b->i',design,teacher-probability,weight)/weight.sum()
                    return loss,gradient
                solved=minimize(objective,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-12,gtol=1e-7))
                coefficient=solved.x/scales;pred=v@coefficient;choice=pred.argmin(-1)
                row=dict(temperature=temperature,strength=strength,converged=bool(solved.success),iterations=int(solved.nit),
                    gain=float((vy[:,0]-vy[np.arange(len(vy)),choice]).mean()),
                    contrast_mae=float(np.abs(pred-pred[:,:1]-vy+vy[:,:1])[:,1:].mean()))
                history.append(row);score=-row['gain']+.01*row['contrast_mae']
                if score<best_score:best_score=score;best=(coefficient.copy(),row)
        saved.update(coefficient=best[0],ranking_selection=best[1],ranking_history=history,
            training_objective='conditional paired GLS plus soft candidate ranking')
        atomic_torch(args.output/f'fleet-{arm}-{args.seed}.pth',saved)
        print(json.dumps(dict(arm=arm,selected=best[1])),flush=True)


def ensemble_fleet(args):
    """Equal-weight bagging; linear energies collapse to one deployment readout."""
    seeds=[int(seed) for seed in args.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or args.seed in seeds:
        raise ValueError('Ensemble members must be distinct from one another and the output seed.')
    contract=('kind','source','horizon','rollout_steps','defenders','train_count',
        'validation_count','candidate_mode','interaction_residual','conditional_edges',
        'measured_target','conditional_covariance','arm','data_hashes','source_sha256','context_normalization','covariance_floor','covariance_local_weight','projection_repeats',
        'active_defenders','absolute_anchor','mixed_only','validation_criterion','response_basis','monotone_readout')
    for arm in args.arms.split(','):
        paths=[args.output/f'fleet-{arm}-{seed}.pth' for seed in seeds]
        members=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
        if any(m.get('kind')!='rollout-ridge' for m in members):
            raise ValueError('Bagging requires linear rollout readouts.')
        for key in contract:
            if any(m.get(key)!=members[0].get(key) for m in members[1:]):
                raise ValueError(f'Ensemble members disagree on {key}.')
        saved={key:members[0].get(key) for key in contract}
        saved.update(seed=args.seed,coefficient=np.mean([m['coefficient'] for m in members],axis=0),
            trust_margin=float(np.mean([m.get('trust_margin',0.) for m in members])),
            ensemble_seeds=seeds,ensemble_weights=[1./len(seeds)]*len(seeds),
            ensemble_hashes=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])
        atomic_torch(args.output/f'fleet-{arm}-{args.seed}.pth',saved)
        print(json.dumps(dict(arm=arm,ensemble=seeds,seed=args.seed)),flush=True)


def fleet_control_scores(features,saved,options,probe='aligned'):
    if saved.get('score_kind')=='physical_model':return np.asarray(features['model_cost']).copy()
    features=fleet_response_basis(features,options,saved.get('response_basis'))
    name='no_peer' if saved['arm'] in ('no_peer','absolute_no_peer') else 'full'
    value_features=features[name]
    if saved.get('interaction_residual') or probe!='aligned':
        interaction=features[name]-features['no_peer']
        if probe=='zero':interaction=np.zeros_like(interaction)
        elif probe=='permuted':
            # Correct unary candidates, with only the joint residual's
            # control-to-vessel correspondence deliberately mismatched.
            varying=np.flatnonzero(np.ptp(options[1:],axis=0)>0.)
            order=np.arange(options.shape[1]);order[varying]=np.roll(varying,1)
            lookup={tuple(option):i for i,option in enumerate(options)}
            interaction=interaction[[lookup[tuple(option[order])] for option in options]]
        elif probe!='aligned':raise ValueError(f'Unknown interaction probe: {probe}')
        value_features=(np.concatenate((features['no_peer'],interaction),-1) if saved.get('interaction_residual')
            else features['no_peer']+interaction)
    value_features=fleet_context_features(value_features,features.get('root'),saved.get('context_normalization'))
    q=value_features@saved['coefficient'];q[1:]+=saved.get('trust_margin',0.)
    return q


class FleetRolloutValuePolicy(FleetValuePolicy):
    def __init__(self,path,period=None):
        from interaction_evaluation import OriginalPolicy
        self.saved=torch.load(path,map_location='cpu',weights_only=False)
        if self.saved.get('source_sha256') and hashlib.sha256(Path(self.saved['source']).read_bytes()).hexdigest()!=self.saved['source_sha256']:
            raise ValueError('The frozen source policy differs from the fitted value model.')
        self.source=OriginalPolicy(self.saved['source'])
        self.period=self.saved['horizon'] if period is None else period;self.reset()

    @torch.no_grad()
    def choose_action(self,packet,deterministic=True):
        base=self.source(packet['obs'])
        if 60.*(1.-float(packet['central'][-1]))+1e-5<getattr(self,'start_time',0.):
            return base,0.
        if self.saved.get('measured_target') and 'target_motion' not in packet:
            raise ValueError('This model requires the declared shared target motion measurement.')
        if self.remaining==0:
            limit=getattr(self,'decision_limit',0)
            if limit and len(self.decisions)>=limit:return base,0.
            radius=getattr(self,'activation_radius',0.)
            if radius and np.sort(packet['obs'][:,2])[1]>radius:return base,0.
        if self.remaining==0:
            options=fleet_active_options(packet,self.saved.get('active_defenders',0))
            steps=self.saved['rollout_steps'];mode=self.saved.get('candidate_mode','source');repeats=self.saved.get('projection_repeats',1) or 1
            cache=getattr(self,'response_cache',None)
            key=(self.saved.get('source_sha256',self.saved.get('source')),steps,mode,repeats,options.tobytes(),float(packet['central'][-1]),
                *(np.asarray(packet[name]).tobytes() if name in packet else b'' for name in ('motion','obs','target_motion')))
            if cache is None or key not in cache:
                features=fleet_rollout_features(packet,self.source,options,steps,mode,repeats)
                if cache is not None:cache[key]=features
            else:features=cache[key]
            q=fleet_control_scores(features,self.saved,options,getattr(self,'inference_probe','aligned'))
            choice=int(q.argmin());self.coefficients=options[choice].copy()
            decision=dict(choice=choice,gain=float(q[0]-q[choice]),selection='argmin')
            decision['public_state_sha256']=hashlib.sha256(repr(key[5:]).encode('utf-8')).hexdigest()
            if self.saved.get('score_kind')=='physical_model':
                decision['score_definition']='unlearned public-model elapsed time, remaining capture estimate, and exact timeout cap for predicted failure'
            temperature=getattr(self,'decision_temperature',0.)
            if temperature>0.:
                if self.saved.get('candidate_mode')!='feedback':
                    raise ValueError('Soft feedback composition requires source/lead feedback candidates.')
                prior=getattr(self,'reference_prior',.9)
                if not 0.<prior<1.:raise ValueError('Reference prior must be strictly between zero and one.')
                log_prior=np.full(len(q),np.log((1.-prior)/(len(q)-1)));log_prior[0]=np.log(prior)
                logits=log_prior-(q-q[0])/temperature;probability=np.exp(logits-logits.max());probability/=probability.sum()
                self.coefficients=(probability[:,None]*np.where(options<0.,1.,options)).sum(0)
                decision.update(selection='soft-mean',reference_probability=float(probability[0]),
                    gain=float(q[0]-probability@q),score_definition='expected discrete candidate score, not the value of the mean control')
            if getattr(self,'fixed_window',None):
                if self.fixed_window=='mean':self.coefficients=.95+.05*np.where(self.coefficients<0.,1.,self.coefficients)
                decision.update(selection='fixed-'+self.fixed_window+'-window',gain=None,
                    score_definition='constant feedback program; no learned candidate value')
            decision['coefficients']=self.coefficients.tolist()
            self.remaining=self.period;self.decisions.append(decision)
        self.remaining-=1
        return fleet_action(base,self.coefficients,packet['obs'],self.saved.get('candidate_mode','source')),0.


def initialize_fleet_evaluation(output,seed,period,start_time=0.,arms='difference,absolute,no_peer,arboids_cbf',temperature=0.,reference_prior=.9,decision_limit=0,activation_radius=0.):
    global FLEET_POLICIES,FLEET_REFERENCE_PRIOR
    initialize_fleet()
    if (period is not None and period<=0) or min(start_time,temperature,decision_limit,activation_radius)<0:
        raise ValueError('Invalid fleet control period, temperature, intervention limit or activation radius.')
    FLEET_REFERENCE_PRIOR=reference_prior
    saved=torch.load(Path(output)/f'fleet-difference-{seed}.pth',map_location='cpu',weights_only=False)
    policy=FleetRolloutValuePolicy if saved.get('kind')=='rollout-ridge' else FleetValuePolicy
    FLEET_POLICIES={}
    for arm in arms.split(','):
        if arm not in ('difference','absolute','no_peer','absolute_no_peer','zero_interaction','permuted_interaction','model_only','arboids_cbf','constant_bias','lead_boids_cbf','fixed_feedback','fixed_lead_window','fixed_mean_window'):
            raise ValueError(f'Unknown fleet evaluation arm: {arm}')
        if arm in ('arboids_cbf','constant_bias','lead_boids_cbf','fixed_feedback'):
            FLEET_POLICIES[arm]=FLEET_SOURCE;continue
        source_arm=arm if arm in ('difference','absolute','no_peer','absolute_no_peer') else 'difference'
        p=policy(Path(output)/f'fleet-{source_arm}-{seed}.pth',period);p.start_time=start_time
        p.decision_temperature=temperature;p.reference_prior=reference_prior
        p.decision_limit=decision_limit;p.activation_radius=activation_radius
        if (decision_limit or activation_radius) and not isinstance(p,FleetRolloutValuePolicy):
            raise ValueError('Finite intervention budgets require a response value policy.')
        if arm in ('fixed_lead_window','fixed_mean_window'):
            if p.saved.get('kind')!='rollout-ridge' or p.saved.get('candidate_mode')!='feedback':
                raise ValueError('Fixed feedback windows require the same feedback candidate contract.')
            p.saved=dict(p.saved,coefficient=np.zeros_like(p.saved['coefficient']),trust_margin=-1.)
            p.fixed_window='lead' if arm=='fixed_lead_window' else 'mean';p.decision_temperature=0.
        if arm=='model_only':
            p.saved=dict(p.saved,score_kind='physical_model')
        p.inference_probe={'zero_interaction':'zero','permuted_interaction':'permuted'}.get(arm,'aligned')
        if p.inference_probe!='aligned' and p.saved.get('kind')!='rollout-ridge':
            raise ValueError('Surgical probes require a decomposable response value model.')
        FLEET_POLICIES[arm]=p


def fleet_episode(task):
    scene,n=task;row=dict(scene=scene,defenders=n)
    response_cache={}
    contracts={p.saved.get('measured_target',False) for p in FLEET_POLICIES.values() if hasattr(p,'saved')}
    if len(contracts)>1:raise ValueError('All learned ablations must have identical public observations.')
    measured_target=next(iter(contracts),False)
    for arm,policy in FLEET_POLICIES.items():
        learned=arm not in ('arboids_cbf','constant_bias','lead_boids_cbf','fixed_feedback')
        seed_random(scene);env=TADEnv(n,protocol='paper-parameters-v1');obs,_=env.reset(2.25,noisy_agility=False)
        np.random.seed(scene+1000000);packet=fleet_public_packet(env,obs,measured_target);done=0
        if learned:policy.reset();policy.response_cache=response_cache
        while not done:
            count=len(policy.decisions) if learned else 0
            action=policy.choose_action(packet)[0]
            if arm=='constant_bias':action[:,2]=np.clip(action[:,2]-.05,0.,1.)
            elif arm=='lead_boids_cbf':action=fleet_action(action,np.zeros(n),packet['obs'],'feedback')
            elif arm=='fixed_feedback':action=fleet_action(action,np.full(n,.5+.5*FLEET_REFERENCE_PRIOR),packet['obs'],'feedback')
            planned=learned and len(policy.decisions)>count
            if planned:reference=filtered(packet,FLEET_SOURCE(packet['obs']),env.boids_actions)
            packet,_,done,thrust,_,_=execute(env,packet,action)
            if measured_target:packet=fleet_public_packet(env,packet['obs'],True)
            if planned:policy.decisions[-1].update(time=float(env.Current_T-.2),executed_control_change=float(np.linalg.norm(thrust-reference)),
                executed_thrust=np.asarray(thrust).tolist(),source_thrust=np.asarray(reference).tolist())
        row[arm]=outcome_row(env,done)
        if learned:row[arm]['decisions']=policy.decisions
    return row


def fleet_static_episode(task):
    scene,n=task;row=dict(scene=scene,defenders=n)
    for arm in ('arboids_cbf','boids_cbf','lead_boids_cbf','constant_bias'):
        seed_random(scene);env=TADEnv(n,protocol='paper-parameters-v1');obs,_=env.reset(2.25,noisy_agility=False)
        np.random.seed(scene+1000000);packet=public_packet(env,obs);done=0
        while not done:
            action=FLEET_SOURCE(packet['obs'])
            if arm=='boids_cbf':action[:,2]=0.
            elif arm=='lead_boids_cbf':action=fleet_action(action,np.zeros(n),packet['obs'],'feedback')
            elif arm=='constant_bias':action[:,2]=np.clip(action[:,2]-.05,0.,1.)
            packet,_,done,_,_,_=execute(env,packet,action)
        row[arm]=outcome_row(env,done)
    return row


def fleet_result_suffix(period,start_time,arms,temperature=0.,reference_prior=.9,decision_limit=0,activation_radius=0.):
    suffix=f'-p{period}'
    if start_time:suffix+=f'-w{start_time:g}'
    if temperature:suffix+=f'-t{temperature:g}-r{reference_prior:g}'
    if decision_limit:suffix+=f'-d{decision_limit}'
    if activation_radius:suffix+=f'-near{activation_radius:g}'
    if arms!='difference,absolute,no_peer,arboids_cbf':
        suffix+=('-'+arms.replace(',','-') if len(arms)<=60 else '-arms-'+hashlib.sha256(arms.encode()).hexdigest()[:12])
    return suffix


def summarize_fleet(args):
    seeds=[int(seed) for seed in args.seeds.split(',')];arms=args.evaluation_arms.split(',')
    suffix=fleet_result_suffix(args.decision_period,args.decision_start,args.evaluation_arms,args.decision_temperature,args.reference_prior,args.decision_limit,args.activation_radius)
    paths=[args.output/f'fleet-episodes-{seed}-{args.offset}-{args.count}{suffix}.json' for seed in seeds]
    results=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
    if any(result.get('complete') is False for result in results):
        raise ValueError('Wait for every frozen-model evaluation to complete before summarizing.')
    scenes=[[row['scene'] for row in r['rows']] for r in results]
    if any(scene!=scenes[0] for scene in scenes[1:]) or len(set(scenes[0]))!=args.count:
        raise ValueError('Crossed comparisons require the same distinct physical scenarios for every fit.')
    rng=np.random.default_rng(804197);draws=20000
    seed_ix=rng.integers(len(seeds),size=(draws,len(seeds)))
    scene_ix=rng.integers(args.count,size=(draws,args.count))
    times={arm:np.array([[r[arm]['capture_time'] for r in result['rows']] for result in results]) for arm in arms}
    summary={};contrasts={}
    for arm in arms:
        captures=np.array([[r[arm]['capture'] for r in result['rows']] for result in results])
        collisions=np.array([[r[arm]['collision'] for r in result['rows']] for result in results])
        summary[arm]=dict(time=float(times[arm].mean()),per_fit_time=times[arm].mean(1).tolist(),
            capture=int(captures.sum()),capture_per_fit=captures.sum(1).tolist(),collision=int(collisions.sum()))
        if arm=='difference':continue
        delta=times['difference']-times[arm]
        sampled=delta[seed_ix[:,:,None],scene_ix[:,None,:]].mean((1,2))
        contrasts[arm]=dict(difference=float(delta.mean()),per_fit_difference=delta.mean(1).tolist(),
            ci95=np.quantile(sampled,[.025,.975]).tolist(),
            ci975=np.quantile(sampled,[.0125,.9875]).tolist(),
            ci99=np.quantile(sampled,[.005,.995]).tolist(),
            relative_reduction=float(-delta.mean()/times[arm].mean()))
    result=dict(seeds=seeds,physical_scenarios=args.count,scenario_min=min(scenes[0]),scenario_max=max(scenes[0]),
        episodes_per_arm=len(seeds)*args.count,summary=summary,contrasts=contrasts,
        bootstrap='20000 crossed resamples of fitted models and physical scenarios, with all comparisons paired',
        metric='capture time; every non-capture is capped at 60 seconds',
        co_primary=['no_peer','absolute'],co_primary_both_negative_975=all(contrasts[a]['ci975'][1]<0 for a in ('no_peer','absolute')),
        co_primary_both_negative_99=all(contrasts[a]['ci99'][1]<0 for a in ('no_peer','absolute')),
        evidence=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),model_hashes=r['model_hashes']) for p,r in zip(paths,results)])
    atomic_json(args.output/f'fleet-summary-{args.seeds.replace(",","-")}-{args.offset}-{args.count}{suffix}.json',result)
    print(json.dumps({key:result[key] for key in ('summary','contrasts','co_primary_both_negative_975')}),flush=True)


def run_fleet(args):
    started=time.monotonic();rows=[]
    if args.mode=='fleet-fit':return fit_fleet(args)
    if args.mode=='fleet-ridge':return fit_fleet_ridge(args)
    if args.mode=='fleet-ensemble':return ensemble_fleet(args)
    if args.mode=='fleet-rank':return rank_fleet(args)
    if args.mode=='fleet-summarize':return summarize_fleet(args)
    if args.mode=='fleet-static':
        tasks=[(3270000000+args.offset+i,args.defenders) for i in range(args.count)]
        initializer=initialize_fleet;initargs=();function=fleet_static_episode
    elif args.mode=='fleet-project':
        tasks=[(args.split,i,args.output,args.projection_steps,args.measured_target,args.projection_repeats) for i in range(args.offset,args.offset+args.count)]
        initializer=initialize_fleet;initargs=();function=project_fleet
    elif args.mode=='fleet-collect':
        tasks=[(i,args.split,args.output,args.defenders,args.intervention_steps,args.repeats,args.resume,args.candidate_mode,args.active_defenders) for i in range(args.offset,args.offset+args.count)]
        initializer=initialize_fleet;initargs=();function=collect_fleet
    else:
        tasks=[(3270000000+args.offset+i,args.defenders) for i in range(args.count)]
        initializer=initialize_fleet_evaluation;initargs=(str(args.output),args.seed,args.decision_period,args.decision_start,args.evaluation_arms,args.decision_temperature,args.reference_prior,args.decision_limit,args.activation_radius);function=fleet_episode
    progress_path=None;metadata=None
    if args.mode=='fleet-episodes':
        suffix=fleet_result_suffix(args.decision_period,args.decision_start,args.evaluation_arms,args.decision_temperature,args.reference_prior,args.decision_limit,args.activation_radius)
        progress_path=args.output/f'fleet-episodes-{args.seed}-{args.offset}-{args.count}{suffix}.json'
        learned_arms=set(arm if arm in ('difference','absolute','no_peer','absolute_no_peer') else 'difference'
            for arm in args.evaluation_arms.split(',') if arm not in ('arboids_cbf','constant_bias','lead_boids_cbf','fixed_feedback'))
        hashes={arm:hashlib.sha256((args.output/f'fleet-{arm}-{args.seed}.pth').read_bytes()).hexdigest() for arm in sorted(learned_arms)}
        metadata=dict(model_hashes=hashes,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            decision_start=args.decision_start,decision_period=args.decision_period,defenders=args.defenders,
            protocol='paper-parameters-v1',agility=2.25,expected_count=args.count,offset=args.offset,arms=args.evaluation_arms.split(','),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),nominal_baseline_version='elapsed-plus-continuation-v2')
        if args.decision_temperature:metadata.update(decision_temperature=args.decision_temperature,reference_prior=args.reference_prior)
        if args.decision_limit or args.activation_radius:
            metadata.update(decision_limit=args.decision_limit,activation_radius=args.activation_radius)
        if args.resume and progress_path.exists():
            previous=json.loads(progress_path.read_text(encoding='utf-8'))
            if any(previous.get(key)!=value for key,value in metadata.items()):
                raise ValueError('Cannot resume with changed models, observations, scenarios or evaluation settings.')
            rows=previous['rows']
            if [row['scene'] for row in rows]!=[task[0] for task in tasks[:len(rows)]]:
                raise ValueError('Saved paired scenarios are not the expected prefix.')
            tasks=tasks[len(rows):]
        atomic_json(progress_path,dict(**metadata,rows=rows,complete=False))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initializer,initargs=initargs) as pool:
        for row in pool.map(function,tasks,chunksize=1):
            rows.append(row)
            if len(rows)%4==0:print(json.dumps(dict(mode=args.mode,split=args.split,completed=len(rows),elapsed=round(time.monotonic()-started,1))),flush=True)
            if progress_path is not None and len(rows)%32==0:
                atomic_json(progress_path,dict(**metadata,rows=rows,complete=False))
    if args.mode=='fleet-static':
        summary={arm:dict(time=float(np.mean([r[arm]['capture_time'] for r in rows])),capture=int(sum(r[arm]['capture'] for r in rows)),
            collision=int(sum(r[arm]['collision'] for r in rows))) for arm in ('arboids_cbf','boids_cbf','lead_boids_cbf','constant_bias')}
        atomic_json(args.output/f'fleet-static-{args.defenders}-{args.offset}-{args.count}.json',dict(rows=rows,summary=summary));print(json.dumps(summary),flush=True)
    elif args.mode=='fleet-episodes':
        summary={arm:dict(time=float(np.mean([r[arm]['capture_time'] for r in rows])),capture=int(sum(r[arm]['capture'] for r in rows)),
            collision=int(sum(r[arm]['collision'] for r in rows))) for arm in args.evaluation_arms.split(',')}
        if any(hashlib.sha256((args.output/f'fleet-{arm}-{args.seed}.pth').read_bytes()).hexdigest()!=digest
               for arm,digest in metadata['model_hashes'].items()):
            raise ValueError('A model changed during paired evaluation; retain partial records for inspection.')
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=metadata['source_sha256']:
            raise ValueError('The source policy changed during paired evaluation.')
        atomic_json(progress_path,dict(**metadata,rows=rows,summary=summary,complete=True))
        print(json.dumps(summary),flush=True)
    else:print(json.dumps(dict(completed=len(rows),mean_oracle_gain=float(np.mean([r.get('oracle_gain',0.) for r in rows])))),flush=True)


def program_options(n, radius,packet=None,active=0):
    """Persistent feedback-gate interventions, including the reference."""
    selected=np.arange(n) if not active else np.argsort(packet['obs'][:,2],kind='stable')[:active]
    bits=((np.arange(2**len(selected))[:,None] >> np.arange(len(selected))) & 1).astype(np.float32)
    options=np.zeros((1+len(bits),n),np.float32);options[1:,selected]=radius*(2.*bits-1.)
    return options


def program_trigger(packet,trigger,active=0):
    remaining=packet['time_remaining'] if 'time_remaining' in packet else packet['central'][-1]
    return (60.*(1.-float(remaining))>=2.-1e-5 and
            float(np.sort(packet['obs'][:,2])[max(0,active-1)])<=trigger)


def program_action(base, correction, bias):
    action=base.copy();action[...,2]=np.clip(action[...,2]+bias+correction,0.,1.)
    return action


def program_features(packet,base,options,bias):
    coefficients=np.clip(base[None,:,2]+bias+options,0.,1.)
    features=fleet_features(packet,base,coefficients)
    mismatched=copy.deepcopy(packet);mismatched['obs'][:,12:14]=np.roll(packet['obs'][:,12:14],1,axis=0)
    other=fleet_features(mismatched,np.roll(base,1,axis=0),np.roll(coefficients,1,axis=1))
    edge=features['edge'].copy();edge[...,14:22]=other['edge'][...,14:22]
    own_path=features['node'][...,20:28].reshape(len(options),len(base),4,2)
    peer_path=other['node'][...,20:28].reshape(len(options),len(base),4,2)
    edge[...,22:30]=((peer_path[:,None]-own_path[:,:,None])*20./15.).reshape(len(options),len(base),len(base),8)
    features['permuted_edge']=edge
    # The persistent displacement is observable even if the initial gate clips.
    features['program']=np.asarray(options,np.float32)
    if 'target_motion' not in packet:raise ValueError('Persistent programs require public target motion.')
    angle=packet['target_motion'][2]-packet['motion'][:,2]
    context=np.column_stack((np.cos(angle),np.sin(angle),np.full(len(base),packet['target_motion'][5])))
    features['target_context']=np.broadcast_to(context,(len(options),len(base),3)).astype(np.float32).copy()
    return features


class ProgramConditionalValue(FleetConditionalValue):
    def __init__(self,width=64,no_peer=False,quadratic=True):
        super().__init__(width,no_peer);self.quadratic=quadratic
        self.node[0]=nn.Linear(38 if quadratic else 37,width);self.edge[0]=nn.Linear(39 if quadratic else 38,width)
        self.state[0]=nn.Linear(16,width)

    def forward(self,features,probe='aligned'):
        x=features['node'];n=x.shape[-2];z=features['program']
        if self.quadratic:z=z/z.abs().amax(dim=(-2,-1),keepdim=True).clamp_min(1e-8)
        peer=z[...,None,:,None].expand(*z.shape[:-1],n,n,1);own=peer.transpose(-3,-2)
        e=features['no_peer_edge'] if self.no_peer or probe=='state-only' else features['edge']
        if self.no_peer or probe=='state-only':peer=torch.zeros_like(peer)
        elif probe=='permuted':
            e=features['permuted_edge'];peer=peer.roll(1,-2)
        # The released evader traverses obstacles in a fixed, public order.
        # Preserve this order in every arm, including the absolute baseline;
        # otherwise a permutation-invariant value cannot represent its return.
        order=(torch.arange(n,device=x.device,dtype=x.dtype)+1.)/n
        order=order.expand(*z.shape)[...,None]
        own_order=order.unsqueeze(-2).expand(*z.shape,n,1)
        peer_order=order.unsqueeze(-3).expand(*z.shape,n,1)
        clearance=e[...,22:30].reshape(*e.shape[:-1],4,2).norm(dim=-1)
        approach=x[...,20:28].reshape(*x.shape[:-1],4,2).norm(dim=-1)
        edge=torch.cat((e,own,peer,*((own*peer,) if self.quadratic else ()),own_order,peer_order,clearance),-1)
        attention=(1.-torch.eye(n,device=x.device))/(1.+e[...,:2].square().sum(-1))
        message=(self.edge(edge)*attention[...,None]).sum(-2)/attention.sum(-1,keepdim=True).clamp_min(1e-6)
        context=features['target_context']
        h=self.node(torch.cat((x,z[...,None],*((z.square()[...,None],) if self.quadratic else ()),order,approach,context),-1))
        q=(self.effect(torch.cat((h,message),-1)).squeeze(-1)*features['weight']).sum(-1)
        b=(self.state(torch.cat((x[...,:12],order,context),-1)).squeeze(-1)*features['weight']).sum(-1)
        return b,q


def program_branch_returns(root,options,bias,futures):
    n=root.environment.defender_num;repeats=len(futures);t=float(root.environment.Current_T)
    if not repeats:return np.empty((0,len(options)),np.float32),np.empty((0,len(options)),int)
    branches=[];packets=[];states=[]
    for future in futures:
        for option in options:
            branch=root.restore(future_seed=future);use_source_current_motion(branch)
            branches.append(branch);packets.append(public_packet(branch));states.append(np.random.get_state())
    live=list(range(len(branches)));outcomes=np.zeros(len(branches),int)
    for step in range(300):
        if not live:break
        bases=FLEET_SOURCE(np.concatenate([packets[i]['obs'] for i in live])).reshape(len(live),n,3)
        actions=[program_action(bases[j],options[i%len(options)],bias) for j,i in enumerate(live)]
        applied=batch_filtered([packets[i] for i in live],actions,[branches[i].boids_actions for i in live])
        following=[]
        for j,i in enumerate(live):
            np.random.set_state(states[i]);obs,_,done,_=branches[i].step(actions[j],'AdaRes',defender_thrust=applied[j])
            states[i]=np.random.get_state();packets[i]=public_packet(branches[i],obs)
            if done:outcomes[i]=int(done)
            else:following.append(i)
        live=following
    if live:raise RuntimeError('Unfinished program intervention.')
    costs=np.array([b.Current_T-t if outcomes[i]==3 else 60.-t for i,b in enumerate(branches)],np.float32).reshape(repeats,-1)
    return costs,outcomes.reshape(repeats,-1)


def collect_program(task):
    index,split,out,n,repeats,bias,radius,trigger,agility,resume,*extra=task
    active=extra[0] if extra else 0
    path=Path(out)/split/f'program-{index}.pth';path.parent.mkdir(parents=True,exist_ok=True)
    config=dict(defenders=n,repeats=repeats,bias=bias,radius=radius,trigger=trigger,agility=agility)
    if active:config['active_defenders']=active
    if resume and path.exists():
        old=torch.load(path,weights_only=False)
        if old['config']!=config:raise ValueError('Program collection protocol changed.')
        return dict(index=index,resumed=True)
    split_id={'train':0,'validation':1,'test':2}[split]
    for attempt in range(32):
        scene=3370000000+split_id*1000000+n*10000+index+attempt*10000000
        seed_random(scene);env=TADEnv(n,protocol='paper-parameters-v1');obs,_=env.reset(agility,noisy_agility=False)
        np.random.seed(scene+4000000);packet=public_packet(env,obs);use_source_current_motion(env);done=0
        while not done and not program_trigger(packet,trigger,active):
            action=program_action(FLEET_SOURCE(packet['obs']),np.zeros(n),bias)
            packet,_,done,_,_,_=execute(env,packet,action)
        if not done:break
    else:raise RuntimeError('No eligible program root after 32 independent reference episodes.')
    root=compact_snapshot(env)
    for boat in [root.environment.attacker,*root.environment.defender_list]:boat.__dict__.pop('compute_motion',None)
    packet=fleet_public_packet(env,measured_target=True)
    base=FLEET_SOURCE(packet['obs']);options=program_options(n,radius,packet,active);features=program_features(packet,base,options,bias)
    futures=[3470000000+split_id*1000000+n*10000+index*64+r for r in range(repeats)]
    if repeats>64:raise ValueError('Future-seed stride supports at most 64 repetitions.')
    costs,outcomes=program_branch_returns(root,options,bias,futures);t=float(root.environment.Current_T)
    atomic_torch(path,dict(config=config,scene=scene,index=index,split=split,packet=packet,base=base,
        options=options,features=features,costs=costs,futures=futures,root_time=t,terminations=outcomes.reshape(repeats,-1),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        objective='complete capped capture time of the deployed persistent feedback program',snapshot=root))
    return dict(index=index,reference=float(costs[:,0].mean()),oracle_gain=float(costs[:,0].mean()-costs.mean(0).min()))


def program_record(path):
    value=torch.load(path,weights_only=False)
    if 'base_path' not in value:return value
    source=Path(value['base_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=value['base_sha256']:raise ValueError('Refined program source changed.')
    row=program_record(source);row['config']=value['config']
    row['costs']=np.concatenate((row['costs'],value['additional_costs']))
    row['terminations']=np.concatenate((row['terminations'],value['additional_terminations']))
    row['futures']+=value['additional_futures']
    return row


def refine_program(task):
    source_dir,original=task;index,split,out,n,repeats,bias,radius,trigger,agility,resume,*extra=original
    source=Path(source_dir)/split/f'program-{index}.pth'
    if not source.exists():return collect_program(original)
    path=Path(out)/split/f'program-{index}.pth';path.parent.mkdir(parents=True,exist_ok=True)
    old=program_record(source);config=dict(old['config'],repeats=repeats)
    if repeats<old['config']['repeats'] or repeats>64:raise ValueError('Refinement must retain existing futures, up to the 64-future stride.')
    if (config['defenders'],config['bias'],config['radius'],config['trigger'],config['agility'],config.get('active_defenders',0))!=(n,bias,radius,trigger,agility,extra[0] if extra else 0):
        raise ValueError('Refinement cannot change the intervention or task.')
    if resume and path.exists():
        row=program_record(path)
        if row['config']!=config:raise ValueError('Different refined program protocol.')
        return dict(index=index,resumed=True)
    futures=[old['futures'][0]+i for i in range(len(old['futures']),repeats)]
    costs,outcomes=program_branch_returns(old['snapshot'],old['options'],bias,futures)
    atomic_torch(path,dict(base_path=str(source.resolve()),base_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        config=config,additional_costs=costs,additional_terminations=outcomes,additional_futures=futures))
    return dict(index=index,additional_futures=len(futures))


def load_program(out,split,count,device):
    paths=[Path(out)/split/f'program-{i}.pth' for i in range(count)];rows=[program_record(p) for p in paths]
    if any(row['config']!=rows[0]['config'] for row in rows):raise ValueError('Mixed program protocols.')
    for row in rows:
        # Older raw banks retain the exact root snapshot. Reconstruct only
        # its public measurements, never water-relative velocity or agility.
        if 'target_context' not in row['features'] or 'permuted_edge' not in row['features']:
            row['packet']=fleet_public_packet(row['snapshot'].environment,measured_target=True)
            row['features']=program_features(row['packet'],row['base'],row['options'],row['config']['bias'])
    features={k:torch.as_tensor(np.stack([r['features'][k] for r in rows]),device=device) for k in rows[0]['features']}
    return dict(features=features,costs=torch.as_tensor(np.stack([r['costs'].mean(0) for r in rows]),device=device),
        rows=rows,sha256=hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()).hexdigest())


def fit_program(args):
    torch.set_num_threads(1);torch.set_grad_enabled(True)
    train=load_program(args.output,'train',args.train_count,args.device);valid=load_program(args.output,'validation',args.validation_count,args.device)
    opts=train['rows'][0]['options'];edges=(np.count_nonzero(opts[:,None]!=opts[None],axis=-1)==1)
    edges[0]=False;edges[:,0]=False;mask=torch.as_tensor(edges,device=args.device)
    for arm in args.arms.split(','):
        if arm not in ('difference','absolute','no_peer','absolute_no_peer'):raise ValueError('Unknown program arm.')
        torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
        model=ProgramConditionalValue(args.width,'no_peer' in arm).to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01)
        best_score=float('inf');history=[];best=None
        for step in range(1,args.updates+1):
            ix=torch.as_tensor(rng.integers(args.train_count,size=min(64,args.train_count)),device=args.device)
            b,q=model({k:v[ix] for k,v in train['features'].items()});cost=train['costs'][ix]/10.;q=q/10.;b=b/10.
            delta=q-q[:,:1];total=b[:,:1]+delta
            if arm in ('absolute','absolute_no_peer'):loss=F.mse_loss(total,cost)
            else:
                error=(q[:,:,None]-q[:,None,:])-(cost[:,:,None]-cost[:,None,:])
                loss=error[:,mask].square().mean()+F.mse_loss(delta,cost-cost[:,:1])+.25*F.mse_loss(b[:,:1],cost[:,:1])
                if args.program_policy_weight:
                    # Complete conditional returns directly train the deployed
                    # energy policy, while difference regression calibrates Q.
                    probability=torch.softmax(-q/.3,dim=-1)
                    loss+=args.program_policy_weight*(probability*(cost-cost[:,:1])).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if step%100==0 or step==args.updates:
                model.eval();metrics=fleet_metrics(model,valid);model.train();history.append(dict(step=step,**metrics))
                score=metrics['regret']+.05*metrics['contrast_mae']
                if score<best_score:
                    best_score=score;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_step=step
                if step%500==0 or step==args.updates:print(json.dumps(dict(arm=arm,step=step,**metrics)),flush=True)
        atomic_torch(args.output/f'program-{arm}-{args.seed}.pth',dict(model=best,seed=args.seed,arm=arm,width=args.width,
            config=train['rows'][0]['config'],train_count=args.train_count,validation_count=args.validation_count,
            selection_step=best_step,history=history,source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            train_sha256=train['sha256'],validation_sha256=valid['sha256'],updates=args.updates,learning_rate=args.learning_rate,
            program_policy_weight=args.program_policy_weight,
            best_fixed_choice=int(train['costs'].mean(0).argmin()),
            measured_target=True,feature_version='public-target-order-clearance-quadratic-v3',
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            objective_version='trigger-matched-program-conditional-value-v1'))


def program_kernel_description(packet,features,options,config):
    """Own candidate responses plus peer kinematics; pairs add both candidates."""
    node=np.asarray(features['node']);n=node.shape[1];active=np.flatnonzero(np.any(options!=0.,axis=0))
    active=active[np.argsort(packet['obs'][active,2],kind='stable')]
    public=np.concatenate((node[0,:,:12],features['target_context'][0],(1.+np.arange(n))[:,None]/n),axis=-1)
    descriptions=[];candidate_context=[]
    for i in active:
        plus=int(np.flatnonzero(options[:,i]>0)[0]);minus=int(np.flatnonzero(options[:,i]<0)[0])
        peers=np.argsort(np.linalg.norm(packet['motion'][:,:2]-packet['motion'][i,:2],axis=-1),kind='stable')
        peers=peers[peers!=i]
        descriptions.append(np.concatenate((public[i],node[0,i,12:28],node[plus,i,17:28]-node[0,i,17:28],
            node[minus,i,17:28]-node[0,i,17:28],public[peers].ravel())))
        candidate_context.append(np.concatenate((node[0,i,12:28],node[plus,i,17:28]-node[0,i,17:28],node[minus,i,17:28]-node[0,i,17:28])))
    own=np.stack(descriptions);pair_indices=[(i,j) for i in range(len(active)) for j in range(i+1,len(active))]
    context=np.stack([np.concatenate((own[i],*(candidate_context[j] for j in range(len(active)) if j!=i))) for i in range(len(active))])
    pair=np.stack([np.concatenate((own[i],own[j])) for i,j in pair_indices])
    z=np.asarray(options[:,active]/config['radius'],np.float64)
    mixed=np.stack([z[:,i]*z[:,j] for i,j in pair_indices],axis=-1)
    return dict(node=own,node_context=context,pair=pair,state=public[np.argsort(packet['obs'][:,2],kind='stable')].ravel(),z=z,mixed=mixed)


def program_kernel_bank(data,device):
    rows=data['rows'];descriptions=[program_kernel_description(r['packet'],r['features'],r['options'],r['config']) for r in rows]
    return {k:torch.as_tensor(np.stack([d[k] for d in descriptions]),dtype=torch.float64,device=device) for k in descriptions[0]}


def program_kernel_normalization(bank,metric='balanced'):
    keys=('node','pair','state')+(('node_context',) if 'node_context' in bank else ())
    result={k:dict(mean=bank[k].flatten(0,-2).mean(0),scale=bank[k].flatten(0,-2).std(0).clamp_min(.1 if metric=='balanced' else .01)) for k in keys}
    if metric=='control':
        # Projected displacements are in units of 20 m and can be much smaller
        # than 0.1 despite materially changing capture. Retain their scale.
        d=bank['node'].shape[-1];weight=torch.ones(d,device=bank['node'].device,dtype=bank['node'].dtype)
        weight[32:54]=2.;weight[54:]=.5
        result['node']['scale']/=weight;result['pair']['scale']/=weight.repeat(2)
        if 'node_context' in result:
            peer=torch.ones(38,device=weight.device,dtype=weight.dtype);peer[16:]=2.
            result['node_context']['scale']/=torch.cat((weight,peer.repeat(bank['node'].shape[1]-1)))
    return result


def program_kernel_components(left,right,normalization,length,node_key='node'):
    def rbf(key):
        a=(left[key]-normalization[key]['mean'])/normalization[key]['scale']
        b=(right[key]-normalization[key]['mean'])/normalization[key]['scale']
        d=torch.cdist(a.reshape(-1,a.shape[-1]),b.reshape(-1,b.shape[-1])).square()/a.shape[-1]
        return torch.exp(-.5*d/length**2)
    nr,ns=left['z'].shape[0],right['z'].shape[0];m=left['z'].shape[-1];p=left['mixed'].shape[-1]
    node=rbf(node_key).reshape(nr,m,ns,m);pair=rbf('pair').reshape(nr,p,ns,p)
    def project(kernel,key,square=False):
        a,b=left[key],right[key]
        if square:a,b=a.square(),b.square()
        return torch.einsum('iap,ipjq,jbq->iajb',a,kernel,b)
    # Independent latent unary and curvature functions, shared across receivers.
    unary=project(node,'z')+.5*project(node,'z',True)
    return dict(unary=unary,pair=project(pair,'mixed'),state=rbf('state')[:,None,:,None])


def program_kernel_covariance(rows,conditional):
    raw=np.stack([r['costs'][:,1:]-r['costs'][:,:1] if conditional else r['costs'] for r in rows]).astype(np.float64)
    residual=raw-raw.mean(1,keepdims=True);r=raw.shape[1]
    pooled=np.einsum('nri,nrj->ij',residual,residual)/(len(rows)*max(1,r-1)*r)
    # Pooled rather than root-local precision avoids weighting a root's target
    # with a variance estimate from the same few unusually good/bad outcomes.
    if not conditional:pooled=np.diag(np.diag(pooled))
    else:pooled=.8*pooled+.2*np.diag(np.diag(pooled))
    return pooled+np.eye(pooled.shape[0])*.25


def fit_program_kernel(args):
    torch.set_num_threads(1);torch.set_grad_enabled(False);device=args.device
    train=load_program(args.output,'train',args.train_count,'cpu');valid=load_program(args.output,'validation',args.validation_count,'cpu')
    tx=program_kernel_bank(train,device);vx=program_kernel_bank(valid,device);normalization=program_kernel_normalization(tx,args.kernel_metric)
    ty=train['costs'].to(device,dtype=torch.float64);vy=valid['costs'].to(device,dtype=torch.float64)
    n,k=ty.shape;nv=vy.shape[0];best={};history={a:[] for a in args.arms.split(',')}
    for length in (.35,.7,1.4):
        kt=program_kernel_components(tx,tx,normalization,length);kv=program_kernel_components(vx,tx,normalization,length)
        ct=program_kernel_components(tx,tx,normalization,length,'node_context') if args.candidate_context else kt
        cv=program_kernel_components(vx,tx,normalization,length,'node_context') if args.candidate_context else kv
        for arm in args.arms.split(','):
            if arm not in ('difference','absolute','no_peer','absolute_no_peer'):raise ValueError('Unknown kernel arm.')
            paired=arm in ('difference','no_peer');conditional=paired and not args.absolute_anchor;nop='no_peer' in arm
            training,validation=(kt,kv) if nop else (ct,cv)
            indices=slice(1,None) if conditional else slice(None);size=k-1 if conditional else k
            noise=torch.as_tensor(program_kernel_covariance(train['rows'],conditional),device=device)
            target=ty[:,1:]-ty[:,:1] if conditional else ty-ty.mean()
            for auxiliary in ((.1,1.,10.,100.) if paired and args.absolute_anchor else (0.,)):
                if auxiliary:
                    delta_noise=torch.as_tensor(program_kernel_covariance(train['rows'],True),device=device)
                    difference=torch.cat((-torch.ones(k-1,1,device=device,dtype=noise.dtype),torch.eye(k-1,device=device,dtype=noise.dtype)),-1)
                    precision=torch.linalg.inv(noise)+auxiliary*difference.T@torch.linalg.solve(delta_noise,difference)
                    whiten=torch.linalg.cholesky(precision).T
                else:whiten=torch.linalg.inv(torch.linalg.cholesky(noise))
                y=(target@whiten.T).reshape(-1)
                for interaction in ((0.,) if nop else (.25,1.,4.)):
                    kernel=training['unary']+interaction*training['pair']+(0. if conditional else 25.*training['state'])
                    cross=validation['unary']+interaction*validation['pair']+(0. if conditional else 25.*validation['state'])
                    matrix=kernel[:,indices,:,indices]
                    matrix=(whiten@matrix.permute(0,2,1,3)@whiten.T).permute(0,2,1,3).reshape(n*size,n*size)
                    eigen,vectors=torch.linalg.eigh(.5*(matrix+matrix.T));eigen=eigen.clamp_min(0.)
                    projected=vectors.T@y;cross=cross[:,:,:,indices].reshape(nv*k,n*size)
                    for ridge in (.001,.01,.1,1.,10.,100.):
                        alpha=((vectors@(projected/(eigen+ridge))).reshape(n,size)@whiten).reshape(-1)
                        q=(cross@alpha).reshape(nv,k);delta=q-q[:,:1];truth=vy-vy[:,:1];choice=q.argmin(-1)
                        chosen=vy.gather(-1,choice[:,None]).squeeze(-1)
                        metrics=dict(length=length,interaction=interaction,ridge=ridge,auxiliary=auxiliary,
                            contrast_mae=float((delta[:,1:]-truth[:,1:]).abs().mean()),gain=float((vy[:,0]-chosen).mean()),
                            regret=float((chosen-vy.min(-1).values).mean()),changes=float((choice!=0).double().mean()))
                        history[arm].append(metrics);score=metrics['regret']+.05*metrics['contrast_mae']
                        if arm not in best or score<best[arm]['score']:
                            best[arm]=dict(score=score,metrics=metrics,alpha=alpha.cpu().numpy().reshape(n,size))
            print(json.dumps(dict(arm=arm,length=length,best=best[arm]['metrics'])),flush=True)
    for arm,value in best.items():
        atomic_torch(args.output/f'program-{arm}-{args.seed}.pth',dict(kind='program-kernel',seed=args.seed,arm=arm,
            config=train['rows'][0]['config'],bank={key:v.cpu().numpy() for key,v in tx.items()},
            normalization={key:{name:v.cpu().numpy() for name,v in norm.items()} for key,norm in normalization.items()},
            alpha=value['alpha'],metrics=value['metrics'],history=history[arm],source=str(SOURCE),
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),train_sha256=train['sha256'],validation_sha256=valid['sha256'],
            train_count=args.train_count,validation_count=args.validation_count,best_fixed_choice=int(ty.mean(0).argmin()),
            measured_target=True,feature_version='public-pair-anova-kernel-v1',kernel_metric=args.kernel_metric,candidate_context=args.candidate_context,
            target_mode='anchored' if args.absolute_anchor and arm in ('difference','no_peer') else ('contrast' if arm in ('difference','no_peer') else 'absolute'),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            objective_version='pooled-covariance-conditional-program-values-v1'))


def program_kernel_predict(description,saved,probe='aligned',pair_description=None,policy=True):
    left={k:torch.as_tensor(v[None],dtype=torch.float64) for k,v in description.items()}
    right={k:torch.as_tensor(v,dtype=torch.float64) for k,v in saved['bank'].items()}
    norm={k:{name:torch.as_tensor(v,dtype=torch.float64) for name,v in n.items()} for k,n in saved['normalization'].items()}
    contextual=saved.get('candidate_context',False) and 'no_peer' not in saved['arm']
    node_key='node_context' if contextual else 'node'
    if contextual and (probe=='state-only' or pair_description is not None):
        value=left['node_context'].clone();d=left['node'].shape[-1]
        value[...,d:]=0. if probe=='state-only' else torch.as_tensor(pair_description['node_context'][None,...,d:],dtype=value.dtype)
        left=dict(left,node_context=value)
    parts=program_kernel_components(left,right,norm,saved['metrics']['length'],node_key)
    if pair_description is not None:
        changed=dict(left,pair=torch.as_tensor(pair_description['pair'][None],dtype=torch.float64))
        parts['pair']=program_kernel_components(changed,right,norm,saved['metrics']['length'],node_key)['pair']
    conditional=saved['arm'] in ('difference','no_peer') and saved.get('target_mode')!='anchored'
    interaction=0. if probe=='state-only' else saved['metrics']['interaction']
    kernel=parts['unary']+interaction*parts['pair']+(0. if conditional else 25.*parts['state'])
    if conditional:kernel=kernel[:,:,:,1:]
    coefficient=np.asarray(saved['alpha']).ravel()
    if policy and 'policy_residual' in saved:coefficient=coefficient+np.asarray(saved['policy_residual']).ravel()
    return kernel.reshape(left['z'].shape[1],-1)@torch.as_tensor(coefficient,dtype=torch.float64)


def fit_program_policy(args):
    """Calibrated conditional values and a separate decision-focused policy.

    The critic is frozen. Conditional arms additionally receive the actual
    paired, complete-return advantage in their actor update. Absolute arms use
    only their fitted critic's advantage, as ordinary policy improvement does.
    All four arms use the same policy class, search grid and update budget.
    """
    torch.set_num_threads(1);torch.set_grad_enabled(True);device=args.device
    train=load_program(args.output,'train',args.train_count,'cpu');valid=load_program(args.output,'validation',args.validation_count,'cpu')
    costs=train['costs'].to(device,dtype=torch.float64);truth=valid['costs'].to(device,dtype=torch.float64)
    vx=program_kernel_bank(valid,device);n,k=costs.shape;nv=len(truth)
    for arm in args.arms.split(','):
        parent=args.output/f'program-{arm}-{args.initial_seed}.pth';saved=torch.load(parent,weights_only=False)
        if saved.get('kind')!='program-kernel' or saved['train_sha256']!=train['sha256']:
            raise ValueError('Actor fitting requires its exact frozen conditional-value training bank.')
        tx={key:torch.as_tensor(v,dtype=torch.float64,device=device) for key,v in saved['bank'].items()}
        norm={key:{name:torch.as_tensor(v,dtype=torch.float64,device=device) for name,v in value.items()} for key,value in saved['normalization'].items()}
        paired=arm in ('difference','no_peer');conditional=paired and saved.get('target_mode')!='anchored'
        indices=slice(1,None) if conditional else slice(None)
        def matrix(left):
            key='node_context' if saved.get('candidate_context',False) and 'no_peer' not in arm else 'node'
            parts=program_kernel_components(left,tx,norm,saved['metrics']['length'],key)
            result=parts['unary']+saved['metrics']['interaction']*parts['pair']+(0. if conditional else 25.*parts['state'])
            return result[:,:,:,indices].reshape(left['z'].shape[0]*k,-1)
        kt=matrix(tx).detach();kv=matrix(vx).detach();gram=kt.reshape(n,k,-1)[:,indices].reshape(kt.shape[1],-1)
        value_alpha=torch.as_tensor(saved['alpha'].ravel(),device=device);qt=(kt@value_alpha).reshape(n,k);qv=(kv@value_alpha).reshape(nv,k)
        target=(costs-costs[:,:1]) if paired else (qt-qt[:,:1]).detach()
        def metrics(q):
            choice=q.argmin(-1);selected=truth.gather(-1,choice[:,None]).squeeze(-1)
            return dict(gain=float((truth[:,0]-selected).mean()),regret=float((selected-truth.min(-1).values).mean()),
                        changes=float((choice!=0).double().mean()))
        initial=metrics(qv);best=dict(score=initial['regret'],residual=np.zeros_like(saved['alpha']),metrics=dict(step=0,**initial));history=[best['metrics']]
        for temperature in (1.,3.):
            prior=torch.log_softmax(-qt/temperature,-1).detach()
            for regularization in (.003,.03,.3):
                residual=nn.Parameter(torch.zeros_like(value_alpha));optimizer=torch.optim.Adam([residual],lr=.01)
                for step in range(1,args.updates+1):
                    energy=(qt+(kt@residual).reshape(n,k));logp=torch.log_softmax(-energy/temperature,-1);probability=logp.exp()
                    risk=(probability*target).sum(-1).mean()
                    loss=risk+regularization*(residual@(gram@residual))/n+.1*(probability*(logp-prior)).sum(-1).mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_((residual,),10.);optimizer.step()
                    if step%50==0 or step==args.updates:
                        with torch.no_grad():current=metrics(qv+(kv@residual).reshape(nv,k))
                        record=dict(temperature=temperature,regularization=regularization,step=step,**current);history.append(record)
                        if current['regret']<best['score']:
                            best=dict(score=current['regret'],residual=residual.detach().cpu().numpy().reshape(saved['alpha'].shape).copy(),metrics=record)
                print(json.dumps(dict(arm=arm,temperature=temperature,regularization=regularization,best=best['metrics'])),flush=True)
        saved.update(seed=args.seed,policy_residual=best['residual'],policy_metrics=best['metrics'],policy_history=history,
            policy_parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),policy_parent=str(parent),
            policy_updates=args.updates,policy_target='paired actual advantage' if paired else 'frozen critic advantage',
            policy_validation_sha256=valid['sha256'],implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_torch(args.output/f'program-{arm}-{args.seed}.pth',saved)


class ProgramPolicy:
    def __init__(self,path,probe='aligned'):
        self.saved=torch.load(path,map_location='cpu',weights_only=False);self.config=self.saved['config'];self.probe=probe
        from interaction_evaluation import OriginalPolicy
        if hashlib.sha256(Path(self.saved['source']).read_bytes()).hexdigest()!=self.saved['source_sha256']:
            raise ValueError('Source policy changed.')
        self.source=OriginalPolicy(self.saved['source'])
        self.model=None
        if self.saved.get('kind')!='program-kernel':
            quadratic=self.saved.get('feature_version','').endswith('-v3')
            self.model=ProgramConditionalValue(self.saved['width'],'no_peer' in self.saved['arm'],quadratic).eval().requires_grad_(False)
            self.model.load_state_dict(self.saved['model'])
        self.reset()

    def reset(self):self.correction=None;self.decisions=[]

    @torch.no_grad()
    def choose_action(self,packet,deterministic=True):
        base=self.source(packet['obs']);c=self.config
        if self.correction is None and program_trigger(packet,c['trigger'],c.get('active_defenders',0)):
            options=program_options(len(base),c['radius'],packet,c.get('active_defenders',0));features=program_features(packet,base,options,c['bias'])
            if self.model is None:
                description=program_kernel_description(packet,features,options,c);mismatch=None
                if self.probe=='permuted':
                    changed=copy.deepcopy(packet);changed['obs'][:,12:14]=np.roll(packet['obs'][:,12:14],1,axis=0)
                    other=program_features(changed,np.roll(base,1,axis=0),options,c['bias'])
                    mismatch=program_kernel_description(changed,other,options,c)
                q=program_kernel_predict(description,self.saved,self.probe,mismatch)
            else:_,q=self.model({k:torch.as_tensor(v) for k,v in features.items()},self.probe)
            choice=self.saved['best_fixed_choice'] if self.probe=='fixed' else int(q.argmin())
            self.correction=options[choice].copy()
            self.decisions.append(dict(choice=choice,correction=self.correction.tolist(),value=q.tolist(),
                public_state_sha256=hashlib.sha256(np.asarray(packet['obs']).tobytes()+np.asarray(packet['motion']).tobytes()+np.asarray(packet['target_motion']).tobytes()).hexdigest()))
        return program_action(base,np.zeros(len(base)) if self.correction is None else self.correction,c['bias']),0.


def initialize_program_evaluation(output,seed,arms):
    global PROGRAM_POLICIES
    initialize_fleet();PROGRAM_POLICIES={}
    for arm in arms.split(','):
        if arm in ('arboids_cbf','constant_bias'):continue
        actual='difference' if arm in ('permuted','state-only','fixed') else arm
        PROGRAM_POLICIES[arm]=ProgramPolicy(Path(output)/f'program-{actual}-{seed}.pth',arm if arm in ('permuted','state-only','fixed') else 'aligned')
    PROGRAM_POLICIES['__arms__']=arms.split(',')


def program_episode(task):
    scene,n,agility,bias=task;row=dict(scene=scene,defenders=n,agility=agility)
    for arm in PROGRAM_POLICIES['__arms__']:
        policy=PROGRAM_POLICIES.get(arm)
        if policy:policy.reset()
        seed_random(scene);env=TADEnv(n,protocol='paper-parameters-v1');obs,_=env.reset(agility,noisy_agility=False)
        np.random.seed(scene+1000000);packet=fleet_public_packet(env,obs,measured_target=True);done=0
        while not done:
            was=len(policy.decisions) if policy else 0
            action=policy.choose_action(packet)[0] if policy else program_action(FLEET_SOURCE(packet['obs']),np.zeros(n),bias if arm=='constant_bias' else 0.)
            planned=policy and len(policy.decisions)>was
            if planned:
                reference=filtered(packet,program_action(FLEET_SOURCE(packet['obs']),np.zeros(n),bias),env.boids_actions)
            packet,_,done,thrust,_,_=execute(env,packet,action)
            packet['target_motion']=np.array([env.attacker.x,env.attacker.y,env.attacker.theta,*env.attacker.velocity])
            if planned:policy.decisions[-1].update(time=float(env.Current_T-.2),executed_thrust=np.asarray(thrust).tolist(),
                reference_thrust=np.asarray(reference).tolist(),executed_control_change=float(np.linalg.norm(thrust-reference)))
        row[arm]=outcome_row(env,done)
        if policy:row[arm]['decisions']=policy.decisions
    return row


def run_program(args):
    if args.mode=='program-fit':return fit_program(args)
    if args.mode=='program-kernel':return fit_program_kernel(args)
    if args.mode=='program-policy':return fit_program_policy(args)
    if args.mode in ('program-collect','program-refine'):
        tasks=[(i,args.split,args.output,args.defenders,args.repeats,args.program_bias,args.program_radius,args.program_trigger,args.agility,args.resume,args.active_defenders)
               for i in range(args.offset,args.offset+args.count)]
        init=initialize_fleet;initargs=();function=collect_program
        if args.mode=='program-refine':
            if args.root_source is None:raise ValueError('Refinement requires --root-source.')
            tasks=[(args.root_source,task) for task in tasks];function=refine_program
    else:
        tasks=[(3570000000+args.offset+i,args.defenders,args.agility,args.program_bias) for i in range(args.count)]
        init=initialize_program_evaluation;initargs=(args.output,args.seed,args.evaluation_arms);function=program_episode
        names=set('difference' if a in ('permuted','state-only','fixed') else a for a in args.evaluation_arms.split(',') if a not in ('arboids_cbf','constant_bias'))
        models={a:args.output/f'program-{a}-{args.seed}.pth' for a in names}
        hashes={a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in models.items()}
        settings=[torch.load(p,weights_only=False)['config'] for p in models.values()]
        if any(c!=settings[0] for c in settings) or any((c['defenders'],c['agility'],c['bias'])!=(args.defenders,args.agility,args.program_bias) for c in settings):
            raise ValueError('Evaluation and all matched programs must use the same task and reference.')
        code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        output=args.output/f'program-episodes-{args.seed}-{args.offset}-{args.count}-{hashlib.sha256(args.evaluation_arms.encode()).hexdigest()[:10]}.json'
        metadata=dict(model_hashes=hashes,implementation_sha256=code_hash,source_sha256=source_hash,
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=initargs) as pool:
        for row in pool.map(function,tasks):
            rows.append(row)
            if len(rows)%16==0:
                print(json.dumps(dict(completed=len(rows),seconds=time.monotonic()-start)),flush=True)
                if args.mode=='program-episodes':atomic_json(output,dict(**metadata,rows=rows,complete=False))
    if args.mode=='program-episodes':
        summary={a:dict(time=float(np.mean([r[a]['capture_time'] for r in rows])),capture=sum(r[a]['capture'] for r in rows),
            collision=sum(r[a]['collision'] for r in rows)) for a in args.evaluation_arms.split(',')}
        if code_hash!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or source_hash!=hashlib.sha256(SOURCE.read_bytes()).hexdigest() or any(hashes[a]!=hashlib.sha256(p.read_bytes()).hexdigest() for a,p in models.items()):
            raise ValueError('Frozen evaluation code or models changed during execution.')
        contrasts={};rng=np.random.default_rng(721503);ix=rng.integers(len(rows),size=(20000,len(rows)))
        for a in args.evaluation_arms.split(','):
            if a=='difference':continue
            d=np.array([r['difference']['capture_time']-r[a]['capture_time'] for r in rows]);sample=d[ix].mean(-1)
            contrasts[a]=dict(delta=float(d.mean()),ci95=np.quantile(sample,[.025,.975]).tolist(),ci99=np.quantile(sample,[.005,.995]).tolist())
        result=dict(**metadata,rows=rows,summary=summary,contrasts=contrasts,complete=True)
        atomic_json(output,result);print(json.dumps(dict(summary=summary,contrasts=contrasts)),flush=True)
    print(json.dumps(dict(completed=len(rows),seconds=time.monotonic()-start)),flush=True)


def reactive_options(packet,all_gates=False,control_radius=30.,gate_levels=3,reference=None):
    """Joint gate optimization; far receivers retain their source coefficient."""
    b,n=packet['obs'].shape[:2];device=packet['obs'].device
    if gate_levels not in (3,5):raise ValueError('Gate optimization supports three or five levels.')
    levels=torch.tensor([0.,-1.,1.] if gate_levels==3 else [0.,-1.,1.,-.5,.5],device=device)
    if all_gates:
        ids=torch.arange(gate_levels**n,device=device)
        digits=(ids[:,None]//(gate_levels**torch.arange(n,device=device))[None])%gate_levels
        values=levels[digits]
        programs=values[None].expand(b,-1,-1)*(packet['obs'][...,2]<control_radius)[:,None]
    else:
        nearest=packet['obs'][...,2].argsort(dim=-1,stable=True)[:,:2]
        grid=torch.cartesian_prod(levels,levels)
        programs=torch.zeros(b,len(grid),n,device=device)
        programs.scatter_(2,nearest[:,None].expand(-1,len(grid),-1),grid[None].expand(b,-1,-1))
    if reference is not None:
        # The source is one available fusion decision. Every receiver can
        # instead assign all authority to either of its current candidates.
        programs=programs*torch.where(programs>=0,1.-reference[:,None],reference[:,None])
    return programs


def reactive_reference(base,settings):
    return (base[...,2]+settings['bias']).clamp(0.,1.) if settings.get('full_gate',False) else None


def reactive_action(base,program,settings):
    action=base.clone()
    reference=reactive_reference(base,settings)
    if program.ndim==base.ndim:
        if program.shape[-1]!=2 or reference is None:
            raise ValueError('Component candidate fusion requires two full-range actuator gates.')
        return torch.cat((base[...,:2],(reference[...,None]+program).clamp(0.,1.)),-1)
    action[...,2]=((base[...,2]+settings['bias']+settings['radius']*program).clamp(0.,1.)
        if reference is None else (reference+program).clamp(0.,1.))
    return action


def reactive_public_model_packet(packet,agility):
    """Reconstruct a nominal dynamics state from public observations only.

    Measured velocity estimates water-relative velocity with zero mean water
    current. Target turn comes from observed headings. The supplied agility
    is the configured nominal model constant, never an observed private state.
    """
    motion,obs=packet['motion'],packet['obs'];count=len(obs)
    angle=motion[:,0,2]+obs[:,0,3]
    position=motion[:,0,:2]+obs[:,0,2,None]*torch.stack((angle.cos(),angle.sin()),-1)
    angle=motion[:,0,2]+obs[:,0,5]
    velocity=obs[:,0,4,None]*torch.stack((angle.cos(),angle.sin()),-1)
    heading=packet.get('target_heading',angle)
    turn=packet.get('target_turn',torch.zeros_like(heading))
    state=torch.cat((motion,torch.cat((position,heading[:,None],velocity,turn[:,None]),-1)[:,None]),1)
    central=torch.cat((state[...,:2]/60.,state[...,2,None].sin(),state[...,2,None].cos(),state[...,3:5]/5.,state[...,5,None]),-1).flatten(1)
    remaining=packet['time_remaining']
    central=torch.cat((central,motion.new_full((count,1),agility/3.),remaining[:,None]),-1)
    return dict(obs=obs,motion=motion,central=central,target_heading=heading,time_remaining=remaining,target_turn=turn)


@torch.no_grad()
def reactive_model_prior(source,packet,programs,settings,repeats=8,supported_only=False,return_response=False):
    """A reproducible public-model response prior, not a true return label.

    Antithetic common futures are fixed across states and every joint option.
    The prior's continuation is the frozen zero-increment source policy;
    paired supervision must still identify the actual collecting policy's
    return. Real environment labels never enter this public prediction.
    """
    from policy.interaction_sac import MirrorCoordinationGate
    if repeats<2 or repeats%2:raise ValueError('Use an even number of antithetic model futures.')
    if settings.get('canonical_grouped_model',False):source=ReactiveGroupedSource(source)
    elif settings.get('canonical_model',False):source=ReactiveCanonicalSource(source)
    public=reactive_public_model_packet(packet,settings['agility'])
    count,n=packet['obs'].shape[:2];options=programs.shape[1]
    generator=torch.Generator().manual_seed(719992)
    half=torch.randn(repeats//2,300,2+3*(n+1),generator=generator)
    futures=torch.stack((half,-half),1).flatten(0,1)
    if settings.get('exact_joint_fleet',False):
        # Saturated gates and ineligible receivers produce identical physical
        # controls. Simulate each distinct control once with the same futures.
        unique=[];origins=[];inverse=[];offset=0
        for row in range(count):
            values,index=torch.unique(programs[row],dim=0,return_inverse=True)
            if supported_only and 'max_coordinated_receivers' in settings:
                allowed=(values.abs()>1e-7).sum(-1)<=settings['max_coordinated_receivers']
                retained=allowed.nonzero().flatten()
                zero=((values[retained]==0.).all(-1)).nonzero().flatten()[0]
                mapping=torch.full((len(values),),int(zero),dtype=torch.long)
                mapping[retained]=torch.arange(len(retained))
                index=mapping[index];values=values[retained]
            unique.append(values);origins.append(torch.full((len(values),),row,dtype=torch.long))
            inverse.append(index+offset);offset+=len(values)
        unique=torch.cat(unique);origins=torch.cat(origins);inverse=torch.stack(inverse)
        expanded={k:v[origins].repeat_interleave(repeats,0) for k,v in public.items()}
        innovations=futures[None].expand(len(unique),-1,-1,-1).reshape(len(unique)*repeats,300,-1)
        cost,_,auxiliary=reactive_rollout(MirrorCoordinationGate(16),source,expanded,innovations,settings,
            first=unique.repeat_interleave(repeats,0),**({'return_prefix':True} if return_response else {}))
        prediction=cost.reshape(len(unique),repeats)[inverse].transpose(1,2)
        if not return_response:return prediction
        response=auxiliary['prefix_response'].reshape(len(unique),repeats,n,2,2).mean(1)[inverse]
        return prediction,response
    innovations=futures[None,:,None].expand(count,-1,options,-1,-1).reshape(count*repeats*options,300,-1)
    expanded={k:v.repeat_interleave(repeats*options,0) for k,v in public.items()}
    first=programs[:,None].expand(-1,repeats,-1,-1).reshape(-1,n)
    cost,_,auxiliary=reactive_rollout(MirrorCoordinationGate(16),source,expanded,innovations,settings,first=first,
        **({'return_prefix':True} if return_response else {}))
    prediction=cost.reshape(count,repeats,options)
    if not return_response:return prediction
    return prediction,auxiliary['prefix_response'].reshape(count,repeats,options,n,2,2).mean(1)


def reactive_prior_options(packet,base,settings):
    """One reference plus all single and simultaneous two-receiver switches."""
    count,n=base.shape[:2];reference=reactive_reference(base,settings)
    allowed=packet['obs'][...,2]<settings.get('control_radius',30.)
    levels=torch.stack((-reference,1.-reference),1)
    programs=[torch.zeros_like(reference)];parts=[(0,0)]
    for i in range(n):
        for a in range(2):
            program=torch.zeros_like(reference);program[:,i]=levels[:,a,i]*allowed[:,i]
            programs.append(program);parts.append((len(programs)-1,0))
    for i in range(n):
        for j in range(i+1,n):
            for a in range(2):
                for b in range(2):
                    programs.append(programs[1+2*i+a]+programs[1+2*j+b])
                    parts.append((1+2*i+a,1+2*j+b))
    return torch.stack(programs,1),torch.tensor(parts,device=base.device)


def reactive_component_options(packet,base,settings):
    """Separate actuator fusion with a complete face for the nearest pair.

    Every vessel retains its own four candidate corners. Joint decoding uses
    the two current closest receivers; their identities are refreshed at each
    decision, without reordering the physical simulator's vessel state.
    """
    count,n=base.shape[:2];reference=reactive_reference(base,settings)
    order=packet['obs'][...,2].argsort(dim=-1,stable=True)
    allowed=packet['obs'][...,2]<settings.get('control_radius',30.)
    corners=base.new_tensor([[0.,0.],[1.,1.],[1.,0.],[0.,1.]])
    programs=[base.new_zeros(count,n,2)];parts=[(0,0)];rows=torch.arange(count,device=base.device)
    for rank in range(n):
        vessel=order[:,rank]
        for corner in corners:
            program=torch.zeros_like(programs[0])
            program[rows,vessel]=(corner-reference[rows,vessel,None])*allowed[rows,vessel,None]
            programs.append(program);parts.append((len(programs)-1,0))
    pairs=[(i,j) for i in range(n) for j in range(i+1,n)] if settings.get('component_all_pairs',False) else [(0,1)]
    for i,j in pairs:
        for left in range(1+4*i,5+4*i):
            for right in range(1+4*j,5+4*j):
                programs.append(programs[left]+programs[right]);parts.append((left,right))
    return torch.stack(programs,1),torch.tensor(parts,device=base.device)


class ReactiveComponentGate(nn.Module):
    """Learn and refresh joint two-actuator candidate fusion in closed loop."""
    def __init__(self,settings,calibration,risk=0.,interaction=True,source=None):
        super().__init__();self.settings=settings;self.risk=risk;self.interaction=interaction
        self.register_buffer('calibration',calibration.float())
        object.__setattr__(self,'source',source)

    @torch.no_grad()
    def decide(self,packet,base,settings,probe='aligned'):
        programs,parts=reactive_component_options(packet,base,settings)
        enabled=self.interaction and probe!='interaction-off'
        if settings.get('component_orthogonal',False):
            return self.cooperative_decision(packet,base,programs,parts,settings,enabled)
        count=programs.shape[1] if enabled else 1+4*base.shape[1]
        public={k:v.cpu() for k,v in packet.items()}
        prior=reactive_model_prior(self.source,public,programs[:,:count].cpu(),settings,4)
        coefficients=self.calibration.cpu().double()
        if not enabled:
            singles=reactive_monotone_estimates(coefficients,prior.double(),parts[:count].cpu(),public,False)
            prediction=singles[...,parts.cpu()].sum(-1)
        else:prediction=reactive_monotone_estimates(coefficients,prior.double(),parts.cpu(),public,True)
        # Common disturbances identify the policy-relevant contrast. Model
        # futures are averaged before measuring learned-head uncertainty.
        prediction=prediction.mean(2).to(base)
        score=prediction.mean(0)+self.risk*prediction.std(0,unbiased=False)
        score=score+settings['prior']*programs.square().mean((-1,-2))
        choice=score.argmin(-1)
        self.last_prediction=dict(prior=prior.cpu().clone(),objective=score.cpu().clone())
        return choice,programs[torch.arange(len(base),device=base.device),choice]

    @torch.no_grad()
    def cooperative_decision(self,packet,base,programs,parts,settings,enabled):
        public={k:v.cpu() for k,v in packet.items()};programs=programs.cpu();parts=parts.cpu()
        response_features=settings.get('component_response_features',False)
        if settings.get('component_contextual',False) and not response_features:
            from conditional_fusion_value import component_relation_features
            public=dict(public,candidate_interactions=component_relation_features(public,base.cpu(),programs,parts,settings['bias']))
        coefficients=self.calibration.cpu().double();singles=1+4*base.shape[1]
        prior=reactive_model_prior(self.source,public,programs[:,:singles],settings,4,return_response=response_features)
        if response_features:prior,single_response=prior
        single=reactive_monotone_estimates(coefficients,prior.double(),parts[:singles],public,False)
        independent=single[...,parts].sum(-1).mean((0,2))
        initial=independent+settings['prior']*programs.square().mean((-1,-2))
        reference=initial.argmin(-1);rows=torch.arange(len(base))
        if not enabled:
            self.last_prediction=dict(prior=prior,objective=initial)
            return reference.to(base.device),programs[rows,reference].to(base.device)
        count=min(settings.get('component_candidate_budget',16),len(parts)-singles)
        joint=initial[:,singles:].argsort(-1)[:,:count]+singles
        selected=programs.gather(1,joint[:,:,None,None].expand(-1,-1,programs.shape[2],2))
        forecast=reactive_model_prior(self.source,public,selected,settings,4,return_response=response_features)
        if response_features:
            from conditional_fusion_value import component_response_features
            forecast,joint_response=forecast
            response=single_response.new_zeros(len(base),len(parts),base.shape[1],2,2)
            response[:,:singles]=single_response
            response.scatter_(1,joint[:,:,None,None,None].expand(-1,-1,base.shape[1],2,2),joint_response)
            public=dict(public,candidate_interactions=component_response_features(public,base.cpu(),programs,parts,response,settings['bias'],settings.get('component_response_scale')))
        complete=prior.new_zeros(len(base),4,len(parts));complete[:,:,:singles]=prior
        complete.scatter_(2,joint[:,None].expand(-1,4,-1),forecast)
        prediction=reactive_monotone_estimates(coefficients,complete.double(),parts,public,True).mean(2)
        score=reactive_component_paired_score(prediction,programs,reference,self.risk,settings['prior'])
        support=torch.zeros_like(score,dtype=torch.bool);support[:,:singles]=True;support.scatter_(1,joint,True)
        if not support[rows,reference].all():raise RuntimeError('The incumbent must remain in the joint decision beam.')
        score=score.masked_fill(~support,float('inf'));choice=score.argmin(-1)
        self.last_prediction=dict(prior=complete,objective=score)
        return choice.to(base.device),programs[rows,choice].to(base.device)


def reactive_component_paired_score(prediction,programs,reference,risk,trust):
    contrast=prediction-prediction.gather(-1,reference[None,:,None].expand(len(prediction),-1,1))
    incumbent=programs[torch.arange(len(programs),device=programs.device),reference]
    return contrast.mean(0)+risk*contrast.std(0,unbiased=False)+trust*(programs-incumbent[:,None]).square().mean((-1,-2))


def collect_reactive_component_credits(args):
    """Measure the complete joint actuator face under one fixed continuation."""
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    reference_path=args.output/f'reactive-additive-{args.initial_seed}.pth'
    reference=torch.load(reference_path,map_location='cpu',weights_only=False)
    reference_hash=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    settings=dict(bank['settings'],exact_joint_fleet=True,canonical_model=True,canonical_grouped_model=True,component_fusion=True)
    if getattr(args,'component_complete_support',False):
        settings['component_all_pairs']=True
    coefficients=reference.get('unary_reference',reference.get('calibration')).clone()
    if coefficients.shape[-1]!=4:coefficients=coefficients[...,:4]
    source=ReactiveSource(args.source_device)
    continuation_kind=getattr(args,'component_continuation','source')
    if continuation_kind=='source':
        from policy.interaction_sac import MirrorCoordinationGate
        continuation=MirrorCoordinationGate(16)
        continuation_record='source'
    else:
        continuation=ReactiveComponentGate(settings,coefficients,0.,False,source)
        continuation_record=dict(kind='reactive-component-v1',calibration=coefficients,interaction=False,risk=0.)
    origins=bank['data']['origin'];generator=torch.Generator().manual_seed(args.seed)
    train=torch.isin(origins,reference['training_origins']).nonzero().flatten()
    held=torch.isin(origins,reference['held_origins']).nonzero().flatten()
    held_count=min(args.validation_count if getattr(args,'component_complete_support',False) else 16,len(held))
    selected=torch.cat((train[torch.randperm(len(train),generator=generator)[:args.count]],
        held[torch.randperm(len(held),generator=generator)[:held_count]]))
    packet={k:v[selected] for k,v in bank['data']['packet'].items()};base=bank['data']['base'][selected]
    programs,parts=reactive_component_options(packet,base,settings)
    costs=[];outcomes=[];priors=[];started=time.monotonic()
    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for begin in range(0,len(selected),args.reactive_batch):
        end=min(len(selected),begin+args.reactive_batch);size=end-begin;branches=programs.shape[1];n=base.shape[1]
        p={k:v[begin:end] for k,v in packet.items()}
        noise=torch.randn(size*args.repeats,300,2+3*(n+1),generator=generator).repeat_interleave(branches,0)
        expanded={k:v.repeat_interleave(args.repeats*branches,0) for k,v in p.items()}
        first=programs[begin:end,None].expand(-1,args.repeats,-1,-1,-1).reshape(-1,n,2)
        cost,outcome,_=reactive_rollout(continuation,source,expanded,noise,settings,first=first)
        costs.append(cost.reshape(size,args.repeats,branches));outcomes.append(outcome.reshape(size,args.repeats,branches))
        priors.append(reactive_model_prior(source,p,programs[begin:end],settings,4))
        data=dict(packet={k:v[:end] for k,v in packet.items()},base=base[:end],options=programs[:end],parts=parts,
            origin=origins[selected[:end]],costs=torch.cat(costs),outcomes=torch.cat(outcomes))
        result=dict(data=data,prior=torch.cat(priors),settings=settings,complete=False,training_count=end,
            noise_seed=args.seed,training_origins=reference['training_origins'],held_origins=reference['held_origins'],
            collecting_policy=continuation_record,
            source_reference=dict(path=str(reference_path),sha256=reference_hash),
            source_bank_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            training_implementation_sha256=implementation)
        atomic_torch(args.output/f'reactive-component-data-{args.seed}.pth',result)
        print(json.dumps(dict(states=end,total=len(selected),seconds=time.monotonic()-started)),flush=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=implementation:raise RuntimeError('The actuator credit implementation changed.')
    result['complete']=True;atomic_torch(args.output/f'reactive-component-data-{args.seed}.pth',result)


def fit_reactive_component(args):
    """Identify individual and joint actuator values from the same returns."""
    if getattr(args,'component_conditional_auxiliary',False):
        from conditional_fusion_value import fit_conditional_component
        return fit_conditional_component(args,sys.modules[__name__])
    torch.set_num_threads(2)
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    if not bank.get('complete'):raise ValueError('Complete component credits are required.')
    data=bank['data'];settings=bank['settings'];origins=data['origin']
    train=torch.isin(origins,bank['training_origins']).nonzero().flatten()
    valid=torch.isin(origins,bank['held_origins']).nonzero().flatten()
    if len(train)<16 or len(valid)<8:raise ValueError('Separate fitting and held origins are required.')
    training=reactive_subset(data,train);validation=reactive_subset(data,valid);parts=data['parts']
    orthogonal=getattr(args,'component_orthogonal',False)
    fit_seed=args.seed if args.bootstrap_seed is None else args.bootstrap_seed
    common=None
    if orthogonal:
        common,_=fit_prior_orthogonal(training,bank['prior'][train],True,False,fit_seed,3,False,False,True)
        settings=dict(settings,component_orthogonal=True,component_all_pairs=True,component_candidate_budget=16)
    for arm in args.evaluation_arms.split(','):
        conditional=arm!='absolute';interaction=arm!='additive'
        if orthogonal:
            coefficients=(fit_reactive_component_interaction(common,training,bank['prior'][train],conditional,fit_seed,
                getattr(args,'component_crossfit_value',False))
                if interaction else common)
        else:coefficients,_=fit_prior_monotone(training,bank['prior'][train],conditional,interaction,fit_seed,3,True,False,False,True)
        predictions=reactive_monotone_estimates(coefficients.double(),bank['prior'][valid].double(),parts,validation['packet'],interaction).mean(2)
        penalty=settings['prior']*validation['options'].square().mean((-1,-2))
        independent=reactive_monotone_estimates(coefficients.double(),bank['prior'][valid].double(),parts,validation['packet'],False).mean((0,2))
        reference=(independent+penalty).argmin(-1)
        trials=[]
        for risk in (0.,.5,1.,2.):
            if orthogonal:
                choice=(reactive_component_paired_score(predictions,validation['options'],reference,risk,settings['prior']).argmin(-1)
                    if interaction else reference)
            else:choice=(predictions.mean(0)+risk*predictions.std(0,unbiased=False)+penalty).argmin(-1)
            selected=validation['costs'].mean(1)[torch.arange(len(valid)),choice]
            trials.append(dict(risk=risk,cost=float(selected.mean())))
        selection=min(trials,key=lambda row:row['cost'])
        actor=ReactiveComponentGate(settings,coefficients,selection['risk'],interaction)
        saved=dict(kind='reactive-component-v1',actor=actor.state_dict(),settings=settings,calibration=coefficients,
            risk=selection['risk'],interaction=interaction,conditional=conditional,seed=args.seed,arm=arm,training_seed=fit_seed,
            crossfit_control_value=getattr(args,'component_crossfit_value',False),
            selection=selection,calibration_trials=trials,training_origins=bank['training_origins'],held_origins=bank['held_origins'],
            collecting_policy=bank['collecting_policy'],data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',saved)
        print(json.dumps(dict(arm=arm,selection=selection,trials=trials)),flush=True)


def reactive_crossfit_control_targets(costs,futures):
    """Estimate each contrast's precision from different future realizations."""
    folds=[]
    if costs.shape[1]<4:raise ValueError('Cross-fitted control values need at least four futures.')
    for half in (0,1):
        weight=futures[:,half::2];weight=weight/weight.sum(-1,keepdim=True)
        target=(costs[:,half::2]*weight[:,:,None]).sum(1)
        target=target[:,1:]-target[:,:1]
        precision=reactive_precision(costs[:,1-half::2],True)
        folds.append((target,precision))
    return folds


def reactive_crossfit_control_loss(energy,folds,rows):
    contrast=energy[:,1:]-energy[:,:1];loss=energy.new_zeros(())
    for target,precision in folds:
        error=contrast-target
        loss=loss+(rows*torch.einsum('bi,bij,bj->b',error,precision,error)).mean()/error.shape[-1]
    return loss/len(folds)


def fit_reactive_component_interaction(common,data,prior,conditional,seed,crossfit=False):
    """A shared single-vessel value and an independently identified joint gain."""
    parts=data['parts'];joints=(parts[:,1]>0).nonzero().flatten();fitted=[]
    costs=data['costs'].double();mixed=costs-costs[:,:,parts].sum(-1)+costs[:,:,:1]
    precision=reactive_precision(mixed[:,:,torch.cat((torch.zeros(1,dtype=torch.long),joints))],True)
    unique=reactive_unique_control_mask(data['options'].flatten(2))
    for head in range(len(common)):
        generator=torch.Generator().manual_seed(seed+head*104729)
        rows,futures=reactive_positive_credit_weights(reactive_state_groups(data),costs.shape[1],generator)
        folds=reactive_crossfit_control_targets(costs,futures) if crossfit and conditional else None
        target=(costs*futures[:,:,None]).sum(1);mixed_target=(mixed*futures[:,:,None]).sum(1)
        unary=common[head:head+1,:4].double()
        independent=reactive_monotone_estimates(common[head:head+1].double(),prior.double(),parts,data['packet'],False)[0].mean(1).detach()
        with torch.enable_grad():
            joint=nn.Parameter(common[head:head+1,4:].double().clone())
            baseline=nn.Parameter((target-independent).mean(-1))
            regrets=reactive_pair_regret_weights(costs,futures)
            optimizer=torch.optim.LBFGS([joint,baseline],max_iter=150,line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad(set_to_none=True)
                value=reactive_monotone_estimates(torch.cat((unary,joint),1),prior.double(),parts,data['packet'],True)[0].mean(1)
                if conditional:
                    if crossfit:loss=reactive_crossfit_control_loss(value,folds,rows)
                    else:
                        error=(value-independent-mixed_target)[:,joints]
                        loss=(rows*torch.einsum('bi,bij,bj->b',error,precision,error)).mean()/len(joints)
                    loss=loss+.1*reactive_pair_regret_loss(value,regrets,rows,temperature=2.,support=unique)
                else:
                    variance=1.+costs.var(1,unbiased=True)/costs.shape[1]
                    error=(value+baseline[:,None]-target).square()/variance
                    loss=(rows*(error*unique).sum(-1)/unique.sum(-1)).mean()
                centered=joint.clone();centered[0,4,0]-=torch.expm1(torch.tensor(1.,dtype=torch.float64)).log()
                loss=loss+.01*centered.square().mean();loss.backward();return loss
            optimizer.step(closure);fitted.append(torch.cat((unary,joint.detach()),1)[0].float())
    return torch.stack(fitted)


def reactive_joint_options(packet,base,settings):
    """Complete joint controls and their exact single-vessel reference indices."""
    n=base.shape[1]
    programs=reactive_options(packet,True,settings.get('control_radius',30.),3,reactive_reference(base,settings))
    place=3**torch.arange(n,device=base.device)
    digits=(torch.arange(3**n,device=base.device)[:,None]//place[None])%3
    return programs,digits*place[None]


def reactive_prior_features(costs,parts):
    """Separate identifiable individual effects and excess joint effects."""
    difference=costs-costs[...,:1]
    unary=difference[...,parts[:,0]]+difference[...,parts[:,1]]
    return torch.stack((unary,difference-unary),-1)


def reactive_credit_parts(data):
    if 'parts' in data:return data['parts']
    if data['costs'].shape[-1]!=9:raise ValueError('Non-ternary credits require explicit factor indices.')
    return torch.tensor([(3*(i//3),i%3) for i in range(9)])


def fit_prior_calibration(data,prior,conditional=True,interaction=True,seed=9970,members=5,decision_regret=False):
    """Calibrate a public response prior with true paired policy returns."""
    parts=reactive_credit_parts(data)
    features=reactive_prior_features(prior,parts).mean(1).double()
    if not interaction:features[...,1]=0.
    state=data['packet']['obs'];baseline=prior.mean(1)[:,0]
    state=torch.stack((torch.ones_like(baseline),baseline/10.,state[...,2].amin(-1)/10.,
        state[...,2].mean(-1)/20.,data['packet']['time_remaining']),-1).double()
    groups=reactive_state_groups(data);coefficients=[];excluded=[]
    for member in range(members):
        generator=torch.Generator().manual_seed(seed+member*104729)
        index,oob=reactive_group_sample(groups,generator);excluded.append(oob)
        costs=data['costs'][index].double()
        future=torch.randint(costs.shape[1],costs.shape[:2],generator=generator)
        costs=costs.gather(1,future[:,:,None].expand_as(costs))
        target=costs.mean(1);x=features[index]
        if conditional:
            x=x[:,1:]-x[:,:1];target=target[:,1:]-target[:,:1]
            precision=reactive_precision(costs,True)
            matrix=torch.einsum('bki,bkl,blj->ij',x,precision,x)
            right=torch.einsum('bki,bkl,bl->i',x,precision,target)
            beta=torch.linalg.solve(matrix+torch.eye(2,dtype=matrix.dtype),right)
            if decision_regret:
                # Weight each ordering error by the actual paired time lost,
                # not by the number of futures in which one option wins.
                # Thus rare expensive failures retain their task importance.
                features_all=features[index]
                regrets=costs[:,:,:,None]-costs[:,:,None,:]
                worse=regrets.clamp_min(0.).mean(1)/2.
                better=(-regrets).clamp_min(0.).mean(1)/2.
                mask=~torch.eye(costs.shape[-1],dtype=torch.bool)
                with torch.enable_grad():
                    fitted=nn.Parameter(beta.clone())
                    optimizer=torch.optim.LBFGS([fitted],max_iter=100,line_search_fn='strong_wolfe')
                    def closure():
                        optimizer.zero_grad(set_to_none=True)
                        error=torch.einsum('bki,i->bk',x,fitted)-target
                        loss=torch.einsum('bk,bkl,bl->b',error,precision,error).mean()/error.shape[-1]
                        energy=features_all@fitted
                        margin=(energy[:,:,None]-energy[:,None,:])/2.
                        loss=loss+(worse[:,mask]*F.softplus(-margin[:,mask])+better[:,mask]*F.softplus(margin[:,mask])).mean()
                        loss=loss+fitted.square().sum()/max(len(index),1)
                        loss.backward();return loss
                    optimizer.step(closure);beta=fitted.detach()
        else:
            x=torch.cat((state[index,None].expand(-1,prior.shape[-1],-1),x),-1).flatten(0,1)
            beta=torch.linalg.solve(x.T@x+torch.eye(x.shape[-1],dtype=x.dtype),x.T@target.flatten())[-2:]
        coefficients.append(beta.float())
    return torch.stack(coefficients),torch.stack(excluded)


def reactive_prior_context(packet,prior,parts):
    difference=prior-prior[...,:1]
    mixed=difference-difference[...,parts[:,0]]-difference[...,parts[:,1]]
    effects=torch.stack((difference,mixed),-1)
    mean=effects.mean(1);spread=effects.std(1,unbiased=False)
    reference=prior.mean(1)[:,0]/20.
    proximity=torch.exp(-packet['obs'][...,2].amin(-1)/15.)
    turn=packet.get('target_turn',torch.zeros_like(reference)).abs().clamp(max=3.)
    shared=lambda x:x[:,None,None].expand_as(mean)
    context=torch.stack((torch.ones_like(mean),shared(reference),spread/10.,mean.abs()/10.,shared(proximity),shared(turn)),-1)
    return effects,spread,context


def reactive_reliability_estimates(coefficients,prior,parts,packet,interaction=True):
    effects,spread,context=reactive_prior_context(packet,prior,parts)
    gates=torch.sigmoid(torch.einsum('bkdc,mdhc->mbkdh',context,coefficients.to(context)))
    calibrated=effects[None]*gates[:,:,None,:,: ,0]+spread[None,:,None]*gates[:,:,None,:,: ,1]
    unary=calibrated[...,0]
    return unary[...,parts[:,0]]+unary[...,parts[:,1]]+(calibrated[...,1] if interaction else 0.)


def reactive_distribution_features(packet,prior,parts):
    """Conditional response distributions, separately for unary/joint effects.

    Bounded moments distinguish consistent control improvements from a few
    long simulated chases. Every feature vanishes for a null intervention.
    Nonlinear moments are computed BEFORE adding individual control factors,
    so removing the mixed factor really removes the joint value term.
    """
    effects,spread,_=reactive_prior_context(packet,prior,parts)
    moments=torch.stack((effects.mean(1),(2.*(effects/2.).tanh()).mean(1),
        (8.*(effects/8.).tanh()).mean(1),effects.abs().mean(1),spread),-1)
    reference=prior.mean(1)[:,0]/20.
    context=torch.stack((torch.ones_like(reference),reference,
        torch.exp(-packet['obs'][...,2].amin(-1)/15.),
        packet.get('target_turn',torch.zeros_like(reference)).abs().clamp(max=3.)),-1)
    raw=(moments[...,None]*context[:,None,None,None,:]).flatten(-2)
    unary=raw[:,:,0]
    return torch.stack((unary[:,parts[:,0]]+unary[:,parts[:,1]],raw[:,:,1]),-2)


def fit_prior_distribution(data,prior,conditional=True,interaction=True,seed=9970,members=5):
    parts=reactive_credit_parts(data)
    features=reactive_distribution_features(data['packet'],prior,parts).double()
    if not interaction:features[:,:,1]=0.
    features=features.flatten(-2);groups=reactive_state_groups(data)
    distance=data['packet']['obs'][...,2].double();reference=prior.mean(1)[:,0].double()
    state=torch.stack((torch.ones_like(reference),reference/10.,distance.amin(-1)/10.,
        distance.mean(-1)/20.,data['packet']['time_remaining'].double()),-1)
    coefficients=[];excluded=[]
    for member in range(members):
        generator=torch.Generator().manual_seed(seed+member*104729)
        index,oob=reactive_group_sample(groups,generator);excluded.append(oob)
        costs=data['costs'][index].double()
        future=torch.randint(costs.shape[1],costs.shape[:2],generator=generator)
        costs=costs.gather(1,future[:,:,None].expand_as(costs));target=costs.mean(1)
        scale=features[index].square().mean((0,1)).sqrt().clamp_min(.1)
        x=features[index]/scale
        if conditional:
            difference=x[:,1:]-x[:,:1];contrast=target[:,1:]-target[:,:1]
            precision=reactive_precision(costs,True)
            matrix=torch.einsum('bki,bkl,blj->ij',difference,precision,difference)/len(index)
            right=torch.einsum('bki,bkl,bl->i',difference,precision,contrast)/len(index)
            beta=torch.linalg.solve(matrix+.1*torch.eye(x.shape[-1],dtype=x.dtype),right)
            with torch.enable_grad():
                fitted=nn.Parameter(beta)
                optimizer=torch.optim.LBFGS([fitted],max_iter=100,line_search_fn='strong_wolfe')
                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    energy=x@fitted;error=difference@fitted-contrast
                    loss=torch.einsum('bk,bkl,bl->b',error,precision,error).mean()/error.shape[-1]
                    # Optimize the control distribution using the actual
                    # paired task loss, not only its value-fitting surrogate.
                    log_policy=(-energy/2.).log_softmax(-1)
                    regret=target-target[:,:1]
                    loss=loss+(log_policy.exp()*regret).sum(-1).mean()/2.
                    loss=loss+.1*fitted.square().sum()/error.shape[-1]
                    loss.backward();return loss
                optimizer.step(closure);beta=fitted.detach()
        else:
            design=torch.cat((state[index,None].expand(-1,x.shape[1],-1),x),-1).flatten(0,1)
            matrix=design.T@design/len(index)
            beta=torch.linalg.solve(matrix+.1*torch.eye(design.shape[-1],dtype=x.dtype),design.T@target.flatten()/len(index))[-x.shape[-1]:]
        coefficients.append((beta/scale).reshape(2,-1).float())
    return torch.stack(coefficients),torch.stack(excluded)


def fit_prior_reliability(data,prior,conditional=True,interaction=True,seed=9970,members=5):
    """Learn when individual and joint candidate-response values are reliable."""
    initial,excluded=fit_prior_calibration(data,prior,conditional,interaction,seed,members,True)
    parts=reactive_credit_parts(data)
    coefficients=[];groups=reactive_state_groups(data)
    for member in range(members):
        generator=torch.Generator().manual_seed(seed+member*104729)
        index,_=reactive_group_sample(groups,generator)
        packet={k:v[index] for k,v in data['packet'].items()};forecast=prior[index].double()
        costs=data['costs'][index].double()
        future=torch.randint(costs.shape[1],costs.shape[:2],generator=generator)
        costs=costs.gather(1,future[:,:,None].expand_as(costs));target=costs.mean(1)
        precision=reactive_precision(costs,True)
        position=packet['obs'][...,2].double();baseline=forecast.mean(1)[:,0]
        state=torch.stack((torch.ones_like(baseline),baseline/10.,position.amin(-1)/10.,position.mean(-1)/20.,packet['time_remaining'].double()),-1)
        with torch.enable_grad():
            weights=torch.zeros(1,2,2,6,dtype=torch.float64)
            weights[0,:,0,0]=torch.logit(initial[member].double().clamp(.02,.98))
            weights[0,:,1,0]=torch.logit(torch.tensor(.02,dtype=torch.float64))
            weights=nn.Parameter(weights)
            nuisance=nn.Parameter(torch.linalg.lstsq(state,target.mean(-1)).solution)
            parameters=[weights] if conditional else [weights,nuisance]
            optimizer=torch.optim.LBFGS(parameters,max_iter=150,line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad(set_to_none=True)
                energy=reactive_reliability_estimates(weights,forecast,parts,packet,interaction)[0].mean(1)
                if conditional:
                    error=energy[:,1:]-energy[:,:1]-(target[:,1:]-target[:,:1])
                    loss=torch.einsum('bk,bkl,bl->b',error,precision,error).mean()/error.shape[-1]
                    regrets=costs[:,:,:,None]-costs[:,:,None,:]
                    worse=regrets.clamp_min(0.).mean(1)/2.;better=(-regrets).clamp_min(0.).mean(1)/2.
                    margin=(energy[:,:,None]-energy[:,None,:])/2.
                    mask=~torch.eye(costs.shape[-1],dtype=torch.bool)
                    loss=loss+(worse[:,mask]*F.softplus(-margin[:,mask])+better[:,mask]*F.softplus(margin[:,mask])).mean()
                else:loss=((state@nuisance)[:,None]+energy-target).square().mean()/100.
                loss=loss+.005*weights[...,1:].square().mean()
                loss.backward();return loss
            optimizer.step(closure)
            coefficients.append(weights[0].detach().float())
    return torch.stack(coefficients),excluded


@torch.no_grad()
def reactive_candidate_interactions(packet,base,programs,parts):
    """Symmetric messages binding each current candidate pair to its geometry."""
    if parts.shape[-1]>2:
        total=base.new_zeros(len(base),len(parts),12);count=base.new_zeros(len(base),len(parts))
        for i in range(parts.shape[-1]):
            for j in range(i+1,parts.shape[-1]):
                selected=parts[:,[i,j]]
                total+=reactive_candidate_interactions(packet,base,programs,selected)
                count+=((programs[:,selected[:,0]].abs().amax(-1)>0.)&
                    (programs[:,selected[:,1]].abs().amax(-1)>0.))
        return total/count.clamp_min(1.)[...,None]
    from policy.interaction_sac import CandidateCoordinationGate
    _,paths=CandidateCoordinationGate.candidate_responses(packet,base)
    displacement=(paths[:,:,-1,1]-paths[:,:,-1,0])/5.
    motion,obs=packet['motion'],packet['obs'];count,options=programs.shape[:2]
    rows=torch.arange(count)[:,None].expand(-1,options)
    first=programs[:,parts[:,0]];second=programs[:,parts[:,1]]
    left=first.abs().argmax(-1);right=second.abs().argmax(-1)
    alpha=first.gather(-1,left[...,None])[...,0];beta=second.gather(-1,right[...,None])[...,0]
    di=displacement[rows,left]*alpha[...,None];dj=displacement[rows,right]*beta[...,None]
    pi,pj=motion[rows,left,:2],motion[rows,right,:2]
    vi,vj=motion[rows,left,3:5],motion[rows,right,3:5]
    bearing=motion[...,2]+obs[...,3]
    direction=torch.stack((bearing.cos(),bearing.sin()),-1)
    ni,nj=direction[rows,left],direction[rows,right]
    distance_i,distance_j=obs[rows,left,2],obs[rows,right,2]
    separation=pi-pj;length=separation.norm(dim=-1)
    unit=separation/length[...,None].clamp_min(1e-6)
    cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
    norm_i,norm_j=di.norm(dim=-1),dj.norm(dim=-1)
    features=torch.stack(((ni*nj).sum(-1),cross(ni,nj).abs(),torch.exp(-length/10.),
        ((vi-vj)*unit).sum(-1)/5.,torch.minimum(distance_i,distance_j)/20.,
        (distance_i-distance_j).abs()/20.,(di*dj).sum(-1),((di-dj)*unit).sum(-1),
        (di*ni+dj*nj).sum(-1),(cross(di,ni)-cross(dj,nj)).abs(),
        norm_i+norm_j,(norm_i-norm_j).abs()),-1).clamp(-3.,3.)
    return features*((parts>0).all(-1)[None]&(alpha!=0.)&(beta!=0.))[...,None]


def reactive_candidate_data(data):
    if 'candidate_interactions' in data['packet']:return data
    features=reactive_candidate_interactions(data['packet'],data['base'],data['options'],reactive_credit_parts(data))
    return dict(data,packet=dict(data['packet'],candidate_interactions=features))


def reactive_coordination_context(packet,base,programs,parts,prior):
    """Public, symmetric coordinates for a local cooperative value update."""
    features=reactive_candidate_interactions(packet,base,programs,parts)
    motion,obs=packet['motion'],packet['obs']
    angle=motion[:,0,2]+obs[:,0,3]
    target=motion[:,0,:2]+obs[:,0,2,None]*torch.stack((angle.cos(),angle.sin()),-1)
    heading=packet['target_heading'];bearing=torch.atan2(-target[:,1],-target[:,0])
    distance=obs[...,2];nearest=distance.amin(-1)
    state=torch.stack((target.norm(dim=-1)/60.,obs[:,0,4]/5.,
        (heading-bearing).cos(),(heading-bearing).sin(),nearest/20.,
        (distance.mean(-1)-nearest)/20.,packet.get('target_turn',torch.zeros_like(heading)),
        packet['time_remaining']),-1).clamp(-3.,3.)
    effect=prior-prior[:,:,:1]
    independent=effect[:,:,parts].sum(-1)
    mixed=effect-independent
    response=torch.stack((independent.mean(1)/20.,mixed.mean(1)/20.,
        effect.std(1,unbiased=False)/20.,mixed.std(1,unbiased=False)/20.),-1).clamp(-3.,3.)
    return torch.cat((features,state[:,None].expand(-1,len(parts),-1),response),-1)


def reactive_selected_coordination_context(packet,base,programs,parts,prior,bias=-.05):
    """Bind all selected candidate mixtures before forming a conditional value.

    A receiver can change its decision because of an unchanged teammate's
    current candidates. Cooperative messages therefore remain present when
    only one receiver departs from the reference mixture.
    """
    from policy.interaction_sac import CandidateCoordinationGate
    _,paths=CandidateCoordinationGate.candidate_responses(packet,base)
    reference=(base[...,2]+bias).clamp(0.,1.)
    gate=(reference[:,None]+programs).clamp(0.,1.)
    selected=paths[:,None,:,:,1]*(1.-gate[...,None,None])+paths[:,None,:,:,0]*gate[...,None,None]
    heading=packet['target_heading'];c=heading.cos();s=heading.sin()
    def rotate(value):
        shape=(len(c),)+(1,)*(value.ndim-2)
        x,y=value[...,0],value[...,1]
        return torch.stack((c.reshape(shape)*x+s.reshape(shape)*y,-s.reshape(shape)*x+c.reshape(shape)*y),-1)
    selected=rotate(selected)/20.
    endpoints=rotate(paths)/20.
    order=packet['obs'][...,2].argsort(dim=-1,stable=True)
    selected=selected.gather(2,order[:,None,:,None,None].expand_as(selected))
    endpoints=endpoints.gather(1,order[:,:,None,None,None].expand_as(endpoints))
    gate=gate.gather(2,order[:,None].expand_as(gate))
    difference=programs.gather(2,order[:,None].expand_as(programs))
    old=reactive_coordination_context(packet,base,programs,parts,prior)
    return torch.cat((old[...,12:],selected.flatten(2),endpoints.flatten(1)[:,None].expand(-1,programs.shape[1],-1),gate,difference),-1)


def reactive_selected_dictionary(specification):
    return any(block.get('coordinates_version') in ('selected-candidates-v1','candidate-edges-v1','candidate-edges-v2','candidate-edges-v3','candidate-edges-v4') for block in specification.get('blocks',[specification]))


def reactive_response_dictionary(specification):
    return any(block.get('coordinates_version')=='candidate-edges-v4' for block in specification.get('blocks',[specification]))


def reactive_edge_context(packet,base,programs,bias=-.05,global_context=False,joint_response=False):
    """Shared directed edge coordinates of the actual fused candidate thrusts.

    Integrate a candidate mixture before encoding its motion. Mixing two
    separately integrated endpoint paths is not the motion of their mixture
    under nonlinear vessel dynamics. Directed edges are pooled symmetrically,
    without a nearest-vessel ordering or vessel-specific learned parameters.
    """
    from policy.interaction_sac import CandidateCoordinationGate
    n=base.shape[1];gate=((base[...,2]+bias).clamp(0.,1.)[:,None]+programs).clamp(0.,1.)
    if joint_response:
        if 'candidate_joint_response' not in packet:raise ValueError('Coupled response features require the recorded public control-window response.')
        selected=packet['candidate_joint_response']
        if selected.shape!=(len(base),programs.shape[1],n,2,2):raise ValueError('Candidate response does not match the actual joint controls.')
    else:
        thrust=750.*(gate[...,None]*base[:,None,:,:2]+(1.-gate[...,None])*packet['obs'][:,None,:,12:14])+250.
        _,selected=CandidateCoordinationGate.candidate_responses(packet,base,applied_thrust=thrust.transpose(1,2).clamp(-500.,1000.))
        selected=selected.permute(0,3,1,2,4)
    _,candidates=CandidateCoordinationGate.candidate_responses(packet,base)
    heading=packet['target_heading'];c=heading.cos();s=heading.sin()
    def rotate(value):
        shape=(len(c),)+(1,)*(value.ndim-2);x,y=value[...,0],value[...,1]
        return torch.stack((c.reshape(shape)*x+s.reshape(shape)*y,-s.reshape(shape)*x+c.reshape(shape)*y),-1)/20.
    selected=rotate(selected).flatten(-2)
    span=rotate(candidates[:,:,:,1]-candidates[:,:,:,0]).flatten(-2)
    node=torch.cat((selected,span[:,None].expand(-1,len(programs[0]),-1,-1),gate[...,None],
        (packet['obs'][...,2]/20.)[:,None,:,None].expand(-1,len(programs[0]),-1,-1)),-1)
    edges=torch.tensor([(i,j) for i in range(n) for j in range(n) if i!=j],device=base.device)
    motion,obs=packet['motion'],packet['obs'];angle=motion[...,2]+obs[...,3]
    target=(motion[...,:2]+obs[...,2,None]*torch.stack((angle.cos(),angle.sin()),-1)).mean(1)
    bearing=torch.atan2(-target[:,1],-target[:,0])
    state=torch.stack((packet.get('target_turn',torch.zeros_like(heading)),obs[...,4].mean(-1)/5.,
        (heading-bearing).cos(),(heading-bearing).sin(),obs[...,2].amin(-1)/20.),-1)
    if global_context:
        state=torch.cat((state,target.norm(dim=-1,keepdim=True)/60.,packet['time_remaining'][:,None]),-1)
    return torch.cat((node[:,:,edges[:,0]],node[:,:,edges[:,1]],
        state[:,None,None].expand(-1,programs.shape[1],len(edges),-1)),-1).clamp(-4.,4.)


def reactive_edge_dictionary(data,max_centers=64,bias=-.05):
    """Label-free shared edge centers; all training joint controls are covered."""
    response='model_response' in data
    packet=dict(data['packet'],candidate_joint_response=data['model_response']) if response else data['packet']
    context=reactive_edge_context(packet,data['base'],data['options'],bias,True,response)
    points=context.flatten(0,2);scale=points.std(0).clamp_min(.15)
    normalized=points/scale;chosen=[0];nearest=(normalized-normalized[0]).square().mean(-1)
    for _ in range(1,min(max_centers,len(points))):
        index=int(nearest.argmax());chosen.append(index)
        nearest=torch.minimum(nearest,(normalized-normalized[index]).square().mean(-1))
    centers=points[chosen];distance=torch.cdist(centers/scale,centers/scale).square()/len(scale)
    distance.fill_diagonal_(float('inf'))
    bandwidth=float(distance.amin(-1).median().sqrt().clamp_min(.25))
    return dict(centers=centers.tolist(),scale=scale.tolist(),bandwidth=bandwidth,bias=bias,
        centered=True,coordinates_version='candidate-edges-v4' if response else 'candidate-edges-v3',response_features=3,
        coordinates='shared directed pairs of nonlinearly integrated current candidate mixtures')


def reactive_edge_moments(edges):
    """Continuous pair coverage and interference relative to the same gate."""
    left=edges[...,:4].reshape(*edges.shape[:3],2,2)
    right=edges[...,10:14].reshape_as(left)
    di=left.norm(dim=-1);dj=right.norm(dim=-1)
    dot=(left*right).sum(-1);separation=(left-right).norm(dim=-1)
    def soft_nearest(temperature):
        return -temperature*torch.logsumexp(torch.stack((-di/temperature,-dj/temperature),-1),-1)
    proximity=torch.exp(-2.*(di+dj))
    coverage=dot/(di*dj).clamp_min(1e-6)*proximity
    moments=torch.stack((dot,separation,soft_nearest(.1),soft_nearest(.4),coverage,proximity),-1).flatten(-2).mean(2)
    return moments-moments[:,:1]


def reactive_dictionary_support(programs,specification):
    blocks=specification.get('blocks',[specification])
    if any(block.get('centered',False) for block in blocks):
        if not all(block.get('centered',False) for block in blocks):
            raise ValueError('Do not mix centered and discontinuous cooperative residual coordinates.')
        return torch.ones(programs.shape[:2],dtype=programs.dtype,device=programs.device)
    return (programs.abs().amax(-1)>0.).float()


def reactive_local_candidate_features(packet,base,programs,parts,prior,specification):
    features=reactive_candidate_interactions(packet,base,programs,parts)
    context=reactive_coordination_context(packet,base,programs,parts,prior)
    blocks=specification.get('blocks',[specification]);kernels=[]
    for block in blocks:
        centers=context.new_tensor(block['centers']);scale=context.new_tensor(block['scale'])
        if block.get('coordinates_version') in ('candidate-edges-v1','candidate-edges-v2','candidate-edges-v3','candidate-edges-v4'):
            edges=reactive_edge_context(packet,base,programs,block['bias'],block['coordinates_version'] in ('candidate-edges-v3','candidate-edges-v4'),
                block['coordinates_version']=='candidate-edges-v4').to(prior)
            if block['coordinates_version']!='candidate-edges-v1':features=reactive_edge_moments(edges)
            distance=torch.cdist((edges/scale).flatten(0,2),centers/scale).square().reshape(*edges.shape[:3],len(centers))/len(scale)
            kernel=torch.exp(-distance/(2.*block['bandwidth']**2)).mean(2)
            kernels.append(kernel-kernel[:,:1])
            difference=prior-prior[:,:,:1];mixed=difference-difference[...,parts].sum(-1)
            response=torch.stack((mixed.mean(1),2.*(mixed/2.).tanh().mean(1),8.*(mixed/8.).tanh().mean(1)),-1)/10.
            kernels.append(response)
            continue
        coordinates=(reactive_selected_coordination_context(packet,base,programs,parts,prior,block['bias'])
            if block.get('coordinates_version')=='selected-candidates-v1' else context)
        distance=((coordinates[:,:,None]-centers[None,None])/scale).square().mean(-1)
        kernel=torch.exp(-distance/(2.*block['bandwidth']**2))
        if block.get('centered',False):
            if bool((programs[:,0]!=0.).any()):raise ValueError('Centered control kernels require a zero-increment reference.')
            kernel=kernel-kernel[:,:1]
        support=(programs.abs().amax(-1)>0. if block.get('coordinates_version')=='selected-candidates-v1'
            else ((parts>0).sum(-1)>=2)[None].expand(programs.shape[:2]))
        kernels.append(kernel*support[...,None])
    result=torch.cat((features,*kernels),-1)
    if 'feature_scale' in specification:result=result/result.new_tensor(specification['feature_scale'])
    return result


def reactive_local_candidate_data(data,prior,specification):
    packet=dict(data['packet'])
    if 'model_response' in data:packet['candidate_joint_response']=data['model_response']
    features=reactive_local_candidate_features(packet,data['base'],data['options'],reactive_credit_parts(data),prior,specification)
    packet=dict(packet,candidate_interactions=features)
    if reactive_selected_dictionary(specification):packet['candidate_residual_support']=reactive_dictionary_support(data['options'],specification)
    return dict(data,packet=packet)


def reactive_coordination_dictionary(data,prior,anchor,selected=False,bias=-.05,centered=False,max_centers=0):
    """Set local coordinates from training geometry, without reading returns."""
    parts=reactive_credit_parts(data)
    encode=(lambda p,b,u,d,q:reactive_selected_coordination_context(p,b,u,d,q,bias)) if selected else reactive_coordination_context
    context=encode(data['packet'],data['base'],data['options'],parts,prior)
    bank=anchor['data'];bank_parts=reactive_credit_parts(bank)
    reference=encode(bank['packet'],bank['base'],bank['options'],bank_parts,anchor['prior'])
    bank_mask=((bank['options'].abs().amax(-1)>0.) if selected else
        (bank_parts>0).all(-1)[None].expand(reference.shape[:2]))
    scale=reference[bank_mask].std(0).clamp_min(.15)
    evaluated=data['evaluation_options'];rows=torch.arange(len(evaluated))[:,None].expand_as(evaluated)
    mask=((data['options'][rows,evaluated].abs().amax(-1)>0.) if selected else (parts[evaluated]>0).sum(-1)>=2)
    centers=context[rows[mask],evaluated[mask]]
    origins=data['origin'][rows[mask]]
    if not len(centers):raise ValueError('Local cooperation needs evaluated joint controls.')
    if max_centers and len(centers)>max_centers:
        normalized=centers/scale;chosen=[0]
        nearest=(normalized-normalized[0]).square().mean(-1)
        for _ in range(1,max_centers):
            index=int(nearest.argmax());chosen.append(index)
            nearest=torch.minimum(nearest,(normalized-normalized[index]).square().mean(-1))
        centers=centers[chosen];origins=origins[chosen]
    nearest=[]
    for center,origin in zip(centers,origins):
        eligible=reference[bank_mask&(bank['origin']!=origin)[:,None]]
        nearest.append(((eligible-center)/scale).square().mean(-1).amin().sqrt())
    bandwidth=float(torch.stack(nearest).median().clamp_min(.25))
    return dict(centers=centers.tolist(),scale=scale.tolist(),bandwidth=bandwidth,bias=bias,
        centered=centered,
        coordinates_version='selected-candidates-v1' if selected else 'paired-increment-v1',
        coordinates='current pair candidates, public pursuit geometry and paired model responses')


def reactive_monotone_regularizer(weights):
    value=weights[...,:4][...,1:].square().mean()
    if weights.shape[-1]>4:value=value+weights[:,4,4:].square().mean()
    if weights.shape[1] in (8,9,10) and weights.shape[-1]>4:value=value+weights[:,7,4:].square().mean()
    if weights.shape[1]==10 and weights.shape[-1]>4:value=value+weights[:,8,4:].square().mean()
    return value


def reactive_monotone_estimates(coefficients,prior,parts,packet,interaction=True):
    """Learn a monotone control-value link and the weight of joint responses.

    The response-link component preserves the sign of consistently harmful
    modeled controls. A separately supervised optional residual can correct
    that model's error using actual conditional policy returns.
    """
    distance=packet['obs'][...,2].to(prior);nearest=distance.amin(-1)
    context=torch.stack((torch.ones_like(nearest),torch.exp(-nearest/15.),
        ((distance.mean(-1)-nearest)/20.).clamp(max=3.),
        packet.get('target_turn',torch.zeros_like(nearest)).to(prior).abs().clamp(max=3.)),-1)
    parameters=torch.einsum('bc,mhc->mbh',context,coefficients[...,:4].to(context))
    weights=parameters[...,:3].softmax(-1)
    scale=parameters[...,3].sigmoid()
    difference=prior-prior[...,:1]
    basis=torch.stack((difference,2.*(difference/2.).tanh(),8.*(difference/8.).tanh()),-1)
    value=torch.einsum('brkh,mbh->mbrk',basis,weights)*scale[:,:,None,None]
    unary=value[...,parts].sum(-1)
    if not interaction:return unary
    individual=difference[...,parts].sum(-1)
    # Joint trust depends on the current candidate interaction, and is
    # learned from its conditional control consequences. State-only return
    # shifts cancel from both this attention signal and the value link.
    strength=((difference-individual).abs().mean(1)/(2.+individual.abs().mean(1))).clamp(max=3.)
    coupling=parameters[...,4,None]
    if coefficients.shape[1]>5:coupling=coupling+parameters[...,5,None]*strength[None]
    if coefficients.shape[1]>6:
        signed=((difference-individual).mean(1)/(2.+individual.abs().mean(1))).clamp(-3.,3.)
        coupling=coupling+parameters[...,6,None]*signed[None]
    if coefficients.shape[-1]>4:
        features=packet['candidate_interactions'].to(context)
        coupling=coupling+torch.einsum('bkc,mc->mbk',features,coefficients[:,4,4:].to(context))
    coupling=F.softplus(coupling) if coefficients.shape[1]==10 else coupling.sigmoid()
    increment=value-unary
    if coefficients.shape[1]==10:
        # The public model is an imperfect response predictor. Its harmful
        # interactions can be overestimated, and beneficial ones understated.
        # Learn both links from actual conditional returns and current messages.
        positive=parameters[...,8,None]+parameters[...,9,None]*strength[None]
        if coefficients.shape[-1]>4:
            positive=positive+torch.einsum('bkc,mc->mbk',features,coefficients[:,8,4:].to(context))
        positive=F.softplus(positive)
        joint=unary+coupling[:,:,None,:]*increment.clamp(max=0.)+positive[:,:,None,:]*increment.clamp(min=0.)
    elif coefficients.shape[1]==9:
        # Candidate benefits cannot be counted independently while a modeled
        # interference penalty is discounted away. Learn separate responses
        # to beneficial synergy and harmful interference; the latter retains
        # at least the explicit joint-model penalty.
        positive=1.+F.softplus(parameters[...,8])
        joint=unary+coupling[:,:,None,:]*increment.clamp(max=0.)+positive[:,:,None,None]*increment.clamp(min=0.)
    else:joint=unary+coupling[:,:,None,:]*increment
    if coefficients.shape[1] in (8,9,10) and coefficients.shape[-1]>4:
        # Actual conditional returns can identify synergy that the nominal
        # response model missed, including an incorrect effect sign. This
        # learned model-error field is therefore not restricted to reweighting
        # the nominal interaction. Selected-candidate messages also identify
        # a single receiver's conditional marginal with its teammates held.
        features=packet['candidate_interactions'].to(context)
        support=(5.*(features[...,10]-features[...,11]).clamp_min(0.)).tanh()
        support=support*((parts>0).sum(-1)>=2)[None]
        if 'candidate_residual_support' in packet:support=packet['candidate_residual_support'].to(support)
        residual=torch.einsum('bkc,mc->mbk',features,coefficients[:,7,4:].to(context))*support[None]
        joint=joint+residual[:,:,None]
    return joint


def fit_prior_monotone(data,prior,conditional=True,interaction=True,seed=9970,members=5,signed_interaction=False,candidate_context=False,conservative=False,adaptive=False):
    if candidate_context:data=reactive_candidate_data(data)
    initial,excluded=fit_prior_calibration(data,prior,conditional,interaction,seed,members,True)
    parts=reactive_credit_parts(data);groups=reactive_state_groups(data);coefficients=[]
    for member in range(members):
        generator=torch.Generator().manual_seed(seed+member*104729)
        index,_=reactive_group_sample(groups,generator)
        forecast=prior[index].double();packet={k:v[index] for k,v in data['packet'].items()}
        costs=data['costs'][index].double()
        future=torch.randint(costs.shape[1],costs.shape[:2],generator=generator)
        costs=costs.gather(1,future[:,:,None].expand_as(costs));target=costs.mean(1)
        precision=reactive_precision(costs,True)
        distance=packet['obs'][...,2].double();reference=forecast.mean(1)[:,0]
        state=torch.stack((torch.ones_like(reference),reference/10.,distance.amin(-1)/10.,
            distance.mean(-1)/20.,packet['time_remaining'].double()),-1)
        with torch.enable_grad():
            weights=torch.zeros(1,10 if adaptive else 9 if conservative else 7 if signed_interaction else 6,16 if candidate_context else 4,dtype=torch.float64)
            if adaptive:weights[0,8,0]=torch.expm1(torch.tensor(1.,dtype=torch.float64)).log()
            elif conservative:weights[0,8,0]=-4.
            weights[0,:3,0]=torch.tensor([.9,.05,.05],dtype=torch.float64).log()
            weights[0,3,0]=torch.logit(initial[member,0].double().clamp(.02,.98))
            ratio=initial[member,1].double()/initial[member,0].double().clamp_min(.02)
            weights[0,4,0]=(ratio.clamp(.02,2.).expm1().log() if adaptive else torch.logit(ratio.clamp(.02,.98)))
            weights=nn.Parameter(weights)
            nuisance=nn.Parameter(torch.linalg.lstsq(state,target.mean(-1)).solution)
            optimizer=torch.optim.LBFGS([weights] if conditional else [weights,nuisance],max_iter=150,line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad(set_to_none=True)
                energy=reactive_monotone_estimates(weights,forecast,parts,packet,interaction)[0].mean(1)
                if conditional:
                    contrast=target-target[:,:1];error=energy[:,1:]-contrast[:,1:]
                    loss=torch.einsum('bk,bkl,bl->b',error,precision,error).mean()/error.shape[-1]
                    loss=loss+((-energy/2.).softmax(-1)*contrast).sum(-1).mean()/2.
                else:loss=((state@nuisance)[:,None]+energy-target).square().mean()/100.
                loss=loss+.01*reactive_monotone_regularizer(weights)
                loss.backward();return loss
            optimizer.step(closure);coefficients.append(weights[0].detach().float())
    return torch.stack(coefficients),excluded


def fit_prior_orthogonal(data,prior,conditional=True,interaction=True,seed=9970,members=5,candidate_context=False,conservative=False,adaptive=False):
    """Identify unary controls first; then learn only their excess joint value.

    Joint-return error cannot rewrite the individually learned controls.
    The conditional additive ablation has exactly the same unary estimator;
    the full model adds a separately supervised cooperative mirror update.
    """
    if candidate_context:data=reactive_candidate_data(data)
    if not conditional:return fit_prior_monotone(data,prior,False,interaction,seed,members,True,candidate_context,conservative,adaptive)
    parts=reactive_credit_parts(data);groups=reactive_state_groups(data);fitted=[]
    index_options=torch.arange(len(parts))
    singles=((parts==index_options[:,None]).any(-1)&(index_options>0)).nonzero().flatten()
    joints=((parts>0).all(-1)).nonzero().flatten()
    single_ids=torch.cat((torch.tensor([0]),singles));joint_ids=torch.cat((torch.tensor([0]),joints))
    unary_data=dict(data,costs=data['costs'][:,:,single_ids],
        parts=torch.stack((torch.arange(len(single_ids)),torch.zeros(len(single_ids),dtype=torch.long)),-1))
    initial,excluded=fit_prior_calibration(unary_data,prior[:,:,single_ids],True,False,seed,members,True)
    for member in range(members):
        generator=torch.Generator().manual_seed(seed+member*104729)
        index,_=reactive_group_sample(groups,generator)
        packet={k:v[index] for k,v in data['packet'].items()};forecast=prior[index].double()
        costs=data['costs'][index].double();future=torch.randint(costs.shape[1],costs.shape[:2],generator=generator)
        costs=costs.gather(1,future[:,:,None].expand_as(costs));target=costs.mean(1)
        unary_truth=target[:,parts[:,0]]+target[:,parts[:,1]]-2.*target[:,:1]
        mixed=costs-costs[:,:,parts[:,0]]-costs[:,:,parts[:,1]]+costs[:,:,:1]
        mixed_truth=mixed.mean(1)
        unary_precision=reactive_precision(costs[:,:,single_ids],True)
        mixed_precision=reactive_precision(mixed[:,:,joint_ids],True)
        with torch.enable_grad():
            weights=torch.zeros(1,10 if adaptive else 9 if conservative else 7,16 if candidate_context else 4,dtype=torch.float64)
            if adaptive:weights[0,8,0]=torch.expm1(torch.tensor(1.,dtype=torch.float64)).log()
            elif conservative:weights[0,8,0]=-4.
            weights[0,:3,0]=torch.tensor([.9,.05,.05],dtype=torch.float64).log()
            weights[0,3,0]=torch.logit(initial[member,0].double().clamp(.02,.98))
            unary=nn.Parameter(weights[:,:4].clone());joint=weights[:,4:].clone()
            optimizer=torch.optim.LBFGS([unary],max_iter=150,line_search_fn='strong_wolfe')
            def unary_closure():
                optimizer.zero_grad(set_to_none=True)
                value=reactive_monotone_estimates(torch.cat((unary,joint),1),forecast,parts,packet,False)[0].mean(1)
                error=value[:,singles]-unary_truth[:,singles]
                loss=torch.einsum('bk,bkl,bl->b',error,unary_precision,error).mean()/len(singles)
                loss=loss+((-value/2.).softmax(-1)*unary_truth).sum(-1).mean()/2.
                loss=loss+.01*unary[...,1:4].square().mean()
                loss.backward();return loss
            optimizer.step(unary_closure);unary=unary.detach()
            if interaction:
                joint=nn.Parameter(joint)
                independent=reactive_monotone_estimates(torch.cat((unary,joint.detach()),1),forecast,parts,packet,False)[0].mean(1)
                reference_log_policy=(-independent/2.).log_softmax(-1)
                optimizer=torch.optim.LBFGS([joint],max_iter=150,line_search_fn='strong_wolfe')
                def joint_closure():
                    optimizer.zero_grad(set_to_none=True)
                    value=reactive_monotone_estimates(torch.cat((unary,joint),1),forecast,parts,packet,True)[0].mean(1)
                    error=(value-independent-mixed_truth)[:,joints]
                    loss=torch.einsum('bk,bkl,bl->b',error,mixed_precision,error).mean()/len(joints)
                    proposed=(-value/2.).log_softmax(-1)
                    # Exact conditional mirror objective: the unary policy is
                    # the reference, and only measured excess interaction
                    # cost supplies the incremental policy advantage.
                    loss=loss+(proposed.exp()*(mixed_truth/2.+proposed-reference_log_policy)).sum(-1).mean()
                    penalized=joint[...,:4].clone()
                    if adaptive:penalized[0,4,0]-=torch.expm1(torch.tensor(1.,dtype=torch.float64)).log()
                    elif conservative:penalized[0,4,0]+=4.
                    penalty=penalized.flatten()
                    if candidate_context:
                        penalty=torch.cat((penalty,joint[0,0,4:]))
                        if conservative or adaptive:penalty=torch.cat((penalty,joint[0,3,4:]))
                        if adaptive:penalty=torch.cat((penalty,joint[0,4,4:]))
                    loss=loss+.01*penalty.square().mean()
                    loss.backward();return loss
                optimizer.step(joint_closure);joint=joint.detach()
            fitted.append(torch.cat((unary,joint),1)[0].float())
    return torch.stack(fitted),excluded


def fit_prior_candidate(data,prior,conditional=True,interaction=True,seed=9970,members=5):
    return fit_prior_orthogonal(data,prior,conditional,interaction,seed,members,True)


def fit_prior_conservative(data,prior,conditional=True,interaction=True,seed=9970,members=5,adaptive=False):
    return fit_prior_orthogonal(data,prior,conditional,interaction,seed,members,True,True,adaptive)


def reactive_positive_credit_weights(groups,repeats,generator):
    """Bayesian bootstrap keeps every newly evaluated control in every head."""
    counts=torch.bincount(groups)
    origin=-torch.rand(len(counts),generator=generator,dtype=torch.float64).clamp_min(1e-12).log()
    rows=origin[groups]/counts[groups];rows=rows/rows.mean()
    futures=-torch.rand(len(groups),repeats,generator=generator,dtype=torch.float64).clamp_min(1e-12).log()
    return rows,futures/futures.sum(-1,keepdim=True)


def update_prior_continuation(coefficients,data,prior,anchor,conditional=True,interaction=True,seed=9978,joint_only=False,local_start=None,paired_policy=False,anchor_strength=1.,regularization=.01,policy_settings=None,unary_reference=None,mixed_supervision=False,decision_focused=False):
    """Evaluate recurrent controls while retaining a functional value prior.

    Active panels directly identify the current continuation's conditional
    value. They do not assume the old critic's action ranking was accurate.
    Legacy complete panels also support paired continuation corrections;
    both modes regularize unsupported changes against the old value field.
    """
    if 'reference_costs' not in data:raise ValueError('A paired reference continuation is required.')
    if joint_only and not interaction:
        return coefficients.clone(),dict(control_value_change_rms=[0.]*len(coefficients),states=len(data['costs']),
            target='joint-only improvement; the additive value is held fixed')
    if coefficients.shape[-1]>4:
        data=reactive_candidate_data(data);anchor=dict(anchor,data=reactive_candidate_data(anchor['data']))
    parts=reactive_credit_parts(data);groups=reactive_state_groups(data)
    anchor_parts=reactive_credit_parts(anchor['data']);anchor_prior=anchor['prior'].double()
    anchor_packet=anchor['data']['packet']
    previous=reactive_monotone_estimates(coefficients.double(),anchor_prior,anchor_parts,anchor_packet,interaction).mean(2)
    variance=1.+previous.var(0,unbiased=False)
    fitted=[];changes=[]
    for member in range(len(coefficients)):
        generator=torch.Generator().manual_seed(seed+member*104729)
        active_panel='evaluation_options' in data
        if active_panel:
            index=torch.arange(len(groups))
            row_weight,future_weight=reactive_positive_credit_weights(groups,data['costs'].shape[1],generator)
        else:
            index,_=reactive_group_sample(groups,generator)
            row_weight=torch.ones(len(index),dtype=torch.float64)
            future_weight=torch.full((len(index),data['costs'].shape[1]),1./data['costs'].shape[1],dtype=torch.float64)
        packet={k:v[index] for k,v in data['packet'].items()};forecast=prior[index].double()
        costs=data['costs'][index].double();reference=data['reference_costs'][index].double()
        if not active_panel:
            futures=torch.randint(costs.shape[1],costs.shape[:2],generator=generator)
            futures=futures[:,:,None].expand_as(costs)
            costs=costs.gather(1,futures);reference=reference.gather(1,futures)
        difference=costs-reference
        correction=(difference*future_weight[:,:,None]).sum(1);correction=correction-correction[:,:1]
        precision=reactive_precision(costs if active_panel else difference,True)
        absolute_precision=1./(costs.var(1,unbiased=costs.shape[1]>1).mean(-1)/costs.shape[1]+1.)
        original=coefficients[member:member+1].double()
        evaluated=data.get('evaluation_options')
        evaluated=evaluated[index] if evaluated is not None else None
        def predict(weights):
            value=reactive_monotone_estimates(weights,forecast,parts,packet,interaction)[0].mean(1)
            return value.gather(-1,evaluated) if evaluated is not None else value
        old=predict(original)
        old_log_policy=(-old/2.).log_softmax(-1)
        distance=packet['obs'][...,2].double();baseline=forecast.mean(1)[:,0]
        state=torch.stack((torch.ones_like(baseline),baseline/10.,distance.amin(-1)/10.,
            distance.mean(-1)/20.,packet['time_remaining'].double()),-1)
        target=(costs*future_weight[:,:,None]).sum(1)
        policy_penalty=torch.zeros_like(target);policy_support=None
        if policy_settings is not None:
            if unary_reference is None:raise ValueError('Continuous cooperation requires the same learned unary reference at training and execution.')
            reference_saved=dict(calibration=original,unary_reference=unary_reference,settings=policy_settings)
            programs=data['options'][index]
            _,reference_choice,_=reactive_credit_unary(reference_saved,packet,programs,parts,forecast)
            reference_program=programs[torch.arange(len(programs)),reference_choice]
            full_penalty=policy_settings['prior']*(programs-reference_program[:,None]).square().mean(-1)
            full_support=(programs.abs()>1e-7).sum(-1)<=policy_settings['max_coordinated_receivers']
            policy_penalty=full_penalty.gather(-1,evaluated) if evaluated is not None else full_penalty
            policy_support=full_support.gather(-1,evaluated) if evaluated is not None else full_support
            measured=programs.gather(1,evaluated[:,:,None].expand(-1,-1,programs.shape[-1])) if evaluated is not None else programs
            unique=(reactive_unique_control_mask(measured) if policy_settings.get('unique_control_training',False)
                else torch.ones_like(policy_support))
            policy_support=policy_support&unique
        else:unique=torch.ones_like(target,dtype=torch.bool)
        paired_regret=reactive_pair_regret_weights(costs+policy_penalty[:,None],future_weight) if paired_policy and conditional else None
        mixed_parts=None
        if mixed_supervision:
            if mixed_supervision=='active-face':
                if evaluated is None:raise ValueError('A recorded closed intervention face is required.')
                programs=data['options'][index]
                face=programs.gather(1,evaluated[:,:4,None].expand(-1,-1,programs.shape[-1]))
                torch.testing.assert_close(face[:,0]+face[:,3],face[:,1]+face[:,2],rtol=0.,atol=1e-6)
                mixed_parts=torch.tensor([[1,2]]);mixed_indices=torch.tensor([3])
            else:
                if evaluated is None or not torch.equal(evaluated,torch.arange(len(parts))[None].expand_as(evaluated)):
                    raise ValueError('Mixed supervision requires a complete control face.')
                mixed_parts=parts[(parts>0).sum(-1)==2]
                mixed_indices=((parts>0).sum(-1)==2).nonzero().flatten()
            mixed_cost=costs[:,:,mixed_indices]-costs[:,:,mixed_parts].sum(-1)+costs[:,:,:1]
            mixed_target=(mixed_cost*future_weight[:,:,None]).sum(1)
            mixed_variance=1.+mixed_cost.var(1,unbiased=costs.shape[1]>1)/costs.shape[1]
            mixed_support=unique[:,mixed_indices]&unique[:,mixed_parts].all(-1)
        with torch.enable_grad():
            weights=nn.Parameter(original.clone())
            if joint_only:
                mask=torch.zeros_like(weights);mask[:,4:]=1.
                if local_start is not None:
                    mask.zero_();mask[:,7 if weights.shape[1]>7 else 4,local_start:]=1.
                weights.register_hook(lambda gradient:gradient*mask)
            nuisance=nn.Parameter(torch.linalg.lstsq(state,(target-old).mean(-1)).solution)
            optimizer=torch.optim.LBFGS([weights] if conditional else [weights,nuisance],max_iter=100,line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad(set_to_none=True)
                energy=predict(weights)
                if conditional:
                    if evaluated is not None:
                        # The prior critic is approximate. Adding a measured
                        # continuation change to it would preserve its old
                        # action-ranking bias. Active credits instead fit the
                        # actual deployed continuation's paired control values;
                        # the old critic supplies a functional regularizer only.
                        error=(energy-energy[:,:1]-target+target[:,:1])[:,1:]
                    else:
                        change=energy-old
                        error=(change-change[:,:1]-correction)[:,1:]
                    if paired_policy:
                        # Pairwise expected lost time is the decision target.
                        # Shared state returns cancel before supervision, and
                        # rare expensive futures keep their real cost weight.
                        loss=(reactive_decision_regret_loss(energy+policy_penalty,target+policy_penalty,row_weight,policy_support)
                            if decision_focused else reactive_pair_regret_loss(energy+policy_penalty,paired_regret,row_weight,support=policy_support))
                        residual=energy-target
                        centered_error=residual-(residual*unique).sum(-1,keepdim=True)/unique.sum(-1,keepdim=True).clamp_min(1)
                        loss=loss+.1*(row_weight*(centered_error.square()*unique).sum(-1)/unique.sum(-1).clamp_min(1)).mean()
                        if mixed_parts is not None:
                            mixed_energy=energy[:,mixed_indices]-energy[:,mixed_parts].sum(-1)+energy[:,:1]
                            loss=loss+(row_weight*((mixed_energy-mixed_target).square()/mixed_variance*mixed_support).sum(-1)/mixed_support.sum(-1).clamp_min(1)).mean()
                    else:
                        loss=(row_weight*torch.einsum('bk,bkl,bl->b',error,precision,error)).mean()/error.shape[-1]
                        regret=target-target[:,:1]
                        log_policy=(-energy/2.).log_softmax(-1)
                        loss=loss+(row_weight*(log_policy.exp()*(regret/2.+log_policy-old_log_policy)).sum(-1)).mean()
                else:
                    # The absolute ablation receives its own return-noise
                    # scaling, rather than an arbitrary weak loss coefficient.
                    scale=absolute_precision if active_panel else torch.full_like(row_weight,.01)
                    error=((state@nuisance)[:,None]+energy-target).square()
                    loss=(row_weight*scale*(error*unique).sum(-1)/unique.sum(-1).clamp_min(1)).mean()
                if anchor_strength:
                    retained=reactive_monotone_estimates(weights,anchor_prior,anchor_parts,anchor_packet,interaction)[0].mean(1)
                    loss=loss+anchor_strength*((retained-previous[member]).square()/variance).mean()
                difference_weights=weights-original
                if not anchor_strength and local_start is not None:
                    active_heads=[4,7]+([8] if weights.shape[1]==10 else [])
                    penalty=difference_weights[:,:,:local_start].square().mean()+difference_weights[:,active_heads,local_start:].square().mean()
                else:penalty=difference_weights.square().mean()
                loss=loss+regularization*penalty
                loss.backward();return loss
            optimizer.step(closure)
            after=predict(weights)
            changes.append(float((after-old).square().mean().sqrt()))
            fitted.append(weights[0].detach().float())
    return torch.stack(fitted),dict(control_value_change_rms=changes,states=len(data['costs']),
        bootstrap=('positive origin and paired-future Bayesian weights' if 'evaluation_options' in data else 'resampled origins and futures'),
        target=('expected paired decision regret and calibrated control-value contrasts' if conditional and paired_policy else
            'actual paired control return under the recurrent continuation' if conditional and 'evaluation_options' in data else
            'paired change of conditional return under the recurrent continuation' if conditional else
            'ordinary return regression with a fitted state baseline'))


def reactive_prior_estimates(coefficients,prior,parts,packet=None,interaction=True):
    if coefficients.ndim==4:
        estimate=reactive_reliability_estimates(coefficients,prior,parts,packet,interaction)
    elif coefficients.ndim==3 and coefficients.shape[1] in (5,6,7,8,9,10) and coefficients.shape[2]>=4:
        estimate=reactive_monotone_estimates(coefficients,prior,parts,packet,interaction)
    elif coefficients.ndim==3:
        features=reactive_distribution_features(packet,prior,parts)
        if not interaction:features[:,:,1]=0.
        estimate=torch.einsum('bkdc,mdc->mbk',features,coefficients.to(features))[:,:,None]
    else:
        features=reactive_prior_features(prior,parts)
        estimate=torch.einsum('brkd,md->mbrk',features,coefficients.to(features))
    return estimate


def reactive_prior_scores(coefficients,prior,parts,risk,packet=None,interaction=True):
    estimate=reactive_prior_estimates(coefficients,prior,parts,packet,interaction)
    means=estimate.mean(2);mean=means.mean(0)
    uncertainty=(means.var(0,unbiased=False)+estimate.var(2,unbiased=False).mean(0)/prior.shape[1]).sqrt()
    return mean+float(risk)*uncertainty


def reactive_cooperative_scores(coefficients,prior,parts,risk,packet,interaction=True,robust=False):
    """A paired improvement certificate relative to the learned unary gate.

    Shared forecast and parameter errors cancel before uncertainty is
    computed. The current unary decision has exactly zero advantage and
    uncertainty, so an unsupported joint change cannot displace it merely
    because that reference has a large absolute return variance.
    """
    independent=reactive_prior_estimates(coefficients,prior,parts,packet,False)
    estimate=(reactive_prior_estimates(coefficients,prior,parts,packet,True)
        if interaction else independent)
    return reactive_paired_improvement(estimate,independent,risk,interaction,robust)


def reactive_paired_improvement(estimate,independent,risk,interaction=True,robust=False):
    reference=independent.mean((0,2)).argmin(-1)
    return reactive_anchored_value_scores(estimate,reference,risk,interaction,robust),reference


def reactive_anchored_value_scores(estimate,reference,risk,interaction=True,robust=False):
    reference_values=estimate.gather(-1,reference[None,:,None,None].expand(*estimate.shape[:-1],1))
    contrast=estimate-reference_values
    means=contrast.mean(2)
    if robust and interaction:
        # Robustness concerns approximate value models, not environment
        # outcomes. Every head first retains its full expected future cost,
        # including expensive failures; only then are paired heads aggregated.
        location=means.quantile(.5,dim=0)
        deviation=1.4826*(means-location[None]).abs().quantile(.5,dim=0)
        return location+float(risk)*deviation
    # The model's futures are generated as independent antithetic pairs.
    # Treat each pair mean as one Monte Carlo observation.
    futures=contrast
    if futures.shape[2]>1:
        futures=futures.reshape(*futures.shape[:2],-1,2,futures.shape[-1]).mean(3)
    uncertainty=(means.var(0,unbiased=False)+futures.var(2,unbiased=False).mean(0)/futures.shape[2]).sqrt()
    return means.mean(0)+(float(risk)*uncertainty if interaction else 0.)


def reactive_mirror_value_update(previous_scores,previous_values,current_values,risk,interaction=True,robust=False,reference=None):
    """Update the current parent's decision with a learned conditional correction.

    The previous decision energy is the mirror prior. An unchanged value
    field gives exactly the previous control; state-only value shifts cancel.
    All inputs are recomputed at today's state, rather than stored at trigger.
    """
    if reference is None:reference=previous_scores.argmin(-1)
    correction=current_values-previous_values
    delta=reactive_anchored_value_scores(correction,reference,risk,interaction,robust)
    score=previous_scores-previous_scores.gather(-1,reference[:,None])+delta
    choice=score.argmin(-1)
    choice=torch.where(score.gather(-1,choice[:,None])[:,0]<-1e-9,choice,reference)
    return score,choice


def reactive_conditional_posterior_update(current_values,programs,reference,risk,trust=.3,interaction=True):
    """Improve the fitted conditional mean, with paired parameter uncertainty.

    The nominal futures are fixed predictor inputs. Training fits their mean;
    executing a second penalty on their untrained dispersion would change
    that target. Bayesian heads represent real paired-return uncertainty.
    The reference is the preceding gate's freshly recomputed joint choice.
    """
    # Ablations change the value representation, not this confidence rule.
    score=reactive_anchored_value_scores(current_values.mean(2,keepdim=True),reference,risk,True)
    held=programs[torch.arange(len(programs)),reference]
    score=score+float(trust)*(programs-held[:,None]).square().mean(-1)
    choice=score.argmin(-1)
    choice=torch.where(score.gather(-1,choice[:,None])[:,0]<-1e-9,choice,reference)
    return score,choice


def reactive_fleet_factors(estimate,parts,receivers):
    """Decode simultaneous fleet controls from identifiable pair factors.

    Every measured zero, single and pair value is preserved exactly. Joint
    controls beyond two changed receivers use the learned pair factorization,
    rather than an artificial restriction on which boats may cooperate.
    """
    expected=1+2*receivers+2*receivers*(receivers-1)
    if len(parts)!=expected:raise ValueError('Fleet decoding requires all single and pair interventions.')
    device=estimate.device
    digits=(torch.arange(3**receivers,device=device)[:,None]//
        3**torch.arange(receivers,device=device)[None])%3
    singles=torch.where(digits>0,1+2*torch.arange(receivers,device=device)[None]+digits-1,0)
    weights=estimate.new_zeros(len(digits),len(parts))
    weights.scatter_add_(1,singles,torch.ones_like(singles,dtype=weights.dtype))
    for k,(i,j) in enumerate(parts.tolist()):
        if i and j:
            active=(singles==i).any(-1)&(singles==j).any(-1)
            weights[active,k]+=1.;weights[active,i]-=1.;weights[active,j]-=1.
    weights[:,0]=-weights[:,1:].sum(-1)
    return torch.einsum('...k,ak->...a',estimate,weights)


def calibrate_cooperative_update(coefficients,excluded,data,prior,interaction=True):
    parts=reactive_credit_parts(data);eligible=(excluded.sum(0)>=2).nonzero().flatten()
    if not interaction:return 0.,dict(eligible=len(eligible),skipped=True)
    if len(eligible)<12:return 1.,dict(eligible=len(eligible),skipped=True)
    truth=data['costs'].mean(1);rows=[]
    for risk in (2.,1.,.5,.25,0.):
        gains=[];changes=0
        for i in eligible:
            packet={k:v[i:i+1] for k,v in data['packet'].items()}
            score,reference=reactive_cooperative_scores(coefficients[excluded[:,i]],prior[i:i+1],parts,risk,packet)
            choice=score[0].argmin();changes+=int(choice!=reference[0])
            gains.append(truth[i,reference[0]]-truth[i,choice])
        gains=torch.stack(gains)
        rows.append(dict(risk=risk,gain=float(gains.mean()),standard_error=float(gains.std()/len(gains)**.5),changes=changes))
    best=max(rows,key=lambda r:r['gain']-.5*r['standard_error'])
    return best['risk'],dict(eligible=len(eligible),skipped=False,choices=rows,
        reference='out-of-origin learned unary decision')


def calibrate_prior_risk(coefficients,excluded,data,prior,step=.3,regularization=.3,interaction=True,risks=(2.,1.,.5,0.)):
    parts=reactive_credit_parts(data)
    eligible=(excluded.sum(0)>=2).nonzero().flatten()
    if len(eligible)<12:return min(1.,max(risks)),dict(eligible=len(eligible),skipped=True)
    truth=data['costs'].mean(1);rows=[]
    for risk in risks:
        gains=[]
        for i in eligible:
            packet={k:v[i:i+1] for k,v in data['packet'].items()}
            score=reactive_prior_scores(coefficients[excluded[:,i]],prior[i:i+1],parts,risk,packet,interaction)[0]
            choice=(step*score+regularization*data['options'][i].square().sum(-1)).argmin()
            gains.append(truth[i,0]-truth[i,choice])
        gains=torch.stack(gains)
        rows.append(dict(risk=risk,gain=float(gains.mean()),standard_error=float(gains.std()/len(gains)**.5)))
    best=max(rows,key=lambda r:r['gain']-.5*r['standard_error'])
    return best['risk'],dict(eligible=len(eligible),skipped=False,choices=rows)


class ReactiveAttentionValue(nn.Module):
    """Candidate messages produce a joint control value, without online rollouts.

    A shared attention field changes each receiver's marginal from the current
    teammate candidates. Explicit mixed factors preserve joint decisions that
    cannot be recovered by selecting independent marginal minima.
    """
    def __init__(self,width=64,interaction=True,bias=-.05,reaction=False):
        super().__init__();self.width=width;self.interaction=interaction;self.bias=bias;self.reaction=reaction
        def net(inputs,outputs,zero=False):
            value=nn.Sequential(nn.Linear(inputs,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,outputs))
            if zero:nn.init.zeros_(value[-1].weight);nn.init.zeros_(value[-1].bias)
            return value
        self.node=net(54,width)
        self.query=nn.Linear(width,width,bias=False);self.key=nn.Linear(width,width,bias=False)
        self.geometry=net(16+24*reaction,1)
        self.unary=net(width,2,True)
        self.contextual=net(2*width,2,True)
        self.mixed=net(4*width+16+8+24*reaction,4,True)
        self.baseline=net(18,1)

    @staticmethod
    @torch.no_grad()
    def inputs(packet,base,bias=-.05,reaction=False):
        from policy.interaction_sac import CandidateCoordinationGate as Gate
        node,_,geometry,state=Gate.public_inputs(packet,base,True)
        # Vessel identifiers are not physical observations. Shared messages
        # must follow the vessels, rather than their array positions.
        node=node.clone();node[...,9]=0.
        geometry=torch.cat((geometry[...,:7],geometry[...,9:]),-1)
        state=torch.cat((state[...,:9],state[...,10:]),-1)
        response,paths=Gate.candidate_responses(packet,base)
        turn=packet.get('target_turn',torch.zeros(len(base),device=base.device))
        own=torch.cat((node,response,turn[:,None,None].expand(-1,base.shape[1],1)),-1)
        left=paths[:,:,None];right=paths[:,None]
        separation=(left[...,None,:]-right[...,None,:,:]).norm(dim=-1)/30.
        coverage=F.cosine_similarity(left[...,None,:],right[...,None,:,:],dim=-1,eps=1e-6)
        paired=torch.stack((separation,coverage),-1).flatten(3)
        # Pool both physical times; the individual response encoding retains
        # their order and signed direction in the receiving vessel's frame.
        paired=paired.reshape(*paired.shape[:3],2,8).mean(-2)
        result=dict(node=own.clamp(-5.,5.),geometry=geometry.clamp(-5.,5.),paired=paired.clamp(-5.,5.),
            state=state.clamp(-5.,5.),reference=(base[...,2]+bias).clamp(0.,1.))
        if reaction:result['reaction']=Gate.candidate_reactions(packet,paths).clamp(-5.,5.)
        return result

    def factors(self,inputs,probe='aligned',return_hidden=False):
        x=self.node(inputs['node']);n=x.shape[1]
        enabled=self.interaction and probe not in ('interaction-off','state-only')
        incoming=x.roll(1,1) if probe=='permuted' else x
        logits=torch.einsum('bih,bjh->bij',self.query(x),self.key(incoming))/self.width**.5
        geometry=torch.cat((inputs['geometry'],inputs['reaction']),-1) if self.reaction else inputs['geometry']
        logits=logits+self.geometry(geometry).squeeze(-1)
        mask=~torch.eye(n,dtype=torch.bool,device=x.device)
        attention=logits.masked_fill(~mask[None],-1e9).softmax(-1)
        message=torch.einsum('bij,bjh->bih',attention,incoming)*float(enabled)
        contextual=self.contextual[:-1](torch.cat((x,message),-1))
        unary=self.unary(x)+self.contextual[-1](contextual)*float(enabled)
        left=torch.cat((x,message),-1)[:,:,None].expand(-1,-1,n,-1)
        right=torch.cat((incoming,message.roll(1,1) if probe=='permuted' else message),-1)[:,None].expand(-1,n,-1,-1)
        pair=(left,right,inputs['geometry'],inputs['paired'])
        if self.reaction:pair=(*pair,inputs['reaction'])
        pair_hidden=self.mixed[:-1](torch.cat(pair,-1))
        mixed=self.mixed[-1](pair_hidden).reshape(-1,n,n,2,2)
        mixed=.5*(mixed+mixed.transpose(1,2).transpose(-1,-2))*mask[None,:,:,None,None]*float(enabled)
        state=self.baseline(inputs['state']).mean(1).squeeze(-1)
        factors=(state,unary,mixed,inputs['reference'])
        return (factors,(contextual,pair_hidden)) if return_hidden else factors

    def forward(self,inputs,programs,probe='aligned'):
        from policy.interaction_sac import CandidateCoordinationGate as Gate
        factors=self.factors(inputs,probe)
        return factors[0],Gate.energies(factors,programs)


def reactive_attention_unary(coefficients,prior,parts,packet):
    """Exact calibrated independent values from the single-control forecasts."""
    singles=1+2*packet['obs'].shape[1]
    values=reactive_monotone_estimates(coefficients.double(),prior[:,:,:singles].double(),parts[:singles],packet,False)
    return values[...,parts].sum(-1).mean((0,2))


@torch.no_grad()
def reactive_attention_readout(member,inputs,programs):
    """Exact linear cooperative readout and its current candidate design."""
    _,(node,pair)=member.factors(inputs,return_hidden=True)
    reference=inputs['reference'][:,None];n=reference.shape[-1]
    basis=torch.stack(((-programs/reference.clamp_min(1e-6)).clamp(0.,1.),
        (programs/(1.-reference).clamp_min(1e-6)).clamp(0.,1.)),-1)
    mask=~torch.eye(n,dtype=torch.bool,device=programs.device)
    unary=torch.einsum('bnf,bkna->bkaf',node,basis).flatten(-2)
    mixed=.5*torch.einsum('bijf,bkia,bkjc->bkacf',pair*mask[None,:,:,None],basis,basis).flatten(-3)
    mixed_bias=.5*torch.einsum('ij,bkia,bkjc->bkac',mask.to(programs),basis,basis).flatten(-2)
    design=torch.cat((unary,basis.sum(2),mixed,mixed_bias),-1)
    weights=torch.cat((member.contextual[-1].weight.flatten(),member.contextual[-1].bias,
        member.mixed[-1].weight.flatten(),member.mixed[-1].bias))
    return design,weights


@torch.no_grad()
def reactive_attention_set_readout(member,weights):
    offset=0
    for parameter in (member.contextual[-1].weight,member.contextual[-1].bias,member.mixed[-1].weight,member.mixed[-1].bias):
        length=parameter.numel();parameter.copy_(weights[offset:offset+length].reshape_as(parameter));offset+=length
    if offset!=len(weights):raise ValueError('The cooperative readout dimensions changed.')


def reactive_gaussian_readout(design,residual,precision,prior_design=None):
    """Paired GLS with an evidence-selected prior, using fitting data only."""
    design=design.double();residual=residual.double();precision=precision.double()
    dimension=design.shape[-1]
    information=torch.einsum('bki,bkl,blj->ij',design,precision,design)
    rhs=torch.einsum('bki,bkl,bl->i',design,precision,residual)
    energy=torch.einsum('bk,bkl,bl->',residual,precision,residual)
    identity=torch.eye(dimension,device=design.device,dtype=design.dtype)
    anchor=torch.zeros_like(identity)
    if prior_design is not None:
        prior_design=prior_design.double().flatten(0,-2)
        anchor=.05*len(design)*design.shape[1]/len(prior_design)*(prior_design.T@prior_design)
    trials=[];best=None
    for scale in (.3,1.,3.,10.):
        prior=identity/scale**2+anchor
        factor=torch.linalg.cholesky(prior+information)
        mean=torch.cholesky_solve(rhs[:,None],factor)[:,0]
        evidence=.5*(energy-rhs@mean+2.*factor.diag().log().sum()-torch.linalg.slogdet(prior)[1])
        trials.append(dict(prior_scale=scale,negative_log_evidence=float(evidence)))
        if best is None or evidence<best[0]:best=(evidence,mean,factor,scale)
    _,mean,factor,scale=best
    covariance=torch.cholesky_inverse(factor)
    return mean,covariance,dict(prior_scale=scale,trials=trials)


def fit_reactive_attention_posterior(args):
    """Update joint decisions with matched paired-return mean and covariance."""
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    if not bank.get('complete') or not isinstance(bank['collecting_policy'],dict):
        raise ValueError('A completed current-policy paired bank is required.')
    parent_path=Path(bank['collecting_policy']['path'])
    if hashlib.sha256(parent_path.read_bytes()).hexdigest()!=bank['collecting_policy']['sha256']:
        raise ValueError('The measured continuation changed.')
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    if parent['kind']!='reactive-attention-v1' or parent.get('unary_reference') is None:
        raise ValueError('The joint readout requires the calibrated independent link.')
    data=bank['data'];settings=bank['settings'];reaction=parent.get('reaction_messages',False)
    fitting=torch.isin(data['origin'],parent['training_origins']).nonzero().flatten()
    if len(fitting)<16:raise ValueError('At least sixteen fitting states are required.')
    data=reactive_subset(data,fitting)
    inputs=ReactiveAttentionValue.inputs(data['packet'],data['base'],settings['bias'],reaction)
    options=data['options'];unary=reactive_attention_unary(parent['unary_reference'],bank['prior'][fitting],data['parts'],data['packet'])
    if 'evaluation_options' in data:
        evaluated=data['evaluation_options']
        options=options.gather(1,evaluated[:,:,None].expand(-1,-1,options.shape[-1]))
        unary=unary.gather(1,evaluated)
    costs=data['costs'].double();truth=costs.mean(1)
    anchor=None
    if args.reactive_augment_data is not None:
        old=torch.load(args.reactive_augment_data,map_location='cpu',weights_only=False)['data']
        old=reactive_subset(old,torch.isin(old['origin'],parent['training_origins']).nonzero().flatten())
        anchor=(ReactiveAttentionValue.inputs(old['packet'],old['base'],settings['bias'],reaction),old['options'])
    for arm in args.evaluation_arms.split(','):
        if arm not in ('difference','absolute','additive'):raise ValueError('Unknown posterior ablation.')
        actor=reactive_prior_actor(parent);actor.load_state_dict(parent['actor'])
        posteriors=[];metrics=[]
        for member in actor.members:
            features,weights=reactive_attention_readout(member,inputs,options)
            old=features@weights;dimension=features.shape[-1]
            af=features if anchor is None else reactive_attention_readout(member,*anchor)[0]
            scale=af.flatten(0,1).std(0).clamp_min(.03)*dimension**.5
            normalized=features.double()/scale.double()
            af=af.double()/scale.double()
            if arm=='absolute':
                hidden=member.baseline[:-1](inputs['state']).mean(1)
                state_design=torch.cat((hidden,torch.ones(len(hidden),1)),-1)
                state_scale=state_design.std(0).clamp_min(.03)*state_design.shape[-1]**.5
                state_scale[-1]=1.
                state_design=state_design.double()/state_scale.double()
                state_weights=torch.cat((member.baseline[-1].weight.flatten(),member.baseline[-1].bias))
                state=member.baseline(inputs['state']).mean(1).squeeze(-1)
                design=torch.cat((normalized,state_design[:,None].expand(-1,options.shape[1],-1)),-1)
                residual=truth-unary-old.double()-state[:,None].double()
                precision=torch.diag_embed((1.+costs.var(1,unbiased=True)/costs.shape[1]).reciprocal())
                anchor_design=F.pad(af,(0,state_design.shape[-1]))
            else:
                design=normalized[:,1:]-normalized[:,:1]
                residual=truth[:,1:]-truth[:,:1]-(unary[:,1:]-unary[:,:1])-(old[:,1:]-old[:,:1]).double()
                precision=reactive_precision(costs,True)
                anchor_design=af
            update,covariance,metric=reactive_gaussian_readout(design,residual,precision,anchor_design)
            if arm=='absolute':
                changed=state_weights.double()+update[dimension:]/state_scale.double()
                member.baseline[-1].weight.copy_(changed[:-1][None]);member.baseline[-1].bias.copy_(changed[-1:])
            reactive_attention_set_readout(member,weights.double()+update[:dimension]/scale.double())
            factor=torch.linalg.cholesky(covariance[:dimension,:dimension]+torch.eye(dimension,dtype=torch.float64)*1e-10)
            posteriors.append(dict(scale=scale.float(),factor=factor.float()));metrics.append(metric)
        saved=dict(parent,arm=arm,seed=args.seed,conditional=arm!='absolute',interaction=arm!='additive',risk=2.,
            value_posterior=posteriors,posterior_metrics=metrics,
            current_training_origins=torch.unique(data['origin']),collecting_policy=bank['collecting_policy'],
            data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            selection=dict(criterion='training marginal likelihood; fixed two-standard-deviation control bound'),
            decision_rule='recurrent conditional joint mean plus paired posterior uncertainty')
        result=reactive_prior_actor(saved)
        result.load_state_dict(actor.state_dict(),strict=False)
        saved['actor']=result.state_dict()
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',saved)
        print(json.dumps(dict(arm=arm,fitting_states=len(fitting),posterior_metrics=metrics)),flush=True)


def reactive_attention_panel(programs,parts,score,choice,reference):
    """A closed pair face plus current, independent and alternative decisions."""
    panels=[];joint=((parts>0).sum(-1)==2).nonzero().flatten()
    for i in range(len(programs)):
        current=int(choice[i]);pair=current if current in joint else int(joint[score[i,joint].argmin()])
        face=[0,*parts[pair].tolist(),pair]
        proposals=[current,int(reference[i]),*score[i].argsort().tolist()]
        for candidate in proposals:
            if candidate not in face:face.append(candidate)
            if len(face)==7:break
        panels.append(face)
    return torch.tensor(panels,device=programs.device,dtype=torch.long)


class ReactiveAttentionGate(nn.Module):
    """Recompute the learned candidate-interaction decision at every boundary."""
    def __init__(self,settings,width=64,members=3,risk=0.,interaction=True,reaction=False,unary_reference=None,source=None,value_posterior=None):
        super().__init__();self.settings=settings;self.risk=risk;self.reaction=reaction
        self.members=nn.ModuleList([ReactiveAttentionValue(width,interaction,settings['bias'],reaction) for _ in range(members)])
        self.has_unary_reference=unary_reference is not None
        if self.has_unary_reference:self.register_buffer('unary_reference',unary_reference.float())
        object.__setattr__(self,'source',source)
        self.has_value_posterior=value_posterior is not None
        if self.has_value_posterior:
            for i,posterior in enumerate(value_posterior):
                self.register_buffer(f'value_scale_{i}',posterior['scale'].float())
                self.register_buffer(f'value_factor_{i}',posterior['factor'].float())

    @torch.no_grad()
    def decide(self,packet,base,settings,probe='aligned'):
        programs,_=reactive_prior_options(packet,base,settings)
        inputs=ReactiveAttentionValue.inputs(packet,base,settings['bias'],self.reaction)
        predictions=torch.stack([member(inputs,programs,probe)[1] for member in self.members])
        if self.has_unary_reference:
            _,parts=reactive_prior_options(packet,base,settings);singles=1+2*base.shape[1]
            public={k:v.cpu() for k,v in packet.items()};own_programs=programs[:,:singles].cpu()
            prior=reactive_model_prior(self.source,public,own_programs,settings,4)
            single=reactive_monotone_estimates(self.unary_reference.cpu().double(),prior.double(),parts[:singles].cpu(),public,False).mean((0,2))
            independent=single[:,parts.cpu()].sum(-1).to(base)
            own=torch.stack([member(inputs,programs,'interaction-off')[1] for member in self.members])
            predictions=independent[None]+predictions-own
            reference=(independent+settings['prior']*programs.square().mean(-1)).argmin(-1)
            delta=predictions-predictions.gather(-1,reference[None,:,None].expand(len(predictions),-1,1))
            variance=delta.var(0,unbiased=False)
            if self.has_value_posterior and self.members[0].interaction and probe=='aligned':
                uncertainty=[]
                for i,member in enumerate(self.members):
                    features,_=reactive_attention_readout(member,inputs,programs)
                    features=features-features[torch.arange(len(base),device=base.device),reference][:,None]
                    features=features/getattr(self,f'value_scale_{i}')
                    uncertainty.append((features@getattr(self,f'value_factor_{i}')).square().sum(-1))
                variance=variance+torch.stack(uncertainty).mean(0)
            score=delta.mean(0)+self.risk*variance.clamp_min(0.).sqrt()
            selected=programs[torch.arange(len(base),device=base.device),reference]
            score=score+settings['prior']*(programs-selected[:,None]).square().mean(-1)
            choice=score.argmin(-1)
            if not self.members[0].interaction or probe in ('interaction-off','state-only'):choice=reference
            self.last_prediction=dict(prior=prior.cpu().clone(),objective=score.cpu().clone())
            return choice,programs[torch.arange(len(base),device=base.device),choice]
        score=predictions.mean(0)+self.risk*predictions.std(0,unbiased=False)
        score=score+settings['prior']*programs.square().mean(-1)
        choice=score.argmin(-1)
        return choice,programs[torch.arange(len(base),device=base.device),choice]


class ReactiveModelPriorGate(nn.Module):
    """A learned, continuously replanned joint candidate-fusion gate.

    A public model supplies candidate responses. Learned individual and joint
    conditional-value calibration controls their use; the model prior itself
    is not presented as a measured return or as an already improved policy.
    """
    def __init__(self,settings,coefficients,risk=1.,repeats=4,source=None,step=.3,interaction=True,
                 history_calibrations=None,history_risks=None,history_initial_posterior=False,unary_reference=None,cooperation_enabled=True):
        super().__init__();self.settings=settings;self.repeats=repeats;self.risk=risk
        self.interaction=interaction
        self.collect_predictions=False
        self.register_buffer('calibration',torch.as_tensor(coefficients).float())
        self.register_buffer('coefficients',torch.tensor([step]))
        history_calibrations=[] if history_calibrations is None else history_calibrations
        self.history_count=len(history_calibrations)
        self.history_initial_posterior=history_initial_posterior
        self.cooperation_enabled=cooperation_enabled;self.has_unary_reference=unary_reference is not None
        if self.has_unary_reference:
            reference=torch.as_tensor(unary_reference).float()
            self.register_buffer('unary_reference',F.pad(reference,(0,self.calibration.shape[-1]-reference.shape[-1])))
        if self.history_count:
            if not settings.get('exact_joint_fleet',False) or not settings.get('posterior_update',False):
                raise ValueError('Recursive value corrections require the exact joint posterior gate.')
            for i,value in enumerate(history_calibrations):self.register_buffer(f'history_{i}',value.float())
            self.register_buffer('history_risks',torch.as_tensor(history_risks).float())
        self.blocks=nn.ModuleList([nn.Identity()])
        object.__setattr__(self,'source',source if source is not None else ReactiveSource('cuda'))

    @torch.no_grad()
    def decide(self,packet,base,settings,probe='aligned'):
        if probe not in ('aligned','interaction-off'):
            raise ValueError('The public model prior supports interaction-off, not a peer-information masking probe.')
        packet={k:v.cpu() for k,v in packet.items()};base=base.cpu()
        programs,parts=(reactive_joint_options(packet,base,settings) if settings.get('exact_joint_fleet',False)
            else reactive_prior_options(packet,base,settings))
        support=((programs.abs()>1e-7).sum(-1)<=int(settings['max_coordinated_receivers'])
            if 'max_coordinated_receivers' in settings else torch.ones(programs.shape[:2],dtype=torch.bool))
        needs_response=reactive_response_dictionary(settings.get('coordination_dictionary',{}))
        response=None
        if needs_response:
            prior,response=reactive_model_prior(self.source,packet,programs,settings,self.repeats,
                supported_only='max_coordinated_receivers' in settings and not self.collect_predictions,return_response=True)
            packet=dict(packet,candidate_joint_response=response)
        else:prior=(reactive_model_prior(self.source,packet,programs,settings,self.repeats,supported_only=True)
            if 'max_coordinated_receivers' in settings and not self.collect_predictions else
            reactive_model_prior(self.source,packet,programs,settings,self.repeats))
        coefficients=self.calibration.cpu().clone()
        if settings.get('canonical_model',False):prior=prior.double();coefficients=coefficients.double()
        interaction=self.interaction and probe!='interaction-off'
        if not interaction and coefficients.ndim==2:coefficients[:,1]=0.
        if coefficients.ndim==3 and coefficients.shape[1] in (5,6,7,8,9,10) and coefficients.shape[2]>4:
            features=(reactive_local_candidate_features(packet,base,programs,parts,prior,settings['coordination_dictionary'])
                if 'coordination_dictionary' in settings else reactive_candidate_interactions(packet,base,programs,parts))
            packet=dict(packet,candidate_interactions=features)
            if 'coordination_dictionary' in settings and reactive_selected_dictionary(settings['coordination_dictionary']):
                packet['candidate_residual_support']=reactive_dictionary_support(programs,settings['coordination_dictionary'])
        anchored_choice=None
        learned_reference=None;learned_reference_score=None
        if self.has_unary_reference:
            reference_value=reactive_monotone_estimates(self.unary_reference.to(coefficients),prior,parts,packet,False)
            learned_reference_score,_=reactive_conditional_posterior_update(reference_value,programs,
                torch.zeros(len(programs),dtype=torch.long),0.,settings['prior'],False)
            learned_reference_score=learned_reference_score.masked_fill(~support,float('inf'))
            learned_reference=learned_reference_score.argmin(-1)
        if settings.get('exact_joint_fleet',False):
            if coefficients.ndim!=3 or coefficients.shape[1] not in (5,6,7,8,9,10):
                raise ValueError('Exact joint response decoding requires the conditional nonlinear value link.')
            independent=reactive_monotone_estimates(coefficients,prior,parts,packet,False)
            estimate=reactive_monotone_estimates(coefficients,prior,parts,packet,interaction)
            score=reactive_paired_improvement(estimate,independent,self.risk,interaction,settings.get('robust_update',False))[0]
            continuous_reference=settings.get('continuous_cooperation_test',False)
            if continuous_reference and learned_reference is None:
                raise ValueError('Continuous cooperation requires a learned independent gate.')
            if settings.get('conditional_posterior_update',False) and (not self.history_count or continuous_reference):
                score,anchored_choice=reactive_conditional_posterior_update(estimate,programs,
                    torch.zeros(len(programs),dtype=torch.long) if learned_reference is None else learned_reference,
                    self.risk,settings['prior'],interaction)
            if self.history_count and not continuous_reference:
                previous=None;parent_score=None;parent_choice=None
                for i in range(self.history_count):
                    calibration=getattr(self,f'history_{i}').to(coefficients)
                    values=reactive_monotone_estimates(calibration,prior,parts,packet,interaction)
                    if i==0:
                        if self.history_initial_posterior:
                            parent_score,_=reactive_conditional_posterior_update(values,programs,
                                torch.zeros(len(programs),dtype=torch.long) if learned_reference is None else learned_reference,
                                float(self.history_risks[i]),settings['prior'],interaction)
                        else:
                            unary=reactive_monotone_estimates(calibration,prior,parts,packet,False)
                            parent_score=reactive_paired_improvement(values,unary,float(self.history_risks[i]),interaction,
                                settings.get('robust_update',False))[0]
                        parent_score=parent_score.masked_fill(~support,float('inf'))
                        parent_choice=parent_score.argmin(-1)
                    else:
                        if settings.get('conditional_posterior_update',False):
                            parent_score,parent_choice=reactive_conditional_posterior_update(values,programs,parent_choice,
                                float(self.history_risks[i]),settings['prior'],interaction)
                        else:
                            parent_score,parent_choice=reactive_mirror_value_update(parent_score,previous,values,
                                float(self.history_risks[i]),interaction,settings.get('robust_update',False),parent_choice)
                        parent_score=parent_score.masked_fill(~support,float('inf'))
                        parent_choice=parent_score.argmin(-1)
                    previous=values
                if settings.get('conditional_posterior_update',False):
                    score,anchored_choice=reactive_conditional_posterior_update(estimate,programs,parent_choice,
                        self.risk,settings['prior'],interaction)
                else:
                    score,anchored_choice=reactive_mirror_value_update(parent_score,previous,estimate,self.risk,
                        interaction,settings.get('robust_update',False),parent_choice)
        elif settings.get('joint_fleet',False):
            n=base.shape[1]
            independent=reactive_fleet_factors(reactive_prior_estimates(coefficients,prior,parts,packet,False),parts,n)
            estimate=(reactive_fleet_factors(reactive_prior_estimates(coefficients,prior,parts,packet,True),parts,n)
                if interaction else independent)
            score=reactive_paired_improvement(estimate,independent,self.risk,interaction,settings.get('robust_update',False))[0]
            programs=reactive_options(packet,True,settings.get('control_radius',30.),3,reactive_reference(base,settings))
        else:
            score=(reactive_cooperative_scores(coefficients,prior,parts,self.risk,packet,interaction,settings.get('robust_update',False))[0]
                if settings.get('cooperative_update',False) else
                reactive_prior_scores(coefficients,prior,parts,self.risk,packet,interaction))
        if learned_reference is not None and (not interaction or not self.cooperation_enabled):
            score=learned_reference_score;anchored_choice=learned_reference
        objective=(score if settings.get('posterior_update',False) else
            float(self.coefficients[0])*score+settings['prior']*programs.square().sum(-1))
        objective=objective.masked_fill(~support,float('inf'))
        choice=objective.argmin(-1) if anchored_choice is None else anchored_choice
        choice=torch.where(support.gather(-1,choice[:,None])[:,0],choice,objective.argmin(-1))
        self.last_prediction=dict(prior=prior.clone(),objective=objective.clone())
        if response is not None:self.last_prediction['response']=response.clone()
        return choice,programs[torch.arange(len(base)),choice]


def reactive_prior_actor(saved,source=None,interaction=None):
    if saved.get('kind')=='reactive-component-v1':
        return ReactiveComponentGate(saved['settings'],saved['calibration'],saved['risk'],
            saved['interaction'] if interaction is None else interaction,source)
    if saved.get('kind')=='reactive-attention-v1':
        return ReactiveAttentionGate(saved['settings'],saved['width'],saved['members'],saved['risk'],
            saved['interaction'] if interaction is None else interaction,saved.get('reaction_messages',False),saved.get('unary_reference'),source,saved.get('value_posterior'))
    return ReactiveModelPriorGate(saved['settings'],saved['calibration'],saved['risk'],saved['model_repeats'],source,
        interaction=saved.get('interaction',True) if interaction is None else interaction,
        history_calibrations=saved.get('history_calibrations'),history_risks=saved.get('history_risks'),
        history_initial_posterior=saved.get('history_initial_posterior',False),unary_reference=saved.get('unary_reference'),
        cooperation_enabled=saved.get('cooperation_enabled',True))


def fit_reactive_prior(args):
    """Fit the same conditional gate used by the deployment entry point."""
    torch.set_num_threads(2)
    if args.reactive_bootstrap_data is None:raise ValueError('Provide the paired prior data file.')
    saved=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    if not saved.get('complete',True):raise ValueError('Conditional policy evaluation has not completed.')
    count=saved['training_count'];data=reactive_subset(saved['data'],torch.arange(count))
    if args.prior_value in ('candidate','conservative'):data=reactive_candidate_data(data)
    if len(torch.unique(data['origin']))<2:raise ValueError('Multiple training origins are required.')
    if bool(torch.isin(data['origin'],saved['data']['origin'][count:]).any()):
        raise ValueError('Training and held states share a trajectory origin.')
    settings=dict(saved['settings'],posterior_update=True,cooperative_update=args.cooperative_update,joint_fleet=args.joint_fleet,
        exact_joint_fleet=args.exact_joint_fleet,canonical_model=args.canonical_model)
    fitting={'linear':fit_prior_calibration,'reliability':fit_prior_reliability,'distribution':fit_prior_distribution,'monotone':fit_prior_monotone,'orthogonal':fit_prior_orthogonal,'candidate':fit_prior_candidate,'conservative':fit_prior_conservative}[args.prior_value]
    source=ReactiveSource(args.source_device)
    for arm in args.evaluation_arms.split(','):
        if arm not in ('difference','absolute','additive','absolute_additive'):raise ValueError('Unknown prior ablation.')
        conditional=arm in ('difference','additive');interaction=arm in ('difference','absolute')
        fit_seed=args.seed if args.bootstrap_seed is None else args.bootstrap_seed
        calibration,excluded=fitting(data,saved['prior'][:count],conditional,interaction,fit_seed,args.critic_members)
        if args.cooperative_update:
            risk,checks=calibrate_cooperative_update(calibration,excluded,data,saved['prior'][:count],interaction)
        else:
            risk,checks=calibrate_prior_risk(calibration,excluded,data,saved['prior'][:count],1.,0.,interaction,
                (0.,) if args.prior_value in ('orthogonal','candidate') else (2.,1.,.5,0.))
        actor=ReactiveModelPriorGate(settings,calibration,risk,saved['prior'].shape[1],source,1.,interaction)
        result=dict(kind='reactive-model-prior-v1',actor=actor.state_dict(),settings=settings,arm=arm,
            seed=args.seed,bootstrap_seed=fit_seed,calibration=calibration,risk=risk,conditional=conditional,interaction=interaction,
            model_repeats=saved['prior'].shape[1],prior_value=args.prior_value,
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            training_origins=data['origin'],held_origins=saved['data']['origin'][count:],
            calibration_risk=checks,collecting_policy=saved.get('collecting_policy','source'),
            training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',result)
        print(json.dumps(dict(arm=arm,risk=risk,training_states=count,risk_calibration=checks)),flush=True)


def reactive_crossfit_cooperation(calibration,data,prior,anchor,settings,unary_reference,conditional,interaction,seed,field_only=False,regularization=.01,mixed_supervision=False,decision_focused=False):
    """Calibrate intervention using returns excluded from each critic update.

    The continuation is the exact common parent in every fold. Only controls
    with measured paired outcomes participate in this calibration; these are
    training cross-validation scores, never independent deployment results.
    """
    if 'evaluation_options' not in data:raise ValueError('Cross-fitting needs matched current-policy intervention panels.')
    groups=reactive_state_groups(data);origins=torch.unique(groups)
    if len(origins)<4:raise ValueError('At least four independent origins are required for cooperation calibration.')
    prediction=None;folds=[]
    for origin in origins:
        train=(groups!=origin).nonzero().flatten();held=(groups==origin).nonzero().flatten()
        fitted,metrics=update_prior_continuation(calibration,reactive_subset(data,train),prior[train],anchor,
            conditional,interaction,seed,joint_only=field_only,local_start=4 if field_only else None,
            paired_policy=True,policy_settings=settings,unary_reference=unary_reference,regularization=regularization,
            mixed_supervision=mixed_supervision,decision_focused=decision_focused)
        packet={k:v[held] for k,v in data['packet'].items()}
        values=reactive_monotone_estimates(fitted.double(),prior[held].double(),reactive_credit_parts(data),packet,interaction)
        if prediction is None:prediction=torch.empty((len(calibration),len(groups),*values.shape[2:]),dtype=values.dtype)
        prediction[:,held]=values
        folds.append(dict(origin=int(origin),training_origins=origins[origins!=origin].tolist(),**metrics))
        print(json.dumps(dict(cooperation_fold=int(origin),conditional=conditional,held_states=len(held))),flush=True)
    evaluated=data['evaluation_options'];programs=data['options']
    context=data['packet'];parts=reactive_credit_parts(data)
    _,reference,_=reactive_credit_unary(dict(settings=settings,calibration=calibration,unary_reference=unary_reference),
        context,programs,parts,prior.double())
    matching=evaluated==reference[:,None]
    if not matching.any(-1).all():raise ValueError('Every panel must contain the deployed learned independent reference.')
    reference_column=matching.to(torch.int64).argmax(-1)
    support=(programs.abs()>1e-7).sum(-1)<=settings['max_coordinated_receivers']
    measured=torch.zeros_like(support).scatter_(1,evaluated,True)&support
    costs=data['costs'].mean(1);rows=torch.arange(len(groups))
    reference_cost=costs[rows,reference_column]
    trials=[]
    for risk in (None,0.,.5,1.,2.,4.):
        if risk is None:choice=reference;columns=reference_column
        else:
            score,_=reactive_conditional_posterior_update(prediction,programs,reference,risk,settings['prior'],interaction)
            choice=score.masked_fill(~measured,float('inf')).argmin(-1)
            columns=(evaluated==choice[:,None]).to(torch.int64).argmax(-1)
        chosen=costs[rows,columns]
        group_gain=torch.stack([(reference_cost-chosen)[groups==origin].mean() for origin in origins])
        trials.append(dict(risk=risk,paired_cost=float(chosen.mean()),gain=float(group_gain.mean()),
            standard_error=float(group_gain.std(unbiased=True)/len(origins)**.5),active=float((choice!=reference).float().mean())))
    selected=min(trials,key=lambda row:(row['paired_cost'],row['active']))
    return selected,dict(trials=trials,folds=folds,origins=origins,held_predictions=prediction,
        target='measured current-parent continuation; leave-one-origin-out cooperation decisions')


def update_reactive_prior(args):
    torch.set_num_threads(2)
    if args.initial_seed is None or args.reactive_bootstrap_data is None or args.reactive_augment_data is None:
        raise ValueError('Provide the value prior, current-policy credits and source-policy anchor data.')
    current=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    anchor=torch.load(args.reactive_augment_data,map_location='cpu',weights_only=False)
    if not current.get('complete',False) or not anchor.get('complete',False):raise ValueError('Incomplete policy evaluation.')
    source=ReactiveSource(args.source_device)
    common_parent=getattr(args,'common_update_parent',None)
    first_arm=common_parent or args.evaluation_arms.split(',')[0]
    first_parent=torch.load(args.output/f'reactive-{first_arm}-{args.initial_seed}.pth',map_location='cpu',weights_only=False)
    previous_dictionary=first_parent['settings'].get('coordination_dictionary')
    specification=previous_dictionary;new_centers=0
    initializer=None;initialization=None
    if getattr(args,'value_initialization_seed',None) is not None:
        initialization_path=args.output/f'reactive-difference-{args.value_initialization_seed}.pth'
        initializer=torch.load(initialization_path,map_location='cpu',weights_only=False)
        if not first_parent.get('shared_edge_value',False) or not initializer.get('shared_edge_value',False):
            raise ValueError('Re-encoding is limited to the shared candidate-edge value family.')
        for key in ('period','bias','defenders','agility','max_coordinated_receivers'):
            if initializer['settings'][key]!=first_parent['settings'][key]:raise ValueError('Re-encoding cannot change the evaluated task or control family.')
        torch.testing.assert_close(initializer['unary_reference'][:,:4,:4],first_parent['unary_reference'][:,:4,:4],rtol=0.,atol=0.)
        if args.local_coordination:raise ValueError('Choose either a new coordinate system or a local dictionary extension.')
        specification=initializer['settings']['coordination_dictionary']
        if reactive_response_dictionary(specification):
            selected=torch.isin(anchor['data']['origin'],first_parent['training_origins']).nonzero().flatten()
            if not len(selected):raise ValueError('The coupled-response anchor must contain the original fitting trajectories.')
            anchor=dict(anchor,data=reactive_subset(anchor['data'],selected),prior=anchor['prior'][selected])
        initialization=dict(path=str(initialization_path),sha256=hashlib.sha256(initialization_path.read_bytes()).hexdigest(),
            collecting_policy=initializer['collecting_policy'])
    if args.value_residual and not args.local_coordination:
        raise ValueError('A free conditional model-error field requires local candidate coordinates.')
    if getattr(args,'selected_coordination',False) and not (args.local_coordination and args.value_residual):
        raise ValueError('Selected-candidate supervision requires the local cooperative residual field.')
    if args.local_coordination:
        if not args.mixed_only:raise ValueError('A local coordination update must freeze the existing unary field.')
        selected=getattr(args,'selected_coordination',False)
        if previous_dictionary is not None and selected!=reactive_selected_dictionary(previous_dictionary):
            raise ValueError('Retain the preceding cooperative value coordinates when extending its dictionary.')
        centered=getattr(args,'centered_coordination',False)
        if previous_dictionary is not None and any(b.get('centered',False)!=centered for b in previous_dictionary.get('blocks',[previous_dictionary])):
            raise ValueError('Retain the preceding control-kernel reference convention.')
        addition=reactive_coordination_dictionary(current['data'],current['prior'],anchor,selected,first_parent['settings']['bias'],
            centered,getattr(args,'coordination_centers',0))
        new_centers=len(addition['centers'])
        specification=(dict(blocks=[*previous_dictionary.get('blocks',[previous_dictionary]),addition])
            if previous_dictionary is not None else addition)
    if specification is not None:
        current=dict(current,data=reactive_local_candidate_data(current['data'],current['prior'],specification))
        anchor=dict(anchor,data=reactive_local_candidate_data(anchor['data'],anchor['prior'],specification))
    for arm in args.evaluation_arms.split(','):
        if common_parent is not None and arm not in ('difference','absolute'):
            raise ValueError('The common-parent supervision comparison changes only the conditional return objective.')
        path=args.output/f'reactive-{common_parent or arm}-{args.initial_seed}.pth'
        parent=torch.load(path,map_location='cpu',weights_only=False)
        if getattr(args,'recursive_value_update',False) and not (args.mixed_only and not parent['interaction']):
            if current['collecting_policy']['sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Recursive value correction must evaluate this arm\'s actual preceding gate.')
        if common_parent is not None:parent=dict(parent,conditional=arm=='difference',arm=arm)
        if parent.get('prior_value') not in ('monotone','orthogonal','candidate','conservative','joint-conservative','candidate-conservative-residual'):
            raise ValueError('The continuation update uses the monotone joint value family.')
        fit_seed=args.seed if args.bootstrap_seed is None else args.bootstrap_seed
        calibration=parent['calibration'];local_start=None
        if parent['settings'].get('coordination_dictionary')!=previous_dictionary:
            raise ValueError('All arms must share the same existing candidate coordinates.')
        if initializer is not None:calibration=initializer['calibration'].clone()
        if args.local_coordination:
            local_start=calibration.shape[-1]
            calibration=F.pad(calibration,(0,new_centers,0,int(args.value_residual and calibration.shape[1]<8)))
        continuous=getattr(args,'continuous_cooperation_test',False)
        field_only=parent.get('shared_edge_value',False)
        regularization=(initializer or parent).get('selection',{}).get('regularization',.01) if field_only else .01
        mixed_supervision='active-face' if field_only else False
        decision_focused=getattr(args,'decision_focused_edge',False) or parent.get('decision_focused_edge',False)
        policy_settings=dict(parent['settings'],unique_control_training=True) if field_only else parent['settings'] if continuous else None
        if field_only:local_start=4;continuous=True
        if continuous and parent.get('unary_reference') is None:raise ValueError('A common learned unary reference is required.')
        coefficients,metrics=update_prior_continuation(calibration,current['data'],current['prior'],anchor,
            parent['conditional'],parent['interaction'],fit_seed,args.mixed_only or field_only,local_start,
            getattr(args,'paired_policy_update',False),regularization=regularization,
            policy_settings=policy_settings,unary_reference=parent.get('unary_reference'),
            mixed_supervision=mixed_supervision,decision_focused=decision_focused)
        # The task minimizes expected terminal cost. Uncertainty regularizes
        # paired learning; the executable gate minimizes that same mean.
        risk=parent['risk'] if 'evaluation_options' in current['data'] else 0.
        settings=dict(parent['settings'],robust_update=args.robust_update or parent['settings'].get('robust_update',False))
        if getattr(args,'conditional_posterior_update',False):
            if parent.get('history_calibrations') and not parent['settings'].get('conditional_posterior_update',False):
                raise ValueError('Cannot reinterpret an existing mirror-update history as posterior improvement.')
            settings['conditional_posterior_update']=True
        if specification is not None:settings['coordination_dictionary']=specification
        if continuous:settings['continuous_cooperation_test']=True
        if field_only:settings['unique_control_training']=True
        if settings['robust_update']:risk=0.
        cooperation_enabled=True;cooperation_calibration=None
        if getattr(args,'crossfit_cooperation',False):
            if not continuous:raise ValueError('Cross-fitted cooperation requires continuous independent-reference testing.')
            selected,cooperation_calibration=reactive_crossfit_cooperation(calibration,current['data'],current['prior'],anchor,
                settings,parent['unary_reference'],parent['conditional'],parent['interaction'],fit_seed,
                field_only,regularization,mixed_supervision,decision_focused)
            cooperation_calibration['selected']=selected
            cooperation_enabled=selected['risk'] is not None
            risk=0. if selected['risk'] is None else selected['risk']
        history=parent.get('history_calibrations',[]);history_risks=parent.get('history_risks',[])
        history_initial_posterior=parent.get('history_initial_posterior',False) if history else parent['settings'].get('conditional_posterior_update',False)
        if getattr(args,'recursive_value_update',False):
            history=[*history,parent['calibration']];history_risks=[*history_risks,parent['risk']]
        if initializer is not None:
            # Old value weights belong to old coordinates. The actual parent
            # remains the recorded continuation; only its value initializer
            # is replaced. Continuous independent-reference improvement does
            # not execute this historical sequence of value estimates.
            history=[];history_risks=[]
        history=[F.pad(value,(0,coefficients.shape[-1]-value.shape[-1])) for value in history]
        actor=ReactiveModelPriorGate(settings,coefficients,risk,parent['model_repeats'],source,
            interaction=parent['interaction'],history_calibrations=history,history_risks=history_risks,
            history_initial_posterior=history_initial_posterior,unary_reference=parent.get('unary_reference'),
            cooperation_enabled=cooperation_enabled)
        result=dict(parent,actor=actor.state_dict(),settings=settings,calibration=coefficients,risk=risk,seed=args.seed,bootstrap_seed=fit_seed,
            history_calibrations=history,history_risks=history_risks,
            history_initial_posterior=history_initial_posterior,
            cooperation_enabled=cooperation_enabled,cooperation_calibration=cooperation_calibration,
            decision_focused_edge=decision_focused,
            policy_iteration=dict(value_prior=str(path),value_prior_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                evaluated_continuation=current['collecting_policy'],paired_reference=current['reference_policy'],
                value_prior_bootstrap_seed=parent.get('bootstrap_seed',parent['seed']),update_seed=args.seed,
                value_initialization=initialization,
                current_credits_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
                anchor_data_sha256=hashlib.sha256(args.reactive_augment_data.read_bytes()).hexdigest(),**metrics),
            decision_rule=('proximal improvement of the fitted paired conditional mean' if settings.get('conditional_posterior_update',False)
                else 'paired conditional mirror improvement of the evaluated recurrent gate'),joint_only_update=args.mixed_only,
            paired_policy_update=getattr(args,'paired_policy_update',False),
            value_residual=coefficients.shape[1]>7,
            ablation_scope=('one recurrent policy improvement from a common parent' if common_parent is not None else
                'arm-specific recurrent policy improvement'),
            training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',result)
        print(json.dumps(dict(arm=arm,**metrics)),flush=True)


def fit_reactive_coordination_value(args):
    """Learn a global candidate-interaction field for the first policy update.

    Source-policy returns identify Q of the unmodified recurrent source.
    Every subsequent policy iteration must collect its own continuation.
    Whole source trajectories form the independent calibration partition.
    """
    torch.set_num_threads(2)
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    if not bank.get('complete',False) or bank.get('collecting_policy')!='source':
        raise ValueError('First-step cooperative value learning requires complete source-policy returns.')
    fit_seed=args.seed if args.bootstrap_seed is None else args.bootstrap_seed
    decision_risk=getattr(args,'posterior_risk',0.)
    count=bank['training_count'];data=reactive_subset(bank['data'],torch.arange(count));prior=bank['prior'][:count]
    groups=reactive_state_groups(data);origins=torch.unique(groups)
    order=origins[torch.randperm(len(origins),generator=torch.Generator().manual_seed(719955))]
    held=max(8,len(order)//4);selection=torch.isin(groups,order[held:]);train_index=selection.nonzero().flatten();valid_index=(~selection).nonzero().flatten()
    training=reactive_subset(data,train_index);validation=reactive_subset(data,valid_index)
    forecast=prior[train_index];valid_forecast=prior[valid_index]
    training=dict(training,evaluation_options=torch.arange(forecast.shape[-1])[None].expand(len(forecast),-1),
        reference_costs=training['costs'])
    anchor=dict(data=training,prior=forecast)
    specification=reactive_coordination_dictionary(training,forecast,anchor,True,bank['settings']['bias'],True,args.coordination_centers or 128)
    encoded=reactive_local_candidate_data(training,forecast,specification)
    checked=reactive_local_candidate_data(validation,valid_forecast,specification)
    encoded_anchor=dict(data=encoded,prior=forecast)
    settings=dict(bank['settings'],exact_joint_fleet=True,canonical_model=True,conditional_posterior_update=True,
        coordination_dictionary=specification,posterior_update=True,
        max_coordinated_receivers=int((training['options'].abs()>1e-7).sum(-1).max()))
    if 'coordination_dictionary' in bank['settings']:raise ValueError('Do not reuse a fitted candidate field as an untrained source bank.')
    common_calibration=None
    adaptive=getattr(args,'adaptive_interaction_link',False)
    if getattr(args,'common_value_initialization',False):
        common_calibration=fit_prior_conservative(training,forecast,False,True,fit_seed,5,adaptive)[0]
    for arm in args.evaluation_arms.split(','):
        if arm not in ('difference','absolute','additive'):raise ValueError('Unknown cooperative-value ablation.')
        conditional=arm!='absolute';interaction=arm!='additive'
        calibration=(common_calibration.clone() if common_calibration is not None else
            fit_prior_conservative(training,forecast,conditional,interaction,fit_seed,5,adaptive)[0])
        start=calibration.shape[-1];calibration=F.pad(calibration,(0,len(specification['centers'])))
        best=None;trials=[]
        joint_refit=getattr(args,'joint_value_refit',False)
        for regularization in ((.01,.1,1.) if interaction or joint_refit else (1.,)):
            coefficients,metrics=update_prior_continuation(calibration,encoded,forecast,encoded_anchor,
                conditional,interaction,fit_seed,not joint_refit,start,True,0.,regularization)
            values=reactive_monotone_estimates(coefficients.double(),valid_forecast.double(),validation['parts'],checked['packet'],interaction)
            _,choice=reactive_conditional_posterior_update(values,validation['options'],torch.zeros(len(valid_forecast),dtype=torch.long),decision_risk,settings['prior'],interaction)
            cost=validation['costs'].mean(1);chosen=cost[torch.arange(len(cost)),choice]
            contrast=values.mean((0,2));contrast=contrast-contrast[:,:1]
            error=float((contrast-cost+cost[:,:1]).square().mean())
            metric=dict(regularization=regularization,paired_cost=float(chosen.mean()),contrast_mse=error,
                gain=float((cost[:,0]-chosen).mean()),**metrics)
            trials.append(metric);score=float(chosen.mean())+.01*error
            if best is None or score<best[0]:best=(score,coefficients,metric)
        _,coefficients,selected=best
        actor=ReactiveModelPriorGate(settings,coefficients,decision_risk,forecast.shape[1],source=object(),interaction=interaction)
        saved=dict(kind='reactive-model-prior-v1',actor=actor.state_dict(),settings=settings,arm=arm,seed=args.seed,
            bootstrap_seed=fit_seed,calibration=coefficients,risk=decision_risk,conditional=conditional,interaction=interaction,
            model_repeats=forecast.shape[1],prior_value='candidate-conservative-residual',
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            training_origins=training['origin'],held_origins=validation['origin'],collecting_policy='source',
            selection=selected,calibration_trials=trials,training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            common_value_initialization=common_calibration is not None,joint_value_refit=joint_refit,
            adaptive_interaction_link=adaptive,
            decision_support='joint intervention cardinality identified by the training controls; the team is reselected each decision',
            decision_rule='conditional-mean proximal improvement from the source, with centered selected-candidate interaction kernels')
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',saved)
        print(json.dumps(dict(arm=arm,training_states=len(training['base']),calibration_states=len(validation['base']),selected=selected)),flush=True)


def calibrate_reactive_coordination_policy(args):
    """Calibrate a conditional cooperative override of the learned unary gate.

    Both supervision arms receive the same independently trained reference.
    The held source-policy panels select whether evidence supports an override.
    No closed-loop development outcomes enter this calibration.
    """
    torch.set_num_threads(2)
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    reference_path=args.output/f'reactive-additive-{args.initial_seed}.pth'
    independent=torch.load(reference_path,map_location='cpu',weights_only=False)
    indices=torch.isin(bank['data']['origin'],independent['held_origins']).nonzero().flatten()
    data=reactive_subset(bank['data'],indices);prior=bank['prior'][indices].double();parts=reactive_credit_parts(data)
    costs=data['costs'].mean(1);programs=data['options'];rows=torch.arange(len(programs));reference=independent['calibration']
    if torch.isin(data['origin'],independent['training_origins']).any():raise ValueError('Policy calibration must exclude fitted origins.')
    for arm in args.evaluation_arms.split(','):
        path=args.output/f'reactive-{arm}-{args.initial_seed}.pth';parent=torch.load(path,map_location='cpu',weights_only=False)
        if torch.isin(data['origin'],parent['training_origins']).any():raise ValueError('Policy calibration overlaps value fitting.')
        settings=parent['settings'];local=reactive_local_candidate_data(data,prior,settings['coordination_dictionary'])
        own=reactive_monotone_estimates(reference.double(),prior,parts,local['packet'],False)
        _,incumbent=reactive_conditional_posterior_update(own,programs,torch.zeros(len(programs),dtype=torch.long),0.,settings['prior'],False)
        value=reactive_monotone_estimates(parent['calibration'].double(),prior,parts,local['packet'],parent['interaction'])
        trials=[];best=None
        for risk in ((None,0.,.5,1.,2.,4.) if parent['interaction'] else (None,)):
            choice=(incumbent if risk is None else reactive_conditional_posterior_update(value,programs,incumbent,risk,settings['prior'])[1])
            cost=float(costs[rows,choice].mean());active=float((choice!=incumbent).float().mean())
            metric=dict(risk=risk,cost=cost,active_fraction=active,gain=float((costs[rows,incumbent]-costs[rows,choice]).mean()))
            trials.append(metric);score=(cost,active)
            if best is None or score<best[0]:best=(score,risk,metric)
        _,risk,selected=best;enabled=risk is not None
        saved=dict(parent,seed=args.seed,unary_reference=reference,cooperation_enabled=enabled,risk=0. if risk is None else risk,
            policy_calibration=dict(reference_path=str(reference_path),reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
                selected=selected,trials=trials,origins=data['origin']),
            deployment_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            decision_rule='conditional cooperative improvement of the current learned unary decision')
        saved['actor']=reactive_prior_actor(saved,object()).state_dict()
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',saved)
        print(json.dumps(dict(arm=arm,calibration_states=len(costs),selected=selected)),flush=True)


def collect_reactive_prior_design(args):
    """Supervise every pair that the recurrent gate actually considers."""
    from policy.interaction_sac import MirrorCoordinationGate
    torch.set_num_threads(2)
    if args.reactive_bootstrap_data is None:raise ValueError('Provide the training state bank.')
    saved=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    if args.count>saved['training_count']:raise ValueError('Do not collect training credits from held states.')
    data=reactive_subset(saved['data'],torch.arange(args.count));settings=saved['settings']
    source=ReactiveSource(args.source_device);actor=MirrorCoordinationGate(16)
    generator=torch.Generator().manual_seed(args.seed);costs=[];outcomes=[];priors=[];programs=[]
    started=time.monotonic()
    for begin in range(0,args.count,args.reactive_batch):
        end=min(begin+args.reactive_batch,args.count);count=end-begin
        packet={k:v[begin:end] for k,v in data['packet'].items()}
        options,parts=reactive_prior_options(packet,data['base'][begin:end],settings)
        branches,n=options.shape[1:];repeats=args.repeats
        expanded={k:v.repeat_interleave(repeats*branches,0) for k,v in packet.items()}
        noise=torch.randn(count*repeats,300,2+3*(n+1),generator=generator).repeat_interleave(branches,0)
        first=options[:,None].expand(-1,repeats,-1,-1).reshape(-1,n)
        cost,outcome,_=reactive_rollout(actor,source,expanded,noise,settings,first=first)
        costs.append(cost.reshape(count,repeats,branches));outcomes.append(outcome.reshape(count,repeats,branches))
        priors.append(reactive_model_prior(source,packet,options,settings,args.prior_repeats));programs.append(options)
        collected=dict(packet={k:v[:end] for k,v in data['packet'].items()},base=data['base'][:end],
            options=torch.cat(programs),parts=parts,origin=data['origin'][:end],
            costs=torch.cat(costs),outcomes=torch.cat(outcomes))
        collected['precision']=reactive_precision(collected['costs'])
        atomic_torch(args.output/f'reactive-model-prior-data-{args.seed}.pth',dict(data=collected,prior=torch.cat(priors),
            settings=settings,training_count=end,complete=end==args.count,collecting_policy='source',
            design='all single and pair candidate assignments used by the deployed gate',noise_seed=args.seed,
            source_data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest()))
        print(json.dumps(dict(states=end,total=args.count,branches=branches,seconds=time.monotonic()-started)),flush=True)


def reactive_decision_credit_design(saved,data,prior,count,branches=4):
    """Spend conditional evaluations on uncertain executable decisions.

    Selection reads only training states and model predictions. The panel
    includes the current deployed choice, its unary alternative and distinct
    bootstrap proposals; no realized return is used to choose the states.
    """
    if saved['settings'].get('joint_fleet',False):
        raise ValueError('The active panel currently requires the measured single/pair decision domain.')
    if 'coordination_dictionary' in saved['settings']:
        data=reactive_local_candidate_data(data,prior,saved['settings']['coordination_dictionary'])
    elif saved['calibration'].shape[-1]>4:data=reactive_candidate_data(data)
    parts=reactive_credit_parts(data);packet=data['packet'];coefficients=saved['calibration']
    independent=reactive_prior_estimates(coefficients,prior,parts,packet,False)
    estimate=reactive_prior_estimates(coefficients,prior,parts,packet,saved['interaction'])
    score,reference=reactive_paired_improvement(estimate,independent,saved['risk'],saved['interaction'],saved['settings'].get('robust_update',False))
    means=estimate.mean(2);chosen=score.argmin(-1);head_choices=means.argmin(-1)
    rows=torch.arange(len(chosen))
    margin=means-means.gather(-1,reference[None,:,None].expand(len(means),-1,1))
    uncertainty=margin.std(0,unbiased=False)
    priority=2.*(chosen!=reference).float()+uncertainty.gather(1,chosen[:,None])[:,0]
    priority=priority+(head_choices!=chosen[None]).float().mean(0)
    selection=priority.argsort(descending=True,stable=True)[:count]
    panels=[]
    for i in selection.tolist():
        proposals=[int(chosen[i]),int(reference[i]),0,*head_choices[:,i].tolist(),*score[i].argsort().tolist()]
        selected=[]
        for proposal in proposals:
            if any(torch.equal(data['options'][i,proposal],data['options'][i,j]) for j in selected):continue
            selected.append(proposal)
            if len(selected)==branches:break
        if len(selected)<branches:raise ValueError('Too few physically distinct candidate assignments.')
        panels.append(selected)
    return selection,torch.tensor(panels),dict(priority=priority[selection],deployed=chosen[selection],
        independent=reference[selection],criterion='current cooperative decision, independent alternative, zero reference and bootstrap disagreement')


def reactive_credit_context(saved,packet,base,programs,parts,prior):
    settings=saved['settings']
    features=(reactive_local_candidate_features(packet,base,programs,parts,prior,settings['coordination_dictionary'])
        if 'coordination_dictionary' in settings else reactive_candidate_interactions(packet,base,programs,parts))
    context=dict(packet,candidate_interactions=features)
    if 'coordination_dictionary' in settings and reactive_selected_dictionary(settings['coordination_dictionary']):
        context['candidate_residual_support']=reactive_dictionary_support(programs,settings['coordination_dictionary'])
    return context


def reactive_credit_unary(saved,context,programs,parts,prior):
    coefficients=saved.get('unary_reference',saved['calibration'])
    width=context['candidate_interactions'].shape[-1]+4
    coefficients=F.pad(coefficients,(0,width-coefficients.shape[-1])).to(prior)
    value=reactive_monotone_estimates(coefficients,prior,parts,context,False)
    settings=saved['settings']
    score,_=reactive_conditional_posterior_update(value,programs,torch.zeros(len(programs),dtype=torch.long),0.,settings.get('prior',.3),False)
    if 'max_coordinated_receivers' in settings:
        score=score.masked_fill((programs.abs()>1e-7).sum(-1)>settings['max_coordinated_receivers'],float('inf'))
    return score,score.argmin(-1),value


def reactive_joint_credit_panel(saved,packet,base,programs,parts,prior,recorded_choice=None,recorded_objective=None):
    """A closed two-receiver intervention around the CURRENT joint decision."""
    coefficients=saved['calibration'];settings=saved['settings']
    context=reactive_credit_context(saved,packet,base,programs,parts,prior)
    unary_score,reference,independent=reactive_credit_unary(saved,context,programs,parts,prior)
    estimate=reactive_monotone_estimates(coefficients,prior,parts,context,True)
    if settings.get('conditional_posterior_update',False):
        anchor=reference if 'unary_reference' in saved else torch.zeros(len(programs),dtype=torch.long)
        score,_=reactive_conditional_posterior_update(estimate,programs,anchor,saved['risk'],settings['prior'])
    else:score,reference=reactive_paired_improvement(estimate,independent,saved['risk'],True,settings.get('robust_update',False))
    if recorded_objective is not None:score=recorded_objective
    allowed=((programs.abs()>1e-7).sum(-1)<=settings.get('max_coordinated_receivers',base.shape[1]))
    score=score.masked_fill(~allowed,float('inf'))
    current=score.argmin(-1) if recorded_choice is None else recorded_choice
    n=base.shape[1];places=3**torch.arange(n)
    panels=[];pairs=[]
    for row,chosen in enumerate(current.tolist()):
        digits=(chosen//places)%3;alternatives=[]
        for i in range(n):
            options=[int(chosen+(level-int(digits[i]))*places[i]) for level in (0,1,2)
                if not torch.equal(programs[row,chosen],programs[row,int(chosen+(level-int(digits[i]))*places[i])])]
            alternatives.append(options or [chosen])
        choices=[]
        for i in range(n):
            for j in range(i+1,n):
                for left in alternatives[i]:
                    for right in alternatives[j]:
                        joint=left+right-chosen
                        if not allowed[row,joint]:continue
                        effect=estimate[:,row,:,joint]-estimate[:,row,:,left]-estimate[:,row,:,right]+estimate[:,row,:,chosen]
                        priority=float(effect.mean(1).std(unbiased=False)+.25*effect.std(unbiased=False))
                        if left==chosen or right==chosen:priority=-1.
                        choices.append((priority,i,j,left,right,joint))
        _,i,j,left,right,joint=max(choices)
        proposed=[chosen,left,right,joint,0,int(reference[row])]
        distinct=[]
        for k in [*proposed,*score[row].argsort().tolist()]:
            if k not in proposed and not allowed[row,k]:continue
            if any(torch.equal(programs[row,k],programs[row,other]) for other in distinct):continue
            distinct.append(k)
            if len(distinct)==7:break
        # The first four positions retain the closed intervention, even if
        # a receiver has no distinct alternative in this physical state.
        panel=proposed+[next((k for k in distinct if k not in proposed),chosen)]
        panels.append(panel);pairs.append((i,j))
    return torch.tensor(panels),dict(current=current,unary=reference,pair=torch.tensor(pairs),
        slots=['current','single_i','single_j','joint_ij','source_zero','unary','additional'],
        criterion='paired conditional uncertainty of a closed intervention, using current model predictions only')


def reactive_joint_disagreement_frames(saved,trajectory,count):
    """Acquire actual returns where learned cooperative and unary gates differ."""
    packet,base,prior=trajectory['packet'],trajectory['base'],trajectory['model_prior']
    if 'model_response' in trajectory:packet=dict(packet,candidate_joint_response=trajectory['model_response'])
    programs,parts=reactive_joint_options(packet,base,saved['settings'])
    context=reactive_credit_context(saved,packet,base,programs,parts,prior)
    _,unary,_=reactive_credit_unary(saved,context,programs,parts,prior)
    score=trajectory['model_objective'];rows=torch.arange(len(base))
    gap=(score[rows,unary]-score[rows,trajectory['choice']]).clamp_min(0.)
    spread=score.masked_fill(~torch.isfinite(score),-float('inf')).amax(-1)-score.amin(-1)
    indices=[]
    for row in range(count):
        candidates=(trajectory['origin']==row).nonzero().flatten().tolist()
        indices.append(max(candidates,key=lambda i:(float(gap[i]),float(spread[i]))))
    return torch.tensor(indices),dict(rule='maximum predicted cooperative-versus-unary decision disagreement, then candidate sensitivity',
        selected_gap=gap[indices],selected_spread=spread[indices])


def encode_reactive_joint_responses(args):
    """Add public coupled responses without changing any actual return label."""
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    if not bank.get('complete',False):raise ValueError('Only completed return measurements can be re-encoded.')
    data=bank['data'];source=ReactiveSource(args.source_device);responses=[];started=time.monotonic()
    settings=dict(bank['settings'],canonical_model=True,exact_joint_fleet=True)
    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for begin in range(0,len(data['base']),args.reactive_batch):
        end=min(begin+args.reactive_batch,len(data['base']))
        packet={k:v[begin:end] for k,v in data['packet'].items()}
        forecast,response=reactive_model_prior(source,packet,data['options'][begin:end],settings,bank['prior'].shape[1],return_response=True)
        torch.testing.assert_close(forecast,bank['prior'][begin:end].to(forecast),rtol=0.,atol=0.)
        responses.append(response)
        if implementation!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise RuntimeError('The public response encoder changed during execution.')
        encoded=reactive_subset(data,torch.arange(end));encoded['model_response']=torch.cat(responses)
        result=dict(bank,data=encoded,prior=bank['prior'][:end],training_count=end,complete=end==len(data['base']),
            response_encoding='joint CBF-filtered first-window dynamics with refreshed source candidates and evader reaction',
            response_implementation_sha256=implementation,response_prior_match_max=0.,
            response_source_data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest())
        atomic_torch(args.output/f'reactive-model-prior-data-{args.seed}.pth',result)
        print(json.dumps(dict(response_states=end,total=len(data['base']),seconds=time.monotonic()-started)),flush=True)


def collect_reactive_complete_pair_credits(args):
    """Complete every executable single/pair face under a common continuation."""
    from policy.interaction_sac import MirrorCoordinationGate
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    if not bank.get('complete',False) or bank.get('collecting_policy')!='source':
        raise ValueError('The initial complete-pair policy improvement starts from source-policy states.')
    count=min(args.count,bank['training_count']);data=reactive_subset(bank['data'],torch.arange(count))
    settings=dict(bank['settings'],canonical_model=True,exact_joint_fleet=True,max_coordinated_receivers=2)
    source=ReactiveSource(args.source_device);reference=MirrorCoordinationGate(16)
    programs,parts=reactive_prior_options(data['packet'],data['base'],settings)
    generator=torch.Generator().manual_seed(args.seed);costs=[];outcomes=[];priors=[]
    code=Path(__file__);core=code.parent/'policy/interaction_sac.py'
    implementation=hashlib.sha256(code.read_bytes()+core.read_bytes()).hexdigest();started=time.monotonic()
    digest=hashlib.sha256()
    for begin in range(0,count,args.reactive_batch):
        end=min(count,begin+args.reactive_batch);size=end-begin;options,n=programs.shape[1:]
        packet={k:v[begin:end] for k,v in data['packet'].items()}
        noise=torch.randn(size*args.repeats,300,2+3*(n+1),generator=generator).repeat_interleave(options,0)
        expanded={k:v.repeat_interleave(args.repeats*options,0) for k,v in packet.items()}
        first=programs[begin:end,None].expand(-1,args.repeats,-1,-1).reshape(-1,n)
        cost,outcome,_=reactive_rollout(reference,source,expanded,noise,settings,first=first)
        costs.append(cost.reshape(size,args.repeats,options));outcomes.append(outcome.reshape(size,args.repeats,options))
        priors.append(reactive_model_prior(source,packet,programs[begin:end],settings,args.prior_repeats))
        digest.update(noise.numpy().tobytes())
        if hashlib.sha256(code.read_bytes()+core.read_bytes()).hexdigest()!=implementation:raise RuntimeError('The complete-pair collector changed during execution.')
        collected=dict(packet={k:v[:end] for k,v in data['packet'].items()},base=data['base'][:end],
            options=programs[:end],parts=parts,origin=data['origin'][:end],costs=torch.cat(costs),outcomes=torch.cat(outcomes))
        collected['precision']=reactive_precision(collected['costs'])
        atomic_torch(args.output/f'reactive-model-prior-data-{args.seed}.pth',dict(data=collected,prior=torch.cat(priors),
            settings=settings,training_count=end,complete=end==count,collecting_policy='source',noise_seed=args.seed,
            paired_innovations_sha256=digest.hexdigest(),source_data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            training_implementation_sha256=implementation,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            design='all executable single and pair controls, with shared actual future innovations'))
        print(json.dumps(dict(complete_pair_states=end,total=count,branches=options,seconds=time.monotonic()-started)),flush=True)


def collect_reactive_attention(args):
    """Matched current-policy returns on fresh times of training trajectories."""
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    path=args.output/f'reactive-{args.collecting_arm}-{args.initial_seed}.pth'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['kind']!='reactive-attention-v1':raise ValueError('The collector requires a direct attention gate.')
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    model_hash=hashlib.sha256(path.read_bytes()).hexdigest();settings=saved['settings']
    source=ReactiveSource(args.source_device);actor=reactive_prior_actor(saved,source);actor.load_state_dict(saved['actor'])
    rng=torch.Generator().manual_seed(args.seed);origins=bank['data']['origin']
    fit=torch.isin(origins,saved['training_origins']).nonzero().flatten()
    held=torch.isin(origins,saved['held_origins']).nonzero().flatten()
    previous=None
    if args.reactive_augment_data is not None:
        previous=torch.load(args.reactive_augment_data,map_location='cpu',weights_only=False)
        if not previous.get('complete',False) or previous['collecting_policy']['sha256']!=model_hash:
            raise ValueError('Append only completed evaluations of the identical actual continuation.')
        fit=fit[~torch.isin(origins[fit],previous['data']['origin'])]
        forecast=bank['prior'][:,:,:1+2*settings['defenders']].mean(1)
        influence=forecast.amax(-1)-forecast.amin(-1)
        fit=fit[influence[fit]>.5]
    fit=fit[torch.randperm(len(fit),generator=rng)[:args.count]]
    held=held[torch.randperm(len(held),generator=rng)[:max(8,args.count//3)]]
    chosen=torch.cat((fit,held));start_packet={k:v[chosen] for k,v in bank['data']['packet'].items()}
    n=settings['defenders'];started=time.monotonic()
    noise=torch.randn(len(chosen),300,2+3*(n+1),generator=rng)
    _,_,trajectory=reactive_rollout(actor,source,start_packet,noise,settings,collect=True)
    active=args.active_joint_credits
    if active:
        if not actor.has_unary_reference:raise ValueError('Active attention credits require the current independent reference.')
        all_options,all_parts=reactive_prior_options(trajectory['packet'],trajectory['base'],settings)
        independent=reactive_attention_unary(saved['unary_reference'],trajectory['model_prior'],all_parts,trajectory['packet'])
        independent_choice=(independent+settings['prior']*all_options.square().mean(-1)).argmin(-1)
        displacement=(trajectory['program']-all_options[torch.arange(len(all_options)),independent_choice]).abs().mean(-1)
        priority=(-trajectory['model_objective'].amin(-1)).clamp_min(0.)+.2*displacement
        span=trajectory['model_prior'].mean(1).amax(-1)-trajectory['model_prior'].mean(1).amin(-1)
        priority=priority*(span/.5).clamp(max=1.)+.01*span
    indices=[];root_indices=[]
    for root in range(len(chosen)):
        frames=(trajectory['origin']==root).nonzero().flatten()
        # Uniformly sample up to two decision times per origin. Selection
        # never reads measured return or selects successful paths.
        pick=(priority[frames].argmax().reshape(1) if active else torch.randperm(len(frames),generator=rng)[:2])
        indices.extend(frames[pick].tolist());root_indices.extend([root]*len(pick))
    indices=torch.tensor(indices);root_indices=torch.tensor(root_indices)
    packet={k:v[indices] for k,v in trajectory['packet'].items()};base=trajectory['base'][indices]
    options,parts=reactive_prior_options(packet,base,settings)
    evaluation=None
    if active:
        evaluation=reactive_attention_panel(options,parts,trajectory['model_objective'][indices],
            trajectory['choice'][indices],independent_choice[indices])
    measured=options if evaluation is None else options.gather(1,evaluation[:,:,None].expand(-1,-1,n))
    costs=[];outcomes=[];repeats=args.repeats
    for begin in range(0,len(indices),args.reactive_batch):
        end=min(len(indices),begin+args.reactive_batch);count=end-begin;branches=measured.shape[1]
        expanded={k:v[begin:end].repeat_interleave(repeats*branches,0) for k,v in packet.items()}
        noise=torch.randn(count*repeats,300,2+3*(n+1),generator=rng).repeat_interleave(branches,0)
        first=measured[begin:end,None].expand(-1,repeats,-1,-1).reshape(-1,n)
        cost,outcome,_=reactive_rollout(actor,source,expanded,noise,settings,first=first)
        costs.append(cost.reshape(count,repeats,branches));outcomes.append(outcome.reshape(count,repeats,branches))
        data=dict(packet={k:v[:end] for k,v in packet.items()},base=base[:end],options=options[:end],parts=parts,
            origin=origins[chosen[root_indices[:end]]],costs=torch.cat(costs),outcomes=torch.cat(outcomes))
        if evaluation is not None:data['evaluation_options']=evaluation[:end]
        combined_packet=start_packet;combined_origins=origins[chosen]
        if previous is not None:
            old=previous['data']
            torch.testing.assert_close(old['parts'],parts,rtol=0.,atol=0.)
            data={k:({j:torch.cat((old[k][j],value)) for j,value in v.items()} if k=='packet' else v if k=='parts' else torch.cat((old[k],v))) for k,v in data.items()}
            joined_origins=torch.cat((previous['initial_origins'],combined_origins));keep=[];seen=set()
            for i,origin in enumerate(joined_origins.tolist()):
                if origin not in seen:keep.append(i);seen.add(origin)
            combined_origins=joined_origins[keep]
            combined_packet={k:torch.cat((previous['initial_packet'][k],v))[keep] for k,v in start_packet.items()}
        result=dict(data=data,settings=settings,complete=False,training_count=end,noise_seed=args.seed,
            collecting_policy=dict(path=str(path),sha256=model_hash),training_origins=saved['training_origins'],held_origins=saved['held_origins'],
            on_policy_states=True,initial_packet=combined_packet,initial_origins=combined_origins,
            source_bank_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            training_implementation_sha256=implementation)
        if 'model_prior' in trajectory:
            result['prior']=trajectory['model_prior'][indices[:end]]
            if previous is not None:result['prior']=torch.cat((previous['prior'],result['prior']))
            result['prior_representation']='single-vessel public forecasts; joint value contains only their calibrated sum'
        if previous is not None:
            result['previous_data']=dict(path=str(args.reactive_augment_data),sha256=hashlib.sha256(args.reactive_augment_data.read_bytes()).hexdigest(),noise_seed=previous['noise_seed'])
        result['training_count']=len(data['origin'])
        result['credit_design']='largest predicted current cooperation disagreement; closed pair face and both executable decisions' if active else 'uniform decision times; complete pair controls'
        atomic_torch(args.output/f'reactive-attention-data-{args.seed}.pth',result)
        print(json.dumps(dict(states=end,total=len(indices),branches=branches,seconds=time.monotonic()-started)),flush=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=implementation or hashlib.sha256(path.read_bytes()).hexdigest()!=model_hash:
        raise RuntimeError('The collecting policy or implementation changed.')
    result['complete']=True;atomic_torch(args.output/f'reactive-attention-data-{args.seed}.pth',result)


def fit_reactive_attention(args):
    """Learn a directly executable joint gate from matched continuation returns."""
    torch.set_num_threads(2);torch.set_grad_enabled(True)
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    if not bank.get('complete',False):raise ValueError('Complete paired controls are required.')
    data=bank['data'];settings=dict(bank['settings']);device=args.device
    reference_path=args.output/f'reactive-additive-{args.initial_seed}.pth'
    reference=torch.load(reference_path,map_location='cpu',weights_only=False)
    train=torch.isin(data['origin'],reference['training_origins']).nonzero().flatten()
    valid=(~torch.isin(data['origin'],reference['training_origins'])).nonzero().flatten()
    if len(train)<16 or len(valid)<8:raise ValueError('Independent training and calibration origins are required.')
    reaction=args.reaction_encoding
    inputs={k:v.to(device) for k,v in ReactiveAttentionValue.inputs(data['packet'],data['base'],settings['bias'],reaction).items()}
    full_options=data['options'].to(device);options=full_options;costs=data['costs'].to(device);parts=data['parts']
    evaluated=data.get('evaluation_options')
    if evaluated is not None:
        evaluated=evaluated.to(device);options=full_options.gather(1,evaluated[:,:,None].expand(-1,-1,full_options.shape[-1]))
        torch.testing.assert_close(options[:,0]+options[:,3],options[:,1]+options[:,2],rtol=0.,atol=1e-6)
    members=3;torch.manual_seed(args.seed)
    anchored=getattr(args,'attention_unary_anchor',False)
    unary_reference=reference.get('unary_reference',reference.get('calibration')) if anchored else None
    if anchored:
        if unary_reference is None:raise ValueError('A calibrated independent value link is required.')
        unary_reference=unary_reference[...,:4]
    initial=ReactiveAttentionGate(settings,args.width,members,0.,False,reaction,unary_reference).to(device)
    train=train.to(device);valid=valid.to(device)
    generator=torch.Generator().manual_seed(args.seed)
    def take(index):return {k:v[index] for k,v in inputs.items()}
    # A common independent response prior initializes all ablations. It is
    # learned only from training origins and is thereafter updated by actual
    # returns. No nominal future search remains in deployed decisions.
    if isinstance(bank['collecting_policy'],dict):
        parent_path=Path(bank['collecting_policy']['path'])
        if hashlib.sha256(parent_path.read_bytes()).hexdigest()!=bank['collecting_policy']['sha256']:
            raise ValueError('The actual continuation checkpoint changed.')
        parent=torch.load(parent_path,map_location='cpu',weights_only=False)
        if parent['kind']!='reactive-attention-v1':raise ValueError('Continue the same direct gate architecture.')
        expected=initial.state_dict();inherited={}
        for key,value in parent['actor'].items():
            if value.shape!=expected[key].shape:
                if not reaction or parent.get('reaction_messages',False) or not key.endswith(('geometry.0.weight','mixed.0.weight')):
                    raise ValueError('An attention update must retain its existing value representation.')
                value=F.pad(value,(0,24))
            inherited[key]=value
        if anchored:inherited['unary_reference']=unary_reference
        initial.load_state_dict(inherited)
    else:
        with torch.no_grad():
            teacher=reactive_monotone_estimates(reference['calibration'][...,:4].double(),bank['prior'].double(),
                parts,data['packet'],False).mean((0,2)).float().to(device)
        optimizer=torch.optim.AdamW(initial.parameters(),lr=.003,weight_decay=.001)
        for step in range(500):
            index=train[torch.randint(len(train),(min(32,len(train)),),generator=generator).to(device)]
            loss=sum((member(take(index),options[index])[1]-teacher[index]).square().mean() for member in initial.members)/members
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(initial.parameters(),10.);optimizer.step()
    if anchored:
        if 'prior' not in bank:raise ValueError('Record the independent public response predictions for current-policy credits.')
        with torch.no_grad():
            all_unary=reactive_attention_unary(unary_reference,bank['prior'],parts,data['packet']).float().to(device)
            incumbent=(all_unary+settings['prior']*full_options.square().mean(-1)).argmin(-1)
            independent_program=full_options[torch.arange(len(full_options),device=device),incumbent]
            unary_values=all_unary if evaluated is None else all_unary.gather(1,evaluated)
    policy_penalty=settings['prior']*((options-independent_program[:,None]) if anchored else options).square().mean(-1)
    common=copy.deepcopy(initial.state_dict());start=time.monotonic()
    recurrent=isinstance(bank['collecting_policy'],dict)
    check_packet=None;check_noise=None;source=None
    if recurrent:
        calibration=torch.isin(bank['initial_origins'],reference['held_origins']).nonzero().flatten()
        if len(calibration)<8:raise ValueError('Recurrent policy selection needs held trajectory origins.')
        if torch.isin(bank['initial_origins'][calibration],data['origin'][train.cpu()]).any():
            raise ValueError('Closed-loop calibration leaked a fitted origin.')
        calibration_repeats=2 if anchored else 4
        check_packet={k:v[calibration].repeat_interleave(calibration_repeats,0) for k,v in bank['initial_packet'].items()}
        check_noise=torch.randn(len(calibration)*calibration_repeats,300,2+3*(settings['defenders']+1),
            generator=torch.Generator().manual_seed(args.seed+880000))
        source=ReactiveSource(args.source_device)
    anchor=None
    if recurrent and args.reactive_augment_data is not None:
        old=torch.load(args.reactive_augment_data,map_location='cpu',weights_only=False)['data']
        select=torch.isin(old['origin'],reference['training_origins']).nonzero().flatten()
        old=reactive_subset(old,select)
        anchor=({k:v.to(device) for k,v in ReactiveAttentionValue.inputs(old['packet'],old['base'],settings['bias'],reaction).items()},old['options'].to(device))
    precision=reactive_precision(costs[train],True)
    unique=reactive_unique_control_mask(options)
    for arm in args.evaluation_arms.split(','):
        if arm not in ('difference','absolute','additive'):raise ValueError('Unknown attention ablation.')
        conditional=arm!='absolute';interaction=arm!='additive'
        actor=ReactiveAttentionGate(settings,args.width,members,0.,interaction,reaction,unary_reference,source).to(device);actor.load_state_dict(common)
        if getattr(args,'attention_readout_update',False):
            for name,parameter in actor.named_parameters():
                parameter.requires_grad_(any(key in name for key in ('.contextual.4.','.mixed.4.','.baseline.')))
        bootstrap=[];weights=[];pair_regrets=[]
        for head in range(members):
            rng=torch.Generator().manual_seed(args.seed+head*104729)
            rows,futures=reactive_positive_credit_weights(reactive_state_groups(reactive_subset(data,train.cpu())),costs.shape[1],rng)
            bootstrap.append((costs[train]*futures.to(costs)[:,:,None]).sum(1));weights.append(rows.to(costs))
            penalty=policy_penalty[train]
            pair_regrets.append(reactive_pair_regret_weights(costs[train]+penalty[:,None],futures.to(costs)))
        target=torch.stack(bootstrap);row_weights=torch.stack(weights);pair_regrets=torch.stack(pair_regrets)
        mixed_indices=(torch.tensor([3],device=device) if evaluated is not None else ((parts>0).sum(-1)==2).nonzero().flatten().to(device))
        mixed_parts=torch.tensor([[1,2]],device=device) if evaluated is not None else parts[mixed_indices.cpu()].to(device)
        mixed_returns=costs[:,:,mixed_indices]-costs[:,:,mixed_parts].sum(-1)+costs[:,:,:1]
        mixed_variance=1.+mixed_returns.var(1,unbiased=False)/costs.shape[1]
        optimizer=torch.optim.AdamW(actor.parameters(),lr=.001,weight_decay=.002)
        best=None;trials=[]
        anchor_values=None
        if anchor is not None:
            with torch.no_grad():
                anchor_values=torch.stack([member(*anchor)[1] for member in actor.members])
                if anchored:anchor_values=anchor_values-torch.stack([member(*anchor,'interaction-off')[1] for member in actor.members])
        if recurrent:
            actor.risk=parent['risk']
            check_cost,check_outcome,_=reactive_rollout(actor,source,check_packet,check_noise,settings)
            metric=dict(step=0,risk=actor.risk,update_fraction=0.,cost=float(check_cost.mean()),captures=int((check_outcome==3).sum()),
                criterion='held-origin recurrent deployment')
            best=(metric['cost'],copy.deepcopy(actor.state_dict()),metric);trials.append(metric)
        iterations=0 if recurrent and anchored and not interaction else args.updates
        for step in range(iterations):
            local=torch.randint(len(train),(min(32,len(train)),),generator=generator).to(device);index=train[local]
            loss=costs.new_zeros(())
            for h,member in enumerate(actor.members):
                state,energy=member(take(index),options[index]);truth=target[h,local];weight=row_weights[h,local]
                if anchored:energy=unary_values[index]+energy-member(take(index),options[index],'interaction-off')[1]
                if conditional:
                    delta=(energy-truth)[:,1:]-(energy-truth)[:,:1]
                    regression=torch.einsum('bi,bij,bj->b',delta,precision[local],delta)/delta.shape[-1]
                    # The policy loss learns the same trust-regularized choice
                    # used during recurrent execution, in units of actual cost.
                    trust=policy_penalty[index]
                    decision=reactive_pair_regret_loss(energy+trust,pair_regrets[h,local],weight,temperature=2.,support=unique[index])
                    loss=loss+(weight*regression).mean()+.1*decision
                    if interaction:
                        predicted_mixed=energy[:,mixed_indices]-energy[:,mixed_parts].sum(-1)+energy[:,:1]
                        expected_mixed=truth[:,mixed_indices]-truth[:,mixed_parts].sum(-1)+truth[:,:1]
                        loss=loss+(weight*((predicted_mixed-expected_mixed).square()/mixed_variance[index]).mean(-1)).mean()
                    loss=loss+.01*(state-(truth-energy.detach()).mean(-1)).square().mean()
                else:
                    variance=1.+costs[index].var(1,unbiased=False).mean(-1)/costs.shape[1]
                    loss=loss+(weight*((state[:,None]+energy-truth).square()*unique[index]).sum(-1)/unique[index].sum(-1)/variance).mean()
                if anchor is not None:
                    subset=torch.randperm(len(anchor[1]),generator=generator)[:32].to(device)
                    retained=member({k:v[subset] for k,v in anchor[0].items()},anchor[1][subset])[1]
                    if anchored:retained=retained-member({k:v[subset] for k,v in anchor[0].items()},anchor[1][subset],'interaction-off')[1]
                    previous=anchor_values[h,subset]
                    loss=loss+.1*((retained-previous).square()/(1.+previous.var(-1,keepdim=True,unbiased=False))).mean()
            optimizer.zero_grad();(loss/members).backward();nn.utils.clip_grad_norm_(actor.parameters(),10.);optimizer.step()
            if recurrent:
                if anchored and step+1!=args.updates:continue
                if (step+1)%400 and step+1!=args.updates:continue
                fitted=copy.deepcopy(actor.state_dict())
                for fraction in (.25,.5,1.):
                    actor.load_state_dict({k:common[k]+fraction*(v-common[k]) for k,v in fitted.items()})
                    check_cost,check_outcome,_=reactive_rollout(actor,source,check_packet,check_noise,settings)
                    metric=dict(step=step+1,risk=actor.risk,update_fraction=fraction,cost=float(check_cost.mean()),captures=int((check_outcome==3).sum()),
                        criterion='held-origin recurrent deployment')
                    trials.append(metric)
                    if metric['cost']<best[0]:best=(metric['cost'],copy.deepcopy(actor.state_dict()),metric)
                actor.load_state_dict(fitted)
                print(json.dumps(dict(arm=arm,step=step+1,loss=float(loss/members),best=best[2],seconds=time.monotonic()-start)),flush=True)
                continue
            if (step+1)%100 and step+1!=args.updates:continue
            with torch.no_grad():
                prediction=torch.stack([member(take(valid),options[valid])[1] for member in actor.members])
                if anchored:
                    own=torch.stack([member(take(valid),options[valid],'interaction-off')[1] for member in actor.members])
                    prediction=unary_values[valid][None]+prediction-own
                    incumbent=(unary_values[valid]+settings['prior']*options[valid].square().mean(-1)).argmin(-1)
                    prediction=prediction-prediction.gather(-1,incumbent[None,:,None].expand(members,-1,1))
                    trust=settings['prior']*(options[valid]-options[valid][torch.arange(len(valid),device=device),incumbent][:,None]).square().mean(-1)
                else:trust=settings['prior']*options[valid].square().mean(-1)
                for risk in (0.,.5,1.,2.):
                    score=prediction.mean(0)+risk*prediction.std(0,unbiased=False)+trust
                    choice=score.argmin(-1);actual=costs[valid].mean(1)[torch.arange(len(valid),device=device),choice]
                    if anchored and not interaction:
                        choice=incumbent;actual=costs[valid].mean(1)[torch.arange(len(valid),device=device),choice]
                    metric=dict(step=step+1,risk=risk,cost=float(actual.mean()),source_gain=float((costs[valid,:,0].mean(1)-actual).mean()))
                    trials.append(metric)
                    if best is None or metric['cost']<best[0]:best=(metric['cost'],copy.deepcopy(actor.state_dict()),metric)
            if (step+1)%400==0:print(json.dumps(dict(arm=arm,step=step+1,loss=float(loss/members),best=best[2],seconds=time.monotonic()-start)),flush=True)
        _,state,selection=best;actor.load_state_dict(state);actor.risk=selection['risk'];actor.cpu()
        saved=dict(kind='reactive-attention-v1',settings=settings,actor=actor.state_dict(),width=args.width,members=members,
            risk=actor.risk,conditional=conditional,interaction=interaction,arm=arm,seed=args.seed,reaction_messages=reaction,
            unary_reference=unary_reference,
            readout_update=getattr(args,'attention_readout_update',False),
            selection=selection,calibration_trials=trials,training_origins=torch.unique(reference['training_origins']),
            held_origins=torch.unique(reference['held_origins']),current_training_origins=torch.unique(data['origin'][train.cpu()]),
            collecting_policy=bank['collecting_policy'],source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            closed_loop_calibration_origins=bank['initial_origins'][calibration] if recurrent else None,
            closed_loop_calibration_noise_seed=args.seed+880000 if recurrent else None,
            training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            decision_rule='recurrent joint attention factors, fitted to paired returns of the recorded continuation')
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',saved)
        print(json.dumps(dict(arm=arm,selection=selection,seconds=time.monotonic()-start)),flush=True)


def fit_reactive_edge_value(args):
    """Shared local candidate messages, identified by full paired control faces."""
    torch.set_num_threads(2)
    bank=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    reference_path=args.output/f'reactive-additive-{args.initial_seed}.pth'
    reference=torch.load(reference_path,map_location='cpu',weights_only=False)
    if not bank.get('complete',False) or bank['collecting_policy']!='source':raise ValueError('Complete source-policy faces are required.')
    data,prior=bank['data'],bank['prior'];origins=data['origin']
    train=torch.isin(origins,reference['training_origins']).nonzero().flatten()
    valid=(~torch.isin(origins,reference['training_origins'])).nonzero().flatten()
    if len(train)<16 or len(valid)<8:raise ValueError('Keep independent source trajectories for cooperative value calibration.')
    training=reactive_subset(data,train);validation=reactive_subset(data,valid)
    if 'model_response' in training:
        training=dict(training,packet=dict(training['packet'],candidate_joint_response=training['model_response']))
    specification=reactive_edge_dictionary(training,args.coordination_centers or 64,bank['settings']['bias'])
    features=reactive_local_candidate_features(training['packet'],training['base'],training['options'],training['parts'],prior[train],specification)
    specification['feature_scale']=features.flatten(0,1).std(0).clamp_min(.03).tolist()
    training=reactive_local_candidate_data(training,prior[train],specification)
    validation=reactive_local_candidate_data(validation,prior[valid],specification)
    training=dict(training,evaluation_options=torch.arange(len(training['parts']))[None].expand(len(train),-1),reference_costs=training['costs'])
    width=training['packet']['candidate_interactions'].shape[-1]+4
    calibration=torch.zeros(len(reference['calibration']),10,width)
    calibration[:,:4,:4]=reference['calibration'][:,:4,:4]
    # The independent value is shared exactly. Only the edge field learns
    # marginal candidate context and the nonadditive response to joint controls.
    calibration[:,4,0]=-30.;calibration[:,8,0]=-30.
    unary_reference=calibration.clone()
    settings=dict(bank['settings'],coordination_dictionary=specification,posterior_update=True,
        conditional_posterior_update=True,continuous_cooperation_test=True,
        unique_control_training=getattr(args,'decision_focused_edge',False))
    _,reference_choice,_=reactive_credit_unary(dict(settings=settings,calibration=calibration,unary_reference=unary_reference),
        validation['packet'],validation['options'],validation['parts'],prior[valid].double())
    valid_cost=validation['costs'].mean(1);rows=torch.arange(len(valid));reference_cost=valid_cost[rows,reference_choice]
    fit_seed=args.seed if args.bootstrap_seed is None else args.bootstrap_seed
    for arm in args.evaluation_arms.split(','):
        conditional=arm!='absolute';interaction=arm!='additive';trials=[];best=None
        if arm not in ('difference','absolute','additive'):raise ValueError('Unknown edge-value ablation.')
        for regularization in ((.1,1.,10.) if interaction else (1.,)):
            coefficients,metrics=update_prior_continuation(calibration,training,prior[train],dict(data=training,prior=prior[train]),
                conditional,interaction,fit_seed,True,4,True,0.,regularization,settings,unary_reference,True,
                getattr(args,'decision_focused_edge',False))
            values=reactive_monotone_estimates(coefficients.double(),prior[valid].double(),validation['parts'],validation['packet'],interaction)
            for risk in ((None,0.,.5,1.,2.,4.) if interaction else (None,)):
                if risk is None:choice=reference_choice
                else:_,choice=reactive_conditional_posterior_update(values,validation['options'],reference_choice,risk,settings['prior'],interaction)
                chosen=valid_cost[rows,choice];gain=reference_cost-chosen
                metric=dict(regularization=regularization,risk=risk,paired_cost=float(chosen.mean()),gain=float(gain.mean()),
                    active=float((choice!=reference_choice).float().mean()),standard_error=float(gain.std()/len(gain)**.5))
                trials.append(metric)
                score=(metric['paired_cost'],metric['active'],-regularization)
                if best is None or score<best[0]:best=(score,coefficients,metric)
        _,coefficients,selected=best;enabled=selected['risk'] is not None;risk=selected['risk'] or 0.
        actor=ReactiveModelPriorGate(settings,coefficients,risk,prior.shape[1],source=object(),interaction=interaction,
            unary_reference=unary_reference,cooperation_enabled=enabled)
        saved=dict(kind='reactive-model-prior-v1',actor=actor.state_dict(),settings=settings,calibration=coefficients,
            unary_reference=unary_reference,cooperation_enabled=enabled,risk=risk,model_repeats=prior.shape[1],
            conditional=conditional,interaction=interaction,arm=arm,seed=args.seed,bootstrap_seed=fit_seed,
            prior_value='candidate-conservative-residual',training_origins=origins[train],held_origins=origins[valid],
            collecting_policy='source',selection=selected,calibration_trials=trials,shared_edge_value=True,
            decision_focused_edge=getattr(args,'decision_focused_edge',False),
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            unary_reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            decision_rule='continuous independent-reference improvement from shared candidate-edge conditional credits')
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',saved)
        print(json.dumps(dict(arm=arm,training_states=len(train),calibration_states=len(valid),selected=selected)),flush=True)


def collect_reactive_joint_credits(args):
    """Actual conditional returns from on-policy states of the joint planner."""
    from policy.interaction_sac import MirrorCoordinationGate
    torch.set_num_threads(2)
    collecting_arm=getattr(args,'collecting_arm','difference')
    path=args.output/f'reactive-{collecting_arm}-{args.initial_seed}.pth'
    parent=torch.load(path,map_location='cpu',weights_only=False);settings=parent['settings']
    if not settings.get('exact_joint_fleet',False):raise ValueError('Use the exact joint decision policy.')
    code=Path(__file__);core=code.parent/'policy/interaction_sac.py'
    implementation=hashlib.sha256(code.read_bytes()+core.read_bytes()).hexdigest()
    model_hash=hashlib.sha256(path.read_bytes()).hexdigest();source=ReactiveSource(args.source_device)
    actor=reactive_prior_actor(parent,source)
    actor.load_state_dict(parent['actor']);reference_actor=MirrorCoordinationGate(16)
    roots,root_hash=reactive_roots(args.root_source,'train',args.train_count)
    generator=torch.Generator().manual_seed(args.seed);available=torch.unique(parent['training_origins'])
    selected=available[torch.randperm(len(available),generator=generator)[:args.count]]
    packet={k:v[selected] for k,v in roots.items()};n=settings['defenders']
    innovations=torch.randn(len(selected),300,2+3*(n+1),generator=generator)
    started=time.monotonic()
    _,_,trajectory=reactive_rollout(actor,source,packet,innovations,settings,collect=True)
    if getattr(args,'active_joint_credits',False):
        indices,state_selection=reactive_joint_disagreement_frames(parent,trajectory,len(selected))
    else:
        indices=[]
        for row in range(len(selected)):
            candidates=(trajectory['origin']==row).nonzero().flatten()
            indices.append(candidates[torch.randint(len(candidates),(1,),generator=generator)])
        indices=torch.cat(indices);state_selection=dict(rule='uniform on-policy frame per selected training origin')
    packet={k:v[indices] for k,v in trajectory['packet'].items()}
    base=trajectory['base'][indices];programs,parts=reactive_joint_options(packet,base,settings)
    # Retain the actual forecasts and gate used when this state was visited.
    # Re-running long-horizon nominal futures in a different numerical batch
    # is not a substitute for the collecting policy's recorded decision.
    prior=trajectory['model_prior'][indices]
    feature_packet=(dict(packet,candidate_joint_response=trajectory['model_response'][indices]) if 'model_response' in trajectory else packet)
    evaluated,design=reactive_joint_credit_panel(parent,feature_packet,base,programs,parts,prior,
        trajectory['choice'][indices],trajectory['model_objective'][indices])
    design['state_selection']=state_selection
    actual=programs.gather(1,evaluated[:,:1,None].expand(-1,-1,n))[:,0]
    torch.testing.assert_close(actual,trajectory['program'][indices],rtol=0.,atol=0.)
    print(json.dumps(dict(on_policy_states=len(indices),branches=evaluated.shape[1],seconds=time.monotonic()-started)),flush=True)
    costs=[];outcomes=[];reference_costs=[];reference_outcomes=[];digest=hashlib.sha256()
    for begin in range(0,len(selected),args.reactive_batch):
        end=min(begin+args.reactive_batch,len(selected));count=end-begin;branches=evaluated.shape[1]
        p={k:v[begin:end] for k,v in packet.items()}
        controls=programs[begin:end].gather(1,evaluated[begin:end,:,None].expand(-1,-1,n))
        noise=torch.randn(count*args.repeats,300,2+3*(n+1),generator=generator).repeat_interleave(branches,0)
        expanded={k:v.repeat_interleave(args.repeats*branches,0) for k,v in p.items()}
        first=controls[:,None].expand(-1,args.repeats,-1,-1).reshape(-1,n)
        cost,outcome,_=reactive_rollout(actor,source,expanded,noise,settings,first=first)
        costs.append(cost.reshape(count,args.repeats,branches));outcomes.append(outcome.reshape(count,args.repeats,branches))
        cost,outcome,_=reactive_rollout(reference_actor,source,expanded,noise,settings,first=first)
        reference_costs.append(cost.reshape(count,args.repeats,branches));reference_outcomes.append(outcome.reshape(count,args.repeats,branches))
        digest.update(noise.numpy().tobytes())
        data=dict(packet={k:v[:end] for k,v in packet.items()},base=base[:end],options=programs[:end],parts=parts,
            evaluation_options=evaluated[:end],origin=selected[:end],costs=torch.cat(costs),outcomes=torch.cat(outcomes),
            reference_costs=torch.cat(reference_costs),reference_outcomes=torch.cat(reference_outcomes))
        if 'model_response' in trajectory:data['model_response']=trajectory['model_response'][indices[:end]]
        data['precision']=reactive_precision(data['costs'])
        if hashlib.sha256(code.read_bytes()+core.read_bytes()).hexdigest()!=implementation or hashlib.sha256(path.read_bytes()).hexdigest()!=model_hash:
            raise RuntimeError('The collecting joint decision policy or implementation changed.')
        result=dict(data=data,prior=prior[:end],settings=settings,training_count=end,complete=end==len(selected),
            noise_seed=args.seed,credit_batch=args.reactive_batch,credit_design=design,
            collecting_policy=dict(path=str(path),sha256=model_hash,continuation='same exact joint parent at every decision boundary'),
            on_policy_states=True,root_sha256=root_hash,training_implementation_sha256=implementation,
            reference_policy='source with the same first control and common future innovations',
            paired_continuation_innovations_sha256=digest.hexdigest())
        suffix=str(args.seed) if collecting_arm=='difference' else f'{collecting_arm}-{args.seed}'
        atomic_torch(args.output/f'reactive-model-prior-data-{suffix}.pth',result)
        print(json.dumps(dict(arm=collecting_arm,states=end,total=len(selected),seconds=time.monotonic()-started)),flush=True)


def collect_reactive_prior_credits(args):
    """Terminal conditional returns under the actual recurrent parent gate."""
    torch.set_num_threads(2)
    if args.reactive_bootstrap_data is None or args.initial_seed is None:
        raise ValueError('Provide the state bank and the deployed parent seed.')
    original=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    path=args.output/f'reactive-difference-{args.initial_seed}.pth'
    parent=torch.load(path,map_location='cpu',weights_only=False);settings=parent['settings']
    from policy.interaction_sac import MirrorCoordinationGate
    source=ReactiveSource(args.source_device);reference_actor=MirrorCoordinationGate(16)
    actor=reactive_prior_actor(parent,source);actor.load_state_dict(parent['actor'])
    generator=torch.Generator().manual_seed(args.seed)
    if args.decision_credits:
        training=reactive_subset(original['data'],torch.arange(original['training_count']))
        selection,evaluated,design=reactive_decision_credit_design(parent,training,original['prior'][:len(training['base'])],args.count)
    else:
        selection=torch.randperm(original['training_count'],generator=generator)[:args.count];design=None
    data=reactive_subset(original['data'],selection)
    if args.decision_credits:data['evaluation_options']=evaluated
    costs=[];outcomes=[];reference_costs=[];reference_outcomes=[]
    noise_digest=hashlib.sha256();started=time.monotonic()
    for begin in range(0,len(selection),args.reactive_batch):
        end=min(begin+args.reactive_batch,len(selection));count=end-begin
        packet={k:v[begin:end] for k,v in data['packet'].items()};programs=data['options'][begin:end]
        if args.decision_credits:
            programs=programs.gather(1,data['evaluation_options'][begin:end,:,None].expand(-1,-1,programs.shape[-1]))
        options,n=programs.shape[1:];repeats=args.repeats
        noise=torch.randn(count*repeats,300,2+3*(n+1),generator=generator).repeat_interleave(options,0)
        expanded={k:v.repeat_interleave(repeats*options,0) for k,v in packet.items()}
        first=programs[:,None].expand(-1,repeats,-1,-1).reshape(-1,n)
        cost,outcome,_=reactive_rollout(actor,source,expanded,noise,settings,first=first)
        costs.append(cost.reshape(count,repeats,options));outcomes.append(outcome.reshape(count,repeats,options))
        reference_cost,reference_outcome,_=reactive_rollout(reference_actor,source,expanded,noise,settings,first=first)
        reference_costs.append(reference_cost.reshape(count,repeats,options))
        reference_outcomes.append(reference_outcome.reshape(count,repeats,options));noise_digest.update(noise.numpy().tobytes())
        collected=reactive_subset(data,torch.arange(end))
        collected.update(costs=torch.cat(costs),outcomes=torch.cat(outcomes),
            reference_costs=torch.cat(reference_costs),reference_outcomes=torch.cat(reference_outcomes))
        collected['precision']=reactive_precision(collected['costs'])
        result=dict(data=collected,prior=original['prior'][selection[:end]],settings=settings,
            training_count=end,complete=end==len(selection),selection=selection[:end],noise_seed=args.seed,
            credit_batch=args.reactive_batch,credit_design=design,
            collecting_policy=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                continuation='same recurrent parent at every decision boundary'),
            reference_policy='source with identical first intervention and exactly shared future noise',
            paired_continuation_innovations_sha256=noise_digest.hexdigest(),
            source_data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest())
        atomic_torch(args.output/f'reactive-model-prior-data-{args.seed}.pth',result)
        print(json.dumps(dict(states=end,total=len(selection),seconds=time.monotonic()-started)),flush=True)


def augment_reactive_prior_anchor(args):
    """Measure the zero-reference control with the panel's exact same futures.

    The continuation remains the evaluated recurrent policy. This anchors
    its cooperative effects relative to the single/control-zero axes without
    substituting a source-policy continuation or changing earlier outcomes.
    """
    from policy.interaction_sac import MirrorCoordinationGate
    torch.set_num_threads(2)
    saved=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    data=saved['data']
    if not saved.get('complete',False) or 'evaluation_options' not in data or saved.get('credit_design') is None:
        raise ValueError('A completed active conditional panel is required.')
    path=Path(saved['collecting_policy']['path']);parent=torch.load(path,map_location='cpu',weights_only=False)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=saved['collecting_policy']['sha256']:
        raise ValueError('The evaluated parent policy changed.')
    settings=saved['settings'];source=ReactiveSource(args.source_device)
    actor=ReactiveModelPriorGate(settings,parent['calibration'],parent['risk'],parent['model_repeats'],source,
        interaction=parent['interaction']);actor.load_state_dict(parent['actor'])
    generator=torch.Generator().manual_seed(saved['noise_seed']);count,repeats,branches=data['costs'].shape
    n=data['base'].shape[1];batch=saved.get('credit_batch',args.reactive_batch)
    noises=[];original_digest=hashlib.sha256();expanded_digest=hashlib.sha256()
    for begin in range(0,count,batch):
        size=min(begin+batch,count)-begin
        noise=torch.randn(size*repeats,300,2+3*(n+1),generator=generator)
        original_digest.update(noise.repeat_interleave(branches,0).numpy().tobytes())
        expanded_digest.update(noise.repeat_interleave(branches+1,0).numpy().tobytes());noises.append(noise)
    if original_digest.hexdigest()!=saved['paired_continuation_innovations_sha256']:
        raise ValueError('The recorded shared panel futures could not be reproduced exactly.')
    costs=[];outcomes=[];references=[];reference_outcomes=[];started=time.monotonic()
    for block,begin in enumerate(range(0,count,batch)):
        end=min(begin+batch,count);size=end-begin
        packet={k:v[begin:end].repeat_interleave(repeats,0) for k,v in data['packet'].items()}
        first=torch.zeros(size*repeats,n)
        cost,outcome,_=reactive_rollout(actor,source,packet,noises[block],settings,first=first)
        reference,reference_outcome,_=reactive_rollout(MirrorCoordinationGate(16),source,packet,noises[block],settings,first=first)
        costs.append(cost.reshape(size,repeats,1));outcomes.append(outcome.reshape(size,repeats,1))
        references.append(reference.reshape(size,repeats,1));reference_outcomes.append(reference_outcome.reshape(size,repeats,1))
        collected=reactive_subset(data,torch.arange(end))
        for key,values in (('costs',costs),('outcomes',outcomes),('reference_costs',references),('reference_outcomes',reference_outcomes)):
            collected[key]=torch.cat((data[key][:end],torch.cat(values)),-1)
        collected['evaluation_options']=torch.cat((data['evaluation_options'][:end],torch.zeros(end,1,dtype=torch.long)),-1)
        collected['precision']=reactive_precision(collected['costs'])
        result=dict(saved,data=collected,prior=saved['prior'][:end],selection=saved['selection'][:end],
            training_count=end,complete=end==count,credit_batch=batch,
            paired_continuation_innovations_sha256=expanded_digest.hexdigest(),
            anchor_augmentation=dict(original_data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
                original_innovations_sha256=original_digest.hexdigest(),continuation='unchanged recurrent parent'),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_torch(args.output/f'reactive-model-prior-data-{args.seed}.pth',result)
        print(json.dumps(dict(anchored_states=end,total=count,seconds=time.monotonic()-started)),flush=True)


def check_reactive_prior(args):
    """A bounded, matched-future development check from stored root states."""
    from policy.interaction_sac import MirrorCoordinationGate
    torch.set_num_threads(2)
    if args.root_source is None or args.bootstrap_seed is None:
        raise ValueError('Provide the root bank and an explicit future seed.')
    arms=args.evaluation_arms.split(',');paths={}
    for arm in arms:
        if arm=='constant_bias':continue
        name='difference' if arm in ('interaction-off','permuted') else arm
        paths[name]=args.output/f'reactive-{name}-{args.seed}.pth'
    models={arm:torch.load(path,map_location='cpu',weights_only=False) for arm,path in paths.items()}
    settings=next(iter(models.values()))['settings']
    comparable=lambda value:{k:v for k,v in value.items() if k!='coordination_dictionary'}
    if any(comparable(m['settings'])!=comparable(settings) for m in models.values()):raise ValueError('Mismatched gate settings.')
    code=Path(__file__);core=code.parent/'policy/interaction_sac.py'
    frozen=dict(implementation=hashlib.sha256(code.read_bytes()+core.read_bytes()).hexdigest(),
        source=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),models={a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in paths.items()})
    packet,roots_hash=reactive_roots(args.root_source,args.split,args.offset+args.count)
    packet={k:v[args.offset:].repeat_interleave(args.repeats,0) for k,v in packet.items()}
    source=ReactiveSource(args.source_device);count=len(packet['obs']);n=settings['defenders']
    noise=torch.randn(count,300,2+3*(n+1),generator=torch.Generator().manual_seed(args.bootstrap_seed))
    costs={};outcomes={};started=time.monotonic()
    path=args.output/f'reactive-model-check-{args.seed}-{args.split}-{args.offset}-{args.count}-{args.bootstrap_seed}.pth'
    metadata=dict(settings=settings,frozen=frozen,root_sha256=roots_hash,split=args.split,
        offset=args.offset,roots=args.count,repeats=args.repeats,noise_seed=args.bootstrap_seed)
    for arm in arms:
        if arm=='constant_bias':actor=MirrorCoordinationGate(16)
        else:
            saved=models['difference' if arm in ('interaction-off','permuted') else arm]
            if saved['kind']=='reactive-factor-kernel-v1':
                actor=ReactiveFactorKernelGate(settings,saved['maps'],saved['weights'],saved['risks'],
                    interaction=saved.get('interaction',True) and arm!='interaction-off',probe=arm if arm=='permuted' else 'aligned')
            else:
                actor=reactive_prior_actor(saved,source,interaction=saved.get('interaction',True) and arm!='interaction-off')
            actor.load_state_dict(saved['actor'])
        execution_settings=settings if arm=='constant_bias' else saved['settings']
        cost,outcome,_=reactive_rollout(actor,source,packet,noise,execution_settings)
        costs[arm]=cost;outcomes[arm]=outcome
        atomic_torch(path,dict(**metadata,costs=costs,outcomes=outcomes,complete=False))
        print(json.dumps(dict(arm=arm,cost=float(cost.mean()),captures=int((outcome==3).sum()),
            episodes=count,seconds=time.monotonic()-started)),flush=True)
    if frozen['source']!=hashlib.sha256(SOURCE.read_bytes()).hexdigest() or frozen['implementation']!=hashlib.sha256(code.read_bytes()+core.read_bytes()).hexdigest() or any(
        frozen['models'][a]!=hashlib.sha256(p.read_bytes()).hexdigest() for a,p in paths.items()):
        raise RuntimeError('The frozen check implementation or policy changed.')
    contrasts={}
    if 'difference' in costs:
        for arm,cost in costs.items():
            if arm=='difference':continue
            gain=(cost-costs['difference']).reshape(args.count,args.repeats).mean(-1)
            contrasts[arm]=dict(gain=float(gain.mean()),root_standard_error=float(gain.std()/args.count**.5))
    atomic_torch(path,dict(**metadata,costs=costs,outcomes=outcomes,contrasts=contrasts,complete=True))
    print(json.dumps(dict(contrasts=contrasts)),flush=True)


@torch.no_grad()
def conditional_pair_decision(actor,packet,base,settings,probe='aligned'):
    """One identified conditional policy improvement per learned value block.

    Every block holds all but two receivers at its predecessor's CURRENT
    decision. The uncertainty is that of the paired value DIFFERENCE, not
    two unrelated pessimistic values. Prefix decisions are recomputed from
    today's candidates, so no historical correction is frozen in time.
    """
    from policy.interaction_sac import CandidateCoordinationGate
    count,n=base.shape[:2];device=base.device
    current=torch.zeros_like(base[...,2]);choice=torch.zeros(count,dtype=torch.long,device=device)
    reference=reactive_reference(base,settings)
    if reference is None:raise ValueError('Conditional candidate policy needs full-range fusion.')
    pairs=[(i,j) for i in range(n) for j in range(i+1,n)];pair_count=len(pairs)
    pair_indices=torch.tensor(pairs,device=device)
    eligible=packet['obs'][...,2]<settings.get('control_radius',30.)
    prepared=actor.prepare_factors(packet,base,probe)
    for coefficient,risk,factors in prepared:
        if float(coefficient)==0.:continue
        before=current
        origin=torch.stack([CandidateCoordinationGate.energies(f,before[:,None])[:,0] for f in factors])
        def objective(programs):
            delta=torch.stack([CandidateCoordinationGate.energies(f,programs) for f in factors])-origin[...,None]
            value=delta.mean(0)+risk*delta.std(0,unbiased=False)
            return coefficient*value/settings['temperature']+settings['prior']*(programs-before[:,None]).square().sum(-1)
        levels=torch.stack((before,torch.zeros_like(before),-reference,1.-reference,-.5*reference,.5*(1.-reference)),1)
        trials=before[:,None,None].expand(-1,pair_count,36,-1).clone()
        for pair,(i,j) in enumerate(pairs):
            left=levels[:,:,i].repeat_interleave(6,1);right=levels[:,:,j].repeat(1,6)
            trials[:,pair,:,i]=torch.where(eligible[:,i,None],left,before[:,i,None])
            trials[:,pair,:,j]=torch.where(eligible[:,j,None],right,before[:,j,None])
        selected=objective(trials.flatten(1,2)).reshape(count,pair_count,36).argmin(-1)
        candidates=trials.gather(2,selected[:,:,None,None].expand(-1,-1,1,n)).squeeze(2)
        if settings.get('refine_gates',False):
            offsets=base.new_tensor([-2.,-1.,0.,1.,2.])
            for step in (.125,.0625,.03125):
                for side in range(2):
                    receivers=pair_indices[:,side]
                    held=candidates.gather(2,receivers[None,:,None].expand(count,-1,1)).squeeze(-1)
                    bound=reference[:,receivers]
                    values=(held[:,:,None]+step*offsets).clamp(-bound[:,:,None],1.-bound[:,:,None])
                    values=torch.where(eligible[:,receivers,None],values,held[:,:,None])
                    trials=candidates[:,:,None].expand(-1,-1,5,-1).clone()
                    trials.scatter_(3,receivers[None,:,None,None].expand(count,-1,5,1),values[:,:,:,None])
                    selected=objective(trials.flatten(1,2)).reshape(count,pair_count,5).argmin(-1)
                    candidates=trials.gather(2,selected[:,:,None,None].expand(-1,-1,1,n)).squeeze(2)
        candidates=torch.cat((before[:,None],candidates),1)
        choice=objective(candidates).argmin(-1)
        current=candidates[torch.arange(count,device=device),choice]
    return choice,current


@torch.no_grad()
def reactive_decision(actor,packet,base,settings,probe='aligned'):
    """Optimize the learned joint fusion energy over the full gate range."""
    if isinstance(actor,nn.Module):
        device=next(actor.parameters(),base).device
        if device!=base.device:
            choice,program=reactive_decision(actor,{k:v.to(device) for k,v in packet.items()},base.to(device),settings,probe)
            return choice.to(base.device),program.to(base.device)
    if hasattr(actor,'blocks') and not len(actor.blocks):
        return torch.zeros(len(base),dtype=torch.long,device=base.device),torch.zeros_like(base[...,2])
    if hasattr(actor,'decide'):return actor.decide(packet,base,settings,probe)
    if settings.get('conditional_policy',False):
        return conditional_pair_decision(actor,packet,base,settings,probe)
    reference=reactive_reference(base,settings)
    options=reactive_options(packet,settings.get('all_gates',False),settings.get('control_radius',30.),settings.get('gate_levels',3),reference)
    if not settings.get('refine_gates',False):
        choice=actor.log_policy(packet,base,options,settings['temperature'],settings['prior'],probe).argmax(-1)
        return choice,options[torch.arange(len(base)),choice]
    if reference is None or not settings.get('all_gates',False) or not hasattr(actor,'prepare_factors'):
        raise ValueError('Continuous joint refinement requires full-range mirror gates.')
    prepared=actor.prepare_factors(packet,base,probe)
    def objective(programs):
        return actor.prepared_energy(prepared,programs)/settings['temperature']+settings['prior']*programs.square().sum(-1)
    choice=objective(options).argmin(-1);current=options[torch.arange(len(base)),choice].clone()
    eligible=packet['obs'][...,2]<settings.get('control_radius',30.)
    offsets=base.new_tensor([-2.,-1.,0.,1.,2.])
    # The complete joint grid initializes the coordinated solution. Refine
    # its actual fusion coefficients while retaining every other gate. Reuse
    # learned factors; this performs no environment rollout or value fitting.
    for step in (.125,.0625,.03125):
        for receiver in range(base.shape[1]):
            trials=current[:,None].expand(-1,5,-1).clone()
            value=(current[:,receiver,None]+step*offsets).clamp(-reference[:,receiver,None],1.-reference[:,receiver,None])
            trials[:,:,receiver]=torch.where(eligible[:,receiver,None],value,current[:,receiver,None])
            selected=objective(trials).argmin(-1)
            current=trials[torch.arange(len(base)),selected]
    return choice,current


def reactive_physics_step(state,measured,agility,action,noise):
    dynamics=DifferentiableInterventionDynamics
    boids=dynamics.boids(state,measured)[0]
    gate=action[...,2:4] if action.shape[-1]==4 else action[...,2:3]
    nominal=gate*(750.*action[...,:2]+250.)+(1.-gate)*boids
    motion=torch.cat((state[:,:-1,:3],measured[:,:-1]),-1)
    from train_response_fusion import SafetyDerivative
    thrust=torch.cat([SafetyDerivative.apply(motion[i:i+32],nominal[i:i+32]) for i in range(0,len(state),32)])
    attacker=dynamics.thrust(dynamics.apf(state)+.3*noise[:,:2],state[:,-1,2],agility[:,None])
    current=noise[:,2:].reshape(state.shape[0],state.shape[1],3)
    current=torch.cat((current[:,1:],current[:,:1]),1)*state.new_tensor([.2,.2,.02])
    return dynamics.vessel(state,torch.cat((thrust,attacker[:,None]),1),current)


@torch.no_grad()
def reactive_rollout(actor,source,packet,innovations,settings,first=None,collect=False,return_prefix=False):
    """Real hydrodynamics and source CBF, followed by the collecting policy.

    An intervention changes only the first H-step gate option. Subsequent
    decisions are recomputed by the SAME actor at the same H-step boundaries.
    Paired rows receive common exogenous innovations at every physical step.
    """
    dynamics=DifferentiableInterventionDynamics
    # Keep the optimizer's actor on CPU. A frozen execution copy batches the
    # same learned gate on the source's accelerator during simulated futures.
    execution_actor=actor
    if isinstance(actor,nn.Module) and str(getattr(source,'device','cpu')).startswith('cuda') and (not hasattr(actor,'blocks') or len(actor.blocks)):
        execution_actor=copy.deepcopy(actor).to(source.device).eval().requires_grad_(False)
    if isinstance(execution_actor,ReactiveModelPriorGate):execution_actor.collect_predictions=collect
    state,measured,agility,remaining=dynamics.from_packet({k:v.double() for k,v in packet.items()})
    count,n=packet['obs'].shape[:2];active=torch.arange(count)
    total=torch.zeros(count);outcome=torch.zeros(count,dtype=torch.long)
    p={k:v.float() for k,v in packet.items()};frames=[];period=settings['period']
    if return_prefix:
        if collect or not 2<=period<=300:raise ValueError('Nominal response extraction requires a valid fixed control window.')
        relative=(state[:,-1:,:2]-state[:,:-1,:2]).float()
        prefix_response=relative[:,:,None].expand(-1,-1,2,-1).clone()
        prefix_steps=(max(1,period//2),period)
    if settings.get('coupled_potential',False):p.setdefault('target_turn',torch.zeros(count))
    upper=torch.triu(torch.ones(n,n,dtype=torch.bool),1)
    for step in range(300):
        if not len(active):break
        base=source(p['obs'].flatten(0,1),True,False)[0].reshape(len(active),n,3)
        if step%period==0:
            if step==0 and first is not None:
                # The specified intervention is the first deployed control.
                # Do not run an expensive gate search whose result is discarded.
                correction=first[active];choice=torch.full((len(active),),-1,dtype=torch.long)
            else:choice,correction=reactive_decision(execution_actor,p,base,settings)
            if collect:
                frames.append(dict(packet={k:v.clone() for k,v in p.items()},base=base.clone(),
                    choice=choice,program=correction.clone(),indices=active.clone(),
                    before=total[active].clone()))
                if hasattr(execution_actor,'last_prediction') and bool((choice>=0).all()):
                    frames[-1].update(model_prior=execution_actor.last_prediction['prior'].clone(),
                        model_objective=execution_actor.last_prediction['objective'].clone())
                    if 'response' in execution_actor.last_prediction:
                        frames[-1]['model_response']=execution_actor.last_prediction['response'].clone()
        action=reactive_action(base,correction,settings)
        duration=(60.*remaining).clamp(max=.2)
        previous_heading=p.get('target_heading')
        state,measured=reactive_physics_step(state,measured,agility,action,innovations[active,step].double())
        remaining=(remaining-.2/60.).clamp_min(0.)
        distance=(state[:,:-1,:2]-state[:,-1:,:2]).norm(dim=-1)
        separation=(state[:,:-1,None,:2]-state[:,None,:-1,:2]).norm(dim=-1)[:,upper]
        collision=separation.amin(-1)<=5.
        breach=state[:,-1,:2].norm(dim=-1)<=15.
        failure=breach|collision
        capture=(distance.amin(-1)<5.)&~failure
        done=failure|capture|(remaining<1e-5)
        if return_prefix and step<period:
            relative=(state[:,-1:,:2]-state[:,:-1,:2]).float()
            for slot,boundary in enumerate(prefix_steps):
                take=torch.ones_like(done) if step+1==boundary else done if step+1<boundary else torch.zeros_like(done)
                prefix_response[active[take],:,slot]=relative[take]
        outcome[active[done]]=torch.where(breach[done],1,torch.where(collision[done],2,torch.where(capture[done],3,4)))
        total[active]+=(duration+failure*60.*remaining).float()
        keep=~done;active=active[keep]
        state,measured,agility,remaining,correction=(v[keep] for v in (state,measured,agility,remaining,correction))
        if len(active):
            p={k:v.float() for k,v in dynamics.packet(state,measured,agility,remaining).items()}
            if 'target_heading' in packet:p['target_heading']=state[:,-1,2].float()
            if 'time_remaining' in packet:p['time_remaining']=remaining.float()
            if settings.get('coupled_potential',False):
                angle=p['target_heading']-previous_heading[keep]
                p['target_turn']=torch.atan2(angle.sin(),angle.cos())/.2
    if len(active):raise RuntimeError('Reactive credit must reach termination.')
    data=None
    if collect:
        data=dict(packet={k:torch.cat([f['packet'][k] for f in frames]) for k in p},
            base=torch.cat([f['base'] for f in frames]),
            origin=torch.cat([f['indices'] for f in frames]),
            program=torch.cat([f['program'] for f in frames]),choice=torch.cat([f['choice'] for f in frames]),
            returns=torch.cat([total[f['indices']]-f['before'] for f in frames]))
        for key in ('model_prior','model_objective','model_response'):
            if all(key in frame for frame in frames):data[key]=torch.cat([frame[key] for frame in frames])
    if return_prefix:data=dict(prefix_response=prefix_response)
    return total,outcome,data


@torch.no_grad()
def reactive_credits(actor,source,packet,settings,repeats,generator,batch=4,complete_pairs=False):
    """Full ternary pair labels identify unary, curvature and mixed effects."""
    results=[];option_rows=[];n=packet['obs'].shape[1]
    base=source(packet['obs'].flatten(0,1),True,False)[0].reshape(-1,n,3)
    for start in range(0,len(packet['obs']),batch):
        p={k:v[start:start+batch] for k,v in packet.items()};roots=len(p['obs'])
        reference=reactive_reference(base[start:start+batch],settings)
        programs=reactive_options(p,reference=reference)
        if settings.get('all_gates',False):
            _,held=reactive_decision(actor,p,base[start:start+batch],settings)
            eligible=p['obs'][...,2]<settings.get('control_radius',30.)
            if complete_pairs:
                if reference is None or not settings.get('conditional_policy',False):
                    raise ValueError('Complete pair credit requires current-centered full-range decisions.')
                # One current reference, two single-receiver endpoints, and
                # all four joint endpoints for every pair. For four boats
                # this is 33 branches, compared with 55 duplicated faces.
                faces=[]
                endpoints=torch.stack((-reference,1.-reference),1)
                for i in range(n):
                    for a in range(2):
                        face=held.clone();face[:,i]=torch.where(eligible[:,i],endpoints[:,a,i],held[:,i]);faces.append(face)
                for i in range(n):
                    for j in range(i+1,n):
                        for a in range(2):
                            for b in range(2):
                                face=held.clone()
                                face[:,i]=torch.where(eligible[:,i],endpoints[:,a,i],held[:,i])
                                face[:,j]=torch.where(eligible[:,j],endpoints[:,b,j],held[:,j])
                                faces.append(face)
                programs=torch.stack(faces,1)
            else:
                priority=torch.rand(roots,n,generator=generator)+2.*(~eligible)
                receivers=priority.argsort(-1)[:,:2]
                grid=torch.cartesian_prod(torch.tensor([0.,-1.,1.]),torch.tensor([0.,-1.,1.]))
                programs=held[:,None].expand(-1,9,-1).clone()
                interventions=grid[None].expand(roots,-1,-1)
                if reference is not None:
                    local=reference.gather(1,receivers)[:,None]
                    interventions=interventions*torch.where(interventions>=0,1.-local,local)
                programs.scatter_(2,receivers[:,None].expand(-1,9,-1),interventions)
                programs*=eligible[:,None]
            if settings.get('conditional_policy',False):
                # The first branch is exactly the collecting policy's joint
                # decision, including any continuously refined gate values.
                programs=torch.cat((held[:,None],programs),1)
        option_rows.append(programs)
        option_count=programs.shape[1]
        expanded={k:v.repeat_interleave(repeats*option_count,0) for k,v in p.items()}
        noise=torch.randn(roots*repeats,300,2+3*(n+1),generator=generator).repeat_interleave(option_count,0)
        first=programs[:,None].expand(-1,repeats,-1,-1).reshape(-1,n)
        cost,outcome,_=reactive_rollout(actor,source,expanded,noise,settings,first=first)
        results.append(dict(costs=cost.reshape(roots,repeats,option_count),outcomes=outcome.reshape(roots,repeats,option_count)))
        if len(packet['obs'])>=32 and (start+roots)%32==0:
            print(json.dumps(dict(credit_roots=start+roots,total_roots=len(packet['obs']))),flush=True)
    return dict(packet=packet,base=base,options=torch.cat(option_rows),
        costs=torch.cat([r['costs'] for r in results]),outcomes=torch.cat([r['outcomes'] for r in results]))


def reactive_value_loss(model,data,conditional=True,orthogonal=False,decision_scale=0.,control_only=False,preference=False):
    value,energy=model(data['packet'],data['base'],data['options'])
    costs=data['costs'];target=costs.mean(1)
    loss=((value[:,None]+energy-target)/10.).square().mean()
    if conditional:
        if control_only:
            loss=energy.new_zeros(())
        elif orthogonal:
            # Fit the nuisance state value without sending its approximation
            # error into the identifiable control factors. Conditional labels
            # alone update those factors; the total value remains anchored.
            loss=((value-(target-energy.detach()).mean(-1))/10.).square().mean()
        # A complete pair surface is supervised in a basis that annihilates
        # every state-only return shift. Whiten with pooled, shrinkage-regularized
        # future covariance; never give a low-variance accident infinite weight.
        if preference:
            # The probability that one control beats another is estimated
            # within each identical future, before averaging. Long-chase
            # outliers cannot manufacture arbitrarily large gate logits.
            paired=(costs[:,:,:,None]-costs[:,:,None,:])/2.
            probability=paired.sigmoid().mean(1).detach()
            logits=energy[:,:,None]-energy[:,None,:]
            mask=~torch.eye(energy.shape[1],dtype=torch.bool,device=energy.device)
            loss=loss+F.binary_cross_entropy_with_logits(logits[:,mask],probability[:,mask])
        else:
            difference=energy[:,1:]-energy[:,:1]-(target[:,1:]-target[:,:1])
            precision=data['precision']
            equation='bi,bij,bj->b' if precision.ndim==3 else 'bi,ij,bj->b'
            loss=loss+torch.einsum(equation,difference,precision,difference).mean()/difference.shape[-1]
        if decision_scale and not preference:
            # Supervise the very conditional policy improvement that the
            # deployed mirror gate will execute, not only unweighted Q error.
            # Common state-return shifts cancel in both distributions.
            old=data['old_log_policy']
            teacher=(old-decision_scale*target).log_softmax(-1).detach()
            proposed=(old-decision_scale*energy).log_softmax(-1)
            loss=loss+(teacher.exp()*(teacher-proposed)).sum(-1).mean()/decision_scale**2
    return loss


def reactive_precision(costs,local=False):
    difference=costs[...,1:]-costs[...,:1]
    centered=difference-difference.mean(1,keepdim=True)
    if local:
        covariance=torch.einsum('bri,brj->bij',centered,centered)/(max(1,costs.shape[1]-1)*costs.shape[1])
        covariance=.5*covariance+.5*covariance.mean(0,keepdim=True)
        covariance=.8*covariance+.2*torch.diag_embed(torch.diagonal(covariance,dim1=-2,dim2=-1))
    else:
        covariance=torch.einsum('bri,brj->ij',centered,centered)/(len(costs)*max(1,costs.shape[1]-1)*costs.shape[1])
        covariance=.8*covariance+.2*torch.diag(torch.diagonal(covariance))
    return torch.linalg.inv(covariance+torch.eye(difference.shape[-1],device=costs.device,dtype=costs.dtype))


def reactive_subset(data,index):
    return {k:({j:x[index] for j,x in v.items()} if k=='packet' else v if k=='parts' or (k=='precision' and v.ndim==2) else v[index]) for k,v in data.items()}


def reactive_state_groups(data):
    """Keep a source trajectory, including its different times, in one fold."""
    if 'origin' in data:
        return torch.unique(data['origin'],return_inverse=True)[1]
    # Compatibility for credits recorded before trajectory provenance.
    state=torch.cat([data['packet'][key].flatten(1) for key in ('central','motion','obs')],-1)
    return torch.unique(state,dim=0,return_inverse=True)[1]


def reactive_root_partition(count,reserved):
    """A fixed split survives optimizer seeds and successive policy updates."""
    if reserved<0 or reserved>=count:raise ValueError('Reserve fewer roots than the training bank contains.')
    if not reserved:return torch.arange(count),torch.empty(0,dtype=torch.long)
    order=torch.randperm(count,generator=torch.Generator().manual_seed(71933))
    return order[:-reserved],order[-reserved:]


@torch.no_grad()
def reactive_recover_origins(data,training,source,settings,count,seed):
    """Recover legacy zero-policy provenance only after exact state matching."""
    from policy.interaction_sac import MirrorCoordinationGate
    generator=torch.Generator().manual_seed(seed+1000000)
    selection=torch.randperm(len(training['obs']),generator=generator)[:count]
    packet={k:v[selection] for k,v in training.items()}
    noise=torch.randn(len(selection),300,2+3*(settings['defenders']+1),generator=generator)
    _,_,trajectory=reactive_rollout(MirrorCoordinationGate(16),source,packet,noise,settings,collect=True)
    indices=torch.randperm(len(trajectory['base']),generator=generator)[:count]
    if len(indices)!=len(data['base']) or any(not torch.equal(v[indices],data['packet'][k])
            for k,v in trajectory['packet'].items() if k in data['packet']):
        raise ValueError('Replayed public states differ; cannot recover the provenance of these credits.')
    packet=dict(data['packet'])
    if 'target_turn' in trajectory['packet']:packet['target_turn']=trajectory['packet']['target_turn'][indices]
    return dict(data,packet=packet,origin=selection[trajectory['origin'][indices]])


def reactive_group_sample(groups,generator=None):
    count=int(groups.max())+1
    selected=torch.randint(count,(count,),generator=generator)
    indices=torch.cat([(groups==group).nonzero().flatten() for group in selected])
    included=torch.zeros(count,dtype=torch.bool);included[selected]=True
    return indices,~included[groups]


@torch.no_grad()
def reactive_calibrate_risk(critic,data,excluded,old,settings,groups):
    """Select pessimism from predictions made without the state's labels.

    A bootstrap member is eligible only if EVERY intervention at that state
    was excluded from its fit. Repeated futures are never treated as new
    physical states. This tunes the optimizer on training data, not a test.
    """
    energies=critic.estimates(data['packet'],data['base'],data['options'])[1]
    if settings.get('conditional_policy',False):energies=energies-energies[:,:,:1]
    mask=torch.stack(excluded);count=mask.sum(0);valid=count>=2
    if int(valid.sum())<32:return dict(selected_risk=critic.risk,eligible_states=0,skipped=True)
    weight=mask[...,None].to(energies.dtype)
    mean=(weight*energies).sum(0)/count.clamp_min(1)[:,None]
    std=((weight*(energies-mean).square()).sum(0)/count.clamp_min(1)[:,None]).sqrt()
    truth=data['costs'].mean(1)
    risks=(2.,1.,.5,.25,0.)
    costs=[];gains=[];unique=groups[valid].unique()
    reference=truth.gather(1,old.argmax(-1)[:,None]).squeeze(-1)
    for risk in risks:
        choice=(old-settings['improvement']*(mean+risk*std)).argmax(-1)
        selected=truth.gather(1,choice[:,None]).squeeze(-1)
        costs.append(torch.stack([selected[valid&(groups==group)].mean() for group in unique]).mean())
        gains.append(torch.stack([(reference-selected)[valid&(groups==group)].mean() for group in unique]).mean())
    best=int(torch.stack(costs).argmin());critic.risk=risks[best]
    return dict(selected_risk=risks[best],candidate_risks=list(risks),
        mean_costs=[float(v) for v in costs],paired_gains=[float(v) for v in gains],
        eligible_states=len(unique),eligible_interventions=int(valid.sum()),skipped=False)


def reactive_update(actor,critic,data,conditional,settings,updates,seed):
    """Critic improvement followed by a KL-regularized joint policy update."""
    torch.manual_seed(seed);data['precision']=reactive_precision(data['costs'],settings.get('local_precision',False))
    decision_scale=settings['improvement'] if settings.get('decision_loss',False) else 0.
    with torch.no_grad():
        if settings.get('conditional_policy',False):
            _,reference=reactive_decision(actor,data['packet'],data['base'],settings)
            data['reference_program']=reference
            data['old_log_policy']=(-settings['prior']*(data['options']-reference[:,None]).square().sum(-1)).log_softmax(-1)
        else:data['old_log_policy']=actor.log_policy(data['packet'],data['base'],data['options'],settings['temperature'],settings['prior'])
    if settings.get('response_encoding',False):
        from policy.interaction_sac import CandidateCoordinationGate
        response,paths=CandidateCoordinationGate.candidate_responses(data['packet'],data['base'])
        data['packet'].update(candidate_response=response,candidate_paths=paths)
        if settings.get('reaction_encoding',False):
            data['packet']['candidate_reaction']=CandidateCoordinationGate.candidate_reactions(data['packet'],paths)
    if settings.get('potential_value',False):
        data['packet']=CandidateCoordinationGate.cache_potentials(data['packet'],data['base'],settings['bias'],critic.peer_candidates,curved=settings.get('curved_potential',False),constrained=settings.get('constrained_potential',False),coupled=settings.get('coupled_potential',False))
    critic.train()
    members=list(critic.members) if hasattr(critic,'members') else [critic]
    calibrate=settings.get('calibrate_risk',False)
    if calibrate and len(members)<3:raise ValueError('Out-of-bag calibration needs at least three members.')
    groups=reactive_state_groups(data) if calibrate else None
    excluded=[]
    for member_index,member in enumerate(members):
        torch.manual_seed(seed+104729*member_index)
        bootstrap=data
        if len(members)>1:
            if calibrate:
                index,held_out=reactive_group_sample(groups);excluded.append(held_out)
            else:index=torch.randint(len(data['base']),(len(data['base']),))
            bootstrap=reactive_subset(data,index)
            repeats=bootstrap['costs'].shape[1]
            future=torch.randint(repeats,bootstrap['costs'].shape[:2])[...,None].expand_as(bootstrap['costs'])
            bootstrap['costs']=bootstrap['costs'].gather(1,future)
            bootstrap['outcomes']=bootstrap['outcomes'].gather(1,future)
        if settings.get('potential_value',False):
            # The response potential has only 96 shared control coefficients.
            # Solve its full paired objective, instead of applying many noisy
            # miniature updates to tiny physical feature differences.
            controls=[*member.motion_potential.parameters(),*member.pair_potential.parameters()]
            baseline_parameters=list(member.baseline.parameters())
            if member.complete_state:baseline_parameters.extend(member.global_value.parameters())
            parameters=controls if conditional else [*controls,*baseline_parameters]
            optimizer=torch.optim.LBFGS(parameters,lr=.7,max_iter=min(updates,200),line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss=reactive_value_loss(member,bootstrap,conditional,settings.get('exact_mirror',False),decision_scale,control_only=conditional,preference=settings.get('preference_value',False))
                energy=member(bootstrap['packet'],bootstrap['base'],bootstrap['options'])[1]
                loss=loss+.03*(energy-energy[:,:1]).square().mean()
                loss.backward()
                return loss
            optimizer.step(closure)
            if conditional:
                # A detached moving state-value target is not a scalar
                # objective for a joint line search. Fit the nuisance value
                # only after fixing the control potential coefficients.
                with torch.no_grad():
                    energy=member(bootstrap['packet'],bootstrap['base'],bootstrap['options'])[1]
                    target=(bootstrap['costs'].mean(1)-energy).mean(-1)
                value_optimizer=torch.optim.Adam(baseline_parameters,lr=.005)
                for _ in range(min(updates,200)):
                    value=member(bootstrap['packet'],bootstrap['base'],bootstrap['options'])[0]
                    value_loss=((value-target)/10.).square().mean()
                    value_optimizer.zero_grad(set_to_none=True);value_loss.backward();value_optimizer.step()
        else:
            optimizer=torch.optim.AdamW(member.parameters(),lr=.001,weight_decay=.01)
            for _ in range(updates):
                ix=torch.randint(len(bootstrap['base']),(min(64,len(bootstrap['base'])),))
                loss=reactive_value_loss(member,reactive_subset(bootstrap,ix),conditional,settings.get('exact_mirror',False),decision_scale,preference=settings.get('preference_value',False))
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(member.parameters(),5.);optimizer.step()
    critic.eval()
    with torch.no_grad():
        old=data['old_log_policy']
        calibration=reactive_calibrate_risk(critic,data,excluded,old,settings,groups) if calibrate else None
        if settings.get('conditional_policy',False):
            extended=torch.cat((data['options'],data['reference_program'][:,None]),1)
            predictions=torch.stack([member(data['packet'],data['base'],extended)[1] for member in members])
            differences=predictions[:,:,:-1]-predictions[:,:,-1:]
            cost=differences.mean(0)+getattr(critic,'risk',0.)*differences.std(0,unbiased=False)
        else:_,cost=critic(data['packet'],data['base'],data['options'])
        # Subtract the OLD policy's conditional mean, then solve the exact
        # entropy-regularized finite joint improvement problem. The detached
        # teacher cannot be changed by the actor's projection gradients.
        advantage=cost-(old.exp()*cost).sum(-1,keepdim=True)
        teacher=(old-settings['improvement']*advantage).log_softmax(-1).exp()
    if settings.get('exact_mirror',False):
        actor.add_value(critic,settings['temperature']*settings['improvement'])
    else:
        optimizer=torch.optim.AdamW(actor.parameters(),lr=.0005,weight_decay=.01)
        for _ in range(max(100,updates//2)):
            ix=torch.randint(len(data['base']),(min(64,len(data['base'])),));batch=reactive_subset(data,ix)
            logp=actor.log_policy(batch['packet'],batch['base'],batch['options'],settings['temperature'],settings['prior'])
            loss=-(teacher[ix]*logp).sum(-1).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),5.);optimizer.step()
    with torch.no_grad():
        current=((old-settings['improvement']*cost).log_softmax(-1) if settings.get('conditional_policy',False) else
            actor.log_policy(data['packet'],data['base'],data['options'],settings['temperature'],settings['prior']))
        prediction=cost-cost[:,:1];truth=data['costs'].mean(1);truth=truth-truth[:,:1]
        preference=conditional and settings.get('preference_value',False)
        diagnostics=(dict(value_units='paired_log_odds',preference_error=float((prediction.sigmoid()-(data['costs']-data['costs'][...,:1]).div(2.).sigmoid().mean(1)).abs().mean()))
            if preference else dict(value_units='seconds',conditional_mae=float((prediction-truth).abs().mean())))
        return dict(value_loss=float(reactive_value_loss(critic,data,conditional,settings.get('exact_mirror',False),decision_scale,preference=preference)),**diagnostics,
            policy_kl=float((current.exp()*(current-old)).sum(-1).mean()),
            changed=float((current.argmax(-1)!=old.argmax(-1)).float().mean()),
            training_gain=float(-truth.gather(1,current.argmax(-1)[:,None]).mean()),risk_calibration=calibration)


@torch.no_grad()
def reactive_factor_inputs(packet,base,probe='aligned'):
    """Own candidates and directed candidate-interaction messages, public only."""
    from policy.interaction_sac import CandidateCoordinationGate as Gate
    node,_,geometry,state=Gate.public_inputs(packet,base,True)
    response,paths=Gate.candidate_responses(packet,base)
    public=torch.cat((geometry.mean(2),geometry.amax(2)),-1)
    own=torch.cat((node,response,public),-1);incoming=own;peer_paths=paths
    if probe=='permuted':
        changed=dict(packet,obs=packet['obs'].clone());changed['obs'][...,12:14]=changed['obs'][...,12:14].roll(1,1)
        incoming_node=Gate.public_inputs(changed,base.roll(1,1),True)[0]
        incoming_response,peer_paths=Gate.candidate_responses(changed,base.roll(1,1))
        incoming=torch.cat((incoming_node,incoming_response,public),-1)
    n=base.shape[1]
    reaction=Gate.candidate_reactions(packet,paths,peer_paths)
    pair=torch.cat((own[:,:,None].expand(-1,-1,n,-1),incoming[:,None].expand(-1,n,-1,-1),geometry,reaction),-1)
    global_state=torch.cat((state.mean(1),state.amin(1),state.amax(1)),-1)
    # Preserve the other vessels' candidate responses and their association
    # with relative geometry; mean/max geometry alone loses this relation.
    fleet=incoming[:,None].expand(-1,n,-1,-1).clone()
    index=torch.arange(n,device=base.device);fleet[:,index,index]=own
    fleet=torch.cat((fleet,geometry),-1)
    order=geometry[...,6].argsort(dim=-1,stable=True)
    fleet=fleet.gather(2,order[...,None].expand_as(fleet)).flatten(2)
    pair_fleet=torch.cat((pair,fleet[:,:,None].expand(-1,-1,n,-1)),-1)
    return dict(node=own,pair=pair,state=global_state,pair_fleet=pair_fleet)


@torch.no_grad()
def fit_reactive_feature_map(values,centers=48,dimensions=20):
    """A training-only Nyström dictionary in public physical coordinates."""
    values=values.reshape(-1,values.shape[-1]).double()
    mean=values.mean(0);scale=values.std(0).clamp_min(.1)
    normalized=((values-mean)/scale).clamp(-5.,5.)
    _,vectors=torch.linalg.eigh(normalized.T@normalized/len(normalized))
    projection=vectors[:,-min(dimensions,len(vectors)):]
    projected=normalized@projection
    # Retain physical PCA amplitudes; do not amplify nearly constant axes.
    projected=projected/(projected.square().mean().sqrt().clamp_min(.1))
    projection=projection/(normalized@projection).square().mean().sqrt().clamp_min(.1)
    candidates=projected[torch.linspace(0,len(projected)-1,min(2048,len(projected))).long()]
    selected=[int(candidates.square().sum(-1).argmin())]
    nearest=torch.full((len(candidates),),float('inf'),dtype=torch.float64)
    for _ in range(min(centers,len(candidates))-1):
        distance=(candidates-candidates[selected[-1]]).square().mean(-1)
        nearest=torch.minimum(nearest,distance);selected.append(int(nearest.argmax()))
    centers_tensor=candidates[selected]
    distance=torch.cdist(candidates,centers_tensor).square()/projected.shape[-1]
    bandwidth=distance.amin(-1).median().sqrt().clamp_min(.35)
    return dict(mean=mean.float(),scale=scale.float(),projection=projection.float(),centers=centers_tensor.float(),
        bandwidth=bandwidth.float())


def reactive_feature_map(values,specification):
    spec={k:v.to(values) for k,v in specification.items()}
    projected=((values-spec['mean'])/spec['scale']).clamp(-5.,5.)@spec['projection']
    distance=(projected[...,None,:]-spec['centers']).square().mean(-1)
    radial=torch.exp(-distance/(2.*spec['bandwidth'].square()))
    return torch.cat((torch.ones_like(projected[...,:1]),projected,radial),-1)


def reactive_factor_design(packet,base,programs,maps,bias=-.05):
    inputs=reactive_factor_inputs(packet,base)
    node=reactive_feature_map(inputs['node'],maps['node'])
    pair=reactive_feature_map(inputs['pair'],maps['pair'])
    if 'pair_fleet' in maps:pair=torch.cat((pair,reactive_feature_map(inputs['pair_fleet'],maps['pair_fleet'])),-1)
    n=base.shape[1];pair=pair*(~torch.eye(n,dtype=torch.bool,device=base.device))[None,:,:,None]
    reference=(base[...,2]+bias).clamp(0.,1.)[:,None]
    basis=torch.stack(((-programs/reference.clamp_min(1e-6)).clamp(0.,1.),
        (programs/(1.-reference).clamp_min(1e-6)).clamp(0.,1.)),-1)
    unary=torch.einsum('bnf,bkna->bkfa',node,basis).flatten(-2)
    mixed=(.5*torch.einsum('bijf,bkia,bkjc->bkfac',pair,basis,basis)).flatten(-3)
    state=reactive_feature_map(inputs['state'],maps['state'])
    return unary.double(),mixed.double(),state.double()


class ReactiveFactorKernelGate(nn.Module):
    """A learned candidate graph; no long-horizon response search at inference.

    Every value stage evaluates the preceding recurrent policy. Its paired
    improvement is anchored at that policy's CURRENT joint decision, which
    is recomputed from today's own and incoming candidate controls.
    """
    def __init__(self,settings,maps,weights,risks,interaction=True,probe='aligned'):
        super().__init__();self.settings=settings;self.interaction=interaction;self.probe=probe
        self.map_names=tuple(maps)
        for name,spec in maps.items():
            for key,value in spec.items():self.register_buffer(name+'_'+key,value.float())
        self.register_buffer('weights',weights.float());self.register_buffer('risks',torch.as_tensor(risks).float())
        self.blocks=nn.ModuleList([nn.Identity()])

    def feature_maps(self):
        return {name:{key:getattr(self,name+'_'+key) for key in ('mean','scale','projection','centers','bandwidth')}
            for name in self.map_names}

    def energies(self,packet,base,programs,probe='aligned'):
        inputs=reactive_factor_inputs(packet,base,probe);maps=self.feature_maps()
        node=reactive_feature_map(inputs['node'],maps['node']);pair=reactive_feature_map(inputs['pair'],maps['pair'])
        if 'pair_fleet' in maps:pair=torch.cat((pair,reactive_feature_map(inputs['pair_fleet'],maps['pair_fleet'])),-1)
        n=base.shape[1];pair=pair*(~torch.eye(n,dtype=torch.bool,device=base.device))[None,:,:,None]
        width=node.shape[-1]*2
        unary=torch.einsum('bnf,lmfa->lmbna',node,self.weights[...,:width].reshape(*self.weights.shape[:2],-1,2))
        mixed=torch.einsum('bijf,lmfac->lmbijac',pair,self.weights[...,width:].reshape(*self.weights.shape[:2],-1,2,2))
        reference=(base[...,2]+self.settings['bias']).clamp(0.,1.)[:,None]
        basis=torch.stack(((-programs/reference.clamp_min(1e-6)).clamp(0.,1.),
            (programs/(1.-reference).clamp_min(1e-6)).clamp(0.,1.)),-1)
        energy=torch.einsum('lmbna,bkna->lmbk',unary,basis)
        if self.interaction and probe not in ('state-only','interaction-off'):
            energy=energy+.5*torch.einsum('lmbijac,bkia,bkjc->lmbk',mixed,basis,basis)
        return energy

    def choices(self,estimates,programs):
        choice=torch.zeros(programs.shape[0],dtype=torch.long,device=programs.device)
        limits=self.settings.get('stage_max_changes',[programs.shape[-1]]*len(self.risks))
        for stage,risk,limit in zip(estimates,self.risks,limits):
            reference=stage.gather(-1,choice[None,:,None].expand(len(stage),-1,1))
            difference=stage-reference
            score=difference.mean(0)+risk*difference.std(0,unbiased=False)
            previous=programs.gather(1,choice[:,None,None].expand(-1,1,programs.shape[-1]))
            score=score+self.settings.get('factor_trust',.1)*(programs-previous).square().sum(-1)
            score=score.masked_fill(((programs-previous).abs()>1e-5).sum(-1)>limit,float('inf'))
            choice=score.argmin(-1)
        return choice

    @torch.no_grad()
    def decide(self,packet,base,settings,probe='aligned'):
        device=self.weights.device;packet={k:v.to(device) for k,v in packet.items()};base=base.to(device)
        if self.probe!='aligned':probe=self.probe
        programs=reactive_options(packet,True,settings.get('control_radius',30.),3,reactive_reference(base,settings))
        choice=self.choices(self.energies(packet,base,programs,probe),programs)
        return choice.cpu(),programs[torch.arange(len(choice),device=device),choice].cpu()


def fit_reactive_factor_kernel(args):
    """Exact regularized conditional fitting of a nonlinear candidate graph."""
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    saved=torch.load(args.reactive_bootstrap_data,map_location='cpu',weights_only=False)
    if saved['collecting_actor'].get('coefficients',torch.ones(1)).numel():
        raise ValueError('Initial factor fitting requires source-policy conditional returns.')
    data=saved['data'];groups=reactive_state_groups(data)
    generator=torch.Generator().manual_seed(731907)
    order=torch.randperm(int(groups.max())+1,generator=generator)
    validation_groups=order[:max(1,len(order)//5)]
    train=(~torch.isin(groups,validation_groups)).nonzero().flatten();valid=torch.isin(groups,validation_groups).nonzero().flatten()
    inputs=reactive_factor_inputs({k:v[train] for k,v in data['packet'].items()},data['base'][train])
    maps={key:fit_reactive_feature_map(value,48 if key!='state' else 32,20 if key!='pair' else 28)
        for key,value in inputs.items() if key!='pair_fleet' or args.complete_state}
    unary,mixed,state=reactive_factor_design(data['packet'],data['base'],data['options'],maps,saved['settings']['bias'])
    parts=reactive_credit_parts(data);singles=torch.tensor([1,2,3,6]);joints=torch.tensor([4,5,7,8])
    if data['costs'].shape[-1]!=9:raise ValueError('The initial graph fit expects the complete ternary pair design.')
    costs=data['costs'].double();truth=costs.mean(1);width=unary.shape[-1];all_width=width+mixed.shape[-1]
    settings=dict(saved['settings'],factor_trust=.1,conditional_policy=False)
    def solve(x,y,precision,ridge,weight):
        if precision is None:
            matrix=torch.einsum('bki,bkj,b->ij',x,x,weight)/(len(x)*x.shape[1])
            right=torch.einsum('bki,bk,b->i',x,y,weight)/(len(x)*x.shape[1])
        else:
            matrix=torch.einsum('bki,bkl,blj,b->ij',x,precision,x,weight)/(len(x)*x.shape[1])
            right=torch.einsum('bki,bkl,bl,b->i',x,precision,y,weight)/(len(x)*x.shape[1])
        diagonal=torch.eye(x.shape[-1],dtype=x.dtype)*ridge/x.shape[-1]
        return torch.linalg.solve(matrix+diagonal,right)
    for arm in args.evaluation_arms.split(','):
        conditional=arm in ('difference','additive');interaction=arm in ('difference','absolute')
        candidates=[]
        for ridge in (.1,1.,10.,100.):
            coefficients=[]
            for member in range(args.critic_members):
                g=torch.Generator().manual_seed(args.seed+104729*member)
                row_weight,future_weight=reactive_positive_credit_weights(groups[train],costs.shape[1],g)
                target=(costs[train]*future_weight[:,:,None]).sum(1)
                if conditional:
                    precision=reactive_precision(costs[train][:,:,torch.cat((torch.tensor([0]),singles))],True)
                    node=solve(unary[train][:,singles],target[:,singles]-target[:,:1],precision,ridge,row_weight)
                    edges=torch.zeros(mixed.shape[-1],dtype=torch.float64)
                    if interaction:
                        residual=costs[train]-costs[train][:,:,parts[:,0]]-costs[train][:,:,parts[:,1]]+costs[train][:,:,:1]
                        precision=reactive_precision(residual[:,:,torch.cat((torch.tensor([0]),joints))],True)
                        value=(residual*future_weight[:,:,None]).sum(1)[:,joints]
                        edges=solve(mixed[train][:,joints],value,precision,ridge,row_weight)
                    coefficient=torch.cat((node,edges))
                else:
                    control=torch.cat((unary,mixed if interaction else torch.zeros_like(mixed)),-1)
                    design=torch.cat((state[:,None].expand(-1,control.shape[1],-1),control),-1)
                    noise=costs[train].var(1).mean(-1)/costs.shape[1]+1.
                    coefficient=solve(design[train],target,None,ridge,row_weight/noise)[state.shape[-1]:]
                coefficients.append(coefficient.float())
            coefficients=torch.stack(coefficients)
            prediction=torch.einsum('bkp,mp->mbk',torch.cat((unary[valid],mixed[valid]),-1).float(),coefficients)
            for risk in (1.,.5,0.):
                score=prediction.mean(0)+risk*prediction.std(0,unbiased=False)+settings['factor_trust']*data['options'][valid].square().sum(-1)
                choice=score.argmin(-1);gain=truth[valid,0]-truth[valid].gather(-1,choice[:,None])[:,0]
                grouped=torch.stack([gain[groups[valid]==origin].mean() for origin in validation_groups])
                metric=dict(ridge=ridge,risk=risk,gain=float(grouped.mean()),standard_error=float(grouped.std()/len(grouped)**.5))
                candidates.append((metric,coefficients))
        metric,coefficients=max(candidates,key=lambda item:item[0]['gain']-.5*item[0]['standard_error'])
        actor=ReactiveFactorKernelGate(settings,maps,coefficients[None],[metric['risk']],interaction)
        result=dict(kind='reactive-factor-kernel-v1',settings=settings,maps=maps,weights=coefficients[None],risks=[metric['risk']],
            actor=actor.state_dict(),interaction=interaction,conditional=conditional,arm=arm,seed=args.seed,
            selection=metric,candidates=[item[0] for item in candidates],training_origins=torch.unique(data['origin'][train]),
            validation_origins=torch.unique(data['origin'][valid]),data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',result)
        print(json.dumps(dict(arm=arm,states=len(train),held_states=len(valid),parameters=all_width,selected=metric)),flush=True)


def reactive_pair_regret_weights(costs,future_weights):
    """Expected time lost by ordering two actual paired controls incorrectly."""
    contrast=costs[:,:,:,None]-costs[:,:,None,:]
    return (contrast.clamp_min(0.)*future_weights[:,:,None,None]).sum(1)


def reactive_unique_control_mask(programs):
    """Count each physical intervention once, regardless of candidate aliases."""
    mask=torch.zeros(programs.shape[:2],dtype=torch.bool,device=programs.device)
    for row in range(len(programs)):
        _,inverse=torch.unique(programs[row],dim=0,return_inverse=True)
        first=torch.full((int(inverse.max())+1,),programs.shape[1],dtype=torch.long,device=programs.device)
        first.scatter_reduce_(0,inverse,torch.arange(programs.shape[1],device=programs.device),reduce='amin')
        mask[row,first]=True
    return mask


def reactive_decision_regret_loss(energy,costs,row_weights,support=None,temperature=.5):
    """Smooth SPO+ control regret, invariant to any state-only value shift.

    The loss-augmented decoder emphasizes controls that can actually displace
    the best measured decision. Paired conditional means retain cost units;
    an optimal action is learned, rather than all pair orderings equally.
    """
    if support is None:support=torch.ones_like(energy,dtype=torch.bool)
    best=costs.masked_fill(~support,float('inf')).argmin(-1)
    augmented=(costs-2.*energy).masked_fill(~support,-float('inf'))
    loss=temperature*torch.logsumexp(augmented/temperature,-1)
    loss=loss+(2.*energy-costs).gather(-1,best[:,None])[:,0]
    return (loss*row_weights).mean()


def reactive_pair_regret_loss(energy,regret,row_weights,temperature=2.,support=None):
    margin=(energy[:,:,None]-energy[:,None,:])/temperature
    loss=regret*F.softplus(-margin)
    if support is None:per_row=loss.mean((1,2))
    else:
        pairs=support[:,:,None]&support[:,None,:]
        per_row=(loss*pairs).sum((1,2))/pairs.sum((1,2)).clamp_min(1)
    return (per_row*row_weights).mean()/temperature


def collect_reactive_factor_credits(args):
    """On-policy states and complete current-centered conditional control faces."""
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    if args.initial_seed is None:raise ValueError('Specify the exact collecting factor policy.')
    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source=ReactiveSource(args.source_device)
    roots,root_hash=reactive_roots(args.root_source,'train',args.train_count)
    for arm in args.evaluation_arms.split(','):
        path=args.output/f'reactive-{arm}-{args.initial_seed}.pth';saved=torch.load(path,weights_only=False)
        if saved['kind']!='reactive-factor-kernel-v1':raise ValueError('Factor credits require a factor policy.')
        model_hash=hashlib.sha256(path.read_bytes()).hexdigest();settings=saved['settings']
        actor=ReactiveFactorKernelGate(settings,saved['maps'],saved['weights'],saved['risks'],saved['interaction'])
        generator=torch.Generator().manual_seed(args.seed)
        available=saved['training_origins'];chosen=available[torch.randperm(len(available),generator=generator)[:args.count]]
        packet={k:v[chosen] for k,v in roots.items()}
        noise=torch.randn(len(chosen),300,2+3*(settings['defenders']+1),generator=generator)
        _,_,trajectory=reactive_rollout(actor,source,packet,noise,settings,collect=True)
        selected=[]
        for i in range(len(chosen)):
            indices=(trajectory['origin']==i).nonzero().flatten()
            selected.append(indices[torch.randint(len(indices),(1,),generator=generator)])
        selected=torch.cat(selected)
        packet={k:v[selected] for k,v in trajectory['packet'].items()}
        started=time.monotonic()
        data=reactive_credits(actor,source,packet,dict(settings,conditional_policy=True),args.repeats,generator,
            batch=args.reactive_batch,complete_pairs=True)
        data['origin']=chosen[trajectory['origin'][selected]]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=model_hash:raise RuntimeError('Collecting policy changed.')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=implementation:raise RuntimeError('Collecting implementation changed.')
        target=args.output/f'reactive-factor-data-{arm}-{args.seed}.pth'
        atomic_torch(target,dict(data=data,settings=settings,collecting_seed=args.initial_seed,
            collecting_policy_sha256=model_hash,collecting_policy=path.name,root_sha256=root_hash,
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),seed=args.seed,
            training_implementation_sha256=implementation,complete=True))
        print(json.dumps(dict(arm=arm,states=len(chosen),repeats=args.repeats,options=data['options'].shape[1],
            mean_reference=float(data['costs'][...,0].mean()),seconds=time.monotonic()-started)),flush=True)


def update_reactive_factor_kernel(args):
    """Matched recurrent-policy evaluation followed by a paired joint improvement."""
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    for arm in args.evaluation_arms.split(','):
        path=args.output/f'reactive-{arm}-{args.initial_seed}.pth';parent=torch.load(path,weights_only=False)
        data_path=args.output/f'reactive-factor-data-{arm}-{args.bootstrap_seed}.pth'
        saved=torch.load(data_path,weights_only=False);data=saved['data']
        if saved['collecting_policy_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('Return continuation differs from the preceding deployed gate.')
        maps=dict(parent['maps']);settings=dict(parent['settings'])
        groups=reactive_state_groups(data)
        order=torch.randperm(int(groups.max())+1,generator=torch.Generator().manual_seed(731907))
        held=order[:max(4,len(order)//5)]
        train=(~torch.isin(groups,held)).nonzero().flatten();valid=torch.isin(groups,held).nonzero().flatten()
        if args.complete_state and 'pair_fleet' not in maps:
            inputs=reactive_factor_inputs({k:v[train] for k,v in data['packet'].items()},data['base'][train])
            maps['pair_fleet']=fit_reactive_feature_map(inputs['pair_fleet'],48,40)
        stages=len(parent['risks']);settings['stage_max_changes']=settings.get('stage_max_changes',[settings['defenders']]*stages)+[2]
        unary,mixed,state=reactive_factor_design(data['packet'],data['base'],data['options'],maps,settings['bias'])
        interaction=parent['interaction'];conditional=parent['conditional']
        control=torch.cat((unary,mixed if interaction else torch.zeros_like(mixed)),-1)
        costs=data['costs'].double();truth=costs.mean(1)
        precision=reactive_precision(costs[train],True)
        candidates=[]
        for ridge in (.1,1.,10.,100.):
            heads=[]
            for member in range(len(parent['weights'][0])):
                g=torch.Generator().manual_seed(args.seed+104729*member)
                row_weight,future_weight=reactive_positive_credit_weights(groups[train],costs.shape[1],g)
                target=(costs[train]*future_weight[:,:,None]).sum(1)
                if conditional:
                    x=(control[:,1:]-control[:,:1])[train];y=target[:,1:]-target[:,:1]
                    matrix=torch.einsum('bki,bkl,blj,b->ij',x,precision,x,row_weight)/(len(x)*x.shape[1])
                    right=torch.einsum('bki,bkl,bl,b->i',x,precision,y,row_weight)/(len(x)*x.shape[1])
                else:
                    x=torch.cat((state[:,None].expand(-1,control.shape[1],-1),control),-1)[train];y=target
                    row_weight=row_weight/(costs[train].var(1).mean(-1)/costs.shape[1]+1.)
                    matrix=torch.einsum('bki,bkj,b->ij',x,x,row_weight)/(len(x)*x.shape[1])
                    right=torch.einsum('bki,bk,b->i',x,y,row_weight)/(len(x)*x.shape[1])
                fitted=torch.linalg.solve(matrix+torch.eye(x.shape[-1],dtype=x.dtype)*ridge/x.shape[-1],right)
                if conditional and args.decision_loss:
                    # The regression head and the deployed gate share one
                    # energy. Paired regret supplies a control-ordering
                    # gradient, including costly rare futures, without any
                    # state-return gradient entering the control factors.
                    regret=reactive_pair_regret_weights(costs[train],future_weight)
                    with torch.enable_grad():
                        fitted=nn.Parameter(fitted)
                        optimizer=torch.optim.LBFGS([fitted],max_iter=80,line_search_fn='strong_wolfe')
                        def closure():
                            optimizer.zero_grad(set_to_none=True)
                            prediction=x@fitted;error=prediction-y
                            loss=(torch.einsum('bk,bkl,bl->b',error,precision,error)*row_weight).mean()/x.shape[1]
                            energy=torch.cat((torch.zeros_like(prediction[:,:1]),prediction),1)
                            loss=loss+reactive_pair_regret_loss(energy,regret,row_weight)
                            loss=loss+ridge*fitted.square().sum()/x.shape[-1]
                            loss.backward();return loss
                        optimizer.step(closure);fitted=fitted.detach()
                heads.append((fitted if conditional else fitted[state.shape[-1]:]).float())
            heads=torch.stack(heads);prediction=torch.einsum('bkp,mp->mbk',control[valid].float(),heads)
            difference=prediction-prediction[...,:1]
            for risk in (2.,1.,.5,0.):
                score=difference.mean(0)+risk*difference.std(0,unbiased=False)
                score=score+settings.get('factor_trust',.1)*(data['options'][valid]-data['options'][valid,:1]).square().sum(-1)
                choice=score.argmin(-1);gain=truth[valid,0]-truth[valid].gather(-1,choice[:,None])[:,0]
                grouped=torch.stack([gain[groups[valid]==origin].mean() for origin in held])
                metric=dict(ridge=ridge,risk=risk,gain=float(grouped.mean()),standard_error=float(grouped.std()/len(grouped)**.5))
                candidates.append((metric,heads))
        metric,heads=max(candidates,key=lambda item:item[0]['gain']-.5*item[0]['standard_error'])
        inherited=F.pad(parent['weights'],(0,heads.shape[-1]-parent['weights'].shape[-1]))
        weights=torch.cat((inherited,heads[None]),0);risks=parent['risks']+[metric['risk']]
        actor=ReactiveFactorKernelGate(settings,maps,weights,risks,interaction)
        result=dict(parent,settings=settings,maps=maps,weights=weights,risks=risks,actor=actor.state_dict(),seed=args.seed,
            selection=metric,candidates=[m for m,_ in candidates],parent_sha256=saved['collecting_policy_sha256'],
            stage_objective='paired-return-and-decision-regret' if conditional and args.decision_loss else 'paired-return' if conditional else 'absolute-return',
            data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),stage_training_origins=data['origin'][train],
            stage_validation_origins=data['origin'][valid],training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',result)
        print(json.dumps(dict(arm=arm,stage=len(risks),states=len(train),held_states=len(valid),selected=metric)),flush=True)


def reactive_roots(directory,split,count):
    packets=[];hashes=[]
    for i in range(count):
        path=Path(directory)/split/f'program-{i}.pth';row=program_record(path)
        env=row['snapshot'].restore();p=public_packet(env);p['target_heading']=np.asarray(env.attacker.theta,np.float32)
        p['time_remaining']=np.asarray(max(0.,1.-env.Current_T/60.),np.float32)
        packets.append(p);hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return {k:torch.as_tensor(np.stack([p[k] for p in packets]),dtype=torch.float32) for k in packets[0]},hashlib.sha256(''.join(hashes).encode()).hexdigest()


@torch.no_grad()
def reactive_kernel_description(packet,base,options,settings,probe='aligned'):
    # Target velocity and position are recovered exclusively from public
    # observations. No water-relative velocity or private attacker state is read.
    p={k:np.asarray(v) for k,v in packet.items()};m=p['motion'];o=p['obs']
    velocity_heading=m[0,2]+o[0,5];heading=float(p['target_heading']) if settings.get('target_pose') else velocity_heading
    bearing=m[0,2]+o[0,3]
    position=m[0,:2]+o[0,2]*np.array([np.cos(bearing),np.sin(bearing)])
    velocity=o[0,4]*np.array([np.cos(velocity_heading),np.sin(velocity_heading)])
    p=dict(p,target_motion=np.array([*position,heading,*velocity,0.]))
    offsets=options*settings['radius']
    feature=program_features(p,base,offsets,settings['bias'])
    description=program_kernel_description(p,feature,offsets,settings)
    if probe=='state-only':
        width=description['node'].shape[-1];description['node_context'][...,width:]=0.
    if probe=='permuted':
        altered=copy.deepcopy(p);altered['obs'][:,12:14]=np.roll(p['obs'][:,12:14],1,axis=0)
        other=program_features(altered,np.roll(base,1,axis=0),offsets,settings['bias'])
        replacement=program_kernel_description(altered,other,offsets,settings)
        width=description['node'].shape[-1];description['node_context'][...,width:]=replacement['node_context'][...,width:]
        description['pair']=replacement['pair']
    return description


def reactive_kernel_bank(data,settings,device='cpu',probe='aligned'):
    rows=[reactive_kernel_description({k:v[i].detach().cpu().numpy() for k,v in data['packet'].items()},
        data['base'][i].detach().cpu().numpy(),data['options'][i].detach().cpu().numpy(),settings,probe) for i in range(len(data['base']))]
    return {k:torch.as_tensor(np.stack([r[k] for r in rows]),dtype=torch.float64,device=device) for k in rows[0]}


def reactive_kernel_components(left,right,normalization,length,node_key):
    # All four mixed ternary factors are needed: signed x signed alone
    # cannot represent a cooperative benefit confined to just one corner.
    base=program_kernel_components(left,right,normalization,length,node_key)
    a=(left['pair']-normalization['pair']['mean'])/normalization['pair']['scale']
    b=(right['pair']-normalization['pair']['mean'])/normalization['pair']['scale']
    kernel=torch.exp(-.5*torch.cdist(a[:,0],b[:,0]).square()/a.shape[-1]/length**2)
    def basis(z):
        x,y=z[...,0],z[...,1]
        return torch.stack((x*y,x*y.square(),x.square()*y,x.square()*y.square()),-1)
    base['pair']=torch.einsum('iab,ij,jcb->iajc',basis(left['z']),kernel,basis(right['z']))
    return base


def fit_reactive_kernel(data,settings,conditional,peer,device='cuda'):
    """Select a conditional kernel critic by leave-one-physical-root-out cost.

    Every option and future of a root is removed together. This prevents a
    shared state or paired outcome from leaking into that root's prediction.
    Selection is part of learning; final native scenes remain independent.
    """
    bank=reactive_kernel_bank(data,settings,device)
    normal=program_kernel_normalization(bank,'control');n,k=data['costs'].shape[0],data['costs'].shape[-1]
    y=data['costs'].mean(1).to(device,dtype=torch.float64)
    raw=data['costs'].to(device,dtype=torch.float64)
    if conditional:raw=raw[...,1:]-raw[...,:1];target=y[:,1:]-y[:,:1]
    else:target=y-y.mean()
    centered=raw-raw.mean(1,keepdim=True);r=raw.shape[1];size=raw.shape[-1]
    noise=torch.einsum('bri,brj->ij',centered,centered)/(n*max(1,r-1)*r)
    noise=(.8*noise+.2*torch.diag(noise.diag())) if conditional else torch.diag(noise.diag())
    noise=noise+.25*torch.eye(size,device=device,dtype=noise.dtype)
    whiten=torch.linalg.inv(torch.linalg.cholesky(noise));response=(target@whiten.T).reshape(-1)
    selected=None;history=[];columns=slice(1,None) if conditional else slice(None)
    for length in (.35,.7,1.4):
        parts=reactive_kernel_components(bank,bank,normal,length,'node_context' if peer else 'node')
        for interaction in ((1.,4.) if peer else (0.,)):
            kernel=parts['unary']+interaction*parts['pair']+(0. if conditional else 25.*parts['state'])
            matrix=kernel[:,columns,:,columns]
            matrix=(whiten@matrix.permute(0,2,1,3)@whiten.T).permute(0,2,1,3).reshape(n*size,n*size)
            eigen,vectors=torch.linalg.eigh(.5*(matrix+matrix.T));eigen=eigen.clamp_min(0.);projected=vectors.T@response
            for ridge in (.1,1.,10.,100.):
                precision=(vectors/(eigen+ridge)[None])@vectors.T
                alpha=precision@response
                # Block deletion, rather than deleting a single gate row.
                blocks=precision.reshape(n,size,n,size)[torch.arange(n),:,torch.arange(n),:]
                omitted=response.reshape(n,size)-torch.linalg.solve(blocks,alpha.reshape(n,size,1)).squeeze(-1)
                prediction=torch.linalg.solve(whiten,omitted.T).T
                if conditional:prediction=torch.cat((torch.zeros(n,1,device=device),prediction),-1)
                choice=(prediction+.5*data['options'].square().sum(-1).to(device)).argmin(-1)
                gain=(y[:,0]-y.gather(1,choice[:,None]).squeeze(-1)).mean()
                mae=((prediction-prediction[:,:1])-(y-y[:,:1])).abs().mean()
                record=dict(length=length,interaction=interaction,ridge=ridge,loo_gain=float(gain),loo_mae=float(mae));history.append(record)
                score=float(gain)-.02*float(mae)
                if selected is None or score>selected['score']:
                    selected=dict(score=score,metrics=record,alpha=(alpha.reshape(n,size)@whiten).detach().cpu())
    return dict(bank={k:v.cpu() for k,v in bank.items()},normal={k:{j:v.cpu() for j,v in x.items()} for k,x in normal.items()},
        alpha=selected['alpha'],metrics=selected['metrics'],history=history,conditional=conditional,peer=peer)


class ReactiveKernelGate:
    def __init__(self,settings,blocks=None,coefficients=None):
        self.settings=settings;self.blocks=blocks or [];self.coefficients=list(coefficients or [])
    def log_policy(self,packet,baseline,programs,temperature=1.,reference_prior=.2,probe='aligned'):
        if not self.blocks:return (-reference_prior*programs.square().sum(-1)).log_softmax(-1)
        bank=reactive_kernel_bank(dict(packet=packet,base=baseline,options=programs),self.settings,probe=probe)
        total=torch.zeros(programs.shape[:2],dtype=torch.float64)
        for coefficient,block in zip(self.coefficients,self.blocks):
            parts=reactive_kernel_components(bank,block['bank'],block['normal'],block['metrics']['length'],'node_context' if block['peer'] else 'node')
            interaction=0. if probe=='state-only' else block['metrics']['interaction']
            kernel=parts['unary']+interaction*parts['pair']+(0. if block['conditional'] else 25.*parts['state'])
            if block['conditional']:kernel=kernel[:,:,:,1:]
            energy=(kernel.reshape(total.numel(),-1)@block['alpha'].reshape(-1)).reshape_as(total)
            total+=coefficient*(energy-energy[:,:1])
        return (-total/temperature-reference_prior*programs.square().sum(-1)).log_softmax(-1).float()


@torch.no_grad()
def reactive_accept_step(actor,source,packet,settings,generator,repeats=4,standard_errors=0.):
    """Backtrack the new factor increment using fresh complete training futures.

    This is part of the optimizer, never a validation/test performance claim.
    All step lengths share initial states and future disturbances, and the
    zero-length candidate is exactly the preceding deployed policy.
    """
    count,n=packet['obs'].shape[:2]
    expanded={k:v.repeat_interleave(repeats,0) for k,v in packet.items()}
    noise=torch.randn(count*repeats,300,2+3*(n+1),generator=generator)
    coefficient=copy.deepcopy(actor.coefficients[-1]);costs=[]
    scales=(0.,1.,.5,.25,2.,4.) if settings.get('expand_step',False) else (0.,1.,.5,.25)
    for scale in scales:
        actor.coefficients[-1]=coefficient*scale
        costs.append(reactive_rollout(actor,source,expanded,noise,settings)[0].reshape(count,repeats).mean(1))
    average=torch.stack(costs).mean(-1);best=int(average.argmin())
    if best and float((costs[0]-costs[best]).mean())<=standard_errors*float((costs[0]-costs[best]).std()/count**.5):best=0
    actor.coefficients[-1]=coefficient*scales[best]
    if best==0:
        del actor.blocks[-1]
        actor.coefficients=actor.coefficients[:-1].clone() if isinstance(actor.coefficients,torch.Tensor) else actor.coefficients[:-1]
    return dict(selected_scale=scales[best],candidate_scales=list(scales),mean_costs=average.tolist(),
        paired_gain=float((costs[0]-costs[best]).mean()),
        paired_standard_error=float((costs[0]-costs[best]).std()/count**.5),
        roots=count,repeats=repeats)


def fit_reactive_kernel_policy(args):
    from interaction_evaluation import OriginalPolicy
    torch.set_num_threads(1);started=time.monotonic();source=OriginalPolicy(str(SOURCE)).actor.eval().requires_grad_(False)
    training,train_hash=reactive_roots(args.root_source,'train',args.train_count)
    validation,valid_hash=reactive_roots(args.root_source,'validation',args.validation_count)
    settings=dict(period=args.intervention_steps,bias=args.program_bias,radius=args.program_radius,temperature=1.,prior=.3,
        improvement=.6,trigger=args.program_trigger,defenders=args.defenders,agility=args.agility,kernel=True,target_pose=args.target_pose)
    generator=torch.Generator().manual_seed(args.seed+160000)
    validation_noise=torch.randn(len(validation['obs']),300,2+3*(args.defenders+1),generator=generator)
    reference=reactive_rollout(ReactiveKernelGate(settings),source,validation,validation_noise,settings)[0]
    models={a:ReactiveKernelGate(settings) for a in args.arms.split(',')};history={a:[] for a in models}
    shared=None
    if args.reactive_bootstrap_data:
        saved=torch.load(args.reactive_bootstrap_data,weights_only=False)
        if saved['collecting_actor'].get('coefficients',torch.ones(1)).numel() or any(saved['settings'][k]!=settings[k] for k in ('period','bias','radius','defenders','agility')):
            raise ValueError('Kernel warm-start labels require identical zero-increment continuation.')
        shared=saved['data']
        if args.target_pose:reactive_restore_public_heading(shared)
    for iteration in range(args.reactive_iterations):
        for arm,actor in models.items():
            generator=torch.Generator().manual_seed(args.seed+iteration*10000+1000000)
            order=torch.randperm(len(training['obs']),generator=generator);selection=order[:args.count]
            before=copy.deepcopy(dict(blocks=actor.blocks,coefficients=actor.coefficients))
            if iteration==0 and shared is not None:data=shared
            else:
                p={k:v[selection] for k,v in training.items()}
                noise=torch.randn(len(p['obs']),300,2+3*(args.defenders+1),generator=generator)
                _,_,trajectory=reactive_rollout(actor,source,p,noise,settings,collect=True)
                ix=torch.randperm(len(trajectory['base']),generator=generator)[:args.count]
                p={k:v[ix] for k,v in trajectory['packet'].items()}
                data=reactive_credits(actor,source,p,settings,args.repeats,generator,args.reactive_batch)
                if iteration==0:shared=data
            fitted=fit_reactive_kernel(data,settings,arm in ('difference','no_peer'),'no_peer' not in arm,args.device)
            actor.blocks.append(fitted);actor.coefficients.append(settings['improvement'])
            eligible=order[args.count:] if len(order)>args.count else order
            check={k:v[eligible[:min(64,len(eligible))]] for k,v in training.items()}
            checked=reactive_accept_step(actor,source,check,settings,torch.Generator().manual_seed(args.seed+iteration*10000+3000000))
            cost,outcomes,_=reactive_rollout(actor,source,validation,validation_noise,settings)
            metrics=dict(iteration=iteration,validation_gain=float((reference-cost).mean()),validation_cost=float(cost.mean()),
                captures=int((outcomes==3).sum()),value_selection=fitted['metrics'],step_check=checked,seconds=time.monotonic()-started)
            history[arm].append(metrics)
            saved=dict(kind='reactive-kernel-v1',settings=settings,blocks=actor.blocks,coefficients=actor.coefficients,
                arm=arm,seed=args.seed,iteration=iteration,history=history[arm],train_sha256=train_hash,validation_sha256=valid_hash,
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                training_settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
            atomic_torch(args.output/f'reactive-{arm}-{args.seed}-iteration-{iteration}.pth',saved)
            atomic_torch(args.output/f'reactive-data-{arm}-{args.seed}-iteration-{iteration}.pth',dict(data=data,collecting_actor=before,settings=settings))
            if iteration==max(range(len(history[arm])),key=lambda i:history[arm][i]['validation_gain']):
                atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',saved)
            print(json.dumps(dict(arm=arm,**metrics)),flush=True)


def reactive_restore_public_heading(data):
    """Recover the recorded public orientation, never a private velocity.

    Older trajectory files stored public pose also inside the centralized
    snapshot. Extracting its sine/cosine here upgrades that observation schema;
    actors at execution read the explicit measured heading only.
    """
    p=data['packet'];n=p['obs'].shape[1]
    if 'target_heading' not in p:
        p['target_heading']=torch.atan2(p['central'][:,7*n+2],p['central'][:,7*n+3])
    if 'time_remaining' not in p:p['time_remaining']=p['central'][:,-1].clone()


class ReactiveGroupedSource:
    """Fixed-shape grouped arithmetic for the internal public response model.

    Group count and matrix shapes remain fixed even in the last block. This
    avoids making nominal candidate futures depend on the evaluation batch.
    The true simulator continues to use the unchanged original source.
    """
    def __init__(self,source,groups=8,block=256):
        self.source=source;self.device=source.device;self.groups=groups;self.block=block

    def __call__(self,observations,deterministic=True,with_logprob=False):
        if not deterministic or with_logprob:raise ValueError('The public model uses deterministic candidates.')
        model=self.source.actor;outputs=[];size=self.groups*self.block
        def linear(value,layer):
            shape=value.shape;value=value.reshape(self.groups,-1,shape[-1])
            result=torch.bmm(value,layer.weight.T[None].expand(self.groups,-1,-1))+layer.bias
            return result.reshape(*shape[:-1],-1)
        for begin in range(0,len(observations),size):
            x=observations[begin:begin+size].to(self.device);count=len(x)
            if count<size:x=torch.cat((x,x[-1:].expand(size-count,-1)))
            x=x.reshape(self.groups,self.block,-1);a=model.feature1_dim;b=a+model.feature2_dim
            first=F.leaky_relu(linear(x[...,:a],model.f1));second=F.leaky_relu(linear(x[...,a:b],model.f2))
            peers=F.leaky_relu(linear(x[...,b:].reshape(self.groups,self.block,-1,2),model.t1)).mean(-2)
            hidden=F.leaky_relu(linear(torch.cat((first,second,peers),-1),model.l1))
            hidden=F.leaky_relu(linear(hidden,model.l2))
            control=linear(hidden,model.mean_layer).tanh()*model.max_action
            gate=F.leaky_relu(linear(torch.cat((control,x[...,b-2:b]),-1),model.a1))
            gate=linear(torch.cat((gate,hidden),-1),model.adap_layer).tanh()*.5+.5
            outputs.append(torch.cat((control,gate),-1).flatten(0,1)[:count].to(observations.device))
        return torch.cat(outputs),None


class ReactiveCanonicalSource:
    """Canonical nominal-policy arithmetic, independent of trajectory batches.

    Only the public response model uses padded fixed-size inference. Source
    weights and the actual environment's controls/CBF remain unchanged.
    """
    def __init__(self,source,block=256):self.source=source;self.device=getattr(source,'device','cpu');self.block=block
    def __call__(self,observations,*arguments):
        actions=[];logps=[]
        for start in range(0,len(observations),self.block):
            value=observations[start:start+self.block];count=len(value)
            if count<self.block:value=torch.cat((value,value[-1:].expand(self.block-count,-1)))
            action,logp=self.source(value,*arguments);actions.append(action[:count])
            if logp is not None:logps.append(logp[:count])
        return torch.cat(actions),torch.cat(logps) if logps else None


class ReactiveSource(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        from interaction_evaluation import OriginalPolicy
        self.actor=OriginalPolicy(str(SOURCE)).actor.to(device).eval().requires_grad_(False);self.device=device
        if str(device).startswith('cuda'):torch.backends.cuda.matmul.allow_tf32=False
    def forward(self,observations,*arguments):
        action,logp=self.actor(observations.to(self.device),*arguments)
        return action.to(observations.device),None if logp is None else logp.to(observations.device)


def fit_reactive(args):
    from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate,EnsembleCoordinationValue
    from interaction_evaluation import OriginalPolicy
    torch.set_num_threads(1);torch.set_grad_enabled(True);started=time.monotonic()
    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'policy/interaction_sac.py').read_bytes()).hexdigest()
    source=ReactiveSource(args.source_device)
    training,train_hash=reactive_roots(args.root_source,'train',args.train_count)
    validation,validation_hash=reactive_roots(args.root_source,'validation',args.validation_count)
    fit_roots,check_roots=reactive_root_partition(len(training['obs']),args.reserve_roots)
    validation={k:v.repeat_interleave(args.validation_repeats,0) for k,v in validation.items()}
    settings=dict(period=args.intervention_steps,bias=args.program_bias,radius=args.program_radius,
        temperature=1.,prior=.3,improvement=.3,trigger=args.program_trigger,
        defenders=args.defenders,agility=args.agility,exact_mirror=args.exact_mirror,
        critic_members=args.critic_members,critic_risk=args.critic_risk,target_pose=args.target_pose,
        complete_state=args.complete_state,all_gates=args.joint_fleet,control_radius=30.,
        gate_levels=args.gate_levels,calibrate_risk=args.calibrate_risk,expand_step=args.expand_step,full_gate=args.full_gate,
        response_encoding=args.response_encoding,decision_loss=args.decision_loss,
        cardinal_gates=args.cardinal_gates,refine_gates=args.refine_gates,conditional_policy=args.conditional_policy,
        reaction_encoding=args.reaction_encoding,potential_value=args.potential_value,preference_value=args.preference_value,curved_potential=args.curved_potential,constrained_potential=args.constrained_potential,coupled_potential=args.coupled_potential,local_precision=args.local_precision)
    if (args.cardinal_gates or args.refine_gates) and not args.full_gate:
        raise ValueError('Candidate cardinal coordinates and joint refinement need full-range gates.')
    if args.conditional_policy and not (args.full_gate and args.joint_fleet and args.exact_mirror):
        raise ValueError('Conditional policy improvement needs full-range joint mirror gates.')
    arms=args.arms.split(',');models={};history={arm:[] for arm in arms}
    for arm in arms:
        torch.manual_seed(args.seed)
        peer=arm not in ('no_peer','absolute_no_peer')
        critic=(CandidateCoordinationGate(args.width,peer,args.target_pose,args.complete_state,args.response_encoding,args.cardinal_gates,args.program_bias,args.reaction_encoding,args.potential_value,args.curved_potential,args.constrained_potential,args.coupled_potential) if args.critic_members==1 else
            EnsembleCoordinationValue(args.width,peer,args.critic_members,args.critic_risk,args.target_pose,args.complete_state,args.response_encoding,args.cardinal_gates,args.program_bias,args.reaction_encoding,args.potential_value,args.curved_potential,args.constrained_potential,args.coupled_potential))
        if args.critic_members>1 and not args.exact_mirror:raise ValueError('Ensemble values require direct factor-space policy updates.')
        actor=MirrorCoordinationGate(args.width,peer,members=args.critic_members,risk=args.critic_risk,target_pose=args.target_pose,complete_state=args.complete_state,response_encoding=args.response_encoding,cardinal_gates=args.cardinal_gates,gate_bias=args.program_bias,reaction_encoding=args.reaction_encoding,potential_value=args.potential_value,curved_potential=args.curved_potential,constrained_potential=args.constrained_potential,coupled_potential=args.coupled_potential) if args.exact_mirror else copy.deepcopy(critic)
        if args.initial_seed is not None:
            parent=torch.load(args.output/f'reactive-{arm}-{args.initial_seed}.pth',weights_only=False)
            parent['settings'].setdefault('reaction_encoding',False)
            parent['settings'].setdefault('potential_value',False)
            parent['settings'].setdefault('preference_value',False)
            parent['settings'].setdefault('curved_potential',False)
            parent['settings'].setdefault('constrained_potential',False)
            parent['settings'].setdefault('coupled_potential',False)
            parent['settings'].setdefault('local_precision',False)
            if parent['settings']!=settings:raise ValueError('Reactive continuation must retain controller settings.')
            if args.reserve_roots!=parent['training_settings'].get('reserve_roots',0):
                raise ValueError('A continuing policy must retain the same reserved root partition.')
            if args.exact_mirror:actor=MirrorCoordinationGate(args.width,peer,parent['mirror_increments'],args.critic_members,args.critic_risk,args.target_pose,args.complete_state,args.response_encoding,args.cardinal_gates,args.program_bias,args.reaction_encoding,args.potential_value,args.curved_potential,args.constrained_potential,args.coupled_potential)
            actor.load_state_dict(parent['actor']);critic.load_state_dict(parent['critic'])
        models[arm]=(actor,critic)
    generator=torch.Generator().manual_seed(args.seed+160000)
    validation_noise=torch.randn(len(validation['obs']),300,2+3*(args.defenders+1),generator=generator)
    reference_actor=MirrorCoordinationGate(args.width,True)
    reference=reactive_rollout(reference_actor,source,validation,validation_noise,settings)[0]
    previous_validation={arm:(reference if not len(actor.blocks) else
        reactive_rollout(actor,source,validation,validation_noise,settings)[0])
        for arm,(actor,_) in models.items()} if args.exact_mirror else {}
    for iteration in range(args.reactive_iterations):
        shared_initial=None
        # Arm-specific on-policy states and conditional continuations are
        # necessary after the first update. Common root/future draws still
        # pair the experimental budgets without forcing an off-policy critic.
        for arm,(actor,critic) in models.items():
            if iteration==0 and args.reactive_bootstrap_data and shared_initial is None:
                bootstrap=torch.load(args.reactive_bootstrap_data,weights_only=False)
                if args.initial_seed is not None or bootstrap['collecting_actor'].get('coefficients',torch.ones(1)).numel():
                    raise ValueError('Reused initial credits must come from the same zero-increment gate policy.')
                if any(bootstrap['settings'][k]!=settings[k] for k in ('period','bias','radius','defenders','agility')) or bootstrap['settings'].get('full_gate',False)!=settings['full_gate']:
                    raise ValueError('Reused initial credits have different physical interventions.')
                shared_initial=bootstrap['data']
                if args.target_pose:reactive_restore_public_heading(shared_initial)
                if len(shared_initial['base'])!=(args.initial_credit_count or args.count) or shared_initial['costs'].shape[1]!=args.repeats:
                    raise ValueError('Reused initial credit budget differs from the requested budget.')
                if ('origin' not in shared_initial or (args.coupled_potential and 'target_turn' not in shared_initial['packet'])) and args.reactive_origin_seed is not None:
                    shared_initial=reactive_recover_origins(shared_initial,training,source,settings,
                        args.initial_credit_count or args.count,args.reactive_origin_seed)
                if args.coupled_potential and 'target_turn' not in shared_initial['packet']:
                    raise ValueError('Coupled evasion needs measured heading history; recover it from the collecting rollout.')
                if args.reserve_roots:
                    if 'origin' not in shared_initial:raise ValueError('Reserved roots require recorded or verified trajectory origins.')
                    shared_initial=reactive_subset(shared_initial,torch.isin(shared_initial['origin'],fit_roots).nonzero().flatten())
            if iteration==0 and args.initial_seed is None and shared_initial is not None:data=shared_initial
            else:
                generator=torch.Generator().manual_seed(args.seed+iteration*10000+1000000)
                selection=fit_roots[torch.randperm(len(fit_roots),generator=generator)[:args.count]]
                p={k:v[selection] for k,v in training.items()}
                noise=torch.randn(len(p['obs']),300,2+3*(args.defenders+1),generator=generator)
                _,_,trajectory=reactive_rollout(actor,source,p,noise,settings,collect=True)
                indices=torch.randperm(len(trajectory['base']),generator=generator)[:args.count]
                sampled={k:v[indices] for k,v in trajectory['packet'].items()}
                data=reactive_credits(actor,source,sampled,settings,args.repeats,generator,args.reactive_batch,args.complete_pair_credits)
                data['origin']=selection[trajectory['origin'][indices]]
                if iteration==0 and args.reactive_augment_data:
                    augmented=torch.load(args.reactive_augment_data,weights_only=False)
                    if args.initial_seed is not None or augmented['collecting_actor'].get('coefficients',torch.ones(1)).numel() or any(augmented['settings'][k]!=settings[k] for k in ('period','bias','radius','defenders','agility')) or augmented['settings'].get('full_gate',False)!=settings['full_gate']:
                        raise ValueError('Extra initial credits must have the same zero-increment policy and intervention.')
                    extra=augmented['data'];reactive_restore_public_heading(extra)
                    if 'origin' not in extra:raise ValueError('Augmented credits need trajectory provenance.')
                    if args.reserve_roots:extra=reactive_subset(extra,torch.isin(extra['origin'],fit_roots).nonzero().flatten())
                    data={k:({j:torch.cat((data[k][j],extra[k][j])) for j in data[k]} if k=='packet' else torch.cat((data[k],extra[k]))) for k in data}
                if iteration==0 and args.initial_seed is None:shared_initial=data
            before=copy.deepcopy(actor.state_dict())
            metrics=reactive_update(actor,critic,data,arm in ('difference','no_peer'),settings,args.updates,args.seed+iteration)
            if args.reactive_step_check:
                if not args.exact_mirror:raise ValueError('Step checks require exact factor-space policy updates.')
                check_generator=torch.Generator().manual_seed(args.seed+iteration*10000+3000000)
                # Fresh roots, when available, are disjoint from the roots
                # used to collect the current update's trajectories.
                generator=torch.Generator().manual_seed(args.seed+iteration*10000+1000000)
                order=torch.randperm(len(training['obs']),generator=generator)
                eligible=check_roots if args.reserve_roots else (order[args.count:] if len(order)>args.count else order)
                check={k:v[eligible[:min(args.accept_roots,len(eligible))]] for k,v in training.items()}
                metrics['step_check']=reactive_accept_step(actor,source,check,settings,check_generator,args.accept_repeats,args.accept_standard_errors)
            validation_cost,outcomes,_=reactive_rollout(actor,source,validation,validation_noise,settings)
            metrics.update(iteration=iteration,validation_gain=float((reference-validation_cost).mean()),
                validation_cost=float(validation_cost.mean()),captures=int((outcomes==3).sum()),validation_roots=args.validation_count,
                validation_repeats=args.validation_repeats,seconds=time.monotonic()-started)
            metrics.update(credit_states=len(data['base']),trajectory_groups=int(reactive_state_groups(data).max())+1,
                reserved_roots=len(check_roots))
            if arm in previous_validation:
                gains=(previous_validation[arm]-validation_cost).reshape(args.validation_count,args.validation_repeats).mean(1)
                metrics.update(incremental_validation_gain=float(gains.mean()),
                    incremental_validation_standard_error=float(gains.std()/len(gains)**.5))
                previous_validation[arm]=validation_cost
            history[arm].append(metrics)
            # Selection is common across arms and uses whole current-policy
            # validation episodes, not the surrogate or the training labels.
            best=max(range(len(history[arm])),key=lambda j:history[arm][j]['validation_gain'])
            saved=dict(kind='reactive-conditional-gates-v1',actor=actor.state_dict(),critic=critic.state_dict(),
                settings=settings,width=args.width,peer_candidates=actor.peer_candidates,arm=arm,seed=args.seed,
                mirror_increments=len(actor.blocks) if args.exact_mirror else 0,
                training_settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),train_sha256=train_hash,
                validation_sha256=validation_hash,training_implementation_sha256=implementation_sha256,history=history[arm],iteration=iteration)
            atomic_torch(args.output/f'reactive-{arm}-{args.seed}-iteration-{iteration}.pth',saved)
            atomic_torch(args.output/f'reactive-data-{arm}-{args.seed}-iteration-{iteration}.pth',dict(data=data,collecting_actor=before,settings=settings))
            if best==iteration:atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',saved)
            print(json.dumps(dict(arm=arm,**metrics)),flush=True)


def refit_conditional_policy(args):
    """Reuse a verified Q of the zero policy with its identified policy domain.

    This changes the optimizer, not the labels or trained critic. Reusing a
    critic of a later, different continuation is explicitly rejected.
    """
    from policy.interaction_sac import CandidateCoordinationGate,EnsembleCoordinationValue,MirrorCoordinationGate
    if args.initial_seed is None:raise ValueError('Conditional readout refit requires a parent seed.')
    torch.set_num_threads(1);torch.set_grad_enabled(False);started=time.monotonic()
    implementation=hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'policy/interaction_sac.py').read_bytes()).hexdigest()
    training,train_hash=reactive_roots(args.root_source,'train',args.train_count)
    validation,validation_hash=reactive_roots(args.root_source,'validation',args.validation_count)
    validation={k:v.repeat_interleave(args.validation_repeats,0) for k,v in validation.items()}
    source=ReactiveSource(args.source_device)
    for arm in args.arms.split(','):
        path=args.output/f'reactive-{arm}-{args.initial_seed}.pth';parent=torch.load(path,weights_only=False)
        labels=torch.load(args.output/f'reactive-data-{arm}-{args.initial_seed}-iteration-0.pth',weights_only=False)
        if parent['iteration']!=0 or labels['collecting_actor'].get('coefficients',torch.ones(1)).numel():
            raise ValueError('Only credits following the identical zero-increment continuation may be reused.')
        if parent['train_sha256']!=train_hash or parent['validation_sha256']!=validation_hash:
            raise ValueError('Conditional refit must retain its parent root banks.')
        settings=dict(parent['settings'],conditional_policy=True)
        if not all(settings.get(key,False) for key in ('full_gate','all_gates','exact_mirror')):
            raise ValueError('Conditional refit requires the full-range paired-value architecture.')
        width=parent['width'];peer=parent['peer_candidates'];members=settings['critic_members']
        kwargs=dict(target_pose=settings.get('target_pose',False),complete_state=settings.get('complete_state',False),
            response_encoding=settings.get('response_encoding',False),cardinal_gates=settings.get('cardinal_gates',False),gate_bias=settings['bias'],reaction_encoding=settings.get('reaction_encoding',False),potential_value=settings.get('potential_value',False),curved_potential=settings.get('curved_potential',False),constrained_potential=settings.get('constrained_potential',False),coupled_potential=settings.get('coupled_potential',False))
        critic=(CandidateCoordinationGate(width,peer,**kwargs) if members==1 else
            EnsembleCoordinationValue(width,peer,members,settings['critic_risk'],**kwargs))
        critic.load_state_dict(parent['critic']);critic.eval().requires_grad_(False)
        actor=MirrorCoordinationGate(width,peer,members=members,risk=settings['critic_risk'],**kwargs)
        generator=torch.Generator().manual_seed(args.seed+160000)
        noise=torch.randn(len(validation['obs']),300,2+3*(settings['defenders']+1),generator=generator)
        reference=reactive_rollout(actor,source,validation,noise,settings)[0]
        actor.add_value(critic,settings['temperature']*settings['improvement'])
        generator=torch.Generator().manual_seed(args.seed+3000000)
        order=torch.randperm(len(training['obs']),generator=generator)[:args.accept_roots]
        check={k:v[order] for k,v in training.items()}
        metrics=dict(step_check=reactive_accept_step(actor,source,check,settings,generator,args.accept_repeats,args.accept_standard_errors))
        cost,outcomes,_=reactive_rollout(actor,source,validation,noise,settings)
        metrics.update(iteration=0,validation_gain=float((reference-cost).mean()),validation_cost=float(cost.mean()),
            captures=int((outcomes==3).sum()),validation_roots=args.validation_count,
            validation_repeats=args.validation_repeats,seconds=time.monotonic()-started)
        saved=dict(parent,actor=actor.state_dict(),critic=critic.state_dict(),settings=settings,seed=args.seed,
            mirror_increments=len(actor.blocks),history=[metrics],iteration=0,
            training_implementation_sha256=implementation,
            conditional_refit=dict(parent=str(path),parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                training_critic_reused=True,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}.pth',saved)
        atomic_torch(args.output/f'reactive-{arm}-{args.seed}-iteration-0.pth',saved)
        print(json.dumps(dict(arm=arm,**metrics)),flush=True)


def reactive_native_action(action,boids):
    """Encode actuator-specific fusion for the unchanged native CBF API."""
    if action.shape[-1]==3:return action
    if action.shape[-1]!=4:raise ValueError('Expected scalar or two-actuator fusion.')
    action=np.asarray(action,dtype=np.float64);gate=action[:,2:4]
    nominal=gate*(750.*action[:,:2]+250.)+(1.-gate)*np.asarray(boids,dtype=np.float64)
    return np.concatenate(((nominal-250.)/750.,np.ones((len(action),1))),-1)


class ReactiveGatePolicy:
    def __init__(self,path,probe='aligned'):
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        from interaction_evaluation import OriginalPolicy
        self.saved=torch.load(path,map_location='cpu',weights_only=False)
        self.settings=self.saved['settings'];self.probe=probe
        if self.saved['kind']=='reactive-kernel-v1':
            self.model=ReactiveKernelGate(self.settings,self.saved['blocks'],self.saved['coefficients'])
        elif self.saved['kind']=='reactive-factor-kernel-v1':
            self.model=ReactiveFactorKernelGate(self.settings,self.saved['maps'],self.saved['weights'],self.saved['risks'],
                interaction=self.saved.get('interaction',True))
        elif self.saved['kind'] in ('reactive-model-prior-v1','reactive-attention-v1','reactive-component-v1'):
            self.model=reactive_prior_actor(self.saved,ReactiveSource('cuda' if torch.cuda.is_available() else 'cpu'))
        elif self.settings.get('exact_mirror',False):
            self.model=MirrorCoordinationGate(self.saved['width'],self.saved['peer_candidates'],self.saved['mirror_increments'],
                self.settings.get('critic_members',1),self.settings.get('critic_risk',.5),self.settings.get('target_pose',False),self.settings.get('complete_state',False),self.settings.get('response_encoding',False),self.settings.get('cardinal_gates',False),self.settings['bias'],self.settings.get('reaction_encoding',False),self.settings.get('potential_value',False),self.settings.get('curved_potential',False),self.settings.get('constrained_potential',False),self.settings.get('coupled_potential',False))
        else:self.model=CandidateCoordinationGate(self.saved['width'],self.saved['peer_candidates'],self.settings.get('target_pose',False),self.settings.get('complete_state',False),self.settings.get('response_encoding',False),self.settings.get('cardinal_gates',False),self.settings['bias'],self.settings.get('reaction_encoding',False),self.settings.get('potential_value',False),self.settings.get('curved_potential',False),self.settings.get('constrained_potential',False),self.settings.get('coupled_potential',False))
        if isinstance(self.model,nn.Module):
            self.model.eval().requires_grad_(False);self.model.load_state_dict(self.saved['actor'])
        self.source=OriginalPolicy(str(SOURCE));self.reset()
    def reset(self):self.remaining=0;self.active=False;self.correction=None;self.decisions=[];self.previous_heading=None
    @torch.no_grad()
    def __call__(self,packet):
        base=self.source(packet['obs']);s=self.settings
        if s.get('coupled_potential',False):
            heading=float(packet['target_heading']);delta=0. if self.previous_heading is None else heading-self.previous_heading
            packet=dict(packet,target_turn=np.asarray(np.arctan2(np.sin(delta),np.cos(delta))/.2,np.float32))
            self.previous_heading=heading
        self.active=self.active or program_trigger(packet,s['trigger'],2)
        if self.active and self.remaining==0:
            p={k:torch.as_tensor(v,dtype=torch.float32)[None] for k,v in packet.items()}
            base_tensor=torch.as_tensor(base)[None]
            choice,program=reactive_decision(self.model,p,base_tensor,s,self.probe)
            choice=int(choice[0]);self.correction=program[0].numpy();self.remaining=s['period']
            displacement=self.correction*(1. if s.get('full_gate',False) else s['radius'])
            remaining=packet['time_remaining'] if 'time_remaining' in packet else packet['central'][-1]
            self.decisions.append(dict(time=60.*(1.-float(remaining)),choice=choice,correction=displacement.tolist(),
                public_sha256=hashlib.sha256(b''.join(np.asarray(packet[k]).tobytes() for k in
                    ('obs','motion','target_heading','time_remaining') if k in packet)).hexdigest()))
        if self.active:self.remaining-=1
        return reactive_action(torch.as_tensor(base),torch.as_tensor(self.correction if self.active else 0.),s).numpy()


def initialize_reactive_evaluation(output,seed,arms):
    global REACTIVE_POLICIES
    initialize_fleet();REACTIVE_POLICIES={'__arms__':arms.split(',')}
    for arm in arms.split(','):
        if arm in ('original','arboids_cbf','constant_bias'):continue
        name='difference' if arm in ('permuted','state-only','interaction-off') else arm
        REACTIVE_POLICIES[arm]=ReactiveGatePolicy(Path(output)/f'reactive-{name}-{seed}.pth',arm if arm in ('permuted','state-only','interaction-off') else 'aligned')


def reactive_episode(task):
    scene,n,agility,bias=task;row=dict(scene=scene,defenders=n,agility=agility)
    for arm in REACTIVE_POLICIES['__arms__']:
        policy=REACTIVE_POLICIES.get(arm)
        if policy:policy.reset()
        seed_random(scene);env=TADEnv(n,protocol='paper-parameters-v1');obs,_=env.reset(agility,noisy_agility=False)
        np.random.seed(scene+1000000);packet=public_packet(env,obs);packet['target_heading']=np.asarray(env.attacker.theta,np.float32);done=0
        packet['time_remaining']=np.asarray(max(0.,1.-env.Current_T/60.),np.float32)
        while not done:
            before=len(policy.decisions) if policy else 0
            action=policy(packet) if policy else program_action(FLEET_SOURCE(packet['obs']),0.,bias if arm=='constant_bias' else 0.)
            changed=policy and len(policy.decisions)>before
            if changed:reference=filtered(packet,program_action(FLEET_SOURCE(packet['obs']),0.,bias),env.boids_actions)
            packet,_,done,thrust,_,_=execute(env,packet,reactive_native_action(action,env.boids_actions),safety=arm!='original')
            packet['target_heading']=np.asarray(env.attacker.theta,np.float32)
            packet['time_remaining']=np.asarray(max(0.,1.-env.Current_T/60.),np.float32)
            if changed:policy.decisions[-1].update(executed_thrust=np.asarray(thrust).tolist(),reference_thrust=np.asarray(reference).tolist(),control_l2=float(np.linalg.norm(thrust-reference)))
        row[arm]=outcome_row(env,done)
        if policy:row[arm]['decisions']=policy.decisions
    return row


def evaluate_reactive(args):
    names=set('difference' if a in ('permuted','state-only','interaction-off') else a for a in args.evaluation_arms.split(',') if a not in ('original','arboids_cbf','constant_bias'))
    paths={a:args.output/f'reactive-{a}-{args.seed}.pth' for a in names}
    frozen={a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in paths.items()}
    code=Path(__file__);core=code.parent/'policy/interaction_sac.py'
    frozen.update(implementation=hashlib.sha256(code.read_bytes()+core.read_bytes()).hexdigest(),source=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    conditional_code=code.with_name('conditional_fusion_value.py')
    frozen['conditional_value']=hashlib.sha256(conditional_code.read_bytes()).hexdigest()
    settings=[torch.load(p,weights_only=False)['settings'] for p in paths.values()]
    if any(s!=settings[0] for s in settings) or (settings[0]['defenders'],settings[0]['bias'],settings[0]['agility'])!=(args.defenders,args.program_bias,args.agility):
        raise ValueError('All arms and evaluation must share the task and controller settings.')
    tasks=[(3670000000+args.offset+i,args.defenders,args.agility,args.program_bias) for i in range(args.count)]
    rows=[];start=time.monotonic()
    filename=args.output/f'reactive-episodes-{args.seed}-{args.offset}-{args.count}-{hashlib.sha256(args.evaluation_arms.encode()).hexdigest()[:10]}.json'
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_reactive_evaluation,initargs=(args.output,args.seed,args.evaluation_arms)) as pool:
        for row in pool.map(reactive_episode,tasks):
            rows.append(row)
            if len(rows)%16==0:
                atomic_json(filename,dict(rows=rows,frozen=frozen,settings=settings[0],complete=False))
                print(json.dumps(dict(completed=len(rows),seconds=time.monotonic()-start)),flush=True)
    if frozen['implementation']!=hashlib.sha256(code.read_bytes()+core.read_bytes()).hexdigest() or frozen['source']!=hashlib.sha256(SOURCE.read_bytes()).hexdigest() or frozen['conditional_value']!=hashlib.sha256(conditional_code.read_bytes()).hexdigest() or any(frozen[a]!=hashlib.sha256(p.read_bytes()).hexdigest() for a,p in paths.items()):
        raise RuntimeError('Evaluation implementation or policy changed during execution.')
    summary={a:dict(mean=float(np.mean([r[a]['capture_time'] for r in rows])),captures=sum(r[a]['capture'] for r in rows),collisions=sum(r[a]['collision'] for r in rows)) for a in args.evaluation_arms.split(',')}
    generator=np.random.default_rng(17731);indices=generator.integers(len(rows),size=(20000,len(rows)));contrasts={}
    for arm in args.evaluation_arms.split(','):
        if arm=='difference':continue
        delta=np.array([r['difference']['capture_time']-r[arm]['capture_time'] for r in rows]);sample=delta[indices].mean(1)
        contrasts[arm]=dict(delta=float(delta.mean()),ci95=np.quantile(sample,[.025,.975]).tolist(),ci99=np.quantile(sample,[.005,.995]).tolist())
    atomic_json(filename,dict(rows=rows,frozen=frozen,settings=settings[0],complete=True,summary=summary,contrasts=contrasts))
    print(json.dumps(dict(summary=summary,contrasts=contrasts)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('audit','roots','outcomes','fit-values','evaluate-values','fit-gates','evaluate-gates','gate-episodes','summarize-identification','fleet-collect','fleet-fit','fleet-episodes','fleet-project','fleet-ridge','fleet-rank','fleet-ensemble','fleet-summarize','fleet-static','program-collect','program-refine','program-fit','program-kernel','program-policy','program-episodes'))
    parser._actions[1].choices=(*parser._actions[1].choices,'reactive-train','reactive-kernel','reactive-evaluate','reactive-refit-conditional','reactive-prior-fit','reactive-prior-credits','reactive-prior-check','reactive-prior-design','reactive-prior-update','reactive-prior-anchor','reactive-factor-fit','reactive-factor-credits','reactive-factor-update')
    parser.add_argument('--prior-repeats',type=int,default=4)
    parser.add_argument('--prior-value',choices=('linear','reliability','distribution','monotone','orthogonal','candidate','conservative'),default='distribution')
    parser.add_argument('--exact-joint-fleet',action='store_true')
    parser.add_argument('--canonical-model',action='store_true')
    parser.add_argument('--recursive-value-update',action='store_true')
    parser.add_argument('--conditional-posterior-update',action='store_true')
    parser.add_argument('--continuous-cooperation-test',action='store_true')
    parser.add_argument('--crossfit-cooperation',action='store_true')
    parser.add_argument('--decision-focused-edge',action='store_true')
    parser.add_argument('--attention-unary-anchor',action='store_true')
    parser.add_argument('--attention-readout-update',action='store_true')
    parser.add_argument('--component-continuation',choices=('source','independent'),default='source')
    parser.add_argument('--component-orthogonal',action='store_true')
    parser.add_argument('--component-crossfit-value',action='store_true')
    parser.add_argument('--component-complete-support',action='store_true')
    parser.add_argument('--component-conditional-auxiliary',action='store_true')
    parser.add_argument('--component-covariance-credit',action='store_true')
    parser.add_argument('--value-initialization-seed',type=int)
    parser.add_argument('--paired-policy-update',action='store_true')
    parser.add_argument('--common-update-parent',choices=('difference','absolute'))
    parser.add_argument('--collecting-arm',choices=('difference','absolute','additive'),default='difference')
    parser.add_argument('--active-joint-credits',action='store_true')
    parser._actions[1].choices=(*parser._actions[1].choices,'reactive-joint-credits')
    parser._actions[1].choices=(*parser._actions[1].choices,'reactive-coordination-fit')
    parser._actions[1].choices=(*parser._actions[1].choices,'reactive-coordination-policy')
    parser._actions[1].choices=(*parser._actions[1].choices,'reactive-complete-pair-credits','reactive-edge-fit')
    parser._actions[1].choices=(*parser._actions[1].choices,'reactive-response-bank','reactive-attention-fit','reactive-attention-credits','reactive-attention-posterior','reactive-component-credits','reactive-component-fit')
    parser.add_argument('--cooperative-update',action='store_true')
    parser.add_argument('--decision-credits',action='store_true')
    parser.add_argument('--robust-update',action='store_true')
    parser.add_argument('--local-coordination',action='store_true')
    parser.add_argument('--selected-coordination',action='store_true')
    parser.add_argument('--centered-coordination',action='store_true')
    parser.add_argument('--coordination-centers',type=int,default=0)
    parser.add_argument('--posterior-risk',type=float,default=0.)
    parser.add_argument('--joint-value-refit',action='store_true')
    parser.add_argument('--common-value-initialization',action='store_true')
    parser.add_argument('--adaptive-interaction-link',action='store_true')
    parser.add_argument('--value-residual',action='store_true')
    parser.add_argument('--reactive-iterations',type=int,default=3)
    parser.add_argument('--reactive-batch',type=int,default=4)
    parser.add_argument('--exact-mirror',action='store_true')
    parser.add_argument('--reactive-step-check',action='store_true')
    parser.add_argument('--critic-members',type=int,default=1)
    parser.add_argument('--critic-risk',type=float,default=.5)
    parser.add_argument('--reactive-bootstrap-data',type=Path)
    parser.add_argument('--reactive-origin-seed',type=int)
    parser.add_argument('--reserve-roots',type=int,default=0)
    parser.add_argument('--target-pose',action='store_true')
    parser.add_argument('--source-device',default='cpu')
    parser.add_argument('--complete-state',action='store_true')
    parser.add_argument('--joint-fleet',action='store_true')
    parser.add_argument('--gate-levels',type=int,choices=(3,5),default=3)
    parser.add_argument('--full-gate',action='store_true')
    parser.add_argument('--response-encoding',action='store_true')
    parser.add_argument('--reaction-encoding',action='store_true')
    parser.add_argument('--potential-value',action='store_true')
    parser.add_argument('--curved-potential',action='store_true')
    parser.add_argument('--constrained-potential',action='store_true')
    parser.add_argument('--coupled-potential',action='store_true')
    parser.add_argument('--local-precision',action='store_true')
    parser.add_argument('--preference-value',action='store_true')
    parser.add_argument('--complete-pair-credits',action='store_true')
    parser.add_argument('--decision-loss',action='store_true')
    parser.add_argument('--cardinal-gates',action='store_true')
    parser.add_argument('--refine-gates',action='store_true')
    parser.add_argument('--conditional-policy',action='store_true')
    parser.add_argument('--calibrate-risk',action='store_true')
    parser.add_argument('--expand-step',action='store_true')
    parser.add_argument('--initial-credit-count',type=int)
    parser.add_argument('--reactive-augment-data',type=Path)
    parser.add_argument('--validation-repeats',type=int,default=1)
    parser.add_argument('--accept-repeats',type=int,default=4)
    parser.add_argument('--accept-roots',type=int,default=64)
    parser.add_argument('--accept-standard-errors',type=float,default=0.)
    parser.add_argument('--program-bias',type=float,default=-.05)
    parser.add_argument('--program-radius',type=float,default=.2)
    parser.add_argument('--program-trigger',type=float,default=35.)
    parser.add_argument('--program-policy-weight',type=float,default=0.)
    parser.add_argument('--kernel-metric',choices=('balanced','control'),default='balanced')
    parser.add_argument('--candidate-context',action='store_true')
    parser.add_argument('--agility',type=float,default=2.25)
    parser.add_argument('--projection-steps',type=int,default=30)
    parser.add_argument('--projection-repeats',type=int,default=1)
    parser.add_argument('--candidate-mode',choices=('source','feedback'),default='source')
    parser.add_argument('--interaction-residual',action='store_true')
    parser.add_argument('--absolute-anchor',action='store_true')
    parser.add_argument('--mixed-only',action='store_true')
    parser.add_argument('--validation-criterion',choices=('gain','mae','mse'),default='gain')
    parser.add_argument('--nonlinear-continuation',action='store_true')
    parser.add_argument('--monotone-readout',action='store_true')
    parser.add_argument('--context-modulation',action='store_true')
    parser.add_argument('--conditional-edges',action='store_true')
    parser.add_argument('--measured-target',action='store_true')
    parser.add_argument('--calibrate-margin',action='store_true')
    parser.add_argument('--bootstrap-seed',type=int)
    parser.add_argument('--no-bootstrap',action='store_true')
    parser.add_argument('--initial-seed',type=int)
    parser.add_argument('--defenders',type=int,default=6)
    parser.add_argument('--active-defenders',type=int,default=0)
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--decision-period',type=int)
    parser.add_argument('--decision-start',type=float,default=0.)
    parser.add_argument('--decision-temperature',type=float,default=0.)
    parser.add_argument('--decision-limit',type=int,default=0)
    parser.add_argument('--activation-radius',type=float,default=0.)
    parser.add_argument('--reference-prior',type=float,default=.9)
    parser.add_argument('--evaluation-arms',default='difference,absolute,no_peer,arboids_cbf')
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--count',type=int,default=64)
    parser.add_argument('--offset',type=int,default=0);parser.add_argument('--split',default='train',choices=('train','validation','test'))
    parser.add_argument('--repeats',type=int,default=6);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--train-count',type=int,default=256);parser.add_argument('--validation-count',type=int,default=64)
    parser.add_argument('--updates',type=int,default=2000);parser.add_argument('--seed',type=int,default=9101)
    parser.add_argument('--seeds',default='9220,9221,9222')
    parser.add_argument('--test-offset',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--batched',action='store_true')
    parser.add_argument('--uncertainty-weighting',action='store_true')
    parser.add_argument('--contrast-covariance',action='store_true')
    parser.add_argument('--covariance-floor',type=float,default=4.)
    parser.add_argument('--covariance-local-weight',type=float,default=.5)
    parser.add_argument('--joint-contrasts',action='store_true')
    parser.add_argument('--event-value',action='store_true')
    parser.add_argument('--additive-value',action='store_true')
    parser.add_argument('--separate-state',action='store_true')
    parser.add_argument('--pair-interaction',action='store_true')
    parser.add_argument('--joint-actor',action='store_true')
    parser.add_argument('--aux-weight',type=float,default=10.)
    parser.add_argument('--arms',default=','.join(VALUE_ARMS))
    parser.add_argument('--candidate-contexts',type=int,default=1)
    parser.add_argument('--intervention-steps',type=int,default=10)
    parser.add_argument('--root-source',type=Path)
    parser.add_argument('--extra-train-output',type=Path)
    parser.add_argument('--extra-train-count',type=int,default=256)
    parser.add_argument('--batch-safety',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--scalar-vessels',action='store_true')
    parser.add_argument('--max-decisions',type=int,default=1)
    parser.add_argument('--policy-kind',choices=('actor','value'),default='actor')
    parser.add_argument('--gate-trust',type=float,default=.03)
    parser.add_argument('--gate-reference-prior',type=float,default=1./3.)
    parser.add_argument('--influence-roots',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);start=time.monotonic();rows=[]
    if args.mode=='reactive-train':return fit_reactive(args)
    if args.mode=='reactive-kernel':return fit_reactive_kernel_policy(args)
    if args.mode=='reactive-evaluate':return evaluate_reactive(args)
    if args.mode=='reactive-refit-conditional':return refit_conditional_policy(args)
    if args.mode=='reactive-prior-fit':return fit_reactive_prior(args)
    if args.mode=='reactive-prior-credits':return collect_reactive_prior_credits(args)
    if args.mode=='reactive-joint-credits':return collect_reactive_joint_credits(args)
    if args.mode=='reactive-coordination-fit':return fit_reactive_coordination_value(args)
    if args.mode=='reactive-coordination-policy':return calibrate_reactive_coordination_policy(args)
    if args.mode=='reactive-complete-pair-credits':return collect_reactive_complete_pair_credits(args)
    if args.mode=='reactive-edge-fit':return fit_reactive_edge_value(args)
    if args.mode=='reactive-attention-fit':return fit_reactive_attention(args)
    if args.mode=='reactive-attention-posterior':return fit_reactive_attention_posterior(args)
    if args.mode=='reactive-component-credits':return collect_reactive_component_credits(args)
    if args.mode=='reactive-component-fit':return fit_reactive_component(args)
    if args.mode=='reactive-attention-credits':return collect_reactive_attention(args)
    if args.mode=='reactive-response-bank':return encode_reactive_joint_responses(args)
    if args.mode=='reactive-prior-check':return check_reactive_prior(args)
    if args.mode=='reactive-prior-design':return collect_reactive_prior_design(args)
    if args.mode=='reactive-prior-update':return update_reactive_prior(args)
    if args.mode=='reactive-prior-anchor':return augment_reactive_prior_anchor(args)
    if args.mode=='reactive-factor-fit':return fit_reactive_factor_kernel(args)
    if args.mode=='reactive-factor-credits':return collect_reactive_factor_credits(args)
    if args.mode=='reactive-factor-update':return update_reactive_factor_kernel(args)
    if args.mode.startswith('fleet-'):return run_fleet(args)
    if args.mode.startswith('program-'):return run_program(args)
    if args.mode=='fit-values':return fit_values(args)
    if args.mode=='evaluate-values':return evaluate_values(args)
    if args.mode=='fit-gates':return fit_gates(args)
    if args.mode=='evaluate-gates':return evaluate_gates(args)
    if args.mode=='gate-episodes':return evaluate_gate_episodes(args)
    if args.mode=='summarize-identification':return summarize_identification(args)
    function={'audit':audit_scene,'roots':collect_root,'outcomes':evaluate_root}[args.mode]
    if args.mode=='outcomes' and args.batched:function=evaluate_root_batched
    if args.mode=='audit':tasks=range(2730000000+args.offset,2730000000+args.offset+args.count)
    elif args.mode=='roots':tasks=[(i,args.split,args.output,args.influence_roots) for i in range(args.offset,args.offset+args.count)]
    else:tasks=[(args.split,i,args.repeats,args.output,dict(candidate_contexts=args.candidate_contexts,root_source=args.root_source,batch_safety=args.batch_safety,resume=args.resume,scalar_vessels=args.scalar_vessels,intervention_steps=args.intervention_steps)) for i in range(args.offset,args.offset+args.count)] if args.batched else [(args.split,i,args.repeats,args.output) for i in range(args.offset,args.offset+args.count)]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        for row in pool.map(function,tasks,chunksize=1):
            rows.append(row)
            if len(rows)%8==0:print(json.dumps(dict(mode=args.mode,split=args.split,completed=len(rows),elapsed=round(time.monotonic()-start,1))),flush=True)
    if args.mode=='audit':
        summary={k:summarize_pairs(rows,k) for k in ('aligned','permuted','state-only')}
        for probe in ('permuted','state-only'):
            d=[d for r in rows for d in r['diagnostics'] if d['probe']==probe]
            summary[probe]['same_state']={k:float(np.mean([x[k] for x in d])) for k in ('gate_l1','gate_max','thrust_l2','thrust_max')}
            summary[probe]['same_state']['changed_control_fraction']=float(np.mean([x['thrust_l2']>1. for x in d]))
        result=dict(checkpoint=str(CHECKPOINT),sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),fusion_scale=.5,option_steps=10,summary=summary,rows=rows)
        atomic_json(args.output/f'audit-{args.offset}-{args.count}.json',result)
        print(json.dumps(summary),flush=True)
    else:print(json.dumps(dict(completed=len(rows),elapsed=round(time.monotonic()-start,1),mean_opportunity=np.mean([r.get('opportunity',0.) for r in rows]))),flush=True)


if __name__=='__main__':main()
