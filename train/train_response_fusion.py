"""Local research on response-supervised relational fusion and intervention values.

An intervention is a gate correction held for H control periods. Candidates,
the adversary, and CBF are recomputed at every period. The value target is the
complete return of precisely that intervention followed by the frozen policy.
The same trajectories supervise relative-motion responses. No rollout or
candidate search is used by the learned deployment policy.
"""
import study_runtime
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from envs.TADgame import TADEnv
from envs.snapshot import seed_random, preserved_random_state
from interaction_evaluation import OriginalPolicy
from interaction_rollout import public_packet, execute, compact_snapshot
from policy.interaction_sac import InteractionActor, tensor_packet
from train_interaction import atomic_json, atomic_torch, outcome_row


SOURCE = None
LEARNED = None
TAIL_POLICY = None
SAFETY_DERIVATIVES = {}


def safety_derivatives(defenders):
    """Differentiate the same upstream QP used for every real execution."""
    if defenders not in SAFETY_DERIVATIVES:
        import jax
        import jax.numpy as jnp
        import qpax
        from interaction_rollout import safety_controller
        controller=safety_controller(defenders)
        def single(motion,thrust):
            value=controller.filter.safety_filter(motion.reshape(-1),thrust.reshape(-1)/1000.)
            return (1000.*jnp.clip(value,-.5,1.)).reshape(defenders,2)
        forward=jax.jit(jax.vmap(single))
        def differentiated(motion,thrust):
            p,q,_,_,g,h=controller.filter.qp_data(motion.reshape(-1),thrust.reshape(-1)/1000.)
            # The upstream default smooths the backward QP at kappa=1e-3,
            # which measurably attenuates feasible fusion derivatives. Only
            # this training derivative uses a tighter complementarity level;
            # both forward solves retain the unchanged source solver/tolerance.
            value=qpax.solve_qp_elastic_primal(p,q,g,h,
                penalty=jnp.asarray(controller.filter.constraint_relaxation_penalties),
                solver_tol=controller.filter.solver_tol,target_kappa=1e-8)
            return (1000.*jnp.clip(value,-.5,1.)).reshape(defenders,2)
        differentiated=jax.jit(jax.vmap(differentiated))
        backward=jax.jit(lambda motion,thrust,cotangent:jax.vjp(differentiated,motion,thrust)[1](cotangent))
        SAFETY_DERIVATIVES[defenders]=(forward,backward)
    return SAFETY_DERIVATIVES[defenders]


class SafetyDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx,motion,thrust):
        forward,_=safety_derivatives(motion.shape[1])
        x=motion.detach().cpu().double().numpy();u=thrust.detach().cpu().double().numpy()
        count=len(x);capacity=max(8,1<<(count-1).bit_length())
        if capacity>count:
            x=np.concatenate((x,np.repeat(x[-1:],capacity-count,0)))
            u=np.concatenate((u,np.repeat(u[-1:],capacity-count,0)))
        ctx.arrays=(x,u);ctx.count=count;ctx.device=motion.device;ctx.dtypes=(motion.dtype,thrust.dtype)
        return torch.as_tensor(np.asarray(forward(x,u))[:count].copy(),dtype=thrust.dtype,device=thrust.device)
    @staticmethod
    def backward(ctx,cotangent):
        x,u=ctx.arrays;_,backward=safety_derivatives(x.shape[1])
        dy=cotangent.detach().cpu().double().numpy()
        if len(x)>ctx.count:dy=np.concatenate((dy,np.zeros((len(x)-ctx.count,*dy.shape[1:]),dtype=dy.dtype)))
        dx,du=backward(x,u,dy)
        return (torch.as_tensor(np.asarray(dx)[:ctx.count].copy(),dtype=ctx.dtypes[0],device=ctx.device),
                torch.as_tensor(np.asarray(du)[:ctx.count].copy(),dtype=ctx.dtypes[1],device=ctx.device))


class DifferentiableInterventionDynamics:
    """Training-only, reparameterized conditional futures with original physics.

    State contains pose and the integrator's water-relative velocity. Measured
    velocity deliberately retains the source simulator's final-substep timing.
    The original CBF QP is called through its own implicit differentiation API.
    """
    @staticmethod
    def thrust(force,heading,agility=1.):
        direction=F.normalize(force,dim=-1,eps=1e-12);c,s=heading.cos(),heading.sin()
        forward=c*direction[...,0]+s*direction[...,1];side=-s*direction[...,0]+c*direction[...,1]
        angle=torch.atan2(side,forward)
        return torch.stack((forward+.75*angle,forward-.75*angle),-1).clamp(-.5,1.)*1000.*agility

    @staticmethod
    def apf(state):
        p=state[:,-1,:2];force=-.1*p;limit=torch.full_like(p[:,0],50.)
        for i in range(state.shape[1]-1):
            delta=p-state[:,i,:2];norm=delta.norm(dim=-1).clamp_min(1e-12);distance=(norm-8.).clamp_min(.1)
            active=distance<limit;size=3000.*(1./distance-1./limit)/distance.square()
            force=force+torch.where(active[:,None],size[:,None]*delta/norm[:,None],torch.zeros_like(delta))
            limit=torch.where(active,distance,limit)
        return force

    @staticmethod
    def vessel(state,thrust,current):
        position,heading,water=state[...,:2],state[...,2],state[...,3:6]
        for _ in range(4):
            measured=water+current
            position=position+.05*measured[...,:2];heading=torch.remainder(heading+.05*measured[...,2],2.*torch.pi)
            c,s=heading.cos(),heading.sin()
            ur=c*water[...,0]+s*water[...,1];vr=-s*water[...,0]+c*water[...,1]
            u=c*measured[...,0]+s*measured[...,1];v=-s*measured[...,0]+c*measured[...,1];r=measured[...,2]
            du=(400.*r*v-(100.+150.*ur.abs())*ur+thrust.sum(-1))/380.
            dv=(-400.*r*u+20.*ur*r-(100.+150.*vr.abs())*vr)/400.
            dr=(-20.*ur*vr-(980.+950.*r.abs())*r+1.25*(thrust[...,0]-thrust[...,1]))/1430.
            ur=ur+.05*du;vr=vr+.05*dv;r=r+.05*dr
            water=torch.stack((c*ur-s*vr,s*ur+c*vr,r),-1)
        return torch.cat((position,heading[...,None],water),-1),measured

    @staticmethod
    def boids(state,measured):
        boats=state[:,:-1];n=boats.shape[1];position=boats[...,:2]
        delta=position[:,:,None]-position[:,None];distance=delta.norm(dim=-1)
        active=(distance<15.)&~torch.eye(n,dtype=torch.bool,device=state.device)[None]
        separation=(active[...,None]*delta/(distance-3.5).clamp_min(.01)[...,None].square()).sum(2)
        alignment=measured[:,:-1,:2].mean(1,keepdim=True).expand(-1,n,-1)
        cohesion=position.mean(1,keepdim=True)-position
        force=.5*(state[:,-1:, :2]-position)+10.*separation+.1*alignment+.1*cohesion
        thrust=DifferentiableInterventionDynamics.thrust(force,boats[...,2])
        return thrust,separation,alignment,cohesion

    @staticmethod
    def packet(state,measured,agility,remaining):
        boats=state[:,:-1];b,n=boats.shape[:2];heading=boats[...,2]
        thrust,separation,alignment,cohesion=DifferentiableInterventionDynamics.boids(state,measured)
        def polar(vector,angle=heading):
            length=vector.norm(dim=-1)
            valid=length>1e-10
            # Zero speed has no direction: retain the source's atan2(0,0)=0
            # value while avoiding an undefined derivative at initial rest.
            vy=torch.where(valid,vector[...,1],torch.zeros_like(length))
            vx=torch.where(valid,vector[...,0],torch.ones_like(length))
            bearing=torch.atan2(vy,vx)-angle
            bearing=torch.atan2(bearing.sin(),bearing.cos())
            return torch.stack((length,bearing),-1)
        peers=boats[:,None,:,:2]-boats[:,:,None,:2]
        peer=polar(peers,heading[:,:,None]);mask=~torch.eye(n,dtype=torch.bool,device=state.device)
        peer=peer[:,mask].reshape(b,n,2*(n-1))
        obs=torch.cat((polar(-boats[...,:2]),polar(state[:,-1:,:2]-boats[...,:2]),
            polar(measured[:,-1:,:2].expand(-1,n,-1)),polar(separation),polar(alignment),polar(cohesion),
            (thrust-250.)/750.,peer),-1)
        central=torch.cat((state[...,:2]/60.,state[...,2,None].sin(),state[...,2,None].cos(),state[...,3:5]/5.,state[...,5,None]),-1).flatten(1)
        central=torch.cat((central,agility[:,None]/3.,remaining[:,None]),-1)
        motion=torch.cat((boats[...,:3],measured[:,:-1]),-1)
        return dict(obs=obs,motion=motion,central=central)

    @staticmethod
    def from_packet(packet):
        n=packet['motion'].shape[1];raw=packet['central'][:,:7*(n+1)].reshape(-1,n+1,7)
        state=torch.cat((60.*raw[...,:2],torch.atan2(raw[...,2],raw[...,3])[...,None],5.*raw[...,4:6],raw[...,6:7]),-1)
        angle=packet['motion'][:,0,2]+packet['obs'][:,0,5]
        velocity=packet['obs'][:,0,4,None]*torch.stack((angle.cos(),angle.sin()),-1)
        # Attacker yaw measurement is not consumed by the APF or observations.
        measured=torch.cat((packet['motion'][...,3:6],torch.cat((velocity,state[:,-1,5,None]),-1)[:,None]),1)
        return state,measured,3.*packet['central'][:,-2],packet['central'][:,-1]

    @staticmethod
    def raw_history(packet):
        h=packet['intent_history'];m,obs=packet['motion'],packet['obs']
        angle=m[:,0,2]+obs[:,0,3];position=m[:,0,:2]+obs[:,0,2,None]*torch.stack((angle.cos(),angle.sin()),-1)
        phi=torch.atan2(position[:,1],position[:,0]);c,s=phi.cos()[:,None],phi.sin()[:,None]
        def world(v):return torch.stack((c*v[...,0]-s*v[...,1],s*v[...,0]+c*v[...,1]),-1)
        return torch.cat((position[:,None]+5.*world(h[...,2:4]),5.*world(h[...,:2]),world(h[...,4:6]),h[...,7:8]),-1)

    @staticmethod
    def advance_history(raw,state,measured):
        force=F.normalize(DifferentiableInterventionDynamics.apf(state),dim=-1,eps=1e-12)
        latest=torch.cat((state[:,-1,:2],measured[:,-1,:2],force,torch.ones_like(force[:,:1])),-1)
        return torch.cat((raw[:,1:],latest[:,None]),1)

    @staticmethod
    def intent_history(raw):
        position=raw[:,-1,:2];phi=torch.atan2(position[:,1],position[:,0]);c,s=phi.cos()[:,None],phi.sin()[:,None]
        def local(v):return torch.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1]),-1)
        return torch.cat((local(raw[...,2:4])/5.,local(raw[...,:2]-position[:,None])/5.,local(raw[...,4:6]),
            raw[...,:2].norm(dim=-1,keepdim=True)/60.,raw[...,6:7]),-1)

    @staticmethod
    def step(state,measured,agility,action,innovations,nominal_delta=None):
        boids=DifferentiableInterventionDynamics.boids(state,measured)[0]
        nominal=action[...,2:3]*(750.*action[...,:2]+250.)+(1.-action[...,2:3])*boids
        if nominal_delta is not None:nominal=nominal+nominal_delta
        motion=torch.cat((state[:,:-1,:3],measured[:,:-1]),-1)
        thrust=SafetyDerivative.apply(motion,nominal)
        attacker=DifferentiableInterventionDynamics.thrust(DifferentiableInterventionDynamics.apf(state)+.3*innovations[:,:2],state[:,-1,2],agility[:,None])
        control=torch.cat((thrust,attacker[:,None]),1)
        # Original draw order: APF, attacker current, then defenders.
        current=innovations[:,2:].reshape(state.shape[0],state.shape[1],3)
        current=torch.cat((current[:,1:],current[:,:1]),1)*state.new_tensor([.2,.2,.02])
        return DifferentiableInterventionDynamics.vessel(state,control,current)


def initialize(source, checkpoint=None,tail_checkpoint=None,fusion_scale=None,fusion_period=None,fusion_bias=0.,candidate_probe='aligned',reference_checkpoint=None):
    global SOURCE, LEARNED, TAIL_POLICY
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore', message='.*Lgh is zero.*')
    SOURCE = ResponsePolicy(source,reference_checkpoint) if reference_checkpoint else OriginalPolicy(source)
    LEARNED = ResponsePolicy(source, checkpoint) if checkpoint else None
    if LEARNED is not None:
        if fusion_scale is not None:LEARNED.fusion_scale=fusion_scale
        if fusion_period is not None:LEARNED.horizon=fusion_period
        LEARNED.fusion_bias=fusion_bias
        if candidate_probe!='aligned' and not getattr(LEARNED.model,'factor_policy',False):
            raise ValueError('Candidate configuration probes require the cooperative factor actor.')
        LEARNED.model.candidate_probe=candidate_probe
    TAIL_POLICY = ResponsePolicy(source,tail_checkpoint) if tail_checkpoint else None


def motion_response(before, after):
    """Public relative responses in each vessel's *root* body frame."""
    x, y = before['motion'], after['motion']
    c, s = np.cos(x[:, 2]), np.sin(x[:, 2])
    def rotate(v):
        return np.stack((c*v[:, 0]+s*v[:, 1], -s*v[:, 0]+c*v[:, 1]), -1)
    angle = y[:, 2]-x[:, 2]
    node = np.concatenate((rotate(y[:, :2]-x[:, :2])/5.,
        rotate(y[:, 3:5]-x[:, 3:5])/5., np.sin(angle)[:, None],
        (np.cos(angle)-1)[:, None], (y[:, 5]-x[:, 5])[:, None]), -1)
    def pair(m):
        delta = m[None, :, :]-m[:, None, :]
        p = np.stack((c[:, None]*delta[..., 0]+s[:, None]*delta[..., 1],
            -s[:, None]*delta[..., 0]+c[:, None]*delta[..., 1]), -1)/5.
        v = np.stack((c[:, None]*delta[..., 3]+s[:, None]*delta[..., 4],
            -s[:, None]*delta[..., 3]+c[:, None]*delta[..., 4]), -1)/5.
        return np.concatenate((p, v, np.sin(delta[..., 2:3]),
                               np.cos(delta[..., 2:3]), delta[..., 5:6]), -1)
    return node.astype(np.float32), (pair(y)-pair(x)).astype(np.float32)


def target_response(before, after):
    """Change of target-relative position/velocity in each root body frame."""
    heading = before['motion'][:,2]
    c,s = np.cos(heading),np.sin(heading)
    def relative(packet):
        obs,m = packet['obs'],packet['motion']
        angle = m[:,2]+obs[:,3]
        position = obs[:,2,None]*np.stack((np.cos(angle),np.sin(angle)),-1)
        velocity_angle = m[:,2]+obs[:,5]
        velocity = obs[:,4,None]*np.stack((np.cos(velocity_angle),np.sin(velocity_angle)),-1)-m[:,3:5]
        return position,velocity
    bp,bv = relative(before);ap,av = relative(after)
    def rotate(v):
        return np.stack((c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1]),-1)/5.
    return np.concatenate((rotate(ap-bp),rotate(av-bv)),-1).astype(np.float32)


def task_aligned_noise(env,noise_sign=1.):
    """Couple auxiliary innovations by capture role, preserving every marginal.

    Sorting independent equal-variance defender innovations and rotating each
    isotropic XY normal are conditional orthogonal maps. They use only the
    state before that step's noise is drawn. The execution environment is not
    modified; this coupling is installed only on counterfactual snapshots.
    """
    original_current,original_force=env.generate_random_current,env._apf_force_noise
    queued=[]
    def rotate(value,angle):
        c,s=np.cos(angle),np.sin(angle);result=value.copy()
        result[:2]=(c*value[0]-s*value[1],s*value[0]+c*value[1])
        return result
    def force():
        nonlocal queued
        # Fix the state-dependent transformation before drawing innovations.
        relative=np.array([env.attacker.pos-d.pos for d in env.defender_list])
        distance=np.linalg.norm(relative,axis=-1)
        order=np.argsort(distance,kind='stable')
        angles=np.arctan2(relative[:,1],relative[:,0])
        priority=np.exp(-(distance-distance.min())/5.);priority/=priority.sum()
        direction=(priority[:,None]*relative/np.maximum(distance[:,None],1e-6)).sum(0)
        attacker_angle=np.arctan2(direction[1],direction[0])
        attacker_heading=float(env.attacker.theta)
        raw_force=noise_sign*original_force()
        innovations=[noise_sign*original_current() for _ in range(len(order)+1)]
        defender=[None]*len(order)
        for rank,index in enumerate(order):defender[index]=rotate(innovations[rank+1],angles[index])
        queued=[rotate(innovations[0],attacker_angle),*defender]
        return rotate(raw_force,attacker_heading)
    def current():
        if not queued:raise RuntimeError('The auxiliary coupling expects one APF draw before the vessel currents.')
        return queued.pop(0)
    env._apf_force_noise,env.generate_random_current=force,current


def branch(snapshot, correction, horizon, future, *, response_only=False, noise_sign=1., prefix_only=False,intervention_link='additive',noise_coupling='cartesian',past=None,innovations=None):
    env = snapshot.restore(future_seed=future)
    if innovations is not None:install_innovations(env,innovations)
    if noise_coupling=='role':
        task_aligned_noise(env,noise_sign)
    elif noise_sign == -1.:
        current,force = env.generate_random_current,env._apf_force_noise
        env.generate_random_current = lambda: -current()
        env._apf_force_noise = lambda: -force()
    start = packet = public_packet(env)
    total, discount, done, steps = 0., 1., 0, 0
    response = None
    history=list(past or [])
    while not done:
        if steps>=horizon and intervention_link=='feedback' and TAIL_POLICY is not None:
            if steps==horizon:
                TAIL_POLICY.remaining=0;TAIL_POLICY.history=history[-7:].copy()
                TAIL_POLICY.last_time=float(start['central'][-1])
            action=TAIL_POLICY.choose_action(packet,deterministic=True)[0]
        else:
            action = SOURCE.choose_action(packet, deterministic=True)[0]
        modal_gates=None
        if steps < horizon:
            if intervention_link=='modal':
                numerator=action[:,2,None]*np.exp(correction)
                modal_gates=numerator/(1.-action[:,2,None]+numerator)
            elif intervention_link in ('odds','feedback'):
                shift=correction
                if intervention_link=='feedback':
                    history.append(dict(obs=packet['obs'],motion=packet['motion']));history=history[-8:]
                    p={k:torch.as_tensor(packet[k],dtype=torch.float32)[None] for k in ('obs','motion','central')}
                    p['intent_history']=torch.as_tensor(public_intent_history(history)[-1:])
                    shift=relational_feedback(LEARNED.model,p,torch.as_tensor(action)[None],
                        torch.as_tensor(correction),LEARNED.cooperation_scale)[0].numpy()
                numerator=action[:,2]*np.exp(shift)
                action[:,2]=numerator/(1.-action[:,2]+numerator)
            else:
                action[:, 2] = np.clip(action[:, 2]+correction, 0., 1.)
        if modal_gates is None:
            packet, reward, done, _, _, _ = execute(env, packet, action, safety=True)
        else:
            packet,reward,done,_,_,_=execute_modal_fusion(env,packet,action,modal_gates)
        total += discount*float(np.mean(reward))
        discount *= .99
        steps += 1
        if response is None and (steps == horizon or done):
            response = motion_response(start, packet)
            target = target_response(start,packet)
            following = packet
            if response_only: return target
        if prefix_only and steps>=horizon: break
    outcome = outcome_row(env,done)
    if not done: outcome['capture_time'] = float('nan')
    return dict(**outcome, discounted_return=total if done else float('nan'), steps=steps,
                node_response=response[0], pair_response=response[1],target_response=target,
                intervention_next=following,completed_episode=bool(done),noise_sign=noise_sign,
                capturer=(int(np.argmin([np.linalg.norm(d.pos-env.attacker.pos) for d in env.defender_list])) if outcome['capture'] else -1))


def install_innovations(env,innovations):
    """A supplied Gaussian path replaces draws only in auxiliary episodes."""
    step=-1;vessel=0
    def force():
        nonlocal step,vessel
        step+=1;vessel=0
        if step>=len(innovations):raise RuntimeError('The auxiliary Gaussian path is exhausted.')
        return .3*innovations[step,:2]
    def current():
        nonlocal vessel
        value=innovations[step,2+3*vessel:5+3*vessel]*np.array([.2,.2,.02]);vessel+=1
        return value
    env._apf_force_noise=force;env.generate_random_current=current


def haar_innovations(normal):
    """Orthonormal low-frequency-first coordinates for independent increments."""
    samples,steps,dimension=normal.shape
    if steps&(steps-1):raise ValueError('The Gaussian Haar path needs a power-of-two length.')
    result=np.broadcast_to(normal[:,:1]/np.sqrt(steps),normal.shape).copy();start=1;blocks=1
    while blocks<steps:
        length=steps//blocks;sign=np.r_[np.ones(length//2),-np.ones(length//2)]/np.sqrt(length)
        result+=(normal[:,start:start+blocks,None,:]*sign[None,None,:,None]).reshape(samples,steps,dimension)
        start+=blocks;blocks*=2
    return result


def collect_quadrature_root(task):
    root,bank,state_source,output,horizon=task
    source_root=(80 if bank==0 else 40)+root;raw=torch.load(state_source/f'{source_root}.pth',weights_only=False)
    snapshot=raw['snapshot'];packet=source_context(raw);n=len(packet['obs']);dimension=2+3*(n+1);steps=512
    index=int(round(snapshot.environment.Current_T/.2));past=[dict(obs=p['obs'],motion=p['motion']) for p in raw['transitions'][max(0,index-7):index]]
    modes=np.array([[0.,0.,0.],[0.,1.,0.],[0.,1.,1.],[0.,-1.,-1.],[1.,0.,0.],[-1.,0.,0.]],dtype=np.float32)
    started=time.perf_counter();families={}
    for method in ('antithetic','sobol_haar'):
        paths=[]
        for group in range(4):
            seed=4080000000+bank*1000000+root*1000+group*11
            if method=='antithetic':normal=np.random.default_rng(seed).standard_normal((4,steps,dimension))
            else:
                engine=torch.quasirandom.SobolEngine(steps*dimension,scramble=True,seed=seed)
                uniform=engine.draw_base2(2,dtype=torch.float64).clamp(1e-12,1.-1e-12)
                normal=(2.**.5*torch.erfinv(2.*uniform-1.)).numpy().reshape(4,steps,dimension)
            paths.append(np.concatenate((haar_innovations(normal),-haar_innovations(normal)),0))
        outcomes=[]
        for mode in modes:
            trials=[]
            for group in range(4):
                for path in paths[group]:
                    trials.append(branch(snapshot,mode,horizon,4080000000+root,intervention_link='feedback',
                        noise_coupling='role',past=past,innovations=path))
            outcomes.append(trials)
        families[method]=outcomes
    statistics={}
    for method,rows in families.items():
        q=-np.array([[t['capture_time'] for t in trials] for trials in rows]).reshape(len(modes),4,8).mean(-1)
        difference=q[1:]-q[:1]
        statistics[method]=dict(contrast_mean=difference.mean(-1).tolist(),
            variance_of_mean=(difference.var(-1,ddof=1)/4.).tolist(),average_group_variance=float(difference.var(-1,ddof=1).mean()))
    atomic_torch(output/f'{root}.pth',dict(root=root,source_root=source_root,bank=bank,horizon=horizon,packet=packet,
        baseline=SOURCE.choose_action(packet)[0],modes=modes,families=families,statistics=statistics,integration_group_size=8))
    return dict(root=root,bank=bank,seconds=round(time.perf_counter()-started,2),
        group_variance={k:v['average_group_variance'] for k,v in statistics.items()})


def collect_modal_root(task):
    root,bank,state_source,output,horizon,repeats,radius=task
    started=time.perf_counter();raw=torch.load(state_source/f'{root}.pth',weights_only=False)
    packet=source_context(raw);snapshot=raw['snapshot'];n=len(packet['obs'])
    if n!=3:raise ValueError('This matched controllability experiment uses three vessels.')
    columns=np.array([3,2,1,7,11,12])
    corners=np.array([[1.-2.*((row&int(c)).bit_count()%2) for c in columns] for row in range(16)])
    scalar_corners=np.repeat(np.array(list(itertools.product((-1.,1.),repeat=n))),2,axis=1)
    scalar_axes=np.repeat(np.concatenate((np.eye(n),-np.eye(n)),0),2,axis=1)
    axes=np.concatenate((np.eye(2*n),-np.eye(2*n)),0)
    choices=[np.zeros(2*n)];modal=[0];scalar=[0]
    for designs,indices in ((np.r_[corners,axes],modal),(np.r_[scalar_corners,scalar_axes],scalar)):
        for value in designs:
            matches=[i for i,x in enumerate(choices) if np.array_equal(x,value)]
            if matches:index=matches[0]
            else:index=len(choices);choices.append(value)
            indices.append(index)
    corrections=(radius*np.array(choices)).reshape(-1,n,2).astype(np.float32);rows=[]
    for correction in corrections:
        trials=[]
        for repeat in range(repeats):
            future=3850000000+bank*1000000+root*1000+100+repeat//2
            trials.append(branch(snapshot,correction,horizon,future,noise_sign=(-1. if repeat%2 else 1.),
                intervention_link='modal',noise_coupling='role'))
        rows.append(trials)
    atomic_torch(output/f'{root}.pth',dict(root=root,bank=bank,horizon=horizon,packet=packet,
        baseline=SOURCE.choose_action(packet)[0],corrections=corrections,rows=rows,repeats=repeats,
        intervention_coordinates='surge_yaw_log_odds',radius=radius,modal_indices=modal,scalar_indices=scalar,
        antithetic=True,noise_coupling='role'))
    return dict(root=root,bank=bank,seconds=round(time.perf_counter()-started,2),
        steps=sum(t['steps'] for trials in rows for t in trials))


@torch.no_grad()
def relational_feedback(model,packet,baseline,weight,scale):
    """A persistent cooperation objective, with freshly coordinated gates."""
    if not torch.count_nonzero(weight):return torch.zeros_like(baseline[...,2])
    utility,influence=model.cooperative_response(packet,baseline)
    prior=baseline[...,2];belief=prior.clone()
    own=influence.diagonal(dim1=-2,dim2=-1)/scale[0]
    initial=torch.einsum('bij,bj->bi',influence,utility.softmax(-1))/scale[1]
    for _ in range(8):
        responsibility=(utility+torch.einsum('bi,bij->bj',belief-prior,influence)).softmax(-1)
        changed=torch.einsum('bij,bj->bi',influence,responsibility)/scale[1]
        shift=2.*torch.tanh((weight[0]*own+weight[1]*initial+weight[2]*(changed-initial))/2.)
        numerator=prior*shift.exp();proposal=numerator/(1.-prior+numerator)
        belief=.5*(belief+proposal)
    return torch.where((prior==0.)|(prior==1.),torch.zeros_like(prior),
        torch.logit(belief.clamp(1e-6,1.-1e-6))-torch.logit(prior.clamp(1e-6,1.-1e-6)))


def collect_feedback_root(task):
    root,bank,state_source,output,horizon,repeats,defenders=task
    started=time.perf_counter()
    if state_source:
        raw=torch.load(state_source/f'{root}.pth',weights_only=False)
        snapshot=raw['snapshot']
        if 'past' in raw:packet=raw['packet'];past=raw['past']
        else:
            packet=source_context(raw)
            index=int(round(snapshot.environment.Current_T/.2))
            past=[dict(obs=p['obs'],motion=p['motion']) for p in raw['transitions'][max(0,index-7):index]]
    else:
        scene=2610000000+bank*1000000+(defenders-3)*100000+root
        with preserved_random_state():
            seed_random(scene);env=TADEnv(defenders,protocol='paper-parameters-v1')
            obs,_=env.reset(2.25,noisy_agility=True);np.random.seed(scene+500000)
            reservoir=np.random.default_rng(scene+600000);current=public_packet(env,obs)
            history=[];done=count=0
            while not done:
                history.append(dict(obs=current['obs'],motion=current['motion']));history=history[-8:];count+=1
                if reservoir.integers(count)==0:
                    snapshot=compact_snapshot(env);packet=current.copy();packet['intent_history']=public_intent_history(history)[-1]
                    past=history[:-1].copy()
                current,_,done,_,_,_=execute(env,current,SOURCE.choose_action(current)[0],safety=True)
    modes=np.array([[0.,0.,0.],[0.,1.,0.],[0.,1.,1.],[0.,-1.,-1.],[1.,0.,0.],[-1.,0.,0.]],dtype=np.float32)
    rows=[]
    for mode in modes:
        trials=[]
        for repeat in range(repeats):
            future=3950000000+(defenders-3)*10000000+bank*1000000+root*1000+repeat//2
            trials.append(branch(snapshot,mode,horizon,future,noise_sign=(-1. if repeat%2 else 1.),
                intervention_link='feedback',noise_coupling='role',past=past))
        rows.append(trials)
    saved=dict(root=root,bank=bank,horizon=horizon,packet=packet,baseline=SOURCE.choose_action(packet)[0],
        modes=modes,rows=rows,repeats=repeats,antithetic=True,intervention_coordinates='relational_feedback',
        noise_coupling='role',completed_episode=True,learned_continuation=TAIL_POLICY is not None,
        graph_source=LEARNED.checkpoint,continuation_source=(TAIL_POLICY.checkpoint if TAIL_POLICY else None))
    atomic_torch(output/f'{root}.pth',saved)
    return dict(root=root,bank=bank,seconds=round(time.perf_counter()-started,2),
        steps=sum(t['steps'] for trials in rows for t in trials),
        mean_gain=(-np.array([[t['capture_time'] for t in trials] for trials in rows]).mean(1)+np.mean([t['capture_time'] for t in rows[0]])).tolist())


def collect_adapted_context(task):
    """Decision-boundary states reached by the current learned team policy."""
    root,bank,output,defenders=task;scene=4010000000+bank*1000000+(defenders-3)*100000+root
    started=time.perf_counter();seed_random(scene);env=TADEnv(defenders,protocol='paper-parameters-v1')
    obs,_=env.reset(2.25,noisy_agility=True);np.random.seed(scene+500000);rng=np.random.default_rng(scene+600000)
    packet=public_packet(env,obs);history=[];done=step=count=0;modified=0.;LEARNED.remaining=0
    selected=None
    while not done:
        if step%LEARNED.horizon==0:
            count+=1
            if selected is None or (step>0 and rng.integers(max(1,count-1))==0):
                chosen=packet.copy();chosen['intent_history']=public_intent_history([*history,packet])[-1]
                selected=dict(snapshot=compact_snapshot(env),packet=chosen,past=history[-7:].copy(),modified_before=modified)
        action=LEARNED.choose_action(packet)[0];baseline=SOURCE.choose_action(packet)[0]
        modified+=float(np.abs(action[:,2]-baseline[:,2]).sum())
        history.append(dict(obs=packet['obs'],motion=packet['motion']));history=history[-7:]
        packet,_,done,_,_,_=execute(env,packet,action,safety=True);step+=1
    atomic_torch(output/f'{root}.pth',dict(root=root,bank=bank,**selected,outcome=outcome_row(env,done)))
    return dict(root=root,bank=bank,seconds=round(time.perf_counter()-started,2),steps=step,
        selected_time=float(selected['snapshot'].environment.Current_T),modified_before=selected['modified_before'])


def collect_source_episode(task):
    root,bank,output,defenders,exact=task;scene=4230000000+bank*100000+root
    started=time.perf_counter();seed_random(scene);env=TADEnv(defenders,protocol='paper-parameters-v1')
    obs,_=env.reset(2.25,noisy_agility=not exact);np.random.seed(scene+500000)
    rng=np.random.default_rng(scene+600000);packet=public_packet(env,obs);transitions=[];done=0
    while not done:
        if rng.integers(len(transitions)+1)==0:snapshot=compact_snapshot(env)
        action=SOURCE.choose_action(packet)[0]
        following,reward,done,_,_,_=execute(env,packet,action,safety=True)
        transitions.append(dict(**packet,**{'next_'+k:v for k,v in following.items()},action=action,
            reward=np.asarray([np.mean(reward)],dtype=np.float32),done=np.asarray([bool(done)],dtype=np.float32)))
        packet=following
    total=0.
    for index in range(len(transitions)-1,-1,-1):
        item=transitions[index];total=float(item['reward'][0])+.99*total
        item['return']=np.asarray([total],dtype=np.float32)
        item['next_action']=transitions[index+1]['action'] if index+1<len(transitions) else SOURCE.choose_action(packet)[0]
    saved=dict(root=root,bank=bank,scene_seed=scene,snapshot=snapshot,transitions=transitions,outcome=outcome_row(env,done))
    saved['packet']=source_context(saved)
    atomic_torch(output/f'{root}.pth',saved)
    return dict(root=root,bank=bank,steps=len(transitions),seconds=round(time.perf_counter()-started,2))


def collect_root(task):
    root, bank, horizon, repeats, output, state_source, radius, antithetic, prefix_only, defenders = task
    started = time.perf_counter()
    if state_source:
        saved = torch.load(Path(state_source)/f'{root}.pth', weights_only=False, map_location='cpu')
        snapshot = saved['snapshot']
    else:
        scene = 2610000000+bank*1000000+(defenders-3)*100000+root
        with preserved_random_state():
            seed_random(scene)
            env = TADEnv(defenders, protocol='paper-parameters-v1')
            obs, _ = env.reset(2.25, noisy_agility=True)
            np.random.seed(scene+500000)
            reservoir = np.random.default_rng(scene+600000)
            packet, done, count = public_packet(env, obs), 0, 0
            history = []
            while not done:
                history.append(dict(obs=packet['obs'],motion=packet['motion']))
                history = history[-8:]
                count += 1
                if reservoir.integers(count) == 0:
                    snapshot = compact_snapshot(env)
                    chosen_history = public_intent_history(history)[-1]
                action = SOURCE.choose_action(packet, deterministic=True)[0]
                packet, _, done, _, _, _ = execute(env, packet, action, safety=True)
    packet = public_packet(snapshot.environment)
    if state_source and 'intent_history' in saved['packet']:
        packet['intent_history'] = saved['packet']['intent_history']
    elif not state_source:
        packet['intent_history'] = chosen_history
    baseline = SOURCE.choose_action(packet, deterministic=True)[0]
    # Full factorial interventions identify pair synergy, unlike isolated
    # one-vessel changes. They are training treatments, never online choices.
    targets = np.asarray(list(itertools.product((0., 1.), repeat=len(baseline))), dtype=np.float32)
    if radius is None:
        changes = targets-baseline[:,2]
    else:
        changes = np.clip(baseline[:,2]+radius*(2.*targets-1.),0.,1.)-baseline[:,2]
    corrections = np.vstack((np.zeros(len(baseline), dtype=np.float32),changes))
    path = Path(output)/f'{root}.pth'
    previous = torch.load(path,weights_only=False,map_location='cpu') if path.exists() else None
    if previous and previous['horizon'] != horizon:
        raise ValueError('Existing data use a different intervention duration.')
    if previous and previous.get('intervention_radius') != radius:
        raise ValueError('Existing data use a different intervention radius.')
    if previous and (previous.get('antithetic',False)!=antithetic or previous.get('prefix_only',False)!=prefix_only):
        raise ValueError('Existing data use a different sampling design.')
    rows = []
    for choice, correction in enumerate(corrections):
        trials = []
        for repeat in range(repeats):
            future = 2650000000+bank*1000000+root*100+(repeat//2 if antithetic else repeat)
            noise_sign = -1. if antithetic and repeat%2 else 1.
            if previous and repeat < len(previous['rows'][choice]):
                row = previous['rows'][choice][repeat]
                if 'target_response' not in row:
                    row['target_response'] = branch(snapshot,correction,horizon,future,response_only=True)
            else:
                row = branch(snapshot,correction,horizon,future,noise_sign=noise_sign,prefix_only=prefix_only)
            trials.append(row)
        rows.append(trials)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch(path, dict(root=root, bank=bank, horizon=horizon, repeats=repeats,
        packet=packet, baseline=baseline, corrections=corrections, rows=rows,
        intervention_radius=radius,antithetic=antithetic,prefix_only=prefix_only))
    return dict(root=root, seconds=round(time.perf_counter()-started, 2),
                steps=sum(r['steps'] for rs in rows for r in rs))


def collect_value_root(task):
    """Pilot-allocated, independent paired estimates of actual task outcomes."""
    root,bank,state_source,context_source,output,horizon,radius,max_groups,precision,noise_coupling=task[:10]
    source_root=task[10] if len(task)>10 else root
    include_axes=task[11] if len(task)>11 else False
    started=time.perf_counter()
    saved=torch.load(Path(state_source)/f'{source_root}.pth',weights_only=False)
    snapshot=saved['snapshot'];packet=source_context(saved)
    baseline=SOURCE.choose_action(packet,deterministic=True)[0]
    n=len(baseline)
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=n)),dtype=np.float32)
    if n>=6:signs=signs[np.prod(signs,axis=1)>0.]
    changes=np.vstack((np.zeros(n,dtype=np.float32),radius*signs))
    if include_axes:changes=np.vstack((changes,radius*np.eye(n,dtype=np.float32),-radius*np.eye(n,dtype=np.float32)))
    def group(index):
        future=3850000000+bank*1000000+root*1000+index
        return [[branch(snapshot,change,horizon,future,noise_sign=sign,intervention_link='odds',noise_coupling=noise_coupling) for sign in (1.,-1.)] for change in changes]
    # Pilot futures only determine allocation. Independent futures below
    # provide the regression targets, avoiding outcome-dependent stopping.
    pilot=[group(i) for i in range(4)]
    pilot_times=np.array([[[t['capture_time'] for t in trials] for trials in g] for g in pilot]).mean(-1)
    contrasts=pilot_times[:,1:]-pilot_times[:,:1]
    variance=np.var(contrasts,axis=0,ddof=1)
    groups=int(np.clip(np.ceil(np.quantile(variance,.75)/precision**2),4,max_groups))
    estimation=[group(100+i) for i in range(groups)]
    rows=[[t for g in estimation for t in g[choice]] for choice in range(len(changes))]
    path=Path(output)/f'{root}.pth';path.parent.mkdir(parents=True,exist_ok=True)
    atomic_torch(path,dict(root=root,bank=bank,horizon=horizon,packet=packet,baseline=baseline,
        corrections=changes,rows=rows,repeats=2*groups,intervention_coordinates='log_odds',intervention_radius=radius,
        antithetic=True,prefix_only=False,pilot_groups=4,pilot_contrast_variance=variance,
        estimation_groups=groups,allocation_precision_seconds=precision,source_policy='original',
        pilot_outcomes=pilot_times,noise_coupling=noise_coupling,source_root=source_root,axial_interventions=include_axes))
    return dict(root=root,groups=groups,seconds=round(time.perf_counter()-started,2),
        steps=sum(t['steps'] for g in pilot+estimation for trials in g for t in trials))


def collect_candidate_contrast(task):
    """Spend matched futures across independent contexts, with opposite treatments."""
    root,bank,state_source,output,horizon,repeats,radius=task
    started=time.perf_counter();raw=torch.load(Path(state_source)/f'{root}.pth',weights_only=False)
    snapshot=raw['snapshot'];packet=source_context(raw)
    baseline=SOURCE.choose_action(packet,deterministic=True)[0]
    rng=np.random.default_rng(4190000000+bank*100000+root)
    direction=rng.normal(size=len(baseline));direction=direction/np.max(np.abs(direction))
    direction=(radius*rng.uniform(.5,1.)*direction).astype(np.float32)
    corrections=np.stack((np.zeros_like(direction),direction,-direction))
    rows=[[branch(snapshot,change,horizon,4180000000+bank*100000+root*100+repeat//2,
        noise_sign=(-1. if repeat%2 else 1.),intervention_link='odds',noise_coupling='role')
        for repeat in range(repeats)] for change in corrections]
    atomic_torch(Path(output)/f'{root}.pth',dict(root=root,bank=bank,horizon=horizon,packet=packet,
        baseline=baseline,corrections=corrections,rows=rows,repeats=repeats,
        intervention_coordinates='log_odds',intervention_radius=radius,antithetic=True,
        prefix_only=False,source_policy='original',noise_coupling='role',source_root=root))
    return dict(root=root,seconds=round(time.perf_counter()-started,2),steps=sum(t['steps'] for trials in rows for t in trials))


def augment_value_root(task):
    """Axial counterfactuals identify individual curvature absent at cube corners."""
    path,state_source=task;started=time.perf_counter()
    row=torch.load(path,weights_only=False)
    if row.get('axial_interventions',False):return dict(root=row['root'],steps=0,already_complete=True)
    snapshot=torch.load(Path(state_source)/f"{row.get('source_root',row['root'])}.pth",weights_only=False)['snapshot']
    n=len(row['baseline']);axes=np.concatenate((np.eye(n),-np.eye(n)),0).astype(np.float32)*row['intervention_radius']
    added=[]
    for axis in axes:
        trials=[]
        for index in range(row['estimation_groups']):
            future=3850000000+row['bank']*1000000+row['root']*1000+100+index
            for sign in (1.,-1.):
                trials.append(branch(snapshot,axis,row['horizon'],future,noise_sign=sign,
                    intervention_link='odds',noise_coupling=row.get('noise_coupling','cartesian')))
        added.append(trials)
    row['corrections']=np.concatenate((row['corrections'],axes),0)
    row['rows'].extend(added);row['axial_interventions']=True
    atomic_torch(path,row)
    return dict(root=row['root'],seconds=round(time.perf_counter()-started,2),steps=sum(t['steps'] for trials in added for t in trials))


def source_context(saved):
    packet=public_packet(saved['snapshot'].environment)
    transitions=saved['transitions']
    index=int(round(saved['snapshot'].environment.Current_T/.2))
    if abs(float(transitions[index]['central'][-1])-float(packet['central'][-1]))>1e-5:
        raise ValueError('The temporal context must end at the exact intervention state.')
    packet['intent_history']=public_intent_history(transitions[max(0,index-7):index+1])[-1]
    return packet


def active_value_tasks(args):
    """Diversify consequential training states in the learned relation space."""
    policy=ResponsePolicy(args.source,args.graph_checkpoint)
    existing={}
    for path in (args.output/'train').glob('*.pth'):
        row=torch.load(path,weights_only=False);existing[int(path.stem)]=row.get('source_root',row['root'])
    ids,features,leverage,eligible=[],[],[],[]
    with torch.no_grad():
        for path in sorted((args.state_source/'train').glob('*.pth'),key=lambda p:int(p.stem)):
            raw=torch.load(path,weights_only=False);packet=source_context(raw)
            base=policy.source.choose_action(packet,deterministic=True)[0]
            p={k:torch.tensor(packet[k],dtype=torch.float32)[None] for k in ('obs','motion','central','intent_history')}
            node,pair=conditional_value_features(policy.model,p,torch.tensor(base)[None])
            ordered=node[0,:,4].argsort(descending=True)
            feature=torch.cat((node[0,ordered,:14].flatten(),pair[0,:,:,:7].mean((0,1)),pair[0,:,:,:7].amax((0,1))))
            ids.append(int(path.stem));features.append(feature.numpy())
            effect=4.*base[:,2]*(1.-base[:,2])*np.linalg.norm(base[:,:2]-packet['obs'][:,12:14],axis=-1)
            leverage.append(float(effect.max()));eligible.append(float(packet['obs'][:,2].min())>8.)
    x=np.asarray(features);x=((x-x.mean(0))/np.maximum(x.std(0),.05)).clip(-5.,5.)
    used=set(existing.values());chosen=[i for i,k in enumerate(ids) if k in used]
    distance=np.min(((x[:,None]-x[chosen][None])**2).mean(-1),axis=1) if chosen else np.ones(len(x))
    leverage=np.asarray(leverage);importance=.25+.75*np.clip(leverage/max(np.quantile(leverage,.75),.01),0.,1.)
    tasks=[];next_root=max(existing,default=-1)+1
    while len(existing)+len(tasks)<args.train_states:
        score=distance*importance
        score=np.where(np.asarray(eligible)&np.asarray([k not in used for k in ids]),score,-np.inf)
        index=int(score.argmax())
        if not np.isfinite(score[index]):break
        source_root=ids[index];used.add(source_root)
        tasks.append((next_root,0,args.state_source/'train',args.context_source/'train',args.output/'train',
            args.horizon,args.radius,args.max_value_groups,args.value_precision,args.noise_coupling,source_root,True))
        next_root+=1;distance=np.minimum(distance,((x-x[index])**2).mean(-1))
    print(dict(active_states=len(tasks),source_roots=[t[10] for t in tasks]),flush=True)
    return tasks


def public_intent_history(packets,length=8):
    """Causal observations only, expressed in the current target-radial frame."""
    obs = np.asarray([p['obs'] for p in packets])
    motion = np.asarray([p['motion'] for p in packets])
    bearing = motion[:,0,2]+obs[:,0,3]
    position = motion[:,0,:2]+obs[:,0,2,None]*np.stack((np.cos(bearing),np.sin(bearing)),-1)
    angle = motion[:,0,2]+obs[:,0,5]
    velocity = obs[:,0,4,None]*np.stack((np.cos(angle),np.sin(angle)),-1)
    force = -.1*position
    limit = np.full(len(position),50.)
    for index in range(motion.shape[1]):
        relative = position-motion[:,index,:2]
        norm = np.linalg.norm(relative,axis=-1).clip(1e-6)
        distance = (norm-8.).clip(.1)
        active = distance<limit
        magnitude = 3000.*(1./distance-1./limit)/distance**2
        force += np.where(active[:,None],magnitude[:,None]*relative/norm[:,None],0.)
        limit = np.where(active,distance,limit)
    force /= np.linalg.norm(force,axis=-1,keepdims=True).clip(1e-6)
    raw = np.arange(len(packets))[:,None]-np.arange(length-1,-1,-1)[None]
    indices = raw.clip(0)
    phi = np.arctan2(position[:,1],position[:,0])
    c,s = np.cos(phi)[:,None],np.sin(phi)[:,None]
    def rotate(v): return np.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1]),-1)
    return np.concatenate((rotate(velocity[indices])/5.,rotate(position[indices]-position[:,None])/5.,
        rotate(force[indices]),np.linalg.norm(position[indices],axis=-1)[...,None]/60.,
        (raw>=0)[...,None]),-1).astype(np.float32)


class TemporalIntent(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent = nn.GRU(8,32,batch_first=True)
        self.output = nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.Linear(32,5))

    def forward(self,history):
        _,state = self.recurrent(history)
        value = self.output(state[0])
        return torch.cat((F.normalize(value[:,:2],dim=-1,eps=1e-6),value[:,2:]),-1)


class RefinedTemporalIntent(TemporalIntent):
    """Refine hidden turning state from a finite causal observation window."""
    def __init__(self):
        super().__init__()
        self.kinematic_input=nn.Sequential(nn.Linear(16,64),nn.SiLU(),nn.Linear(64,64))
        layer=nn.TransformerEncoderLayer(64,4,128,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.kinematic_attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.kinematic_output=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,5))
        nn.init.zeros_(self.kinematic_output[-1].weight);nn.init.zeros_(self.kinematic_output[-1].bias)

    def forward(self,history):
        with torch.no_grad():reference=super().forward(history)
        valid=history[...,7]>0.
        difference=torch.zeros_like(history[...,:7])
        difference[:,1:]=(history[:,1:,:7]-history[:,:-1,:7])/.2
        difference[:,1:]*=(valid[:,1:]&valid[:,:-1])[...,None]
        age=torch.linspace(-1.,0.,history.shape[1],device=history.device,dtype=history.dtype)[None,:,None].expand(len(history),-1,-1)
        token=self.kinematic_input(torch.cat((history,difference,age),-1))
        token=self.kinematic_attention(token,src_key_padding_mask=~valid)
        correction=self.kinematic_output(token[:,-1])
        direction=F.normalize(reference[:,:2]+.3*correction[:,:2],dim=-1,eps=1e-6)
        yaw=reference[:,2:3]+.5*correction[:,2:3]
        yaw=torch.where((valid.sum(-1)>1)[:,None],yaw,torch.zeros_like(yaw))
        # Ground velocity is a measured anchor. The learned residual estimates
        # water velocity and the final integration substep, rather than asking
        # an unconstrained latent decoder to reconstruct the whole velocity.
        water=history[:,-1,:2]+.1*correction[:,3:5]
        return torch.cat((direction,yaw,water),-1)


class CaptureResponsibility(nn.Module):
    """Competing receiver/time outcomes, fitted from complete team episodes."""
    def __init__(self,inputs=86):
        super().__init__()
        self.head=nn.Sequential(nn.LayerNorm(inputs),nn.Linear(inputs,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,8))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('time_centers',torch.tensor([.5,1.5,3.,6.,12.,24.,45.]))
    def logits(self,features,prior):
        residual=self.head(features)
        event=prior[...,None]+residual[...,:7]-np.log(7.)
        failure=torch.logsumexp(prior,-1)-np.log(20.)+residual[...,7].mean(-1)
        return event,failure
    def forward(self,features,prior):
        event,_=self.logits(features,prior)
        return torch.logsumexp(event-self.time_centers/8.,-1)


class CandidateAttentionBlock(nn.Module):
    def __init__(self,width=64,heads=4):
        super().__init__();self.heads=heads;self.width=width
        self.norm=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.output=nn.Linear(width,width)
        self.feed=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width))
    def forward(self,token,bias):
        b,n,d=token.shape
        q,k,v=self.qkv(self.norm(token)).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        attention=(q@k.transpose(-1,-2)/(d//self.heads)**.5+bias).softmax(-1)
        token=token+self.output((attention@v).transpose(1,2).reshape(b,n,d))
        return token+self.feed(token)


class CandidateInterventionModel(nn.Module):
    """Candidate-pair attention, coupled fusion and anchored team consequences."""
    def __init__(self,width=64,competing_risks=False,degree_feature=False,modal_fusion=False,fusion_link='odds',candidate_projection=False,response_projection=False,intent_refinement=False):
        super().__init__();self.width=width;self.intent=RefinedTemporalIntent() if intent_refinement else TemporalIntent();self.temporal=True
        self.intent_refinement=intent_refinement
        self.modal_fusion=modal_fusion;self.fusion_link=fusion_link
        self.candidate_projection=candidate_projection;self.response_projection=response_projection
        self.token=nn.Sequential(nn.Linear(24,width),nn.SiLU(),nn.Linear(width,width))
        self.pair=nn.Sequential(nn.Linear(13,width),nn.SiLU(),nn.Linear(width,4))
        self.blocks=nn.ModuleList([CandidateAttentionBlock(width) for _ in range(2)])
        if candidate_projection:
            self.projection_node=nn.Sequential(nn.Linear(36,width),nn.SiLU(),nn.Linear(width,width))
            self.projection_pair=nn.Sequential(nn.Linear(27,width),nn.SiLU(),nn.Linear(width,4))
            for layer in (self.projection_node[-1],self.projection_pair[-1]):
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.degree_feature=degree_feature
        if degree_feature:
            self.degree=nn.Linear(1,width,bias=False);nn.init.zeros_(self.degree.weight)
        if modal_fusion:
            self.mode_condition=nn.Linear(2,width,bias=False);nn.init.zeros_(self.mode_condition.weight)
        self.response=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,11))
        self.value_node=nn.Sequential(nn.Linear(width+13,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.value_attention=nn.Linear(width,1)
        self.values=nn.ModuleList([nn.Sequential(nn.Linear(3*width,96),nn.SiLU(),nn.Linear(96,1)) for _ in range(3)])
        self.competing_risks=competing_risks
        if competing_risks:
            self.register_buffer('capture_times',torch.tensor([0.,1.,2.,3.,4.,6.,8.,10.,12.,16.,20.,24.,32.,40.,50.,60.]))
            self.capture_heads=nn.ModuleList([nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,16)) for _ in range(3)])
            self.failure_heads=nn.ModuleList([nn.Sequential(nn.Linear(3*width,64),nn.SiLU(),nn.Linear(64,1)) for _ in range(3)])
        self.adapter=nn.Sequential(nn.LayerNorm(3*width+1),nn.Linear(3*width+1,width),nn.SiLU(),nn.Linear(width,2 if modal_fusion else 1))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)

    def project_candidates(self,packet,baseline,eta=None):
        """Public nominal candidate motion, not a task rollout or controller.

        Individual vessels retain each candidate thrust for 0.4, 1, and 2 s.
        Measured velocity initializes the mean-noise hydrodynamics; the target
        uses a public constant-turn forecast, or a nominal adversary response
        to the proposed joint fusion. Exact safety and stochastic task outcomes
        are evaluated in the separate complete-consequence learning process.
        """
        original_inputs=(packet['obs'],packet['motion'],packet['intent_history'],baseline)
        inputs=original_inputs+((eta,) if self.response_projection and eta is not None else ())
        versions=tuple(x._version for x in inputs)
        cache=getattr(self,'_projection_cache',None)
        reusable=not any(x.requires_grad for x in inputs)
        if reusable and cache is not None and versions==cache[1] and all(a is b for a,b in zip(inputs,cache[0])):
            return cache[2],cache[3]
        obs,m=inputs[:2];b,n=m.shape[:2]
        candidates=torch.stack((baseline[...,:2],obs[...,12:14]),2)
        motion_versions=tuple(x._version for x in original_inputs)
        motion_cache=getattr(self,'_candidate_motion_cache',None)
        motion_reusable=not any(x.requires_grad for x in original_inputs)
        if motion_reusable and motion_cache is not None and motion_versions==motion_cache[1] and all(a is b for a,b in zip(original_inputs,motion_cache[0])):
            future=motion_cache[2]
        else:
            state=m[:,:,None].expand(-1,-1,2,-1)
            thrust=750.*candidates+250.;current=torch.zeros_like(state[...,3:6]);snapshots=[]
            for step in range(1,11):
                state,_=DifferentiableInterventionDynamics.vessel(state,thrust,current)
                if step in (2,5,10):snapshots.append(state)
            future=torch.stack(snapshots,-2)
            if motion_reusable:self._candidate_motion_cache=(original_inputs,motion_versions,future)
        bearing=m[:,0,2]+obs[:,0,3]
        position=m[:,0,:2]+obs[:,0,2,None]*torch.stack((bearing.cos(),bearing.sin()),-1)
        velocity_angle=m[:,0,2]+obs[:,0,5]
        velocity=obs[:,0,4,None]*torch.stack((velocity_angle.cos(),velocity_angle.sin()),-1)
        with torch.no_grad():intent=self.intent(packet['intent_history'])
        yaw=intent[:,2]
        times=m.new_tensor([.4,1.,2.]);angle=yaw[:,None]*times
        def turn(v,a):return torch.stack((a.cos()*v[...,0]-a.sin()*v[...,1],a.sin()*v[...,0]+a.cos()*v[...,1]),-1)
        target_position=position[:,None]+times[None,:,None]*torch.sinc(angle[...,None]/(2.*torch.pi))*turn(velocity[:,None],.5*angle)
        target_velocity=turn(velocity[:,None],angle)
        if self.response_projection:
            # A public belief initializes a nominal response to the proposed
            # joint fusion. The population-mean agility is a prior, not the
            # private episode agility. No future labels or private state enter
            # this actor representation, and no task-value search is performed.
            phi=torch.atan2(position[:,1],position[:,0]);cp,sp=phi.cos(),phi.sin()
            water=5.*torch.stack((cp*intent[:,3]-sp*intent[:,4],sp*intent[:,3]+cp*intent[:,4]),-1)
            heading=phi+torch.atan2(intent[:,0],intent[:,1])
            target=torch.cat((position,heading[:,None],water,yaw[:,None]),-1)
            reference=torch.cat((m,target[:,None]),1)
            if eta is None:eta=torch.zeros_like(baseline[...,2])
            theta=self.fusion_weights(baseline[...,2],eta)
            if theta.ndim==3:nominal=self.modal_nominal(baseline[...,:2],obs[...,12:14],theta)
            else:nominal=theta[...,None]*baseline[...,:2]+(1.-theta[...,None])*obs[...,12:14]
            defender_thrust=750.*nominal+250.;positions=[];velocities=[]
            for step in range(1,11):
                attacker_thrust=DifferentiableInterventionDynamics.thrust(DifferentiableInterventionDynamics.apf(reference),reference[:,-1,2],reference.new_full((b,1),2.25))
                controls=torch.cat((defender_thrust,attacker_thrust[:,None]),1)
                reference,_=DifferentiableInterventionDynamics.vessel(reference,controls,torch.zeros_like(reference[...,3:6]))
                if step in (2,5,10):positions.append(reference[:,-1,:2]);velocities.append(reference[:,-1,3:5])
            target_position=torch.stack(positions,1);target_velocity=torch.stack(velocities,1)
        c,s=m[...,2].cos()[:,:,None,None],m[...,2].sin()[:,:,None,None]
        def local(v):return torch.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1]),-1)
        relative=target_position[:,None,None]-future[...,:2]
        relative_velocity=target_velocity[:,None,None]-future[...,3:5]
        changed_angle=future[...,2]-m[:,:,None,None,2]
        node=torch.cat((local(future[...,:2]-m[:,:,None,None,:2])/5.,
            local(future[...,3:5]-m[:,:,None,None,3:5])/5.,changed_angle.sin()[...,None],
            (changed_angle.cos()-1.)[...,None],(future[...,5]-m[:,:,None,None,5])[...,None],
            local(relative)/15.,local(relative_velocity)/5.,((relative.norm(dim=-1)-5.)/5.).tanh()[...,None]),-1).flatten(-2)
        projected=future.flatten(1,2);unit=F.normalize(relative.flatten(1,2),dim=-1,eps=1e-6)
        delta=projected[:,None]-projected[:,:,None]
        heading=m[...,2].repeat_interleave(2,1);ec,es=heading.cos()[:,:,None,None],heading.sin()[:,:,None,None]
        def edge_local(v):return torch.stack((ec*v[...,0]+es*v[...,1],-es*v[...,0]+ec*v[...,1]),-1)
        coverage=(unit[:,:,None]*unit[:,None]).sum(-1)
        handedness=unit[:,:,None,:,0]*unit[:,None,:,:,1]-unit[:,:,None,:,1]*unit[:,None,:,:,0]
        pair=torch.cat((edge_local(delta[...,:2])/15.,delta[...,2:3].sin(),delta[...,2:3].cos(),
            edge_local(delta[...,3:5])/5.,delta[...,5:6],coverage[...,None],handedness[...,None]),-1).flatten(-2)
        node=node.flatten(1,2)
        if reusable:self._projection_cache=(inputs,versions,node,pair)
        return node,pair

    def encode(self,packet,baseline,eta,projection=None):
        obs,m=packet['obs'],packet['motion'];b,n=baseline.shape[:2]
        c,s=m[...,2].cos(),m[...,2].sin()
        physical=torch.stack((obs[...,0]/60.,obs[...,1].sin(),obs[...,1].cos(),obs[...,2]/60.,
            obs[...,3].sin(),obs[...,3].cos(),obs[...,4]/5.,obs[...,5].sin(),obs[...,5].cos(),
            (c*m[...,3]+s*m[...,4])/5.,(-s*m[...,3]+c*m[...,4])/5.,m[...,5]),-1)
        with torch.no_grad():intent=self.intent(packet['intent_history'])
        bearing=m[...,2]+obs[...,3];attacker=(m[...,:2]+obs[...,2,None]*torch.stack((bearing.cos(),bearing.sin()),-1)).mean(1)
        phi=torch.atan2(attacker[:,1],attacker[:,0]);angle=m[...,2]-phi[:,None];ci,si=angle.cos(),angle.sin()
        local=torch.stack((intent[:,0,None]*ci-intent[:,1,None]*si,intent[:,1,None]*ci+intent[:,0,None]*si,
            intent[:,2,None].expand_as(ci),ci*intent[:,3,None]+si*intent[:,4,None],-si*intent[:,3,None]+ci*intent[:,4,None]),-1)
        prior=baseline[...,2]
        if self.modal_fusion:
            if eta.ndim==2:eta=eta[...,None].expand(-1,-1,2)
            mode_theta=self.fusion_weights(prior,eta)
            theta=mode_theta.mean(-1)
        else:
            theta=self.fusion_weights(prior,eta)
        probability=torch.stack((theta,1.-theta),-1);reference=torch.stack((prior,1.-prior),-1)
        candidate=torch.stack((baseline[...,:2],obs[...,12:14]),2)
        expand=lambda x:x[:,:,None].expand(-1,-1,2,-1)
        kind=torch.tensor([1.,-1.],device=obs.device)[None,None,:,None].expand(b,n,-1,-1)
        clock=packet['central'][:,-1,None,None,None].expand(-1,n,2,1)
        inputs=torch.cat((expand(physical),expand(local),candidate,reference[...,None],probability[...,None],
            kind,expand(prior[...,None]),clock),-1)
        token=self.token(inputs.flatten(1,2))
        if self.candidate_projection:
            if projection is None:projection=self.project_candidates(packet,baseline,eta)
            token=token+self.projection_node(projection[0])
        if self.modal_fusion:
            mode_delta=torch.stack((mode_theta-prior[...,None],prior[...,None]-mode_theta),2)
            token=token+self.mode_condition(mode_delta.flatten(1,2))
        if self.degree_feature:
            token=token+self.degree(token.new_full((b,1,1),np.log(n/3.)))
        proxy=type('RelationInputs',(),{'peer_candidates':True})()
        geometry=InteractionActor.relation_features(proxy,m,baseline[...,:2],obs[...,12:14])[...,:7]
        geometry=geometry.repeat_interleave(2,1).repeat_interleave(2,2);candidate=candidate.flatten(1,2);count=2*n
        left=candidate[:,:,None].expand(-1,-1,count,-1);right=candidate[:,None].expand(-1,count,-1,-1)
        ids=torch.arange(count,device=obs.device);same_boat=((ids[:,None]//2)==(ids[None]//2)).float()
        same_kind=((ids[:,None]%2)==(ids[None]%2)).float()
        edges=torch.cat((geometry,left,right,same_boat[None,:,:,None].expand(b,-1,-1,-1),same_kind[None,:,:,None].expand(b,-1,-1,-1)),-1)
        bias=self.pair(edges).permute(0,3,1,2)
        if self.candidate_projection:bias=bias+self.projection_pair(projection[1]).permute(0,3,1,2)
        for block in self.blocks:token=block(token,bias)
        token=token.reshape(b,n,2,self.width)
        return token,(token*probability[...,None]).sum(2)

    def value_features(self,packet,baseline,eta,encoded=None):
        _,node=self.encode(packet,baseline,eta) if encoded is None else encoded;b,n=baseline.shape[:2]
        state=packet['central'][:,:7*n].reshape(b,n,7);att=packet['central'][:,7*n:7*(n+1)]
        heading=packet['motion'][...,2];c,s=heading.cos(),heading.sin()
        def local(v):return torch.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1]),-1)
        relative=local(att[:,None,:2]-state[...,:2]);own_velocity=local(state[...,4:6]);target_velocity=local(att[:,None,4:6].expand(-1,n,-1))
        relative_angle=torch.stack((att[:,None,2]*c-att[:,None,3]*s,att[:,None,3]*c+att[:,None,2]*s),-1)
        clock_agility=packet['central'][:,-2:,None].transpose(1,2).expand(-1,n,-1)
        order=torch.arange(1,n+1,device=node.device,dtype=node.dtype)[None,:,None].expand(b,-1,-1)/7.
        private=torch.cat((relative,own_velocity,state[...,6:7],target_velocity,relative_angle,
            att[:,None,6:7].expand(-1,n,-1),clock_agility,order),-1)
        value=self.value_node(torch.cat((node,private),-1));attention=self.value_attention(value).softmax(1)
        pooled=torch.cat((value.mean(1),value.amax(1),(value*attention).sum(1)),-1)
        return node,value,pooled

    def consequences(self,packet,baseline,eta):
        b,n=baseline.shape[:2];node,value,pooled=self.value_features(packet,baseline,eta)
        if self.competing_risks:
            logits=torch.stack([torch.cat((event(value).flatten(1),failure(pooled)),-1)
                for event,failure in zip(self.capture_heads,self.failure_heads)],1)
            probability=logits.softmax(-1)
            remaining=60.*packet['central'][:,-1]
            times=torch.minimum(self.capture_times[None,None].expand(b,n,-1),remaining[:,None,None]).flatten(1)
            times=torch.cat((times,remaining[:,None]),-1)
            q=-(probability*times[:,None]).sum(-1)/20.
            return q,self.response(node),probability
        q=-torch.cat([head(pooled) for head in self.values],-1).sigmoid()*3.*packet['central'][:,-1,None]
        return q,self.response(node),None

    def forward(self,packet,baseline,eta):
        q,response,_=self.consequences(packet,baseline,eta)
        return q,response

    def fusion(self,packet,baseline):
        eta=torch.zeros_like(baseline[...,2])
        if self.modal_fusion:eta=eta[...,None].expand(-1,-1,2)
        projection=self.project_candidates(packet,baseline) if self.candidate_projection and not self.response_projection else None
        for _ in range(3):
            token,_=self.encode(packet,baseline,eta,projection)
            features=torch.cat((token[:,:,0],token[:,:,1],token[:,:,0]-token[:,:,1],baseline[...,2:3]),-1)
            if 'fusion_readout' in packet:
                hidden=self.adapter[:3](features);readout=packet['fusion_readout']
                raw=self.adapter[3](hidden)+torch.einsum('bnh,bmh->bnm',hidden,readout[...,:-1])+readout[:,None,:,-1]
            else:raw=self.adapter(features)
            proposal=2.*raw.squeeze(-1).tanh()
            eta=.5*(eta+proposal)
        return eta

    @staticmethod
    def modal_nominal(learned,boids,theta):
        def modes(value):return torch.stack((.5*(value[...,0]+value[...,1]),.5*(value[...,0]-value[...,1])),-1)
        mixed=theta*modes(learned)+(1.-theta)*modes(boids)
        return torch.stack((mixed[...,0]+mixed[...,1],mixed[...,0]-mixed[...,1]),-1)

    def fusion_weights(self,prior,eta):
        if eta.ndim==prior.ndim+1:prior=prior[...,None]
        if self.fusion_link in ('probability','exchange'):
            # A source mixing coefficient is not a calibrated confidence.
            # Euclidean exchange remains learnable near either boundary;
            # the control metric, rather than vanishing prior odds, limits
            # the magnitude of a coordinated update.
            # Historical probability checkpoints use a quarter-width chart.
            # The exchange chart reaches both original candidate endpoints
            # from every prior; learned control trust still limits updates.
            scale=1. if self.fusion_link=='exchange' else .25
            return (prior+scale*eta).clamp(0.,1.)
        numerator=prior*eta.exp()
        return numerator/(1.-prior+numerator)

    @staticmethod
    def modal_change(learned,boids,theta,prior):
        difference=learned-boids
        modes=torch.stack((.5*(difference[...,0]+difference[...,1]),.5*(difference[...,0]-difference[...,1])),-1)
        change=(theta-prior[...,None])*modes
        return torch.stack((change[...,0]+change[...,1],change[...,0]-change[...,1]),-1)


class CooperativeFactorModel(CandidateInterventionModel):
    """Joint fusion with explicitly learned candidate-interaction potentials.

    The public policy solves one strictly convex, signed interaction problem
    for the whole team. Its edge is the mixed contrast of the four candidate
    combinations, not an attention weight. The centralized critic has separate
    unary, curvature and mixed-interaction factors, so paired outcome contrasts
    identify the terms that actually coordinate the public decision.
    """
    def __init__(self,width=64,candidate_projection=True,response_projection=True,intent_refinement=False,coordination_solver='convex'):
        super().__init__(width,True,True,True,'exchange',candidate_projection,response_projection,intent_refinement)
        self.factor_policy=True
        self.coordination_solver=coordination_solver
        def head(inputs,outputs):
            net=nn.Sequential(nn.LayerNorm(inputs),nn.Linear(inputs,width),nn.SiLU(),nn.Linear(width,outputs))
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
            return net
        self.coordination_node=head(3*width+1,2)
        self.coordination_pair=head(2*width+7,1)
        self.factor_node=head(5*width,6)
        self.factor_global=nn.Linear(3*width,width)
        self.factor_pair=head(3*width+7,3)

    @staticmethod
    def geometry(packet,baseline):
        proxy=type('RelationInputs',(),{'peer_candidates':True})()
        return InteractionActor.relation_features(proxy,packet['motion'],baseline[...,:2],packet['obs'][...,12:14])[...,:7]

    @staticmethod
    def symmetric_edges(value):
        n=value.shape[-1]
        return .5*(value+value.transpose(-1,-2))*(1.-torch.eye(n,device=value.device,dtype=value.dtype))

    def public_factors(self,packet,baseline):
        b,n=baseline.shape[:2];zero=baseline.new_zeros(b,n,2)
        token,_=self.encode(packet,baseline,zero)
        difference=token[:,:,0]-token[:,:,1]
        node=self.coordination_node(torch.cat((token[:,:,0],token[:,:,1],difference,baseline[...,2:3]),-1))
        flat=token.flatten(1,2);geometry=self.geometry(packet,baseline).repeat_interleave(2,1).repeat_interleave(2,2)
        left=flat[:,:,None].expand(-1,-1,2*n,-1);right=flat[:,None].expand(-1,2*n,-1,-1)
        inputs=(left,right,geometry)
        if getattr(self,'factor_projection',False):inputs+= (self.project_candidates(packet,baseline,zero)[1],)
        combination=self.coordination_pair(torch.cat(inputs,-1)).squeeze(-1)
        combination=combination.reshape(b,n,2,n,2).permute(0,1,3,2,4)
        mixed=combination[...,0,0]-combination[...,0,1]-combination[...,1,0]+combination[...,1,1]
        gap=.5*(baseline[...,:2]-packet['obs'][...,12:14]).norm(dim=-1)
        interaction=self.symmetric_edges(20.*mixed*gap[:,:,None]*gap[:,None,:])
        return 20.*node[...,0]*gap,interaction,node[...,1]

    @staticmethod
    def solve_coordination(prior,unary,interaction,curvature,steps=80):
        # This signed Laplacian keeps the learned mixed coefficient intact.
        # The additional .2 row norm bounds the Jacobi contraction by 5/6,
        # independent of fleet size or the magnitude/sign of the interactions.
        row_norm=interaction.abs().sum(-1)
        diagonal=4.+F.softplus(curvature)+1.2*row_norm
        displacement=torch.zeros_like(prior)
        for _ in range(steps):
            message=torch.einsum('bij,bj->bi',interaction,displacement)
            displacement=torch.maximum(torch.minimum(-(unary+message)/diagonal,1.-prior),-prior)
        return prior+displacement

    def joint_weights(self,packet,baseline):
        solver=self.solve_signed_coordination if self.coordination_solver=='signed' else self.solve_coordination
        return solver(baseline[...,2],*self.public_factors(packet,baseline))

    def enable_execution_residual(self,base_scale=.5):
        self.execution_base_scale=base_scale
        self.execution_edges=nn.Sequential(nn.Linear(23,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
        self.execution_gate=nn.Sequential(nn.Linear(28,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.execution_gate[-1].weight);nn.init.zeros_(self.execution_gate[-1].bias)

    def adapted_weights(self,packet,baseline):
        reference=self.joint_weights(packet,baseline)
        if not hasattr(self,'execution_gate'):return reference
        reference=baseline[...,2]+self.execution_base_scale*(reference-baseline[...,2])
        with torch.no_grad():
            obs,motion=packet['obs'],packet['motion'];b,n=baseline.shape[:2]
            nominal=750.*(reference[...,None]*baseline[...,:2]+(1.-reference[...,None])*obs[...,12:14])+250.
            actual=SafetyDerivative.apply(motion,nominal)
            suppression=(actual-nominal)/1500.;applied=(actual-250.)/750.
            proxy=type('RelationInputs',(),{'peer_candidates':True})()
            relations=InteractionActor.relation_features(proxy,motion,baseline[...,:2],obs[...,12:14])
            left=lambda x:x[:,:,None].expand(-1,-1,n,-1)
            right=lambda x:x[:,None].expand(-1,n,-1,-1)
            edges=torch.cat((relations,left(suppression),right(suppression),right(applied),
                left(reference[...,None]),right(reference[...,None])),-1)
            distance=(motion[:,:,None,:2]-motion[:,None,:,:2]).norm(dim=-1)
            mask=1.-torch.eye(n,device=baseline.device)[None]
            proximity=(-distance/15.).exp()*mask
            proximity=proximity/proximity.sum(-1,keepdim=True).clamp_min(1e-6)
            activity=(10.*(suppression.norm(dim=-1)+
                torch.einsum('bij,bj->bi',proximity,suppression.norm(dim=-1)))).clamp(0.,1.)
            task=torch.stack((obs[...,2]/30.,obs[...,3].cos(),obs[...,3].sin()),-1)
            node=torch.cat((baseline,obs[...,12:14],applied,suppression,task),-1)
        message=(proximity[...,None]*self.execution_edges(edges)).sum(2)
        residual=.25*self.execution_gate(torch.cat((node,message),-1)).squeeze(-1).tanh()*activity
        return (reference+residual).clamp(0.,1.)

    @staticmethod
    def solve_signed_coordination(prior,unary,interaction,curvature,steps=48):
        """Preserve nonconvex cooperative benefits during joint inference.

        Making the Hessian diagonally dominant can remove a benefit available
        only when two otherwise unhelpful changes occur together. Here signed
        edges remain intact. Projected energy descent starts at the reference
        and the candidate-combination corners; it performs no environment or
        private-critic queries. A common spectral-norm upper bound guarantees
        descent at each start. Selecting the lowest energy is differentiable
        almost everywhere through the selected continuous solution.
        """
        result_dtype=prior.dtype
        prior,unary,interaction,curvature=(value.double() for value in (prior,unary,interaction,curvature))
        b,n=prior.shape;diagonal=4.+F.softplus(curvature)
        if n<=8:
            ids=torch.arange(2**n,device=prior.device)
            corners=((ids[:,None]>>torch.arange(n,device=prior.device))&1).to(prior.dtype)
            starts=corners[None].expand(b,-1,-1)
        else:
            eye=torch.eye(n,device=prior.device,dtype=prior.dtype)
            starts=torch.cat((torch.zeros(1,n,device=prior.device),torch.ones(1,n,device=prior.device),eye,1.-eye),0)[None].expand(b,-1,-1)
        displacement=torch.cat((prior[:,None],starts),1)-prior[:,None]
        lipschitz=(diagonal+interaction.abs().sum(-1)).amax(-1)[:,None,None]
        for _ in range(steps):
            gradient=unary[:,None]+diagonal[:,None]*displacement+torch.einsum('bij,bkj->bki',interaction,displacement)
            displacement=(prior[:,None]+displacement-gradient/lipschitz).clamp(0.,1.)-prior[:,None]
        energy=(unary[:,None]*displacement+.5*diagonal[:,None]*displacement.square()).sum(-1)+\
            .5*torch.einsum('bij,bki,bkj->bk',interaction,displacement,displacement)
        chosen=energy.argmin(-1)
        return (prior+displacement[torch.arange(b,device=prior.device),chosen]).to(result_dtype)

    def fusion(self,packet,baseline):
        # Both motors share the same continuous source coefficient. The joint
        # decision therefore remains a convex blend of the original candidates.
        probe=getattr(self,'candidate_probe','aligned')
        if probe=='aligned':
            weights=self.adapted_weights(packet,baseline)
        else:
            # Each receiver retains its own true controls and all geometry.
            # Only the candidates it receives from other vessels are changed;
            # execution still blends the original, physically valid controls.
            # This is an inference diagnostic, not a retrained ablation.
            b,n=baseline.shape[:2]
            expand=lambda value:value[:,None].expand(-1,n,*value.shape[1:]).reshape(b*n,*value.shape[1:]).clone()
            altered={key:expand(value) for key,value in packet.items()}
            base=expand(baseline)
            for receiver in range(n):
                batch=torch.arange(b,device=baseline.device)*n+receiver
                peers=torch.tensor([j for j in range(n) if j!=receiver],device=baseline.device)
                if probe=='permuted':
                    senders=peers.roll(1)
                    base[batch[:,None],peers[None],:2]=baseline[:,senders,:2]
                    altered['obs'][batch[:,None],peers[None],12:14]=packet['obs'][:,senders,12:14]
                elif probe=='state-only':
                    base[batch[:,None],peers[None],:2]=0.
                    base[batch[:,None],peers[None],2]=.5
                    altered['obs'][batch[:,None],peers[None],12:14]=0.
                else:raise ValueError('Unknown candidate configuration probe.')
            choices=self.adapted_weights(altered,base).reshape(b,n,n)
            weights=choices.diagonal(dim1=1,dim2=2)
        return (weights-baseline[...,2])[...,None].expand(-1,-1,2)

    def centralized_factors(self,packet,baseline):
        b,n=baseline.shape[:2];zero=baseline.new_zeros(b,n,2)
        encoded=self.encode(packet,baseline,zero)
        _,value,pooled=self.value_features(packet,baseline,zero,encoded)
        contrast=encoded[0][:,:,0]-encoded[0][:,:,1]
        node_input=torch.cat((value,contrast,pooled[:,None].expand(-1,n,-1)),-1)
        node=self.factor_node(node_input).reshape(b,n,3,2).permute(0,2,1,3)
        left=value[:,:,None].expand(-1,-1,n,-1);right=value[:,None].expand(-1,n,-1,-1)
        global_feature=self.factor_global(pooled)[:,None,None].expand(-1,n,n,-1)
        pair=self.factor_pair(torch.cat((left,right,global_feature,self.geometry(packet,baseline)),-1)).permute(0,3,1,2)
        gap=.5*(baseline[...,:2]-packet['obs'][...,12:14]).norm(dim=-1)
        interaction=self.symmetric_edges(20.*pair*gap[:,None,:,None]*gap[:,None,None,:])
        unary=20.*node[...,0]*gap[:,None]
        curvature=20.*node[...,1]*gap[:,None].square()
        logits=torch.stack([torch.cat((event(value).flatten(1),failure(pooled)),-1)
            for event,failure in zip(self.capture_heads,self.failure_heads)],1)
        remaining=60.*packet['central'][:,-1]
        times=torch.minimum(self.capture_times[None,None].expand(b,n,-1),remaining[:,None,None]).flatten(1)
        times=torch.cat((times,remaining[:,None]),-1)
        reference=(logits.softmax(-1)*times[:,None]).sum(-1)
        return reference,unary,curvature,interaction

    @staticmethod
    def factor_cost(factors,weights,prior):
        reference,unary,curvature,interaction=factors
        if weights.ndim==2:weights=weights[:,None]
        change=weights-prior[:,None]
        return reference[:,None]+torch.einsum('bhn,bkn->bkh',unary,change)+\
            .5*torch.einsum('bhn,bkn->bkh',curvature,change.square())+\
            .5*torch.einsum('bhij,bki,bkj->bkh',interaction,change,change)


class CooperativeEventModel(CooperativeFactorModel):
    """Cooperative fusion effects on cause and time of the terminal event.

    Directed factors explain how receiver j changes boat i's capture hazard;
    mixed factors explain their joint effect. A normalized competing-event
    distribution turns these effects into a bounded expected team cost. This
    retains changes of capture responsibility even when their mean time effect
    cancels, and avoids imposing a quadratic shape on the final task value.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        def head(inputs,outputs):
            net=nn.Sequential(nn.LayerNorm(inputs),nn.Linear(inputs,self.width),nn.SiLU(),nn.Linear(self.width,outputs))
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
            return net
        self.event_unary=head(5*self.width,3*16)
        self.event_pair=head(5*self.width+7,3*2*16)
        self.factor_projection=self.candidate_projection
        if self.factor_projection:self.coordination_pair=head(2*self.width+7+27,1)

    def centralized_factors(self,packet,baseline):
        b,n=baseline.shape[:2];zero=baseline.new_zeros(b,n,2)
        encoded=self.encode(packet,baseline,zero)
        _,value,pooled=self.value_features(packet,baseline,zero,encoded)
        contrast=encoded[0][:,:,0]-encoded[0][:,:,1]
        node=self.event_unary(torch.cat((value,contrast,pooled[:,None].expand(-1,n,-1)),-1))
        node=node.reshape(b,n,3,16).permute(0,2,1,3)
        receiver=torch.cat((value,contrast),-1)
        left=receiver[:,:,None].expand(-1,-1,n,-1);right=receiver[:,None].expand(-1,n,-1,-1)
        context=self.factor_global(pooled)[:,None,None].expand(-1,n,n,-1)
        pair=self.event_pair(torch.cat((left,right,context,self.geometry(packet,baseline)),-1))
        pair=pair.reshape(b,n,n,3,2,16).permute(0,3,1,2,4,5)
        gap=.5*(baseline[...,:2]-packet['obs'][...,12:14]).norm(dim=-1)
        mask=(1.-torch.eye(n,device=baseline.device))[None,None,:,:,None]
        own=3.*node*gap[:,None,:,None]
        influence=3.*pair[...,0,:]*gap[:,None,None,:,None]*mask
        mixed=3.*pair[...,1,:]*gap[:,None,:,None,None]*gap[:,None,None,:,None]*mask
        logits=torch.stack([torch.cat((event(value).flatten(1),failure(pooled)),-1)
            for event,failure in zip(self.capture_heads,self.failure_heads)],1)
        remaining=60.*packet['central'][:,-1]
        times=torch.minimum(self.capture_times[None,None].expand(b,n,-1),remaining[:,None,None]).flatten(1)
        times=torch.cat((times,remaining[:,None]),-1)
        return dict(reference=logits,own=own,influence=influence,mixed=mixed,times=times)

    @staticmethod
    def factor_logits(factors,weights,prior):
        if weights.ndim==2:weights=weights[:,None]
        change=weights-prior[:,None];b,k,n=change.shape
        reference=factors['reference'];capture=reference[...,:-1].reshape(b,3,n,16)
        altered=capture[:,None]+torch.einsum('bhit,bki->bkhit',factors['own'],change)+\
            torch.einsum('bhijt,bkj->bkhit',factors['influence'],change)+\
            torch.einsum('bhijt,bki,bkj->bkhit',factors['mixed'],change,change)
        return torch.cat((altered.flatten(-2),reference[:,None,:,-1:].expand(-1,k,-1,-1)),-1)

    @staticmethod
    def factor_cost(factors,weights,prior):
        logits=CooperativeEventModel.factor_logits(factors,weights,prior)
        return (logits.softmax(-1)*factors['times'][:,None,None]).sum(-1)

    def event_loss(self,factors,weights,prior,cost,capturer,branch_limit=6):
        """Proper cause likelihood with linear time-bin interpolation."""
        logits=self.factor_logits(factors,weights,prior)[:,:branch_limit].transpose(1,2)
        log_probability=logits.log_softmax(-1);cost=cost[...,:branch_limit];capturer=capturer[...,:branch_limit]
        if cost.ndim==4:log_probability=log_probability[:,:,None].expand(-1,-1,cost.shape[2],-1,-1)
        bins=factors['times'][:,:len(self.capture_times)].contiguous()
        upper=torch.searchsorted(bins,cost.flatten(1).contiguous()).clamp(1,len(self.capture_times)-1)
        lower=upper-1;low_time=bins.gather(1,lower);high_time=bins.gather(1,upper)
        fraction=((cost.flatten(1)-low_time)/(high_time-low_time).clamp_min(1e-6)).clamp(0.,1.).reshape_as(cost)
        lower=lower.reshape_as(capturer);upper=upper.reshape_as(capturer)
        final=logits.shape[-1]-1
        lower_label=torch.where(capturer>=0,16*capturer+lower,final)
        upper_label=torch.where(capturer>=0,16*capturer+upper,final)
        left=log_probability.gather(-1,lower_label[...,None]).squeeze(-1)
        right=log_probability.gather(-1,upper_label[...,None]).squeeze(-1)
        return -((1.-fraction)*left+fraction*right).mean()


@torch.no_grad()
def grouped_factor_outcomes(model,source,packet,pairs,repeats,*,source_continuation=False,option_steps=1,proposals=None):
    """Grouped complete futures identify unary, curvature and mixed effects.

    All seven branches share environmental innovations. Their first actions
    are reference, half/full i, half/full j, full i+j, and half i+j. Each
    selected coefficient moves to its farther original candidate endpoint.
    The last branch is reserved for a quadratic-interaction consistency check.
    A branch applies its coefficient displacement for one fusion option.
    The original candidate controllers and safety filter still update at 0.2 s.
    Afterwards every branch follows the same feedback policy and option cadence.
    """
    d=DifferentiableInterventionDynamics;b,n=packet['obs'].shape[:2];k=7
    base=source(packet['obs'].flatten(0,1),True,False)[0].reshape(b,n,3)
    reference=base[...,2] if source_continuation else model.joint_weights(packet,base)
    weights=reference[:,None].expand(-1,k,-1).clone();row=torch.arange(b,device=base.device)
    i,j=pairs.unbind(-1);di=torch.where(reference[row,i]<.5,1.,0.)-reference[row,i]
    dj=torch.where(reference[row,j]<.5,1.,0.)-reference[row,j]
    for branch,amount in ((1,.5),(2,1.),(5,1.),(6,.5)):weights[row,branch,i]+=amount*di
    for branch,amount in ((3,.5),(4,1.),(5,1.),(6,.5)):weights[row,branch,j]+=amount*dj
    motor_fusion=proposals is not None and proposals.ndim==4
    if motor_fusion:weights=weights[...,None].expand(-1,-1,-1,2)
    if proposals is not None:
        weights=torch.cat((weights,proposals),1);k=weights.shape[1]
    p={key:value.repeat_interleave(repeats*k,0) for key,value in packet.items()}
    reference_prior=base[...,2,None] if motor_fusion else base[...,2]
    first=weights-reference_prior[:,None]
    first=first[:,None].expand(-1,repeats,*first.shape[1:]).flatten(0,2)
    noise=torch.randn(b*repeats,300,2+3*(n+1),device=base.device).repeat_interleave(k,0)
    state,measured,agility,remaining=d.from_packet(p);raw_history=d.raw_history(p)
    active=torch.arange(len(state),device=base.device);total=torch.zeros(len(state),device=base.device)
    capturers=torch.full((len(state),),-1,device=base.device,dtype=torch.long)
    mask=torch.triu(torch.ones(n,n,device=base.device,dtype=torch.bool),1)
    for step in range(300):
        a=source(p['obs'].flatten(0,1),True,False)[0].reshape(len(active),n,3)
        prior=a[...,2,None].expand(-1,-1,2) if motor_fusion else a[...,2]
        if step==0:program=first[active]
        elif step%option_steps==0:
            program=torch.zeros_like(prior) if source_continuation else model.joint_weights(p,a)-a[...,2]
            if motor_fusion and program.ndim==2:program=program[...,None].expand(-1,-1,2)
        theta=(prior+program).clamp(0.,1.)
        change=(theta-prior)*(a[...,:2]-p['obs'][...,12:14]) if motor_fusion else (theta-prior)[...,None]*(a[...,:2]-p['obs'][...,12:14])
        duration=torch.minimum(60.*remaining,torch.full_like(remaining,.2))
        state,measured=d.step(state,measured,agility,a,noise[active,step],750.*change)
        remaining=(remaining-.2/60.).clamp_min(0.)
        distance=(state[:,:-1,:2]-state[:,-1:,:2]).norm(dim=-1)
        separation=(state[:,:-1,None,:2]-state[:,None,:-1,:2]).norm(dim=-1)[:,mask]
        failure=(state[:,-1,:2].norm(dim=-1)<=15.)|(separation.amin(-1)<=5.)
        capture=(distance.amin(-1)<5.)&~failure
        done=failure|capture|(remaining<1e-5)
        capturers[active[capture]]=distance[capture].argmin(-1)
        total.index_add_(0,active,duration+failure*60.*remaining)
        keep=~done
        if not bool(keep.any()):break
        raw_history=d.advance_history(raw_history,state,measured)[keep]
        active=active[keep];state=state[keep];measured=measured[keep];program=program[keep]
        agility=agility[keep];remaining=remaining[keep]
        p=d.packet(state,measured,agility,remaining);p['intent_history']=d.intent_history(raw_history)
    else:raise ValueError('Factor labels require complete terminal futures.')
    return dict(packet=packet,baseline=base,weights=weights,returns=total.reshape(b,repeats,k),
        capturers=capturers.reshape(b,repeats,k),pairs=pairs)


def factor_contrasts(value):
    """The first five contrasts identify the quadratic; sixth checks it."""
    base,half_i,full_i,half_j,full_j,both,half_both=value.unbind(-1)
    return torch.stack((full_i-base,full_j-base,full_i-2.*half_i+base,
        full_j-2.*half_j+base,both-full_i-full_j+base,
        half_both-half_i-half_j+base),-1)


def fit_cooperative_factors(args):
    """Learn an explicit cooperative game, then its public joint-fusion rule."""
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    saved=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    model_type=CooperativeEventModel if args.event_factors else CooperativeFactorModel
    model=model_type(saved['width'],saved.get('candidate_projection',True),
        saved.get('response_projection',True),saved.get('intent_refinement',False),args.coordination_solver).eval()
    weights=dict(saved.get('score',{}).get('critic',saved['model'])) if not saved.get('factor_policy') else dict(saved['model'])
    if weights['adapter.3.weight'].shape[0]==1:
        weights['adapter.3.weight']=weights['adapter.3.weight'].repeat(2,1)
        weights['adapter.3.bias']=weights['adapter.3.bias'].repeat(2)
    compatible={key:value for key,value in weights.items() if key in model.state_dict() and value.shape==model.state_dict()[key].shape}
    model.load_state_dict(compatible,strict=False)
    source=OriginalPolicy(args.source).actor
    path=args.factor_data or args.output/f'factor-data-{args.seed}.pth';started=time.perf_counter()
    if path.exists():datasets=torch.load(path,weights_only=True)
    else:
        datasets=[]
        for directory,count in [(args.context_source,args.train_states)]+([(args.mixed_source,args.mixed_train_states)] if args.mixed_source else []):
            packets=[]
            for root in range(count):
                raw=torch.load(directory/'train'/f'{root}.pth',weights_only=False)
                if args.horizon==1:packets.append(source_context(raw))
                else:
                    index=int(round(raw['snapshot'].environment.Current_T/.2))//args.horizon*args.horizon
                    transitions=raw['transitions'];packet={key:transitions[index][key] for key in ('obs','motion','central')}
                    packet['intent_history']=public_intent_history(transitions[max(0,index-7):index+1])[-1]
                    packets.append(packet)
            packet={key:torch.as_tensor(np.asarray([p[key] for p in packets]),dtype=torch.float32) for key in ('obs','motion','central','intent_history')}
            n=packet['obs'].shape[1];i=packet['obs'][...,2].argmin(-1)
            j=(i+torch.as_tensor(rng.integers(1,n,size=count)))%n;pairs=torch.stack((i,j),-1)
            parts=[]
            for start in range(0,count,8):
                ix=torch.arange(start,min(start+8,count))
                parts.append(grouped_factor_outcomes(model,source,subset(packet,ix),pairs[ix],args.repeats,
                    source_continuation=not saved.get('factor_policy',False),option_steps=args.horizon))
                print(dict(stage='factor_futures',defenders=n,roots=min(start+8,count),total_roots=count,seconds=round(time.perf_counter()-started,1)),flush=True)
            data={key:torch.cat([part[key] for part in parts]) for key in ('baseline','weights','returns','capturers','pairs')}
            data['packet']=packet;data['defenders']=n;data['option_steps']=args.horizon;datasets.append(data)
        atomic_torch(path,datasets)
    if args.mode=='collect_factor_outcomes':return
    if any(data['returns'].shape[1]!=args.repeats or data.get('option_steps',1)!=args.horizon for data in datasets):
        raise ValueError('The fitted option duration and number of futures must match the collected data.')
    if args.event_factors and any('capturers' not in data for data in datasets):
        raise ValueError('Event-factor learning requires recorded terminal causes.')
    # Whole episodes, never branches, are held out. The seventh branch is
    # also excluded from the fitted contrast objective at every training root.
    train=[];valid=[]
    for data in datasets:
        cut=3*len(data['baseline'])//4
        def take(ix):return {key:subset(value,ix) if key=='packet' else value[ix] if isinstance(value,torch.Tensor) else value for key,value in data.items()}
        train.append(take(torch.arange(cut)));valid.append(take(torch.arange(cut,len(data['baseline']))))
    representation=('token.','pair.','blocks.','degree.','mode_condition.','projection_')
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('factor_','event_','capture_heads.','failure_heads.','value_node.','value_attention.')) or
            (args.fine_tune_relations and name.startswith(representation)))
    head_parameters=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith(representation)]
    representation_parameters=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith(representation)]
    groups=[dict(params=head_parameters,lr=args.value_lr)]
    if representation_parameters:groups.append(dict(params=representation_parameters,lr=.1*args.value_lr))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    scales=torch.cat([factor_contrasts(data['returns']).mean(1)[:,:5] for data in train]).square().mean(0).clamp_min(.25)
    history=[];best=None;best_score=float('inf')
    def calibration(data):
        with torch.no_grad():
            factors=model.centralized_factors(data['packet'],data['baseline'])
            prediction=model.factor_cost(factors,data['weights'],data['baseline'][...,2]).mean(-1)
            truth=factor_contrasts(data['returns']).mean(1);estimated=factor_contrasts(prediction)
            error=(estimated-truth).square().mean(0);zero=truth.square().mean(0)
            result=dict(defenders=data['defenders'],mse=error.tolist(),zero_mse=zero.tolist(),
                absolute_mse=float((prediction[:,0]-data['returns'][:,:,0].mean(1)).square().mean()),
                contrast_noise=float(factor_contrasts(data['returns']).var(1).mean()/args.repeats))
            if args.event_factors:
                cost=data['returns'][:,None].expand(-1,3,-1,-1);cause=data['capturers'][:,None].expand(-1,3,-1,-1)
                result['event_nll']=float(model.event_loss(factors,data['weights'],data['baseline'][...,2],cost,cause))
                probability=model.factor_logits(factors,data['weights'],data['baseline'][...,2]).softmax(-1).mean(2)
                n=data['defenders'];marginal=torch.cat((probability[...,:-1].reshape(len(probability),7,n,16).sum(-1),probability[...,-1:]),-1)
                labels=F.one_hot(torch.where(data['capturers']>=0,data['capturers'],n),n+1).float().mean(1)
                difference=marginal[:,1:]-marginal[:,:1];truth_difference=labels[:,1:]-labels[:,:1]
                result['responsibility_difference_mse']=float((difference-truth_difference).square().sum(-1).mean())
                result['responsibility_zero_mse']=float(truth_difference.square().sum(-1).mean())
            return result
    for update in range(1,args.updates+1):
        data=train[(update-1)%len(train)];ix=torch.as_tensor(rng.integers(len(data['baseline']),size=args.batch_size))
        packet=subset(data['packet'],ix);base=data['baseline'][ix]
        factors=model.centralized_factors(packet,base)
        predicted=model.factor_cost(factors,data['weights'][ix],base[...,2]).transpose(1,2)
        # Each head gets an independent bootstrap of matched futures, preserving
        # the within-future branch dependence used by all mixed contrasts.
        draws=torch.randint(args.repeats,(len(ix),3,args.repeats))
        outcome=data['returns'][ix];targets=outcome[torch.arange(len(ix))[:,None,None],draws].mean(2)
        difference=factor_contrasts(predicted)[...,:5]-factor_contrasts(targets)[...,:5]
        if args.event_factors:
            outcome_index=draws[:,:,0]
            sampled_cost=outcome[torch.arange(len(ix))[:,None],outcome_index]
            sampled_capturer=data['capturers'][ix][torch.arange(len(ix))[:,None],outcome_index]
            loss=model.event_loss(factors,data['weights'][ix],base[...,2],sampled_cost,sampled_capturer)
            loss+=.5*(difference[...,[0,1,4]].square()/scales[[0,1,4]]).mean()
        else:loss=(difference.square()/scales).mean()+.05*((predicted[...,0]-targets[...,0])/20.).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if update%100==0 or update==args.updates:
            metrics=[calibration(data) for data in valid]
            if args.event_factors:score=float(np.mean([np.mean(np.array(row['mse'])[[0,1,4]]/scales.numpy()[[0,1,4]])+.2*row['event_nll'] for row in metrics]))
            else:score=float(np.mean([np.mean(np.array(row['mse'])[:5]/scales.numpy()) for row in metrics]))
            row=dict(stage='factor_identification',update=update,score=score,metrics=metrics);history.append(row);print(row,flush=True)
            if score<best_score:best_score=score;best={key:value.detach().clone() for key,value in model.state_dict().items()}
    model.load_state_dict(best)
    if args.event_factors:
        for data in train+valid:
            with torch.no_grad():
                factors=model.centralized_factors(data['packet'],data['baseline'])
                predictions=model.factor_cost(factors,data['weights'],data['baseline'][...,2]).mean(-1)
                data['relation_target']=factor_contrasts(predictions)[:,4]
    # Public edges are trained from the SAME measured mixed consequences as
    # the centralized factors. They cannot change merely to game actor loss.
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('coordination_pair.'))
    edge_optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.value_lr,weight_decay=.02)
    edge_best=None;edge_score=float('inf')
    for update in range(1,args.updates+1):
        data=train[(update-1)%len(train)];ix=torch.as_tensor(rng.integers(len(data['baseline']),size=args.batch_size));base=data['baseline'][ix]
        _,interaction,_=model.public_factors(subset(data['packet'],ix),base)
        pairs=data['pairs'][ix];row=torch.arange(len(ix));i,j=pairs.unbind(-1)
        change=data['weights'][ix,5]-data['weights'][ix,0]
        prediction=interaction[row,i,j]*change[row,i]*change[row,j]
        truth=data['relation_target'][ix] if args.event_factors else factor_contrasts(data['returns'][ix]).mean(1)[:,4]
        loss=(prediction-truth).square().mean()/scales[4]
        edge_optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);edge_optimizer.step()
        if update%100==0 or update==args.updates:
            metrics=[]
            with torch.no_grad():
                for data in valid:
                    _,interaction,_=model.public_factors(data['packet'],data['baseline'])
                    row=torch.arange(len(data['baseline']));i,j=data['pairs'].unbind(-1)
                    change=data['weights'][:,5]-data['weights'][:,0]
                    prediction=interaction[row,i,j]*change[row,i]*change[row,j]
                    truth=factor_contrasts(data['returns']).mean(1)[:,4]
                    metrics.append(dict(defenders=data['defenders'],mixed_mse=float((prediction-truth).square().mean()),zero_mse=float(truth.square().mean())))
            score=float(np.mean([r['mixed_mse'] for r in metrics]));print(dict(stage='public_relations',update=update,metrics=metrics),flush=True)
            if score<edge_score:edge_score=score;edge_best={key:value.detach().clone() for key,value in model.coordination_pair.state_dict().items()}
    model.coordination_pair.load_state_dict(edge_best)
    # The value and relation heads now remain fixed while a public unary field
    # learns the joint policy through the differentiable coordination solve.
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('coordination_node.'))
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.pathwise_lr)
    for data in train+valid:
        with torch.no_grad():
            factors=model.centralized_factors(data['packet'],data['baseline'])
            data['factors']={key:value.detach() for key,value in factors.items()} if isinstance(factors,dict) else tuple(x.detach() for x in factors)
    for update in range(1,args.actor_updates+1):
        data=train[(update-1)%len(train)];ix=torch.as_tensor(rng.integers(len(data['baseline']),size=args.batch_size));base=data['baseline'][ix]
        packet=subset(data['packet'],ix);theta=model.joint_weights(packet,base)
        factors=subset(data['factors'],ix) if isinstance(data['factors'],dict) else tuple(x[ix] for x in data['factors'])
        value=model.factor_cost(factors,theta,base[...,2])[:,0]
        reference=model.factor_cost(factors,base[...,2],base[...,2])[:,0]
        change=(theta-base[...,2])[...,None]*(base[...,:2]-packet['obs'][...,12:14])
        objective=(value-reference).mean(-1)+args.trust*change.square().mean((1,2))
        loss=objective.mean();optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if update%100==0 or update==args.actor_updates:
            with torch.no_grad():
                metrics=[]
                for data in valid:
                    base=data['baseline'];theta=model.joint_weights(data['packet'],base)
                    delta=model.factor_cost(data['factors'],theta,base[...,2])-model.factor_cost(data['factors'],base[...,2],base[...,2])
                    metrics.append(dict(defenders=data['defenders'],predicted_change=float(delta.mean()),coefficient_change=float((theta-base[...,2]).abs().mean())))
            print(dict(stage='joint_policy',update=update,metrics=metrics),flush=True)
    output=dict(model=model.state_dict(),width=model.width,horizon=args.horizon,seed=args.seed,candidate_attention=True,
        factor_policy=True,event_factors=args.event_factors,coordination_solver=model.coordination_solver,competing_risks=True,degree_feature=True,modal_fusion=True,fusion_link='exchange',
        candidate_projection=model.candidate_projection,response_projection=model.response_projection,intent_refinement=model.intent_refinement,
        factors=dict(history=history,training_data=str(path),repeats=args.repeats,trust=args.trust,whole_root_validation=True,fine_tune_relations=args.fine_tune_relations,
            continuation=str(args.graph_checkpoint) if saved.get('factor_policy') else 'original ARBoids',
            branch_order=['reference','half_i','full_i','half_j','full_j','full_pair','half_pair']),
        source=str(args.source),readout=None)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',output)
    print(dict(stage='factor_complete',seconds=round(time.perf_counter()-started,1)),flush=True)


class ResponseGraph(nn.Module):
    """Candidate-conditioned directed edges, nonlinear aggregation, and values."""
    def __init__(self, width=64, heads=3, task_response=False, distributional=False, mediated=False, physical_mediator=False, temporal=False, cooperation=False,responsibility=False):
        super().__init__()
        self.width, self.heads = width, heads
        self.task_response, self.distributional = task_response,distributional
        self.mediated = mediated
        self.physical_mediator = physical_mediator
        self.temporal = temporal
        self.cooperation = cooperation
        if responsibility:self.responsibility=CaptureResponsibility(width+22)
        if cooperation:
            self.cooperation_nodes = nn.Sequential(nn.Linear(width+17,64),nn.SiLU(),nn.Linear(64,3 if cooperation=='coverage' else 2))
            self.cooperation_edges = nn.Sequential(nn.Linear(4*width,64),nn.SiLU(),nn.Linear(64,2 if cooperation=='coverage' else 1))
        if temporal:
            self.intent = TemporalIntent()
            self.intent_context = nn.Linear(5,width,bias=False)
            nn.init.zeros_(self.intent_context.weight)
            if physical_mediator:
                self.intent_force = nn.Linear(5,32,bias=False)
                nn.init.zeros_(self.intent_force.weight)
        self.node = nn.Sequential(nn.Linear(17, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.edge = nn.Sequential(nn.Linear(15, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.attention = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, 1))
        self.context = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.condition = nn.Sequential(nn.Linear(width+3, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.condition_edge = nn.Sequential(nn.Linear(3*width+6, width), nn.SiLU(),
                                            nn.Linear(width, width), nn.SiLU())
        self.response_node = nn.Linear(width, 7)
        self.response_pair = nn.Linear(width, 7)
        if task_response:
            self.action_context = nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
            self.response_target = nn.Linear(width,4)
        if mediated:
            self.attacker_node = nn.Sequential(nn.Linear(width+15,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
            self.attacker_attention = nn.Linear(64,1)
            self.attacker_response = nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,4))
            if physical_mediator:
                self.attacker_force_response = nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,4))
        self.value_node = nn.Sequential(nn.Linear(width+8, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        # Centralized execution state is only available to the value head.
        self.value = nn.ModuleList([nn.Sequential(nn.Linear(2*width+23, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 61 if distributional else 1)) for _ in range(heads)])
        self.gate = (nn.Sequential(nn.LayerNorm(89),nn.Linear(89,width),nn.SiLU(),nn.Linear(width,1))
            if task_response else nn.Sequential(nn.Linear(2*width+1, width), nn.SiLU(), nn.Linear(width, 1)))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def node_inputs(self, packet, baseline):
        obs, motion = packet['obs'], packet['motion']
        c, s = motion[..., 2].cos(), motion[..., 2].sin()
        local_velocity = torch.stack((c*motion[..., 3]+s*motion[..., 4],
            -s*motion[..., 3]+c*motion[..., 4], 5.*motion[..., 5]), -1)/5.
        task = torch.stack((obs[..., 0]/60., obs[..., 1].sin(), obs[..., 1].cos(),
            obs[..., 2]/60., obs[..., 3].sin(), obs[..., 3].cos(),
            obs[..., 4]/5., obs[..., 5].sin(), obs[..., 5].cos()), -1)
        candidates = torch.cat((baseline[..., :2], obs[..., 12:14]), -1)
        return torch.cat((task, local_velocity, candidates, baseline[..., 2:3]), -1)

    def encode(self, packet, baseline, peer=True):
        obs, motion = packet['obs'], packet['motion']
        b, n = baseline.shape[:2]
        node = self.node(self.node_inputs(packet,baseline))
        # Identical public relative coordinates to the existing mechanism.
        proxy = type('RelationInputs', (), {'peer_candidates': peer})()
        features = InteractionActor.relation_features(proxy, motion, baseline[..., :2], obs[..., 12:14])
        edges = self.edge(features)
        mask = ~torch.eye(n, dtype=torch.bool, device=obs.device)[None]
        weights = self.attention(torch.cat((node[:, :, None].expand(-1, -1, n, -1), edges), -1)).squeeze(-1)
        weights = weights.masked_fill(~mask, -torch.inf).softmax(-1)
        message = (weights[..., None]*edges).sum(-2)
        context = self.context(torch.cat((node, message), -1))
        if self.temporal:
            intent = self.intent(packet['intent_history'])
            bearing = motion[...,2]+obs[...,3]
            attacker = (motion[...,:2]+obs[...,2,None]*torch.stack((bearing.cos(),bearing.sin()),-1)).mean(1)
            phi = torch.atan2(attacker[:,1],attacker[:,0])
            angle = motion[...,2]-phi[:,None]
            c,s = angle.cos(),angle.sin()
            sine,cosine = intent[:,0,None],intent[:,1,None]
            local = torch.stack((sine*c-cosine*s,cosine*c+sine*s,intent[:,2,None].expand_as(c),
                c*intent[:,3,None]+s*intent[:,4,None],-s*intent[:,3,None]+c*intent[:,4,None]),-1)
            context = context+self.intent_context(local)
        return context, edges, mask

    def readout_features(self, packet, baseline, mode='latent'):
        node, edge, mask = self.encode(packet,baseline)
        if mode == 'latent':
            return torch.cat((node,node.mean(1,keepdim=True).expand_as(node),baseline[...,2:3]),-1)
        if mode in ('projected','projected_future'):
            if not self.task_response: raise ValueError('Task-projected relations require the target response model.')
            n = node.shape[1]
            influences = []
            if mode == 'projected_future':
                corrections = []
                for sender in range(n):
                    for gate in (0.,1.):
                        correction = torch.zeros_like(baseline[...,2]);correction[:,sender] = gate-baseline[:,sender,2]
                        corrections.append(correction)
                corrections.append(torch.zeros_like(baseline[...,2]))
                choices = len(corrections)
                expand = lambda v:v[:,None].expand(-1,choices,*v.shape[1:]).flatten(0,1)
                expanded_packet = {k:expand(v) for k,v in packet.items()}
                result = self.condition_on((expand(node),expand(edge),mask),expanded_packet,expand(baseline),
                    torch.stack(corrections,1).flatten(0,1),details=True)[4].reshape(len(node),choices,n,4)
                influence = (result[:,:2*n].reshape(len(node),n,2,n,4)[:,:,1]
                             -result[:,:2*n].reshape(len(node),n,2,n,4)[:,:,0])
                reference_response = result[:,-1]
            else:
                for sender in range(n):
                    responses = []
                    for gate in (0.,1.):
                        correction = torch.zeros_like(baseline[...,2]);correction[:,sender] = gate-baseline[:,sender,2]
                        responses.append(self.condition_on((node,edge,mask),packet,baseline,correction,details=True)[4])
                    influences.append(responses[1]-responses[0])
                influence = torch.stack(influences,1)
            obs,m = packet['obs'],packet['motion']
            direction = torch.stack((obs[...,3].cos(),obs[...,3].sin()),-1)
            # Each response lives in its RECEIVER's body frame. Project there
            # first; a scalar closing effect can then be pooled across boats.
            benefit = -((influence[...,:2]+.5*influence[...,2:])*direction[:,None]).sum(-1)
            priority = (-obs[...,2]/10.).softmax(-1)
            if mode == 'projected_future':
                predicted_relative = obs[...,2,None]*direction+5.*reference_response[...,:2]
                priority = (-predicted_relative.norm(dim=-1)/7.).softmax(-1)
            diagonal = benefit[:,torch.arange(n),torch.arange(n)]
            helping = (benefit*mask*priority[:,None]).sum(-1)
            helped = (benefit*mask).sum(1)/(n-1)
            if mode == 'projected_future':
                diagonal = n*priority*diagonal
                helping = n*helping
                helped = n*priority*helped
            c,s = m[...,2].cos(),m[...,2].sin()
            own_velocity = torch.stack((c*m[...,3]+s*m[...,4],-s*m[...,3]+c*m[...,4]),-1)
            target_velocity = obs[...,4,None]*torch.stack((obs[...,5].cos(),obs[...,5].sin()),-1)
            closing = ((own_velocity-target_velocity)*direction).sum(-1)/5.
            intercept = -(target_velocity*direction).sum(-1)/5.
            logit = torch.logit(baseline[...,2].clamp(1e-5,1.-1e-5))/3.
            return torch.stack((logit,diagonal,helping,helped,closing,intercept,n*priority-1.,
                                obs[...,2]/60.),-1)
        if mode != 'response': raise ValueError('Unknown fusion readout representation.')
        n = node.shape[1]
        controls, nodes = [], []
        for gate in (0.,1.):
            nominal = gate*baseline[...,:2]+(1.-gate)*packet['obs'][...,12:14]
            control = torch.cat((nominal,gate-baseline[...,2:3]),-1)
            controls.append(control)
            nodes.append(self.condition(torch.cat((node,control),-1)))
        responses = {}
        for own,peer in itertools.product((0,1),repeat=2):
            left = nodes[own][:,:,None].expand(-1,-1,n,-1)
            right = nodes[peer][:,None].expand(-1,n,-1,-1)
            a = controls[own][:,:,None].expand(-1,-1,n,-1)
            b = controls[peer][:,None].expand(-1,n,-1,-1)
            responses[own,peer] = self.response_pair(self.condition_edge(torch.cat((left,right,edge,a,b),-1)))
        # Explicit own, teammate, and joint control-response contrasts. This
        # fixed feedforward relation encoding does not score or select actions.
        relation = torch.cat((responses[1,0]-responses[0,0],
            responses[0,1]-responses[0,0],
            responses[1,1]-responses[1,0]-responses[0,1]+responses[0,0]),-1)
        proxy = type('RelationInputs', (), {'peer_candidates': True})()
        geometry = InteractionActor.relation_features(proxy,packet['motion'],baseline[...,:2],packet['obs'][...,12:14])[...,:7]
        relation = torch.cat((geometry,relation),-1)
        average = (relation*mask[...,None]).sum(-2)/(n-1)
        score = relation.abs().masked_fill(~mask[...,None],-torch.inf)
        extreme = relation.gather(-2,score.argmax(-2,keepdim=True)).squeeze(-2)
        features = [self.node_inputs(packet,baseline),average,extreme]
        if self.task_response:
            influences = []
            encoded = (node,edge,mask)
            for sender in range(n):
                endpoints = []
                for gate in (0.,1.):
                    correction = torch.zeros_like(baseline[...,2])
                    correction[:,sender] = gate-baseline[:,sender,2]
                    endpoints.append(self.condition_on(encoded,packet,baseline,correction,details=True)[4])
                influences.append(endpoints[1]-endpoints[0])
            influence = torch.stack(influences,1)  # sender, receiver, response
            diagonal = influence[:,torch.arange(n),torch.arange(n)]
            outgoing = (influence*mask[...,None]).sum(2)/(n-1)
            incoming = (influence*mask[...,None]).sum(1)/(n-1)
            magnitude = influence.abs().masked_fill(~mask[...,None],-torch.inf)
            strongest = influence.gather(2,magnitude.argmax(2,keepdim=True)).squeeze(2)
            features.extend((diagonal,outgoing,incoming,strongest))
        return torch.cat(features,-1)

    def cooperative_effects(self,packet,baseline):
        if self.cooperation=='factorized':
            utility,influence = self.cooperative_response(packet,baseline)
            responsibility = utility.softmax(-1)
            team = torch.einsum('bij,bj->bi',influence,responsibility)
            own = influence.diagonal(dim1=-2,dim2=-1)
            coupling = torch.einsum('bij,bj,bkj->bik',influence,responsibility,influence)-team[:,:,None]*team[:,None]
            coupling = coupling*(1.-torch.eye(team.shape[1],device=team.device))
            return torch.stack((own,team),-1),coupling
        node,edge,mask = self.encode(packet,baseline)
        fields = self.cooperation_nodes(torch.cat((node,self.node_inputs(packet,baseline)),-1))
        n = node.shape[1]
        left,right = node[:,:,None].expand(-1,-1,n,-1),node[:,None].expand(-1,n,-1,-1)
        paired = torch.cat(((left+right)*.5,(left-right).abs(),left*right,(edge+edge.transpose(1,2))*.5),-1)
        coupling = self.cooperation_edges(paired)
        coupling = coupling*mask[...,None] if self.cooperation=='coverage' else coupling.squeeze(-1)*mask
        return fields,coupling

    def cooperative_response(self,packet,baseline):
        node,edge,mask = self.encode(packet,baseline)
        n=node.shape[1]
        own=self.cooperation_nodes(torch.cat((node,self.node_inputs(packet,baseline)),-1))
        left,right=node[:,:,None].expand(-1,-1,n,-1),node[:,None].expand(-1,n,-1,-1)
        influence=self.cooperation_edges(torch.cat((left,right,left-right,edge),-1)).squeeze(-1)*mask
        influence=influence+torch.diag_embed(own[...,0])
        utility=-packet['obs'][...,2]/5.+own[...,1]
        if hasattr(self,'responsibility'):
            features=self.responsibility_inputs(node,packet,baseline,own,influence)
            utility=self.responsibility(features,utility)
        return utility,influence

    def responsibility_inputs(self,node,packet,baseline,own,influence):
        clock=packet['central'][:,-1,None,None].expand(-1,baseline.shape[1],1)
        return torch.cat((node,self.node_inputs(packet,baseline),own,influence.mean(1)[...,None],
                          influence.mean(2)[...,None],clock),-1)

    def condition_on(self, encoded, packet, baseline, correction, details=False):
        node, edge, mask = encoded
        n = node.shape[1]
        gate = (baseline[..., 2:3]+correction[..., None]).clamp(0., 1.)
        nominal = gate*baseline[..., :2]+(1.-gate)*packet['obs'][..., 12:14]
        control = torch.cat((nominal, correction[..., None]), -1)
        nodes = self.condition(torch.cat((node, control), -1))
        left, right = nodes[:, :, None].expand(-1,-1,n,-1), nodes[:, None].expand(-1,n,-1,-1)
        a, z = control[:, :, None].expand(-1,-1,n,-1), control[:, None].expand(-1,n,-1,-1)
        pairs = self.condition_edge(torch.cat((left, right, edge, a, z), -1))
        if self.task_response:
            message = (pairs*mask[...,None]).sum(2)/(n-1)
            nodes = self.action_context(torch.cat((nodes,message),-1))
        pair_mean = (pairs*mask[..., None]).sum((1,2))/(n*(n-1))
        central = packet['central']
        state = central[:, :7*n].reshape(-1,n,7)
        # Pair each dynamics state and APF traversal index with its own
        # action-conditioned node before pooling. Moments alone would alias
        # configurations with different dynamics-to-action assignments.
        order = torch.arange(1,n+1,device=nodes.device,dtype=nodes.dtype)[None,:,None].expand(len(nodes),-1,-1)/7.
        value_nodes = self.value_node(torch.cat((nodes,state,order),-1))
        privileged = torch.cat((state.mean(1), state.square().mean(1), central[:, 7*n:]), -1)
        features = torch.cat((value_nodes.mean(1), pair_mean, privileged), -1)
        logits = torch.stack([head(features) for head in self.value],1)
        if self.distributional:
            bins = torch.arange(61,device=nodes.device,dtype=nodes.dtype)
            remaining = (60.*central[:,-1]).ceil()
            logits = logits.masked_fill(bins[None,None]>remaining[:,None,None],-1e4)
            q = -(logits.softmax(-1)*bins).sum(-1)/20.
        else:
            q = logits.squeeze(-1)
        motion = self.response_node(nodes)
        target_motion = self.response_target(nodes) if self.task_response else None
        if self.mediated:
            # A shared attacker response mediates effects on every receiver.
            # Own displacement is separately supervised, so peer effects
            # cannot be hidden in the much larger self-motion target.
            obs,m = packet['obs'],packet['motion']
            bearing = m[...,2]+obs[...,3]
            attacker_position = m[...,:2]+obs[...,2,None]*torch.stack((bearing.cos(),bearing.sin()),-1)
            direction = F.normalize(attacker_position.mean(1),dim=-1,eps=1e-6)
            phi = torch.atan2(direction[:,1],direction[:,0])
            angle = m[...,2]-phi[:,None]
            c,s = angle.cos(),angle.sin()
            def canonical(v):
                return torch.stack((c*v[...,0]-s*v[...,1],s*v[...,0]+c*v[...,1]),-1)
            own_motion = torch.cat((canonical(motion[...,:2]),canonical(motion[...,2:4])), -1)
            relative = obs[...,2,None]*torch.stack((obs[...,3].cos(),obs[...,3].sin()),-1)/60.
            target_velocity = obs[...,4,None]*torch.stack((obs[...,5].cos(),obs[...,5].sin()),-1)/5.
            geometry = torch.cat((canonical(relative),canonical(target_velocity),
                angle.sin()[...,None],angle.cos()[...,None],motion[...,4:7],own_motion),-1)
            # 13 response/geometry components and the two nominal controls.
            local = self.attacker_node(torch.cat((nodes,geometry,nominal),-1))
            weights = (self.attacker_attention(local).squeeze(-1)-obs[...,2]/15.).softmax(-1)
            common = self.attacker_response(torch.cat((local.mean(1),(local*weights[...,None]).sum(1)),-1))
            if self.physical_mediator:
                # A control-conditioned interaction prior: defenders alter
                # the attacker's preferred direction through relative motion.
                # It is a feature of the response model, never a replacement
                # policy or an online rollout/action selector.
                origin = attacker_position.mean(1)
                velocity = 5.*canonical(target_velocity).mean(1)
                cg,sg = phi.cos(),phi.sin()
                velocity_world = torch.stack((cg*velocity[:,0]-sg*velocity[:,1],sg*velocity[:,0]+cg*velocity[:,1]),-1)
                cm,sm = m[...,2].cos(),m[...,2].sin()
                displacement = 5.*torch.stack((cm*motion[...,0]-sm*motion[...,1],sm*motion[...,0]+cm*motion[...,1]),-1)
                def preferred(fraction):
                    attacker = origin+2.*fraction*velocity_world
                    defenders = m[...,:2]+fraction*displacement
                    force = -.1*attacker
                    limit = torch.full_like(attacker[:,0],50.)
                    for index in range(n):
                        difference = attacker-defenders[:,index]
                        distance = (difference.norm(dim=-1)-8.).clamp_min(.1)
                        active = distance<limit
                        magnitude = 3000.*(1./distance-1./limit)/distance.square()
                        force = force+torch.where(active[:,None],magnitude[:,None]*F.normalize(difference,dim=-1),torch.zeros_like(force))
                        limit = torch.where(active,distance,limit)
                    force = F.normalize(force,dim=-1,eps=1e-6)
                    return torch.stack((cg*force[:,0]+sg*force[:,1],-sg*force[:,0]+cg*force[:,1]),-1)
                force_features = torch.cat((preferred(0.),preferred(.5),preferred(1.),velocity/5.),-1)
                if self.temporal:
                    hidden = self.attacker_force_response[0](force_features)+self.intent_force(self.intent(packet['intent_history']))
                    common = common+self.attacker_force_response[2](self.attacker_force_response[1](hidden))
                else:
                    common = common+self.attacker_force_response(force_features)
            expanded = common[:,None].expand(-1,n,-1)
            def body(v):
                return torch.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1]),-1)
            target_motion = torch.cat((body(expanded[...,:2]),body(expanded[...,2:])), -1)-motion[...,:4]
        result = (q,motion,self.response_pair(pairs))
        if details:
            return (*result,logits,target_motion)
        return result

    def forward(self, packet, baseline, correction, peer=True, details=False):
        return self.condition_on(self.encode(packet, baseline, peer), packet, baseline, correction,details)

    def correction(self, packet, baseline, peer=True):
        if self.task_response:
            features = self.readout_features(packet,baseline,'response')
        else:
            node, _, _ = self.encode(packet, baseline, peer)
            team = node.mean(1, keepdim=True).expand_as(node)
            features = torch.cat((node, team, baseline[..., 2:3]), -1)
        raw = self.gate(features).squeeze(-1)
        # Direct probability correction permits movement away from a saturated
        # source adapter without changing either candidate control.
        return raw.tanh().clamp(-baseline[...,2],1.-baseline[...,2])


class ResponsePolicy:
    def __init__(self, source, checkpoint):
        self.source = OriginalPolicy(source)
        self.checkpoint=str(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.candidate_attention=saved.get('candidate_attention',False)
        factor_type=CooperativeEventModel if saved.get('event_factors',False) else CooperativeFactorModel
        self.model = (factor_type(saved['width'],saved.get('candidate_projection',True),saved.get('response_projection',True),saved.get('intent_refinement',False),saved.get('coordination_solver','convex')) if saved.get('factor_policy',False) else CandidateInterventionModel(saved['width'],saved.get('competing_risks',False),saved.get('degree_feature',False),saved.get('modal_fusion',False),saved.get('fusion_link','odds'),saved.get('candidate_projection',False),saved.get('response_projection',False),saved.get('intent_refinement',False)) if self.candidate_attention else ResponseGraph(saved['width'],task_response=saved.get('task_response',False),
                                  distributional=saved.get('distributional',False),mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),temporal=saved.get('temporal',False),cooperation=saved.get('cooperation',False),responsibility=saved.get('responsibility',False)).eval().requires_grad_(False)
        )
        if saved.get('execution_residual',False):
            self.model.enable_execution_residual(saved.get('execution_base_scale',.5))
        self.model.eval().requires_grad_(False)
        self.model.load_state_dict(saved['model'])
        self.horizon = saved['horizon']
        self.fusion_scale = saved.get('fusion_scale',1.)
        self.fusion_bias = saved.get('fusion_bias',0.)
        self.remaining, self.current = 0, None
        self.readout = saved.get('readout')
        self.local_contrast = saved.get('local_contrast')
        self.response_value_actor = saved.get('response_value_actor')
        self.conditional_value = saved.get('conditional_value')
        self.training_noisy_agility = saved.get('training_noisy_agility',True)
        self.history,self.last_time = [],None
        self.minimum_history = saved.get('minimum_history',0)
        self.cooperation_scale = saved.get('cooperation_field_scale')
        self.feedback_value = saved.get('feedback_value')
        self.feedback_mode = None

    @torch.no_grad()
    def choose_action(self, packet, deterministic=True):
        action = self.source.choose_action(packet, deterministic=True)[0]
        if self.fusion_scale==0.:
            action[:,2]=np.clip(action[:,2]+self.fusion_bias,0.,1.)
            return action,0.
        if self.model.temporal:
            remaining = float(packet['central'][-1])
            if self.last_time is None or remaining>=self.last_time:
                self.history = []
            self.history.append(dict(obs=packet['obs'],motion=packet['motion']))
            self.history = self.history[-8:]
            self.last_time = remaining
            if len(self.history)<self.minimum_history:
                self.remaining=0
                return action,0.
        if self.feedback_value is not None:
            p={k:v[None] for k,v in tensor_packet(packet).items()}
            p['intent_history']=torch.as_tensor(public_intent_history(self.history)[-1:])
            base=torch.as_tensor(action[None])
            if self.remaining==0:
                if self.feedback_value.get('continuous',False):
                    self.feedback_mode=optimize_feedback_program(self.model,p,base,self.feedback_value)
                else:
                    mean,variance=feedback_values(self.model,p,base,self.feedback_value)
                    score=mean-self.feedback_value['risk']*variance.clamp_min(0.).sqrt()
                    chosen=torch.cat((torch.zeros_like(score[:,:1]),score),-1).argmax(-1)[0]
                    self.feedback_mode=self.feedback_value['modes'][chosen]
                self.remaining=self.horizon
                if self.feedback_value.get('baseline_event',False) and not torch.count_nonzero(self.feedback_mode):
                    self.remaining=1
            shift=relational_feedback(self.model,p,base,self.feedback_mode,self.cooperation_scale)[0].numpy()
            numerator=action[:,2]*np.exp(shift);action[:,2]=numerator/(1.-action[:,2]+numerator)
            self.remaining-=1
            return action,0.
        if self.remaining == 0:
            p = {k:v[None] for k,v in tensor_packet(packet).items()}
            if self.model.temporal:
                p['intent_history'] = torch.as_tensor(public_intent_history(self.history)[-1:])
            base = torch.as_tensor(action[None])
            if self.candidate_attention:
                correction=self.model.fusion(p,base)
            elif self.conditional_value is not None:
                correction = conditional_value_correction(self.model,p,base,self.conditional_value)
            elif self.response_value_actor is not None:
                features = contrast_features(self.model,p,base)[0]
                features = contrast_transform(features,self.response_value_actor['transform'])
                correction = self.response_value_actor['radius']*torch.tanh(features@self.response_value_actor['weight'])
                correction = correction.clamp(-base[...,2],1.-base[...,2])
            elif self.local_contrast is not None:
                correction = contrast_correction(self.model,p,base,self.local_contrast)
            elif self.readout is None:
                correction = self.model.correction(p,base)
            elif self.readout.get('link') == 'coordination':
                fields,coupling = self.model.cooperative_effects(p,base)
                response = self.model.cooperative_response(p,base) if self.model.cooperation=='factorized' else None
                weights = self.readout['weight']
                fields = fields/self.readout['field_scale']
                coupling = coupling/self.readout['pair_scale']
                # The bounded interaction operator gives a contractive
                # shared update; every boat responds to the others' weights.
                if self.model.cooperation=='coverage':
                    coupling = (coupling*weights[4:]).sum(-1)
                    coupling = 1.5*coupling/(1.+coupling.abs().sum(-1).amax(-1)[:,None,None])
                else:
                    coupling = 1.5*weights[3].tanh()*coupling/(1.+coupling.abs().sum(-1).amax(-1)[:,None,None])
                prior = base[...,2]
                floor = self.readout.get('prior_floor',0.)
                reference = floor+(1.-2.*floor)*prior
                logit = torch.logit(reference.clamp(1e-6,1.-1e-6))
                field = weights[0]*logit/3.+weights[1]*fields[...,0]+weights[2]*fields[...,1]
                if self.model.cooperation=='coverage': field = field+weights[3]*fields[...,2]
                center = prior if self.readout.get('conditional_effects',False) else .5
                belief = prior.clone()
                for _ in range(8):
                    interaction = torch.einsum('bij,bj->bi',coupling,belief-center)
                    if response is not None:
                        utility,influence = response
                        responsibility = (utility+torch.einsum('bi,bij->bj',belief-prior,influence)).softmax(-1)
                        initial = torch.einsum('bij,bj->bi',influence,utility.softmax(-1))
                        changed = torch.einsum('bij,bj->bi',influence,responsibility)
                        interaction = weights[3]*(changed-initial)/self.readout['field_scale'][1]
                    shift = 3.*torch.tanh((field+interaction)/3.)
                    numerator = reference*shift.exp()
                    proposal = (numerator/(1.-reference+numerator)-reference+prior).clamp(0.,1.)
                    belief = .5*(belief+proposal)
                correction = torch.logit((belief-prior+reference).clamp(1e-6,1.-1e-6))-logit
                if floor == 0.:
                    correction = torch.where((prior==0.)|(prior==1.),torch.zeros_like(correction),correction)
            else:
                features = self.model.readout_features(p,base,self.readout.get('features_mode','latent'))
                if self.readout.get('allocation',False):
                    features = features-features.mean(1,keepdim=True)
                features = (features-self.readout['mean'])@self.readout['projection']
                if self.readout.get('allocation',False):
                    scores = .5*torch.tanh(features@self.readout['weight'][:-1])
                    scores = scores-scores.mean(-1,keepdim=True)
                    desired = base[...,2]+scores
                    budget = (base[...,2].sum(-1)+base.shape[1]*.5*torch.tanh(self.readout['weight'][-1])).clamp(0.,float(base.shape[1]))
                    if torch.count_nonzero(self.readout['weight']) == 0:
                        correction = torch.zeros_like(base[...,2])
                    else:
                        lower,upper = desired.min(-1).values-1.,desired.max(-1).values+1.
                        for _ in range(24):
                            midpoint = (lower+upper)*.5
                            excess = (desired-midpoint[:,None]).clamp(0.,1.).sum(-1)>budget
                            lower = torch.where(excess,midpoint,lower)
                            upper = torch.where(excess,upper,midpoint)
                        correction = (desired-(lower+upper)[:,None]*.5).clamp(0.,1.)-base[...,2]
                else:
                    features = torch.cat((features,torch.ones_like(features[...,:1])),-1)
                    if self.readout.get('link') == 'odds':
                        correction = 3.*torch.tanh((features@self.readout['weight'])/3.)
                    else:
                        correction = .5*torch.tanh(features@self.readout['weight'])
                        correction = correction.clamp(-base[...,2],1.-base[...,2])
            self.current = self.fusion_scale*correction[0].numpy()+self.fusion_bias
            self.remaining = self.horizon
        if self.candidate_attention and self.model.modal_fusion:
            if not np.count_nonzero(self.current):
                self.remaining-=1
                return action,0.
            if self.model.fusion_link=='odds':
                numerator=action[:,2,None]*np.exp(self.current)
                theta=numerator/(1.-action[:,2,None]+numerator)
            else:theta=self.model.fusion_weights(torch.as_tensor(action[:,2]),torch.as_tensor(self.current))
            nominal=self.model.modal_nominal(torch.as_tensor(action[:,:2]),torch.as_tensor(packet['obs'][:,12:14]),torch.as_tensor(theta)).numpy()
            action=np.column_stack((nominal,np.ones(len(action),dtype=np.float32)))
        elif self.candidate_attention and self.model.fusion_link=='probability':
            action[:,2]=self.model.fusion_weights(torch.as_tensor(action[:,2]),torch.as_tensor(self.current)).numpy()
        elif self.readout and self.readout.get('link') in ('odds','coordination'):
            odds = np.exp(self.current)
            floor = self.readout.get('prior_floor',0.)
            reference = floor+(1.-2.*floor)*action[:,2]
            numerator = reference*odds
            action[:,2] = np.clip(numerator/(1.-reference+numerator)-reference+action[:,2],0.,1.)
        else:
            action[:, 2] = np.clip(action[:, 2]+self.current, 0., 1.)
        self.remaining -= 1
        return action, 0.


def load_rows(directory, count, device):
    rows = [torch.load(Path(directory)/f'{i}.pth', weights_only=False, map_location='cpu') for i in range(count)]
    if len({r['horizon'] for r in rows}) != 1:
        raise ValueError('Cannot mix intervention durations without conditioning the value on duration.')
    tensor = lambda a:torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    keys = ('obs','motion','central','intent_history') if 'intent_history' in rows[0]['packet'] else ('obs','motion','central')
    return dict(packet={k:tensor([r['packet'][k] for r in rows]) for k in keys},
        baseline=tensor([r['baseline'] for r in rows]), correction=tensor([r['corrections'] for r in rows]),
        returns=tensor([[[b['discounted_return'] for b in bs] for bs in r['rows']] for r in rows]),
        response_node=tensor([[np.mean([b['node_response'] for b in bs], axis=0) for bs in r['rows']] for r in rows]),
        response_pair=tensor([[np.mean([b['pair_response'] for b in bs], axis=0) for bs in r['rows']] for r in rows]),
        response_target=(tensor([[np.mean([b['target_response'] for b in bs],axis=0) for bs in r['rows']] for r in rows])
                         if 'target_response' in rows[0]['rows'][0][0] else None),
        times=tensor([[[b['capture_time'] for b in bs] for bs in r['rows']] for r in rows]),
        horizon=rows[0]['horizon'],complete_returns=all(b.get('completed_episode',True) for r in rows for trials in r['rows'] for b in trials))


def subset(packet, index):
    return {k:v[index] for k,v in packet.items()}


@torch.no_grad()
def contrast_features(model, packet, baseline):
    """Public task response features for local node and pair value effects."""
    node = model.readout_features(packet,baseline,'response')
    obs,m = packet['obs'],packet['motion']
    distance,angle = obs[...,2],obs[...,3]
    position = distance[...,None]*torch.stack((angle.cos(),angle.sin()),-1)
    c,s = m[...,2].cos(),m[...,2].sin()
    velocity = obs[...,4,None]*torch.stack((obs[...,5].cos(),obs[...,5].sin()),-1)
    velocity -= torch.stack((c*m[...,3]+s*m[...,4],-s*m[...,3]+c*m[...,4]),-1)
    future = torch.stack([(position+t*velocity).norm(dim=-1)/20. for t in (1.,2.,4.,8.)],-1)
    kinetic = torch.cat((position/20.,velocity/5.,future,
        (distance-distance.min(-1,keepdim=True).values)[...,None]/20.,
        (distance-distance.mean(-1,keepdim=True))[...,None]/20.),-1)
    node = torch.cat((node,kinetic),-1)
    n = baseline.shape[1]
    left,right = node[:,:,None].expand(-1,-1,n,-1),node[:,None].expand(-1,n,-1,-1)
    proxy = type('RelationInputs',(),{'peer_candidates':True})()
    geometry = InteractionActor.relation_features(proxy,m,baseline[...,:2],obs[...,12:14])
    pair = torch.cat(((left+right)*.5,(left-right).abs(),
                      (geometry+geometry.transpose(1,2))*.5),-1)
    return node,pair


def contrast_transform(features, transform):
    value = ((features-transform['mean'])/transform['scale'])@transform['projection']
    if 'kernel_projection' in transform:
        kernel=torch.exp(-torch.cdist(value.reshape(-1,value.shape[-1]),transform['kernel_centers']).square()/(2.*value.shape[-1]))
        nonlinear=(kernel@transform['kernel_projection']).reshape(*value.shape[:-1],-1)
        value=torch.cat((.2*value,nonlinear),-1)
    if 'centers' in transform:
        kernel = torch.exp(-torch.cdist(value.reshape(-1,value.shape[-1]),transform['centers']).square()/(2.*value.shape[-1]))
        kernel = ((kernel-transform['kernel_mean'])/transform['kernel_scale']).reshape(*value.shape[:-1],-1)
        value = torch.cat((value,kernel),-1)
    return torch.cat((value,torch.ones_like(value[...,:1])),-1)


def contrast_design(node, pair, correction, radius):
    """Exact zero at the reference; joint terms vanish for isolated changes."""
    x = correction/radius
    n = x.shape[-1]
    unary = (node*x[...,None]).sum(-2)
    mask = torch.triu(torch.ones((n,n),dtype=torch.bool,device=x.device),1)
    coupled = (pair*(x[...,None]*x[...,None,:])[...,None])[:,mask].sum(1)
    return torch.cat((unary,coupled),-1)


@torch.no_grad()
def conditional_value_features(model,packet,baseline):
    """Keep response-to-task binding explicit in the conditional value basis."""
    latent,edge,mask=model.encode(packet,baseline)
    utility,influence=model.cooperative_response(packet,baseline)
    n=baseline.shape[1];prior=baseline[...,2];responsibility=utility.softmax(-1)
    own=influence.diagonal(dim1=-2,dim2=-1)
    team=torch.einsum('bij,bj->bi',influence,responsibility)
    received=(influence*mask).sum(1)/(n-1)
    helping=(influence*mask*responsibility[:,None]).sum(-1)
    obs,m=packet['obs'],packet['motion']
    direction=torch.stack((obs[...,3].cos(),obs[...,3].sin()),-1)
    c,s=m[...,2].cos(),m[...,2].sin()
    velocity=torch.stack((c*m[...,3]+s*m[...,4],-s*m[...,3]+c*m[...,4]),-1)
    target=obs[...,4,None]*torch.stack((obs[...,5].cos(),obs[...,5].sin()),-1)
    relative=target-velocity;closing=-(relative*direction).sum(-1)/5.
    semantic=torch.stack((own,team,helping,received,n*responsibility,utility/10.,
        (utility+obs[...,2]/5.)/2.,prior,prior*(1.-prior),
        torch.logit(prior.clamp(1e-5,1.-1e-5))/3.,closing,obs[...,2]/60.,
        packet['central'][:,-1,None].expand_as(prior),torch.full_like(prior,n/6.)),-1)
    node=torch.cat((semantic,model.node_inputs(packet,baseline),latent),-1)
    pair_effect=torch.einsum('bij,bj,bkj->bik',influence,responsibility,influence)-team[:,:,None]*team[:,None]
    du=utility[:,:,None]-utility[:,None]
    separation=(m[:,:,None,:2]-m[:,None,:,:2]).norm(dim=-1)/30.
    pair_semantic=torch.stack((pair_effect,influence+influence.transpose(1,2),
        (influence-influence.transpose(1,2)).abs(),own[:,:,None]*own[:,None],
        n*n*responsibility[:,:,None]*responsibility[:,None],du.abs()/10.,separation),-1)
    pair=torch.cat((pair_semantic,(edge+edge.transpose(1,2))*.5,
        (semantic[:,:,None]+semantic[:,None])*.5,(semantic[:,:,None]-semantic[:,None]).abs()),-1)
    return node,pair


def conditional_design(node,pair,eta,radius):
    x=eta/radius;n=x.shape[-1]
    mask=torch.triu(torch.ones(n,n,dtype=torch.bool,device=x.device),1)
    return torch.cat(((node*x[...,None]).sum(1),(node*x.square()[...,None]).sum(1),
        (pair*(x[:,:,None]*x[:,None])[...,None])[:,mask].sum(1)),-1)


@torch.no_grad()
def feedback_context(model,packet,baseline,transform):
    node,_=conditional_value_features(model,packet,baseline)
    responsibility=(node[...,4]/baseline.shape[1]).unsqueeze(-1)
    node=((node-transform['mean'])/transform['scale'])@transform['projection']
    return torch.cat((node.mean(1),node.amax(1),node.amin(1),(node*responsibility).sum(1)),-1)


def feedback_mode_kernel(left,right):
    # A difference kernel: every covariance with the unchanged program is zero.
    cross=torch.exp(-torch.cdist(left,right).square()/2.)
    return left@right.T/3.+cross-torch.exp(-left.square().sum(-1)[:,None]/2.)-torch.exp(-right.square().sum(-1)[None]/2.)+1.


@torch.no_grad()
def feedback_response_features(model,packet,baseline,modes,scale):
    """Bind a program to its candidate-conditioned physical effects."""
    corrections=[];prior=baseline[...,2]
    for mode in modes:
        shift=relational_feedback(model,packet,baseline,mode,scale)
        numerator=prior*shift.exp();corrections.append(numerator/(1.-prior+numerator)-prior)
    changes=torch.stack(corrections,1);choices=len(modes);b,n=prior.shape
    expand=lambda x:x[:,None].expand(-1,choices,*x.shape[1:]).flatten(0,1)
    encoded=model.encode(packet,baseline);p={k:expand(v) for k,v in packet.items()}
    result=model.condition_on((expand(encoded[0]),expand(encoded[1]),encoded[2]),p,expand(baseline),changes.flatten(0,1),details=True)
    own=result[1].reshape(b,choices,n,7);target=result[4].reshape(b,choices,n,4)
    own=own-own[:,:1];target=target-target[:,:1]
    angle=packet['obs'][...,3][:,None];c,s=angle.cos(),angle.sin()
    def rotate(v):return torch.stack((c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1]),-1)
    response=torch.cat((rotate(own[...,:2]),rotate(own[...,2:4]),own[...,4:],rotate(target[...,:2]),rotate(target[...,2:])), -1)
    utility,_=model.cooperative_response(packet,baseline);priority=utility.softmax(-1)[:,None,:,None]
    physical=changes[...,None]*(baseline[...,:2]-packet['obs'][...,12:14])[:,None]
    physical=torch.stack((physical.sum(-1)*.5,(physical[...,0]-physical[...,1])*.5),-1)
    response=torch.cat((response,physical),-1)
    return torch.cat((response.mean(2),response.amax(2),response.amin(2),(response*priority).sum(2)),-1)


def response_difference_kernel(left,right):
    dimension=left.shape[-1]
    return left@right.T/dimension+torch.exp(-torch.cdist(left,right).square()/(2.*dimension))-torch.exp(-left.square().sum(-1)[:,None]/(2.*dimension))-torch.exp(-right.square().sum(-1)[None]/(2.*dimension))+1.


@torch.no_grad()
def feedback_values(model,packet,baseline,saved):
    context=feedback_context(model,packet,baseline,saved['transform'])
    context=(context-saved['context_mean'])/saved['context_scale']
    if 'context_weight' in saved:context=context*saved['context_weight']
    if 'forest' in saved:
        means,variances=[],[]
        for tree in saved['forest']:
            mean,var=honest_tree_predict(tree,context.numpy())
            means.append(mean);variances.append(var)
        means=np.array(means)
        return torch.tensor(means.mean(0),dtype=torch.float32),torch.tensor(np.mean(variances,0)+means.var(0),dtype=torch.float32)
    affinity=torch.exp(-torch.cdist(context,saved['centers']).square()/(2.*context.shape[-1]*saved['length']**2))+.1
    if saved.get('response_kernel',False):
        effect=feedback_response_features(model,packet,baseline,saved['modes'],saved['field_scale'])[:,1:]/saved['effect_scale']
        response_kernel=response_difference_kernel(effect.flatten(0,1),saved['effect_centers'])
        cross=saved['amplitude']*affinity[:,None,:,None]*response_kernel.reshape(len(context),effect.shape[1],len(saved['centers']),-1)
        cross=cross.flatten(2)
        size=effect.square().sum(-1)/effect.shape[-1]
        diagonal=saved['amplitude']*1.1*(size+2.-2.*torch.exp(-size/2.))
    else:
        mode_kernel=feedback_mode_kernel(saved['modes'][1:],saved['modes'][1:])
        cross=(saved['amplitude']*affinity[:,None,:,None]*mode_kernel[None,:,None,:]).flatten(2)
        diagonal=saved['amplitude']*1.1*mode_kernel.diag()[None]
    mean=cross@saved['alpha']
    projected=cross@saved['inverse']
    variance=diagonal-(cross*projected).sum(-1)
    return mean,variance.clamp_min(0.)


@torch.no_grad()
def optimize_feedback_program(model,packet,baseline,saved):
    """Optimize a continuous joint feedback objective inside observed support."""
    if len(baseline)!=1:raise ValueError('The online program optimizer expects one team state.')
    context=(feedback_context(model,packet,baseline,saved['transform'])-saved['context_mean'])/saved['context_scale']
    if 'context_weight' in saved:context=context*saved['context_weight']
    affinity=(torch.exp(-torch.cdist(context,saved['centers']).square()/(2.*context.shape[-1]*saved['length']**2))+.1)[0]
    modes=saved['modes'];observed=modes[1:];actions=len(observed)
    operator=(saved['amplitude']*affinity[None,:,None]*torch.eye(actions)[:,None,:]).flatten(1)
    coefficient=operator@saved['alpha'];reduction=operator@saved['inverse']@operator.T
    strength=saved.get('program_trust',.1);risk=saved['risk']
    def energy(weight,gradient=False):
        difference=weight[:,None]-observed[None]
        cross=(-.5*difference.square().sum(-1)).exp();self_kernel=(-.5*weight.square().sum(-1)).exp()
        kernel=feedback_mode_kernel(weight,observed)
        mean=kernel@coefficient
        variance=saved['amplitude']*1.1*(weight.square().sum(-1)/3.+2.-2.*self_kernel)-(kernel@reduction*kernel).sum(-1)
        sd=variance.clamp_min(0.).sqrt()
        value=mean-risk*sd-strength*weight.square().sum(-1)
        if not gradient:return value
        derivative=observed[None]/3.-difference*cross[...,None]+weight[:,None]*self_kernel[:,None,None]
        variance_gradient=saved['amplitude']*1.1*(2.*weight/3.+2.*weight*self_kernel[:,None])-2.*torch.einsum('ba,bad->bd',kernel@reduction,derivative)
        variance_gradient=torch.where((variance>0.)[:,None],variance_gradient,torch.zeros_like(variance_gradient))
        grad=torch.einsum('a,bad->bd',coefficient,derivative)-.5*risk*variance_gradient/sd.clamp_min(1e-5)[:,None]-2.*strength*weight
        return value,grad
    def simplex(value):
        sorted_value=value.sort(-1,descending=True).values
        cumulative=sorted_value.cumsum(-1)-1.
        count=torch.arange(1,len(modes)+1,dtype=value.dtype)[None]
        rho=(sorted_value-cumulative/count>0.).sum(-1).clamp_min(1)
        threshold=cumulative.gather(1,(rho-1)[:,None])/rho[:,None]
        return (value-threshold).clamp_min(0.)
    mixture=torch.eye(len(modes));weights=mixture@modes
    for _ in range(24):
        current,gradient=energy(weights,True);direction=gradient@modes.T
        rate=.4/(1.+direction.norm(dim=-1,keepdim=True))
        for _ in range(4):
            proposed=simplex(mixture+rate*direction);changed=proposed@modes
            good=energy(changed)>=current
            mixture=torch.where(good[:,None],proposed,mixture);weights=mixture@modes
            rate=rate*.5
    values=energy(weights);chosen=int(values.argmax())
    return weights[chosen] if values[chosen]>1e-6 else torch.zeros(3)


def fit_feedback_value(args):
    """Conditional consequences of closed-loop cooperation, with paired noise."""
    torch.set_num_threads(1);rng=np.random.default_rng(args.seed)
    saved=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    policy=ResponsePolicy(args.source,args.graph_checkpoint);model=policy.model
    def load(directory,count):
        rows=[torch.load(directory/f'{i}.pth',weights_only=False) for i in range(count)]
        if any(r['intervention_coordinates']!='relational_feedback' or r['horizon']!=args.horizon for r in rows):
            raise ValueError('Feedback programs and their duration must match their value targets.')
        packet={k:torch.tensor(np.array([r['packet'][k] for r in rows]),dtype=torch.float32) for k in ('obs','motion','central','intent_history')}
        base=torch.tensor(np.array([r['baseline'] for r in rows]),dtype=torch.float32)
        means,noise=[],[]
        for row in rows:
            outcomes=-np.array([[t['capture_time'] for t in trials] for trials in row['rows']]).reshape(len(row['modes']),-1,2).mean(-1)
            difference=outcomes[1:]-outcomes[:1];means.append(difference.mean(-1))
            c=np.cov(difference,ddof=1)/difference.shape[-1]
            noise.append(.5*c+.5*np.diag(np.diag(c))+.25*np.eye(len(c)))
        return dict(packet=packet,baseline=base,y=np.array(means),noise=np.array(noise),modes=torch.tensor(rows[0]['modes']))
    training=[load(args.output/'train',args.train_states)];validation=[load(args.output/'development',args.test_states)]
    if args.mixed_source:
        training.append(load(args.mixed_source/'train',args.mixed_train_states))
        validation.append(load(args.mixed_source/'development',args.mixed_test_states))
    train={k:np.concatenate([r[k] for r in training]) for k in ('y','noise')};train['modes']=training[0]['modes']
    with torch.no_grad():
        flat=torch.cat([conditional_value_features(model,r['packet'],r['baseline'])[0].flatten(0,1) for r in training])
        mean=flat.mean(0);scale=flat.std(0).clamp_min(.05);normalized=(flat-mean)/scale
        _,singular,vt=torch.linalg.svd(normalized[:,31:],full_matrices=False)
        projection=torch.zeros(flat.shape[-1],35);projection[:31,:31]=torch.eye(31)
        projection[31:,31:]=vt[:4].T/(singular[:4]/np.sqrt(len(flat))).clamp_min(.3)
        transform=dict(mean=mean,scale=scale,projection=projection)
        context=torch.cat([feedback_context(model,r['packet'],r['baseline'],transform) for r in training])
        cm,cs=context.mean(0),context.std(0).clamp_min(.1);context=(context-cm)/cs
    if args.honest_forest:
        fit_honest_feedback(args,saved,model,train,validation,context,transform,cm,cs)
        return
    modes=train['modes'];actions=len(modes)-1;count=len(context)
    km=feedback_mode_kernel(modes[1:],modes[1:]).double().numpy()
    effect=None;effect_scale=None;conditional_kernel=None
    if args.response_kernel:
        effect=torch.cat([feedback_response_features(model,r['packet'],r['baseline'],modes,policy.cooperation_scale)[:,1:] for r in training])
        effect_scale=effect.square().mean((0,1)).sqrt().clamp_min(.01)
        effect=effect/effect_scale;conditional_kernel=response_difference_kernel(effect.flatten(0,1),effect.flatten(0,1)).double().numpy()
    amplitude=max(.5,float(np.mean(train['y']**2)-np.mean(np.diagonal(train['noise'],axis1=1,axis2=2))))
    if conditional_kernel is not None:amplitude/=max(.1,float(np.diag(conditional_kernel).mean()))
    noise=np.zeros((count*actions,count*actions))
    for i,c in enumerate(train['noise']):noise[i*actions:(i+1)*actions,i*actions:(i+1)*actions]=c
    y=train['y'].reshape(-1);folds=np.empty(count,dtype=int);folds[rng.permutation(count)]=np.arange(count)%4
    distance=torch.cdist(context,context).square().double().numpy()/context.shape[-1]
    diagnostics=[];risk=.5
    def make_kernel(length):
        state=np.exp(-distance/(2.*length**2))+.1
        return amplitude*(np.kron(state,km) if conditional_kernel is None else np.repeat(np.repeat(state,actions,0),actions,1)*conditional_kernel)
    for length in (.5,1.,2.):
        kernel=make_kernel(length)
        for regularization in (.3,1.,3.,10.):
            mu=np.zeros_like(train['y']);var=np.zeros_like(mu)
            for fold in range(4):
                fit=np.repeat(folds!=fold,actions);test=~fit
                inverse=np.linalg.inv(kernel[np.ix_(fit,fit)]+regularization*noise[np.ix_(fit,fit)]+1e-6*np.eye(fit.sum()))
                cross=kernel[np.ix_(test,fit)];mu[folds==fold]=(cross@inverse@y[fit]).reshape(-1,actions)
                var[folds==fold]=(kernel.diagonal()[test]-(cross@inverse*cross).sum(-1)).reshape(-1,actions).clip(0.)
            chosen=np.c_[np.zeros(count),mu-risk*np.sqrt(var)].argmax(-1)
            gain=np.c_[np.zeros(count),train['y']][np.arange(count),chosen]
            diagnostics.append(dict(length=length,regularization=regularization,mse=float(np.mean((mu-train['y'])**2)),
                selected_gain=float(gain.mean()),gain_se=float(gain.std(ddof=1)/np.sqrt(count)),active=int(np.count_nonzero(chosen))))
    selected=max(diagnostics,key=lambda r:r['selected_gain']-.5*r['gain_se'])
    length=selected['length'];kernel=make_kernel(length)
    metric_weight=None;metric_history=[]
    if args.adaptive_metric:
        if args.response_kernel:raise ValueError('Use the response metric directly in this experiment.')
        # Tie each learned relevance across mean, extrema and responsibility
        # pooling. Paired outcome covariance stays inside the likelihood.
        log_weight=nn.Parameter(torch.zeros(context.shape[-1]//4,dtype=torch.float64))
        optimizer=torch.optim.Adam([log_weight],lr=.03)
        inputs=context.double();mode_cov=torch.tensor(km,dtype=torch.float64)
        observation=selected['regularization']*torch.tensor(noise,dtype=torch.float64)+1e-6*torch.eye(len(y),dtype=torch.float64)
        target=torch.tensor(y,dtype=torch.float64)
        for step in range(1,201):
            relevance=log_weight.clamp(-2.,2.).exp().repeat(4)
            points=inputs*relevance
            distances=(points.square().sum(-1)[:,None]+points.square().sum(-1)[None]-2.*points@points.T).clamp_min(0.)/points.shape[-1]
            covariance=amplitude*torch.kron(torch.exp(-distances/(2.*length**2))+.1,mode_cov)+observation
            cholesky=torch.linalg.cholesky(covariance)
            coefficient=torch.cholesky_solve(target[:,None],cholesky)[:,0]
            loss=.5*(target*coefficient).sum()+cholesky.diag().log().sum()+2.*log_weight.square().sum()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step%50==0:
                metric_history.append(dict(step=step,negative_log_evidence=float(loss.detach()),min_weight=float(relevance.min().detach()),max_weight=float(relevance.max().detach())))
        metric_weight=log_weight.detach().clamp(-2.,2.).exp().repeat(4).float()
        context=context*metric_weight
        distance=torch.cdist(context,context).square().double().numpy()/context.shape[-1]
        kernel=amplitude*np.kron(np.exp(-distance/(2.*length**2))+.1,km)
    inverse=np.linalg.inv(kernel+selected['regularization']*noise+1e-6*np.eye(len(y)))
    fitted=dict(transform=transform,context_mean=cm,context_scale=cs,centers=context,modes=modes,
        length=length,amplitude=amplitude,inverse=torch.tensor(inverse,dtype=torch.float32),
        alpha=torch.tensor(inverse@y,dtype=torch.float32),risk=risk,diagnostics=diagnostics,selected=selected)
    if metric_weight is not None:fitted.update(context_weight=metric_weight,metric_history=metric_history)
    if effect is not None:fitted.update(response_kernel=True,effect_centers=effect.flatten(0,1),effect_scale=effect_scale,field_scale=policy.cooperation_scale)
    metrics=[]
    for valid in validation:
        pred,var=feedback_values(model,valid['packet'],valid['baseline'],fitted)
        choice=torch.cat((torch.zeros(len(pred),1),pred-risk*var.sqrt()),-1).argmax(-1).numpy()
        gains=np.c_[np.zeros(len(pred)),valid['y']][np.arange(len(pred)),choice]
        metrics.append(dict(defenders=valid['baseline'].shape[1],development_gain=float(gains.mean()),development_gain_se=float(gains.std(ddof=1)/np.sqrt(len(gains))),
            development_active=int(np.count_nonzero(choice)),difference_mae=float(np.abs(pred.numpy()-valid['y']).mean()),
            zero_mae=float(np.abs(valid['y']).mean()),choices=choice.tolist()))
    result=dict(selected=selected,metrics=metrics)
    fitted['development']=result
    saved.update(feedback_value=fitted,horizon=args.horizon,conditional_value=None,readout=None,
        local_contrast=None,response_value_actor=None,seed=args.seed)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved);print(result,flush=True)


def honest_tree_predict(tree,x):
    mean=np.zeros((len(x),len(tree['mean'])));variance=np.zeros_like(mean)
    def visit(node,indices):
        if not len(indices):return
        if 'feature' not in node:
            mean[indices]=node['mean'];variance[indices]=node['variance'];return
        left=x[indices,node['feature']]<=node['threshold']
        visit(node['left'],indices[left]);visit(node['right'],indices[~left])
    visit(tree,np.arange(len(x)));return mean,variance


def fit_honest_feedback(args,saved,model,train,validation,context,transform,cm,cs):
    """Outcome-independent leaf estimation for discontinuous team consequences."""
    rng=np.random.default_rng(args.seed);x=context.numpy();y=train['y'];noise=train['noise'];n,d=y.shape
    precision=np.linalg.inv(noise);rhs=np.einsum('bij,bj->bi',precision,y)
    signal=max(.5,float(np.mean(y*y)-np.diagonal(noise,axis1=1,axis2=2).mean()));regularization=np.eye(d)/signal
    def posterior(indices):
        a=regularization+precision[indices].sum(0);b=rhs[indices].sum(0)
        covariance=np.linalg.inv(a);mean=covariance@b
        evidence=.5*b@mean-.5*np.linalg.slogdet(a)[1]+.5*np.linalg.slogdet(regularization)[1]
        return mean,covariance.diagonal(),evidence
    def grow(indices,depth):
        mean,var,parent=posterior(indices);node=dict(mean=mean.tolist(),variance=var.tolist())
        if depth==0 or len(indices)<12:return node
        candidates=np.flatnonzero(x[indices].std(0)>.01)
        if not len(candidates):return node
        candidates=rng.choice(candidates,size=min(18,len(candidates)),replace=False);best_gain=.5;best=None
        for feature in candidates:
            for threshold in np.unique(np.quantile(x[indices,feature],[.2,.35,.5,.65,.8])):
                left=x[indices,feature]<=threshold
                if min(left.sum(),(~left).sum())<6:continue
                gain=posterior(indices[left])[2]+posterior(indices[~left])[2]-parent
                if gain>best_gain:best_gain=gain;best=(feature,float(threshold),indices[left],indices[~left])
        if best is not None:
            feature,threshold,left,right=best
            node.update(feature=int(feature),threshold=threshold,left=grow(left,depth-1),right=grow(right,depth-1))
        return node
    def estimate(node,indices):
        mean,var,_=posterior(indices);node['mean']=mean.tolist();node['variance']=var.tolist()
        if 'feature' in node:
            left=x[indices,node['feature']]<=node['threshold']
            if min(left.sum(),(~left).sum())<3:
                for k in ('feature','threshold','left','right'):node.pop(k)
            else:estimate(node['left'],indices[left]);estimate(node['right'],indices[~left])
    forest=[];oob_mean=np.zeros((n,d));oob_second=np.zeros_like(oob_mean);oob_variance=np.zeros_like(oob_mean);counts=np.zeros(n)
    for _ in range(96):
        perm=rng.permutation(n);size=int(.8*n);split=perm[:size//2];estimation=perm[size//2:size];out=perm[size:]
        tree=grow(split,3);estimate(tree,estimation);forest.append(tree)
        mean,var=honest_tree_predict(tree,x[out]);oob_mean[out]+=mean;oob_second[out]+=mean**2;oob_variance[out]+=var;counts[out]+=1
    mean=oob_mean/counts[:,None]
    variance=oob_variance/counts[:,None]+oob_second/counts[:,None]-mean**2
    candidates=[]
    for risk in (0.,.25,.5,1.):
        choice=np.c_[np.zeros(n),mean-risk*np.sqrt(variance.clip(0.))].argmax(-1)
        gain=np.c_[np.zeros(n),y][np.arange(n),choice]
        candidates.append(dict(risk=risk,gain=float(gain.mean()),se=float(gain.std(ddof=1)/np.sqrt(n)),active=int(np.count_nonzero(choice))))
    selected=max(candidates,key=lambda v:v['gain']-.5*v['se'])
    fitted=dict(transform=transform,context_mean=cm,context_scale=cs,modes=train['modes'],forest=forest,
        risk=selected['risk'],oob=candidates,selected=selected)
    metrics=[]
    for valid in validation:
        pred,var=feedback_values(model,valid['packet'],valid['baseline'],fitted)
        choice=torch.cat((torch.zeros(len(pred),1),pred-fitted['risk']*var.clamp_min(0.).sqrt()),-1).argmax(-1).numpy()
        gain=np.c_[np.zeros(len(pred)),valid['y']][np.arange(len(pred)),choice]
        metrics.append(dict(defenders=valid['baseline'].shape[1],selected_gain=float(gain.mean()),gain_se=float(gain.std(ddof=1)/np.sqrt(len(gain))),
            active=int(np.count_nonzero(choice)),difference_mae=float(np.abs(pred.numpy()-valid['y']).mean()),zero_mae=float(np.abs(valid['y']).mean()),choices=choice.tolist()))
    result=dict(selected=selected,metrics=metrics);fitted['development']=result
    saved.update(feedback_value=fitted,horizon=args.horizon,conditional_value=None,readout=None,local_contrast=None,response_value_actor=None,seed=args.seed)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved);print(result,flush=True)


def fit_capture_responsibility(args):
    """Learn prospective receiver roles instead of assigning them by proximity."""
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    saved=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    graph=ResponsePolicy(args.source,args.graph_checkpoint).model
    def load(split,count):
        packets=[];actions=[];labels=[];roots=[];delays=[]
        for root in range(count):
            row=torch.load(args.state_source/split/f'{root}.pth',weights_only=False);ts=row['transitions'];last=ts[-1]
            n=len(last['obs']);attacker=last['next_central'][7*n:7*n+2]*60.
            captured=last['next_obs'][:,2].min()<5. and np.linalg.norm(attacker)>15.
            end=60.*(1.-float(last['next_central'][-1])) if captured else 60.
            winner=int(last['next_obs'][:,2].argmin());history=public_intent_history(ts)
            for index in range(0,len(ts),2):
                t=ts[index];delay=end-60.*(1.-float(t['central'][-1]))
                packet={k:t[k] for k in ('obs','motion','central')};packet['intent_history']=history[index]
                packets.append(packet);actions.append(t['action']);roots.append(root);delays.append(delay)
                labels.append(winner*7+int(np.searchsorted([1.,2.,4.,8.,16.,32.],delay)) if captured else n*7)
        packet={k:torch.tensor(np.array([p[k] for p in packets]),dtype=torch.float32) for k in packets[0]}
        baseline=torch.tensor(np.array(actions),dtype=torch.float32);features=[];priors=[]
        with torch.no_grad():
            for start in range(0,len(packets),512):
                p=subset(packet,slice(start,start+512));base=baseline[start:start+512]
                node=graph.encode(p,base)[0];utility,influence=graph.cooperative_response(p,base)
                own=torch.stack((influence.diagonal(dim1=-2,dim2=-1),utility+p['obs'][...,2]/5.),-1)
                features.append(graph.responsibility_inputs(node,p,base,own,influence));priors.append(utility)
        return dict(features=torch.cat(features),prior=torch.cat(priors),label=torch.tensor(labels),
            root=torch.tensor(roots),delay=torch.tensor(delays),n=baseline.shape[1])
    train,valid=load('train',args.train_states),load('test',args.test_states)
    model=CaptureResponsibility(train['features'].shape[-1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    def predict(data,index=slice(None)):
        event,failure=model.logits(data['features'][index],data['prior'][index])
        return torch.cat((event.flatten(1),failure[:,None]),-1)
    def metrics():
        with torch.no_grad():
            logits=predict(valid);event=logits[:,:-1].reshape(-1,valid['n'],7)
            role=torch.logsumexp(event-model.time_centers/8.,-1)
            captured=valid['label']<valid['n']*7;actual=valid['label']//7
            correct=(role.argmax(-1)==actual);near=captured&(valid['delay']<=8.)
            prior_correct=(valid['prior'].argmax(-1)==actual)
            return dict(joint_nll=float(F.cross_entropy(logits,valid['label'])),
                receiver_accuracy=float(correct[captured].float().mean()),baseline_accuracy=float(prior_correct[captured].float().mean()),
                within8_accuracy=float(correct[near].float().mean()),within8_baseline=float(prior_correct[near].float().mean()))
    initial=metrics();print(dict(step=0,**initial),flush=True)
    best,best_score,history,selected=None,float('inf'),[],0
    for step in range(1,args.updates+1):
        index=torch.tensor(rng.integers(len(train['label']),size=256))
        logits=predict(train,index);loss=F.cross_entropy(logits,train['label'][index])
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%100==0 or step==args.updates:
            row=dict(step=step,loss=float(loss.detach()),**metrics());history.append(row);print(row,flush=True)
            if row['joint_nll']<best_score:
                best_score,selected=row['joint_nll'],step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
    graph.responsibility=CaptureResponsibility(train['features'].shape[-1]);graph.responsibility.load_state_dict(best)
    saved.update(model=graph.state_dict(),responsibility=True,responsibility_history=history,
        responsibility_initial=initial,responsibility_selected_step=selected,seed=args.seed)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


@torch.no_grad()
def conditional_value_correction(model,packet,baseline,saved):
    node,pair=conditional_value_features(model,packet,baseline)
    node=contrast_transform(node,saved['node_transform']);pair=contrast_transform(pair,saved['pair_transform'])
    w=saved['weights'];d=node.shape[-1];n=baseline.shape[1]
    unary=torch.einsum('bnf,hf->bhn',node,w[:,:d])
    diagonal=torch.einsum('bnf,hf->bhn',node,w[:,d:2*d])
    coupling=torch.einsum('bnmf,hf->bhnm',pair,w[:,2*d:])*(1.-torch.eye(n,device=node.device))[None,None]
    def energy(x):
        return (unary*x[:,None]+diagonal*x[:,None].square()).sum(-1)+.5*torch.einsum('bhnm,bn,bm->bh',coupling,x,x)
    x=torch.zeros_like(baseline[...,2])
    curvature=(2.*diagonal.abs()+coupling.abs().sum(-1)).amax((1,2))+2.*saved['trust_seconds']+.01
    for _ in range(16):
        values=energy(x);gradients=unary+2.*diagonal*x[:,None]+torch.einsum('bhnm,bm->bhn',coupling,x)
        centered=values-values.mean(-1,keepdim=True)
        risk=(centered[...,None]*gradients).mean(1)/(values.var(-1,unbiased=False)+.04).sqrt()[:,None]
        gradient=gradients.mean(1)-saved['risk']*risk-2.*saved['trust_seconds']*x
        x=(x+gradient/curvature[:,None]).clamp(-1.,1.)
    values=energy(x)
    benefit=values.mean(-1)-saved['risk']*values.std(-1,unbiased=False)-saved['trust_seconds']*x.square().sum(-1)
    return saved['radius']*torch.where((benefit>0.)[:,None],x,torch.zeros_like(x))


def pathwise_team_cost(actor,critic,source,packet,innovations,*,adapt=True,temperature=.25,interpolated_forward=False,stop_source_jacobian=False,rescue_weight=0.,adjoint_horizon=0,opportunity_weight=0.,opportunity_width=1.):
    """Exact event costs, with an interpolated hitting-time surrogate gradient.

    Forward capture/failure and time quantization match the source task. The
    backward signal differentiates the within-period crossing time. This is a
    surrogate gradient, not a claim that a hard stopping time is differentiable.
    """
    dynamics=DifferentiableInterventionDynamics
    state,measured,agility,remaining=dynamics.from_packet(packet)
    history=dynamics.raw_history(packet).detach();b,n=packet['obs'].shape[:2]
    active=torch.arange(b,device=state.device)
    cost=torch.zeros(b,device=state.device);trust=torch.zeros_like(cost)
    closest=torch.full_like(remaining,float('inf'))
    encounter_mass=torch.zeros_like(remaining);encounter_moment=torch.zeros_like(remaining);elapsed=torch.zeros_like(remaining)
    p=packet
    mask=torch.triu(torch.ones(n,n,dtype=torch.bool,device=state.device),1)
    for step in range(innovations.shape[1]):
        if adjoint_horizon and step and step%adjoint_horizon==0:
            # Keep the complete physical future and terminal outcomes, while
            # limiting an event's adjoint to a local causal control segment.
            # The retained closest-encounter graph still credits its own
            # segment when a much later terminal failure is observed.
            state=state.detach();measured=measured.detach()
            p={k:v.detach() for k,v in p.items()}
        source_obs=p['obs'].detach() if stop_source_jacobian else p['obs']
        baseline=source(source_obs.flatten(0,1),True,False)[0].reshape(len(active),n,3)
        eta=actor.fusion(p,baseline) if adapt else torch.zeros_like(baseline[...,2])
        prior=baseline[...,2]
        theta=actor.fusion_weights(prior,eta) if adapt else prior
        nominal_delta=None
        if eta.ndim==3:
            change=actor.modal_change(baseline[...,:2],p['obs'][...,12:14],theta,prior)
            action=baseline;nominal_delta=750.*change
        else:
            action=torch.cat((baseline[...,:2],theta[...,None]),-1)
            change=(theta-prior)[...,None]*(baseline[...,:2]-p['obs'][...,12:14])
        trust=trust.index_add(0,active,change.square().mean((1,2))/innovations.shape[1])
        before_distance=(state[:,:-1,:2]-state[:,-1:,:2]).norm(dim=-1)
        duration=torch.minimum(60.*remaining,torch.full_like(remaining,.2))
        elapsed=elapsed+duration
        state,measured=dynamics.step(state,measured,agility,action,innovations[active,step],nominal_delta)
        remaining=(remaining-.2/60.).clamp_min(0.)
        expired=remaining<1e-5
        distance=(state[:,:-1,:2]-state[:,-1:,:2]).norm(dim=-1)
        separation=(state[:,:-1,None,:2]-state[:,None,:-1,:2]).norm(dim=-1)[:,mask]
        failure=(state[:,-1,:2].norm(dim=-1)<=15.)|(separation.amin(-1)<=5.)
        minimum,winner=distance.min(-1);capture=(minimum<5.)&~failure
        closest=torch.minimum(closest,minimum)
        if opportunity_weight:
            # A running envelope counts an encounter only as its opportunity
            # improves; successive frames are not independent capture events.
            # This is a geometric opportunity kernel, not a calibrated event
            # probability. Its value is the observed remaining task time that
            # an earlier cooperative interception could save.
            mass=torch.maximum(encounter_mass,torch.sigmoid((5.-minimum)/opportunity_width))
            encounter_moment=encounter_moment+(mass-encounter_mass)*elapsed.detach()
            encounter_mass=mass
        previous=before_distance.gather(1,winner[:,None])[:,0]
        crossing=duration*((previous-5.)/(previous-minimum).clamp_min(1e-5)).clamp(0.,1.)
        correction=crossing-duration
        event_correction=correction if interpolated_forward else correction-correction.detach()
        increment=duration+failure*60.*remaining+capture*event_correction
        if rescue_weight:
            # A capped failure has a flat pathwise derivative even when the
            # team almost captured the target. Its closest future encounter
            # supplies a bounded-slope auxiliary direction. Forward costs
            # remain the exact task costs; paired functional interventions
            # must validate any benefit from this explicitly surrogate credit.
            unsuccessful=failure|(expired&~capture)
            rescue=rescue_weight*F.softplus((closest-5.)/.5)
            increment=increment+unsuccessful*(rescue-rescue.detach())
        if opportunity_weight:
            terminal=failure|capture|expired
            factual_time=(elapsed+failure*60.*remaining).detach()
            opportunity=factual_time*encounter_mass-encounter_moment
            increment=increment-opportunity_weight*terminal*(opportunity-opportunity.detach())
        cost=cost.index_add(0,active,increment)
        keep=(~(failure|capture))&~expired
        # An absorbed trajectory has no subsequent dynamics or adjoint.
        # Multiplying its later cost by zero is insufficient: an ill-conditioned
        # post-terminal QP can otherwise inject 0 * NaN into the backward pass.
        if not bool(keep.any()):return cost,trust
        history=dynamics.advance_history(history,state,measured).detach()
        active=active[keep];state=state[keep];measured=measured[keep]
        agility=agility[keep];remaining=remaining[keep];history=history[keep];closest=closest[keep]
        encounter_mass=encounter_mass[keep];encounter_moment=encounter_moment[keep];elapsed=elapsed[keep]
        p=dynamics.packet(state,measured,agility,remaining)
        p['intent_history']=dynamics.intent_history(history)
    if innovations.shape[1]>=300:
        continuation=60.*remaining
    else:
        source_obs=p['obs'].detach() if stop_source_jacobian else p['obs']
        baseline=source(source_obs.flatten(0,1),True,False)[0].reshape(len(active),n,3)
        continuation=-20.*critic(p,baseline,torch.zeros_like(baseline[...,2]))[0].mean(-1)
    cost=cost.index_add(0,active,continuation)
    return cost,trust


def functional_control_step(model,source,parameters,gradient,packets,rng,*,radius,damping,iterations):
    """Precondition team credit in the metric of changed candidate mixtures.

    A Gauss--Newton control metric retains all candidate and peer dependence.
    Its action on a vector is computed exactly by autodiff; no parameter-sized
    matrix is formed. Complete paired outcomes still accept or reject the step.
    """
    metric=0.
    for packet in packets:
        indices=torch.as_tensor(rng.integers(len(packet['obs']),size=64),device=packet['obs'].device)
        p={k:v[indices] for k,v in packet.items()};b,n=p['obs'].shape[:2]
        with torch.no_grad():baseline=source(p['obs'].flatten(0,1),True,False)[0].reshape(b,n,3)
        eta=model.fusion(p,baseline);prior=baseline[...,2]
        theta=model.fusion_weights(prior,eta)
        if eta.ndim==3:
            nominal=model.modal_nominal(baseline[...,:2],p['obs'][...,12:14],theta)
        else:
            nominal=theta[...,None]*baseline[...,:2]+(1.-theta[...,None])*p['obs'][...,12:14]
        metric=metric+.5*(nominal-nominal.detach()).square().mean()/len(packets)
    first=torch.autograd.grad(metric,parameters,create_graph=True)
    def dot(a,b):return sum((x*y).sum() for x,y in zip(a,b))
    def multiply(vector):
        product=torch.autograd.grad(dot(first,vector),parameters,retain_graph=True)
        return [p.detach()+damping*v for p,v in zip(product,vector)]
    direction=[torch.zeros_like(g) for g in gradient]
    residual=[g.detach().clone() for g in gradient];search=[g.clone() for g in residual]
    squared=dot(residual,residual);initial=float(squared)
    for iteration in range(iterations):
        product=multiply(search);curvature=dot(search,product)
        if not bool(torch.isfinite(curvature)) or curvature<=1e-20:break
        alpha=squared/curvature
        direction=[d+alpha*p for d,p in zip(direction,search)]
        residual=[r-alpha*h for r,h in zip(residual,product)]
        following=dot(residual,residual)
        if following<1e-8*max(initial,1e-20):squared=following;break
        beta=following/squared.clamp_min(1e-20)
        search=[r+beta*p for r,p in zip(residual,search)];squared=following
    curvature=dot(direction,multiply(direction))
    scale=radius/curvature.clamp_min(1e-20).sqrt()
    if not bool(torch.isfinite(scale)):raise FloatingPointError('Non-finite functional control step.')
    with torch.no_grad():
        for p,d in zip(parameters,direction):p.add_(d,alpha=-float(scale))
    return dict(control_radius=radius,damping=damping,iterations=iteration+1,
        relative_residual=float((squared/max(initial,1e-20)).sqrt()),parameter_step_norm=float(scale*dot(direction,direction).sqrt()))


def fit_pathwise_intervention(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    device=torch.device(args.device);saved=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    degree=args.degree_feature or saved.get('degree_feature',False)
    modal=args.modal_fusion or saved.get('modal_fusion',False)
    link=args.fusion_link or saved.get('fusion_link','odds')
    if saved.get('pathwise') and link!=saved.get('fusion_link','odds'):
        raise ValueError('Changing fusion coordinates requires an identity-initialized source adapter, not reinterpreting trained shifts.')
    model=CandidateInterventionModel(saved['width'],saved.get('competing_risks',False),degree,modal,link,saved.get('candidate_projection',False),saved.get('response_projection',False),saved.get('intent_refinement',False)).to(device)
    def compatible(weights):
        result=dict(weights)
        if modal and result['adapter.3.weight'].shape[0]==1:
            result['adapter.3.weight']=result['adapter.3.weight'].repeat(2,1)
            result['adapter.3.bias']=result['adapter.3.bias'].repeat(2)
        return result
    model.load_state_dict(compatible(saved['model']),strict=not (degree or modal));critic=copy.deepcopy(model).eval().requires_grad_(False)
    if saved.get('pathwise'):
        critic.load_state_dict(compatible(saved['pathwise']['bootstrap_model']),strict=not (degree or modal))
    else:
        nn.init.zeros_(model.adapter[-1].weight);nn.init.zeros_(model.adapter[-1].bias)
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('adapter.','token.','pair.','blocks.','degree.','mode_condition.','projection_')))
    source=OriginalPolicy(args.source).actor.to(device)
    rows=[torch.load(args.state_source/'train'/f'{i}.pth',weights_only=False) for i in range(args.train_states)]
    packet={k:torch.as_tensor(np.asarray([r['packet'][k] for r in rows]),dtype=torch.float32,device=device)
        for k in ('obs','motion','central','intent_history')}
    if args.start_states:
        initial=[]
        for root in range(args.train_states):
            transitions=torch.load(args.context_source/'train'/f'{root}.pth',weights_only=False)['transitions']
            p={k:transitions[0][k] for k in ('obs','motion','central')}
            p['intent_history']=public_intent_history(transitions[:1])[0];initial.append(p)
        packet={k:torch.cat((v,torch.as_tensor(np.asarray([p[k] for p in initial]),device=device,dtype=torch.float32))) for k,v in packet.items()}
    packet_sets=[packet]
    if args.mixed_source:
        rows=[torch.load(args.mixed_source/'train'/f'{i}.pth',weights_only=False) for i in range(args.mixed_train_states)]
        mixed={k:torch.as_tensor(np.asarray([r['packet'][k] for r in rows]),device=device,dtype=torch.float32) for k in packet}
        if args.start_states:
            initial=[]
            for r in rows:
                first=r['transitions'][0];p={k:first[k] for k in ('obs','motion','central')}
                p['intent_history']=public_intent_history([first])[0];initial.append(p)
            mixed={k:torch.cat((v,torch.as_tensor(np.asarray([p[k] for p in initial]),device=device,dtype=torch.float32))) for k,v in mixed.items()}
        packet_sets.append(mixed)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.pathwise_lr,weight_decay=.01)
    history=[];started=time.perf_counter()
    for step in range(1,args.updates+1):
        if args.paired_step_check and (step-1)%args.check_interval==0:
            anchor=[p.detach().clone() for p in parameters]
            accumulated=[torch.zeros_like(p) for p in parameters]
        gradients=[];metrics=[]
        for packet in packet_sets:
            ix=torch.tensor(rng.integers(len(packet['obs']),size=args.batch_size),device=device)
            batch={k:v[ix].repeat_interleave(2,0) for k,v in packet.items()}
            noise=torch.randn(args.batch_size,args.horizon,2+3*(packet['obs'].shape[1]+1),device=device)
            noise=torch.stack((noise,-noise),1).flatten(0,1)
            with torch.no_grad():reference,_=pathwise_team_cost(model,critic,source,batch,noise,adapt=False)
            cost,trust=pathwise_team_cost(model,critic,source,batch,noise,stop_source_jacobian=args.stop_source_jacobian,
                rescue_weight=args.rescue_weight,adjoint_horizon=args.adjoint_horizon,
                opportunity_weight=args.opportunity_weight,opportunity_width=args.opportunity_width)
            loss=(cost-reference).mean()+args.trust_seconds*4.*trust.mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient=nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(gradient):raise FloatingPointError('Non-finite conditional pathwise gradient.')
            gradients.append([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in parameters])
            metrics.append(dict(defenders=packet['obs'].shape[1],model_gain=float((reference-cost).mean().detach()),
                control_change_rms=float(trust.mean().sqrt().detach()),gradient_norm=float(gradient)))
        adjusted=[[g.clone() for g in gs] for gs in gradients];cosine=None
        if len(gradients)==2:
            dot=sum((a*b).sum() for a,b in zip(*gradients));norms=[sum(g.square().sum() for g in gs) for gs in gradients]
            cosine=float(dot/(norms[0]*norms[1]).clamp_min(1e-12).sqrt())
            if args.project_conflicts and dot<0:
                for i in range(2):
                    adjusted[i]=[a-dot/norms[1-i].clamp_min(1e-12)*b for a,b in zip(gradients[i],gradients[1-i])]
        objective_weights=[args.primary_weight]+[1.]*(len(adjusted)-1)
        for index,p in enumerate(parameters):
            p.grad=sum(w*gs[index] for w,gs in zip(objective_weights,adjusted))/sum(objective_weights)
        natural=None
        if args.functional_natural:
            if not args.paired_step_check:raise ValueError('Functional control steps require paired complete-outcome checks.')
            accumulated=[a+p.grad for a,p in zip(accumulated,parameters)]
            if step%args.check_interval==0 or step==args.updates:
                count=(step-1)%args.check_interval+1
                natural=functional_control_step(model,source,parameters,[g/count for g in accumulated],packet_sets,rng,
                    radius=args.control_step,damping=args.natural_damping,iterations=args.natural_iterations)
        else:optimizer.step()
        check=None
        if args.paired_step_check and (step%args.check_interval==0 or step==args.updates):
            # The interpolated event derivative proposes a functional change.
            # Complete paired outcomes, including changed terminal events,
            # determine its sign and scale. These are training interventions;
            # no rollout or selection is performed at deployment.
            displacement=[p.detach()-a for p,a in zip(parameters,anchor)]
            scales=(-1.,0.,1.,3.)
            check_batches=[]
            for packet in packet_sets:
                start=len(packet['obs'])//2 if args.check_initial_only and args.start_states else 0
                ix=torch.tensor(rng.integers(start,len(packet['obs']),size=args.check_batch_size),device=device)
                batch={k:v[ix].repeat_interleave(2,0) for k,v in packet.items()}
                noise=torch.randn(args.check_batch_size,args.horizon,2+3*(packet['obs'].shape[1]+1),device=device)
                noise=torch.stack((noise,-noise),1).flatten(0,1)
                check_batches.append((batch,noise))
            costs=[];efforts=[]
            with torch.no_grad():
                for scale in scales:
                    for p,a,d in zip(parameters,anchor,displacement):p.copy_(a+scale*d)
                    team_costs=[];team_efforts=[]
                    for batch,noise in check_batches:
                        cost,trust=pathwise_team_cost(model,critic,source,batch,noise)
                        team_costs.append(cost.reshape(-1,2).mean(-1).cpu().numpy())
                        team_efforts.append(trust.mean().item())
                    costs.append(team_costs);efforts.append(team_efforts)
            costs=np.asarray(costs);efforts=np.asarray(efforts)
            changes=costs-costs[1:2]
            mean=changes.mean(-1)
            # Paired antithetic paths share a root; root averages are the units.
            standard_error=changes.std(-1,ddof=1)/np.sqrt(args.check_batch_size)
            eligible=(mean[:,0]<=args.team_tolerance) if args.primary_only_check else (mean<=args.team_tolerance).all(-1)
            weights=np.asarray(objective_weights);weights=weights/weights.sum()
            scores=(mean*weights).sum(-1)+args.trust_seconds*4.*((efforts-efforts[1:2])*weights).sum(-1)
            scores=scores+args.value_confidence*np.sqrt((standard_error.square() if isinstance(standard_error,torch.Tensor) else standard_error**2)@(weights**2))
            scores[~eligible]=np.inf
            chosen=int(np.argmin(scores));scale=scales[chosen]
            with torch.no_grad():
                for p,a,d in zip(parameters,anchor,displacement):p.copy_(a+scale*d)
            check=dict(scales=list(scales),selected_scale=scale,paired_time_changes=mean.tolist(),
                paired_standard_errors=standard_error.tolist(),team_tolerance=args.team_tolerance,
                objective_weights=weights.tolist(),value_confidence=args.value_confidence,
                root_averaged_costs=costs.tolist())
            if natural is not None:check['control_metric']=natural
        if step%10==0 or step==1 or step==args.updates:
            row=dict(pathwise_step=step,teams=metrics,gradient_cosine=cosine,
                elapsed_seconds=round(time.perf_counter()-started,2))
            if check is not None:row['functional_intervention']=check
            history.append(row)
            print({k:({a:b for a,b in v.items() if a!='root_averaged_costs'} if k=='functional_intervention' else v) for k,v in row.items()},flush=True)
        elif check is not None:
            row=dict(pathwise_step=step,functional_intervention=check,elapsed_seconds=round(time.perf_counter()-started,2))
            history.append(row)
            print({**row,'functional_intervention':{k:v for k,v in check.items() if k!='root_averaged_costs'}},flush=True)
    saved.update(model={k:v.detach().cpu() for k,v in model.state_dict().items()},horizon=1,seed=args.seed,degree_feature=degree,modal_fusion=modal,fusion_link=link,
        pathwise=dict(horizon=args.horizon,training_states=args.train_states,updates=args.updates,
            history=history,temperature=.25,start_states=args.start_states,bootstrap_model={k:v.detach().cpu() for k,v in critic.state_dict().items()},
            gradient='original-physics-and-QP-with-tight-complementarity',event_model='hard-forward-interpolated-event-time-gradient',
            stop_source_jacobian=args.stop_source_jacobian,project_conflicts=args.project_conflicts,
            paired_step_check=args.paired_step_check,check_interval=args.check_interval,check_batch_size=args.check_batch_size,
            learning_rate=args.pathwise_lr,check_initial_only=args.check_initial_only,absorbing_rollout=True,rescue_weight=args.rescue_weight,
            adjoint_horizon=args.adjoint_horizon,
            opportunity_weight=args.opportunity_weight,opportunity_width=args.opportunity_width,
            functional_natural=args.functional_natural,control_step=args.control_step,
            primary_weight=args.primary_weight,primary_only_check=args.primary_only_check,value_confidence=args.value_confidence,
            mixed_source=str(args.mixed_source) if args.mixed_source else None))
    saved.pop('temporal_diagnostic',None)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


@torch.no_grad()
def score_intervention_rollout(actor,critic,source,packet,innovations,exploration,*,sigma,trace,option_steps=1,transitions=False):
    """Collect exact terminal consequences and conditional fusion credits.

    Exploration is a Gaussian latent before the existing feasible fusion map.
    Its likelihood remains defined when that map clips a coefficient. Physics,
    safety, and capture indicators are never differentiated by this estimator.
    """
    dynamics=DifferentiableInterventionDynamics
    state,measured,agility,remaining=dynamics.from_packet(packet)
    raw_history=dynamics.raw_history(packet);b,n=packet['obs'].shape[:2]
    active=torch.arange(b,device=state.device);total=torch.zeros(b,device=state.device)
    capturer=torch.full((b,),-2,dtype=torch.long,device=state.device)
    mask=torch.triu(torch.ones(n,n,dtype=torch.bool,device=state.device),1)
    p=packet;frames=[]
    for step in range(0,innovations.shape[1],option_steps):
        baseline=source(p['obs'].flatten(0,1),True,False)[0].reshape(len(active),n,3)
        mean=actor.fusion(p,baseline);eta=mean+sigma*exploration[active,step]
        evaluations=[mean]
        for receiver in range(n):
            replacement=eta.clone();replacement[:,receiver]=mean[:,receiver]
            evaluations.append(replacement)
        repeated={k:v.repeat((n+1,)+(1,)*(v.ndim-1)) for k,v in p.items()}
        estimates=-20.*critic(repeated,baseline.repeat(n+1,1,1),torch.cat(evaluations))[0].mean(-1).reshape(n+1,len(active))
        value=estimates[0];conditional=estimates[1:].transpose(0,1)
        record={k:v.clone() for k,v in p.items()}
        record.update(baseline=baseline,mean=mean,eta=eta,value=value,
            conditional=conditional,indices=active.clone())
        if transitions:
            for key in ('obs','motion','central','intent_history'):record['next_'+key]=torch.zeros_like(p[key])
            record['duration']=torch.zeros_like(value)
        local=torch.arange(len(active),device=state.device)
        option_cost=torch.zeros_like(value);option_done=torch.zeros_like(value,dtype=torch.bool)
        for offset in range(min(option_steps,innovations.shape[1]-step)):
            if offset:baseline=source(p['obs'].flatten(0,1),True,False)[0].reshape(len(active),n,3)
            theta=actor.fusion_weights(baseline[...,2],eta)
            change=actor.modal_change(baseline[...,:2],p['obs'][...,12:14],theta,baseline[...,2])
            duration=torch.minimum(60.*remaining,torch.full_like(remaining,.2))
            state,measured=dynamics.step(state,measured,agility,baseline,innovations[active,step+offset],750.*change)
            remaining=(remaining-.2/60.).clamp_min(0.)
            distance=(state[:,:-1,:2]-state[:,-1:,:2]).norm(dim=-1)
            separation=(state[:,:-1,None,:2]-state[:,None,:-1,:2]).norm(dim=-1)[:,mask]
            failure=(state[:,-1,:2].norm(dim=-1)<=15.)|(separation.amin(-1)<=5.)
            capture=(distance.amin(-1)<5.)&~failure
            done=failure|capture|(remaining<1e-5)
            capturer[active[done]]=torch.where(capture[done],distance[done].argmin(-1),-1)
            increment=duration+failure*60.*remaining
            total.index_add_(0,active,increment);option_cost.index_add_(0,local,increment);option_done[local]=done
            if transitions:
                record['duration'].index_add_(0,local,duration)
                advanced=dynamics.advance_history(raw_history,state,measured)
                finish=done|(offset==min(option_steps,innovations.shape[1]-step)-1)
                if bool(finish.any()):
                    final_packet=dynamics.packet(state[finish],measured[finish],agility[finish],remaining[finish])
                    final_packet['intent_history']=dynamics.intent_history(advanced[finish])
                    for key,value in final_packet.items():record['next_'+key][local[finish]]=value
            keep=~done
            if not bool(keep.any()):active=active[keep];break
            raw_history=dynamics.advance_history(raw_history,state,measured)
            active=active[keep];state=state[keep];measured=measured[keep];eta=eta[keep];local=local[keep]
            agility=agility[keep];remaining=remaining[keep];raw_history=raw_history[keep]
            following=dynamics.packet(state,measured,agility,remaining)
            if 'fusion_readout' in p:following['fusion_readout']=p['fusion_readout'][keep]
            p=following;p['intent_history']=dynamics.intent_history(raw_history)
        record.update(cost=option_cost,done=option_done);frames.append(record)
        if len(active)==0:break
    if bool((~frames[-1]['done']).any()):
        raise ValueError('Score credits require complete terminal futures, not truncated labels.')
    next_value=torch.zeros_like(total);accumulated=torch.zeros_like(total);returns=torch.zeros_like(total)
    for frame in reversed(frames):
        ix=frame['indices'];alive=(~frame['done']).float()
        delta=frame['cost']+alive*next_value[ix]-frame['value']
        advantage=delta+trace**option_steps*alive*accumulated[ix]
        returns[ix]=frame['cost']+alive*returns[ix]
        # The conditional baseline replaces only this receiver's sampled
        # fusion while retaining every peer's actual sampled fusion.
        frame['advantage']=advantage[:,None]+frame['value'][:,None]-frame['conditional']
        frame['return']=returns[ix].clone()
        frame['capturer']=capturer[ix]
        next_value[ix]=frame['value'];accumulated[ix]=advantage
    keys=('obs','motion','central','intent_history','baseline','mean','eta','advantage','return','capturer')
    if transitions:keys+=('next_obs','next_motion','next_central','next_intent_history','duration','done','indices')
    return {k:torch.cat([f[k] for f in frames]) for k in keys},total


@torch.no_grad()
def conditional_score_contrasts(actor,source,packet,rng,*,roots,repeats,sigma,option_steps,radius):
    """Single-receiver interventions with identical future innovations.

    The two paths differ only in their initial receiver fusion. Afterwards
    both follow the same stochastic fusion policy, with common environmental
    and exploration innovations. Labels therefore match the trained option Q.
    """
    class ZeroValue:
        def __call__(self,p,base,eta):return base.new_zeros((len(base),3)),None
    device=packet['obs'].device;n=packet['obs'].shape[1]
    distance=packet['obs'][...,2].amin(-1)
    eligible=torch.where((distance>5.5)&(distance<18.))[0]
    if not len(eligible):eligible=torch.arange(len(distance),device=device)
    ix=eligible[torch.as_tensor(rng.integers(len(eligible),size=roots),device=device)]
    p={k:v[ix] for k,v in packet.items()}
    base=source(p['obs'].flatten(0,1),True,False)[0].reshape(roots,n,3)
    mean=actor.fusion(p,base);direction=torch.zeros_like(mean)
    receivers=torch.as_tensor(rng.integers(n,size=roots),device=device)
    vector=torch.randn(roots,2,device=device);vector=vector/vector.abs().amax(-1,keepdim=True).clamp_min(1e-6)
    direction[torch.arange(roots,device=device),receivers]=radius*vector
    batch={k:v.repeat_interleave(repeats*2,0) for k,v in p.items()}
    noise=torch.randn(roots*repeats,300,2+3*(n+1),device=device).repeat_interleave(2,0)
    exploration=torch.randn(roots*repeats,300,n,2,device=device).repeat_interleave(2,0)
    first=direction.repeat_interleave(repeats,0)/sigma
    exploration[0::2,0]=first;exploration[1::2,0]=-first
    _,cost=score_intervention_rollout(actor,ZeroValue(),source,batch,noise,exploration,
        sigma=sigma,trace=1.,option_steps=option_steps)
    cost=cost.reshape(roots,repeats,2);delta=cost[:,:,0]-cost[:,:,1]
    return dict(packet=p,baseline=base,plus=mean+direction,minus=mean-direction,
        returns=cost.mean(1),delta=delta.mean(1),standard_error=delta.std(1)/repeats**.5,
        roots=ix,receivers=receivers)


@torch.no_grad()
def paired_policy_credits(actor,source,data,rng,*,roots,repeats,sigma,option_steps):
    """Antithetic receiver-policy gradients from coupled complete futures.

    Roots are uniform on-policy decision states. Peers' sampled initial
    fusions are fixed across each pair, and only one receiver's Gaussian
    perturbation changes sign. Environment and continuation-policy innovations
    are identical within each pair. At the collecting mean, symmetric centered
    returns give the exact Gaussian mean score estimator of this conditional Q;
    subsequent clipped optimization is a local policy-improvement surrogate.
    """
    class ZeroValue:
        def __call__(self,p,base,eta):return base.new_zeros((len(base),3)),None
    device=data['obs'].device;n=data['obs'].shape[1]
    ix=torch.as_tensor(rng.integers(len(data['obs']),size=roots),device=device)
    p={k:data[k][ix] for k in ('obs','motion','central','intent_history')}
    base=data['baseline'][ix];mean=actor.fusion(p,base)
    receivers=torch.as_tensor(rng.integers(n,size=roots),device=device)
    row=torch.arange(roots,device=device)
    perturbation=torch.randn_like(mean)
    plus=mean+sigma*perturbation;minus=plus.clone()
    minus[row,receivers]=mean[row,receivers]-sigma*perturbation[row,receivers]
    batch={k:v.repeat_interleave(repeats*2,0) for k,v in p.items()}
    noise=torch.randn(roots*repeats,300,2+3*(n+1),device=device).repeat_interleave(2,0)
    exploration=torch.randn(roots*repeats,300,n,2,device=device).repeat_interleave(2,0)
    exploration[0::2,0]=perturbation.repeat_interleave(repeats,0)
    opposite=perturbation.clone();opposite[row,receivers]*=-1.
    exploration[1::2,0]=opposite.repeat_interleave(repeats,0)
    _,cost=score_intervention_rollout(actor,ZeroValue(),source,batch,noise,exploration,
        sigma=sigma,trace=1.,option_steps=option_steps)
    cost=cost.reshape(roots,repeats,2);difference=cost[:,:,0]-cost[:,:,1]
    delta=difference.mean(1)
    # A cyclic re-pairing retains both conditional cost marginals while removing
    # their common future. This diagnoses coupling, not a policy/task advantage.
    shuffled=cost[:,:,0]-cost[:,:,1].roll(1,1)
    return dict(packet=p,baseline=base,mean=mean,plus=plus,minus=minus,
        returns=cost.mean(1),paired_returns=cost,delta=delta,
        standard_error=difference.std(1)/repeats**.5,roots=ix,receivers=receivers,
        perturbation=perturbation[row,receivers],
        coupling_second_moment=float(difference.square().mean()),
        shuffled_second_moment=float(shuffled.square().mean()))


@torch.no_grad()
def paired_factor_credits(actor,source,data,rng,roots,repeats,sigma,option_steps,mean_peers=False):
    """Antithetic receiver interventions followed by the deployed policy.

    Only the first option is perturbed. The common continuation is the same
    deterministic feedback used for collecting states and executing the policy.
    Thus the finite-noise credit differentiates a locally smoothed Q of that
    policy, rather than the value of a persistently noisy continuation.
    """
    ix=torch.as_tensor(rng.integers(len(data['obs']),size=roots))
    packet={key:data[key][ix] for key in ('obs','motion','central','intent_history')}
    base=data['baseline'][ix];n=base.shape[1];branches=2*n if mean_peers else n+1
    mean=actor.fusion(packet,base)[...,0]
    epsilon=torch.randn(roots,repeats,n)
    if mean_peers:
        # Condition on teammates' deployed choices, not a jointly perturbed
        # team. Each pair changes precisely one receiver's first option.
        initial=torch.zeros(roots,repeats,branches,n)
        positive=torch.arange(n)*2;negative=positive+1
        for receiver in range(n):
            initial[:,:,2*receiver,receiver]=epsilon[:,:,receiver]
            initial[:,:,2*receiver+1,receiver]=-epsilon[:,:,receiver]
    else:
        initial=epsilon[:,:,None].expand(-1,-1,branches,-1).clone()
        for receiver in range(n):initial[:,:,receiver+1,receiver]*=-1.
        positive=torch.zeros(n,dtype=torch.long);negative=torch.arange(n)+1
    samples=mean[:,None,None]+sigma*initial
    expanded={key:value.repeat_interleave(repeats*branches,0) for key,value in packet.items()}
    innovations=torch.randn(roots*repeats,300,2+3*(n+1)).repeat_interleave(branches,0)
    exploration=torch.zeros(roots*repeats*branches,300,n,2)
    exploration[:,0]=initial.flatten(0,2)[...,None].expand(-1,-1,2)
    class ZeroValue:
        def __call__(self,p,b,e):return b.new_zeros(len(b),3),None
    trajectories,cost=score_intervention_rollout(actor,ZeroValue(),source,expanded,innovations,exploration,
        sigma=sigma,trace=1.,option_steps=option_steps)
    cause=trajectories['capturer'][:len(expanded['obs'])]
    return dict(packet=packet,baseline=base,mean=mean,epsilon=epsilon,samples=samples,
        returns=cost.reshape(roots,repeats,branches),capturers=cause.reshape(roots,repeats,branches),
        positive=positive,negative=negative)


@torch.no_grad()
def check_factor_policy_step(model,source,parameters,before,initial_sets,rng,*,roots,repeats,option_steps):
    """Check the whole updated feedback policy on fresh paired training futures.

    This is an optimizer acceptance check, not held-out evidence of a benefit.
    Both policies use their deterministic deployed controls throughout. A
    rejected proposal restores both its parameters and (in the caller) Adam
    state; its outcome data still remain useful for learning the critic.
    """
    class ZeroValue:
        def __call__(self,p,b,e):return b.new_zeros(len(b),3),None
    after=[p.detach().clone() for p in parameters];records=[]
    for initial in initial_sets:
        ix=torch.as_tensor(rng.choice(len(initial['obs']),size=min(roots,len(initial['obs'])),replace=False))
        packet={key:value[ix].repeat_interleave(repeats,0) for key,value in initial.items()}
        b,n=packet['obs'].shape[:2]
        innovations=torch.randn(b,300,2+3*(n+1));exploration=torch.zeros(b,300,n,2)
        for parameter,value in zip(parameters,before):parameter.copy_(value)
        _,reference=score_intervention_rollout(model,ZeroValue(),source,packet,innovations,exploration,
            sigma=0.,trace=1.,option_steps=option_steps)
        for parameter,value in zip(parameters,after):parameter.copy_(value)
        _,candidate=score_intervention_rollout(model,ZeroValue(),source,packet,innovations,exploration,
            sigma=0.,trace=1.,option_steps=option_steps)
        difference=(candidate-reference).reshape(len(ix),repeats).mean(-1)
        records.append(dict(defenders=n,scenes=len(ix),repeats=repeats,difference=float(difference.mean()),
            standard_error=float(difference.std()/len(ix)**.5),reference=float(reference.mean()),candidate=float(candidate.mean())))
    # Separate fleet checks prevent an improvement in one fleet from hiding
    # a deterioration in the other. Final evidence always uses unseen futures.
    accepted=all(row['difference']<=0. for row in records) and any(row['difference']<0. for row in records)
    if not accepted:
        for parameter,value in zip(parameters,before):parameter.copy_(value)
    return dict(accepted=accepted,regimes=records)


def project_event_backup(probability,times,duration,done,capturer,bins):
    """Categorical Bellman projection retaining the capturing-boat identity."""
    b,h,c=probability.shape;n=(c-1)//len(bins)
    location=times[:,:-1]+duration[:,None]
    current_bins=torch.minimum(bins[None],(times[:,-1]+duration)[:,None]).contiguous()
    upper=torch.searchsorted(current_bins,location.contiguous()).clamp(1,len(bins)-1);lower=upper-1
    low_time=current_bins.gather(1,lower);high_time=current_bins.gather(1,upper)
    fraction=((location-low_time)/(high_time-low_time).clamp_min(1e-6)).clamp(0.,1.)
    offset=(torch.arange(n,device=bins.device)*len(bins)).repeat_interleave(len(bins))[None]
    result=torch.zeros_like(probability)
    result.scatter_add_(-1,(offset+lower)[:,None].expand(-1,h,-1),probability[...,:-1]*(1.-fraction[:,None]))
    result.scatter_add_(-1,(offset+upper)[:,None].expand(-1,h,-1),probability[...,:-1]*fraction[:,None])
    result[...,-1]=probability[...,-1]
    terminal=torch.zeros_like(probability)
    upper=torch.searchsorted(current_bins,duration[:,None].contiguous()).squeeze(1).clamp(1,len(bins)-1);lower=upper-1
    low_time=current_bins.gather(1,lower[:,None]).squeeze(1);high_time=current_bins.gather(1,upper[:,None]).squeeze(1)
    fraction=((duration-low_time)/(high_time-low_time).clamp_min(1e-6)).clamp(0.,1.)
    low_label=torch.where(capturer>=0,len(bins)*capturer+lower,c-1)
    high_label=torch.where(capturer>=0,len(bins)*capturer+upper,c-1)
    terminal.scatter_add_(-1,low_label[:,None,None].expand(-1,h,1),(1.-fraction)[:,None,None].expand(-1,h,1))
    terminal.scatter_add_(-1,high_label[:,None,None].expand(-1,h,1),fraction[:,None,None].expand(-1,h,1))
    return torch.where(done[:,None,None],terminal,result)


@torch.no_grad()
def temporal_event_targets(actor,target,source,data,sigma,trace,batch_size):
    """Distributional TD(lambda) along complete current-policy trajectories."""
    probabilities=[];time_values=[]
    for start in range(0,len(data['obs']),batch_size):
        ix=torch.arange(start,min(start+batch_size,len(data['obs'])))
        packet={key:data['next_'+key][ix] for key in ('obs','motion','central','intent_history')}
        n=packet['obs'].shape[1];base=source(packet['obs'].flatten(0,1),True,False)[0].reshape(len(ix),n,3)
        mean=actor.fusion(packet,base)[...,0]
        proposals=(base[:,None,:,2]+mean[:,None]+sigma*torch.randn(len(ix),4,n)).clamp(0.,1.)
        factors=target.centralized_factors(packet,base)
        probabilities.append(target.factor_logits(factors,proposals,base[...,2]).softmax(-1).mean(1))
        time_values.append(factors['times'])
    probabilities=torch.cat(probabilities);time_values=torch.cat(time_values)
    result=torch.zeros_like(probabilities);continuations={}
    for row in range(len(probabilities)-1,-1,-1):
        episode=int(data['indices'][row]);future=probabilities[row]
        if episode in continuations:future=(1.-trace)*future+trace*continuations[episode]
        target_row=project_event_backup(future[None],time_values[row:row+1],data['duration'][row:row+1],
            data['done'][row:row+1],data['capturer'][row:row+1],target.capture_times)[0]
        result[row]=target_row;continuations[episode]=target_row
    return result


def mediated_option_gradient(actor,critic,source,packet,mean,steps):
    """Local consequence slope through physics and the opponent response.

    The returned slope is detached before forming a LINEAR control variate.
    Consequently approximate tail values, event-boundary derivatives, and
    controller switches affect its variance reduction, not the expectation of
    the corrected Gaussian policy gradient.
    """
    d=DifferentiableInterventionDynamics
    displacement=mean.detach().requires_grad_(True)
    state,measured,agility,remaining=d.from_packet(packet);history=d.raw_history(packet)
    b,n=mean.shape;alive=mean.new_ones(b);total=mean.new_zeros(b);p=packet
    noise=mean.new_zeros(b,2+3*(n+1));mask=torch.triu(torch.ones(n,n,dtype=torch.bool),1)
    for _ in range(steps):
        baseline=source(p['obs'].flatten(0,1),True,False)[0].reshape(b,n,3)
        theta=(baseline[...,2]+displacement).clamp(0.,1.)
        change=(theta-baseline[...,2])[...,None]*(baseline[...,:2]-p['obs'][...,12:14])
        duration=torch.minimum(60.*remaining,torch.full_like(remaining,.2))
        state,measured=d.step(state,measured,agility,baseline,noise,750.*change)
        remaining=(remaining-.2/60.).clamp_min(0.)
        distance=(state[:,:-1,:2]-state[:,-1:,:2]).norm(dim=-1)
        separation=(state[:,:-1,None,:2]-state[:,None,:-1,:2]).norm(dim=-1)[:,mask]
        failure=(state[:,-1,:2].norm(dim=-1)<=15.)|(separation.amin(-1)<=5.)
        capture=(distance.amin(-1)<5.)&~failure
        total=total+alive*(duration+failure*60.*remaining)
        alive=alive*(~(failure|capture|(remaining<1e-5))).to(alive.dtype)
        history=d.advance_history(history,state,measured)
        p=d.packet(state,measured,agility,remaining);p['intent_history']=d.intent_history(history)
    baseline=source(p['obs'].flatten(0,1),True,False)[0].reshape(b,n,3)
    theta=actor.joint_weights(p,baseline)
    tail=critic.factor_cost(critic.centralized_factors(p,baseline),theta,baseline[...,2])[:,0].mean(-1)
    consequence=total+alive*tail
    gradient=torch.autograd.grad(consequence.sum(),displacement)[0]
    if not bool(torch.isfinite(gradient).all()):raise FloatingPointError('The mediated conditional-value slope is non-finite.')
    return gradient.detach()


def fit_corrected_factor_policy(args):
    """On-policy joint learning with an unbiased differentiable control variate.

    The actor's signed interaction solve defines the mean of one scalar fusion
    displacement per boat. A displacement persists for the whole learned
    option; original candidate controllers still update at every physics step.
    For a Gaussian option a=mu+sigma*epsilon and fixed event predictor C, use
        (G-C(s,a))*grad log p(a|s) + grad_mu C(s,mu+sigma*epsilon).
    The two C terms cancel in expectation. Thus predictor errors cannot become
    a systematic policy gradient, while an accurate conditional predictor can
    reduce the noise of the true complete-team return. Every actor gradient is
    accumulated before changing its collecting policy or fitting the critic.
    In paired mode, collection and all continuations are deterministic; only
    the tested first option is perturbed. Its corrected credit estimates the
    gradient of a locally smoothed conditional value. A complete-policy check
    can reject updates that do not improve the deployed feedback policy.
    """
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    saved=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    if not saved.get('event_factors'):raise ValueError('Corrected factor learning requires a competing-event factor model.')
    policy=ResponsePolicy(args.source,args.graph_checkpoint);model=policy.model;source=policy.source.actor
    if args.execution_residual and not hasattr(model,'execution_gate'):
        model.enable_execution_residual(args.execution_base_scale)
    args.execution_residual=args.execution_residual or hasattr(model,'execution_gate')
    if args.source_device!='cpu':
        class DeviceSource(nn.Module):
            def __init__(self,actor,device):
                super().__init__();self.actor=actor.to(device);self.device=device
            def forward(self,obs,*arguments):
                action,log_probability=self.actor(obs.to(self.device),*arguments)
                return action.to(obs.device),None if log_probability is None else log_probability.to(obs.device)
        source=DeviceSource(source,args.source_device)
    critic=copy.deepcopy(model)
    if saved.get('corrected_factors'):critic.load_state_dict(saved['corrected_factors']['critic'],strict=not args.execution_residual)
    target=copy.deepcopy(critic).requires_grad_(False) if args.temporal_event_backup else None
    if target is not None and saved.get('corrected_factors',{}).get('target_critic'):
        target.load_state_dict(saved['corrected_factors']['target_critic'],strict=not args.execution_residual)
    if args.reset_factor_policy:
        for head in (model.coordination_node,model.coordination_pair):
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    relation_names=('token.','pair.','blocks.','degree.','mode_condition.','projection_')
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('execution_gate.','execution_edges.')) if args.execution_residual else
            (name.startswith(('coordination_node.','coordination_pair.')) or
             (args.fine_tune_relations and name.startswith(relation_names))))
    actor_parameters=[p for p in model.parameters() if p.requires_grad]
    actor_optimizer=torch.optim.Adam([
        dict(params=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith(('coordination_node.','coordination_pair.','execution_gate.','execution_edges.'))],lr=args.pathwise_lr),
        dict(params=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith(relation_names)],lr=.1*args.pathwise_lr)])
    value_names=('event_','factor_global.','capture_heads.','failure_heads.','value_node.','value_attention.')
    for name,p in critic.named_parameters():p.requires_grad_(name.startswith(value_names))
    value_parameters=[p for p in critic.parameters() if p.requires_grad]
    value_optimizer=torch.optim.AdamW(value_parameters,lr=args.value_lr,weight_decay=.01)
    initial_sets=[]
    for directory,count in [(args.context_source,args.train_states)]+([(args.mixed_source,args.mixed_train_states)] if args.mixed_source else []):
        packets=[]
        for root in range(count):
            if directory is None:
                with preserved_random_state():
                    seed_random(2100000000+args.seed*10000+root)
                    env=TADEnv(args.defenders,protocol='paper-parameters-v1')
                    obs,_=env.reset(args.agility,noisy_agility=False)
                    first=public_packet(env,obs)
            else:
                row=torch.load(directory/'train'/f'{root}.pth',weights_only=False);first=row['transitions'][0]
            packet={key:first[key] for key in ('obs','motion','central')};packet['intent_history']=public_intent_history([first])[0]
            packets.append(packet)
        initial_sets.append({key:torch.as_tensor(np.asarray([p[key] for p in packets]),dtype=torch.float32) for key in packets[0]})
    class ZeroValue:
        def __call__(self,p,b,e):return b.new_zeros(len(b),3),None
    history=[];started=time.perf_counter();sigma=args.exploration_sigma
    cv_scale=saved.get('corrected_factors',{}).get('next_cv_scale',1.)
    for iteration in range(1,args.iterations+1):
        torch.manual_seed(args.seed+65537*iteration);datasets=[];metrics=[]
        for packet in initial_sets:
            ix=torch.as_tensor(rng.integers(len(packet['obs']),size=args.episodes));n=packet['obs'].shape[1]
            if args.paired_policy_credit:
                batch={key:value[ix] for key,value in packet.items()}
                innovations=torch.randn(args.episodes,300,2+3*(n+1))
                exploration=torch.zeros(args.episodes,300,n,2)
            else:
                batch={key:value[ix].repeat_interleave(2,0) for key,value in packet.items()}
                innovations=torch.randn(args.episodes,300,2+3*(n+1)).repeat_interleave(2,0)
                epsilon=torch.randn(args.episodes,300,n,1).expand(-1,-1,-1,2)
                exploration=torch.stack((epsilon,-epsilon),1).flatten(0,1)
            data,cost=score_intervention_rollout(model,ZeroValue(),source,batch,innovations,exploration,
                sigma=sigma,trace=1.,option_steps=args.horizon,transitions=args.temporal_event_backup)
            datasets.append(data)
            metrics.append(dict(defenders=n,episodes=len(cost),mean_cost=float(cost.mean()),decisions=len(data['obs'])))
        paired=[]
        if args.paired_policy_credit:
            paired=[paired_factor_credits(model,source,data,rng,args.credit_roots,args.credit_repeats,sigma,args.horizon,args.mean_peer_credit) for data in datasets]
        # Freeze targets under the collecting policy. Mixing its actual tail
        # with a just-updated policy's bootstrap would describe neither policy.
        if target is not None:
            continuation_sigma=0. if args.paired_policy_credit else sigma
            for data in datasets:data['event_target']=temporal_event_targets(model,target,source,data,continuation_sigma,args.credit_trace,args.batch_size)
        actor_optimizer.zero_grad(set_to_none=True);moments=[];cv_numerator=0.;cv_denominator=0.
        before=[p.detach().clone() for p in actor_parameters]
        optimizer_before=copy.deepcopy(actor_optimizer.state_dict()) if args.paired_step_check else None
        for auxiliary in paired:
            base=auxiliary['baseline'];packet=auxiliary['packet'];samples=auxiliary['samples'].detach().requires_grad_(True)
            with torch.no_grad():factors=critic.centralized_factors(packet,base)
            theta=(base[:,None,None,:,2]+samples).clamp(0.,1.)
            prediction=critic.factor_cost(factors,theta.flatten(1,2),base[...,2]).mean(-1).reshape_as(auxiliary['returns'])
            slope=torch.autograd.grad(prediction.sum(),samples)[0]
            positive,negative=auxiliary['positive'],auxiliary['negative'];receivers=torch.arange(base.shape[1])
            delta=auxiliary['returns'][:,:,positive]-auxiliary['returns'][:,:,negative]
            predicted_delta=prediction[:,:,positive]-prediction[:,:,negative]
            derivative=.5*(slope[:,:,positive,receivers]+slope[:,:,negative,receivers])
            credit=((delta-predicted_delta.detach())*auxiliary['epsilon']/(2.*sigma)+derivative.detach()).mean(1)
            auxiliary['actor_credit']=credit.detach()
            mean=model.fusion(packet,base)[...,0]
            loss=(mean*credit.detach()).sum(-1).mean()/(20.*len(paired))
            loss.backward()
            moments.append(dict(conditional_difference_mse=float((predicted_delta.detach()-delta).square().mean()),
                conditional_zero_mse=float(delta.square().mean()),credit_rms=float(credit.square().mean().sqrt())))
        for data in ([] if paired else datasets):
            moments_one=[]
            for start in range(0,len(data['obs']),args.batch_size):
                ix=torch.arange(start,min(start+args.batch_size,len(data['obs'])))
                packet={key:data[key][ix] for key in ('obs','motion','central','intent_history')};base=data['baseline'][ix]
                with torch.no_grad():factors=critic.centralized_factors(packet,base)
                if args.mediated_control_variate:
                    gradient=mediated_option_gradient(model,critic,source,packet,data['mean'][ix,...,0],args.horizon)
                    with torch.no_grad():
                        reference_theta=(base[...,2]+data['mean'][ix,...,0]).clamp(0.,1.)
                        reference=critic.factor_cost(factors,reference_theta,base[...,2])[:,0].mean(-1)
                mean=model.fusion(packet,base)[...,0]
                collected=data['eta'][ix,...,0];noise=collected-data['mean'][ix,...,0]
                sample=mean+noise;theta=(base[...,2]+sample).clamp(0.,1.)
                if args.mediated_control_variate:
                    prediction=reference+cv_scale*(gradient*(sample-data['mean'][ix,...,0])).sum(-1)
                    with torch.no_grad():
                        score=noise/sigma**2
                        raw=(data['return'][ix]-reference)[:,None]*score
                        control=(gradient*noise).sum(-1)[:,None]*score-gradient
                        cv_numerator+=float((raw*control).sum());cv_denominator+=float(control.square().sum())
                else:prediction=critic.factor_cost(factors,theta,base[...,2])[:,0].mean(-1)
                log_probability=-.5*((collected-mean)/sigma).square().sum(-1)
                actual=data['return'][ix]
                loss=((actual-prediction.detach())*log_probability+prediction).sum()
                # A fixed control-cost regularizer is explicit in the optimizer;
                # all reported task returns remain the original capped time.
                nominal=mean[...,None]*(base[...,:2]-packet['obs'][...,12:14])
                loss+=args.trust*nominal.square().mean((1,2)).sum()
                (loss/(20.*len(data['obs'])*len(datasets))).backward()
                moments_one.append(torch.stack((actual.detach(),prediction.detach()),-1))
            observations=torch.cat(moments_one)
            moments.append(dict(return_prediction_mse=float((observations[:,0]-observations[:,1]).square().mean())))
        norm=float(nn.utils.clip_grad_norm_(actor_parameters,5.));actor_optimizer.step()
        actor_epochs=1
        # Reuse the expensive conditional credits as a local affine value
        # model. The credit is estimated at the collecting policy; subsequent
        # fits optimize that local surrogate, not a newly unbiased gradient.
        # Both the global trust region and full-policy acceptance check below
        # still apply to the entire proposed update.
        for _ in range(1,args.credit_actor_epochs if paired else 1):
            actor_optimizer.zero_grad(set_to_none=True)
            for auxiliary in paired:
                mean=model.fusion(auxiliary['packet'],auxiliary['baseline'])[...,0]
                loss=((mean-auxiliary['mean'].detach())*auxiliary['actor_credit']).sum(-1).mean()/(20.*len(paired))
                loss.backward()
            nn.utils.clip_grad_norm_(actor_parameters,5.);actor_optimizer.step();actor_epochs+=1
            with torch.no_grad():
                local_kl=max(float((.5*(model.fusion(a['packet'],a['baseline'])[...,0]-a['mean']).square().sum(-1)/sigma**2).mean()) for a in paired)
            if local_kl>=args.score_kl:break
        with torch.no_grad():
            def actual_kl():
                values=[]
                for data in datasets:
                    index=torch.arange(0,len(data['obs']),max(1,len(data['obs'])//128))
                    packet={key:data[key][index] for key in ('obs','motion','central','intent_history')}
                    changed=model.fusion(packet,data['baseline'][index])[...,0]-data['mean'][index,...,0]
                    values.append(float((.5*changed.square().sum(-1)/sigma**2).mean()))
                return values
            kl=actual_kl();applied_kl=kl
            for _ in range(6):
                largest=max(applied_kl)
                if largest<=args.score_kl:break
                scale=.9*(args.score_kl/largest)**.5
                for parameter,old in zip(actor_parameters,before):parameter.copy_(old+scale*(parameter-old))
                applied_kl=actual_kl()
            if max(applied_kl)>args.score_kl:
                for parameter,old in zip(actor_parameters,before):parameter.copy_(old)
                applied_kl=actual_kl()
        step_check=None
        if args.paired_step_check:
            step_check=check_factor_policy_step(model,source,actor_parameters,before,initial_sets,rng,
                roots=args.check_batch_size,repeats=2,option_steps=args.horizon)
            if not step_check['accepted']:
                actor_optimizer.load_state_dict(optimizer_before)
                applied_kl=[0. for _ in datasets]
        # Fit only after the complete actor gradient has been formed. The next
        # rollout therefore uses an independent, fixed control variate.
        value_loss=0.
        for _ in range(args.score_epochs):
            for data in datasets:
                order=torch.randperm(len(data['obs']))
                for start in range(0,len(order),args.batch_size):
                    ix=order[start:start+args.batch_size]
                    packet={key:data[key][ix] for key in ('obs','motion','central','intent_history')};base=data['baseline'][ix]
                    theta=(base[...,2]+data['eta'][ix,...,0]).clamp(0.,1.)
                    factors=critic.centralized_factors(packet,base)
                    cost=data['return'][ix,None,None].expand(-1,3,1);cause=data['capturer'][ix,None,None].expand(-1,3,1)
                    objective=critic.event_loss(factors,theta[:,None],base[...,2],cost,cause)
                    if target is not None:
                        log_probability=critic.factor_logits(factors,theta[:,None],base[...,2])[:,0].log_softmax(-1)
                        objective=.1*objective-.9*(data['event_target'][ix]*log_probability).sum(-1).mean()
                    prediction=critic.factor_cost(factors,theta,base[...,2])[:,0]
                    objective+=.25*((prediction-cost[...,0])/20.).square().mean()
                    value_optimizer.zero_grad(set_to_none=True);objective.backward();nn.utils.clip_grad_norm_(value_parameters,5.);value_optimizer.step()
                    if target is not None:
                        with torch.no_grad():
                            for averaged,current in zip(target.parameters(),critic.parameters()):averaged.lerp_(current,.02)
                    value_loss=float(objective.detach())
            for auxiliary in paired:
                base=auxiliary['baseline'];factors=critic.centralized_factors(auxiliary['packet'],base)
                theta=(base[:,None,None,:,2]+auxiliary['samples']).clamp(0.,1.).flatten(1,2)
                cost=auxiliary['returns'].flatten(1,2)[:,None].expand(-1,3,-1)
                cause=auxiliary['capturers'].flatten(1,2)[:,None].expand(-1,3,-1)
                objective=critic.event_loss(factors,theta,base[...,2],cost,cause,branch_limit=None)
                prediction=critic.factor_cost(factors,theta,base[...,2]).mean(-1).reshape_as(auxiliary['returns'])
                positive,negative=auxiliary['positive'],auxiliary['negative']
                truth=auxiliary['returns'][:,:,positive]-auxiliary['returns'][:,:,negative]
                difference=prediction[:,:,positive]-prediction[:,:,negative]
                objective+=(difference-truth).square().mean()/truth.square().mean().clamp_min(1.)
                value_optimizer.zero_grad(set_to_none=True);objective.backward();nn.utils.clip_grad_norm_(value_parameters,5.);value_optimizer.step()
                if target is not None:
                    with torch.no_grad():
                        for averaged,current in zip(target.parameters(),critic.parameters()):averaged.lerp_(current,.02)
        next_cv_scale=float(np.clip(cv_numerator/max(cv_denominator,1e-8),0.,1.)) if args.mediated_control_variate else cv_scale
        row=dict(iteration=iteration,rollouts=metrics,model_error=moments,gradient_norm=norm,actor_epochs=actor_epochs,proposed_kl=kl,applied_kl=applied_kl,step_check=step_check,cv_scale=cv_scale,next_cv_scale=next_cv_scale,
            value_loss=value_loss,seconds=round(time.perf_counter()-started,1));history.append(row);print(row,flush=True)
        cv_scale=next_cv_scale
    saved.update(model=model.state_dict(),seed=args.seed,horizon=args.horizon,
        execution_residual=hasattr(model,'execution_gate'),execution_base_scale=getattr(model,'execution_base_scale',1.),
        training_settings={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
        corrected_factors=dict(critic=critic.state_dict(),history=history,sigma=sigma,reset_policy=args.reset_factor_policy,
            option_steps=args.horizon,unbiased_control_variate=True,trust=args.trust,mediated_control_variate=args.mediated_control_variate,next_cv_scale=cv_scale,
            temporal_event_backup=args.temporal_event_backup,event_trace=args.credit_trace,target_critic=target.state_dict() if target is not None else None,
            paired_policy_credit=args.paired_policy_credit,credit_roots=args.credit_roots,credit_repeats=args.credit_repeats,
            continuation_policy='deterministic' if args.paired_policy_credit else 'gaussian',paired_step_check=args.paired_step_check,
            mean_peer_credit=args.mean_peer_credit,fine_tune_relations=args.fine_tune_relations,
            credit_actor_epochs=args.credit_actor_epochs,
            credit_target='smoothed-first-option-value' if args.paired_policy_credit else 'stochastic-policy-return'))
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


def fit_score_intervention(args):
    """Learn relational fusion from event-complete conditional policy credit."""
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    device=torch.device(args.device);saved=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    response_projection=args.response_projection or saved.get('response_projection',False)
    event_likelihood=args.event_likelihood or saved.get('score',{}).get('event_likelihood',False)
    projection=args.candidate_projection or saved.get('candidate_projection',False) or response_projection
    fusion_link=args.fusion_link or saved.get('fusion_link','probability')
    if fusion_link=='odds':fusion_link='probability'
    intent_saved=torch.load(args.intent_checkpoint,map_location='cpu',weights_only=True) if args.intent_checkpoint else None
    intent_refinement=saved.get('intent_refinement',False) or bool(intent_saved and intent_saved.get('intent_refinement'))
    model=CandidateInterventionModel(saved['width'],saved.get('competing_risks',False),True,True,fusion_link,projection,response_projection,intent_refinement).to(device)
    weights=dict(saved['model'])
    if weights['adapter.3.weight'].shape[0]==1:
        weights['adapter.3.weight']=weights['adapter.3.weight'].repeat(2,1)
        weights['adapter.3.bias']=weights['adapter.3.bias'].repeat(2)
    model.load_state_dict(weights,strict=False)
    if intent_saved:model.intent.load_state_dict(intent_saved['model'])
    if saved.get('fusion_link','odds')!=fusion_link or not (saved.get('pathwise') or saved.get('score')):
        nn.init.zeros_(model.adapter[-1].weight);nn.init.zeros_(model.adapter[-1].bias)
    critic=copy.deepcopy(model)
    if saved.get('score'):critic.load_state_dict(saved['score']['critic'],strict=not (projection or intent_refinement))
    if intent_saved:critic.intent.load_state_dict(intent_saved['model'])
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('adapter.','token.','pair.','blocks.','degree.','mode_condition.','projection_')))
    for name,p in critic.named_parameters():p.requires_grad_(not name.startswith(('intent.','adapter.','response.')))
    parameters=[p for p in model.parameters() if p.requires_grad]
    value_parameters=[p for p in critic.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=args.pathwise_lr)
    value_optimizer=torch.optim.Adam(value_parameters,lr=args.value_lr)
    source=OriginalPolicy(args.source).actor.to(device)
    packet_sets=[];context_sets=[]
    for directory,count in [(args.context_source,args.train_states)]+([(args.mixed_source,args.mixed_train_states)] if args.mixed_source else []):
        initial=[];contexts=[]
        for root in range(count):
            row=torch.load(directory/'train'/f'{root}.pth',weights_only=False)
            first=row['transitions'][0];p={k:first[k] for k in ('obs','motion','central')}
            p['intent_history']=public_intent_history([first])[0];initial.append(p)
            if args.score_contrast_roots:contexts.append(source_context(row))
        packet_sets.append({k:torch.as_tensor(np.asarray([p[k] for p in initial]),device=device,dtype=torch.float32) for k in initial[0]})
        if contexts:context_sets.append({k:torch.as_tensor(np.asarray([p[k] for p in contexts]),device=device,dtype=torch.float32) for k in packet_sets[-1]})
    history=[];started=time.perf_counter();steps=0
    for iteration in range(1,args.iterations+1):
        # Paired architecture experiments keep future draws aligned even if
        # earlier policies produced different numbers of surviving decisions.
        torch.manual_seed(args.seed+65537*iteration)
        datasets=[];metrics=[];contrasts=[];paired=[]
        for packet in packet_sets:
            ix=torch.as_tensor(rng.integers(len(packet['obs']),size=args.episodes),device=device)
            batch={k:v[ix].repeat_interleave(2,0) for k,v in packet.items()};n=batch['obs'].shape[1]
            noise=torch.randn(args.episodes,300,2+3*(n+1),device=device).repeat_interleave(2,0)
            exploration=torch.randn(args.episodes,300,n,2,device=device)
            exploration=torch.stack((exploration,-exploration),1).flatten(0,1)
            class ZeroValue:
                def __call__(self,p,base,eta):return base.new_zeros((len(base),3)),None
            rollout_critic=ZeroValue() if args.paired_policy_credit else critic
            data,cost=score_intervention_rollout(model,rollout_critic,source,batch,noise,exploration,sigma=args.exploration_sigma,
                trace=args.credit_trace,option_steps=args.score_option_steps)
            with torch.no_grad():reference,_=pathwise_team_cost(model,critic,source,batch,noise,adapt=False)
            scale=data['advantage'].std().clamp_min(1.)
            data['advantage']=(data['advantage']-data['advantage'].mean())/scale
            data['old_log_probability']=-.5*((data['eta']-data['mean'])/args.exploration_sigma).square().sum(-1)
            datasets.append(data);steps+=len(data['obs'])
            metrics.append(dict(defenders=n,exploratory_time_change=float((cost-reference).mean()),
                paired_standard_error=float((cost-reference).reshape(-1,2).mean(-1).std()/args.episodes**.5),fusion_decisions=len(data['obs'])))
        if args.paired_policy_credit:
            for data in datasets:
                paired.append(paired_policy_credits(model,source,data,rng,roots=args.credit_roots,repeats=args.credit_repeats,
                    sigma=args.exploration_sigma,option_steps=args.score_option_steps))
            scale=torch.cat([a['delta'] for a in paired]).square().mean().sqrt().clamp_min(1.)
            for a in paired:a['advantage']=torch.stack((.5*a['delta'],-.5*a['delta']),-1)/scale
            contrasts=paired
        else:
            for packet in context_sets:
                contrasts.append(conditional_score_contrasts(model,source,packet,rng,roots=args.score_contrast_roots,
                    repeats=args.score_contrast_repeats,sigma=args.exploration_sigma,option_steps=args.score_option_steps,radius=args.score_contrast_radius))
        actor_updates=0;last_kl=0.;last_value=0.
        for epoch in range(args.score_epochs):
            orders=[torch.randperm(len(d['obs']),device=device) for d in datasets]
            count=max((len(x)+args.batch_size-1)//args.batch_size for x in orders)
            for update in range(count):
                optimizer.zero_grad(set_to_none=True);value_optimizer.zero_grad(set_to_none=True)
                actor_loss=0.;value_loss=0.;kl=0.
                for team,(data,order) in enumerate(zip(datasets,orders)):
                    start=(update*args.batch_size)%len(order);ix=order[start:start+args.batch_size]
                    p={k:data[k][ix] for k in ('obs','motion','central','intent_history')};base=data['baseline'][ix]
                    if paired:
                        a=paired[team];actor_packet=a['packet'];actor_base=a['baseline']
                        mean=model.fusion(actor_packet,actor_base)
                        row_index=torch.arange(len(mean),device=device);receiver=a['receivers']
                        selected=mean[row_index,receiver]
                        actions=torch.stack((a['plus'][row_index,receiver],a['minus'][row_index,receiver]),1)
                        log_probability=-.5*((actions-selected[:,None])/args.exploration_sigma).square().sum(-1)
                        old_log_probability=-.5*a['perturbation'].square().sum(-1)[:,None]
                        ratio=(log_probability-old_log_probability).exp();advantage=a['advantage']
                        surrogate=mean.shape[1]*torch.maximum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                        old_mean=a['mean']
                    else:
                        actor_packet=p;actor_base=base;mean=model.fusion(p,base)
                        log_probability=-.5*((data['eta'][ix]-mean)/args.exploration_sigma).square().sum(-1)
                        log_ratio=log_probability-data['old_log_probability'][ix];ratio=log_ratio.exp()
                        advantage=data['advantage'][ix]
                        surrogate=torch.maximum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).sum(-1).mean()
                        old_mean=data['mean'][ix]
                    theta=model.fusion_weights(actor_base[...,2],mean)
                    change=model.modal_change(actor_base[...,:2],actor_packet['obs'][...,12:14],theta,actor_base[...,2])
                    actor_loss=actor_loss+(surrogate+args.trust_seconds*4.*change.square().mean())/len(datasets)
                    if event_likelihood:
                        q,_,probability=critic.consequences(p,base,data['eta'][ix])
                        prediction=-q
                        costs=data['return'][ix];remaining=60.*p['central'][:,-1]
                        times=torch.minimum(costs,remaining).clamp_min(0.)
                        bins=critic.capture_times;count_bins=len(bins);capturer=data['capturer'][ix]
                        upper=torch.searchsorted(bins,times).clamp(1,count_bins-1);lower=upper-1
                        left=torch.minimum(bins[lower],remaining);right=torch.minimum(bins[upper],remaining)
                        weight=((times-left)/(right-left).clamp_min(1e-6)).clamp(0.,1.)
                        target=torch.zeros_like(probability[:,0]);row_indices=torch.arange(len(ix),device=device)
                        receiver=capturer.clamp_min(0)*count_bins
                        target[row_indices,receiver+lower]=1.-weight;target[row_indices,receiver+upper]+=weight
                        target[capturer<0]=0.;target[capturer<0,-1]=1.
                        likelihood=-(target[:,None]*probability.clamp_min(1e-8).log()).sum(-1).mean()
                        mean_error=F.mse_loss(prediction,costs[:,None].expand_as(prediction)/20.)
                        value_loss=value_loss+(likelihood+.25*mean_error)/len(datasets)
                    else:
                        prediction=-critic(p,base,data['eta'][ix])[0]
                        value_loss=value_loss+F.smooth_l1_loss(prediction,data['return'][ix,None].expand_as(prediction)/20.)/len(datasets)
                    if contrasts:
                        auxiliary=contrasts[team];ap=auxiliary['packet'];ab=auxiliary['baseline']
                        plus=-20.*critic(ap,ab,auxiliary['plus'])[0]
                        minus=-20.*critic(ap,ab,auxiliary['minus'])[0]
                        reliability=1./(1.+(auxiliary['standard_error']/2.).square())
                        residual=(plus-minus-auxiliary['delta'][:,None])/5.
                        contrast_error=residual.square() if event_likelihood else F.smooth_l1_loss(residual,torch.zeros_like(residual),reduction='none')
                        value_loss=value_loss+args.score_contrast_weight*(contrast_error.mean(-1)*reliability).mean()/len(datasets)
                    kl=kl+float((.5*((mean-old_mean)/args.exploration_sigma).square().sum(-1)).mean().detach())/len(datasets)
                if kl<=args.score_kl:
                    actor_loss.backward();nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();actor_updates+=1
                value_loss.backward();nn.utils.clip_grad_norm_(value_parameters,1.);value_optimizer.step()
                last_kl=kl;last_value=float(value_loss.detach())
        row=dict(iteration=iteration,teams=metrics,actor_updates=actor_updates,last_kl=last_kl,value_loss=last_value,
            total_fusion_decisions=steps,elapsed_seconds=round(time.perf_counter()-started,2))
        if paired:
            row['paired_credit']=[dict(defenders=a['baseline'].shape[1],roots=len(a['roots']),repeats=args.credit_repeats,
                mean_absolute_difference=float(a['delta'].abs().mean()),mean_standard_error=float(a['standard_error'].mean()),
                coupling_second_moment=a['coupling_second_moment'],shuffled_second_moment=a['shuffled_second_moment']) for a in paired]
        if contrasts:
            with torch.no_grad():
                row['conditional_contrasts']=[dict(roots=a['roots'].tolist(),receivers=a['receivers'].tolist(),
                    measured_delta=a['delta'].tolist(),standard_error=a['standard_error'].tolist(),
                    fitted_delta=(-20.*(critic(a['packet'],a['baseline'],a['plus'])[0]-critic(a['packet'],a['baseline'],a['minus'])[0]).mean(-1)).tolist()) for a in contrasts]
                row['contrast_fit_mae']=[float(np.mean(np.abs(np.asarray(a['fitted_delta'])-a['measured_delta']))) for a in row['conditional_contrasts']]
                row['contrast_zero_mae']=[float(np.mean(np.abs(a['measured_delta']))) for a in row['conditional_contrasts']]
        history.append(row);print({k:v for k,v in row.items() if k!='conditional_contrasts'},flush=True)
    saved.update(model={k:v.detach().cpu() for k,v in model.state_dict().items()},horizon=args.score_option_steps,seed=args.seed,degree_feature=True,modal_fusion=True,fusion_link=fusion_link,candidate_projection=projection,response_projection=response_projection,intent_refinement=intent_refinement,
        score=dict(history=history,critic={k:v.detach().cpu() for k,v in critic.state_dict().items()},sigma=args.exploration_sigma,
            trace=args.credit_trace,epochs=args.score_epochs,kl_limit=args.score_kl,source=str(args.graph_checkpoint),
            task_cost='complete-capped-capture-time',credit=('paired-receiver-fusion-policy-score' if args.paired_policy_credit else 'receiver-fusion-conditional-score'),antithetic_exploration=True,
            option_steps=args.score_option_steps,contrast_roots=args.score_contrast_roots,contrast_repeats=args.score_contrast_repeats,
            contrast_radius=args.score_contrast_radius,contrast_weight=args.score_contrast_weight,
            stochastic_seed_schedule='seed+65537*iteration',event_likelihood=event_likelihood,
            paired_policy_credit=args.paired_policy_credit,credit_roots=args.credit_roots,credit_repeats=args.credit_repeats))
    saved.pop('temporal_diagnostic',None)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


def fit_candidate_intervention(args):
    """Jointly ground candidate relations and a duration-matched intervention critic.

    Whole roots, rather than correlated branches, are bootstrap units. Paired
    future covariance weights team-return contrasts; the physical contrasts
    ground attention in the effects of candidates on every recipient vessel.
    Only the public fusion adapter is subsequently optimized through the critic.
    """
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);device=torch.device(args.device)
    model=CandidateInterventionModel(args.width,args.competing_risks).to(device)
    reference=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    model.intent.load_state_dict({k.removeprefix('intent.'):v for k,v in reference['model'].items() if k.startswith('intent.')})
    model.intent.requires_grad_(False)
    if args.warm_candidate:
        model.load_state_dict(torch.load(args.warm_candidate,map_location=device,weights_only=True)['model'],strict=not args.competing_risks)
        nn.init.zeros_(model.adapter[-1].weight);nn.init.zeros_(model.adapter[-1].bias)
    def event_targets(times,capturers,n):
        times=torch.as_tensor(np.asarray(times),dtype=torch.float32,device=device)
        capturers=torch.as_tensor(np.asarray(capturers),dtype=torch.long,device=device)
        upper=torch.searchsorted(model.capture_times,times).clamp(1,len(model.capture_times)-1);lower=upper-1
        fraction=((times-model.capture_times[lower])/(model.capture_times[upper]-model.capture_times[lower])).clamp(0.,1.)
        target=torch.zeros((*times.shape,n*len(model.capture_times)+1),device=device)
        left=capturers.clamp_min(0)*len(model.capture_times)+lower;right=capturers.clamp_min(0)*len(model.capture_times)+upper
        target.scatter_add_(-1,left[...,None],((1.-fraction)*(capturers>=0))[...,None])
        target.scatter_add_(-1,right[...,None],(fraction*(capturers>=0))[...,None]);target[...,-1]=(capturers<0)
        return target
    def dataset(split,count):
        rows=[torch.load(args.output/split/f'{i}.pth',weights_only=False) for i in range(count)]
        if any(r.get('intervention_coordinates')!='log_odds' or r['horizon']!=args.horizon for r in rows):
            raise ValueError('Candidate interventions must match the deployed log-odds coordinates and duration.')
        tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
        packet={k:tensor([r['packet'][k] for r in rows]) for k in ('obs','motion','central','intent_history')}
        baseline=tensor([r['baseline'] for r in rows]);eta=tensor([r['corrections'] for r in rows])
        target,precision,response,events=[],[],[],[]
        for r in rows:
            times=np.asarray([[t['capture_time'] for t in trials] for trials in r['rows']])
            elapsed=60.*(1.-float(r['packet']['central'][-1]))
            q=-(times-elapsed)/20.;target.append(q.mean(-1))
            if args.competing_risks:
                capturers=[[t['capturer'] for t in trials] for trials in r['rows']]
                events.append(event_targets(times-elapsed,capturers,len(r['baseline'])).mean(1))
            group=q.reshape(len(q),-1,2).mean(-1);difference=group[1:]-group[:1]
            cov=np.atleast_2d(np.cov(difference,ddof=1))/group.shape[1]
            cov=.5*cov+.5*np.diag(np.diag(cov))+.25/400.*np.eye(len(difference))
            precision.append(np.linalg.inv(cov))
            response.append([np.mean([np.concatenate((t['node_response'],t['target_response']),-1) for t in trials],0) for trials in r['rows']])
        return dict(packet=packet,baseline=baseline,eta=eta,target=tensor(target),precision=tensor(precision),response=tensor(response),
            events=(torch.stack(events) if events else None))
    train,valid=dataset('train',args.train_states),dataset('development',args.test_states)
    if args.equal_context:
        # Preserve the task's context measure. Raw inverse-variance weighting
        # otherwise makes almost-no-effect contexts dominate the shared model.
        for data in (train,valid):
            value,vectors=torch.linalg.eigh(data['precision'])
            center=value.mean(-1,keepdim=True)
            value=value.clamp(min=.25*center,max=4.*center)
            value=400.*value/value.mean(-1,keepdim=True)
            data['precision']=(vectors*value[:,None])@vectors.transpose(-1,-2)
    bootstrap=torch.tensor(rng.poisson(1.,(args.train_states,3)),dtype=torch.float32,device=device)
    scale=train['response'].std((0,1,2)).clamp_min(.1)
    delta_scale=(train['response'][:,1:]-train['response'][:,:1]).std((0,1,2)).clamp_min(.05)
    def predictions(data,index=None):
        if index is None:index=torch.arange(len(data['baseline']),device=device)
        k=data['eta'].shape[1];p={name:x[index].repeat_interleave(k,0) for name,x in data['packet'].items()}
        base=data['baseline'][index].repeat_interleave(k,0);eta=data['eta'][index].flatten(0,1)
        q,r,prob=model.consequences(p,base,eta)
        return q.reshape(len(index),k,3),r.reshape(len(index),k,*r.shape[1:]),(prob.reshape(len(index),k,*prob.shape[1:]) if prob is not None else None)
    @torch.no_grad()
    def diagnose(data):
        q,response,_=predictions(data);d=(q[:,1:]-q[:,:1])*20.;target=(data['target'][:,1:]-data['target'][:,:1])*20.
        score=d.mean(-1)-.5*d.std(-1,unbiased=False)
        chosen=torch.cat((torch.zeros_like(score[:,:1]),score),-1).argmax(-1)
        true_gain=torch.cat((torch.zeros_like(target[:,:1]),target),-1)[torch.arange(len(chosen),device=device),chosen]
        rd=response[:,1:]-response[:,:1];rt=data['response'][:,1:]-data['response'][:,:1]
        return dict(difference_mae=float((d.mean(-1)-target).abs().mean()),zero_mae=float(target.abs().mean()),
            selected_gain=float(true_gain.mean()),selected_se=float(true_gain.std()/len(true_gain)**.5),active=int(torch.count_nonzero(chosen)),
            response_contrast_mse=float((rd-rt).square().mean()),response_contrast_zero=float(rt.square().mean()))
    event_history=[];factual_train=None
    if args.competing_risks:
        def factual(split,count):
            packets={k:[] for k in ('obs','motion','central','intent_history')};actions=[];times=[];capturers=[]
            for root in range(count):
                transitions=torch.load(args.state_source/split/f'{root}.pth',weights_only=False)['transitions'];last=transitions[-1]
                distances=last['next_obs'][:,2];attacker=last['next_central'][21:23]*60.
                separation=last['next_motion'][:,None,:2]-last['next_motion'][None,:,:2]
                pair_distance=np.linalg.norm(separation,axis=-1)+np.eye(len(distances))*1000.
                capture=distances.min()<5. and np.linalg.norm(attacker)>15. and pair_distance.min()>5.
                end=60.*(1.-float(last['next_central'][-1])) if capture else 60.
                histories=public_intent_history(transitions)
                for k in ('obs','motion','central'):packets[k].extend([t[k] for t in transitions])
                packets['intent_history'].extend(histories);actions.extend([t['action'] for t in transitions])
                times.extend([end-60.*(1.-float(t['central'][-1])) for t in transitions]);capturers.extend([int(distances.argmin()) if capture else -1]*len(transitions))
            tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
            return dict(packet={k:tensor(v) for k,v in packets.items()},baseline=tensor(actions),
                target=event_targets(times,capturers,len(actions[0])),times=tensor(times))
        factual_train=factual('train',args.train_states);factual_valid=factual('test',args.test_states)
        for name,p in model.named_parameters():p.requires_grad_(name.startswith(('value_node.','value_attention.','capture_heads.','failure_heads.')))
        event_optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=7e-4,weight_decay=.01)
        best_event=None;best_event_score=float('inf')
        for step in range(1,args.pretrain_updates+1):
            ix=torch.tensor(rng.integers(len(factual_train['baseline']),size=256),device=device)
            p={k:v[ix] for k,v in factual_train['packet'].items()};base=factual_train['baseline'][ix]
            q,_,prob=model.consequences(p,base,torch.zeros_like(base[...,2]))
            loss=-(factual_train['target'][ix,None]*prob.clamp_min(1e-8).log()).sum(-1).mean()
            event_optimizer.zero_grad(set_to_none=True);loss.backward();event_optimizer.step()
            if step%250==0 or step==args.pretrain_updates:
                with torch.no_grad():
                    qs=[];ps=[]
                    for start in range(0,len(factual_valid['baseline']),512):
                        p={k:v[start:start+512] for k,v in factual_valid['packet'].items()};base=factual_valid['baseline'][start:start+512]
                        q,_,prob=model.consequences(p,base,torch.zeros_like(base[...,2]));qs.append(q.mean(-1));ps.append(prob.mean(1))
                    q=torch.cat(qs);prob=torch.cat(ps)
                    row=dict(event_step=step,time_mae=float((-20.*q-factual_valid['times']).abs().mean()),
                        event_nll=float(-(factual_valid['target']*prob.clamp_min(1e-8).log()).sum(-1).mean()))
                event_history.append(row);print(row,flush=True)
                if row['time_mae']<best_event_score:best_event_score=row['time_mae'];best_event={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_event)
        for name,p in model.named_parameters():p.requires_grad_(not name.startswith(('adapter.','intent.')))
    critic_parameters=[p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('adapter.')]
    optimizer=torch.optim.AdamW(critic_parameters,lr=3e-4,weight_decay=.01)
    best_score=float('inf');best=None;history=[]
    for step in range(1,args.updates+1):
        ix=torch.tensor(rng.integers(args.train_states,size=min(args.batch_size,24)),device=device)
        q,response,prob=predictions(train,ix)
        error=q[:,1:]-q[:,:1]-(train['target'][ix,1:]-train['target'][ix,:1])[...,None]
        per_head=torch.einsum('bch,bcd,bdh->bh',error,train['precision'][ix],error)/error.shape[1]
        loss=(per_head*bootstrap[ix]).mean()+.2*((q-train['target'][ix,...,None]).square().mean(1)*bootstrap[ix]).mean()
        if args.competing_risks:
            # The same paired futures identify recipient/time redistribution.
            # Factual anchors retain the source policy's calibrated survival law.
            ce=-(train['events'][ix,:,None]*prob.clamp_min(1e-8).log()).sum(-1).mean(1)
            cumulative=prob[...,:-1].reshape(*prob.shape[:-1],train['baseline'].shape[1],-1).sum(-2).cumsum(-1)
            target_cdf=train['events'][ix,...,:-1].reshape(len(ix),-1,train['baseline'].shape[1],len(model.capture_times)).sum(-2).cumsum(-1)
            cdf_delta=cumulative[:,1:]-cumulative[:,:1]-(target_cdf[:,1:]-target_cdf[:,:1])[:,:,None]
            loss=.1*loss+(ce*bootstrap[ix]).mean()+2.*cdf_delta.square().mean()
            fx=torch.tensor(rng.integers(len(factual_train['baseline']),size=64),device=device)
            fp={k:v[fx] for k,v in factual_train['packet'].items()};fb=factual_train['baseline'][fx]
            _,_,fprob=model.consequences(fp,fb,torch.zeros_like(fb[...,2]))
            loss+=.5*-(factual_train['target'][fx,None]*fprob.clamp_min(1e-8).log()).sum(-1).mean()
        target=train['response'][ix]
        loss+=args.response_weight*(((response-target)/scale).square().mean()+
            ((response[:,1:]-response[:,:1]-target[:,1:]+target[:,:1])/delta_scale).square().mean())
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(critic_parameters,5.);optimizer.step()
        if step%200==0 or step==args.updates:
            row=dict(step=step,**diagnose(valid));history.append(row);print(row,flush=True)
            score=row['difference_mae']+.2*(row['response_contrast_mse']/max(row['response_contrast_zero'],1e-8))
            if score<best_score:best_score=score;best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best);critic=diagnose(valid)
    model.requires_grad_(False);model.adapter.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=3e-4,weight_decay=.01)
    actor_history=[]
    for step in range(1,args.actor_updates+1):
        ix=torch.tensor(rng.integers(args.train_states,size=args.batch_size),device=device)
        p={k:v[ix] for k,v in train['packet'].items()};base=train['baseline'][ix]
        eta=model.fusion(p,base)
        with torch.no_grad():q0=model(p,base,torch.zeros_like(eta))[0]
        q,_=model(p,base,eta);advantage=20.*(q-q0)
        prior=base[...,2];numerator=prior*eta.exp();theta=numerator/(1.-prior+numerator)
        control=(theta-prior)[...,None]*(base[...,:2]-p['obs'][...,12:14])
        loss=-(advantage.mean(-1)-.5*advantage.std(-1,unbiased=False)).mean()
        loss+=args.trust_seconds*4.*control.square().mean()+.01*eta.square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.adapter.parameters(),2.);optimizer.step()
        if step%100==0 or step==args.actor_updates:
            with torch.no_grad():
                eta=model.fusion(valid['packet'],valid['baseline']);q=model(valid['packet'],valid['baseline'],eta)[0]
                q0=model(valid['packet'],valid['baseline'],torch.zeros_like(eta))[0];gain=20.*(q-q0)
                # Descriptive critic estimates only, never evidence of task gains.
                row=dict(actor_step=step,predicted_gain=float(gain.mean()),eta_rms=float(eta.square().mean().sqrt()),
                    head_disagreement=float(gain.std(-1,unbiased=False).mean()))
            actor_history.append(row);print(row,flush=True)
    saved=dict(model={k:v.cpu() for k,v in model.state_dict().items()},width=args.width,candidate_attention=True,
        temporal=True,horizon=args.horizon,readout=dict(link='odds',prior_floor=0.),seed=args.seed,
        critic_diagnostics=critic,critic_history=history,actor_history=actor_history,trust_seconds=args.trust_seconds,
        graph_source=str(args.graph_checkpoint),source_policy=str(args.source))
    saved['equal_context']=args.equal_context
    saved['competing_risks']=args.competing_risks;saved['event_history']=event_history
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


def fit_conditional_value(args):
    """GLS for correlated paired returns, including learned value curvature."""
    torch.set_num_threads(1);rng=np.random.default_rng(args.seed)
    saved=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    model=ResponseGraph(saved['width'],task_response=True,distributional=saved.get('distributional',False),
        mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),
        temporal=saved.get('temporal',False),cooperation=saved['cooperation']).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    def dataset(split,count):
        rows=[torch.load(args.output/split/f'{i}.pth',weights_only=False) for i in range(count)]
        if any(r.get('intervention_coordinates')!='log_odds' or r['horizon']!=args.horizon or r['intervention_radius']!=args.radius for r in rows):
            raise ValueError('The conditional critic and deployed intervention must use the same coordinates and duration.')
        keys=('obs','motion','central','intent_history')
        packet={k:torch.tensor(np.asarray([r['packet'][k] for r in rows]),dtype=torch.float32) for k in keys}
        base=torch.tensor(np.asarray([r['baseline'] for r in rows]),dtype=torch.float32)
        eta=torch.tensor(np.asarray([r['corrections'] for r in rows]),dtype=torch.float32)
        means,covariances=[],[]
        for r in rows:
            q=-np.asarray([[t['capture_time'] for t in trials] for trials in r['rows']])
            group=q.reshape(len(q),-1,2).mean(-1)
            difference=group[1:]-group[:1]
            means.append(difference.mean(-1))
            covariance=np.atleast_2d(np.cov(difference,ddof=1))/group.shape[1]
            covariance=.5*covariance+.5*np.diag(np.diag(covariance))+.5*np.eye(len(difference))
            covariances.append(covariance)
        node,pair=conditional_value_features(model,packet,base)
        return dict(packet=packet,baseline=base,eta=eta,node=node,pair=pair,
            y=np.asarray(means),covariance=np.asarray(covariances))
    train,valid=dataset('train',args.train_states),dataset('development',args.test_states)
    def transform(features,keep):
        flat=features.reshape(-1,features.shape[-1]);mean=flat.mean(0);scale=flat.std(0).clamp_min(.05)
        normalized=(flat-mean)/scale
        _,singular,vt=torch.linalg.svd(normalized[:,keep:],full_matrices=False)
        count=min(args.readout_dim,len(singular));projection=torch.zeros(flat.shape[-1],keep+count)
        projection[:keep,:keep]=torch.eye(keep)
        projection[keep:,keep:]=vt[:count].T/(singular[:count]/np.sqrt(len(flat))).clamp_min(.3)
        return dict(mean=mean,scale=scale,projection=projection)
    n=train['baseline'].shape[1];mask=torch.triu(torch.ones(n,n,dtype=torch.bool),1)
    nt=transform(train['node'],14);pt=transform(train['pair'][:,mask],7)
    if args.kernel:
        for feature,t in ((train['node'],nt),(train['pair'][:,mask],pt)):
            flat=feature.reshape(-1,feature.shape[-1]);centers=((flat-t['mean'])/t['scale'])@t['projection']
            kernel=torch.exp(-torch.cdist(centers,centers).square()/(2.*centers.shape[-1]))
            eigen,vectors=torch.linalg.eigh(kernel.double());keep=eigen>1e-5
            t.update(kernel_centers=centers,kernel_projection=(vectors[:,keep]/eigen[keep].sqrt()).float())
    def design(data):
        node,pair=contrast_transform(data['node'],nt),contrast_transform(data['pair'],pt)
        return torch.stack([conditional_design(node,pair,data['eta'][:,j],args.radius) for j in range(1,data['eta'].shape[1])],1).double().numpy()
    x,vx=design(train),design(valid);y=train['y'];precision=np.linalg.inv(train['covariance'])
    normal=np.einsum('bcf,bcd,bdg->bfg',x,precision,x);rhs=np.einsum('bcf,bcd,bd->bf',x,precision,y)
    folds=np.empty(len(x),dtype=int);folds[rng.permutation(len(x))]=np.arange(len(x))%4
    def solve(indices,alpha,weight=None):
        a,b=normal[indices],rhs[indices]
        if weight is not None:a=a*weight[:,None,None];b=b*weight[:,None]
        return np.linalg.solve(a.sum(0)+alpha*np.eye(x.shape[-1]),b.sum(0))
    scores=[]
    for alpha in ((.1,1.,10.,100.) if args.kernel else (1.,10.,100.,1000.)):
        pred=np.zeros_like(y)
        for fold in range(4):pred[folds==fold]=x[folds==fold]@solve(folds!=fold,alpha)
        scores.append(dict(alpha=alpha,mse=float(np.mean((pred-y)**2)),zero_mse=float(np.mean(y*y))))
    alpha=min(scores,key=lambda r:r['mse'])['alpha']
    weights=np.stack([solve(np.arange(len(x)),alpha,rng.poisson(1.,len(x))) for _ in range(16)])
    estimate=(vx@weights.T).mean(-1);pred=np.c_[np.zeros(len(vx)),estimate];actual=np.c_[np.zeros(len(vx)),valid['y']]
    chosen=pred.argmax(-1)
    diagnostics=dict(cross_validation=scores,alpha=alpha,development_difference_mae=float(np.abs(estimate-valid['y']).mean()),
        zero_mae=float(np.abs(valid['y']).mean()),development_selected_gain=float(actual[np.arange(len(vx)),chosen].mean()))
    saved.update(conditional_value=dict(node_transform=nt,pair_transform=pt,weights=torch.tensor(weights,dtype=torch.float32),
        radius=args.radius,trust_seconds=args.trust_seconds,risk=.5,diagnostics=diagnostics),
        readout=dict(link='odds',prior_floor=0.),local_contrast=None,response_value_actor=None,
        horizon=args.horizon,seed=args.seed,graph_source=str(args.graph_checkpoint))
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved);print(diagnostics,flush=True)


@torch.no_grad()
def continuation_features(model, packet, baseline):
    """SO(2)-invariant privileged state, attached to its public response node."""
    node = model.encode(packet,baseline)[0]
    public = model.node_inputs(packet,baseline)
    b,n = baseline.shape[:2]
    central = packet['central']
    states = central[:,:7*(n+1)].reshape(b,n+1,7)
    direction = F.normalize(states[:,-1,:2],dim=-1,eps=1e-6)
    c,s = direction[:,0,None],direction[:,1,None]
    state = states.clone()
    for first,second in ((0,1),(4,5)):
        state[...,first] = c*states[...,first]+s*states[...,second]
        state[...,second] = -s*states[...,first]+c*states[...,second]
    state[...,2] = c*states[...,2]-s*states[...,3]
    state[...,3] = c*states[...,3]+s*states[...,2]
    attacker = state[:,-1:,].expand(-1,n,-1)
    extra = central[:,-2:,None].transpose(1,2).expand(-1,n,-1)
    order = torch.arange(1,n+1,dtype=node.dtype,device=node.device)[None,:,None].expand(b,-1,-1)/7.
    return torch.cat((node,public,state[:,:n],attacker,extra,order),-1)


class ContinuationValue(nn.Module):
    def __init__(self,inputs):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(inputs,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.attention = nn.Linear(64,1)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(192,96),nn.SiLU(),nn.Linear(96,1)) for _ in range(3)])

    def forward(self,features,remaining):
        nodes = self.node(features)
        attention = self.attention(nodes).softmax(1)
        pooled = torch.cat((nodes.mean(1),nodes.amax(1),(nodes*attention).sum(1)),-1)
        return torch.cat([head(pooled) for head in self.heads],-1).sigmoid()*remaining[:,None]/20.


def fit_continuation(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    saved = torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    graph = ResponseGraph(saved['width'],task_response=saved.get('task_response',False),
                          distributional=saved.get('distributional',False),mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),temporal=saved.get('temporal',False),cooperation=saved.get('cooperation',False)).eval().requires_grad_(False)
    graph.load_state_dict(saved['model'])
    def load(split,count):
        rows,goals,costs,root_ids = [],[],[],[]
        for root in range(count):
            transitions = torch.load(args.state_source/split/f'{root}.pth',weights_only=False)['transitions']
            last = transitions[-1]
            attacker = last['next_central'][21:23]*60.
            capture = last['next_obs'][:,2].min()<5. and np.linalg.norm(attacker)>15.
            end = (1.-float(last['next_central'][-1]))*60. if capture else 60.
            for t in transitions:
                remaining = float(t['central'][-1])*60.
                goals.append((end-(60.-remaining))/20.)
                costs.append((remaining if bool(t['done'][0]) and not capture else .2)/20.)
                root_ids.append(root)
                rows.append(t)
        tensors = {k:torch.as_tensor(np.stack([r[k] for r in rows]),dtype=torch.float32)
                   for k in rows[0]}
        packet = {k:tensors[k] for k in ('obs','motion','central')}
        next_packet = {k:tensors['next_'+k] for k in packet}
        def encode(p,a):
            return torch.cat([continuation_features(graph,subset(p,slice(i,i+512)),a[i:i+512])
                              for i in range(0,len(a),512)])
        return dict(features=encode(packet,tensors['action']),following=encode(next_packet,tensors['next_action']),
                    remaining=60.*tensors['central'][:,-1],next_remaining=60.*tensors['next_central'][:,-1],
                    target=torch.tensor(goals),cost=torch.tensor(costs),done=tensors['done'][:,0],
                    roots=torch.tensor(root_ids))
    train,valid = load('train',args.train_states),load('test',args.test_states)
    model = ContinuationValue(train['features'].shape[-1])
    target = copy.deepcopy(model).requires_grad_(False)
    weights = torch.tensor(rng.poisson(1.,(args.train_states,3)),dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    history = []
    for step in range(1,args.updates+1):
        ix = torch.tensor(rng.integers(len(train['target']),size=256))
        value = model(train['features'][ix],train['remaining'][ix])
        with torch.no_grad():
            td = train['cost'][ix,None]+(1.-train['done'][ix,None])*target(train['following'][ix],train['next_remaining'][ix])
        mc = (value-train['target'][ix,None]).square()
        bellman = (value-td).square()
        loss = ((mc+.5*bellman)*weights[train['roots'][ix]]).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        with torch.no_grad():
            for t,p in zip(target.parameters(),model.parameters()): t.lerp_(p,.02)
        if step%250==0 or step==args.updates:
            with torch.no_grad():
                pred = torch.cat([model(valid['features'][i:i+512],valid['remaining'][i:i+512]).mean(-1)
                                  for i in range(0,len(valid['target']),512)])
                row = dict(step=step,mc_loss=float(mc.mean()),td_loss=float(bellman.mean()),
                           development_time_mae=float(20.*(pred-valid['target']).abs().mean()))
            history.append(row);print(row,flush=True)
    saved['graph_source'] = str(args.graph_checkpoint)
    saved['continuation'] = dict(model=model.state_dict(),inputs=train['features'].shape[-1],history=history,
        training_episodes=args.train_states,training_transitions=len(train['target']),seed=args.seed)
    atomic_torch(args.output/f'continuation-{args.seed}.pth',saved)


def fit_paired_continuation(args):
    """Calibrate successor-state value differences with matched full outcomes."""
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    saved=torch.load(args.continuation_checkpoint,map_location='cpu',weights_only=True)
    graph=ResponseGraph(saved['width'],**{k:saved.get(k,False) for k in
        ('task_response','distributional','mediated','physical_mediator','temporal','cooperation')}).eval().requires_grad_(False)
    graph.load_state_dict(saved['model'])
    value=ContinuationValue(saved['continuation']['inputs']).eval().requires_grad_(False)
    value.load_state_dict(saved['continuation']['model']);source=OriginalPolicy(args.source)
    for head in value.heads:head[-1].requires_grad_(True)
    if args.full_value:value.requires_grad_(True)
    anchors=[(head[-1].weight.detach().clone(),head[-1].bias.detach().clone()) for head in value.heads]
    full_anchors={k:v.detach().clone() for k,v in value.named_parameters()}
    def load(split,count):
        data=[]
        with torch.no_grad():
            for root in range(count):
                row=torch.load(args.output/split/f'{root}.pth',weights_only=False)
                trials=[t for rows in row['rows'] for t in rows];n=len(row['baseline'])
                packet={k:torch.tensor(np.stack([t['intervention_next'][k] for t in trials]),dtype=torch.float32) for k in ('obs','motion','central')}
                action=source.actor(packet['obs'].flatten(0,1),True,False)[0].reshape(len(trials),n,3)
                encoded=continuation_features(graph,packet,action);nodes=value.node(encoded)
                attention=value.attention(nodes).softmax(1)
                pooled=torch.cat((nodes.mean(1),nodes.amax(1),(nodes*attention).sum(1)),-1)
                hidden=torch.stack([head[:-1](pooled) for head in value.heads],1)
                finished=torch.tensor([t['steps']<=row['horizon'] for t in trials])
                actual=torch.tensor([t['capture_time'] for t in trials],dtype=torch.float32)
                remaining=60.*packet['central'][:,-1]
                offset=torch.where(finished,actual,60.-remaining)
                scale=torch.where(finished,torch.zeros_like(remaining),remaining)
                choices,repeats=len(row['rows']),row['repeats']
                grouped=actual.reshape(choices,-1,2).mean(-1).numpy()
                contrast=grouped[1:]-grouped[:1]
                covariance=np.cov(contrast,ddof=1)/grouped.shape[1]
                covariance=.5*covariance+.5*np.diag(np.diag(covariance))+.5*np.eye(len(contrast))
                data.append(dict(encoded=encoded,hidden=hidden,pooled=pooled,offset=offset,scale=scale,choices=choices,repeats=repeats,
                    actual=actual,
                    target=actual.reshape(choices,repeats).mean(-1),precision=torch.tensor(np.linalg.inv(covariance),dtype=torch.float32)))
        return data
    train,valid=load('train',args.train_states),load('development',args.test_states)
    if args.differential_decoder:
        fit_differential_decoder(args,saved,value,train,valid)
        return
    def predict(row):
        if args.full_value:
            time=row['offset'][:,None]+20.*value(row['encoded'],row['scale'])
        else:
            logits=torch.cat([head[-1](row['hidden'][:,i]) for i,head in enumerate(value.heads)],-1)
            time=row['offset'][:,None]+row['scale'][:,None]*logits.sigmoid()
        return time.reshape(row['choices'],row['repeats'],3).mean(1)
    def metrics():
        errors,zeros,gains=[],[],[]
        with torch.no_grad():
            for row in valid:
                p=predict(row).mean(-1);target=row['target']
                delta=p[1:]-p[0];truth=target[1:]-target[0]
                errors.append((delta-truth).abs().mean().item());zeros.append(truth.abs().mean().item())
                gains.append((target[0]-target[p.argmin()]).item())
        return dict(difference_mae=float(np.mean(errors)),zero_mae=float(np.mean(zeros)),selected_gain=float(np.mean(gains)))
    initial=metrics();print(dict(step=0,**initial),flush=True)
    optimizer=torch.optim.Adam([p for p in value.parameters() if p.requires_grad],lr=(1e-4 if args.full_value else 1e-3))
    bootstrap=torch.tensor(rng.poisson(1.,(len(train),3)),dtype=torch.float32)
    best,best_error,selected={k:v.detach().clone() for k,v in value.state_dict().items()},initial['difference_mae'],0
    history=[]
    for step in range(1,args.updates+1):
        indices=rng.integers(len(train),size=8);loss=0.
        for index in indices:
            row=train[index];p=predict(row);target=row['target']
            error=(p[1:]-p[:1])-(target[1:]-target[:1])[:,None]
            paired=torch.einsum('ch,cd,dh->h',error,row['precision'],error)/(row['choices']-1)
            baseline=.02*(p[0]-target[0]).square()
            loss=loss+((paired+baseline)*bootstrap[index]).mean()/len(indices)
            if args.full_value:
                prediction=row['offset'][:,None]+20.*value(row['encoded'],row['scale'])
                loss=loss+.02*((prediction-row['actual'][:,None]).square()*bootstrap[index]).mean()/len(indices)
        if args.full_value:
            loss=loss+sum((p-full_anchors[k]).square().sum() for k,p in value.named_parameters())
        else:
            loss=loss+10.*sum((head[-1].weight-w).square().sum()+(head[-1].bias-b).square().sum()
                for head,(w,b) in zip(value.heads,anchors))
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(value.parameters(),5.);optimizer.step()
        if step%50==0 or step==args.updates:
            row=dict(step=step,loss=float(loss.detach()),**metrics());history.append(row);print(row,flush=True)
            if row['difference_mae']<best_error:
                best_error,selected=row['difference_mae'],step;best={k:v.detach().clone() for k,v in value.state_dict().items()}
    saved['continuation'].update(model=best,paired_history=history,paired_initial=initial,paired_selected_step=selected)
    atomic_torch(args.output/f'paired-continuation-{args.seed}.pth',saved)


class DifferentialValue(nn.Module):
    def __init__(self,inputs,heads=3):
        super().__init__();self.inputs=inputs
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(2*inputs,64),nn.SiLU(),nn.Linear(64,inputs)) for _ in range(heads)])
        for head in self.heads:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    def forward(self,reference,intervention):
        difference=intervention-reference
        symmetric=torch.cat(((reference+intervention)*.5,difference.abs()),-1)
        return torch.stack([(head(symmetric)*difference).sum(-1)/np.sqrt(self.inputs) for head in self.heads],-1)


def fit_differential_decoder(args,saved,value,train,valid):
    """A nonlinear antisymmetric costate field for paired successor states."""
    rng=np.random.default_rng(args.seed)
    flat=torch.cat([r['pooled'] for r in train]);mean,scale=flat.mean(0),flat.std(0).clamp_min(.05)
    for row in train+valid:
        state=(row['pooled']-mean)/scale
        state=state.reshape(row['choices'],row['repeats'],-1)
        row['reference']=state[:1].expand_as(state).reshape(-1,state.shape[-1]);row['intervention']=state.flatten(0,1)
        with torch.no_grad():
            logits=torch.cat([head[-1](row['hidden'][:,i]) for i,head in enumerate(value.heads)],-1)
            time=row['offset'][:,None]+row['scale'][:,None]*logits.sigmoid()
            time=time.reshape(row['choices'],row['repeats'],3)
            row['base_difference']=(time-time[:1]).mean(1)
    model=DifferentialValue(flat.shape[-1]);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.02)
    def predict(row):
        residual=model(row['reference'],row['intervention']).reshape(row['choices'],row['repeats'],3).mean(1)
        return row['base_difference']+residual
    def metrics():
        errors,zeros,gains=[],[],[]
        with torch.no_grad():
            for row in valid:
                p=predict(row).mean(-1);truth=row['target']-row['target'][0]
                errors.append((p[1:]-truth[1:]).abs().mean().item());zeros.append(truth[1:].abs().mean().item())
                gains.append((-truth[p.argmin()]).item())
        return dict(difference_mae=float(np.mean(errors)),zero_mae=float(np.mean(zeros)),selected_gain=float(np.mean(gains)))
    initial=metrics();print(dict(step=0,**initial),flush=True)
    best,best_score,selected,history=None,float('inf'),0,[]
    bootstrap=torch.tensor(rng.poisson(1.,(len(train),3)),dtype=torch.float32)
    for step in range(1,args.updates+1):
        loss=0.
        for index in rng.integers(len(train),size=8):
            row=train[index];prediction=predict(row)
            error=prediction[1:]-(row['target'][1:]-row['target'][0])[:,None]
            loss=loss+(torch.einsum('ch,cd,dh->h',error,row['precision'],error)*bootstrap[index]).mean()/(8.*(row['choices']-1))
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%50==0 or step==args.updates:
            row=dict(step=step,loss=float(loss.detach()),**metrics());history.append(row);print(row,flush=True)
            if row['difference_mae']<best_score:
                best_score,selected=row['difference_mae'],step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
    saved['differential_value']=dict(model=best,inputs=flat.shape[-1],mean=mean,scale=scale,
        history=history,selected_step=selected,initial=initial,source=str(args.continuation_checkpoint))
    atomic_torch(args.output/f'differential-value-{args.seed}.pth',saved)


@torch.no_grad()
def continuation_targets(args, graph, saved, split, count):
    value = ContinuationValue(saved['inputs']).eval()
    value.load_state_dict(saved['model'])
    policy = OriginalPolicy(args.source)
    outcomes = []
    for root in range(count):
        row = torch.load(args.output/split/f'{root}.pth',weights_only=False)
        elapsed = 60.*(1.-float(row['packet']['central'][-1]))
        choices = []
        for trials in row['rows']:
            estimates = []
            for trial in trials:
                if trial['steps']<=row['horizon']:
                    estimate = trial['capture_time']-elapsed
                else:
                    packet = {k:v[None] for k,v in tensor_packet(trial['intervention_next']).items()}
                    action = torch.as_tensor(policy.choose_action(trial['intervention_next'],deterministic=True)[0])[None]
                    remaining = 60.*packet['central'][:,-1]
                    estimate = row['horizon']*.2+float(20.*value(continuation_features(graph,packet,action),remaining).mean())
                estimates.append(estimate+elapsed)
            choices.append(estimates)
        outcomes.append(choices)
    return np.asarray(outcomes,dtype=np.float64)


def response_value_data(args,graph,value,split,count):
    """The same counterfactual futures provide transition and value labels."""
    data = load_rows(args.output/split,count,'cpu')
    if not data['complete_returns']:
        raise ValueError('This task-value experiment requires complete continuation outcomes.')
    policy = OriginalPolicy(args.source)
    futures,stops,costs,continuations = [],[],[],[]
    with torch.no_grad():
        for root in range(count):
            row = torch.load(args.output/split/f'{root}.pth',weights_only=False)
            root_time = 60.*(1.-float(row['packet']['central'][-1]))
            all_features,all_stops,all_costs,all_values = [],[],[],[]
            for trials in row['rows']:
                packets = {k:torch.stack([torch.as_tensor(t['intervention_next'][k],dtype=torch.float32) for t in trials])
                           for k in ('obs','motion','central')}
                actions = torch.as_tensor(np.stack([policy.choose_action(t['intervention_next'],deterministic=True)[0] for t in trials]))
                encoded = continuation_features(graph,packets,actions)
                stop = torch.tensor([float(t['steps']<=row['horizon']) for t in trials])
                cost = torch.tensor([(t['capture_time']-root_time if t['steps']<=row['horizon'] else row['horizon']*.2) for t in trials])
                tail = 20.*value(encoded,60.*packets['central'][:,-1]).mean(-1)
                all_features.append(encoded.mean(0)); all_stops.append(stop.mean()); all_costs.append(cost.mean())
                all_values.append((cost+(1.-stop)*tail).mean())
            futures.append(torch.stack(all_features));stops.append(torch.stack(all_stops))
            costs.append(torch.stack(all_costs));continuations.append(torch.stack(all_values))
        data.update(future=torch.stack(futures),stop=torch.stack(stops),cost=torch.stack(costs),
                    continuation=torch.stack(continuations),
                    state=continuation_features(graph,data['packet'],data['baseline']))
        # Ordered source/receiver pairs retain which control perturbs which
        # vessel, instead of pooling all three actions before prediction.
        proxy = type('RelationInputs',(),{'peer_candidates':True})()
        data['edge'] = InteractionActor.relation_features(proxy,data['packet']['motion'],
            data['baseline'][...,:2],data['packet']['obs'][...,12:14])
    return data


class ResponseTransition(nn.Module):
    def __init__(self,inputs):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(inputs,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.edge = nn.Sequential(nn.Linear(143,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.base = nn.Linear(64,inputs)
        self.own_effect = nn.Linear(64,inputs)
        self.peer_effect = nn.Linear(64,inputs)
        self.joint_effect = nn.Linear(64,inputs)
        self.events = nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,2))

    def forward(self,state,edge,change):
        nodes = self.node(state)
        n = state.shape[1]
        sender = nodes[:,:,None].expand(-1,-1,n,-1)
        receiver = nodes[:,None].expand(-1,n,-1,-1)
        pair = self.edge(torch.cat((sender,receiver,edge),-1))
        mask = (~torch.eye(n,dtype=torch.bool,device=state.device))[None,:,:,None]
        own,peer = change[:,:,None,None],change[:,None,:,None]
        influence = (self.peer_effect(pair)*peer+self.joint_effect(pair)*own*peer)*mask
        future = state+self.base(nodes)+self.own_effect(nodes)*change[...,None]+influence.sum(2)/(n-1)
        context = (pair*mask).sum((1,2))/(n*(n-1))
        # Events condition on the predicted response, not a direct gate path.
        response = self.node(future)
        event = self.events(torch.cat((response.mean(1),context),-1))
        return future,event[:,0].sigmoid(),F.softplus(event[:,1])


def fit_response_value(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    saved = torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    graph = ResponseGraph(saved['width'],task_response=saved.get('task_response',False),
                          distributional=saved.get('distributional',False),mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),temporal=saved.get('temporal',False),cooperation=saved.get('cooperation',False)).eval().requires_grad_(False)
    graph.load_state_dict(saved['model'])
    continuation = torch.load(args.continuation_checkpoint,map_location='cpu',weights_only=True)['continuation']
    value = ContinuationValue(continuation['inputs']).eval().requires_grad_(False)
    value.load_state_dict(continuation['model'])
    train = response_value_data(args,graph,value,'train',args.train_states)
    valid = response_value_data(args,graph,value,'development',args.test_states)
    model = ResponseTransition(train['state'].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    scale = train['future'].reshape(-1,train['future'].shape[-1]).std(0).clamp_min(.05)
    def predict(state,edge,correction,remaining):
        future,stop,cost = model(state,edge,correction/args.intervention_radius)
        duration = args.horizon*.2
        next_remaining = (remaining-duration).clamp_min(0.)
        q = cost+(1.-stop)*20.*value(future,next_remaining).mean(-1)
        return q,future,stop,cost
    history = []
    for step in range(1,args.updates+1):
        ix = torch.tensor(rng.integers(args.train_states,size=64))
        choice = torch.tensor(rng.integers(1,9,size=64))
        state,edge,remaining = train['state'][ix],train['edge'][ix],60.*train['packet']['central'][ix,-1]
        q0,f0,s0,c0 = predict(state,edge,train['correction'][ix,0],remaining)
        q,f,s,c = predict(state,edge,train['correction'][ix,choice],remaining)
        next_target = train['future'][ix,choice]
        effect = (next_target-train['future'][ix,0])/scale
        response = F.mse_loss((f-f0)/scale,effect)+.1*F.mse_loss((f-next_target)/scale,torch.zeros_like(f))
        events = F.binary_cross_entropy(s,train['stop'][ix,choice])+F.mse_loss(c/20.,train['cost'][ix,choice]/20.)
        values = F.mse_loss((q-q0)/5.,(train['continuation'][ix,choice]-train['continuation'][ix,0])/5.)
        absolute = F.mse_loss(q/20.,train['continuation'][ix,choice]/20.)
        loss = response+events+values+.1*absolute
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%250==0 or step==args.updates:
            with torch.no_grad():
                prediction = torch.stack([predict(valid['state'],valid['edge'],valid['correction'][:,i],
                    60.*valid['packet']['central'][:,-1])[0] for i in range(9)],1)
                difference = prediction-prediction[:,:1]
                truth = valid['continuation']-valid['continuation'][:,:1]
                choice = prediction.argmin(1)
                row = dict(step=step,response_loss=float(response),value_difference_mse=float((difference[:,1:]-truth[:,1:]).square().mean()),
                    zero_mse=float(truth[:,1:].square().mean()),
                    selected_actual_time=float((valid['times'].mean(-1)[torch.arange(len(choice)),choice]-valid['times'][:,0].mean(-1)).mean()))
            history.append(row);print(row,flush=True)
    model.eval().requires_grad_(False)
    features = contrast_features(graph,train['packet'],train['baseline'])[0]
    flat = features.flatten(0,1)
    mean,scale = flat.mean(0),flat.std(0).clamp_min(.05)
    _,singular,vt = torch.linalg.svd((flat-mean)/scale,full_matrices=False)
    dimension = args.readout_dim
    transform = dict(mean=mean,scale=scale,projection=vt[:dimension].T/(singular[:dimension]/np.sqrt(len(flat))).clamp_min(.2))
    features = contrast_transform(features,transform)
    weight = nn.Parameter(torch.zeros(features.shape[-1]))
    optimizer = torch.optim.Adam([weight],lr=.01)
    remaining = 60.*train['packet']['central'][:,-1]
    for _ in range(args.actor_updates):
        correction = (args.intervention_radius*torch.tanh(features@weight)).clamp(-train['baseline'][...,2],1.-train['baseline'][...,2])
        q = predict(train['state'],train['edge'],correction,remaining)[0]
        loss = q.mean()+args.trust_seconds*(correction/args.intervention_radius).square().mean()+.01*weight.square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    saved['readout'] = None;saved['local_contrast'] = None
    saved['response_value_actor'] = dict(transform=transform,weight=weight.detach(),radius=args.intervention_radius)
    saved['transition'] = dict(model=model.state_dict(),history=history,inputs=train['state'].shape[-1])
    saved['continuation_source'] = str(args.continuation_checkpoint)
    saved['horizon'] = args.horizon;saved['seed'] = args.seed
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


def fit_intent_refinement(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    device=torch.device(args.device);saved=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    original={k.removeprefix('intent.'):v for k,v in saved['model'].items() if k.startswith('intent.')}
    reference=TemporalIntent().to(device).eval();reference.load_state_dict(original)
    model=RefinedTemporalIntent().to(device);model.load_state_dict(original,strict=False)
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('kinematic_'))
    def dataset(directory,split,roots):
        histories=[];targets=[]
        for root in roots:
            row=torch.load(directory/split/f'{root}.pth',weights_only=False)
            transitions=row['transitions'];h=public_intent_history(transitions)
            state=np.asarray([t['central'] for t in transitions]);n=transitions[0]['obs'].shape[0]
            att=state[:,7*n:7*(n+1)];phi=np.arctan2(att[:,1],att[:,0]);c,s=np.cos(phi),np.sin(phi)
            y=np.column_stack((c*att[:,2]-s*att[:,3],c*att[:,3]+s*att[:,2],att[:,6],
                c*att[:,4]+s*att[:,5],-s*att[:,4]+c*att[:,5]))
            histories.append(h);targets.append(y)
        return (torch.as_tensor(np.concatenate(histories),dtype=torch.float32,device=device),
            torch.as_tensor(np.concatenate(targets),dtype=torch.float32,device=device))
    train=[];development=[];test=[]
    for directory,count,test_count in [(args.context_source,args.train_states,args.test_states)]+([(args.mixed_source,args.mixed_train_states,args.mixed_test_states)] if args.mixed_source else []):
        boundary=3*count//4
        train.append(dataset(directory,'train',range(boundary)))
        development.append(dataset(directory,'train',range(boundary,count)))
        test.append(dataset(directory,'test',range(test_count)))
    def objective(prediction,target,history):
        warm=(history[...,7].sum(-1)>=3).float()
        angular=2.*(1.-(prediction[:,:2]*target[:,:2]).sum(-1)).clamp_min(0.)/.2**2
        return (angular*warm).sum()/warm.sum().clamp_min(1.)+((prediction[:,2]-target[:,2])/.3).square().mean()+((prediction[:,3:]-target[:,3:])/.06).square().mean()
    @torch.no_grad()
    def assess(net,sets):
        net.eval();results=[]
        for x,y in sets:
            prediction=torch.cat([net(x[i:i+512]) for i in range(0,len(x),512)])
            warm=x[...,7].sum(-1)>=3
            angle=torch.atan2(prediction[:,0],prediction[:,1])-torch.atan2(y[:,0],y[:,1])
            error=torch.atan2(angle.sin(),angle.cos()).abs()*180./torch.pi
            results.append(dict(score=float(objective(prediction,y,x)),heading_mae=float(error.mean()),
                warm_heading_mae=float(error[warm].mean()),yaw_rmse=float((prediction[:,2]-y[:,2]).square().mean().sqrt()),
                water_vector_rmse=float(5.*(prediction[:,3:]-y[:,3:]).square().sum(-1).mean().sqrt()),transitions=len(x)))
        return results
    reference_development=assess(reference,development)
    print(dict(reference_development=reference_development),flush=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    best=float('inf');best_weights=None;best_step=0;history=[];started=time.perf_counter()
    for step in range(1,args.updates+1):
        model.train();loss=0.
        for x,y in train:
            ix=torch.as_tensor(rng.integers(len(x),size=args.batch_size),device=device)
            loss=loss+objective(model(x[ix]),y[ix],x[ix])/len(train)
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%250==0 or step==args.updates:
            measures=assess(model,development);score=float(np.mean([m['score'] for m in measures]))
            row=dict(step=step,development=measures,elapsed_seconds=round(time.perf_counter()-started,2));history.append(row);print(row,flush=True)
            if score<best:
                best=score;best_step=step;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_weights)
    comparison=dict(reference=assess(reference,test),refined=assess(model,test))
    print(dict(selected_step=best_step,test=comparison),flush=True)
    atomic_torch(args.output/f'intent-{args.seed}.pth',dict(model=best_weights,intent_refinement=True,history=history,
        selected_step=best_step,test=comparison,source=str(args.graph_checkpoint),
        training_roots=[3*args.train_states//4]+([3*args.mixed_train_states//4] if args.mixed_source else []),
        supervision='dimensionally-scaled-heading-turn-rate-water-velocity',measurement_anchor=True))


def fit_intent(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    def dataset(split,count):
        histories,targets,naive = [],[],[]
        for root in range(count):
            source = torch.load(args.state_source/split/f'{root}.pth',weights_only=False)
            trajectory = source['transitions']
            history = public_intent_history(trajectory)
            state = np.stack([t['central'] for t in trajectory])
            n = trajectory[0]['obs'].shape[0]
            attacker = state[:,7*n:7*(n+1)]
            phi = np.arctan2(attacker[:,1],attacker[:,0]);c,s = np.cos(phi),np.sin(phi)
            target = np.column_stack((c*attacker[:,2]-s*attacker[:,3],
                c*attacker[:,3]+s*attacker[:,2],attacker[:,6],
                c*attacker[:,4]+s*attacker[:,5],-s*attacker[:,4]+c*attacker[:,5]))
            histories.append(history);targets.append(target.astype(np.float32))
            naive.append(np.arctan2(history[:,-1,1],history[:,-1,0]))
            # Attach the causal window to the existing response bank. Root
            # matching uses timestamps, never any future observations.
            bank = args.output/('train' if split=='train' else 'development')/f'{root}.pth'
            if bank.exists():
                row = torch.load(bank,weights_only=False)
                index = int(np.argmin(np.abs(state[:,-1]-row['packet']['central'][-1])))
                if abs(float(state[index,-1]-row['packet']['central'][-1]))>1e-6:
                    raise ValueError('Counterfactual root does not match its source history.')
                row['packet']['intent_history'] = history[index]
                atomic_torch(bank,row)
        return torch.tensor(np.concatenate(histories)),torch.tensor(np.concatenate(targets)),torch.tensor(np.concatenate(naive))
    x,y,_ = dataset('train',args.train_states)
    vx,vy,naive = dataset('test',args.test_states)
    model = TemporalIntent()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    history = []
    for step in range(1,args.updates+1):
        index = torch.tensor(rng.integers(len(x),size=256))
        prediction = model(x[index])
        loss = F.mse_loss(prediction[:,:2],y[index,:2])+.2*F.mse_loss(prediction[:,2:],y[index,2:])
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%250==0 or step==args.updates:
            with torch.no_grad():
                pred = torch.cat([model(vx[i:i+512]) for i in range(0,len(vx),512)])
                angle = torch.atan2(pred[:,0],pred[:,1]);truth = torch.atan2(vy[:,0],vy[:,1])
                error = torch.atan2((angle-truth).sin(),(angle-truth).cos()).abs()*180./np.pi
                naive_error = torch.atan2((naive-truth).sin(),(naive-truth).cos()).abs()*180./np.pi
                row = dict(step=step,heading_mean_degrees=float(error.mean()),heading_median_degrees=float(error.median()),
                    velocity_heading_mean_degrees=float(naive_error.mean()),
                    yaw_rate_mse=float(F.mse_loss(pred[:,2],vy[:,2])))
            history.append(row);print(row,flush=True)
    atomic_torch(args.output/f'intent-{args.seed}.pth',dict(model=model.state_dict(),history=history,
        training_episodes=args.train_states,training_transitions=len(x),sequence_length=8,seed=args.seed))


def fit_cooperation(args):
    """Learn node and edge effects from matched factorial capture geometry."""
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    saved = torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    model = ResponseGraph(saved['width'],task_response=True,distributional=saved.get('distributional',False),
        mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),
        temporal=saved.get('temporal',False),cooperation=True)
    model.load_state_dict(saved['model'],strict=False)
    for name,p in model.named_parameters(): p.requires_grad_(name.startswith('cooperation_'))
    signs = 2.*torch.tensor(list(itertools.product((0.,1.),repeat=3)))-1.
    def dataset(split,count):
        data = load_rows(args.output/split,count,'cpu')
        node_labels,pair_labels = [],[]
        for root in range(count):
            row = torch.load(args.output/split/f'{root}.pth',weights_only=False)
            obs = torch.as_tensor(row['packet']['obs'],dtype=torch.float32)
            root_vector = obs[:,2,None]*torch.stack((obs[:,3].cos(),obs[:,3].sin()),-1)
            response = torch.tensor(np.asarray([[t['target_response'] for t in trials] for trials in row['rows'][1:]]))
            distance = (root_vector[None,None]+5.*response[...,:2]).norm(dim=-1)
            own = (-distance/5.).mean(1)
            team = torch.logsumexp(-distance/5.,dim=-1).mean(1)
            self_effect = torch.stack([(own[:,i]*signs[:,i]).sum()/4. for i in range(3)])
            team_effect = (team[:,None]*signs).sum(0)/4.
            pair = torch.zeros(3,3)
            for i,j in itertools.combinations(range(3),2):
                pair[i,j] = pair[j,i] = (team*signs[:,i]*signs[:,j]).sum()/2.
            node_labels.append(torch.stack((self_effect,team_effect),-1));pair_labels.append(pair)
        data.update(node_labels=torch.stack(node_labels),pair_labels=torch.stack(pair_labels))
        return data
    train,valid = dataset('train',args.train_states),dataset('development',args.test_states)
    mask = ~torch.eye(3,dtype=torch.bool)
    field_scale = train['node_labels'].square().mean((0,1)).sqrt().clamp_min(.03)
    pair_scale = train['pair_labels'][:,mask].square().mean().sqrt().clamp_min(.01)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4,weight_decay=.01)
    best,best_score,selected,history = None,float('inf'),0,[]
    for step in range(1,args.updates+1):
        ix = torch.tensor(rng.integers(args.train_states,size=64))
        field,pair = model.cooperative_effects(subset(train['packet'],ix),train['baseline'][ix])
        loss = ((field-train['node_labels'][ix])/field_scale).square().mean()
        loss += ((pair-train['pair_labels'][ix])[:,mask]/pair_scale).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%250==0 or step==args.updates:
            with torch.no_grad():
                f,e = model.cooperative_effects(valid['packet'],valid['baseline'])
                error = (f-valid['node_labels']).square().mean((0,1))
                zero = valid['node_labels'].square().mean((0,1))
                pair_error = (e-valid['pair_labels'])[:,mask].square().mean()
                pair_zero = valid['pair_labels'][:,mask].square().mean()
                row = dict(step=step,self_mse=float(error[0]),self_zero=float(zero[0]),team_mse=float(error[1]),
                    team_zero=float(zero[1]),synergy_mse=float(pair_error),synergy_zero=float(pair_zero))
                score = float(error[1]/zero[1]+pair_error/pair_zero)
            history.append(row);print(row,flush=True)
            if score<best_score:
                best_score,selected = score,step
                best = {k:v.detach().clone() for k,v in model.state_dict().items()}
    saved.update(model=best,cooperation=True,cooperation_history=history,cooperation_selected_step=selected,
        cooperation_field_scale=field_scale,cooperation_pair_scale=pair_scale,seed=args.seed,
        readout=None,local_contrast=None,response_value_actor=None)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


def cooperative_geometry_labels(row, conditional=True):
    """Factorial marginal effects on capture proximity and escape coverage.

    These are auxiliary state consequences, never replacement task returns.
    Conditioning integrates the other boats at their current fusion priors.
    """
    n = len(row['baseline'])
    bits = torch.tensor(list(itertools.product((0.,1.),repeat=n)))
    signs = 2.*bits-1.
    obs = torch.tensor(np.asarray([[t['intervention_next']['obs'] for t in trials] for trials in row['rows'][1:]]),dtype=torch.float32)
    motion = torch.tensor(np.asarray([[t['intervention_next']['motion'] for t in trials] for trials in row['rows'][1:]]),dtype=torch.float32)
    own = (-obs[...,2]/5.).mean(1)
    close = torch.logsumexp(-obs[...,2]/5.,dim=-1).mean(1)
    bearing = motion[...,2]+obs[...,3]
    attacker = (motion[...,:2]+obs[...,2,None]*torch.stack((bearing.cos(),bearing.sin()),-1)).mean(2)
    angle = torch.arange(48)*2.*torch.pi/48.
    directions = torch.stack((angle.cos(),angle.sin()),-1)
    speed = obs[:,:,0,4].clamp(2.,8.)
    velocity = speed[:,:,None,None,None]*directions[None,None,None]-motion[:,:,:,None,3:5]
    relative = attacker[:,:,None,None]-motion[:,:,:,None,:2]
    closest_time = (-(relative*velocity).sum(-1)/velocity.square().sum(-1).clamp_min(.01)).clamp(0.,4.)
    closest = (relative+closest_time[...,None]*velocity).norm(dim=-1)
    interception = torch.sigmoid((5.-closest)/1.5)
    escape = (1.-interception).prod(2)
    coverage = -escape.mean(-1).clamp_min(1e-5).log().mean(1)
    utility = torch.stack((close,coverage),-1)
    prior = torch.as_tensor(row['baseline'][:,2]) if conditional else torch.full((n,),.5)
    probabilities = torch.where(bits.bool(),prior,1.-prior)
    fields,pairs = torch.zeros(n,3),torch.zeros(n,n,2)
    for i in range(n):
        other = [j for j in range(n) if j!=i]
        derivative = signs[:,i]*probabilities[:,other].prod(-1)
        fields[i,0] = derivative@own[:,i]
        fields[i,1:] = derivative@utility
        for j in range(i+1,n):
            remainder = [k for k in range(n) if k not in (i,j)]
            derivative_pair = signs[:,i]*signs[:,j]*probabilities[:,remainder].prod(-1)
            pairs[i,j] = pairs[j,i] = derivative_pair@utility
    return fields,pairs


def fit_coverage(args):
    """Shared relational heads across team sizes, with conditional joint effects."""
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    saved = torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    model = ResponseGraph(saved['width'],task_response=True,distributional=saved.get('distributional',False),
        mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),
        temporal=saved.get('temporal',False),cooperation='coverage')
    state = {k:v for k,v in saved['model'].items() if k in model.state_dict() and v.shape==model.state_dict()[k].shape}
    model.load_state_dict(state,strict=False)
    for name,p in model.named_parameters(): p.requires_grad_(name.startswith('cooperation_'))
    def dataset(directory,count):
        rows = [torch.load(directory/f'{i}.pth',weights_only=False) for i in range(count)]
        fields,pairs = zip(*(cooperative_geometry_labels(r) for r in rows))
        keys = ('obs','motion','central','intent_history')
        packet = {k:torch.tensor(np.asarray([r['packet'][k] for r in rows]),dtype=torch.float32) for k in keys}
        return dict(packet=packet,baseline=torch.tensor(np.asarray([r['baseline'] for r in rows])),
            fields=torch.stack(fields),pairs=torch.stack(pairs),n=len(rows[0]['baseline']))
    train = [dataset(args.output/'train',args.train_states)]
    valid = [dataset(args.output/'development',args.test_states)]
    if args.mixed_source:
        train.append(dataset(args.mixed_source/'train',args.mixed_train_states))
        valid.append(dataset(args.mixed_source/'development',args.mixed_test_states))
    field_scale = torch.cat([x['fields'].flatten(0,1) for x in train]).square().mean(0).sqrt().clamp_min(.02)
    pair_scale = torch.cat([x['pairs'][:,~torch.eye(x['n'],dtype=torch.bool)] .flatten(0,1) for x in train]).square().mean(0).sqrt().clamp_min(.005)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4,weight_decay=.02)
    best,best_score,history,selected = None,float('inf'),[],0
    for step in range(1,args.updates+1):
        data = train[(step-1)%len(train)]
        ix = torch.tensor(rng.integers(len(data['baseline']),size=args.batch_size))
        f,e = model.cooperative_effects(subset(data['packet'],ix),data['baseline'][ix])
        mask = ~torch.eye(data['n'],dtype=torch.bool)
        loss = ((f-data['fields'][ix])/field_scale).square().mean()
        loss += ((e-data['pairs'][ix])[:,mask]/pair_scale).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%250==0 or step==args.updates:
            metrics=[]
            with torch.no_grad():
                for v in valid:
                    f,e=model.cooperative_effects(v['packet'],v['baseline']);mask=~torch.eye(v['n'],dtype=torch.bool)
                    fm=(f-v['fields']).square().mean((0,1));fz=v['fields'].square().mean((0,1))
                    em=(e-v['pairs'])[:,mask].square().mean((0,1));ez=v['pairs'][:,mask].square().mean((0,1))
                    metrics.append(dict(defenders=v['n'],field_mse=fm.tolist(),field_zero=fz.tolist(),pair_mse=em.tolist(),pair_zero=ez.tolist()))
            score = np.mean([np.mean(np.array(v['field_mse'])[1:]/np.maximum(np.array(v['field_zero'])[1:],.0001))+np.mean(np.array(v['pair_mse'])/np.maximum(np.array(v['pair_zero']),.00001)) for v in metrics])
            row=dict(step=step,score=float(score),metrics=metrics);history.append(row);print(row,flush=True)
            if score<best_score:
                best_score,selected=score,step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
    saved.update(model=best,cooperation='coverage',cooperation_history=history,cooperation_selected_step=selected,
        cooperation_field_scale=field_scale,cooperation_pair_scale=pair_scale,conditional_effects=True,
        seed=args.seed,readout=None,local_contrast=None,response_value_actor=None)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


def fit_factorized(args):
    """Learn candidate-to-receiver responses; bind credit to future responsibility."""
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    saved=torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    model=ResponseGraph(saved['width'],task_response=True,distributional=saved.get('distributional',False),
        mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),
        temporal=saved.get('temporal',False),cooperation='factorized')
    model.load_state_dict(saved['model'],strict=False)
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith('cooperation_') or (args.fine_tune_relations and name.startswith(('node.','edge.','attention.','context.','intent_context.'))))
    def dataset(directory,count):
        rows=[torch.load(directory/f'{i}.pth',weights_only=False) for i in range(count)]
        n=len(rows[0]['baseline']);bits=torch.tensor(list(itertools.product((0.,1.),repeat=n)));signs=2.*bits-1.
        keys=('obs','motion','central','intent_history')
        packet={k:torch.tensor(np.asarray([r['packet'][k] for r in rows]),dtype=torch.float32) for k in keys}
        utilities,influences,fields,pairs=[],[],[],[]
        for row in rows:
            obs=torch.tensor(np.asarray([[t['intervention_next']['obs'] for t in trials] for trials in row['rows']]),dtype=torch.float32)
            utility=(-obs[...,2]/5.).mean(1)
            prior=torch.as_tensor(row['baseline'][:,2]);probability=torch.where(bits.bool(),prior,1.-prior)
            derivative=torch.stack([signs[:,i]*probability[:,[j for j in range(n) if j!=i]].prod(-1) for i in range(n)],0)
            influence=derivative@utility[1:]
            f,j=cooperative_geometry_labels(row)
            utilities.append(utility[0]);influences.append(influence);fields.append(f[:,:2]);pairs.append(j[...,0])
        return dict(packet=packet,baseline=torch.tensor(np.asarray([r['baseline'] for r in rows])),n=n,
            utility=torch.stack(utilities),influence=torch.stack(influences),fields=torch.stack(fields),pairs=torch.stack(pairs))
    train=[dataset(args.output/'train',args.train_states)];valid=[dataset(args.output/'development',args.test_states)]
    if args.mixed_source:
        train.append(dataset(args.mixed_source/'train',args.mixed_train_states));valid.append(dataset(args.mixed_source/'development',args.mixed_test_states))
    if args.adapted_source:
        train.append(dataset(args.adapted_source/'train',args.adapted_train_states));valid.append(dataset(args.adapted_source/'development',args.adapted_test_states))
    field_scale=torch.cat([x['fields'].flatten(0,1) for x in train]).square().mean(0).sqrt().clamp_min(.02)
    pair_scale=torch.cat([x['pairs'][:,~torch.eye(x['n'],dtype=torch.bool)].flatten() for x in train]).square().mean().sqrt().clamp_min(.005)
    peer_scale=torch.cat([x['influence'][:,~torch.eye(x['n'],dtype=torch.bool)].flatten() for x in train]).square().mean().sqrt().clamp_min(.01)
    utility_scale=torch.cat([(x['utility']+x['packet']['obs'][...,2]/5.).flatten() for x in train]).square().mean().sqrt().clamp_min(.05)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=4e-4,weight_decay=.02)
    best,best_score,selected,history=None,float('inf'),0,[]
    for step in range(1,args.updates+1):
        data=train[(step-1)%len(train)];ix=torch.tensor(rng.integers(len(data['baseline']),size=args.batch_size))
        packet,base=subset(data['packet'],ix),data['baseline'][ix]
        utility,influence=model.cooperative_response(packet,base);mask=~torch.eye(data['n'],dtype=torch.bool)
        own=influence.diagonal(dim1=-2,dim2=-1);responsibility=utility.softmax(-1)
        team=torch.einsum('bij,bj->bi',influence,responsibility)
        pair=torch.einsum('bij,bj,bkj->bik',influence,responsibility,influence)-team[:,:,None]*team[:,None]
        loss=((own-data['fields'][ix,:,0])/field_scale[0]).square().mean()
        loss+=((utility-data['utility'][ix])/utility_scale).square().mean()
        loss+=.5*((influence-data['influence'][ix])[:,mask]/peer_scale).square().mean()
        loss+=((team-data['fields'][ix,:,1])/field_scale[1]).square().mean()
        loss+=.25*((pair-data['pairs'][ix])[:,mask]/pair_scale).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%250==0 or step==args.updates:
            metrics=[]
            with torch.no_grad():
                for v in valid:
                    f,e=model.cooperative_effects(v['packet'],v['baseline']);mask=~torch.eye(v['n'],dtype=torch.bool)
                    fm=(f-v['fields']).square().mean((0,1));fz=v['fields'].square().mean((0,1))
                    em=(e-v['pairs'])[:,mask].square().mean();ez=v['pairs'][:,mask].square().mean()
                    u,h=model.cooperative_response(v['packet'],v['baseline'])
                    metrics.append(dict(defenders=v['n'],field_mse=fm.tolist(),field_zero=fz.tolist(),pair_mse=float(em),pair_zero=float(ez),utility_mse=float((u-v['utility']).square().mean()),peer_mse=float((h-v['influence'])[:,mask].square().mean())))
            score=np.mean([v['field_mse'][1]/v['field_zero'][1]+.5*v['pair_mse']/v['pair_zero'] for v in metrics])
            row=dict(step=step,score=float(score),metrics=metrics);history.append(row);print(row,flush=True)
            if score<best_score:
                best_score,selected=score,step;best={k:v.detach().clone() for k,v in model.state_dict().items()}
    saved.update(model=best,cooperation='factorized',cooperation_history=history,cooperation_selected_step=selected,
        cooperation_field_scale=field_scale,cooperation_pair_scale=pair_scale,conditional_effects=True,
        seed=args.seed,readout=None,local_contrast=None,response_value_actor=None)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


def fit_mediated(args):
    """Identify receiver-specific effects through a shared attacker mediator."""
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    saved = torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    model = ResponseGraph(saved['width'],task_response=True,distributional=saved.get('distributional',False),mediated=True,
                          physical_mediator=args.physical_mediator,temporal=bool(args.intent_checkpoint))
    model.load_state_dict(saved['model'],strict=False)
    if args.intent_checkpoint:
        model.intent.load_state_dict(torch.load(args.intent_checkpoint,weights_only=True)['model'])
    anchors = {name:p.detach().clone() for name,p in model.named_parameters() if name in saved['model']}
    for name,p in model.named_parameters():
        p.requires_grad_(not name.startswith(('value','gate','response_pair','response_target','intent.recurrent','intent.output')))
    train = load_rows(args.output/'train',args.train_states,'cpu')
    valid = load_rows(args.output/'development',args.test_states,'cpu')
    bits = torch.tensor(list(itertools.product((0.,1.),repeat=3)))
    signs = 2.*bits-1.
    diagonal = torch.eye(3,dtype=torch.bool)
    def effects(target,packet):
        contrast = torch.einsum('cs,bcjf->bsjf',signs,target[:,1:])/4.
        obs = packet['obs']
        direction = torch.stack((obs[...,3].cos(),obs[...,3].sin()),-1)
        return -((contrast[...,:2]+.5*contrast[...,2:])*direction[:,None]).sum(-1)
    def synergies(target,packet):
        pair_signs = torch.stack([signs[:,i]*signs[:,j] for i,j in itertools.combinations(range(3),2)],1)
        contrast = torch.einsum('cs,bcjf->bsjf',pair_signs,target[:,1:])/2.
        direction = torch.stack((packet['obs'][...,3].cos(),packet['obs'][...,3].sin()),-1)
        return -((contrast[...,:2]+.5*contrast[...,2:])*direction[:,None]).sum(-1)
    def common(node,target,packet):
        motion,obs = packet['motion'],packet['obs']
        bearing = motion[...,2]+obs[...,3]
        attacker = (motion[...,:2]+obs[...,2,None]*torch.stack((bearing.cos(),bearing.sin()),-1)).mean(1)
        phi = torch.atan2(attacker[:,1],attacker[:,0])
        angle = motion[...,2]-phi[:,None]
        c,s = angle.cos()[:,None],angle.sin()[:,None]
        body = node[...,:4]+target
        def rotate(v): return torch.stack((c*v[...,0]-s*v[...,1],s*v[...,0]+c*v[...,1]),-1)
        return torch.cat((rotate(body[...,:2]),rotate(body[...,2:])), -1).mean(2)
    train['common'] = common(train['response_node'],train['response_target'],train['packet'])
    train['effects'] = effects(train['response_target'],train['packet'])
    train['synergy'] = synergies(train['response_target'],train['packet'])
    own_scale = (train['response_node'][:,1:]-train['response_node'][:,:1]).square().mean((0,1,2)).clamp_min(.001)
    common_scale = (train['common'][:,1:]-train['common'][:,:1]).square().mean((0,1)).clamp_min(.0005)
    self_scale = train['effects'][:,diagonal].square().mean().clamp_min(.001)
    peer_scale = train['effects'][:,~diagonal].square().mean().clamp_min(.0005)
    synergy_scale = train['synergy'].square().mean().clamp_min(.0005)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-3)
    def predictions(data,index):
        packet = subset(data['packet'],index)
        count = len(packet['obs'])
        expanded = {k:v[:,None].expand(-1,9,*v.shape[1:]).flatten(0,1) for k,v in packet.items()}
        base = data['baseline'][index,None].expand(-1,9,-1,-1).flatten(0,1)
        correction = data['correction'][index].flatten(0,1)
        _,motion,_,_,target = model(expanded,base,correction,details=True)
        return motion.reshape(count,9,3,7),target.reshape(count,9,3,4),packet
    history = []
    best_model,best_error,best_step = None,float('inf'),0
    for step in range(1,args.updates+1):
        ix = torch.tensor(rng.integers(args.train_states,size=32))
        motion,target,packet = predictions(train,ix)
        shared = common(motion,target,packet)
        influence = effects(target,packet)
        node_difference = motion[:,1:]-motion[:,:1]-(train['response_node'][ix,1:]-train['response_node'][ix,:1])
        common_difference = shared[:,1:]-shared[:,:1]-(train['common'][ix,1:]-train['common'][ix,:1])
        loss = (node_difference.square()/own_scale).mean()+.2*F.mse_loss(motion,train['response_node'][ix])
        loss += (common_difference.square()/common_scale).mean()+.2*F.mse_loss(shared,train['common'][ix])
        error = influence-train['effects'][ix]
        loss += error[:,diagonal].square().mean()/self_scale+error[:,~diagonal].square().mean()/peer_scale
        if args.synergy_supervision:
            loss += F.mse_loss(synergies(target,packet),train['synergy'][ix])/synergy_scale
        loss += .05*sum((p-anchors[name]).square().sum() for name,p in model.named_parameters()
                        if p.requires_grad and name in anchors)
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%250==0 or step==args.updates:
            with torch.no_grad():
                motion,target,packet = predictions(valid,torch.arange(args.test_states))
                estimate = effects(target,packet);truth = effects(valid['response_target'],packet)
                row = dict(step=step,self_mse=float((estimate-truth)[:,diagonal].square().mean()),
                    self_zero=float(truth[:,diagonal].square().mean()),
                    peer_mse=float((estimate-truth)[:,~diagonal].square().mean()),
                    peer_zero=float(truth[:,~diagonal].square().mean()),
                    peer_correlation=float(torch.corrcoef(torch.stack((estimate[:,~diagonal].flatten(),truth[:,~diagonal].flatten())))[0,1]))
                synergy_truth = synergies(valid['response_target'],packet)
                row['synergy_mse'] = float((synergies(target,packet)-synergy_truth).square().mean())
                row['synergy_zero'] = float(synergy_truth.square().mean())
            history.append(row);print(row,flush=True)
            selection = row['peer_mse']/row['peer_zero']
            if args.synergy_supervision: selection += row['synergy_mse']/row['synergy_zero']
            if selection<best_error:
                best_error,best_step = selection,step
                best_model = {name:weight.detach().clone() for name,weight in model.state_dict().items()}
    saved.update(model=best_model,mediated=True,physical_mediator=args.physical_mediator,temporal=bool(args.intent_checkpoint),seed=args.seed,mediation_history=history,
                 mediation_selected_step=best_step,selection_split='development',
                 synergy_supervision=args.synergy_supervision,
                 intent_source=str(args.intent_checkpoint) if args.intent_checkpoint else None,
                 task_response=True,readout=None,local_contrast=None,response_value_actor=None)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


@torch.no_grad()
def contrast_correction(model, packet, baseline, saved):
    """Fixed unrolled coordination in a learned local response energy.

    This changes only fusion weights. It uses no environment, future noise,
    rollout, or candidate search at deployment.
    """
    node,pair = contrast_features(model,packet,baseline)
    node = contrast_transform(node,saved['node_transform'])
    pair = contrast_transform(pair,saved['pair_transform'])
    weights = saved['weights']
    split = node.shape[-1]
    unary = torch.einsum('bnf,hf->bhn',node,weights[:,:split])
    coupling = torch.einsum('bnmf,hf->bhnm',pair,weights[:,split:])
    n = baseline.shape[1]
    mask = ~torch.eye(n,dtype=torch.bool,device=baseline.device)
    coupling = coupling*mask[None,None]
    x = torch.zeros_like(baseline[...,2])
    radius = saved['radius']
    lower,upper = (-baseline[...,2]/radius).clamp_min(-1.),((1.-baseline[...,2])/radius).clamp_max(1.)
    # The trust curvature makes each step respond to estimated task benefit
    # in seconds, rather than blindly moving to a factorial corner.
    curvature = 2.*saved['trust_seconds']+coupling.abs().sum(-1).amax((1,2))
    for _ in range(saved['coordination_steps']):
        values = (unary*x[:,None]).sum(-1)+.5*torch.einsum('bhnm,bn,bm->bh',coupling,x,x)
        gradients = unary+torch.einsum('bhnm,bm->bhn',coupling,x)
        centered = values-values.mean(-1,keepdim=True)
        risk = (centered[...,None]*gradients).mean(1)/(values.var(-1,unbiased=False)+.04).sqrt()[:,None]
        gradient = gradients.mean(1)-saved['risk']*risk-2.*saved['trust_seconds']*x
        x = (x+gradient/curvature[:,None]).clamp(lower,upper)
    return radius*x


def fit_local_contrast(args):
    torch.set_num_threads(1)
    saved = torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    model = ResponseGraph(saved['width'],task_response=saved.get('task_response',False),
                          distributional=saved.get('distributional',False),mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),temporal=saved.get('temporal',False),cooperation=saved.get('cooperation',False)).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    data = load_rows(args.output/'train',args.train_states,'cpu')
    valid = load_rows(args.output/'development',args.test_states,'cpu')
    if not data['complete_returns'] or not valid['complete_returns']:
        raise ValueError('Task-value fitting requires complete returns, not response-only prefixes.')
    radius = args.intervention_radius
    if radius is None or radius <= 0.: raise ValueError('A positive local intervention radius is required.')
    for split in ('train','development'):
        row = torch.load(args.output/split/'0.pth',weights_only=False,map_location='cpu')
        if row.get('intervention_radius') != radius or row['horizon'] != args.horizon:
            raise ValueError('Local value targets and deployed interventions must agree.')
    node,pair = contrast_features(model,data['packet'],data['baseline'])
    def transform(value):
        flat = value.reshape(-1,value.shape[-1])
        mean,scale = flat.mean(0),flat.std(0).clamp_min(.05)
        _,singular,vt = torch.linalg.svd((flat-mean)/scale,full_matrices=False)
        count = min(args.readout_dim,len(singular))
        projection = vt[:count].T/(singular[:count]/np.sqrt(len(flat))).clamp_min(.2)
        result = dict(mean=mean,scale=scale,projection=projection)
        if args.kernel:
            projected = ((flat-mean)/scale)@projection
            generator = torch.Generator().manual_seed(args.seed)
            centers = projected[torch.randperm(len(projected),generator=generator)[:64]]
            kernel = torch.exp(-torch.cdist(projected,centers).square()/(2.*projected.shape[-1]))
            result.update(centers=centers,kernel_mean=kernel.mean(0),kernel_scale=kernel.std(0).clamp_min(.05))
        return result
    n = node.shape[1]
    nt = transform(node)
    pt = transform(pair[:,torch.triu(torch.ones(n,n,dtype=torch.bool),1)])
    def matrix(bank):
        nf,pf = contrast_features(model,bank['packet'],bank['baseline'])
        nf,pf = contrast_transform(nf,nt),contrast_transform(pf,pt)
        return torch.stack([contrast_design(nf,pf,bank['correction'][:,choice],radius)
                            for choice in range(bank['correction'].shape[1])],1)
    design = matrix(data).double().numpy()
    # Center every complete factorial group. This cancels common baseline
    # branch noise without inventing extra independent training examples.
    if args.continuation_checkpoint:
        continuation = torch.load(args.continuation_checkpoint,map_location='cpu',weights_only=True)
        # The value uses the same response representation that was used to
        # train it; its average replaces the long stochastic continuation.
        if continuation['graph_source'] != str(args.graph_checkpoint):
            raise ValueError('Continuation and relation encoders must match.')
        outcome = -continuation_targets(args,model,continuation['continuation'],'train',args.train_states).mean(-1)
        saved['continuation_source'] = str(args.continuation_checkpoint)
    else:
        outcome = -data['times'].mean(-1).double().numpy()
    x = design-design.mean(1,keepdims=True)
    y = outcome-outcome.mean(1,keepdims=True)
    rng = np.random.default_rng(args.seed)
    folds = np.empty(len(x),dtype=int); folds[rng.permutation(len(x))] = np.arange(len(x))%4
    def solve(indices,alpha,root_weight=None):
        a,b = x[indices].reshape(-1,x.shape[-1]),y[indices].reshape(-1)
        if root_weight is not None:
            weight = np.repeat(root_weight,len(x[0]))**.5
            a,b = a*weight[:,None],b*weight
        return np.linalg.solve(a.T@a+alpha*np.eye(a.shape[1]),a.T@b)
    scores = []
    for alpha in (1.,10.,100.,1000.):
        prediction = np.zeros_like(y)
        for fold in range(4):
            weight = solve(folds!=fold,alpha)
            prediction[folds==fold] = x[folds==fold]@weight
        scores.append(dict(alpha=alpha,mse=float(np.mean((prediction-y)**2)),
                           zero_mse=float(np.mean(y*y))))
    alpha = min(scores,key=lambda row:row['mse'])['alpha']
    weights = np.stack([solve(np.arange(len(x)),alpha,rng.poisson(1.,len(x))) for _ in range(12)])
    saved['local_contrast'] = dict(node_transform=nt,pair_transform=pt,weights=torch.as_tensor(weights,dtype=torch.float32),
        radius=radius,trust_seconds=args.trust_seconds,risk=.5,coordination_steps=6,
        cross_validation=scores,alpha=alpha,training_roots=args.train_states)
    saved['readout'] = None
    saved['horizon'] = args.horizon
    saved['seed'] = args.seed
    saved['graph_source'] = str(args.graph_checkpoint)
    saved['training_noisy_agility'] = False
    vd = matrix(valid).numpy()
    prediction = (vd@weights.T).mean(-1)
    actual = (-valid['times'].mean(-1)+valid['times'][:,0].mean(-1,keepdim=True)).numpy()
    index = np.arange(len(vd));choice = prediction.argmax(1)
    diagnostics = dict(cross_validation=scores,alpha=alpha,
        difference_mae=float(np.abs(prediction[:,1:]-actual[:,1:]).mean()),
        zero_mae=float(np.abs(actual[:,1:]).mean()),
        development_selected_time=float(-actual[index,choice].mean()))
    saved['local_contrast']['diagnostics'] = diagnostics
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)
    print(diagnostics,flush=True)


@torch.no_grad()
def calibration(model, data):
    p, a = data['packet'], data['baseline']
    predictions = torch.stack([model(p,a,data['correction'][:,i])[0] for i in range(data['correction'].shape[1])],1)
    predicted = 20.*(predictions.mean(-1)-predictions[:, :1].mean(-1))
    outcome = -data['times'] if model.distributional else data['returns']
    measured = outcome.mean(-1)-outcome[:, :1].mean(-1)
    chosen = predicted.argmax(1)
    index = torch.arange(len(chosen), device=chosen.device)
    return dict(target='negative_capped_time' if model.distributional else 'paper_return',
        difference_mae=float((predicted[:,1:]-measured[:,1:]).abs().mean()),
        zero_mae=float(measured[:,1:].abs().mean()),
        selected_return=float(measured[index,chosen].mean()),
        selected_time=float((data['times'].mean(-1)[index,chosen]-data['times'].mean(-1)[:,0]).mean()))


def fit(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = load_rows(args.output/'train', args.train_states, args.device)
    valid = load_rows(args.output/'development', args.test_states, args.device)
    if not data['complete_returns'] or not valid['complete_returns']:
        raise ValueError('Task-value fitting requires complete returns, not response-only prefixes.')
    if data['horizon'] != args.horizon or valid['horizon'] != args.horizon:
        raise ValueError('Deployment and intervention durations must match.')
    model = ResponseGraph(args.width,task_response=args.task_response,distributional=args.distributional).to(args.device)
    gate_ids = {id(p) for p in model.gate.parameters()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if id(p) not in gate_ids], lr=3e-4)
    n = args.train_states
    # Each value head sees a fixed bootstrap of independent roots, preserving
    # within-root pairing instead of treating nine treatments as independent.
    weights = torch.tensor(rng.poisson(1.,size=(n,3)),dtype=torch.float32,device=args.device)
    history = []
    for step in range(1,args.updates+1):
        index = torch.as_tensor(rng.integers(n,size=args.batch_size),device=args.device)
        choice = torch.as_tensor(rng.integers(1,9,size=args.batch_size),device=args.device)
        p, base = subset(data['packet'],index), data['baseline'][index]
        q0,r0,e0,l0,t0 = model(p,base,data['correction'][index,0],details=True)
        q,r,e,l,t = model(p,base,data['correction'][index,choice],details=True)
        if args.distributional:
            def distribution(times):
                remaining = (times-(60.-60.*p['central'][:,-1,None])).clamp(0.,60.)
                low = remaining.floor().long(); high = (low+1).clamp_max(60)
                fraction = remaining-low
                target = torch.zeros((len(times),61),device=args.device)
                target.scatter_add_(1,low,(1.-fraction)/times.shape[1])
                target.scatter_add_(1,high,fraction/times.shape[1])
                return target
            d0,d = distribution(data['times'][index,0]),distribution(data['times'][index,choice])
            absolute = ((-(d[:,None]*l.log_softmax(-1)).sum(-1)
                -(d0[:,None]*l0.log_softmax(-1)).sum(-1))*.5*weights[index]).mean()
            target_difference = d.cumsum(-1)-d0.cumsum(-1)
            predicted_difference = l.softmax(-1).cumsum(-1)-l0.softmax(-1).cumsum(-1)
            paired = ((predicted_difference-target_difference[:,None]).square().mean(-1)*weights[index]).mean()
        else:
            g0 = data['returns'][index,0].mean(-1,keepdim=True)
            g = data['returns'][index,choice].mean(-1,keepdim=True)
            paired = ((q-q0-(g-g0)/20.).square()*weights[index]).mean()
            absolute = ((q0-g0/20.).square()*weights[index]).mean()
        # Counterfactual displacement teaches how control combinations alter
        # relative motion; absolute prediction anchors the common dynamics.
        mask = ~torch.eye(base.shape[1],dtype=torch.bool,device=args.device)
        response = F.mse_loss(r-r0,data['response_node'][index,choice]-data['response_node'][index,0])
        response += F.mse_loss((e-e0)[:,mask],(data['response_pair'][index,choice]-data['response_pair'][index,0])[:,mask])
        response += .2*F.mse_loss(r0,data['response_node'][index,0])
        if args.task_response:
            response += F.mse_loss(t-t0,data['response_target'][index,choice]-data['response_target'][index,0])
            response += .2*F.mse_loss(t0,data['response_target'][index,0])
        loss = (2.*paired+absolute if args.distributional else paired+.2*absolute)+args.response_weight*response
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step%250==0 or step==args.updates:
            row = dict(stage='value',step=step,paired=float(paired.detach()),
                response=float(response.detach()),development=calibration(model,valid))
            history.append(row)
            print(row,flush=True)
    # Freeze the learned response/value representation while improving its
    # fusion readout. No labels chosen using the development bank are used.
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.gate.parameters(): p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.gate.parameters(),lr=3e-4)
    for step in range(1,args.actor_updates+1):
        index = torch.as_tensor(rng.integers(n,size=args.batch_size),device=args.device)
        p, base = subset(data['packet'],index),data['baseline'][index]
        correction = model.correction(p,base)
        q = model(p,base,correction)[0]
        reference = model(p,base,torch.zeros_like(correction))[0].detach()
        difference = q-reference
        conservative = difference.mean(-1)-.5*difference.std(-1)
        loss = -conservative.mean()+args.trust*correction.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    saved = dict(model=model.cpu().state_dict(),width=args.width,horizon=args.horizon,
                 seed=args.seed,updates=args.updates,actor_updates=args.actor_updates,
                 response_weight=args.response_weight,trust=args.trust,
                 train_states=args.train_states,development_states=args.test_states,history=history,
                 task_response=args.task_response,distributional=args.distributional)
    atomic_torch(args.output/f'candidate-{args.seed}.pth',saved)


def evaluate_scene(task):
    scene, agility, defenders = task[:3]
    if len(task)>3:LEARNED.minimum_history=task[3]
    rows = []
    for policy in (SOURCE, LEARNED):
        if isinstance(policy,ResponsePolicy): policy.remaining = 0
        with preserved_random_state():
            seed_random(scene)
            env = TADEnv(defenders,protocol='paper-parameters-v1')
            obs,_ = env.reset(agility,noisy_agility=False)
            np.random.seed(scene+1000000)
            packet,done,total,discount = public_packet(env,obs),0,0.,1.
            while not done:
                action = policy.choose_action(packet,deterministic=True)[0]
                packet,reward,done,_,_,_ = execute(env,packet,action,safety=True)
                total += discount*float(np.mean(reward)); discount *= .99
            rows.append(dict(**outcome_row(env,done),discounted_return=total))
    return dict(scene=scene,agility=agility,defenders=defenders,baseline=rows[0],candidate=rows[1])


def tune_fusion_scene(task):
    scene,agility,defenders,settings=task
    rows=[]
    for setting in [None]+settings:
        policy=SOURCE if setting is None else LEARNED
        if setting is not None:
            policy.fusion_scale,policy.horizon=setting
            policy.remaining=0
        with preserved_random_state():
            seed_random(scene)
            env=TADEnv(defenders,protocol='paper-parameters-v1')
            obs,_=env.reset(agility,noisy_agility=False)
            np.random.seed(scene+1000000)
            packet,done,total,discount=public_packet(env,obs),0,0.,1.
            while not done:
                action=policy.choose_action(packet,deterministic=True)[0]
                packet,reward,done,_,_,_=execute(env,packet,action,safety=True)
                total+=discount*float(np.mean(reward));discount*=.99
            rows.append(dict(**outcome_row(env,done),discounted_return=total))
    return dict(scene=scene,baseline=rows[0],candidates=rows[1:])


def tune_fusion(args):
    settings=list(itertools.product((.5,1.,1.5),(1,5,10,20)))
    tasks=[(2730000000+args.bank*100000+i,args.agility,args.defenders,settings) for i in range(args.episodes)]
    rows=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(args.source,args.graph_checkpoint)) as pool:
        for row in pool.map(tune_fusion_scene,tasks):
            rows.append(row)
            if len(rows)%8==0:print(dict(completed=len(rows),episodes=args.episodes),flush=True)
    summary=[]
    for j,(scale,period) in enumerate(settings):
        delta=np.asarray([r['candidates'][j]['capture_time']-r['baseline']['capture_time'] for r in rows])
        summary.append(dict(scale=scale,period=period,time_difference=float(delta.mean()),
            standard_error=float(delta.std(ddof=1)/len(delta)**.5),
            baseline_capture=float(np.mean([r['baseline']['capture'] for r in rows])),
            candidate_capture=float(np.mean([r['candidates'][j]['capture'] for r in rows])),
            candidate_collision=float(np.mean([r['candidates'][j]['collision'] for r in rows]))))
    suffix=f'-bank{args.bank}-n{args.defenders}-a{args.agility:g}'
    atomic_json(args.output/f'fusion-tuning-{args.seed}{suffix}.json',dict(
        purpose='development-only hyperparameter selection',checkpoint=str(args.graph_checkpoint),
        settings=settings,summary=summary,rows=rows))
    for row in sorted(summary,key=lambda r:r['time_difference']):print(row,flush=True)


def constant_fusion_scene(task):
    scene,agility,defenders,offsets=task
    rows=[]
    for offset in offsets:
        with preserved_random_state():
            seed_random(scene);env=TADEnv(defenders,protocol='paper-parameters-v1')
            obs,_=env.reset(agility,noisy_agility=False);np.random.seed(scene+1000000)
            packet,done=public_packet(env,obs),0
            while not done:
                action=SOURCE.choose_action(packet,deterministic=True)[0]
                action[:,2]=np.clip(action[:,2]+offset,0.,1.)
                packet,_,done,_,_,_=execute(env,packet,action,safety=True)
            rows.append(outcome_row(env,done))
    return dict(scene=scene,baseline=rows[0],candidates=rows[1:])


def tune_constant(args):
    offsets=[0.,-.025,-.05,-.1,-.2,.05,.1]
    tasks=[(2730000000+args.bank*100000+i,args.agility,args.defenders,offsets) for i in range(args.episodes)]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(args.source,)) as pool:
        rows=list(pool.map(constant_fusion_scene,tasks))
    summary=[]
    for j,offset in enumerate(offsets[1:]):
        delta=np.asarray([r['candidates'][j]['capture_time']-r['baseline']['capture_time'] for r in rows])
        summary.append(dict(offset=offset,time_difference=float(delta.mean()),
            standard_error=float(delta.std(ddof=1)/len(delta)**.5),
            capture=float(np.mean([r['candidates'][j]['capture'] for r in rows]))))
    atomic_json(args.output/f'constant-tuning-bank{args.bank}-n{args.defenders}-a{args.agility:g}.json',
        dict(purpose='development-only constant-gate comparator',offsets=offsets,summary=summary,rows=rows))
    for row in sorted(summary,key=lambda r:r['time_difference']):print(row,flush=True)


def function_intervention(task):
    """Paired complete closed-loop policies, not mislabeled one-step Q targets."""
    scene, weights = task[:2]
    defenders = task[2] if len(task)>2 else 3
    rows = []
    for weight in weights:
        LEARNED.readout['weight'] = torch.as_tensor(weight,dtype=torch.float32)
        LEARNED.remaining = 0
        with preserved_random_state():
            seed_random(scene)
            env = TADEnv(defenders,protocol='paper-parameters-v1')
            obs,_ = env.reset(2.25,noisy_agility=LEARNED.training_noisy_agility)
            np.random.seed(scene+1000000)
            packet,done,total,discount = public_packet(env,obs),0,0.,1.
            while not done:
                action = LEARNED.choose_action(packet,deterministic=True)[0]
                packet,reward,done,_,_,_ = execute(env,packet,action,safety=True)
                total += discount*float(np.mean(reward)); discount *= .99
            rows.append(dict(**outcome_row(env,done),discounted_return=total))
    return rows


def execute_modal_fusion(env, packet, baseline, gates):
    """Fuse candidate surge/yaw components, then use the same physical CBF.

    This changes the fusion operator only. The original learned and Boids
    candidates, opponent, task reward, and actuator/safety constraints remain
    the inputs to exactly the same physical execution path.
    """
    if np.array_equal(gates[:,0],gates[:,1]):
        action = baseline.copy()
        action[:,2] = gates[:,0]
    else:
        learned, boids = baseline[:,:2],packet['obs'][:,12:14]
        def modes(value):
            return np.stack((.5*(value[:,0]+value[:,1]),.5*(value[:,0]-value[:,1])),-1)
        mixed = gates*modes(learned)+(1.-gates)*modes(boids)
        nominal = np.stack((mixed[:,0]+mixed[:,1],mixed[:,0]-mixed[:,1]),-1)
        # Exact representation of the resulting nominal thrust for the
        # existing CBF API; this is not a newly generated candidate policy.
        action = np.column_stack((nominal,np.ones(len(baseline),dtype=np.float32)))
    return execute(env,packet,action,safety=True)


def modal_probe_scene(task):
    scene, offsets = task
    rows = []
    for offset in offsets:
        with preserved_random_state():
            seed_random(scene)
            env = TADEnv(3,protocol='paper-parameters-v1')
            obs,_ = env.reset(2.25,noisy_agility=False)
            np.random.seed(scene+1000000)
            packet,done,total,discount = public_packet(env,obs),0,0.,1.
            while not done:
                baseline = SOURCE.choose_action(packet,deterministic=True)[0]
                gates = np.clip(baseline[:,2:3]+np.asarray(offset)[None,:],0.,1.)
                packet,reward,done,_,_,_ = execute_modal_fusion(env,packet,baseline,gates)
                total += discount*float(np.mean(reward)); discount *= .99
            rows.append(dict(**outcome_row(env,done),discounted_return=total))
    return rows


def probe_modes(args):
    offsets = [(0.,0.)]+[(a,b) for a,b in itertools.product((-.2,0.,.2),repeat=2) if (a,b)!=(0.,0.)]
    tasks = [(3010000000+index,offsets) for index in range(args.episodes)]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(args.source,)) as pool:
        results = list(pool.map(modal_probe_scene,tasks))
    summary = []
    for choice,offset in enumerate(offsets):
        delta = np.asarray([r[choice]['capture_time']-r[0]['capture_time'] for r in results])
        row = dict(correction=offset,time_change=float(delta.mean()),
            time_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),
            captures=int(sum(r[choice]['capture'] for r in results)),
            return_change=float(np.mean([r[choice]['discounted_return']-r[0]['discounted_return'] for r in results])))
        summary.append(row)
        print(row,flush=True)
    atomic_json(args.output/'modal-fusion-probe.json',dict(summary=summary,episodes=results))


def confidence_probe_scene(task):
    scene, sharpness = task
    rows = []
    for value in sharpness:
        with preserved_random_state():
            seed_random(scene)
            env = TADEnv(3,protocol='paper-parameters-v1')
            obs,_ = env.reset(2.25,noisy_agility=False)
            np.random.seed(scene+1000000)
            packet,done,total,discount = public_packet(env,obs),0,0.,1.
            while not done:
                action = SOURCE.choose_action(packet,deterministic=True)[0]
                if value != 1.:
                    gate = action[:,2].astype(np.float64).clip(1e-7,1.-1e-7)
                    logit = np.log(gate)-np.log1p(-gate)
                    action[:,2] = 1./(1.+np.exp(-np.clip(value*logit,-40.,40.)))
                packet,reward,done,_,_,_ = execute(env,packet,action,safety=True)
                total += discount*float(np.mean(reward)); discount *= .99
            rows.append(dict(**outcome_row(env,done),discounted_return=total))
    return rows


def probe_confidence(args):
    sharpness = (.25,.5,1.,2.,4.)
    tasks = [(3110000000+index,sharpness) for index in range(args.episodes)]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(args.source,)) as pool:
        results = list(pool.map(confidence_probe_scene,tasks))
    summary = []
    for choice,value in enumerate(sharpness):
        delta = np.asarray([r[choice]['capture_time']-r[2]['capture_time'] for r in results])
        row = dict(sharpness=value,time_change=float(delta.mean()),
            time_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),
            captures=int(sum(r[choice]['capture'] for r in results)),
            return_change=float(np.mean([r[choice]['discounted_return']-r[2]['discounted_return'] for r in results])))
        summary.append(row)
        print(row,flush=True)
    atomic_json(args.output/'fusion-confidence-probe.json',dict(summary=summary,episodes=results))


def calibrate_readout(args):
    """Local conditional value fitting for coherent fusion-function treatments.

    Paired full-policy returns fit a regularized local value derivative. This
    calibrates the fusion readout in the response representation without
    trusting a global critic's unsupported action derivatives. The evaluated
    treatment and the learned value have identical temporal semantics.
    """
    torch.set_num_threads(1)
    saved = torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    model = ResponseGraph(saved['width'],task_response=saved.get('task_response',False),
                          distributional=saved.get('distributional',False),mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),temporal=saved.get('temporal',False),cooperation=saved.get('cooperation',False)).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    data = load_rows(args.output/'train',args.train_states,'cpu')
    with torch.no_grad():
        x = model.readout_features(data['packet'],data['baseline'],args.readout_features)
        if args.allocation: x = x-x.mean(1,keepdim=True)
        x = x.flatten(0,1)
        mean = x.mean(0)
        scale = x.std(0).clamp_min(.01)
        _,singular,right = torch.linalg.svd((x-mean)/scale,full_matrices=False)
        projection = right[:args.readout_dim].T/(singular[:args.readout_dim]/np.sqrt(len(x)-1)).clamp_min(.1)
        projection = projection/scale[:,None]
    weights = np.zeros(args.readout_dim+1,dtype=np.float32)
    saved['readout'] = dict(mean=mean,projection=projection,weight=torch.as_tensor(weights),features_mode=args.readout_features,allocation=args.allocation)
    saved['horizon'] = args.horizon
    saved['seed'] = args.seed
    saved['training_noisy_agility'] = not args.exact_agility
    saved['calibration'] = []
    path = args.output/f'candidate-{args.seed}.pth'
    atomic_torch(path,saved)
    rng = np.random.default_rng(args.seed)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(args.source,path)) as pool:
        for iteration in range(args.iterations):
            directions = rng.normal(size=(args.directions,len(weights)))
            if args.allocation:
                # Identify total fusion use separately from its distribution
                # among vessels. Transfer treatments retain the same budget.
                directions[:,-1] = 0.
                directions[0] = 0.
                directions[0,-1] = 1.
            directions /= np.linalg.norm(directions,axis=1,keepdims=True)
            scenes = [2810000000+args.seed*10000+iteration*100+index for index in range(args.scenes)]
            tasks = [(scene,(weights+args.radius*d,weights-args.radius*d)) for d in directions for scene in scenes]
            results = list(pool.map(function_intervention,tasks))
            key = 'discounted_return' if args.target=='paper_return' else 'capture_time'
            sign = 1. if args.target=='paper_return' else -1.
            g = sign*np.asarray([[r[key] for r in pair] for pair in results]).reshape(args.directions,args.scenes,2)
            contrasts = (g[:,:,0]-g[:,:,1])/20.
            y = contrasts.mean(1)/(2.*args.radius)
            # Root/scenario pairing supplies the contrast. Ridge shrinkage
            # limits amplification along sparsely observed treatment directions.
            gradient = np.linalg.solve(directions.T@directions+.2*np.eye(len(weights)),directions.T@y)
            gradient /= max(np.linalg.norm(gradient),1e-8)
            proposal = weights+args.step_size*gradient
            # Evaluate the proposed finite step on fresh scenarios, retaining
            # every outcome; this is development, not confirmation evidence.
            checks = [(2810500000+args.seed*10000+iteration*100+index,(weights,proposal)) for index in range(args.check_scenes)]
            verification = list(pool.map(function_intervention,checks))
            change = np.asarray([p[1]['discounted_return']-p[0]['discounted_return'] for p in verification])
            time_change = np.asarray([p[1]['capture_time']-p[0]['capture_time'] for p in verification])
            accepted = bool(change.mean()>0 and time_change.mean()<0)
            if accepted: weights = proposal.astype(np.float32)
            record = dict(iteration=iteration,accepted=accepted,return_change=float(change.mean()),
                time_change=float(time_change.mean()),contrast_std=float(contrasts.std()),
                weights=weights.tolist(),verification=verification,search_returns=g.tolist(),
                treatment_directions=directions.tolist(),radius=args.radius,step_size=args.step_size,
                objective=args.target,search_outcomes=results)
            saved['calibration'].append(record)
            saved['readout']['weight'] = torch.as_tensor(weights)
            atomic_torch(path,saved)
            print({k:v for k,v in record.items() if k not in ('weights','verification','search_returns','treatment_directions','search_outcomes')},flush=True)


def probe_readout(args):
    checkpoint = args.output/f'candidate-{args.seed}.pth'
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    offsets = (-2.,-.4,-.1,0.,.1,.4,2.)
    weights = np.zeros((len(offsets),len(saved['readout']['weight'])),dtype=np.float32)
    weights[:,-1] = offsets
    tasks = [(2910000000+index,weights) for index in range(args.episodes)]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(args.source,checkpoint)) as pool:
        results = list(pool.map(function_intervention,tasks))
    rows = []
    for choice,offset in enumerate(offsets):
        delta = np.asarray([r[choice]['capture_time']-r[3]['capture_time'] for r in results])
        row = dict(raw_correction=offset,gate_correction=float(.5*np.tanh(offset)),
            time_change=float(delta.mean()),time_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),
            mean_time=float(np.mean([r[choice]['capture_time'] for r in results])),
            captures=int(sum(r[choice]['capture'] for r in results)),
            return_change=float(np.mean([r[choice]['discounted_return']-r[3]['discounted_return'] for r in results])))
        rows.append(row)
        print(row,flush=True)
    atomic_json(args.output/f'fusion-scale-{args.seed}.json',dict(summary=rows,episodes=results))


def optimize_odds(args):
    """Paired functional interventions calibrate a covariance-adapted policy step."""
    torch.set_num_threads(1)
    saved = torch.load(args.graph_checkpoint,map_location='cpu',weights_only=True)
    graph = ResponseGraph(saved['width'],task_response=True,distributional=saved.get('distributional',False),mediated=saved.get('mediated',False),physical_mediator=saved.get('physical_mediator',False),temporal=saved.get('temporal',False),cooperation=saved.get('cooperation',False)).eval().requires_grad_(False)
    graph.load_state_dict(saved['model'])
    data = load_rows(args.output/'train',args.train_states,'cpu')
    features_mode = 'projected_future' if args.future_priority else 'projected'
    if args.coupled:
        dimension = 6 if saved.get('cooperation')=='coverage' else 4
        readout = dict(link='coordination',field_scale=saved['cooperation_field_scale'],pair_scale=saved['cooperation_pair_scale'],prior_floor=args.prior_floor,
            conditional_effects=saved.get('conditional_effects',False))
    else:
        features = graph.readout_features(data['packet'],data['baseline'],features_mode).flatten(0,1)
        mean,scale = features.mean(0),features.std(0).clamp_min(.03 if args.future_priority else .1)
        dimension = features.shape[-1]+1
        readout = dict(features_mode=features_mode,mean=mean,projection=torch.diag(1./scale),link='odds')
    weights = np.zeros(dimension,dtype=np.float32)
    readout['weight'] = torch.as_tensor(weights)
    saved.update(readout=readout,local_contrast=None,response_value_actor=None,
        training_noisy_agility=False,horizon=args.horizon,seed=args.seed,calibration=[])
    path = args.output/f'candidate-{args.seed}.pth'
    atomic_torch(path,saved)
    rng = np.random.default_rng(args.seed)
    covariance,sigma = np.eye(dimension),args.radius
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(args.source,path)) as pool:
        for iteration in range(args.iterations):
            directions = rng.normal(size=(args.directions//2,dimension))@np.linalg.cholesky(covariance).T
            if args.coupled:
                directions[:min(dimension,len(directions))] = np.eye(dimension)[:min(dimension,len(directions))]
            elif args.future_priority and len(directions)>=4:
                directions[:4] = 0.
                directions[0,1:3] = 1./np.sqrt(2.)
                directions[1,0] = 1.
                directions[2,2] = 1.
                directions[3,-1] = 1.
            directions = np.concatenate((directions,-directions),0)
            candidates = np.vstack((weights,weights+sigma*directions)).astype(np.float32)
            team_size = lambda index: (3 if index%2==0 else 6) if args.mixed_teams else args.defenders
            tasks = [(3210000000+args.seed*10000+iteration*100+index,candidates,team_size(index)) for index in range(args.scenes)]
            outcomes = list(pool.map(function_intervention,tasks))
            times = np.asarray([[r['capture_time'] for r in scene] for scene in outcomes])
            change = times-times[:,:1]
            ranks = np.argsort(change.mean(0))
            count = min(3,len(ranks))
            elite_weights = np.log(count+.5)-np.log(np.arange(1,count+1));elite_weights /= elite_weights.sum()
            refinement = []
            if args.mode_preserving:
                # Preserve an evaluated cooperation mode. Averaging weights
                # from opposite modes need not preserve either one's value.
                shortlist = np.r_[0,[i for i in ranks if i!=0][:3]]
                refinements = [(3410000000+args.seed*10000+iteration*100+index,candidates[shortlist],team_size(index)) for index in range(args.scenes)]
                refinement = list(pool.map(function_intervention,refinements))
                extra_times = np.asarray([[r['capture_time'] for r in scene] for scene in refinement])
                extra_change = extra_times-extra_times[:,:1]
                combined = np.vstack((change[:,shortlist],extra_change))
                selected = int(shortlist[combined.mean(0).argmin()])
                proposal = candidates[selected].copy()
            else:
                proposal = .5*weights+.5*(candidates[ranks[:count]]*elite_weights[:,None]).sum(0)
            checks = [(3310000000+args.seed*10000+iteration*100+index,(weights,proposal),team_size(index)) for index in range(args.check_scenes)]
            verification = list(pool.map(function_intervention,checks))
            delta = np.array([r[1]['capture_time']-r[0]['capture_time'] for r in verification])
            captures = sum(r[1]['capture']-r[0]['capture'] for r in verification)
            accepted = bool(delta.mean()<0. and captures>=0)
            team_checks = []
            for defenders in sorted(set(t[2] for t in checks)):
                indices = [i for i,t in enumerate(checks) if t[2]==defenders]
                gain = sum(verification[i][1]['capture']-verification[i][0]['capture'] for i in indices)
                team_checks.append(dict(defenders=defenders,time_change=float(delta[indices].mean()),capture_change=gain))
            if args.mixed_teams:
                accepted = accepted and all(r['time_change']<=0. and r['capture_change']>=0 for r in team_checks)
            pooled_checks = []
            if args.pooled_update and args.mode_preserving:
                refined_index = int(np.flatnonzero(shortlist==selected)[0])
                paired = [(r[0],r[selected]) for r in outcomes]
                paired += [(r[0],r[refined_index]) for r in refinement]
                paired += [(r[0],r[1]) for r in verification]
                sizes = [team_size(i) for i in range(args.scenes)]*2+[team_size(i) for i in range(args.check_scenes)]
                for defenders in sorted(set(sizes)):
                    group = [p for p,n in zip(paired,sizes) if n==defenders]
                    pooled_checks.append(dict(defenders=defenders,episodes=len(group),
                        time_change=float(np.mean([b['capture_time']-a['capture_time'] for a,b in group])),
                        capture_change=sum(b['capture']-a['capture'] for a,b in group)))
                accepted = all(v['time_change']<0. and v['capture_change']>=0 for v in pooled_checks)
            standardized = (candidates[ranks[:count]]-weights)/sigma
            covariance = .8*covariance+.2*np.einsum('b,bi,bj->ij',elite_weights,standardized,standardized)+1e-4*np.eye(dimension)
            if accepted: weights = proposal.astype(np.float32)
            sigma *= 1.05 if accepted else .85
            row = dict(iteration=iteration,accepted=accepted,time_change=float(delta.mean()),
                time_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),capture_change=captures,
                search_time_changes=change.mean(0).tolist(),weights=weights.tolist(),sigma=sigma,
                search_outcomes=outcomes,verification=verification,candidates=candidates.tolist(),
                refinement=refinement,mode_preserving=args.mode_preserving,team_checks=team_checks,pooled_checks=pooled_checks)
            saved['calibration'].append(row);saved['readout']['weight'] = torch.as_tensor(weights)
            atomic_torch(path,saved)
            print({k:v for k,v in row.items() if k not in ('search_outcomes','verification','candidates','weights','refinement')},flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=('tune_constant','tune_fusion','fit_corrected_factor_policy','collect_factor_outcomes','collect','collect_value','collect_active_value','collect_candidate_contrast','collect_source_episode','collect_feedback','collect_modal','collect_adapted_context','collect_quadrature','augment_value','fit','fit_contrast','fit_pathwise_intervention','fit_score_intervention','fit_cooperative_factors','fit_candidate_intervention','fit_conditional_value','fit_feedback_value','fit_capture_responsibility','fit_continuation','fit_paired_continuation','fit_response_value','fit_mediated','fit_intent','fit_intent_refinement','fit_cooperation','fit_coverage','fit_factorized','optimize_odds','evaluate','calibrate','probe','probe_modes','probe_confidence'))
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--train-states',type=int,default=64)
    p.add_argument('--test-states',type=int,default=32)
    p.add_argument('--horizon',type=int,default=10)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--updates',type=int,default=1500)
    p.add_argument('--actor-updates',type=int,default=600)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--response-weight',type=float,default=.25)
    p.add_argument('--trust',type=float,default=.05)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--device',default='cpu')
    p.add_argument('--source-device',default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--episodes',type=int,default=64)
    p.add_argument('--bank',type=int,default=0)
    p.add_argument('--state-source',type=Path)
    p.add_argument('--graph-checkpoint',type=Path)
    p.add_argument('--continuation-checkpoint',type=Path)
    p.add_argument('--readout-dim',type=int,default=12)
    p.add_argument('--readout-features',choices=('latent','response'),default='latent')
    p.add_argument('--iterations',type=int,default=4)
    p.add_argument('--directions',type=int,default=12)
    p.add_argument('--scenes',type=int,default=12)
    p.add_argument('--check-scenes',type=int,default=12)
    p.add_argument('--radius',type=float,default=.4)
    p.add_argument('--step-size',type=float,default=.3)
    p.add_argument('--target',choices=('paper_return','capped_time'),default='paper_return')
    p.add_argument('--allocation',action='store_true')
    p.add_argument('--exact-agility',action='store_true')
    p.add_argument('--task-response',action='store_true')
    p.add_argument('--distributional',action='store_true')
    p.add_argument('--defenders',type=int,default=3)
    p.add_argument('--agility',type=float,default=2.25)
    p.add_argument('--mean-peer-credit',action='store_true')
    p.add_argument('--execution-residual',action='store_true')
    p.add_argument('--execution-base-scale',type=float,default=.5)
    p.add_argument('--credit-actor-epochs',type=int,default=1)
    p.add_argument('--fusion-scale',type=float)
    p.add_argument('--fusion-period',type=int)
    p.add_argument('--fusion-bias',type=float,default=0.)
    p.add_argument('--candidate-probe',choices=('aligned','permuted','state-only'),default='aligned')
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--intervention-radius',type=float)
    p.add_argument('--trust-seconds',type=float,default=1.)
    p.add_argument('--kernel',action='store_true')
    p.add_argument('--antithetic',action='store_true')
    p.add_argument('--prefix-only',action='store_true')
    p.add_argument('--physical-mediator',action='store_true')
    p.add_argument('--future-priority',action='store_true')
    p.add_argument('--synergy-supervision',action='store_true')
    p.add_argument('--intent-checkpoint',type=Path)
    p.add_argument('--coupled',action='store_true')
    p.add_argument('--mode-preserving',action='store_true')
    p.add_argument('--prior-floor',type=float,default=0.)
    p.add_argument('--mixed-source',type=Path)
    p.add_argument('--mixed-train-states',type=int,default=48)
    p.add_argument('--mixed-test-states',type=int,default=24)
    p.add_argument('--mixed-teams',action='store_true')
    p.add_argument('--pooled-update',action='store_true')
    p.add_argument('--context-source',type=Path)
    p.add_argument('--max-value-groups',type=int,default=16)
    p.add_argument('--value-precision',type=float,default=1.5)
    p.add_argument('--noise-coupling',choices=('cartesian','role'),default='cartesian')
    p.add_argument('--differential-decoder',action='store_true')
    p.add_argument('--full-value',action='store_true')
    p.add_argument('--adaptive-metric',action='store_true')
    p.add_argument('--tail-checkpoint',type=Path)
    p.add_argument('--adapted-source',type=Path)
    p.add_argument('--adapted-train-states',type=int,default=64)
    p.add_argument('--adapted-test-states',type=int,default=32)
    p.add_argument('--fine-tune-relations',action='store_true')
    p.add_argument('--response-kernel',action='store_true')
    p.add_argument('--honest-forest',action='store_true')
    p.add_argument('--warm-candidate',type=Path)
    p.add_argument('--equal-context',action='store_true')
    p.add_argument('--competing-risks',action='store_true')
    p.add_argument('--pretrain-updates',type=int,default=1500)
    p.add_argument('--start-states',action='store_true')
    p.add_argument('--stop-source-jacobian',action='store_true')
    p.add_argument('--degree-feature',action='store_true')
    p.add_argument('--modal-fusion',action='store_true')
    p.add_argument('--fusion-link',choices=('odds','probability','exchange'))
    p.add_argument('--project-conflicts',action='store_true')
    p.add_argument('--paired-step-check',action='store_true')
    p.add_argument('--check-interval',type=int,default=5)
    p.add_argument('--check-batch-size',type=int,default=16)
    p.add_argument('--check-initial-only',action='store_true')
    p.add_argument('--team-tolerance',type=float,default=0.)
    p.add_argument('--pathwise-lr',type=float,default=1e-4)
    p.add_argument('--rescue-weight',type=float,default=0.)
    p.add_argument('--adjoint-horizon',type=int,default=0)
    p.add_argument('--opportunity-weight',type=float,default=0.)
    p.add_argument('--opportunity-width',type=float,default=1.)
    p.add_argument('--functional-natural',action='store_true')
    p.add_argument('--control-step',type=float,default=.01)
    p.add_argument('--natural-damping',type=float,default=1e-3)
    p.add_argument('--natural-iterations',type=int,default=8)
    p.add_argument('--primary-weight',type=float,default=1.)
    p.add_argument('--primary-only-check',action='store_true')
    p.add_argument('--value-confidence',type=float,default=0.)
    p.add_argument('--exploration-sigma',type=float,default=.3)
    p.add_argument('--credit-trace',type=float,default=.97)
    p.add_argument('--score-epochs',type=int,default=4)
    p.add_argument('--score-kl',type=float,default=.015)
    p.add_argument('--value-lr',type=float,default=3e-4)
    p.add_argument('--score-option-steps',type=int,default=1)
    p.add_argument('--score-contrast-roots',type=int,default=0)
    p.add_argument('--score-contrast-repeats',type=int,default=4)
    p.add_argument('--score-contrast-radius',type=float,default=.6)
    p.add_argument('--score-contrast-weight',type=float,default=2.)
    p.add_argument('--candidate-projection',action='store_true')
    p.add_argument('--response-projection',action='store_true')
    p.add_argument('--event-likelihood',action='store_true')
    p.add_argument('--paired-policy-credit',action='store_true')
    p.add_argument('--credit-roots',type=int,default=64)
    p.add_argument('--credit-repeats',type=int,default=4)
    p.add_argument('--coordination-solver',choices=('convex','signed'),default='convex')
    p.add_argument('--factor-data',type=Path)
    p.add_argument('--event-factors',action='store_true')
    p.add_argument('--reset-factor-policy',action='store_true')
    p.add_argument('--mediated-control-variate',action='store_true')
    p.add_argument('--temporal-event-backup',action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='fit_corrected_factor_policy':
        fit_corrected_factor_policy(args)
        return
    if args.mode in ('fit_cooperative_factors','collect_factor_outcomes'):
        fit_cooperative_factors(args)
        return
    if args.mode=='fit_intent_refinement':
        fit_intent_refinement(args)
        return
    if args.mode == 'fit':
        fit(args)
        return
    if args.mode == 'fit_contrast':
        fit_local_contrast(args)
        return
    if args.mode == 'fit_conditional_value':
        fit_conditional_value(args)
        return
    if args.mode == 'fit_candidate_intervention':
        fit_candidate_intervention(args)
        return
    if args.mode == 'fit_pathwise_intervention':
        fit_pathwise_intervention(args)
        return
    if args.mode == 'fit_score_intervention':
        fit_score_intervention(args)
        return
    if args.mode == 'fit_feedback_value':
        fit_feedback_value(args)
        return
    if args.mode == 'fit_capture_responsibility':
        fit_capture_responsibility(args)
        return
    if args.mode == 'fit_continuation':
        fit_continuation(args)
        return
    if args.mode == 'fit_paired_continuation':
        fit_paired_continuation(args)
        return
    if args.mode == 'fit_response_value':
        fit_response_value(args)
        return
    if args.mode == 'optimize_odds':
        optimize_odds(args)
        return
    if args.mode == 'fit_mediated':
        fit_mediated(args)
        return
    if args.mode == 'fit_intent':
        fit_intent(args)
        return
    if args.mode == 'fit_cooperation':
        fit_cooperation(args)
        return
    if args.mode == 'fit_coverage':
        fit_coverage(args)
        return
    if args.mode == 'fit_factorized':
        fit_factorized(args)
        return
    if args.mode == 'calibrate':
        calibrate_readout(args)
        return
    if args.mode == 'probe':
        probe_readout(args)
        return
    if args.mode == 'probe_modes':
        probe_modes(args)
        return
    if args.mode == 'probe_confidence':
        probe_confidence(args)
        return
    if args.mode=='tune_constant':
        tune_constant(args)
        return
    if args.mode=='tune_fusion':
        tune_fusion(args)
        return
    checkpoint = (args.graph_checkpoint or args.output/f'candidate-{args.seed}.pth') if args.mode=='evaluate' else (args.graph_checkpoint if args.mode in ('collect_feedback','collect_adapted_context','collect_quadrature') else None)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),
            initializer=initialize,initargs=(args.source,checkpoint,args.tail_checkpoint,args.fusion_scale,args.fusion_period,args.fusion_bias,args.candidate_probe,args.reference_checkpoint)) as pool:
        if args.mode=='collect_source_episode':
            tasks=[]
            for bank,(split,count) in enumerate((('train',args.train_states),('test',args.test_states))):
                (args.output/split).mkdir(parents=True,exist_ok=True)
                for root in range(count):
                    if not (args.output/split/f'{root}.pth').exists():tasks.append((root,bank,args.output/split,args.defenders,args.exact_agility))
            futures=[pool.submit(collect_source_episode,t) for t in tasks]
            for i,future in enumerate(as_completed(futures),1):print(dict(completed=i,total=len(tasks),**future.result()),flush=True)
        elif args.mode=='collect_candidate_contrast':
            tasks=[]
            for bank,(split,count) in enumerate((('train',args.train_states),('development',args.test_states))):
                (args.output/split).mkdir(parents=True,exist_ok=True)
                for root in range(count):
                    if not (args.output/split/f'{root}.pth').exists():
                        tasks.append((root,bank,args.state_source/('train' if bank==0 else 'test'),args.output/split,args.horizon,args.repeats,args.radius))
            futures=[pool.submit(collect_candidate_contrast,t) for t in tasks]
            for i,future in enumerate(as_completed(futures),1):print(dict(completed=i,total=len(tasks),**future.result()),flush=True)
        elif args.mode=='collect_quadrature':
            tasks=[]
            for bank,(split,count) in enumerate((('train',args.train_states),('development',args.test_states))):
                (args.output/split).mkdir(parents=True,exist_ok=True)
                for root in range(count):
                    if not (args.output/split/f'{root}.pth').exists():tasks.append((root,bank,args.state_source/('train' if bank==0 else 'test'),args.output/split,args.horizon))
            futures=[pool.submit(collect_quadrature_root,t) for t in tasks]
            for i,future in enumerate(as_completed(futures),1):print(dict(completed=i,total=len(tasks),**future.result()),flush=True)
        elif args.mode=='collect_adapted_context':
            tasks=[]
            for bank,(split,count) in enumerate((('train',args.train_states),('test',args.test_states))):
                (args.output/split).mkdir(parents=True,exist_ok=True)
                for root in range(count):
                    if not (args.output/split/f'{root}.pth').exists():tasks.append((root,bank,args.output/split,args.defenders))
            futures=[pool.submit(collect_adapted_context,t) for t in tasks]
            for i,future in enumerate(as_completed(futures),1):print(dict(completed=i,total=len(tasks),**future.result()),flush=True)
        elif args.mode=='collect_modal':
            tasks=[]
            for bank,(split,count) in enumerate((('train',args.train_states),('development',args.test_states))):
                (args.output/split).mkdir(parents=True,exist_ok=True)
                for root in range(count):
                    if not (args.output/split/f'{root}.pth').exists():
                        tasks.append((root,bank,args.state_source/('train' if bank==0 else 'test'),args.output/split,args.horizon,args.repeats,args.radius))
            futures=[pool.submit(collect_modal_root,t) for t in tasks]
            for i,future in enumerate(as_completed(futures),1):print(dict(completed=i,total=len(tasks),**future.result()),flush=True)
        elif args.mode=='collect_feedback':
            tasks=[]
            for bank,(split,count) in enumerate((('train',args.train_states),('development',args.test_states))):
                for root in range(count):
                    if not (args.output/split/f'{root}.pth').exists():
                        source=args.state_source/('train' if bank==0 else 'test') if args.state_source else None
                        tasks.append((root,bank,source,args.output/split,args.horizon,args.repeats,args.defenders))
                (args.output/split).mkdir(parents=True,exist_ok=True)
            futures=[pool.submit(collect_feedback_root,t) for t in tasks]
            for i,future in enumerate(as_completed(futures),1):print(dict(completed=i,total=len(tasks),**future.result()),flush=True)
        elif args.mode=='collect_active_value':
            tasks=active_value_tasks(args)
            futures=[pool.submit(collect_value_root,t) for t in tasks]
            for i,future in enumerate(as_completed(futures),1):print(dict(completed=i,total=len(tasks),**future.result()),flush=True)
        elif args.mode=='augment_value':
            tasks=[]
            for split,count in (('train',args.train_states),('development',args.test_states)):
                for root in range(count):
                    tasks.append((args.output/split/f'{root}.pth',args.state_source/('train' if split=='train' else 'test')))
            futures=[pool.submit(augment_value_root,t) for t in tasks]
            for i,future in enumerate(as_completed(futures),1):print(dict(completed=i,total=len(tasks),**future.result()),flush=True)
        elif args.mode=='collect_value':
            tasks=[]
            for bank,(split,count) in enumerate((('train',args.train_states),('development',args.test_states))):
                for root in range(count):
                    path=args.output/split/f'{root}.pth'
                    if path.exists():continue
                    tasks.append((root,bank,args.state_source/('train' if bank==0 else 'test'),
                        args.context_source/split,args.output/split,args.horizon,args.radius,args.max_value_groups,args.value_precision,args.noise_coupling))
            futures=[pool.submit(collect_value_root,t) for t in tasks]
            for i,future in enumerate(as_completed(futures),1):print(dict(completed=i,total=len(tasks),**future.result()),flush=True)
        elif args.mode=='collect':
            tasks = []
            for bank,(split,count) in enumerate((('train',args.train_states),('development',args.test_states))):
                for i in range(count):
                    existing = args.output/split/f'{i}.pth'
                    if existing.exists():
                        saved = torch.load(existing,weights_only=False,map_location='cpu')
                        if saved.get('intervention_radius') != args.intervention_radius:
                            raise ValueError('Intervention radius differs from the existing bank.')
                        if saved.get('antithetic',False)!=args.antithetic or saved.get('prefix_only',False)!=args.prefix_only:
                            raise ValueError('The existing bank has a different sampling design.')
                        if saved['repeats']>=args.repeats and 'target_response' in saved['rows'][0][0]: continue
                    source = args.state_source/('train' if bank==0 else 'test') if args.state_source else None
                    tasks.append((i,bank,args.horizon,args.repeats,args.output/split,source,args.intervention_radius,args.antithetic,args.prefix_only,args.defenders))
            futures = [pool.submit(collect_root,task) for task in tasks]
            for index,future in enumerate(as_completed(futures),1):
                print(dict(completed=index,total=len(futures),**future.result()),flush=True)
        else:
            tasks = [(2710000000+args.bank*100000+index,args.agility,args.defenders) for index in range(args.episodes)]
            rows = []
            for index,row in enumerate(pool.map(evaluate_scene,tasks),1):
                rows.append(row)
                if index%8==0:
                    print(dict(completed=index,episodes=args.episodes,
                        time_difference=float(np.mean([r['candidate']['capture_time']-r['baseline']['capture_time'] for r in rows]))),flush=True)
            delta = np.array([r['candidate']['capture_time']-r['baseline']['capture_time'] for r in rows])
            rng = np.random.default_rng(792)
            ci = np.quantile(np.mean(rng.choice(delta,size=(10000,len(delta))),axis=1),[.025,.975])
            summary = dict(time_difference=float(delta.mean()),ci=ci.tolist(),
                baseline={k:float(np.mean([r['baseline'][k] for r in rows])) for k in rows[0]['baseline']},
                candidate={k:float(np.mean([r['candidate'][k] for r in rows])) for k in rows[0]['candidate']})
            suffix = '' if args.defenders==3 else f'-n{args.defenders}'
            if args.agility!=2.25:suffix+=f'-a{args.agility:g}'
            if args.fusion_scale is not None:suffix+=f'-k{args.fusion_scale:g}'
            if args.fusion_period is not None:suffix+=f'-p{args.fusion_period}'
            if args.fusion_bias!=0.:suffix+=f'-b{args.fusion_bias:g}'
            if args.candidate_probe!='aligned':suffix+=f'-{args.candidate_probe}'
            if args.reference_checkpoint is not None:suffix+=f'-vs-{args.reference_checkpoint.stem}'
            atomic_json(args.output/f'evaluation-{args.seed}-bank{args.bank}{suffix}.json',dict(
                checkpoint=str(checkpoint),fusion_scale=args.fusion_scale,fusion_period=args.fusion_period,fusion_bias=args.fusion_bias,
                reference_checkpoint=str(args.reference_checkpoint) if args.reference_checkpoint else None,
                candidate_probe=args.candidate_probe,purpose='task-evaluation' if args.candidate_probe=='aligned' else 'inference-only mechanism diagnostic',summary=summary,rows=rows))
            print(summary,flush=True)


if __name__ == '__main__':
    main()
