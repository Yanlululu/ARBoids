"""Develop candidate-interaction fusion on the untouched author environment.

The frozen baseline calls the author's ActorAdap, SAC.choose_action and
TADEnv.step('AdaRes'). Every variant uses the original convex-mixture executor.
Gate-only variants retain both original proposals. Explicit proposal variants
fit a shared motion-control head in the new method; the baseline stays frozen.
Development searches and conditional-return fitting never use confirmation banks.
"""
import study_runtime
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from source_arboids import SourcePolicy, numerical_source, SOURCE_COMMIT

SOURCE = Path(r'D:\ARBoids\train\experiments\main-seed42-20260921\adares1.pth')
BASELINE_SHA256 = 'fe0c34101379582eac83ec6e89b4090d754308b604a97babf777f5b01d6c6ca2'
ROOT = Path(__file__).resolve().parent / 'experiments' / 'author-source-fusion-20261011'
CELLS = ((3, 2.), (3, 2.5), (6, 2.), (6, 2.5))
FEATURES = ('bias', 'attacker_distance', 'neighbor_proximity', 'closing',
            'risk_LL', 'risk_BL', 'risk_LB', 'risk_BB', 'relative_candidate_risk',
            'candidate_pair_interaction', 'turn_disagreement', 'thrust_disagreement')
FEATURES += ('original_gate_confidence', 'crowded_gate_confidence')
REFINER_FEATURES = ('bias','conditional_candidate_risk','current_joint_risk',
                    'mixture_curvature','risk_weighted_preference','prior_conditioned_preference')
BASE = None
PEER_CANDIDATES = True


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def initialize(source, without_peer=False):
    global BASE, PEER_CANDIDATES
    PEER_CANDIDATES = not without_peer
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError('The author-protocol reproduction baseline is frozen for this study.')
    BASE = SourcePolicy(source)


def relations(env, obs, base):
    """Current public motion and both same-cycle commands; no future labels."""
    n = len(base)
    motion = np.asarray([[b.x, b.y, b.theta, *b.velocity] for b in env.defender_list])
    p, v, heading = motion[:, :2], motion[:, 3:5], motion[:, 2]
    learned = 750. * base[:, :2] + 250.
    boids = env.boids_actions.copy()
    candidates = np.stack((learned, boids), 1)
    direction = np.stack((np.cos(heading), np.sin(heading)), -1)
    side = np.stack((-np.sin(heading), np.cos(heading)), -1)
    # A short kinematic response encoding, used only as policy information.
    # It is not an executed force, a safety filter, or an online return search.
    forward_speed = (v * direction).sum(-1)
    acc = (candidates.sum(-1) - ((100. + 150. * abs(forward_speed)) * forward_speed)[:, None]) / 380.
    yaw = 1.25 * (candidates[..., 0] - candidates[..., 1]) / 1430.
    response = p[:, None] + 1.2 * v[:, None] + .5 * 1.2**2 * acc[..., None] * direction[:, None]
    response += .3 * yaw[..., None] * side[:, None]
    delta = p[:, None] - p[None]
    distance = np.linalg.norm(delta, axis=-1)
    off = 1. - np.eye(n)
    near = np.exp(-distance**2 / 100.) * off
    closing = -(delta * (v[:, None] - v[None])).sum(-1) / np.maximum(distance, 1.)
    risks = []
    for own, peer in ((0, 0), (1, 0), (0, 1), (1, 1)):
        peer_response = response[:,peer] if PEER_CANDIDATES else p+1.2*v
        d = response[:, own, None] - peer_response[None]
        risks.append(np.exp(-(d*d).sum(-1) / 50.) * off)
    ll, bl, lb, bb = risks
    pool = lambda a: a.sum(-1) / max(1, n-1)
    # Taking the nearest risk as well as its mean keeps dense encounters visible.
    pool_risk = lambda a: .5 * (pool(a) + a.max(-1))
    theta = np.clip(base[:, 2], 1e-6, 1.-1e-6)
    confidence = np.log(theta/(1.-theta)) / 6.
    f = np.stack((np.ones(n), np.clip(obs[:, 2] / 60., 0., 2.), pool(near),
        pool(near * np.clip(closing / 5., -2., 2.)),
        pool_risk(ll), pool_risk(bl), pool_risk(lb), pool_risk(bb),
        pool_risk(.5 * (ll + lb)) - pool_risk(.5 * (bl + bb)),
        pool(ll + bb - lb - bl),
        ((learned[:, 0]-learned[:, 1])-(boids[:, 0]-boids[:, 1])) / 1500.,
        (learned.sum(-1)-boids.sum(-1)) / 3000.,
        confidence, confidence*near.max(-1)), -1)
    return np.clip(f, -3., 3.).astype(np.float32)


def policy_weights(record):
    weights = np.asarray(record['weights'], dtype=np.float32)
    return np.pad(weights, (0, len(FEATURES)-len(weights)))


def controller_spec(record):
    if record.get('controller') in ('trajectory','proposal'):
        return dict(controller=record['controller'],settings=record.get('settings',{}),
                    proposal=record.get('proposal'),
                    weights=policy_weights(record),
                    refiner=np.asarray(record.get('refiner_weights',np.zeros(6)),dtype=np.float32))
    weights=policy_weights(record)
    if record.get('refiner_weights') is not None:
        return dict(weights=weights,refiner=np.asarray(record['refiner_weights'],dtype=np.float32))
    return weights


def composition_features(position,velocity,heading,learned,boids,prior,without_peer=False):
    """Encode candidate responses conditional on concurrent nominal mixtures."""
    n=len(prior)
    direction=np.stack((np.cos(heading),np.sin(heading)),-1)
    side=np.stack((-np.sin(heading),np.cos(heading)),-1)
    candidates=np.stack((learned,boids),1)
    speed=(velocity*direction).sum(-1)
    acceleration=(candidates.sum(-1)-((100.+150.*abs(speed))*speed)[:,None])/380.
    yaw=1.25*(candidates[...,0]-candidates[...,1])/1430.
    response=position[:,None]+1.2*velocity[:,None]+.5*1.2**2*acceleration[...,None]*direction[:,None]
    response+=.3*yaw[...,None]*side[:,None]
    current=prior[:,None]*response[:,0]+(1.-prior[:,None])*response[:,1]
    peer=position+1.2*velocity if without_peer else current
    off=1.-np.eye(n)
    def risk(own):
        delta=own[:,None]-peer[None]
        values=np.exp(-(delta*delta).sum(-1)/50.)*off
        return .5*(values.sum(-1)/max(1,n-1)+values.max(-1))
    left,right,nominal=risk(response[:,0]),risk(response[:,1]),risk(current)
    difference=left-right
    theta=np.clip(prior,1e-6,1.-1e-6)
    confidence=np.log(theta/(1.-theta))/6.
    return np.stack((np.ones(n),difference,nominal,left+right-2.*nominal,
                     nominal*difference,confidence*difference),-1).astype(np.float32)


def live_composition_features(env,base,prior):
    motion=np.asarray([[b.x,b.y,*b.velocity,b.theta] for b in env.defender_list])
    return composition_features(motion[:,:2],motion[:,2:4],motion[:,5],
                                750.*base[:,:2]+250.,env.boids_actions,prior,
                                without_peer=not PEER_CANDIDATES)


def critic_context(env, base, features):
    """Centralized measured state, expressed in the target-attacker frame."""
    angle = np.arctan2(env.attacker.y, env.attacker.x)
    rotation = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle),np.cos(angle)]])
    motion = np.array([[b.x,b.y,*b.velocity,b.theta] for b in env.defender_list])
    position,velocity = motion[:,:2]@rotation/60.,motion[:,2:4]@rotation/5.
    heading = motion[:,5]-angle
    attacker = env.attacker
    common = np.array([np.linalg.norm(attacker.pos)/60.,
        *(attacker.velocity[:2]@rotation/5.),np.cos(attacker.theta-angle),
        np.sin(attacker.theta-angle),attacker.velocity[2]/2.])
    return np.concatenate((features,position,velocity,motion[:,4:5]/2.,
        np.cos(heading)[:,None],np.sin(heading)[:,None],base,
        (env.boids_actions-250.)/750.,np.broadcast_to(common,(len(base),len(common)))),axis=-1).astype(np.float32)


def gate_action(base, features, weights):
    action = base.copy()
    if np.any(weights):
        offset = features @ np.asarray(weights, dtype=np.float32)
        # Odds adjustment preserves the scalar learned/Boids convex mixture.
        theta = np.clip(base[:, 2], 1e-6, 1.-1e-6)
        z = np.log(theta / (1.-theta)) + np.clip(offset, -12., 12.)
        action[:, 2] = 1. / (1. + np.exp(-z))
    return action


def decision(env, obs, weights):
    if (isinstance(weights,dict) and weights.get('proposal') is not None
            and weights['proposal'].get('blend',1.)==1.
            and (weights.get('controller')=='proposal' or 'gate_prior' in weights.get('settings',{}))):
        # This fitted motion head completely replaces the proposal, so evaluating
        # the frozen backbone would not affect the executed action. Baseline calls
        # still go through the unmodified SourcePolicy in episode().
        base=motion_proposal(env,np.zeros((len(obs),3),dtype=np.float32),weights['proposal'])
        action=base.copy()
        if weights.get('controller')=='proposal':action[:,2]=weights['proposal'].get('gate',1.)
        else:
            period=weights['settings'].get('decision_steps',1)
            slot=round(env.Current_T/.2)//period
            cached=getattr(env,'_fusion_commit',None)
            if period>1 and cached is not None and cached[0]==slot:
                action[:,2]=cached[1]
            else:
                prior=np.full(len(obs),weights['settings']['gate_prior'])
                action[:,2]=trajectory_gates(env,base,prior,weights['settings'])
                if period>1:env._fusion_commit=(slot,action[:,2].copy())
        return action,base,np.zeros((len(obs),len(FEATURES)),dtype=np.float32)
    base = BASE(obs)
    if isinstance(weights,dict) and weights.get('proposal') is not None:
        base=motion_proposal(env,base,weights['proposal'])
    f = relations(env, obs, base)
    if isinstance(weights,dict):
        if weights.get('controller')=='proposal':
            action=base.copy();action[:,2]=weights['proposal'].get('gate',1.)
            return action,base,f
        action=gate_action(base,f,weights['weights'])
        if np.any(weights['refiner']):
            context=live_composition_features(env,base,action[:,2])
            theta=np.clip(action[:,2],1e-6,1.-1e-6)
            z=np.log(theta/(1.-theta))+np.clip(context@weights['refiner'],-6.,6.)
            action[:,2]=1./(1.+np.exp(-z))
        if weights.get('controller') == 'trajectory':
            if 'gate_prior' in weights['settings']:
                action[:,2]=weights['settings']['gate_prior']
            action[:,2]=trajectory_gates(env,base,action[:,2],weights['settings'])
    else:
        action=gate_action(base,f,weights)
    return action,base,f


def motion_proposal(env,base,parameters):
    """A shared, trainable motion head for the new method's control proposal."""
    motion=np.asarray([[b.x,b.y,b.theta,*b.velocity] for b in env.defender_list])
    position,heading,velocity=motion[:,:2],motion[:,2],motion[:,3:5]
    delta=position[:,None]-position[None]
    distance=np.linalg.norm(delta,axis=-1)+np.eye(len(base))*1e4
    relative=velocity[:,None]-velocity[None]
    direction=delta/np.maximum(distance[...,None],1e-6)
    closing=-(direction*relative).sum(-1)
    predicted=distance-parameters.get('anticipation',.6)*np.maximum(closing,0.)
    separation=(direction*np.maximum(0.,parameters.get('space',8.)-predicted)[...,None]**2).sum(1)
    target=parameters.get('intercept',1.)*env.attacker.pos+parameters.get('lead',0.)*env.attacker.velocity[:2]-position
    target/=np.maximum(np.linalg.norm(target,axis=-1,keepdims=True),1e-6)
    desired=target+parameters.get('separation',.4)*separation
    desired_heading=np.arctan2(desired[:,1],desired[:,0])
    angle=(desired_heading-heading+np.pi)%(2*np.pi)-np.pi
    speed=(velocity*np.stack((np.cos(heading),np.sin(heading)),-1)).sum(-1)
    target_speed=parameters.get('speed',3.)*np.cos(angle)
    surge=380.*parameters.get('surge',1.)*(target_speed-speed)+(100.+150.*abs(speed))*speed
    r=motion[:,5]
    moment=1430.*(parameters.get('turn',1.)*angle-parameters.get('damping',1.5)*r)
    moment+=(980.+950.*abs(r))*r
    force=np.stack(((surge+moment/1.25)*.5,(surge-moment/1.25)*.5),-1)
    force=np.clip(force,-500.,1000.)
    blend=parameters.get('blend',1.)
    output=base.copy();output[:,:2]=(1.-blend)*base[:,:2]+blend*(force-250.)/750.
    return output


def proposal_search(args,pool):
    initial=json.loads(args.start.read_text())
    spec=controller_spec(initial)
    if not isinstance(spec,dict):spec=dict(weights=spec,refiner=np.zeros(6))
    spec['controller']='proposal'
    scenes=bank(args.bank,args.per_cell);rng=np.random.default_rng(args.seed)
    best_rows=None;history=[]
    default=dict(separation=.4,anticipation=.6,space=8.,lead=0.,speed=3.,surge=1.,turn=1.,damping=1.5,blend=1.,gate=1.)
    best_parameters=initial.get('proposal') or default
    choices=dict(separation=(.05,.1,.2,.4,.8,1.6,3.),anticipation=(0.,.3,.6,1.,1.5,2.),space=(6.,7.,8.,9.,10.,12.),
        lead=(0.,1.,2.,3.),speed=(1.5,2.,2.5,3.,3.5,4.),surge=(.5,1.,2.,3.),turn=(.3,.5,1.,1.5,2.),
        damping=(.5,1.,1.5,2.,3.),blend=(.5,.75,1.),gate=(.5,.75,1.),intercept=(.3,.5,.6,.7,.8,1.))
    for iteration in range(args.iterations):
        parameters=best_parameters.copy()
        if iteration:
            if iteration<=5:
                parameters['intercept']=(.3,.5,.6,.7,.8)[iteration-1]
            else:
                for key in rng.choice(list(choices),size=2,replace=False):
                    parameters[key]=float(rng.choice(choices[key]))
        spec['proposal']=parameters
        rows=evaluate(pool,scenes,spec)
        accepted=best_rows is None or score(rows)>score(best_rows)
        if accepted:best_rows,best_parameters=rows,parameters.copy()
        summary=summarize(rows)
        history.append(dict(iteration=iteration,proposal=parameters,summary=summary,accepted=accepted))
        print(dict(iteration=iteration,success=summary['success'],collision=summary['collision'],
                   best=summarize(best_rows)['success'],proposal=parameters),flush=True)
        save_json(args.output/'policy.json',dict(controller='proposal',weights=spec['weights'].tolist(),
            refiner_weights=spec['refiner'].tolist(),proposal=best_parameters,without_peer=args.without_peer,
            source_commit=SOURCE_COMMIT,source_sha256=BASELINE_SHA256,bank=args.bank,training_seed=args.seed,
            selected=summarize(best_rows),selected_rows=best_rows,trials=history,
            status='return-selected motion proposal initialization; not a two-innovation result'))


def predicted_feedback_commands(state,peer_position,peer_velocity,attacker,attacker_velocity,parameters):
    n,k=state.shape[:2]
    difference=state[:,None,:,:2]-peer_position[None,:,None,:]
    distance=np.linalg.norm(difference,axis=-1)+np.eye(n)[:,:,None]*1e4
    unit=difference/np.maximum(distance[...,None],1e-6)
    closing=-(unit*(state[:,None,:,3:5]-peer_velocity[None,:,None,:])).sum(-1)
    separation=(unit*np.maximum(0.,parameters.get('space',8.)-distance+
        parameters.get('anticipation',.6)*np.maximum(closing,0.))[...,None]**2).sum(1)
    goal=parameters.get('intercept',1.)*attacker+parameters.get('lead',0.)*attacker_velocity-state[...,:2]
    goal/=np.maximum(np.linalg.norm(goal,axis=-1,keepdims=True),1e-6)
    desired=goal+parameters.get('separation',.4)*separation
    angle=(np.arctan2(desired[...,1],desired[...,0])-state[...,2]+np.pi)%(2*np.pi)-np.pi
    c,s=np.cos(state[...,2]),np.sin(state[...,2])
    speed=state[...,3]*c+state[...,4]*s;r=state[...,5]
    surge=380.*parameters.get('surge',1.)*(parameters.get('speed',3.)*np.cos(angle)-speed)+(100.+150.*abs(speed))*speed
    moment=1430.*(parameters.get('turn',1.)*angle-parameters.get('damping',1.5)*r)+(980.+950.*abs(r))*r
    learned=np.clip(np.stack(((surge+moment/1.25)*.5,(surge-moment/1.25)*.5),-1),-500.,1000.)
    separation=(difference/np.maximum(distance-3.5,.01)[...,None]**2*(distance<15.)[...,None]).sum(1)
    alignment=(peer_velocity.sum(0)[None,None]-peer_velocity[:,None]+state[...,3:5])/n
    cohesion=(peer_position.sum(0)[None,None]-peer_position[:,None]+state[...,:2])/n-state[...,:2]
    field=.5*(attacker-state[...,:2])+10.*separation+.1*alignment+.1*cohesion
    field/=np.maximum(np.linalg.norm(field,axis=-1,keepdims=True),1e-6)
    forward=field[...,0]*c+field[...,1]*s
    side=-field[...,0]*s+field[...,1]*c
    turn=np.arctan2(side,forward)
    boids=np.clip(1000.*np.stack((forward+.75*turn,forward-.75*turn),-1),-500.,1000.)
    return learned,boids


def candidate_trajectories(env,base,gates,horizon,dt=.1,settings=None):
    """Public measured motion propagated under each present mixture candidate.

    The nominal model uses ground velocity as its zero-current estimate; the
    simulator's water-relative velocity and future disturbances are never read.
    Predictions encode candidate intent and are not branch-return simulations.
    """
    motion=np.asarray([[b.x,b.y,b.theta,*b.velocity] for b in env.defender_list])
    state=np.broadcast_to(motion[:,None,:],(*gates.shape,6)).copy()
    force=env.boids_actions[:,None,:]+gates[...,None]*(750.*base[:,:2,None].transpose(0,2,1)+250.-env.boids_actions[:,None,:])
    paths=[]
    feedback=settings is not None and settings.get('response_model')=='feedback'
    for step in range(round(horizon/dt)):
        if feedback and step and step%2==0:
            t=step*dt
            learned,boids=predicted_feedback_commands(state,motion[:,:2]+t*motion[:,3:5],motion[:,3:5],
                env.attacker.pos+t*env.attacker.velocity[:2],env.attacker.velocity[:2],settings['proposal_model'])
            force=gates[...,None]*learned+(1.-gates[...,None])*boids
        state[...,:2]+=dt*state[...,3:5]
        state[...,2]+=dt*state[...,5]
        c,s=np.cos(state[...,2]),np.sin(state[...,2])
        u=c*state[...,3]+s*state[...,4]
        v=-s*state[...,3]+c*state[...,4]
        r=state[...,5]
        du=(400.*r*v-(100.+150.*abs(u))*u+force.sum(-1))/380.
        dv=(-380.*r*u-(100.+150.*abs(v))*v)/400.
        dr=(-20.*u*v-(980.+950.*abs(r))*r+1.25*(force[...,0]-force[...,1]))/1430.
        u=u+dt*du;v=v+dt*dv
        state[...,3]=c*u-s*v;state[...,4]=s*u+c*v
        state[...,5]+=dt*dr
        if (step+1)%2==0:
            paths.append(state[...,:2].copy())
    return np.stack(paths,axis=2),state


def trajectory_tables(env,base,prior,settings):
    n=len(base)
    gates=np.broadcast_to(np.linspace(0.,1.,settings.get('options',9)),(n,settings.get('options',9))).copy()
    gates=np.concatenate((prior[:,None],gates),axis=1)
    horizon=settings.get('horizon',3.)
    paths,state=candidate_trajectories(env,base,gates,horizon,settings=settings)
    times=np.arange(1,paths.shape[2]+1)*.2
    pos=np.array([b.pos for b in env.defender_list])
    vel=np.array([b.velocity[:2] for b in env.defender_list])
    peers=paths if PEER_CANDIDATES else np.broadcast_to(
        (pos[:,None,None,:]+times[None,None,:,None]*vel[:,None,None,:]),paths.shape)
    delta=paths[:,None,:,None]-peers[None,:,None,:]
    distance=np.linalg.norm(delta,axis=-1).min(-1)
    safe=settings.get('clearance',6.5)
    pair=np.maximum(0.,safe-distance)**2
    pair+=settings.get('tail',.1)*np.exp(-(distance-5.)/2.)
    pair[np.arange(n),np.arange(n)]=0.
    # Terminal bearing/progress are shared features, not privileged attack intent.
    target=env.attacker.pos+settings.get('lead',0.)*env.attacker.velocity[:2]
    direction=target-pos
    direction/=np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-6)
    progress=((state[...,:2]-pos[:,None,:])*direction[:,None,:]).sum(-1)
    align=(state[...,3:5]*direction[:,None,:]).sum(-1)
    unary=-settings.get('progress',1.)*progress-settings.get('align',0.)*align
    unary+=settings.get('prior',.2)*(gates-prior[:,None])**2
    future_attacker=env.attacker.pos+horizon*env.attacker.velocity[:2]
    auxiliary=dict(attacker_distance=np.linalg.norm(state[...,:2]-future_attacker,axis=-1),
        radius=np.linalg.norm(state[...,:2],axis=-1),attacker_radius=np.linalg.norm(future_attacker),
        paths=paths,position=pos,velocity=vel,times=times,
        attacker_paths=env.attacker.pos+times[:,None]*env.attacker.velocity[:2],
        attacker_velocity=env.attacker.velocity[:2].copy(),
        time=float(env.Current_T)/80.,motion=np.array([[b.theta,*b.velocity] for b in env.defender_list]),
        candidates=np.concatenate((base[:,:2],(env.boids_actions-250.)/750.),axis=-1))
    return gates,unary,settings.get('risk',20.)*pair,auxiliary


def joint_response_paths(choices,auxiliary,gates,settings):
    """Nominal joint feedback response using public motion, not simulator futures."""
    key=choices.tobytes();cache=auxiliary.setdefault('_joint_response_cache',{})
    if key in cache:return cache[key]
    m,n=choices.shape;dt=.1;parameters=settings['proposal_model']
    motion=np.concatenate((auxiliary['position'],auxiliary['motion']),axis=-1)
    state=np.broadcast_to(motion,(m,n,6)).copy()
    theta=gates[np.arange(n)[None],choices][...,None]
    current=250.+750.*auxiliary['candidates']
    force=theta*current[None,:,:2]+(1.-theta)*current[None,:,2:]
    av=auxiliary['attacker_velocity'];attacker=auxiliary['attacker_paths'][0]-auxiliary['times'][0]*av
    paths=[]
    for step in range(round(settings.get('horizon',1.)/dt)):
        if step and step%2==0:
            position=state[...,:2];velocity=state[...,3:5]
            difference=position[:,:,None]-position[:,None,:]
            distance=np.linalg.norm(difference,axis=-1)+np.eye(n)[None]*1e4
            unit=difference/np.maximum(distance[...,None],1e-6)
            closing=-(unit*(velocity[:,:,None]-velocity[:,None,:])).sum(-1)
            separation=(unit*np.maximum(0.,parameters['space']-distance+
                parameters['anticipation']*np.maximum(closing,0.))[...,None]**2).sum(2)
            target=attacker+step*dt*av
            goal=parameters['intercept']*target+parameters['lead']*av-position
            goal/=np.maximum(np.linalg.norm(goal,axis=-1,keepdims=True),1e-6)
            desired=goal+parameters['separation']*separation
            angle=(np.arctan2(desired[...,1],desired[...,0])-state[...,2]+np.pi)%(2*np.pi)-np.pi
            c,s=np.cos(state[...,2]),np.sin(state[...,2]);r=state[...,5]
            speed=velocity[...,0]*c+velocity[...,1]*s
            surge=380.*parameters['surge']*(parameters['speed']*np.cos(angle)-speed)+(100.+150.*abs(speed))*speed
            moment=1430.*(parameters['turn']*angle-parameters['damping']*r)+(980.+950.*abs(r))*r
            learned=np.clip(np.stack(((surge+moment/1.25)*.5,(surge-moment/1.25)*.5),-1),-500.,1000.)
            separation=(difference/np.maximum(distance-3.5,.01)[...,None]**2*(distance<15.)[...,None]).sum(2)
            field=.5*(target-position)+10.*separation+.1*velocity.mean(1,keepdims=True)
            field+=.1*(position.mean(1,keepdims=True)-position)
            field/=np.maximum(np.linalg.norm(field,axis=-1,keepdims=True),1e-6)
            forward=field[...,0]*c+field[...,1]*s
            turn=np.arctan2(-field[...,0]*s+field[...,1]*c,forward)
            boids=np.clip(1000.*np.stack((forward+.75*turn,forward-.75*turn),-1),-500.,1000.)
            force=theta*learned+(1.-theta)*boids
        state[...,:2]+=dt*state[...,3:5];state[...,2]+=dt*state[...,5]
        c,s=np.cos(state[...,2]),np.sin(state[...,2]);r=state[...,5]
        u=c*state[...,3]+s*state[...,4];v=-s*state[...,3]+c*state[...,4]
        du=(400.*r*v-(100.+150.*abs(u))*u+force.sum(-1))/380.
        dv=(-380.*r*u-(100.+150.*abs(v))*v)/400.
        dr=(-20.*u*v-(980.+950.*abs(r))*r+1.25*(force[...,0]-force[...,1]))/1430.
        u+=dt*du;v+=dt*dv
        state[...,3]=c*u-s*v;state[...,4]=s*u+c*v;state[...,5]+=dt*dr
        if (step+1)%2==0:paths.append(state[...,:2].copy())
    cache[key]=np.stack(paths,axis=2)
    return cache[key]


def trajectory_embedding(choices,auxiliary,gates,subject=None,angular=False,retain_gates=False,joint_paths=None):
    n=gates.shape[0];rows=np.arange(n)[None,:]
    indices=np.unique(np.linspace(0,len(auxiliary['times'])-1,4).astype(int))
    paths=(auxiliary['paths'][rows,choices] if joint_paths is None else joint_paths)[:,:,indices]
    if subject is not None:
        motion=auxiliary['position'][:,None]+auxiliary['times'][indices][None,:,None]*auxiliary['velocity'][:,None]
        own=paths[:,subject].copy();paths=np.broadcast_to(motion,paths.shape).copy();paths[:,subject]=own
    attacker=auxiliary['attacker_paths'][indices]
    i,j=np.triu_indices(n,1)
    distance=np.linalg.norm(paths[:,i]-paths[:,j],axis=-1)
    attack_distance=np.linalg.norm(paths-attacker[None,None],axis=-1)
    radii=np.linalg.norm(paths,axis=-1)
    features=[distance.min(1)/10.,np.exp(-distance/5.).mean(1),
        np.maximum(0.,7.-distance).max(1)/7.,np.maximum(0.,10.-distance).mean(1)/10.,
        attack_distance.min(1)/60.,attack_distance.mean(1)/60.,
        (radii.min(1)-np.linalg.norm(attacker,axis=-1)[None])/30.,radii.mean(1)/60.]
    if angular:
        direction=(paths-attacker[None,None])/np.maximum(attack_distance[...,None],1e-6)
        inward=-attacker/np.maximum(np.linalg.norm(attacker,axis=-1,keepdims=True),1e-6)
        coverage=(direction*inward[None,None]).sum(-1)
        features.extend((np.linalg.norm(direction.mean(1),axis=-1),coverage.mean(1),coverage.min(1),coverage.max(1)))
    selected=gates[rows,choices]
    if subject is not None and not retain_gates:
        own=selected[:,subject].copy();selected=np.zeros_like(selected);selected[:,subject]=own
    features.extend((np.full((len(choices),1),n/6.),np.full((len(choices),1),auxiliary['time']),
        selected.mean(-1,keepdims=True),selected.std(-1,keepdims=True)))
    return np.clip(np.concatenate(features,axis=-1),-4.,4.).astype(np.float32)


class TrajectoryValue(nn.Module):
    def __init__(self,dimensions):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dimensions,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)


class TrajectoryOutcomeValue(nn.Module):
    """Separate failure, capture and timeout probabilities under the continuation."""
    def __init__(self,dimensions,linear=False,mirror_canonical=False):
        super().__init__()
        self.mirror_canonical=mirror_canonical
        if mirror_canonical:
            sign=torch.ones(dimensions);sign[[53,55,57,59,61,63,65,67,78,79]]=-1
            if dimensions==96:sign[[89,91,93,95]]=-1
            order=torch.arange(dimensions);order[[80,81,82,83]]=order[[81,80,83,82]]
            self.register_buffer('mirror_sign',sign,persistent=False)
            self.register_buffer('mirror_order',order,persistent=False)
        self.net=(nn.Linear(dimensions,3) if linear else
            nn.Sequential(nn.Linear(dimensions,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3)))
    def forward(self,x):
        if self.mirror_canonical:
            # Reflection preserves the task and controller dynamics. Express
            # lateral motion, yaw and left/right forces on the attacker's side.
            reflected=x[:,53]<0
            x=torch.where(reflected[:,None],(x*self.mirror_sign)[:,self.mirror_order],x)
        return self.net(x).softmax(-1)


def trajectory_value_features(choices,auxiliary,gates,settings,subject=None,without_peer=None):
    use_peer=PEER_CANDIDATES if without_peer is None else not without_peer
    mask=subject if not use_peer else None
    joint_paths=(joint_response_paths(choices,auxiliary,gates,settings)
        if use_peer and settings.get('joint_response_features',False) else None)
    common=trajectory_embedding(choices,auxiliary,gates,mask,settings.get('angular_features',False),
        settings.get('gate_conditioning',False),joint_paths)
    if not settings.get('node_value',False):return common
    if subject is None:raise ValueError('A conditional node value needs the intervened vessel.')
    n=gates.shape[0];m=len(choices)
    indices=np.unique(np.linspace(0,len(auxiliary['times'])-1,4).astype(int))
    paths=(auxiliary['paths'][np.arange(n)[None,:],choices] if joint_paths is None else joint_paths)[:,:,indices]
    own=paths[:,subject]
    if not use_peer:
        motion=auxiliary['position'][:,None]+auxiliary['times'][indices][None,:,None]*auxiliary['velocity'][:,None]
        paths=np.broadcast_to(motion,paths.shape).copy();paths[:,subject]=own
    peer=np.arange(n)!=subject
    difference=own[:,None]-paths[:,peer]
    distance=np.linalg.norm(difference,axis=-1)
    target=auxiliary['attacker_paths'][indices][None]-own
    angle=auxiliary['motion'][subject,0]
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    local=target@rotation/30.
    relative=(difference@rotation/10.).mean(1)
    static=np.r_[auxiliary['velocity'][subject]@rotation/5.,auxiliary['motion'][subject,3]/2.,auxiliary['candidates'][subject]]
    extra=np.concatenate((local.reshape(m,-1),relative.reshape(m,-1),distance.min(1)/10.,
        np.exp(-distance/5.).mean(1),gates[subject,choices[:,subject,None]],np.broadcast_to(static,(m,len(static)))),axis=-1)
    if settings.get('gate_conditioning',False):
        selected=gates[np.arange(n)[None],choices]
        own_gate=selected[:,subject,None];peer_gates=selected[:,peer]
        displacement=(auxiliary['position'][peer]-auxiliary['position'][subject])@rotation/10.
        relative_velocity=(auxiliary['velocity'][peer]-auxiliary['velocity'][subject])@rotation/5.
        weight=np.exp(-np.linalg.norm(displacement,axis=-1));weight/=weight.sum()
        weighted=peer_gates*weight[None]
        gate_mean=peer_gates.mean(-1,keepdims=True);gate_near=weighted.sum(-1,keepdims=True)
        context=weighted@np.concatenate((displacement,relative_velocity),axis=-1)
        extra=np.concatenate((extra,gate_mean,own_gate*gate_mean,gate_near,own_gate*gate_near,
                              context,own_gate*context),axis=-1)
    return np.concatenate((common,np.clip(extra,-4.,4.)),axis=-1).astype(np.float32)


@lru_cache(maxsize=6)
def trajectory_value_models(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    models=[]
    for state in saved['models']:
        model=(TrajectoryOutcomeValue(saved['dimensions'],linear=saved.get('value_architecture')=='linear',
                                      mirror_canonical=saved.get('mirror_canonical',False))
            if saved.get('outcome_value',False) else TrajectoryValue(saved['dimensions']))
        model.load_state_dict(state);model.eval();models.append(model)
    return models


def trajectory_value(choices,auxiliary,gates,settings,subject=None):
    values=torch.tensor(trajectory_value_features(choices,auxiliary,gates,settings,subject))
    models=trajectory_value_models(settings['value_path'])
    prediction=torch.stack([model(values) for model in models])
    if settings.get('centered_uncertainty',False):
        reference=torch.tensor(trajectory_value_features(np.zeros((1,gates.shape[0]),dtype=int),auxiliary,gates,settings,subject))
        prediction-=torch.stack([model(reference) for model in models])
    return (prediction.mean(0)-settings.get('uncertainty',.5)*prediction.std(0)).detach().numpy()


def trajectory_gates(env,base,prior,settings):
    gates,unary,pair,auxiliary=trajectory_tables(env,base,prior,settings)
    n,k=gates.shape
    choice=np.zeros(n,dtype=int)
    if settings.get('value_path') and settings.get('pair_readout',False):
        reference=trajectory_gates(env,base,prior,{key:value for key,value in settings.items() if key!='value_path'})
        anchor=np.abs(gates-reference[:,None]).argmin(-1)
        position=auxiliary['position']
        attack_distance=np.linalg.norm(position-env.attacker.pos,axis=-1).min()
        if attack_distance>settings.get('feedback_radius',30.) and env.def_def_dists.min()>8.:
            return reference
        # The four-arm training design includes both orders of each paired move.
        # Read out one bounded plan, rather than iterating beyond trained anchors.
        flips=np.array([np.abs(gates[i]-(1. if reference[i]<.5 else 0.)).argmin() for i in range(n)])
        plans=[anchor.copy()]
        for i in range(n):
            plan=anchor.copy();plan[i]=flips[i];plans.append(plan)
        for i in range(n):
            for j in range(i+1,n):
                plan=anchor.copy();plan[i]=flips[i];plan[j]=flips[j];plans.append(plan)
        plans=np.array(plans);changed=plans!=anchor[None]
        models=trajectory_value_models(settings['value_path'])
        advantages=np.zeros((len(models),len(plans),3))
        cost=np.zeros(len(plans));future=position+auxiliary['times'][-1]*auxiliary['velocity']
        target=auxiliary['attacker_paths'][-1]
        for i in range(n):
            values=torch.tensor(trajectory_value_features(plans,auxiliary,gates,settings,i))
            prediction=torch.stack([model(values) for model in models]).detach().numpy()
            advantages+=(prediction-prediction[:,:1])*changed[None,:,i,None]
            if not PEER_CANDIDATES:
                peers=np.arange(n)!=i;indices=plans[:,i]
                local=unary[i,indices]+pair[i,:,:,0].sum(0)[indices]
                local+=settings.get('nearest',0.)*np.minimum(auxiliary['attacker_distance'][i,indices],np.linalg.norm(future[peers]-target,axis=-1).min())
                radius=np.minimum(auxiliary['radius'][i,indices],np.linalg.norm(future[peers],axis=-1).min())
                local+=settings.get('guard',0.)*np.maximum(0.,radius-auxiliary['attacker_radius']+5.)**2
                cost+=local-local[0]
        advantages/=np.maximum(changed.sum(-1),1)[None,:,None]
        if PEER_CANDIDATES:
            rows=np.arange(n)[None];i,j=np.triu_indices(n,1)
            cost=unary[rows,plans].sum(-1)+pair[i[None],j[None],plans[:,i],plans[:,j]].sum(-1)
            cost+=settings.get('nearest',0.)*auxiliary['attacker_distance'][rows,plans].min(-1)
            radius=auxiliary['radius'][rows,plans].min(-1)
            cost+=settings.get('guard',0.)*np.maximum(0.,radius-auxiliary['attacker_radius']+5.)**2
            cost-=cost[0]
        safety=-advantages[...,0];capture=advantages[...,1];uncertainty=settings.get('uncertainty',.5)
        permissible=safety.mean(0)-uncertainty*safety.std(0,ddof=1)>=-settings.get('safety_tolerance',0.)
        score=cost-settings.get('value_scale',10.)*(capture.mean(0)-uncertainty*capture.std(0,ddof=1))
        score=np.where(permissible,score,np.inf);score[0]=0.
        return gates[np.arange(n),plans[int(score.argmin())]]
    if settings.get('value_path') and settings.get('anchored_readout',False):
        reference_settings={key:value for key,value in settings.items() if key!='value_path'}
        reference=trajectory_gates(env,base,prior,reference_settings)
        anchor=np.abs(gates-reference[:,None]).argmin(-1)
        proposals=[];gains=[]
        future=auxiliary['position']+auxiliary['times'][-1]*auxiliary['velocity']
        target=auxiliary['attacker_paths'][-1]
        def physical(choices):
            rows=np.arange(n)[None];i,j=np.triu_indices(n,1)
            cost=unary[rows,choices].sum(-1)+pair[i[None],j[None],choices[:,i],choices[:,j]].sum(-1)
            cost+=settings.get('nearest',0.)*auxiliary['attacker_distance'][rows,choices].min(-1)
            cost+=settings.get('guard',0.)*np.maximum(0.,auxiliary['radius'][rows,choices].min(-1)-auxiliary['attacker_radius']+5.)**2
            return cost
        for i in range(n):
            choices=np.broadcast_to(anchor,(k,n)).copy();choices[:,i]=np.arange(k)
            queries=np.concatenate((choices,anchor[None]))
            if PEER_CANDIDATES:cost=physical(queries)
            else:
                peers=np.arange(n)!=i;indices=queries[:,i]
                cost=unary[i,indices]+pair[i,:,:,0].sum(0)[indices]
                cost+=settings.get('nearest',0.)*np.minimum(auxiliary['attacker_distance'][i,indices],np.linalg.norm(future[peers]-target,axis=-1).min())
                radius=np.minimum(auxiliary['radius'][i,indices],np.linalg.norm(future[peers],axis=-1).min())
                cost+=settings.get('guard',0.)*np.maximum(0.,radius-auxiliary['attacker_radius']+5.)**2
            x=torch.tensor(trajectory_value_features(queries,auxiliary,gates,settings,i))
            prediction=torch.stack([model(x) for model in trajectory_value_models(settings['value_path'])])
            if settings.get('outcome_value',False):
                survival=1.-prediction[...,0]
                capture=prediction[...,1]
                safety=survival[:,:-1]-survival[:,-1:]
                advantage=capture[:,:-1]-capture[:,-1:]
                permissible=(safety.mean(0)-settings.get('uncertainty',.5)*safety.std(0)>=-settings.get('safety_tolerance',0.)).detach().numpy()
            else:
                advantage=prediction[:,:-1]-prediction[:,-1:]
                permissible=np.ones(k,dtype=bool)
            estimate=(advantage.mean(0)-settings.get('uncertainty',.5)*advantage.std(0)).detach().numpy()
            difference=cost[:-1]-cost[-1]-settings.get('value_scale',10.)*estimate
            difference=np.where(permissible,difference,np.inf)
            alternative=int(difference.argmin());proposals.append(alternative);gains.append(difference[alternative])
        selected=int(np.argmin(gains))
        if gains[selected]<-1e-8:anchor[selected]=proposals[selected]
        return gates[np.arange(n),anchor]
    if not PEER_CANDIDATES:
        future=auxiliary['position']+auxiliary['times'][-1]*auxiliary['velocity']
        target=auxiliary['attacker_paths'][-1]
        for i in range(n):
            peers=np.arange(n)!=i
            cost=unary[i]+pair[i,:,:,0].sum(0)
            nearest=np.minimum(auxiliary['attacker_distance'][i],np.linalg.norm(future[peers]-target,axis=-1).min())
            radius=np.minimum(auxiliary['radius'][i],np.linalg.norm(future[peers],axis=-1).min())
            cost+=settings.get('nearest',0.)*nearest
            cost+=settings.get('guard',0.)*np.maximum(0.,radius-auxiliary['attacker_radius']+5.)**2
            if settings.get('value_path'):
                options=np.zeros((k,n),dtype=int);options[:,i]=np.arange(k)
                cost-=settings.get('value_scale',10.)*trajectory_value(options,auxiliary,gates,settings,subject=i)
            choice[i]=cost.argmin()
        return gates[np.arange(n),choice]
    if settings.get('nearest',0.) or settings.get('guard',0.) or settings.get('value_path'):
        def energy(choices):
            m=len(choices);rows=np.arange(n)[None,:]
            value=unary[rows,choices].sum(-1)
            i,j=np.triu_indices(n,1)
            value+=pair[i[None,:],j[None,:],choices[:,i],choices[:,j]].sum(-1)
            value+=settings.get('nearest',0.)*auxiliary['attacker_distance'][rows,choices].min(-1)
            margin=auxiliary['radius'][rows,choices].min(-1)-auxiliary['attacker_radius']+5.
            value+=settings.get('guard',0.)*np.maximum(0.,margin)**2
            if settings.get('value_path') and not settings.get('node_value',False):
                value-=settings.get('value_scale',10.)*trajectory_value(choices,auxiliary,gates,settings)
            return value
        for _ in range(settings.get('sweeps',4)*n):
            options=np.broadcast_to(choice,(n*k,n)).copy()
            options[np.arange(n*k),np.repeat(np.arange(n),k)]=np.tile(np.arange(k),n)
            values=energy(options)
            if settings.get('value_path') and settings.get('node_value',False):
                for i in range(n):
                    queries=options[i*k:(i+1)*k]
                    x=torch.tensor(trajectory_value_features(np.concatenate((queries,choice[None])),auxiliary,gates,settings,i))
                    prediction=torch.stack([model(x) for model in trajectory_value_models(settings['value_path'])])
                    advantages=prediction[:,:-1]-prediction[:,-1:]
                    values[i*k:(i+1)*k]-=settings.get('value_scale',10.)*(advantages.mean(0)-settings.get('uncertainty',.5)*advantages.std(0)).detach().numpy()
            index=int(values.argmin())
            if values[index]>=energy(choice[None])[0]-1e-8:break
            choice=options[index]
        return gates[np.arange(n),choice]
    # Parallel conditional proposals followed by the largest improving update
    # make the readout permutation equivariant except for exact numerical ties.
    for _ in range(settings.get('sweeps',4)*n):
        conditional=unary+pair[np.arange(n)[:,None],np.arange(n)[None,:],:,choice[None,:]].sum(1)
        candidate=conditional.argmin(-1)
        gain=conditional[np.arange(n),choice]-conditional[np.arange(n),candidate]
        i=int(gain.argmax())
        if gain[i]<1e-8:
            if not settings.get('pair_moves',False) or not PEER_CANDIDATES:break
            best_gain=1e-8;move=None
            for i in range(n):
                for j in range(i+1,n):
                    cost=conditional[i,:,None]+conditional[j,None,:]
                    cost-=pair[i,j,:,choice[j]][:,None]+pair[j,i,:,choice[i]][None,:]
                    cost+=pair[i,j]
                    a,b=np.unravel_index(cost.argmin(),cost.shape)
                    improvement=cost[choice[i],choice[j]]-cost[a,b]
                    if improvement>best_gain:best_gain,move=improvement,(i,j,a,b)
            if move is None:break
            i,j,a,b=move;choice[i]=a;choice[j]=b
            continue
        choice[i]=candidate[i]
    return gates[np.arange(n),choice]


def value_branch(root,obs,policy,first,seed):
    env=copy.deepcopy(root);np.random.seed(seed);action=first.copy();done=0;steps=0;total=0.;discount=1.
    period=policy['settings'].get('decision_steps',1)
    if period>1:env._fusion_commit=(round(env.Current_T/.2)//period,first[:,2].copy())
    while not done:
        obs,reward,done,_=env.step(action,'AdaRes');steps+=1
        total+=discount*float(np.mean(reward))/100.;discount*=.99
        if not done:action,_,_=decision(env,obs,policy)
    return np.array([int(done>2)+.5*int(done==3)+.02*total,
                     int(done>2),int(done==3),int(done==2),steps])


def trajectory_intervention(task):
    scene,policy,repetitions=task
    sampled=episode((*scene,policy,True));root,obs=sampled['roots'][0]
    action,base,f=decision(root,obs,policy);rng=np.random.default_rng(scene[0]^99131)
    i=int(rng.integers(len(action)));future=rng.integers(0,2**32-1,repetitions)
    settings=policy['settings'].copy();settings.pop('value_path',None)
    gates,_,_,auxiliary=trajectory_tables(root,base,action[:,2],settings)
    selected=np.zeros(len(action),dtype=int)
    if settings.get('factorial',False):
        position=auxiliary['position'];distances=np.linalg.norm(position[:,None]-position[None,:],axis=-1)
        np.fill_diagonal(distances,np.inf)
        attack=np.linalg.norm(position-root.attacker.pos,axis=-1)
        gap=np.linalg.norm(base[:,:2]-(root.boids_actions-250.)/750.,axis=-1)
        weights=.05+gap*(np.exp(-attack/20.)+np.exp(-distances.min(-1)/8.))
        i=int(rng.choice(len(action),p=weights/weights.sum()));j=int(distances[i].argmin())
        first=selected.copy();second=selected.copy();both=selected.copy()
        first[i]=int(np.abs(gates[i]-(1. if action[i,2]<.5 else 0.)).argmin())
        second[j]=int(np.abs(gates[j]-(1. if action[j,2]<.5 else 0.)).argmin())
        both[i]=first[i];both[j]=second[j]
        choices=[selected,first,second,both];outcomes=[]
        for choice in choices:
            changed=action.copy();changed[:,2]=gates[np.arange(len(action)),choice]
            outcomes.append(np.stack([value_branch(root,obs,policy,changed,int(seed)) for seed in future]))
        records=[]
        for before,after,subject in ((0,1,i),(0,2,j),(2,3,i),(1,3,j)):
            record=dict(original=outcomes[before],other=outcomes[after],seed=scene[0],defender=subject,
                        arm_before=before,arm_after=after,root_time=float(root.Current_T))
            for without_peer,suffix in ((False,''),(True,'_no_peer')):
                record['x'+suffix]=trajectory_value_features(choices[before][None],auxiliary,gates,settings,subject,without_peer)[0]
                record['y'+suffix]=trajectory_value_features(choices[after][None],auxiliary,gates,settings,subject,without_peer)[0]
            records.append(record)
        return records
    subject=i if not PEER_CANDIDATES or settings.get('node_value',False) else None
    x=trajectory_value_features(selected[None],auxiliary,gates,settings,subject)[0]
    original=np.stack([value_branch(root,obs,policy,action,int(seed)) for seed in future])
    records=[]
    for alternative in (0.,.5,1.):
        changed=action.copy();changed[i,2]=alternative
        other=np.stack([value_branch(root,obs,policy,changed,int(seed)) for seed in future])
        choice=selected.copy();choice[i]=int(np.argmin(abs(gates[i]-alternative)))
        y=trajectory_value_features(choice[None],auxiliary,gates,settings,subject)[0]
        records.append(dict(x=x,y=y,original=original,other=other,seed=scene[0],defender=i))
    return records


def trajectory_observation_episode(task):
    scene,policy=task
    seed,n,agility=scene;np.random.seed(seed);rng=np.random.default_rng(seed^19201)
    env=numerical_source().TADEnv(n);obs,_=env.reset(agility=agility,noisy_agility=False)
    done=0;step=0;features=[];masked_features=[];offsets=[];rewards=[]
    period=policy['settings'].get('decision_steps',1)
    while not done:
        action,base,_=decision(env,obs,policy)
        if step%max(period,5)==0:
            settings=policy['settings'];gates,_,_,auxiliary=trajectory_tables(env,base,action[:,2],settings)
            subject=int(rng.integers(n)) if settings.get('node_value',False) or not PEER_CANDIDATES else None
            features.append(trajectory_value_features(np.zeros((1,n),dtype=int),auxiliary,gates,settings,subject)[0])
            if settings.get('factorial',False):
                masked_features.append(trajectory_value_features(np.zeros((1,n),dtype=int),auxiliary,gates,settings,subject,True)[0])
            offsets.append(step)
        obs,reward,done,_=env.step(action,'AdaRes');rewards.append(float(np.mean(reward))/100.);step+=1
    discounted=0.;returns=np.zeros(len(rewards))
    for t in range(len(rewards)-1,-1,-1):
        discounted=rewards[t]+.99*discounted;returns[t]=discounted
    targets=int(done>2)+.5*int(done==3)+.02*returns[offsets]
    result=dict(x=np.array(features,dtype=np.float32),value=targets.astype(np.float32),seed=seed,outcome=int(done))
    if masked_features:result['x_no_peer']=np.array(masked_features,dtype=np.float32)
    return result


def collect_trajectory_observations(args,pool):
    initial=json.loads(args.start.read_text())
    initial['settings'].update(angular_features=True,centered_uncertainty=True,node_value=args.node_value)
    if args.factorial:
        initial['settings'].update(factorial=True,gate_conditioning=True,interaction_sampling=True,node_value=True)
    policy=controller_spec(initial);scenes=bank(args.bank,args.per_cell);records=[]
    for index,record in enumerate(pool.map(trajectory_observation_episode,[(s,policy) for s in scenes],chunksize=1)):
        records.append(record)
        if (index+1)%32==0:print(dict(episodes=index+1,total=len(scenes)),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(records=records,policy=initial,without_peer=args.without_peer,bank=args.bank,
        source_sha256=BASELINE_SHA256),args.output/'observations.pth')


def reencode_intervention(task):
    scene,policy,records=task
    sampled=episode((*scene,policy,True));root,obs=sampled['roots'][0]
    if abs(root.Current_T-records[0]['root_time'])>1e-7:
        raise ValueError('Replayed state does not match the labelled intervention.')
    action,base,_=decision(root,obs,policy);settings=policy['settings']
    gates,_,_,auxiliary=trajectory_tables(root,base,action[:,2],settings)
    i,j=records[0]['defender'],records[1]['defender'];selected=np.zeros(len(action),dtype=int)
    first=selected.copy();second=selected.copy();both=selected.copy()
    first[i]=int(np.abs(gates[i]-(1. if action[i,2]<.5 else 0.)).argmin())
    second[j]=int(np.abs(gates[j]-(1. if action[j,2]<.5 else 0.)).argmin())
    both[i]=first[i];both[j]=second[j];choices=[selected,first,second,both]
    result=[]
    for record in records:
        encoded=record.copy();subject=record['defender']
        for key,arm in (('x_joint',record['arm_before']),('y_joint',record['arm_after'])):
            encoded[key]=trajectory_value_features(choices[arm][None],auxiliary,gates,settings,subject,False)[0]
        result.append(encoded)
    return result


def reencode_trajectory_data(args,pool):
    if args.without_peer:raise ValueError('Create both feature views from the common full-information behaviour.')
    data=torch.load(args.data,weights_only=False);groups={}
    for record in data['records']:groups.setdefault(record['seed'],[]).append(record)
    policy=controller_spec(data['policy']);policy['settings']['joint_response_features']=True
    tasks=[((seed,*CELLS[(seed-data['bank'])//10000]),policy,records) for seed,records in groups.items()]
    records=[]
    for index,rows in enumerate(pool.map(reencode_intervention,tasks,chunksize=1)):
        records.extend(rows)
        if (index+1)%64==0:print(dict(encoded_roots=index+1,total=len(tasks)),flush=True)
    data['records']=records;data['feature_views']=['independent','joint','no_peer']
    temporary=args.data.with_suffix('.tmp');torch.save(data,temporary);temporary.replace(args.data)
    if args.observations:
        data=torch.load(args.observations,weights_only=False);policy=controller_spec(data['policy'])
        policy['settings']['joint_response_features']=True
        tasks=[((r['seed'],*CELLS[(r['seed']-data['bank'])//10000]),policy) for r in data['records']]
        for index,encoded in enumerate(pool.map(trajectory_observation_episode,tasks,chunksize=1)):
            original=data['records'][index]
            if encoded['outcome']!=original['outcome'] or encoded['x'].shape!=original['x'].shape:
                raise ValueError('Replayed observation trajectory changed its outcome or sampling times.')
            original['x_joint']=encoded['x']
            if (index+1)%64==0:print(dict(encoded_episodes=index+1,total=len(tasks)),flush=True)
        data['feature_views']=['independent','joint','no_peer']
        temporary=args.observations.with_suffix('.tmp');torch.save(data,temporary);temporary.replace(args.observations)


def trajectory_fit(args,pool):
    initial=json.loads(args.start.read_text());policy=controller_spec(initial)
    if args.angular_features or args.node_value:
        initial['settings'].update(angular_features=True,centered_uncertainty=True)
        policy['settings'].update(angular_features=True,centered_uncertainty=True)
    if args.node_value:
        initial['settings']['node_value']=True;policy['settings']['node_value']=True
    if args.factorial:
        initial['settings'].update(factorial=True,gate_conditioning=True,interaction_sampling=True,node_value=True)
        policy=controller_spec(initial)
    path=args.data or args.output/'labels.pth';args.output.mkdir(parents=True,exist_ok=True)
    if path.exists():
        saved=torch.load(path,weights_only=False)
        if saved['without_peer']!=args.without_peer and not (args.shared_data and all('x_no_peer' in r for r in saved['records'])):
            raise ValueError('Information contract mismatch.')
        initial=copy.deepcopy(saved['policy']);initial['without_peer']=args.without_peer;policy=controller_spec(initial)
    else:
        scenes=bank(args.bank,args.per_cell)
        records=[]
        for index,rows in enumerate(pool.map(trajectory_intervention,[(s,policy,args.repetitions) for s in scenes],chunksize=1)):
            records.extend(rows)
            if (index+1)%16==0:print(dict(roots=index+1,total=len(scenes)),flush=True)
        saved=dict(records=records,policy=initial,without_peer=args.without_peer,bank=args.bank,
            repetitions=args.repetitions,source_sha256=BASELINE_SHA256)
        torch.save(saved,path)
    if args.collect_only:
        print(dict(labels=str(path),roots=len({r['seed'] for r in saved['records']}),pairs=len(saved['records'])),flush=True)
        return
    if args.joint_response_features:
        initial['settings']['joint_response_features']=True;policy['settings']['joint_response_features']=True
    records=saved['records']
    suffix=('_no_peer' if args.without_peer and args.shared_data else '_joint' if args.joint_response_features else '')
    x=torch.tensor(np.stack([r['x'+suffix] for r in records]));y=torch.tensor(np.stack([r['y'+suffix] for r in records]))
    if args.pair_readout:
        initial['settings'].update(pair_readout=True);policy['settings'].update(pair_readout=True)
    if args.anchored_readout:
        initial['settings']['anchored_readout']=True;policy['settings']['anchored_readout']=True
    if args.outcome_value:
        initial['settings'].update(anchored_readout=True,outcome_value=True)
        policy['settings'].update(anchored_readout=True,outcome_value=True)
    experience=None
    if args.observations:
        observation_data=torch.load(args.observations,weights_only=False)
        if observation_data['without_peer']!=args.without_peer and not (args.shared_data and all('x_no_peer' in r for r in observation_data['records'])):
            raise ValueError('Experience information contract mismatch.')
        experience=(torch.tensor(np.concatenate([r['x'+suffix] for r in observation_data['records']])),
                    torch.tensor(np.concatenate([r['value'] for r in observation_data['records']])))
        if args.outcome_value:
            experience=(experience[0],torch.tensor(np.concatenate([np.broadcast_to(
                [int(r['outcome']<=2),int(r['outcome']==3),int(r['outcome']==4)],(len(r['x']),3))
                for r in observation_data['records']]),dtype=torch.float32))
        if experience[0].shape[1]!=x.shape[1]:raise ValueError('Experience encoder dimension mismatch.')
    a=torch.tensor(np.stack([r['original'][:,0] for r in records]),dtype=torch.float32)
    b=torch.tensor(np.stack([r['other'][:,0] for r in records]),dtype=torch.float32)
    target,alternate=a.mean(-1),b.mean(-1)
    # Coupled future returns preserve the common random component before differencing.
    delta=(a-b).mean(-1)
    if args.outcome_value:
        def probabilities(key):
            outcomes=np.stack([r[key][:,1:3] for r in records]).mean(1)
            return torch.tensor(np.stack((1.-outcomes[:,0],outcomes[:,1],outcomes[:,0]-outcomes[:,1]),-1),dtype=torch.float32)
        target,alternate=probabilities('original'),probabilities('other')
        delta=target-alternate
    states=[];validation=[]
    torch.set_grad_enabled(True)
    for member in range(0 if args.critic else 3):
        torch.manual_seed(args.seed+member)
        model=(TrajectoryOutcomeValue(x.shape[-1],linear=args.value_architecture=='linear',mirror_canonical=args.mirror_canonical)
            if args.outcome_value else TrajectoryValue(x.shape[-1]))
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
        train_indices=torch.arange(len(x));experience_indices=None;held_out=None
        if args.bootstrap:
            rng=np.random.default_rng(args.seed+member)
            scene_seeds=np.array([r['seed'] for r in records]);unique=np.unique(scene_seeds)
            train_indices=torch.tensor(np.concatenate([np.flatnonzero(scene_seeds==seed)
                for seed in rng.choice(unique,len(unique),replace=True)]))
            if args.out_of_bag_select:
                held_out=torch.tensor(np.flatnonzero(~np.isin(scene_seeds,scene_seeds[train_indices.numpy()])))
            if experience is not None:
                ends=np.cumsum([0]+[len(r['x']) for r in observation_data['records']])
                experience_indices=torch.tensor(np.concatenate([np.arange(ends[index],ends[index+1])
                    for index in rng.integers(0,len(ends)-1,len(ends)-1)]))
        best_validation=float('inf');best_state=None;member_validation=[]
        for update in range(args.updates):
            indices=train_indices[torch.randint(len(train_indices),(128,))];qx,qy=model(x[indices]),model(y[indices])
            if args.outcome_value:
                loss=-.05*(target[indices]*qx.clamp_min(1e-6).log()+alternate[indices]*qy.clamp_min(1e-6).log()).sum(-1).mean()
            else:loss=.1*((qx-target[indices]).square()+(qy-alternate[indices]).square()).mean()
            if args.loss_mode=='conditional':loss+=args.difference_weight*((qx-qy)-delta[indices]).square().mean()
            if experience is not None:
                sampled=torch.randint(len(experience[0]),(128,))
                if experience_indices is not None:
                    sampled=experience_indices[torch.randint(len(experience_indices),(128,))]
                if args.outcome_value:
                    loss-=.05*(experience[1][sampled]*model(experience[0][sampled]).clamp_min(1e-6).log()).sum(-1).mean()
                else:loss+=.25*(model(experience[0][sampled])-experience[1][sampled]).square().mean()
            opt.zero_grad();loss.backward();opt.step()
            if held_out is not None and len(held_out) and ((update+1)%250==0 or update+1==args.updates):
                with torch.no_grad():
                    vx,vy=model(x[held_out]),model(y[held_out])
                    difference_error=((vx-vy)-delta[held_out]).square().mean()
                    if args.outcome_value:
                        absolute_error=-.05*(target[held_out]*vx.clamp_min(1e-6).log()+alternate[held_out]*vy.clamp_min(1e-6).log()).sum(-1).mean()
                    else:absolute_error=.1*((vx-target[held_out]).square()+(vy-alternate[held_out]).square()).mean()
                    error=float(absolute_error+(args.difference_weight*difference_error if args.loss_mode=='conditional' else 0.))
                    member_validation.append(dict(update=update+1,loss=error,difference_mse=float(difference_error),
                        zero_difference_mse=float(delta[held_out].square().mean())))
                    if error<best_validation:best_validation=error;best_state=copy.deepcopy(model.state_dict())
        states.append(best_state if best_state is not None else model.state_dict());validation.append(member_validation)
        print(dict(member=member,loss=float(loss.detach()),out_of_bag_best=best_validation if best_state is not None else None),flush=True)
    torch.set_grad_enabled(False)
    critic_path=args.critic or args.output/'value.pth'
    if not args.critic:
        torch.save(dict(models=states,dimensions=x.shape[-1],loss_mode=args.loss_mode,
            outcome_value=args.outcome_value,value_architecture=args.value_architecture,
            bootstrap=args.bootstrap,out_of_bag_select=args.out_of_bag_select,validation=validation,
            mirror_canonical=args.mirror_canonical,
            source_sha256=BASELINE_SHA256),critic_path)
    scenes=bank(args.selection_bank,args.selection_per_cell)
    original=evaluate(pool,scenes,policy);best_rows=original;best_settings=policy['settings'].copy();history=[]
    def objective(rows):
        value=summarize(rows)
        # After satisfying the required development success floor, improve capture.
        safety=min(cell['success'] for cell in value['cells'].values()) if args.cell_floor else value['success']
        feasible=value['success']>=.9 and safety>=(args.cell_floor or .9)
        return (int(feasible),value['capture'] if feasible else safety,value['success'],-value['collision'])
    configurations=([(scale,tolerance) for tolerance in (0.,.02) for scale in (1.,10.,100.)]
        if args.risk_budget_search else [(scale,0.) for scale in (1.,3.,10.,30.,100.,300.)])
    for scale,tolerance in configurations:
        spec=copy.deepcopy(policy);spec['settings'].update(value_path=str(critic_path.resolve()),value_scale=scale,
                                                        safety_tolerance=tolerance)
        rows=evaluate(pool,scenes,spec);accepted=objective(rows)>objective(best_rows)
        if accepted:best_rows,best_settings=rows,spec['settings'].copy()
        summary=summarize(rows);history.append(dict(scale=scale,safety_tolerance=tolerance,summary=summary,accepted=accepted))
        print(dict(scale=scale,safety_tolerance=tolerance,success=summary['success'],capture=summary['capture'],best=objective(best_rows)),flush=True)
    result=initial.copy();result.update(controller='trajectory',settings=best_settings,selected=summarize(best_rows),
        initial_selected=summarize(original),selected_rows=best_rows,loss_mode=args.loss_mode,training_seed=args.seed,
        data=str(path),value_critic=str(critic_path),selection_bank=args.selection_bank,candidates=history,
        observations=str(args.observations) if args.observations else None,
        value_architecture=args.value_architecture,bootstrap=args.bootstrap,without_peer=args.without_peer,
        shared_data=args.shared_data,out_of_bag_select=args.out_of_bag_select,cell_floor=args.cell_floor,
        risk_budget_search=args.risk_budget_search,
        mirror_canonical=args.mirror_canonical,
        status='paired single-vessel conditional value improvement; development selection')
    save_json(args.output/'policy.json',result)


def trajectory_search(args,pool):
    initial=json.loads(args.start.read_text())
    spec=controller_spec(initial)
    if not isinstance(spec,dict):spec=dict(weights=spec,refiner=np.zeros(6))
    spec['controller']='trajectory'
    scenes=bank(args.bank,args.per_cell)
    rng=np.random.default_rng(args.seed)
    best_rows=None;history=[]
    default=dict(horizon=2.,clearance=6.,risk=20.,progress=0.,align=0.,prior=10.,tail=0.,lead=0.,options=9,sweeps=4)
    if spec.get('proposal') is not None:
        default.update(gate_prior=1.,progress=0.,nearest=1.,guard=10.,prior=.2,risk=50.,clearance=7.,pair_moves=True)
    best_settings=initial.get('settings',default).copy()
    if args.decision_steps is not None:best_settings['decision_steps']=args.decision_steps
    if args.feedback_response:
        best_settings.update(response_model='feedback',proposal_model=spec['proposal'])
    for iteration in range(args.iterations):
        settings=best_settings.copy()
        if iteration:
            if iteration<8:
                settings['horizon']=(1.,1.5,2.,2.5,3.5,4.,5.)[iteration-1]
            else:
                keys=rng.choice(('horizon','clearance','risk','progress','prior','lead','pair_moves'),size=2,replace=False)
                for key in keys:
                    settings[key]=float(rng.choice(dict(horizon=(1.,1.5,2.,2.5,3.,4.,5.),
                        clearance=(5.,5.5,6.,6.5,7.,8.),risk=(2.,5.,10.,20.,50.,100.),
                        progress=(0.,.1,.3,1.),prior=(1.,3.,10.,30.,100.),lead=(0.,1.,2.,3.),pair_moves=(0.,1.))[key]))
        spec['settings']=settings
        rows=evaluate(pool,scenes,spec)
        accepted=best_rows is None or score(rows)>score(best_rows)
        if accepted:best_rows,best_settings=rows,settings.copy()
        summary=summarize(rows)
        history.append(dict(iteration=iteration,settings=settings,summary=summary,accepted=accepted))
        print(dict(iteration=iteration,success=summary['success'],collision=summary['collision'],
                   best=summarize(best_rows)['success'],settings=settings),flush=True)
        save_json(args.output/'policy.json',dict(controller='trajectory',weights=spec['weights'].tolist(),
            refiner_weights=spec['refiner'].tolist(),settings=best_settings,proposal=spec.get('proposal'),without_peer=args.without_peer,
            source_commit=SOURCE_COMMIT,source_sha256=BASELINE_SHA256,bank=args.bank,training_seed=args.seed,
            selected=summarize(best_rows),selected_rows=best_rows,trials=history,
            status='development initialization of trajectory-conditioned value readout'))


def outcome(env, done):
    return dict(success=int(done > 2), capture=int(done == 3), collision=int(done == 2),
        source_failure=int(done == 1), timeout=int(done == 4),
        physical_breach=int(np.linalg.norm(env.attacker.pos) < env.Target_R),
        duration=float(env.Current_T), outcome=int(done))


def episode(task):
    seed, n, agility, weights, keep = task
    np.random.seed(int(seed))
    env = numerical_source().TADEnv(n)
    obs, _ = env.reset(agility=agility, noisy_agility=False)
    done, reward_sum, discount, changes = 0, 0., 1., 0
    roots = []
    root_rng = np.random.default_rng(seed ^ 917)
    root_mass = 0.
    step = 0
    while not done:
        if weights is None:
            action = BASE(obs)
        else:
            action, base, features = decision(env, obs, weights)
            changes += int(np.any(abs(action[:, 2]-base[:, 2]) > 1e-5))
        period=weights.get('settings',{}).get('decision_steps',1) if isinstance(weights,dict) else 1
        if keep and step%period==0:
            priority = .2 + 6.*np.exp(-(env.def_def_dists.min()-8.)**2/8.)
            if isinstance(weights,dict) and weights.get('settings',{}).get('interaction_sampling',False):
                attack=np.linalg.norm(np.array([b.pos for b in env.defender_list])-env.attacker.pos,axis=-1).min()
                priority=.05+3.*np.exp(-((attack-12.)/12.)**2)+.5*np.exp(-(env.def_def_dists.min()-7.)**2/8.)
            root_mass += priority
            if root_rng.random() < priority/root_mass:
                frozen = copy.deepcopy(env)
                for name in ('Pos_Att', 'Phi_Att', 'Pos_Def', 'Phi_Def', 'Rewards'):
                    value = getattr(frozen, name)
                    if np.ndim(value) > 1:
                        setattr(frozen, name, value[-1:].copy())
                roots = [(frozen, obs.copy())]
        obs, reward, done, _ = env.step(action, 'AdaRes')
        reward_sum += discount * float(np.mean(reward)); discount *= .99
        step += 1
    return dict(seed=int(seed), defenders=n, agility=agility, **outcome(env, done),
                discounted_return=reward_sum, changed_steps=changes, roots=roots)


def bank(start, per_cell, cells=None):
    return [(start+c*10000+i, n, a) for c,(n,a) in enumerate(CELLS if cells is None else cells) for i in range(per_cell)]


def stationary_episode(scene):
    """Zero physical thrust diagnoses timeout success; it is not ARBoids."""
    seed,n,agility=scene;np.random.seed(seed)
    env=numerical_source().TADEnv(n);env.reset(agility=agility,noisy_agility=False)
    action=np.full((n,3),-1./3.,dtype=np.float64);action[:,2]=1.
    done=0;total=0.;discount=1.
    while not done:
        _,reward,done,_=env.step(action,'AdaRes')
        total+=discount*float(np.mean(reward));discount*=.99
    return dict(seed=int(seed),defenders=n,agility=agility,**outcome(env,done),discounted_return=total)


def summarize(rows):
    keys = ('success', 'capture', 'collision', 'source_failure', 'timeout',
            'physical_breach', 'duration', 'discounted_return')
    result = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    result['episodes'] = len(rows)
    result['cells'] = {f'n{n}-a{a:g}': {key: float(np.mean([r[key] for r in rows
                          if r['defenders']==n and r['agility']==a])) for key in keys}
                       for n,a in sorted({(r['defenders'],r['agility']) for r in rows})}
    return result


def score(rows):
    s = summarize(rows)
    return (s['success'], -s['collision'], s['discounted_return'])


def evaluate(pool, scenes, weights, keep=False):
    return list(pool.map(episode, [(*s, weights, keep) for s in scenes], chunksize=1))


def paired_summary(left, right):
    result = {}
    rng = np.random.default_rng(20261011)
    for key in ('success', 'capture', 'collision', 'source_failure'):
        d = np.asarray([a[key]-b[key] for a,b in zip(left,right)])
        boot = np.concatenate([rng.choice(d, (min(1000, 10000-i), len(d)), replace=True).mean(-1)
                               for i in range(0, 10000, 1000)])
        result[key] = dict(difference=float(d.mean()), interval95=np.quantile(boot,[.025,.975]).tolist(),
                           wins=int((d>0).sum()), losses=int((d<0).sum()))
    return result


def search(args, pool):
    scenes = bank(args.bank, args.per_cell)
    original = evaluate(pool, scenes, None)
    best = np.zeros(len(FEATURES), dtype=np.float32)
    rows = original
    if args.start:
        best = policy_weights(json.loads(args.start.read_text()))
        rows = evaluate(pool, scenes, best)
    initial = best.copy()
    rng = np.random.default_rng(args.seed)
    history = []
    print('original', summarize(original), flush=True)
    for k in range(args.iterations):
        if k < 12 and not args.start:
            candidate = np.zeros_like(best)
            if k < 6:
                candidate[0] = (-2., -1., -.5, .5, 1., 2.)[k]
            else:
                candidate[8] = (-12., -6., -3., 3., 6., 12.)[k-6]
        elif k < 12:
            candidate = initial.copy()
            index = (12,13,0)[k//4]
            candidate[index] += ((-3.,-6.,-9.,-12.) if k<8 else (-6.,-3.,3.,6.))[k%4]
        else:
            candidate = best.copy()
            scale = (.2, .5, 1., 2.)[k % 4]
            if k % 3:
                eligible=np.arange(len(best))
                indices = rng.choice(eligible, rng.integers(1, 4), replace=False)
                candidate[indices] += rng.normal(0., scale, len(indices))
            else:
                candidate += rng.normal(0., scale, len(best))
        candidate_rows = evaluate(pool, scenes, candidate)
        accepted = score(candidate_rows) > score(rows)
        if accepted:
            best, rows = candidate, candidate_rows
        summary = summarize(candidate_rows)
        history.append(dict(iteration=k, weights=candidate.tolist(), summary=summary, accepted=accepted))
        print(dict(iteration=k, success=summary['success'], collision=summary['collision'],
                   best_success=score(rows)[0], accepted=accepted), flush=True)
        save_json(args.output / 'development.json', dict(protocol='untouched-author-source-80s',
            source_commit=SOURCE_COMMIT, source=str(args.source),
            source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
            features=FEATURES, bank=args.bank, weights=best.tolist(),
            without_peer=args.without_peer,training_seed=args.seed,
            original=summarize(original), selected=summarize(rows), trials=history,
            original_rows=original, selected_rows=rows,
            status='development-selected initialization; conditional critic fitting follows'))
    return best


def branch_return(root, obs, weights, first, future_seed):
    env = copy.deepcopy(root)
    np.random.seed(future_seed)
    action = first.copy()
    total, discount, steps, done = 0., 1., 0, 0
    while not done:
        obs, reward, done, _ = env.step(action, 'AdaRes')
        total += discount * float(np.mean(reward)); discount *= .99
        steps += 1
        if not done:
            action, _, _ = decision(env, obs, weights)
    return total / 100., int(done>2), int(done==2), steps


def intervention(task):
    scene, weights, repetitions = task
    sampled = episode((*scene, weights, True))
    if not sampled['roots']:
        return None
    root, obs = sampled['roots'][0]
    action, base, f = decision(root, obs, weights)
    rng = np.random.default_rng(scene[0] ^ 71297)
    # Focus labels on vessels whose candidate mixtures can influence close interactions.
    influence = .05 + abs(f[:, 8]) + f[:, 2] + 2.*f[:,4:8].max(-1)
    i = int(rng.choice(len(action), p=influence/influence.sum()))
    seeds = rng.integers(0,2**32-1,repetitions)
    a = [branch_return(root,obs,weights,action,int(seed)) for seed in seeds]
    context = critic_context(root,base,f)
    if isinstance(weights,dict):
        context = np.concatenate((context,action[:,2,None]),axis=-1)
    records = []
    for j,reference in enumerate((0.,.5,1.)):
        alt = action.copy()
        alt[i,2] = reference
        b = [branch_return(root,obs,weights,alt,int(seed)) for seed in seeds]
        records.append(dict(features=f,context=context,gates=action[:,2].copy(),
            alternate=alt[:,2].copy(),baseline=base[:,2].copy(),
            value=np.mean(a,axis=0),other=np.mean(b,axis=0),seed=scene[0],defender=i,
            branch_steps=sum(x[3] for x in b)+(sum(x[3] for x in a) if j==0 else 0)))
    return records


def factorial_intervention(task):
    """Add a crossed teammate anchor; each contrast still changes one gate."""
    scene, weights, repetitions, original = task
    sampled = episode((*scene, weights, True))
    root, obs = sampled['roots'][0]
    action, base, features = decision(root, obs, weights)
    context = critic_context(root, base, features)
    if isinstance(weights,dict):
        context = np.concatenate((context,action[:,2,None]),axis=-1)
    reference = original[0]
    if (not np.allclose(action[:, 2], reference['gates'], atol=1e-6)
            or not np.allclose(context, reference['context'], atol=1e-6)):
        raise ValueError('A reused return must belong to the identical root and continuation.')
    rng = np.random.default_rng(scene[0] ^ 71297)
    influence = .05 + abs(features[:, 8]) + features[:, 2] + 2.*features[:, 4:8].max(-1)
    i = int(rng.choice(len(action), p=influence/influence.sum()))
    if i != reference['defender']:
        raise ValueError('The replayed intervention must select the original vessel.')
    seeds = rng.integers(0, 2**32-1, repetitions)
    positions = np.array([boat.pos for boat in root.defender_list])
    distance = np.linalg.norm(positions-positions[i], axis=-1)
    distance[i] = np.inf
    j = int(distance.argmin())
    endpoint_i, endpoint_j = float(action[i, 2] < .5), float(action[j, 2] < .5)
    existing_i = next(row for row in original if row['alternate'][i] == endpoint_i)
    one = action.copy(); one[j, 2] = endpoint_j
    both = one.copy(); both[i, 2] = endpoint_i
    returns_j = [branch_return(root, obs, weights, one, int(seed)) for seed in seeds]
    returns_ij = [branch_return(root, obs, weights, both, int(seed)) for seed in seeds]
    value_j, value_ij = np.mean(returns_j, axis=0), np.mean(returns_ij, axis=0)
    records = []
    contrasts = (
        (action[:, 2], one[:, 2], reference['value'], value_j, j,
         sum(row[3] for row in returns_j), 'teammate-at-policy'),
        (existing_i['alternate'], both[:, 2], existing_i['other'], value_ij, j,
         sum(row[3] for row in returns_ij), 'teammate-at-intervention'),
        (one[:, 2], both[:, 2], value_j, value_ij, i, 0, 'original-at-teammate-intervention'))
    for gates, alternate, value, other, defender, steps, anchor in contrasts:
        records.append(dict(features=features, context=context, gates=gates.copy(),
            alternate=alternate.copy(), baseline=base[:, 2].copy(), value=value, other=other,
            seed=scene[0], defender=defender, branch_steps=steps, anchor=anchor))
    return records


def augment(args, pool):
    saved = torch.load(args.data, weights_only=False)
    if saved.get('without_peer', False) != args.without_peer:
        raise ValueError('Crossed anchors must use the same candidate information.')
    if saved.get('crossed_anchors'):
        raise ValueError('This label bank already contains crossed anchors.')
    grouped = {}
    for row in saved['records']:
        grouped.setdefault(row['seed'], []).append(row)
    weights = controller_spec(saved)
    tasks = []
    for seed, rows in grouped.items():
        n, agility = CELLS[(seed-saved['bank'])//10000]
        if n != len(rows[0]['gates']):
            raise ValueError('The reconstruction bank must match the original scene cells.')
        tasks.append(((seed, n, agility), weights, saved['repetitions'], rows))
    records = list(saved['records'])
    for index, added in enumerate(pool.map(factorial_intervention, tasks, chunksize=1)):
        records.extend(added)
        if (index+1) % 32 == 0:
            print('crossed roots', index+1, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(saved, records=records, crossed_anchors=True, original_data=str(args.data),
                    roots=len(grouped)), args.output/'interventions.pt')
    print('crossed pairs', len(records), 'branch steps',
          sum(row['branch_steps'] for row in records), flush=True)


class ConditionalValue(nn.Module):
    """Joint configuration value with explicit conditional gate interactions."""
    def __init__(self, d):
        super().__init__()
        self.nodes = nn.Sequential(nn.Linear(d,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3))
        self.pairs = nn.Sequential(nn.Linear(2*d,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self, features, gates):
        coefficients = self.nodes(features)
        value = (coefficients[...,0]+coefficients[...,1]*gates+coefficients[...,2]*gates.square()).mean(-1)
        n = features.shape[-2]
        fi, fj = features.unsqueeze(-2).expand(-1,n,n,-1), features.unsqueeze(-3).expand(-1,n,n,-1)
        pair = self.pairs(torch.cat((fi+fj, abs(fi-fj)), -1)).squeeze(-1)
        mask = 1.-torch.eye(n, device=features.device)
        value += (pair*gates.unsqueeze(-1)*gates.unsqueeze(-2)*mask).sum((-1,-2))/(n*(n-1))
        return value


class AdditiveValue(ConditionalValue):
    """Matched value ablation: joint gate value has no cross-vessel coupling."""
    def forward(self, features, gates):
        coefficients = self.nodes(features)
        return (coefficients[...,0]+coefficients[...,1]*gates
                +coefficients[...,2]*gates.square()).mean(-1)


class CenteredConditionalValue(nn.Module):
    """Separate local gate effects from complementarity around the current policy."""
    def __init__(self, d, weights):
        super().__init__()
        self.register_buffer('prior_weights', torch.tensor(weights, dtype=torch.float32))
        self.nodes = nn.Sequential(nn.Linear(d+1,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3))
        self.pairs = nn.Sequential(nn.Linear(2*(d+1),32),nn.SiLU(),nn.Linear(32,1))

    def forward(self, features, gates):
        source = features[...,23].clamp(1e-6,1.-1e-6)
        shift = (features[...,:len(FEATURES)]*self.prior_weights).sum(-1).clamp(-12.,12.)
        prior = torch.sigmoid(torch.logit(source)+shift)
        if features.shape[-1] == 33:
            prior = features[...,32]
        context = torch.cat((features,prior.unsqueeze(-1)),-1)
        displacement = gates-prior
        coefficients = self.nodes(context)
        value = (coefficients[...,0]+coefficients[...,1]*displacement
                 +coefficients[...,2]*displacement.square()).mean(-1)
        n = features.shape[-2]
        fi = context.unsqueeze(-2).expand(-1,n,n,-1)
        fj = context.unsqueeze(-3).expand(-1,n,n,-1)
        pair = self.pairs(torch.cat((fi+fj,abs(fi-fj)),-1)).squeeze(-1)
        mask = 1.-torch.eye(n,device=features.device)
        value += (pair*displacement.unsqueeze(-1)*displacement.unsqueeze(-2)*mask).sum((-1,-2))/(n*(n-1))
        return value


def make_critic(d, weights, architecture):
    if architecture == 'centered':
        return CenteredConditionalValue(d, weights)
    if architecture == 'additive':
        return AdditiveValue(d)
    return ConditionalValue(d)


def fit(args, pool):
    start=json.loads(args.start.read_text())
    weights=policy_weights(start)
    continuation=controller_spec(start)
    continuation_refiner=start.get('refiner_weights')
    data_path=args.data or args.output/'interventions.pt'
    if data_path.exists():
        data=torch.load(data_path,weights_only=False)
        if isinstance(data,dict):
            generating_weights=np.asarray(data['weights'],dtype=np.float32)
            if data.get('without_peer',False)!=args.without_peer:
                raise ValueError('The labels and policy must have the same candidate inputs.')
            if args.data:
                weights=generating_weights
                continuation_refiner=data.get('refiner_weights')
            elif not np.array_equal(weights,generating_weights):
                raise ValueError('Conditional returns must use the policy that generated them.')
            elif not np.array_equal(np.asarray(continuation_refiner),np.asarray(data.get('refiner_weights'))):
                raise ValueError('Conditional returns must use the same frozen readout.')
            data=data['records']
    else:
        tasks=[(s,continuation,args.repetitions) for s in bank(args.bank,args.per_cell)]
        data=[]
        for root_index,records in enumerate(pool.map(intervention,tasks,chunksize=1)):
            if records is not None:data.extend(records)
            if (root_index+1)%16==0:print('paired roots',root_index+1,flush=True)
        args.output.mkdir(parents=True,exist_ok=True)
        torch.save(dict(records=data,weights=weights.tolist(),bank=args.bank,
                        repetitions=args.repetitions,source_sha256=BASELINE_SHA256,
                        without_peer=args.without_peer,refiner_weights=continuation_refiner),data_path)
    args.output.mkdir(parents=True,exist_ok=True)
    initial_weights=weights.copy()
    torch.set_grad_enabled(True); torch.manual_seed(args.seed)
    tensors={n: {k:torch.tensor(np.stack([r[k] for r in data if len(r['gates'])==n]),dtype=torch.float32)
             for k in ('features','context','gates','alternate','baseline','value','other','defender')} for n in (3,6)}
    models=[]
    for member in range(3):
        torch.manual_seed(args.seed+member)
        model=make_critic(tensors[3]['context'].shape[-1],weights,args.critic_architecture)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
        for epoch in range(args.updates):
            total=0.
            for n,batch in tensors.items():
                index=torch.randint(len(batch['gates']),(min(64,len(batch['gates'])),))
                f,g,a=(batch[k][index] for k in ('context','gates','alternate'))
                # Terminal success is the task objective. The published return
                # supplies a small shaping term, without changing evaluation.
                y=batch['value'][index,1]+.05*batch['value'][index,0]
                z=batch['other'][index,1]+.05*batch['other'][index,0]
                q,qa=model(f,g),model(f,a)
                loss=.1*((q-y).square().mean()+(qa-z).square().mean())
                if args.loss_mode=='conditional':loss=loss+args.difference_weight*((q-qa)-(y-z)).square().mean()
                optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss.detach())
            if epoch==args.updates-1:print('critic',member,'loss',total,flush=True)
        model.eval().requires_grad_(False); models.append(model)
    torch.save(dict(models=[m.state_dict() for m in models],weights=weights.tolist(),features=FEATURES,
                    loss_mode=args.loss_mode,difference_weight=args.difference_weight,
                    architecture=args.critic_architecture,
                    refiner_weights=continuation_refiner),args.output/'critic.pth')
    if args.critic_only:return
    if continuation_refiner is not None:
        args.data=data_path
        args.critic=args.output/'critic.pth'
        return refine(args,pool)
    parameter=nn.Parameter(torch.tensor(weights))
    parameter_mask=torch.ones(len(weights))
    optimizer=torch.optim.Adam([parameter],lr=.03)
    initial=parameter.detach().clone()
    candidates=[]
    for iteration in range(80):
        total=0.
        for batch in tensors.values():
            f,base=batch['features'],batch['baseline'].clamp(1e-6,1.-1e-6)
            gates=torch.sigmoid(torch.logit(base)+(f*parameter*parameter_mask).sum(-1))
            reference=batch['gates']
            if args.actor_update=='conditional':
                intervention_mask=nn.functional.one_hot(batch['defender'].long(),gates.shape[-1])
                conditional_gates=reference+intervention_mask*(gates-reference)
                values=torch.stack([m(batch['context'],conditional_gates)-m(batch['context'],reference)
                                    for m in models])
                objective=values.mean(0)-.5*values.std(0)
                total=total-gates.shape[-1]*objective.mean()+.5*((conditional_gates-reference).square().sum(-1)).mean()
            else:
                reference=torch.sigmoid(torch.logit(base)+(f*initial).sum(-1)).detach()
                values=torch.stack([m(batch['context'],gates) for m in models])
                objective=values.mean(0)-.5*values.std(0)
                total=total-objective.mean()+.5*(gates-reference).square().mean()
        optimizer.zero_grad();total.backward();optimizer.step()
        if iteration in (4,9,19,39,79):candidates.append((parameter.detach()*parameter_mask).numpy().copy())
    torch.set_grad_enabled(False)
    scenes=bank(args.selection_bank,args.selection_per_cell)
    best_rows=evaluate(pool,scenes,weights)
    history=[]
    for i,candidate in enumerate(candidates):
        rows=evaluate(pool,scenes,candidate)
        accepted=score(rows)>score(best_rows)
        history.append(dict(candidate=i,weights=candidate.tolist(),summary=summarize(rows),accepted=accepted))
        print('critic-policy',i,summarize(rows)['success'],'accepted',accepted,flush=True)
        if accepted:weights,best_rows=candidate,rows
    save_json(args.output/'policy.json',dict(weights=weights.tolist(),features=FEATURES,
        source=str(args.source),source_commit=SOURCE_COMMIT,source_sha256=BASELINE_SHA256,
        initialization=str(args.start),initial_weights=initial_weights.tolist(),
        objective='terminal_success + 0.05 * published_return/100',loss_mode=args.loss_mode,
        difference_weight=args.difference_weight,
        actor_update=args.actor_update,
        without_peer=args.without_peer,
        critic=str(args.output/'critic.pth'),roots=len(set(r['seed'] for r in data)),
        pairs=len(data),branch_steps=sum(r.get('branch_steps',0) for r in data),
        repetitions=args.repetitions,training_seed=args.seed,
        selection_bank=args.selection_bank,selected=summarize(best_rows),candidates=history,
        policy_update_accepted=any(x['accepted'] for x in history),
        conditional_policy_update_accepted=args.loss_mode=='conditional' and any(x['accepted'] for x in history)))


def refine(args,pool):
    """Learn a composition-conditioned readout while freezing the first pass."""
    saved=torch.load(args.data,weights_only=False)
    if saved.get('without_peer',False)!=args.without_peer:
        raise ValueError('The refinement labels must match the candidate information.')
    weights=np.asarray(saved['weights'],dtype=np.float32)
    bundle=torch.load(args.critic,weights_only=True)
    if not np.array_equal(weights,np.asarray(bundle['weights'],dtype=np.float32)):
        raise ValueError('The critic and labels must use the same frozen continuation.')
    initial_readout=saved.get('refiner_weights')
    if not np.array_equal(np.asarray(initial_readout),np.asarray(bundle.get('refiner_weights'))):
        raise ValueError('The critic and labels must use the same frozen readout.')
    initial_readout=np.asarray(initial_readout if initial_readout is not None
                               else np.zeros(len(REFINER_FEATURES)),dtype=np.float32)
    unique={}
    for row in saved['records']:
        unique.setdefault(row['seed'],row)
    records=list(unique.values())
    tensors={}
    for n in (3,6):
        rows=[row for row in records if len(row['gates'])==n]
        extra=[];priors=[]
        for row in rows:
            c=row['context']
            source=np.clip(c[:,23],1e-6,1.-1e-6)
            prior=1./(1.+np.exp(-(np.log(source/(1.-source))
                                  +np.clip(c[:,:len(FEATURES)]@weights,-12.,12.))))
            priors.append(prior)
            extra.append(composition_features(c[:,14:16]*60.,c[:,16:18]*5.,
                np.arctan2(c[:,20],c[:,19]),c[:,21:23]*750.+250.,c[:,24:26]*750.+250.,
                prior,without_peer=args.without_peer))
        tensors[n]=dict(context=torch.tensor(np.stack([row['context'] for row in rows])),
            reference=torch.tensor(np.stack([row['gates'] for row in rows])),
            prior=torch.tensor(np.stack(priors)),
            extra=torch.tensor(np.stack(extra)))
    models=[]
    for state in bundle['models']:
        model=make_critic(tensors[3]['context'].shape[-1],weights,bundle.get('architecture','polynomial'))
        model.load_state_dict(state);model.eval().requires_grad_(False);models.append(model)
    torch.set_grad_enabled(True)
    parameter=nn.Parameter(torch.tensor(initial_readout))
    optimizer=torch.optim.Adam([parameter],lr=.05)
    candidates=[]
    for iteration in range(120):
        total=0.
        for batch in tensors.values():
            reference=batch['reference']
            shift=(batch['extra']*parameter).sum(-1).clamp(-6.,6.)
            gates=torch.sigmoid(torch.logit(batch['prior'].clamp(1e-6,1.-1e-6))+shift)
            effects=torch.stack([model(batch['context'],gates)-model(batch['context'],reference)
                                 for model in models])
            uncertainty=effects.var(0,unbiased=False).clamp_min(1e-8).sqrt()
            total=total-(effects.mean(0)-.5*uncertainty).mean()+.25*(gates-reference).square().mean()
        optimizer.zero_grad();total.backward();optimizer.step()
        if iteration in (4,9,19,39,79,119):candidates.append(parameter.detach().numpy().copy())
    torch.set_grad_enabled(False)
    scenes=bank(args.selection_bank,args.selection_per_cell)
    selected=initial_readout.copy()
    best=evaluate(pool,scenes,dict(weights=weights,refiner=selected))
    history=[]
    print('composition initial',summarize(best)['success'],flush=True)
    for i,candidate in enumerate(candidates):
        rows=evaluate(pool,scenes,dict(weights=weights,refiner=candidate))
        accepted=score(rows)>score(best)
        history.append(dict(candidate=i,weights=candidate.tolist(),summary=summarize(rows),accepted=accepted))
        print('composition candidate',i,summarize(rows)['success'],'accepted',accepted,flush=True)
        if accepted:selected,best=candidate,rows
    save_json(args.output/'policy.json',dict(weights=weights.tolist(),refiner_weights=selected.tolist(),
        initial_refiner_weights=initial_readout.tolist(),
        features=FEATURES,refiner_features=REFINER_FEATURES,without_peer=args.without_peer,
        source=str(args.source),source_commit=SOURCE_COMMIT,source_sha256=BASELINE_SHA256,
        critic=str(args.critic),data=str(args.data),loss_mode=bundle['loss_mode'],
        critic_architecture=bundle.get('architecture','polynomial'),
        difference_weight=bundle.get('difference_weight',1. if args.loss_mode=='conditional' else 0.),
        policy_update_accepted=any(row['accepted'] for row in history),
        roots=len(records),pairs=len(saved['records']),selection_bank=args.selection_bank,
        selected=summarize(best),candidates=history))


def mechanism_packet(env, obs, base):
    """Public measurements and the two unchanged author commands, in body frames."""
    motion=np.asarray([[b.x,b.y,b.theta,*b.velocity] for b in env.defender_list])
    p,v,heading=motion[:,:2],motion[:,3:5],motion[:,2]
    c,s=np.cos(heading),np.sin(heading)
    def local(vector):
        return np.stack((c*vector[:,0]+s*vector[:,1],-s*vector[:,0]+c*vector[:,1]),-1)
    n=len(p);candidates=np.concatenate((base[:,:2],(env.boids_actions-250.)/750.),-1)
    node=np.concatenate((local(-p)/30.,local(env.attacker.pos-p)/30.,
        local(np.broadcast_to(env.attacker.vel,(n,2)))/5.,local(v)/5.,
        motion[:,5:6],candidates,base[:,2:3],
        np.full((n,1),n/6.),np.full((n,1),env.Current_T/80.)),axis=-1)
    dp=p[None]-p[:,None];dv=v[None]-v[:,None]
    rot=lambda x:np.stack((c[:,None]*x[...,0]+s[:,None]*x[...,1],
                           -s[:,None]*x[...,0]+c[:,None]*x[...,1]),-1)
    distance=np.linalg.norm(dp,axis=-1)
    relative_heading=heading[None]-heading[:,None]
    geometry=np.concatenate((rot(dp)/15.,rot(dv)/5.,np.sin(relative_heading)[...,None],
        np.cos(relative_heading)[...,None],np.broadcast_to(motion[None,:,5:6],(n,n,1)),
        distance[...,None]/15.,((dp*dv).sum(-1)/np.maximum(distance,1.))[...,None]/5.),-1)
    thrust=candidates.reshape(n,2,2)*750.+250.
    u=local(v)[:,0]
    acceleration=(thrust.sum(-1)-(100.+150.*abs(u))[:,None]*u[:,None])/380.
    yaw=1.25*(thrust[...,0]-thrust[...,1])/1430.
    forward=np.stack((c,s),-1);side=np.stack((-s,c),-1)
    response=p[:,None]+1.2*v[:,None]+.72*acceleration[...,None]*forward[:,None]+.3*yaw[...,None]*side[:,None]
    def edges(peer):
        other=response if peer else np.broadcast_to((p+1.2*v)[:,None],response.shape)
        effects=[]
        for a,b in ((0,0),(0,1),(1,0),(1,1)):
            delta=other[None,:,b]-response[:,None,a]
            effects.extend((rot(delta)/15.,np.linalg.norm(delta,axis=-1)[...,None]/15.))
        return np.concatenate((geometry,np.broadcast_to(candidates[:,None],(n,n,4)),
            np.broadcast_to((candidates if peer else np.zeros_like(candidates))[None],(n,n,4)),*effects),-1)
    return {k:np.clip(value,-6.,6.).astype(np.float32) for k,value in
            dict(node=node,edge=edges(True),edge_no_peer=edges(False),prior=base[:,2]).items()}


class MechanismAdapter(nn.Module):
    """Learned candidate-interaction messages; the only output is a scalar gate."""
    def __init__(self,peer=True,width=64,source_prior=False,coupled=False):
        super().__init__();self.peer=peer;self.source_prior=source_prior;self.coupled=coupled
        self.edge=nn.Sequential(nn.Linear(29,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.gate=nn.Sequential(nn.Linear(16+2*width,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        if coupled:
            self.coupling=nn.Sequential(nn.Linear(35,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2))
            nn.init.zeros_(self.coupling[-1].weight);nn.init.zeros_(self.coupling[-1].bias)

    def forward(self,packet):
        edge=self.edge(packet['edge' if self.peer else 'edge_no_peer'])
        n=edge.shape[1];mask=~torch.eye(n,dtype=torch.bool,device=edge.device)[None,:,:,None]
        mean=(edge*mask).sum(2)/(n-1)
        maximum=edge.masked_fill(~mask,-1e5).amax(2)
        shift=self.gate(torch.cat((packet['node'],mean,maximum),-1)).squeeze(-1)
        # A shared interior initialization permits gate learning even when the
        # frozen source Adapter has saturated; source commands remain frozen.
        if self.source_prior:shift=shift+torch.logit(packet['prior'].clamp(1e-6,1.-1e-6))
        gates=torch.sigmoid(shift)
        if self.coupled:
            context=packet['edge' if self.peer else 'edge_no_peer']
            responses=context[...,17:].reshape(*context.shape[:3],2,2,3)
            for _ in range(3):
                own=gates[:,:,None,None].expand(-1,-1,n,-1)
                peers=gates if self.peer else torch.full_like(gates,.5)
                peer=peers[:,None,:,None].expand(-1,n,-1,-1)
                weights=torch.cat((own,1.-own),-1)[...,:,None]*torch.cat((peer,1.-peer),-1)[...,None,:]
                mixed=(responses[...,:2]*weights[...,None]).sum((3,4))
                messages=self.coupling(torch.cat((context,own,peer,own*peer,mixed,mixed.norm(dim=-1,keepdim=True)),-1))
                attention=torch.softmax(messages[...,1].masked_fill(~mask.squeeze(-1),-1e5),-1)
                update=(messages[...,0]*attention).sum(2)
                gates=torch.sigmoid(shift+2.*torch.tanh(update))
        return gates


class MechanismCritic(nn.Module):
    """A joint value over all gates; pair terms identify conditional effects."""
    def __init__(self,width=64,channels=3):
        super().__init__();self.channels=channels
        self.edge=nn.Sequential(nn.Linear(29,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.node=nn.Sequential(nn.Linear(16+width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.state=nn.Sequential(nn.Linear(2*width,128),nn.SiLU(),nn.Linear(128,channels))
        self.unary=nn.Linear(width,2*channels);self.pair=nn.Linear(width,4*channels)
        self.control_node=nn.Sequential(nn.Linear(20,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.control_pair=nn.Sequential(nn.Linear(35,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.control_value=nn.Sequential(nn.Linear(4*width,128),nn.SiLU(),nn.Linear(128,channels))
        nn.init.normal_(self.control_value[-1].weight,std=.005);nn.init.zeros_(self.control_value[-1].bias)
        nn.init.normal_(self.unary.weight,std=.005);nn.init.zeros_(self.unary.bias)
        nn.init.normal_(self.pair.weight,std=.005);nn.init.zeros_(self.pair.bias)

    def forward(self,packet,gates):
        edge=self.edge(packet['edge']);n=edge.shape[1]
        mask=~torch.eye(n,dtype=torch.bool,device=edge.device)[None,:,:,None]
        mean=(edge*mask).sum(2)/(n-1)
        node=self.node(torch.cat((packet['node'],mean),-1))
        state=self.state(torch.cat((node.mean(1),mean.mean(1)),-1))
        z=2.*gates-1.;basis=torch.stack((z,z.square()),-1)
        unary=(self.unary(node).reshape(*node.shape[:2],2,self.channels)*basis[...,None]).sum(2).mean(1)
        product=(basis[:,:,None,:,None]*basis[:,None,:,None,:]).flatten(-2)
        pair=(self.pair(edge).reshape(*edge.shape[:3],4,self.channels)*product[...,None]).sum(3)
        pair=(pair*mask).sum((1,2))/(n*(n-1))
        own=gates[:,:,None,None].expand(-1,-1,n,-1)
        peer=gates[:,None,:,None].expand(-1,n,-1,-1)
        responses=packet['edge'][...,17:].reshape(*edge.shape[:3],2,2,3)
        weights=torch.cat((own,1.-own),-1)[..., :,None]*torch.cat((peer,1.-peer),-1)[...,None,:]
        mixed=(responses[...,:2]*weights[...,None]).sum((3,4))
        controlled=self.control_pair(torch.cat((packet['edge'],own,peer,own*peer,mixed,
            mixed.norm(dim=-1,keepdim=True)),-1))
        nominal=gates[...,None]*packet['node'][...,9:11]+(1.-gates[...,None])*packet['node'][...,11:13]
        controlled_nodes=self.control_node(torch.cat((packet['node'],gates[...,None],gates[...,None].square(),nominal),-1))
        joint=self.control_value(torch.cat((controlled_nodes.mean(1),controlled_nodes.amax(1),
            (controlled*mask).sum((1,2))/(n*(n-1)),controlled.masked_fill(~mask,-1e5).amax((1,2))),-1))
        value=state+unary+pair+joint
        return torch.cat((torch.softmax(value[:,:3],-1),value[:,3:]),-1)


def mechanism_tensor(packet,device='cpu'):
    return {k:torch.as_tensor(value,device=device,dtype=torch.float32).unsqueeze(0) for k,value in packet.items()}


@lru_cache(maxsize=4)
def mechanism_load(path,revision):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    source_prior=saved.get('source_prior',not any('control_value' in k for k in saved['critics']))
    actor=MechanismAdapter(saved['peer'],source_prior=source_prior,coupled=saved.get('coupled',False))
    actor.load_state_dict(saved['actor']);actor.eval()
    actor.warm=saved.get('warm')
    return actor


def mechanism_action(env,obs,actor):
    global PEER_CANDIDATES
    base=BASE(obs);packet=mechanism_packet(env,obs,base)
    if getattr(actor,'warm',None):
        # A shared, previously learned local gate initializes every mechanism
        # cell. It contains no replacement thrust head or peer command input.
        previous=PEER_CANDIDATES;PEER_CANDIDATES=False
        try:
            reference=gate_action(base,relations(env,obs,base),np.asarray(actor.warm['weights']))
            extra=live_composition_features(env,base,reference[:,2])
            theta=np.clip(reference[:,2],1e-6,1.-1e-6)
            z=np.log(theta/(1.-theta))+np.clip(extra@np.asarray(actor.warm['refiner_weights']),-6.,6.)
            packet['prior']=(1./(1.+np.exp(-z))).astype(np.float32)
        finally:PEER_CANDIDATES=previous
    with torch.no_grad():gates=actor(mechanism_tensor(packet))[0].numpy()
    action=base.copy();action[:,2]=gates
    return action,packet


def mechanism_rollout(env,obs,actor,first_action,future_seed,horizon):
    np.random.seed(future_seed)
    branch=copy.deepcopy(env);obs=obs.copy();done=0;step=0;total=0.;discount=1.
    while not done and step<horizon:
        action=first_action if step==0 else mechanism_action(branch,obs,actor)[0]
        obs,reward,done,_=branch.step(action,'AdaRes');step+=1
        total+=discount*float(np.mean(reward))/100.;discount*=.99
    if done:
        packet=None
    else:
        # The bootstrap must use exactly the same gate prior as continuation.
        # A raw source packet would silently switch a warm policy back to the
        # original Adapter when evaluating its next action.
        packet=mechanism_action(branch,obs,actor)[1]
    return dict(done=done,packet=packet,steps=step,reward=total,discount=discount)


def mechanism_episode(task):
    scene,path,revision,collect,exploration,repeats,horizon=task[:7]
    root_count=task[7] if len(task)>7 else 1
    seed,n,agility=scene;np.random.seed(seed)
    actor=mechanism_load(path,revision)
    env=numerical_source().TADEnv(n);obs,_=env.reset(agility=agility,noisy_agility=False)
    rng=np.random.default_rng(seed^180311);done=0;transitions=[];root=None;roots=[];mass=0.;source_return=0.;discount=1.
    while not done:
        action,packet=mechanism_action(env,obs,actor)
        if collect:
            # Candidate-independent exploration is identical in all four cells.
            action[:,2]=np.clip(action[:,2]+rng.normal(0.,exploration,n),0.,1.)
            separation=env.def_def_dists.min()
            priority=.05+4.*np.exp(-((separation-7.)/3.)**2)
            if root_count==1:
                mass+=priority;keep_root=rng.random()<priority/mass;priority_key=0.
            else:
                # Broader coverage prevents a sparse last-moment intervention
                # set from serving as the support of an entire feedback policy.
                priority=1.+np.exp(-((separation-7.)/3.)**2)
                priority_key=float(np.log(max(rng.random(),1e-12))/priority)
                keep_root=len(roots)<root_count or priority_key>min(item[0] for item in roots)
            if keep_root:
                frozen=copy.deepcopy(env)
                for name in ('Pos_Att','Phi_Att','Pos_Def','Phi_Def','Rewards'):
                    value=getattr(frozen,name)
                    if np.ndim(value)>1:setattr(frozen,name,value[-1:].copy())
                root=(frozen,obs.copy(),packet,action.copy())
                if root_count>1:
                    roots.append((priority_key,root))
                    if len(roots)>root_count:roots.pop(min(range(len(roots)),key=lambda k:roots[k][0]))
        obs,reward,done,_=env.step(action,'AdaRes')
        source_return+=discount*float(np.mean(reward));discount*=.99
        if collect:transitions.append(dict(packet=packet,gates=action[:,2].copy(),done=done,reward=float(np.mean(reward))/100.))
    row=dict(seed=int(seed),defenders=n,agility=agility,**outcome(env,done),discounted_return=source_return)
    if not collect:return row
    event=np.eye(3,dtype=np.float32)[0 if done<=2 else done-2]
    total=0.
    for transition in reversed(transitions):
        total=transition['reward']+.99*total;transition['return']=total
    # On-policy n-step TD and terminal supervision share the same state rows.
    for t,transition in enumerate(transitions):
        end=min(t+16,len(transitions))
        transition.update(next=transitions[end]['packet'] if end<len(transitions) else None,
            event=event,terminal=end==len(transitions),discount=.99**(end-t),
            reward_prefix=sum(.99**(j-t)*transitions[j]['reward'] for j in range(t,end)))
    branch_rows=[]
    selected_roots=[root] if root_count==1 and root is not None else [item[1] for item in roots]
    for root in selected_roots:
        frozen,root_obs,packet,action=root
        root_step=int(round(frozen.Current_T/.2))
        positions=np.asarray([boat.pos for boat in frozen.defender_list])
        distances=np.linalg.norm(positions[:,None]-positions[None],axis=-1)+1e6*np.eye(n)
        pair=np.unravel_index(distances.argmin(),distances.shape)
        subjects=tuple(int(i) for i in pair)
        arms=[action.copy() for _ in range(4)]
        for k,subject in enumerate(subjects):
            value=float(rng.integers(2)) if .2<action[subject,2]<.8 else float(action[subject,2]<.5)
            arms[k+1][subject,2]=value;arms[3][subject,2]=value
        for _ in range(repeats):
            future=int(rng.integers(1,2**31-1))
            outcomes=[mechanism_rollout(frozen,root_obs,actor,arm,future,horizon) for arm in arms]
            for a,b,subject in ((0,1,subjects[0]),(0,2,subjects[1]),(2,3,subjects[0]),(1,3,subjects[1])):
                branch_rows.append(dict(packet=packet,left=arms[a][:,2].copy(),right=arms[b][:,2].copy(),
                    subject=subject,before=outcomes[a],after=outcomes[b],seed=int(seed),root_step=root_step,future=future,arms=(a,b)))
    return dict(row=row,transitions=transitions,branches=branch_rows)


def mechanism_stack(packets,device):
    return {k:torch.tensor(np.stack([p[k] for p in packets]),device=device) for k in packets[0]}


@torch.no_grad()
def mechanism_improvement_targets(critics,packet,reference,alternatives,utility):
    q0=torch.stack([(critic(packet,reference)*utility).sum(-1) for critic in critics])
    advantages=torch.stack([torch.stack([(critic(packet,gates)*utility).sum(-1)
        for critic in critics])-q0 for gates in alternatives])
    estimate=advantages.mean(1)-.5*advantages.std(1,unbiased=False)
    best,choice=estimate.max(0);strength=torch.tanh(best.clamp_min(0.)/.02)
    selected=torch.stack(alternatives)[choice,torch.arange(len(reference),device=reference.device)]
    return (reference+strength[:,None]*(selected-reference)).clamp(0.,1.),best


def mechanism_train(args,pool):
    """Train the actual Adapter--joint TD Critic loop on the released plant."""
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(True);torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor=MechanismAdapter(not args.without_peer,source_prior=bool(args.warm),coupled=args.coupled_adapter).to(device)
    actor.warm=None
    if args.warm:
        warm=json.loads(args.warm.read_text())
        if not warm['without_peer'] or warm.get('proposal') is not None:raise ValueError('The shared initializer must be a local gate with unchanged author commands.')
        actor.warm={k:warm[k] for k in ('weights','refiner_weights')}
    channels=4 if args.reward_critic else 3
    critics=nn.ModuleList([MechanismCritic(channels=channels).to(device) for _ in range(2)])
    targets=copy.deepcopy(critics).eval().requires_grad_(False)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=3e-4)
    critic_opt=torch.optim.Adam(critics.parameters(),lr=1e-3)
    start=0;history=[]
    if args.start:
        old=torch.load(args.start,map_location=device,weights_only=False)
        actor.source_prior=old.get('source_prior',False);actor.warm=old.get('warm')
        added_coupling=args.coupled_adapter and not old.get('coupled',False)
        missing=actor.load_state_dict(old['actor'],strict=not added_coupling)
        if added_coupling and (missing.unexpected_keys or any(not k.startswith('coupling.') for k in missing.missing_keys)):
            raise ValueError('Only the new zero-initialized coupling messages may be absent.')
        critics.load_state_dict(old['critics']);targets.load_state_dict(old['targets'])
        if not added_coupling:actor_opt.load_state_dict(old['actor_optimizer'])
        critic_opt.load_state_dict(old['critic_optimizer'])
        if not added_coupling:start=old['iteration'];history=old['history']
    current=args.output/'current.pth';best_path=args.output/'policy.pth';replay=[];branches=[]
    best_score=(-1.,-1.)
    if history:
        best_score=max((r['development']['success'],r['development']['capture']) for r in history)
    def save(path,iteration):
        torch.save(dict(actor={k:v.detach().cpu() for k,v in actor.state_dict().items()},
            critics=critics.state_dict(),targets=targets.state_dict(),actor_optimizer=actor_opt.state_dict(),
            critic_optimizer=critic_opt.state_dict(),peer=not args.without_peer,
            conditional=args.loss_mode=='conditional',source_prior=actor.source_prior,warm=actor.warm,
            value_channels=channels,conditional_actor=args.conditional_actor,coupled=args.coupled_adapter,
            paired_futures_averaged=True,
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            parent=str(args.start) if args.start else None,iteration=iteration,history=history,
            source=str(args.source),source_commit=SOURCE_COMMIT,source_sha256=BASELINE_SHA256,
            mechanism='learned candidate-message Adapter / joint event TD plus paired intervention calibration',
            control_period=.2,proposal='frozen author SAC + unchanged author Boids',seed=args.seed),path)
    save(current,start)
    if history:save(best_path,start)
    if not history:
        initial_rows=list(pool.map(mechanism_episode,[(scene,str(current),current.stat().st_mtime_ns,False,0.,0,0)
            for scene in bank(args.selection_bank,args.selection_per_cell)],chunksize=1))
        initial=summarize(initial_rows);best_score=(initial['success'],initial['capture'])
        history.append(dict(iteration=start,development=initial,rows=initial_rows,accepted=True,initialization=True))
        save(best_path,start);print(dict(phase='initial',success=best_score[0],capture=best_score[1]),flush=True)
    for iteration in range(start,start+args.iterations):
        revision=current.stat().st_mtime_ns
        scenes=bank(args.bank+iteration*100000,args.per_cell)
        rollouts=[]
        for completed,result in enumerate(pool.map(mechanism_episode,[(scene,str(current),revision,True,
            0.,args.repetitions,args.branch_horizon,args.roots_per_episode) for scene in scenes],chunksize=1),1):
            rollouts.append(result)
            if completed%32==0:print(dict(phase='collect',iteration=iteration,episodes=completed,total=len(scenes)),flush=True)
        # Every target refers to this frozen continuation. Old-policy terminal
        # labels must not silently become targets for the current policy.
        replay=[r for episode in rollouts for r in episode['transitions']]
        branches=[r for episode in rollouts for r in episode['branches']]
        torch.save(dict(transitions=replay,branches=branches,iteration=iteration,
                        continuation={k:v.detach().cpu() for k,v in actor.state_dict().items()},
                        source_prior=actor.source_prior,warm=actor.warm,peer=actor.peer),args.output/'experience.pth')
        terminal_pairs=[r for r in branches if r['before']['done'] and r['after']['done']]
        unique_branches={}
        for row in branches:
            for arm,key in zip(row['arms'],('before','after')):
                unique_branches[(row['seed'],row.get('root_step',0),row['future'],arm)]=row[key]
        causal=dict(roots=len({(r['seed'],r.get('root_step',0)) for r in branches}),pairs=len(branches),both_terminal=len(terminal_pairs),
            terminal_outcome_changed=sum(r['before']['done']!=r['after']['done'] for r in terminal_pairs),
            unique_branches=len(unique_branches),simulation_steps=len(replay)+sum(r['steps'] for r in unique_branches.values()),
            mean_absolute_source_return_difference=float(np.mean([abs(r['before']['reward']-r['after']['reward']) for r in branches])))
        print(dict(phase='causal_data',**causal),flush=True)
        teacher=copy.deepcopy(actor).eval().requires_grad_(False)
        groups={n:[r for r in replay if len(r['gates'])==n] for n in args.teams}
        contrast={};actor_roots={}
        for n in args.teams:
            grouped={}
            for row in branches:
                if len(row['left'])==n:grouped.setdefault((row['seed'],row.get('root_step',0),tuple(row['arms'])),[]).append(row)
            contrast[n]=list(grouped.values())
            roots={}
            for row in branches:
                if len(row['left'])==n and row['arms'][0]==0:
                    roots.setdefault((row['seed'],row.get('root_step',0)),{})[row['arms'][1]]=row
            actor_roots[n]=list(roots.values())
        generator=np.random.default_rng(args.seed+iteration)
        diagnostics=[]
        utility=torch.tensor([0.,1.5,1.]+([.05] if channels==4 else []),device=device)
        def value_target(entries):
            target=torch.zeros(len(entries),channels,device=device)
            live=[i for i,e in enumerate(entries) if not e['done']]
            for i,e in enumerate(entries):
                if e['done']:target[i,0 if e['done']<=2 else e['done']-2]=1.
                if channels==4:target[i,3]=e['reward']
            if live:
                packets=mechanism_stack([entries[i]['packet'] for i in live],device)
                gates=teacher(packets)
                continuation=torch.stack([net(packets,gates) for net in targets]).mean(0)
                target[live,:3]=continuation[:,:3]
                if channels==4:
                    discount=torch.tensor([entries[i]['discount'] for i in live],device=device)
                    target[live,3]+=discount*continuation[:,3]
            return target
        for update in range(args.updates):
            n=int(generator.choice(args.teams));rows=groups[n]
            batch=[rows[i] for i in generator.integers(len(rows),size=256)]
            packet=mechanism_stack([r['packet'] for r in batch],device)
            gates=torch.tensor(np.stack([r['gates'] for r in batch]),device=device)
            events=torch.tensor(np.stack([r['event'] for r in batch]),device=device)
            if channels==4:events=torch.cat((events,torch.tensor([[r['return']] for r in batch],device=device)),1)
            with torch.no_grad():
                temporal=value_target([dict(done=(1 if r['event'][0] else 3 if r['event'][1] else 4) if r['terminal'] else 0,
                    packet=r['next'],reward=r['reward_prefix'],discount=r['discount']) for r in batch])
                reference=teacher(packet)
            crows=contrast[n];cgroups=[crows[i] for i in generator.integers(len(crows),size=64)]
            cbatch=[group[0] for group in cgroups]
            cp=mechanism_stack([r['packet'] for r in cbatch],device)
            left=torch.tensor(np.stack([r['left'] for r in cbatch]),device=device)
            right=torch.tensor(np.stack([r['right'] for r in cbatch]),device=device)
            with torch.no_grad():
                before_values=value_target([r['before'] for group in cgroups for r in group])
                after_values=value_target([r['after'] for group in cgroups for r in group])
                counts=[len(group) for group in cgroups]
                before=torch.stack([part.mean(0) for part in before_values.split(counts)])
                after=torch.stack([part.mean(0) for part in after_values.split(counts)])
            def ce(prediction,target):
                loss=-(target[:,:3]*prediction[:,:3].clamp_min(1e-7).log()).sum(-1).mean()
                if channels==4:loss=loss+(prediction[:,3]-target[:,3]).square().mean()
                return loss
            critic_loss=0.;difference=0.
            for net in critics:
                predicted=net(packet,gates);qa,qb=net(cp,left),net(cp,right)
                delta=(qa-qb-(before-after)).square().mean()
                critic_loss=critic_loss+ce(predicted,temporal)+.5*ce(predicted,events)+.25*(ce(qa,before)+ce(qb,after))
                difference=difference+delta
            if args.loss_mode=='conditional':critic_loss=critic_loss+args.difference_weight*difference
            critic_opt.zero_grad(set_to_none=True);critic_loss.backward()
            nn.utils.clip_grad_norm_(critics.parameters(),5.);critic_opt.step()
            if update>=args.updates//4 and update%2==0:
                critics.requires_grad_(False);predicted_gates=actor(packet)
                values=torch.stack([(net(packet,predicted_gates)*utility).sum(-1) for net in critics])
                actor_loss=-values.min(0).values.mean()+args.trust_weight*(predicted_gates-reference).square().mean()
                if args.conditional_actor and args.loss_mode=='conditional':
                    available=actor_roots[n]
                    rows=[available[i] for i in generator.integers(len(available),size=64)]
                    ap=mechanism_stack([row[1]['packet'] for row in rows],device)
                    anchor=torch.tensor(np.stack([row[1]['left'] for row in rows]),device=device)
                    alternatives=[torch.tensor(np.stack([row[k]['right'] for row in rows]),device=device) for k in (1,2)]
                    with torch.no_grad():
                        desired,_=mechanism_improvement_targets(critics,ap,anchor,alternatives,utility)
                    actor_loss=args.trust_weight*nn.functional.binary_cross_entropy(predicted_gates.clamp(1e-7,1.-1e-7),reference)
                    actor_loss=actor_loss+2.*nn.functional.binary_cross_entropy(actor(ap).clamp(1e-7,1.-1e-7),desired)
                actor_opt.zero_grad(set_to_none=True);actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),2.);actor_opt.step();critics.requires_grad_(True)
            with torch.no_grad():
                for net,target in zip(critics,targets):
                    for parameter,following in zip(net.parameters(),target.parameters()):following.lerp_(parameter,.005)
            if (update+1)%250==0:
                info=dict(phase='learn',iteration=iteration,update=update+1,td_and_mc=float(critic_loss.detach()),
                    contrast_error=float(difference.detach()),gate_change=float((actor(packet)-reference).abs().mean().detach()))
                diagnostics.append(info);print(info,flush=True)
        trained=copy.deepcopy(actor.state_dict());reference_state=teacher.state_dict()
        candidates=[];chosen=None;chosen_score=best_score
        for scale in ((.25,1.) if args.conditional_actor else (1.,)):
            actor.load_state_dict({k:reference_state[k]+scale*(trained[k]-reference_state[k]) for k in trained})
            save(current,iteration+1);revision=current.stat().st_mtime_ns
            development=list(pool.map(mechanism_episode,[(scene,str(current),revision,False,0.,0,0)
                for scene in bank(args.selection_bank,args.selection_per_cell)],chunksize=1))
            summary=summarize(development);current_score=(summary['success'],summary['capture'])
            candidates.append(dict(scale=scale,summary=summary,rows=development))
            print(dict(phase='policy_step',iteration=iteration+1,scale=scale,success=current_score[0],capture=current_score[1]),flush=True)
            if current_score>chosen_score:
                chosen_score=current_score;chosen=copy.deepcopy(actor.state_dict());selected=candidates[-1]
        accepted=chosen is not None
        if accepted:
            actor.load_state_dict(chosen);summary=selected['summary'];development=selected['rows'];best_score=chosen_score
        else:
            actor.load_state_dict(reference_state)
            prior=max(history,key=lambda r:(r['development']['success'],r['development']['capture']))
            summary=prior['development'];development=prior['rows']
        actor_opt=torch.optim.Adam(actor.parameters(),lr=3e-4)
        history.append(dict(iteration=iteration+1,development=summary,rows=development,candidates=candidates,
            collection=summarize([r['row'] for r in rollouts]),transitions=len(replay),interventions=len(branches),
            diagnostics=diagnostics,causal_data=causal,accepted=accepted))
        save(current,iteration+1)
        if accepted:save(best_path,iteration+1)
        save_json(args.output/'development.json',dict(history=history,peer=not args.without_peer,
            conditional=args.loss_mode=='conditional',best_success=best_score[0],best_capture=best_score[1],
            source_sha256=BASELINE_SHA256,development_bank=args.selection_bank))
        print(dict(phase='development',iteration=iteration+1,success=summary['success'],capture=summary['capture'],
            collision=summary['collision'],accepted=accepted,best=best_score),flush=True)


def mechanism_refit(args,pool):
    """Policy improvement restricted to evaluated single-vessel alternatives."""
    args.output.mkdir(parents=True,exist_ok=True);torch.set_grad_enabled(True);torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved=torch.load(args.start,map_location=device,weights_only=False)
    data=torch.load(args.data,map_location='cpu',weights_only=False)
    coupled=any(k.startswith('coupling.') for k in data['continuation'])
    actor=MechanismAdapter(data['peer'],source_prior=data['source_prior'],coupled=coupled).to(device)
    actor.load_state_dict(data['continuation']);actor.warm=data['warm']
    channels=saved.get('value_channels',3)
    critics=nn.ModuleList([MechanismCritic(channels=channels).to(device) for _ in range(2)])
    critics.load_state_dict(saved['critics']);critics.eval().requires_grad_(False)
    utility=torch.tensor([0.,1.5,1.]+([.05] if channels==4 else []),device=device)
    roots={}
    for row in data['branches']:
        if row['arms'][0]==0:roots.setdefault((row['seed'],row.get('root_step',0)),{})[row['arms'][1]]=row
    targets={};counts={}
    with torch.no_grad():
        for n in args.teams:
            rows=[root for root in roots.values() if len(root[1]['left'])==n]
            packet=mechanism_stack([root[1]['packet'] for root in rows],device)
            reference=torch.tensor(np.stack([root[1]['left'] for root in rows]),device=device)
            alternatives=[torch.tensor(np.stack([root[k]['right'] for root in rows]),device=device) for k in (1,2)]
            desired,best=mechanism_improvement_targets(critics,packet,reference,alternatives,utility)
            targets[n]=(packet,desired)
            counts[n]=dict(roots=len(rows),positive_targets=int((best>0).sum()),
                mean_gate_change=float((desired-reference).abs().mean()))
    print(dict(phase='supported_policy_targets',cells=counts),flush=True)
    observations={n:[r for r in data['transitions'] if len(r['gates'])==n] for n in args.teams}
    rng=np.random.default_rng(args.seed);opt=torch.optim.Adam(actor.parameters(),lr=3e-4)
    current=args.output/'current.pth';policy=args.output/'policy.pth';history=[];best_score=(-1.,-1.)
    milestones=sorted(set((0,max(1,args.updates//10),max(1,args.updates//3),args.updates)))
    def save(path,updates):
        record=dict(saved,actor={k:v.detach().cpu() for k,v in actor.state_dict().items()},
            peer=actor.peer,source_prior=actor.source_prior,warm=actor.warm,coupled=coupled,
            actor_optimizer=opt.state_dict(),history=history,
            parent=str(args.start),experience=str(args.data),conditional_improvement=True,refit_updates=updates,
            target_rule='best positive lower-ensemble single-vessel advantage; other gates retained',
            source_sha256=BASELINE_SHA256)
        torch.save(record,path)
    for update in range(args.updates+1):
        if update in milestones:
            save(current,update);revision=current.stat().st_mtime_ns
            rows=list(pool.map(mechanism_episode,[(scene,str(current),revision,False,0.,0,0)
                for scene in bank(args.selection_bank,args.selection_per_cell)],chunksize=1))
            summary=summarize(rows);quality=(summary['success'],summary['capture']);accepted=quality>best_score
            history.append(dict(updates=update,development=summary,rows=rows,accepted=accepted))
            if accepted:best_score=quality;save(policy,update)
            save_json(args.output/'development.json',dict(history=history,targets=counts,parent=str(args.start),
                data=str(args.data),best_success=best_score[0],best_capture=best_score[1],
                development_bank=args.selection_bank,source_sha256=BASELINE_SHA256,
                new_training_simulation_steps=0))
            print(dict(phase='supported_policy_development',updates=update,success=quality[0],
                capture=quality[1],accepted=accepted,best=best_score),flush=True)
        if update==args.updates:break
        n=int(rng.choice(args.teams));packet,desired=targets[n]
        indices=torch.tensor(rng.integers(len(desired),size=128),device=device)
        cp={k:v[indices] for k,v in packet.items()}
        rows=observations[n];batch=[rows[i] for i in rng.integers(len(rows),size=256)]
        ordinary=mechanism_stack([row['packet'] for row in batch],device)
        reference=torch.tensor(np.stack([row['gates'] for row in batch]),device=device)
        # All coefficients are supervised. Only one entry per conditional
        # target changes, so fitting a better i does not license altering -i.
        loss=nn.functional.binary_cross_entropy(actor(cp).clamp(1e-7,1.-1e-7),desired[indices])
        loss=loss+.25*nn.functional.binary_cross_entropy(actor(ordinary).clamp(1e-7,1.-1e-7),reference)
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),2.);opt.step()


def joint_packet(env, obs):
    packet=mechanism_packet(env,obs,np.zeros((len(obs),3),dtype=np.float32))
    packet['obs']=obs.astype(np.float32)
    packet['motion']=np.asarray([[b.x,b.y,b.theta,*b.velocity] for b in env.defender_list],dtype=np.float32)
    packet['water']=np.asarray([b.velocity_r for b in env.defender_list],dtype=np.float32)/5.
    a=env.attacker
    packet['task']=np.asarray([a.x/60.,a.y/60.,np.sin(a.theta),np.cos(a.theta),
        *(a.velocity_r/5.),a.agility/3.,env.Current_T/80.],dtype=np.float32)
    return packet


class JointLearningActor(nn.Module):
    """Continue the original neural proposal and learn its cooperative Adapter."""
    def __init__(self,peer=True,graph=False,levels=2,risk_prior=0.,risk_horizon=1.6,pair_scale=1.,gate_period=1,risk_clearance=6.,forecast_inputs=False,frozen_proposal=False,risk_feedback=False,intent_strength=0.,coupled_shortlist=0):
        super().__init__();self.peer=peer;self.graph=graph;self.levels=levels
        self.risk_prior=risk_prior;self.risk_horizon=risk_horizon;self.pair_scale=pair_scale
        self.risk_clearance=risk_clearance
        self.forecast_inputs=forecast_inputs;self.frozen_proposal=frozen_proposal
        self.risk_feedback=risk_feedback
        self.intent_strength=intent_strength
        self.coupled_shortlist=coupled_shortlist
        self.base=numerical_source().ActorAdap(6,8,3,512)
        self.gate_period=gate_period
        if gate_period>1:
            for name,parameter in self.base.named_parameters():
                parameter.requires_grad_(name.startswith(('a1.','adap_layer.')))
        if frozen_proposal:self.base.requires_grad_(False)
        self.adapter=MechanismAdapter(peer=False if graph else peer,source_prior=not bool(intent_strength),coupled=not graph)
        self.gate_logstd=nn.Parameter(torch.tensor(-.7))
        if graph:
            self.gate_pairs=nn.Sequential(nn.Linear(29+(levels*levels if forecast_inputs else 0),64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,levels*levels))
            nn.init.zeros_(self.gate_pairs[-1].weight);nn.init.zeros_(self.gate_pairs[-1].bias)

    def context(self,packet,proposals,prior):
        node=packet['node'].clone();node[...,9:11]=proposals
        candidates=torch.cat((proposals,node[...,11:13]),-1)
        motion=packet['motion'];p=motion[...,:2];v=motion[...,3:5];heading=motion[...,2]
        c,s=heading.cos(),heading.sin();n=p.shape[1]
        thrust=candidates.reshape(-1,n,2,2)*750.+250.
        u=c*v[...,0]+s*v[...,1]
        acc=(thrust.sum(-1)-(100.+150.*u.abs())[...,None]*u[...,None])/380.
        yaw=1.25*(thrust[...,0]-thrust[...,1])/1430.
        forward=torch.stack((c,s),-1);side=torch.stack((-s,c),-1)
        response=p[:,:,None]+1.2*v[:,:,None]+.72*acc[...,None]*forward[:,:,None]+.3*yaw[...,None]*side[:,:,None]
        def edge(peer):
            other=response if peer else (p+1.2*v)[:,:,None].expand_as(response)
            effects=[]
            for i,j in ((0,0),(0,1),(1,0),(1,1)):
                d=other[:,None,:,j]-response[:,:,None,i]
                rotated=torch.stack((c[:,:,None]*d[...,0]+s[:,:,None]*d[...,1],
                    -s[:,:,None]*d[...,0]+c[:,:,None]*d[...,1]),-1)/15.
                effects.extend((rotated,d.norm(dim=-1,keepdim=True)/15.))
            peers=candidates if peer else torch.zeros_like(candidates)
            return torch.cat((packet['edge'][...,:9],candidates[:,:,None].expand(-1,-1,n,-1),
                peers[:,None].expand(-1,n,-1,-1),*effects),-1).clamp(-6.,6.)
        return dict(node=node,edge=edge(True),edge_no_peer=edge(False),prior=prior)

    @torch.no_grad()
    def candidate_risk(self,packet,proposals,return_distance=False,pairwise=False):
        motion=packet['motion'];b,n=motion.shape[:2];k=self.levels
        grid=torch.linspace(0.,1.,k,device=motion.device)
        force=750.*(grid[None,None,:,None]*proposals[:,:,None]+
            (1.-grid[None,None,:,None])*packet['node'][:,:,None,11:13])+250.
        x,y,heading,vx,vy,r=[motion[...,i,None].expand(-1,-1,k).clone() for i in range(6)]
        paths=[];dt=.1
        for step in range(round(self.risk_horizon/dt)):
            if self.risk_feedback and step and step%2==0:
                state=torch.stack((x,y,heading,vx,vy,r),-1).transpose(1,2)
                forecast_obs,boids=self.feedback_observation(packet,state,step*dt)
                future=self.proposal_distribution(forecast_obs.reshape(b*k,n,-1))[0].tanh().reshape(b,k,n,2).transpose(1,2)
                force=750.*(grid[None,None,:,None]*future+
                    (1.-grid[None,None,:,None])*boids.transpose(1,2))+250.
            x=x+dt*vx;y=y+dt*vy;heading=heading+dt*r
            c,s=heading.cos(),heading.sin();u=c*vx+s*vy;v=-s*vx+c*vy
            du=(400.*r*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
            dv=(-380.*r*u-(100.+150.*v.abs())*v)/400.
            dr=(-20.*u*v-(980.+950.*r.abs())*r+1.25*(force[...,0]-force[...,1]))/1430.
            u=u+dt*du;v=v+dt*dv;vx=c*u-s*v;vy=s*u+c*v;r=r+dt*dr
            paths.append(torch.stack((x,y),-1))
        paths=torch.stack(paths,-2)
        if self.peer or pairwise:
            delta=paths[:,:,None,:,None]-paths[:,None,:,None]
            minimum=delta.norm(dim=-1).amin(-1)
            if return_distance:return minimum
            return (self.risk_clearance-minimum).clamp_min(0.).square()
        time=torch.arange(1,len(paths[0,0,0])+1,device=motion.device)*dt
        constant=motion[:,:,:2][:,:,None]+time[None,None,:,None]*motion[:,:,3:5][:,:,None]
        delta=paths[:,:,None]-constant[:,None,:,None]
        minimum=delta.norm(dim=-1).amin(-1)
        mask=~torch.eye(n,dtype=torch.bool,device=motion.device)[None,:,:,None]
        return ((self.risk_clearance-minimum).clamp_min(0.).square()*mask).amax(2)

    def feedback_observation(self,packet,state,time,joint=False):
        """Original source observations at nominal candidate poses.

        Each candidate refreshes the unchanged source neural and Boids commands;
        neighbors follow public constant-velocity forecasts. These are predictor
        inputs only. No new controller or simulator-private state is executed.
        """
        motion=packet['motion'];b,k,n=state.shape[:3]
        position=state[...,:2];velocity=state[...,3:5];heading=state[...,2]
        peers=motion[:,:,:2]+time*motion[:,:,3:5]
        relative=(position[:,:,None]-position[:,:,:,None] if joint else
            peers[:,None,None]-position[:,:,:,None])
        distance=relative.norm(dim=-1)
        mask=~torch.eye(n,dtype=torch.bool,device=state.device)[None,None]
        separation=(-relative/(distance-3.5).clamp_min(.01)[...,None].square()
            *((distance<15.)&mask)[...,None]).sum(-2)
        alignment=(velocity.mean(2,keepdim=True).expand_as(velocity) if joint else
            (motion[:,:,3:5].sum(1)[:,None,None]-motion[:,None,:,3:5]+velocity)/n)
        cohesion=(position.mean(2,keepdim=True)-position if joint else
            (peers.sum(1)[:,None,None]-peers[:,None]+position)/n-position)
        h0=motion[:,0,2];c0,s0=h0.cos(),h0.sin()
        def world(vector):
            return torch.stack((c0*vector[:,0]-s0*vector[:,1],s0*vector[:,0]+c0*vector[:,1]),-1)
        attacker=motion[:,0,:2]+world(packet['node'][:,0,2:4]*30.)
        att_velocity=world(packet['node'][:,0,4:6]*5.)
        target=attacker+time*att_velocity
        attraction=target[:,None,None]-position
        field=.5*attraction+10.*separation+.1*alignment+.1*cohesion
        field=field/field.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        c,s=heading.cos(),heading.sin()
        forward=c*field[...,0]+s*field[...,1]
        turn=torch.atan2(-s*field[...,0]+c*field[...,1],forward)
        thrust=(1000.*torch.stack((forward+.75*turn,forward-.75*turn),-1)).clamp(-500.,1000.)
        boids=(thrust-250.)/750.
        def polar(vector):
            angle=torch.atan2(vector[...,1],vector[...,0])-heading
            return torch.stack((vector.norm(dim=-1),torch.remainder(angle+np.pi,2*np.pi)-np.pi),-1)
        neighbor_angle=torch.atan2(relative[...,1],relative[...,0])-heading[...,None]
        neighbor_polar=torch.stack((distance,torch.remainder(neighbor_angle+np.pi,2*np.pi)-np.pi),-1)
        neighbors=neighbor_polar[:,:,mask[0,0]].reshape(b,k,n,2*(n-1))
        obs=torch.cat((polar(-position),polar(attraction),
            polar(att_velocity[:,None,None].expand_as(position)),polar(separation),
            polar(alignment),polar(cohesion),boids,neighbors),-1)
        return obs,boids

    @torch.no_grad()
    def refine_joint_scores(self,packet,proposals,plans,short_scores,base_scores):
        """Re-evaluate shortlisted mixtures with mutually consistent feedback.

        Every vessel in a candidate combination responds to the positions and
        velocities generated by that same combination. The no-peer control
        retains the constant-velocity neighbor predictor and separable costs.
        """
        if not self.coupled_shortlist:return short_scores
        b,n=proposals.shape[:2];k=self.levels;device=proposals.device
        powers=k**torch.arange(n,device=device)
        best=short_scores.argmax(-1);digits=(plans[best]*(k-1)).round().long()
        neighbors=best[:,None,None]+(torch.arange(k,device=device)[None,None]-digits[...,None])*powers[None,:,None]
        uniform=torch.arange(k,device=device)*powers.sum()
        indices=torch.cat((short_scores.topk(min(self.coupled_shortlist,len(plans)),dim=-1).indices,
            neighbors.flatten(1),uniform[None].expand(b,-1)),1)
        # Duplicate plans are harmless and preserve a rectangular batched
        # rollout without changing the set of candidate combinations.
        gates=plans[indices];m=gates.shape[1]
        state=packet['motion'][:,None].expand(-1,m,-1,-1).clone()
        force=750.*(gates[...,None]*proposals[:,None]+(1.-gates[...,None])*packet['node'][:,None,:,11:13])+250.
        minimum=None;dt=.1
        for step in range(round(self.risk_horizon/dt)):
            if step and step%2==0:
                obs,boids=self.feedback_observation(packet,state,step*dt,joint=self.peer)
                future=self.proposal_distribution(obs.reshape(b*m,n,-1))[0].tanh().reshape(b,m,n,2)
                force=750.*(gates[...,None]*future+(1.-gates[...,None])*boids)+250.
            x,y,heading,vx,vy,r=state.unbind(-1)
            x=x+dt*vx;y=y+dt*vy;heading=heading+dt*r
            c,s=heading.cos(),heading.sin();u=c*vx+s*vy;v=-s*vx+c*vy
            du=(400.*r*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
            dv=(-380.*r*u-(100.+150.*v.abs())*v)/400.
            dr=(-20.*u*v-(980.+950.*r.abs())*r+1.25*(force[...,0]-force[...,1]))/1430.
            u=u+dt*du;v=v+dt*dv;vx=c*u-s*v;vy=s*u+c*v;r=r+dt*dr
            state=torch.stack((x,y,heading,vx,vy,r),-1);position=state[...,:2]
            peers=(position if self.peer else
                (packet['motion'][:,:,:2]+dt*(step+1)*packet['motion'][:,:,3:5])[:,None])
            distance=(position[:,:,:,None]-peers[:,:,None]).norm(dim=-1)
            minimum=distance if minimum is None else torch.minimum(minimum,distance)
        risk=(self.risk_clearance-minimum).clamp_min(0.).square()
        mask=~torch.eye(n,dtype=torch.bool,device=device)
        cost=((risk*torch.triu(mask,diagonal=1)).sum((-1,-2)) if self.peer else
            (risk*mask).amax(-1).sum(-1))
        rescored=base_scores.gather(1,indices)-self.risk_prior*cost
        result=torch.full_like(short_scores,-1e6)
        result.scatter_(1,indices,rescored)
        return result

    def proposal_distribution(self,obs):
        b,n,d=obs.shape;net=self.base;act=net.activation
        x=torch.cat((act(net.f1(obs[...,:6])),act(net.f2(obs[...,6:14])),
            act(net.t1(obs[...,14:].reshape(b,n,n-1,2))).mean(-2)),-1)
        h=act(net.l2(act(net.l1(x))))
        pm=net.mean_layer(h);ps=net.log_std_layer(h).clamp(-3.,-.3).exp()
        return pm,ps,h

    @torch.no_grad()
    def fast_action(self,packet,intent):
        """Replan physical fusion every source step under a persistent preference.

        The preference biases the unchanged source-proposal mixture; candidate
        interaction prediction stays responsive at 0.2 seconds. Holding a
        preference never holds the actual mixture coefficient or thrust.
        """
        proposals,_,h=self.proposal_distribution(packet['obs']);proposals=proposals.tanh()
        b,n=proposals.shape[:2];net=self.base;k=self.levels
        own=net.activation(net.a1(torch.cat((proposals,packet['node'][...,11:13]),-1)))
        prior=.02+.96*(net.adap_layer(torch.cat((own,h),-1)).tanh().squeeze(-1)*.5+.5)
        center=(2.*torch.tanh(torch.logit(prior)/2.)).sigmoid()
        indices=torch.arange(k**n,device=proposals.device)
        choices=((indices[:,None]//(k**torch.arange(n,device=proposals.device)))%k).long()
        plans=choices.to(proposals.dtype)/(k-1)
        scores=-((plans[None]-center[:,None])/.4).square().sum(-1)
        scores=scores+self.intent_strength*torch.einsum('bn,kn->bk',2.*intent[...,2]-1.,plans)
        base_scores=scores.clone() if self.coupled_shortlist else None
        risk=self.candidate_risk(packet,proposals)
        if self.peer:
            for i in range(n):
                for j in range(i+1,n):scores=scores-self.risk_prior*risk[:,i,j,choices[:,i],choices[:,j]]
        else:
            for i in range(n):scores=scores-self.risk_prior*risk[:,i,choices[:,i]]
        if self.coupled_shortlist:scores=self.refine_joint_scores(packet,proposals,plans,scores,base_scores)
        gates=plans[scores.argmax(-1)]
        return torch.cat((proposals,gates[...,None]),-1)

    @torch.no_grad()
    def value_encoding(self,packet):
        """Expose the public fusion decoder to the training-only option critic."""
        obs=packet.get('obs')
        if obs is None:
            obs,_=self.feedback_observation(packet,packet['motion'][:,None],0.)
            obs=obs[:,0]
            obs[...,12:14]=packet['node'][...,11:13]
        pm,_,h=self.proposal_distribution(obs);proposals=pm.tanh();b,n=proposals.shape[:2];k=self.levels
        own=self.base.activation(self.base.a1(torch.cat((proposals,packet['node'][...,11:13]),-1)))
        prior=.02+.96*(self.base.adap_layer(torch.cat((own,h),-1)).tanh().squeeze(-1)*.5+.5)
        center=(2.*torch.tanh(torch.logit(prior)/2.)).sigmoid()
        indices=torch.arange(k**n,device=proposals.device)
        choices=((indices[:,None]//(k**torch.arange(n,device=proposals.device)))%k).long()
        plans=choices.to(proposals.dtype)/(k-1)
        scores=-((plans[None]-center[:,None])/.4).square().sum(-1)
        risk=packet.get('_candidate_risk')
        if risk is None:risk=self.candidate_risk(dict(packet,obs=obs),proposals)
        if self.peer:
            for i in range(n):
                for j in range(i+1,n):scores-=self.risk_prior*risk[:,i,j,choices[:,i],choices[:,j]]
        else:
            for i in range(n):scores-=self.risk_prior*risk[:,i,choices[:,i]]
        return dict(_fusion_scores=scores,_fusion_proposals=proposals,_fusion_risk=risk)

    def forward(self,packet,latent=None,deterministic=False):
        obs=packet['obs'];b,n,d=obs.shape;net=self.base;act=net.activation
        pm,ps,h=self.proposal_distribution(obs)
        pz=(pm if deterministic or self.gate_period>1 or self.frozen_proposal else pm+ps*torch.randn_like(pm)) if latent is None else latent[...,:2]
        proposals=pz.tanh();boids=packet['node'][...,11:13]
        own=act(net.a1(torch.cat((proposals,boids),-1)))
        prior=.02+.96*(net.adap_layer(torch.cat((own,h),-1)).tanh().squeeze(-1)*.5+.5)
        context=self.context(packet,proposals,prior)
        gm=torch.logit(self.adapter(context).clamp(1e-5,1.-1e-5))
        if self.graph:
            # Retain the full joint distribution. Averaging independent gates
            # would discard mutually exclusive, coordinated combinations.
            k=self.levels;indices=torch.arange(k**n,device=obs.device)
            choices=((indices[:,None]//(k**torch.arange(n,device=obs.device)))%k).long()
            bits=choices.to(obs.dtype)/(k-1)
            unary=2.*torch.tanh(gm/2.)
            scores=(torch.einsum('bn,kn->bk',unary,bits) if k==2 else
                -((bits[None]-unary.sigmoid()[:,None])/(.3 if self.intent_strength else .4)).square().sum(-1))
            risk=packet.get('_candidate_risk')
            if risk is None and (self.risk_prior or self.forecast_inputs):risk=self.candidate_risk(packet,proposals)
            pair_context=context['edge' if self.peer else 'edge_no_peer']
            if self.forecast_inputs:
                forecast=(risk if self.peer else risk[:,:,None,:,None].expand(-1,-1,n,-1,k))
                pair_context=torch.cat((pair_context,forecast.flatten(-2).clamp_max(36.)/9.),-1)
            if self.peer:
                factors=self.gate_pairs(pair_context).reshape(b,n,n,k,k)
                for i in range(n):
                    for j in range(i+1,n):
                        pair=factors[:,i,j,choices[:,i],choices[:,j]]
                        # Preserve the older binary prototypes on reload. The
                        # mixed-action graph symmetrizes both receiver frames.
                        if k>2:pair=.5*(pair+factors[:,j,i,choices[:,j],choices[:,i]])
                        scores=scores+self.pair_scale*pair/max(1,n-1)
            elif self.pair_scale!=1.:
                # Capacity control for the scaled interaction model: retain
                # the same learned network and score range, but allow only a
                # separable own-candidate correction against neighbor motion.
                # There are no peer candidate inputs or joint-gate factors.
                local=self.gate_pairs(pair_context).reshape(b,n,n,k,k).mean(-1)
                mask=~torch.eye(n,dtype=torch.bool,device=obs.device)[None,:,:,None]
                local=(local*mask).sum(2)/max(1,n-1)
                for i in range(n):scores=scores+self.pair_scale*local[:,i,choices[:,i]]
            if self.risk_prior and not self.intent_strength:
                base_scores=scores.clone() if self.coupled_shortlist else None
                if self.peer:
                    for i in range(n):
                        for j in range(i+1,n):scores=scores-self.risk_prior*risk[:,i,j,choices[:,i],choices[:,j]]
                else:
                    for i in range(n):scores=scores-self.risk_prior*risk[:,i,choices[:,i]]
                if self.coupled_shortlist:scores=self.refine_joint_scores(packet,proposals,bits,scores,base_scores)
            dist=torch.distributions.Categorical(logits=scores)
            if latent is None:choice=scores.argmax(-1) if deterministic else dist.sample()
            else:choice=((latent[...,2]*(k-1)).round().long()*(k**torch.arange(n,device=obs.device))).sum(-1)
            gates=bits[choice]
            logp=torch.stack((torch.distributions.Normal(pm,ps).log_prob(pz).sum(-1),
                dist.log_prob(choice)[:,None].expand(-1,n)),-1)
            entropy=torch.stack((torch.distributions.Normal(pm,ps).entropy().sum(-1),
                (dist.entropy()/n)[:,None].expand(-1,n)),-1)
            return torch.cat((proposals,gates[...,None]),-1),torch.cat((pz,gates[...,None]),-1),logp,entropy,scores,bits
        gs=self.gate_logstd.clamp(-2.5,.1).exp().expand_as(gm)
        gz=(gm if deterministic else gm+gs*torch.randn_like(gm)) if latent is None else latent[...,2]
        action=torch.cat((proposals,gz.sigmoid()[...,None]),-1)
        p_dist=torch.distributions.Normal(pm,ps);g_dist=torch.distributions.Normal(gm,gs)
        logp=torch.stack((p_dist.log_prob(pz).sum(-1),g_dist.log_prob(gz)),-1)
        entropy=torch.stack((p_dist.entropy().sum(-1),g_dist.entropy()),-1)
        return action,torch.cat((pz,gz[...,None]),-1),logp,entropy,gm,gs


class JointLearningValue(nn.Module):
    """Central state value and joint physical-control value with paired effects."""
    def __init__(self,width=96,response=False,options=False,relational=False,intents=False,probability=False,
            decoded=False,levels=3,intent_strength=0.,invariant=False):
        super().__init__();self.response=response;self.options=options;self.relational=relational;self.intents=intents
        self.probability=probability
        self.decoded=decoded;self.levels=levels;self.intent_strength=intent_strength
        self.invariant=invariant
        self.node=nn.Sequential(nn.Linear(14,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.pair=nn.Sequential(nn.Linear(9,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.value=nn.Sequential(nn.Linear(width*4+9,192),nn.SiLU(),nn.Linear(192,1))
        self.control_node=nn.Sequential(nn.Linear((21 if response else 16)+(5 if options else 0)+(3 if decoded else 0),width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.control_pair=nn.Sequential(nn.Linear((16 if response else 13)+(10 if options else 0)+(7 if decoded else 0),width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.advantage=nn.Sequential(nn.Linear(width*4+9,192),nn.SiLU(),nn.Linear(192,1))
        if relational:
            self.messages=nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
            self.control_update=nn.Sequential(nn.Linear(3*width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        nn.init.zeros_(self.advantage[-1].weight);nn.init.zeros_(self.advantage[-1].bias)

    def response_features(self,packet,force):
        # A nominal response is an input to the learned value, never a new
        # executed controller. Only the training critic reads water velocity.
        motion=packet['motion'];x,y,heading=[motion[...,i].clone() for i in range(3)]
        vx,vy,r=[packet['water'][...,i]*5. for i in range(3)]
        thrust=750.*force+250.;minimum=None;closest=None;capture=None
        attacker=packet['task'][...,:2]*60.;att_vel=packet['task'][...,4:6]*5.
        for step in range(24):
            x=x+.1*vx;y=y+.1*vy;heading=heading+.1*r
            c,s=heading.cos(),heading.sin();u=c*vx+s*vy;v=-s*vx+c*vy
            du=(400.*r*v-(100.+150.*u.abs())*u+thrust.sum(-1))/380.
            dv=(-380.*r*u-(100.+150.*v.abs())*v)/400.
            dr=(-20.*u*v-(980.+950.*r.abs())*r+1.25*(thrust[...,0]-thrust[...,1]))/1430.
            u=u+.1*du;v=v+.1*dv;vx=c*u-s*v;vy=s*u+c*v;r=r+.1*dr
            p=torch.stack((x,y),-1)
            distance=(p[:,:,None]-p[:,None]).norm(dim=-1)
            close=(p-(attacker+.1*(step+1)*att_vel)[:,None]).norm(dim=-1)
            if minimum is None:
                minimum=distance;closest=torch.full_like(distance,(step+1)/24.);capture=close
            else:
                closest=torch.where(distance<minimum,(step+1)/24.,closest)
                minimum=torch.minimum(minimum,distance);capture=torch.minimum(capture,close)
        node=torch.stack((capture/30.,(x.square()+y.square()).clamp_min(1e-8).sqrt()/30.,
            (heading-motion[...,2]).sin(),(heading-motion[...,2]).cos(),(vx.square()+vy.square()).clamp_min(1e-8).sqrt()/5.),-1)
        edge=torch.stack((minimum/15.,distance/15.,closest),-1)
        return node.clamp(-6.,6.),edge.clamp(-6.,6.)

    def forward(self,packet,action=None):
        water=packet['water'];task=packet['task']
        if self.invariant:
            # Match the Adapter's body-frame geometry. A joint rotation of the
            # encounter must not change its return or a single-vessel effect.
            heading=packet['motion'][...,2];c,s=heading.cos(),heading.sin()
            water=torch.stack((c*water[...,0]+s*water[...,1],
                -s*water[...,0]+c*water[...,1],water[...,2]),-1)
            radial=task[:,:2];radius=radial.norm(dim=-1).clamp_min(1e-8)
            radial=radial/radius[:,None];velocity=task[:,4:6]
            direction=torch.stack((task[:,3],task[:,2]),-1)
            def cross(a,b):return a[:,0]*b[:,1]-a[:,1]*b[:,0]
            closest=(packet['motion'][...,:2]/60.-task[:,None,:2]).norm(dim=-1).amin(1)
            task=torch.stack((radius,(radial*velocity).sum(-1),cross(radial,velocity),
                (radial*direction).sum(-1),cross(radial,direction),task[:,6],task[:,7],task[:,8],closest),-1)
        node=torch.cat((packet['node'][...,:9],packet['node'][...,14:16],water),-1)
        edge=packet['edge'][...,:9];n=node.shape[1]
        mask=~torch.eye(n,dtype=torch.bool,device=node.device)[None,:,:,None]
        def pool(a,b):
            return torch.cat((a.mean(1),a.amax(1),(b*mask).sum((1,2))/(n*(n-1)),
                b.masked_fill(~mask,-1e5).amax((1,2)),task),-1)
        state=self.value(pool(self.node(node),self.pair(edge))).squeeze(-1)
        def project(raw):
            if not self.probability:return raw
            potential=-(packet['motion'][...,:2]-packet['task'][:,:2,None].transpose(1,2)*60.).norm(dim=-1).amin(1)/60.
            return raw.tanh()-.2*potential
        if action is None:return project(state)
        force=action[...,2:3]*action[...,:2]+(1.-action[...,2:3])*packet['node'][...,11:13]
        if self.intents:
            # A latent fusion preference is not a physical thrust. Encode it
            # explicitly alongside both source candidates in the option input.
            preference=2.*action[...,2]-1.
            force=torch.stack((preference,preference.square()),-1)
        ni=[node,force];ei=[edge,force[:,:,None].expand(-1,-1,n,-1),force[:,None].expand(-1,n,-1,-1)]
        if self.options:
            option=torch.cat((action,packet['node'][...,11:13]),-1)
            ni.append(option);ei.extend((option[:,:,None].expand(-1,-1,n,-1),option[:,None].expand(-1,n,-1,-1)))
        if self.decoded:
            with torch.no_grad():
                k=self.levels;index=torch.arange(k**n,device=action.device)
                choices=((index[:,None]//(k**torch.arange(n,device=action.device)))%k).long()
                plans=choices.to(action.dtype)/(k-1)
                score=packet['_fusion_scores']+self.intent_strength*torch.einsum('bn,kn->bk',2.*action[...,2]-1.,plans)
                selected=score.argmax(-1);digits=choices[selected];gate=plans[selected]
                applied=gate[...,None]*packet['_fusion_proposals']+(1.-gate[...,None])*packet['node'][...,11:13]
                decoded=torch.cat((applied,gate[...,None]),-1)
                risk=packet['_fusion_risk'];batch=torch.arange(len(action),device=action.device)
                boat=torch.arange(n,device=action.device)
                if risk.ndim==5:
                    chosen_risk=risk[batch[:,None,None],boat[None,:,None],boat[None,None,:],digits[:,:,None],digits[:,None,:]]
                else:
                    own_risk=risk[batch[:,None],boat[None],digits]
                    chosen_risk=torch.maximum(own_risk[:,:,None],own_risk[:,None,:])
            ni.append(decoded)
            ei.extend((decoded[:,:,None].expand(-1,-1,n,-1),decoded[:,None].expand(-1,n,-1,-1),
                chosen_risk.clamp_max(36.)[...,None]/9.))
        if self.response:
            rn,re=self.response_features(packet,force);ni.append(rn);ei.append(re)
        cn=self.control_node(torch.cat(ni,-1));ce=self.control_pair(torch.cat(ei,-1))
        if self.relational:
            ce=self.messages(torch.cat((cn[:,:,None].expand(-1,-1,n,-1),
                cn[:,None].expand(-1,n,-1,-1),ce),-1))
            cn=self.control_update(torch.cat((cn,(ce*mask).sum(2)/max(1,n-1),
                ce.masked_fill(~mask,-1e5).amax(2)),-1))
        return project(state+self.advantage(pool(cn,ce)).squeeze(-1))


def joint_task_step(env,action,gamma=.995,step_cost=.001):
    # Common training objective for every new-method ablation. Author dynamics,
    # rewards and reported terminal classes remain untouched.
    potential=-float(env.def_att_dists.min())/60.
    obs,source_reward,done,_=env.step(action,'AdaRes')
    following=0. if done else -float(env.def_att_dists.min())/60.
    reward=-step_cost+(1. if done>2 else -1. if done else 0.)+.2*(gamma*following-potential)
    return obs,float(reward),done,float(np.mean(source_reward))


def joint_conditional_indices(gates,levels):
    """All single-vessel alternatives, with every other coefficient fixed."""
    digits=(gates*(levels-1)).round().long();n=digits.shape[-1]
    powers=levels**torch.arange(n,device=gates.device)
    index=(digits*powers).sum(-1)
    return index[:,None,None]+(torch.arange(levels,device=gates.device)[None,None]-digits[...,None])*powers[None,:,None]


def joint_factor_advantages(actor,packet,actions,scale,reference=None):
    """Use the deployed joint Adapter scores as the conditional value factors.

    Center on the frozen behavior action during fitted policy improvement.
    This state-only offset cancels in every single-vessel contrast and
    preserves exactly the deployed greedy ranking.
    """
    packet=dict(packet)
    if 'obs' not in packet:
        with torch.no_grad():
            obs,_=actor.feedback_observation(packet,packet['motion'][:,None],0.)
            packet['obs']=obs[:,0];packet['obs'][...,12:14]=packet['node'][...,11:13]
    if '_fusion_risk' in packet:packet['_candidate_risk']=packet['_fusion_risk']
    _,_,_,_,scores,_=actor(packet,deterministic=True)
    value_scores=scores-packet['_direct_prior_logits'] if '_direct_prior_logits' in packet else scores
    n=packet['node'].shape[1];powers=actor.levels**torch.arange(n,device=scores.device)
    if reference is None:
        neutral=int(((actor.levels-1)//2)*powers.sum());center=value_scores[:,neutral]
    else:
        reference_index=((reference[...,2]*(actor.levels-1)).round().long()*powers).sum(-1)
        center=value_scores.gather(1,reference_index[:,None]).squeeze(1)
    advantages=[]
    for action in actions:
        index=((action[...,2]*(actor.levels-1)).round().long()*powers).sum(-1)
        advantages.append(scale*(value_scores.gather(1,index[:,None]).squeeze(1)-center))
    return advantages,scores


@lru_cache(maxsize=3)
def joint_load(path,revision):
    data=torch.load(path,map_location='cpu',weights_only=False)
    actor=JointLearningActor(data['peer'],graph=data.get('graph',False),levels=data.get('levels',2),
        risk_prior=data.get('risk_prior',0.),risk_horizon=data.get('risk_horizon',1.6),
        pair_scale=data.get('pair_scale',1.),gate_period=data.get('gate_period',1),
        risk_clearance=data.get('risk_clearance',6.),forecast_inputs=data.get('forecast_inputs',False),
        frozen_proposal=data.get('frozen_proposal',False),risk_feedback=data.get('risk_feedback',False),
        intent_strength=data.get('intent_strength',0.),coupled_shortlist=data.get('coupled_shortlist',0))
    actor.load_state_dict(data['actor']);actor.eval()
    actor.initial_root=data.get('initial_root',False)
    actor.failure_roots=data.get('failure_roots',False)
    actor.failure_lookback=data.get('failure_lookback',0.)
    actor.failure_subjects=data.get('failure_subjects',False)
    actor.joint_contexts=data.get('joint_contexts',0)
    actor.training_gamma=1. if data.get('success_objective',False) else .995
    actor.training_step_cost=0. if data.get('success_objective',False) else .001
    actor.branch_value_data=data.get('branch_value_data',False)
    actor.branch_grid=data.get('branch_grid',False)
    actor.fitted_iteration=data.get('fitted_iteration',False)
    return actor


def joint_advance(env,obs,actor,action,maximum=0):
    """One coordinated gate decision; both source proposals still update at 5 Hz."""
    period=min(actor.gate_period,maximum) if maximum else actor.gate_period
    reward=0.;source_return=0.;done=0;control=action.copy()
    for step in range(period):
        if actor.intent_strength:
            with torch.no_grad():
                control=actor.fast_action(mechanism_tensor(joint_packet(env,obs)),torch.tensor(action[None]))[0].numpy()
        elif step:
            tensor=torch.tensor(obs[None],dtype=torch.float32)
            control[:,:2]=actor.proposal_distribution(tensor)[0][0].tanh().numpy()
        obs,r,done,sr=joint_task_step(env,control,actor.training_gamma,actor.training_step_cost)
        reward+=(actor.training_gamma**step)*r;source_return+=(.99**step)*sr
        if done:break
    return obs,reward,done,source_return,step+1


def joint_episode(task):
    scene,path,revision,collect,roots_count,repeats,horizon,limit=task
    seed,n,agility=scene;np.random.seed(seed)
    actor=joint_load(path,revision);torch.manual_seed(seed);env=numerical_source().TADEnv(n)
    obs,_=env.reset(agility=agility,noisy_agility=False)
    done=0;steps=[];roots=[];rng=np.random.default_rng(seed^917803);total=0.;discount=1.;physical_steps=0
    while not done and (not limit or physical_steps<limit):
        packet=joint_packet(env,obs)
        with torch.no_grad():action,latent,lp,_,_,_=actor(mechanism_tensor(packet),deterministic=not collect or actor.fitted_iteration)
        action=action[0].numpy()
        if collect and roots_count:
            weight=1.+np.exp(-((env.def_def_dists.min()-7.)/3.)**2)
            hindsight=actor.failure_roots and actor.failure_lookback>0.
            key=(float(env.Current_T) if hindsight else
                1. if not physical_steps and getattr(actor,'initial_root',False) else
                float(np.log(max(rng.random(),1e-12))/weight))
            capacity=(int(np.ceil(actor.failure_lookback*roots_count/(.2*actor.gate_period)))+2
                if hindsight else roots_count)
            if len(roots)<capacity or key>min(r[0] for r in roots):
                root=copy.deepcopy(env)
                for name in ('Pos_Att','Phi_Att','Pos_Def','Phi_Def','Rewards'):
                    value=getattr(root,name)
                    if np.ndim(value)>1:setattr(root,name,value[-1:].copy())
                roots.append((key,root,obs.copy(),packet,action.copy()))
                if len(roots)>capacity:roots.pop(min(range(len(roots)),key=lambda i:roots[i][0]))
        obs,reward,done,sr,span=joint_advance(env,obs,actor,action,limit-physical_steps if limit else 0)
        total+=discount*sr;discount*=.99**span;physical_steps+=span
        if collect:steps.append(dict(packet=packet,action=action,latent=latent[0].numpy(),
            logp=lp[0].numpy(),reward=reward,done=done,discount=actor.training_gamma**span,physical_steps=span))
    row=dict(seed=int(seed),defenders=n,agility=agility,**outcome(env,done),discounted_return=total)
    if not collect:return row
    tail=None if done else joint_packet(env,obs)
    if actor.failure_roots and done>2:roots=[]
    elif roots and actor.failure_roots and actor.failure_lookback>0.:
        # Hindsight chooses training states only. Deployed decisions never
        # receive the eventual failure time or any realized future noise.
        selected={}
        for j in range(roots_count):
            target=env.Current_T-actor.failure_lookback*(j+1)
            eligible=[root for root in roots if root[0]<=target+1e-6]
            root=max(eligible,key=lambda r:r[0]) if eligible else min(roots,key=lambda r:r[0])
            selected[root[0]]=root
        roots=list(selected.values())
    branches=[];value_steps=[];simulation_steps=physical_steps
    event_subjects=None
    if actor.failure_subjects and done in (1,2):
        if done==2:
            pos=np.asarray([v.pos for v in env.defender_list])
            distances=np.linalg.norm(pos[:,None]-pos[None],axis=-1)+1e6*np.eye(n)
            event_subjects=np.unravel_index(distances.argmin(),distances.shape)
        else:
            distances=np.asarray([np.linalg.norm(v.pos-env.attacker.pos) for v in env.defender_list])
            event_subjects=tuple(np.argsort(distances)[:2])
    for _,root,root_obs,packet,reference in roots:
        p=packet['motion'][:,:2];distance=np.linalg.norm(p[:,None]-p[None],axis=-1)+1e6*np.eye(n)
        subjects=np.unravel_index(distance.argmin(),distance.shape)
        if actor.branch_grid:
            if event_subjects is not None:
                subjects=event_subjects
            else:
                with torch.no_grad():
                    predicted=actor.candidate_risk(mechanism_tensor(packet),
                        torch.tensor(reference[None,:,:2]),return_distance=True,pairwise=True)[0].numpy()
                if actor.intent_strength:
                    with torch.no_grad():physical=actor.fast_action(mechanism_tensor(packet),torch.tensor(reference[None]))[0].numpy()
                else:physical=reference
                digits=np.rint(physical[:,2]*(actor.levels-1)).astype(int)
                distance=predicted[np.arange(n)[:,None],np.arange(n)[None,:],digits[:,None],digits[None,:]]+1e6*np.eye(n)
                subjects=np.unravel_index(distance.argmin(),distance.shape)
            arms=[];edges=[];k=actor.levels
            for x in range(k):
                for y in range(k):
                    arm=reference.copy();arm[subjects[0],2]=x/(k-1);arm[subjects[1],2]=y/(k-1);arms.append(arm)
            for fixed in range(k):
                for left in range(k):
                    for right in range(left+1,k):
                        edges.extend(((left*k+fixed,right*k+fixed),(fixed*k+left,fixed*k+right)))
            reference_index=int(round(reference[subjects[0],2]*(k-1)))*k+int(round(reference[subjects[1],2]*(k-1)))
        else:
            arms=[reference.copy() for _ in range(4)];edges=((0,1),(0,2),(2,3),(1,3))
            reference_index=0
            for k,i in enumerate(subjects):
                alternate=1. if reference[i,2]<.5 else 0.
                arms[k+1][i,2]=alternate;arms[3][i,2]=alternate
        if actor.joint_contexts:
            # Explore coordinated contexts before assigning single-vessel
            # credit. A pair-only neighborhood of an unsuccessful joint action
            # can be flat even when several coordinated changes would succeed.
            edges=list(edges);known={tuple(a[:,2]):j for j,a in enumerate(arms)}
            def add_arm(action):
                key=tuple(action[:,2])
                if key not in known:known[key]=len(arms);arms.append(action)
                return known[key]
            for context_id in range(actor.joint_contexts):
                context=reference.copy()
                if context_id<2:context[:,2]=float(context_id)
                elif context_id%2:
                    context[:,2]=rng.integers(0,2,size=n)
                else:
                    probability=np.asarray([.25,.5,.25]) if actor.levels==3 else None
                    context[:,2]=rng.choice(actor.levels,size=n,p=probability)/(actor.levels-1)
                changed=np.flatnonzero(context[:,2]!=reference[:,2])
                if not len(changed):continue
                subject=int(rng.choice(changed));deletion=context.copy()
                deletion[subject,2]=reference[subject,2]
                edge=(add_arm(context),add_arm(deletion))
                if edge not in edges and edge[::-1] not in edges:edges.append(edge)
        for _ in range(repeats):
            future=int(rng.integers(1,2**31-1));out=[]
            for action in arms:
                np.random.seed(future);torch.manual_seed(future)
                branch=copy.deepcopy(root);bo=root_obs.copy();bd=0;reward=0.;gamma=1.;count=0;continuation=[];first_step=None
                while not bd and count<horizon:
                    branch_packet=packet if not count else joint_packet(branch,bo)
                    if count:
                        with torch.no_grad():action=actor(mechanism_tensor(branch_packet),deterministic=actor.fitted_iteration)[0][0].numpy()
                    entry=None
                    if actor.branch_value_data and count and count%max(4,actor.gate_period)==0:
                        # The intervention has ended: these actions come from
                        # the frozen continuation policy, so their suffixes
                        # supervise V^pi and Q^pi without a root-action bias.
                        entry=dict(packet={key:branch_packet[key] for key in ('node','edge','motion','water','task')},action=action.copy())
                    bo,r,bd,_,span=joint_advance(branch,bo,actor,action,horizon-count)
                    if not count and actor.branch_value_data:
                        first_step=dict(reward=r,discount=actor.training_gamma**span,
                            next=None if bd else {key:value for key,value in joint_packet(branch,bo).items()
                                if key in ('node','edge','motion','water','task')})
                    reward+=gamma*r;gamma*=actor.training_gamma**span;count+=span
                    if actor.branch_value_data:continuation.append((entry,r,actor.training_gamma**span))
                simulation_steps+=count
                if actor.branch_value_data:
                    if not bd:raise ValueError('Continuation value labels require true terminal branches.')
                    suffix=0.
                    for entry,r,discount_step in reversed(continuation):
                        suffix=r+discount_step*suffix
                        if entry is not None:entry['mc']=suffix;value_steps.append(entry)
                out.append(dict(reward=reward,discount=gamma,next=None if bd else joint_packet(branch,bo),
                    terminal=bd,success=int(bd>2) if bd else None,first_step=first_step))
            for a,b in edges:
                branches.append(dict(packet=packet,left=arms[a],right=arms[b],before=out[a],after=out[b],
                    reference_action=reference,reference_outcome=out[reference_index],
                    root=(int(seed),int(round(root.Current_T/.2)),a,b)))
    return dict(row=row,steps=steps,tail=tail,branches=branches,value_steps=value_steps,simulation_steps=simulation_steps)


def joint_risk_search(args,pool):
    """Tune the untrained candidate predictor only on the development bank."""
    args.output.mkdir(parents=True,exist_ok=True);torch.manual_seed(args.seed)
    actor=JointLearningActor(not args.without_peer,True,args.joint_levels,args.joint_risk_prior,args.joint_risk_horizon)
    actor.risk_feedback=args.joint_risk_feedback
    actor.coupled_shortlist=args.joint_coupled_shortlist
    actor.base.load_state_dict(torch.load(args.source,map_location='cpu',weights_only=True));actor.eval()
    values=nn.ModuleList([JointLearningValue() for _ in range(2)])
    if args.joint_gate_lr:
        opt=torch.optim.Adam([dict(params=actor.base.parameters(),lr=args.joint_actor_lr),
            dict(params=[p for name,p in actor.named_parameters() if not name.startswith('base.')],lr=args.joint_gate_lr)])
    else:opt=torch.optim.Adam(actor.parameters(),lr=args.joint_actor_lr)
    vopt=torch.optim.Adam(values.parameters(),lr=5e-4);trials=[];best=(-1.,-1.)
    settings=[(h,r,c) for h in (args.joint_risk_horizons or (1.6,2.4,3.2,4.))
        for r in (args.joint_risk_scales or (args.joint_risk_prior,))
        for c in (args.joint_risk_clearances or (args.joint_risk_clearance,))]
    for horizon,risk,clearance in settings:
        actor.risk_horizon=horizon;actor.risk_prior=risk;actor.risk_clearance=clearance
        model=dict(format='joint-neural-source-ppo-v1',actor=actor.state_dict(),values=values.state_dict(),
            actor_optimizer=opt.state_dict(),value_optimizer=vopt.state_dict(),iteration=0,history=[],
            peer=actor.peer,graph=True,levels=actor.levels,risk_prior=actor.risk_prior,risk_horizon=horizon,
            risk_clearance=clearance,
            risk_feedback=actor.risk_feedback,
            coupled_shortlist=actor.coupled_shortlist,
            counterfactual=not args.joint_no_cf,simulation_steps=0,source_sha256=BASELINE_SHA256,
            source_commit=SOURCE_COMMIT,seed=args.seed,initialization='original source neural proposal; no intervention learning')
        path=args.output/'current.pth';torch.save(model,path)
        rows=list(pool.map(joint_episode,[(s,str(path),path.stat().st_mtime_ns,False,0,0,0,0)
            for s in bank(args.selection_bank,args.selection_per_cell)],chunksize=1))
        summary=summarize(rows);score=(summary['success'],summary['capture'])
        trial=dict(horizon=horizon,risk_prior=actor.risk_prior,risk_clearance=clearance,development=summary,rows=rows);trials.append(trial)
        if score>best:
            best=score;model['history']=[dict(iteration=0,development=summary,rows=rows,initialization=True)]
            torch.save(model,args.output/'policy.pth')
        save_json(args.output/'development.json',dict(trials=trials,best_success=best[0],best_capture=best[1],
            selection_bank=args.selection_bank,source_sha256=BASELINE_SHA256,training_steps=0))
        print(dict(phase='predictor_development',horizon=horizon,risk_prior=actor.risk_prior,risk_clearance=clearance,
            success=score[0],capture=score[1],collision=summary['collision'],best=best),flush=True)


def joint_interpolate(args,pool):
    """Development-only trust-region selection between one common initializer and a learned policy."""
    args.output.mkdir(parents=True,exist_ok=True)
    learned=torch.load(args.start,map_location='cpu',weights_only=False)
    initial=torch.load(args.warm,map_location='cpu',weights_only=False)
    if learned['levels']!=initial['levels'] or learned['peer']!=initial['peer']:
        raise ValueError('Interpolation requires the same mechanism and control representation.')
    trials=[];best=(-1.,-1.)
    proposal_prefix=('base.f1.','base.f2.','base.t1.','base.l1.','base.l2.','base.mean_layer.','base.log_std_layer.')
    for pscale in (0.,.5,1.):
        for gscale in (.5,1.):
            model=dict(learned)
            model['actor']={k:initial['actor'][k]+(pscale if k.startswith(proposal_prefix) else gscale)*
                (v-initial['actor'][k]) for k,v in learned['actor'].items()}
            model['interpolation']=dict(proposal_scale=pscale,adapter_scale=gscale,learned=str(args.start),initial=str(args.warm))
            path=args.output/'current.pth';torch.save(model,path)
            rows=list(pool.map(joint_episode,[(s,str(path),path.stat().st_mtime_ns,False,0,0,0,0)
                for s in bank(args.selection_bank,args.selection_per_cell)],chunksize=1))
            summary=summarize(rows);score=(summary['success'],summary['capture'])
            trial=dict(proposal_scale=pscale,adapter_scale=gscale,development=summary,rows=rows);trials.append(trial)
            if score>best:
                best=score;model['selection']=trial;torch.save(model,args.output/'policy.pth')
            save_json(args.output/'development.json',dict(trials=trials,best_success=best[0],best_capture=best[1],
                learned=str(args.start),initial=str(args.warm),selection_bank=args.selection_bank,
                source_sha256=BASELINE_SHA256,new_training_steps=0))
            print(dict(phase='joint_interpolation',proposal_scale=pscale,adapter_scale=gscale,
                success=score[0],capture=score[1],best=best),flush=True)


def joint_value_diagnostic(args,pool):
    """Measure whether the learned conditional value ranks real terminal effects."""
    model=torch.load(args.start,map_location='cpu',weights_only=False)
    values=nn.ModuleList([JointLearningValue(response=model.get('response_critic',False),
        options=model.get('gate_period',1)>1,relational=model.get('relational_critic',False),
        intents=bool(model.get('intent_strength',0.)),probability=model.get('probability_value',False),
        decoded=model.get('decoded_value',False),levels=model.get('levels',3),
        intent_strength=model.get('intent_strength',0.),invariant=model.get('invariant_value',False)) for _ in range(2)])
    values.load_state_dict(model['values']);values.eval();torch.set_grad_enabled(False)
    revision=args.start.stat().st_mtime_ns;groups={};used=0
    tasks=[(s,str(args.start),revision,True,args.roots_per_episode,args.repetitions,400,0)
        for s in bank(args.bank,args.per_cell)]
    for rollout in pool.map(joint_episode,tasks,chunksize=1):
        used+=rollout['simulation_steps']
        for row in rollout['branches']:groups.setdefault(row['root'],[]).append(row)
    prior_actor=None
    if model.get('direct_value_prior',False):
        prior_actor=copy.deepcopy(joint_load(str(args.start),revision))
        prior_actor.load_state_dict(model['value_prior_actor'])
    records=[]
    for root,rows in groups.items():
        assert all(r['before']['terminal'] and r['after']['terminal'] for r in rows)
        first=rows[0];packet=mechanism_tensor(first['packet'])
        if model.get('decoded_value',False):packet.update(joint_load(str(args.start),revision).value_encoding(packet))
        left=torch.tensor(first['left'][None]);right=torch.tensor(first['right'][None])
        if model.get('direct_value_scale',0.):
            if prior_actor is not None:
                _,prior_logits=joint_factor_advantages(prior_actor,packet,[],1.)
                packet['_direct_prior_logits']=prior_logits
            (a,b),_=joint_factor_advantages(joint_load(str(args.start),revision),packet,
                [left,right],model['direct_value_scale'])
            estimated=np.asarray([float(b-a)])
        else:estimated=np.asarray([float(v(packet,right)-v(packet,left)) for v in values])
        observed=np.asarray([r['after']['reward']-r['before']['reward'] for r in rows])
        successes=np.asarray([r['after']['success']-r['before']['success'] for r in rows])
        records.append(dict(root=list(root),defenders=len(first['left']),predicted=float(estimated.mean()),
            ensemble_std=float(estimated.std()),actual=float(observed.mean()),
            actual_std=float(observed.std()),success_effect=float(successes.mean()),
            success_effects=successes.tolist()))
    def statistics(rows):
        q=np.asarray([r['predicted'] for r in rows]);d=np.asarray([r['actual'] for r in rows]);s=np.asarray([r['success_effect'] for r in rows])
        changed=np.abs(s)>0.;strong=np.abs(q)>.05
        return dict(pairs=len(rows),changed_terminal_success=int(changed.sum()),strong_predictions=int(strong.sum()),
            difference_mae=float(np.mean(abs(q-d))),
            return_correlation=float(np.corrcoef(q,d)[0,1]) if q.std()>0 and d.std()>0 else None,
            changed_success_direction_accuracy=float(np.mean(q[changed]*s[changed]>0)) if changed.any() else None,
            strong_prediction_terminal_change_rate=float(np.mean(changed[strong])) if strong.any() else None,
            predicted_abs_mean=float(np.mean(abs(q))),actual_abs_mean=float(np.mean(abs(d))))
    result=dict(model=str(args.start),model_sha256=hashlib.sha256(args.start.read_bytes()).hexdigest(),
        training_diagnostic_bank=args.bank,simulation_steps=used,summary=statistics(records),
        teams={str(n):statistics([r for r in records if r['defenders']==n]) for n in args.teams},records=records)
    save_json(args.output,result);print(json.dumps({k:result[k] for k in ('summary','teams','simulation_steps')}),flush=True)


def joint_train(args,pool):
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(True);torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor=JointLearningActor(not args.without_peer,graph=args.joint_graph,levels=args.joint_levels,
        risk_prior=args.joint_risk_prior,risk_horizon=args.joint_risk_horizon,
        pair_scale=args.joint_pair_scale,gate_period=args.joint_gate_period,
        risk_clearance=args.joint_risk_clearance,forecast_inputs=args.joint_forecast_inputs,
        frozen_proposal=args.joint_frozen_proposal,risk_feedback=args.joint_risk_feedback,
        intent_strength=args.joint_intent_strength,coupled_shortlist=args.joint_coupled_shortlist).to(device)
    actor.base.load_state_dict(torch.load(args.source,map_location=device,weights_only=True))
    values=nn.ModuleList([JointLearningValue(response=args.joint_response_critic,
        options=args.joint_gate_period>1,relational=args.joint_relational_critic,
        intents=bool(args.joint_intent_strength),probability=args.joint_probability_value,
        decoded=args.joint_decoded_value,levels=args.joint_levels,intent_strength=args.joint_intent_strength,
        invariant=args.joint_invariant_value).to(device) for _ in range(2)])
    if args.joint_gate_lr:
        aopt=torch.optim.Adam([dict(params=actor.base.parameters(),lr=args.joint_actor_lr),
            dict(params=[p for name,p in actor.named_parameters() if not name.startswith('base.')],lr=args.joint_gate_lr)])
    else:aopt=torch.optim.Adam(actor.parameters(),lr=args.joint_actor_lr)
    vopt=torch.optim.Adam(values.parameters(),lr=5e-4)
    start=0;history=[];best=(-1.,-1.);total_steps=0;reset_values=False
    replay=torch.load(args.joint_replay,map_location='cpu',weights_only=False) if args.joint_replay else None
    if replay and args.iterations!=1:raise ValueError('A frozen rollout dataset supports one policy-improvement iteration per refit.')
    if args.start or replay:
        old=replay['model'] if replay else torch.load(args.start,map_location=device,weights_only=False)
        if old.get('graph',False)!=actor.graph:raise ValueError('The gate distribution changed; use a common fresh initialization.')
        common_initialization=(old.get('initialization')=='original source neural proposal; no intervention learning'
            and old['iteration']==0 and old['simulation_steps']==0)
        same_period=old.get('gate_period',1)==args.joint_gate_period
        same_intent=old.get('intent_strength',0.)==args.joint_intent_strength
        if not same_intent and not common_initialization:
            raise ValueError('Fusion preference options require a common source initialization.')
        if old.get('risk_clearance',6.)!=args.joint_risk_clearance and not common_initialization:
            raise ValueError('Changing prediction clearance requires a common fresh initialization.')
        if not same_period and not common_initialization:
            raise ValueError('Changing the temporal policy requires a common original initialization.')
        if old.get('pair_scale',1.)!=args.joint_pair_scale and not common_initialization:
            raise ValueError('Changing the pair score scale requires the common untrained initialization.')
        if (old['peer']!=actor.peer or old['counterfactual']!=(not args.joint_no_cf)) and not common_initialization:
            raise ValueError('A mechanism ablation must start from the common original initialization.')
        if old.get('joint_value_guidance',True)!=(not args.joint_no_value_guidance) and not common_initialization:
            raise ValueError('Removing joint value guidance requires the common original initialization.')
        same_forecast=old.get('forecast_inputs',False)==args.joint_forecast_inputs
        same_proposals=old.get('frozen_proposal',False)==args.joint_frozen_proposal
        if old.get('success_objective',False)!=args.joint_success_objective and not common_initialization:
            raise ValueError('Changing the return objective requires a common fresh initialization.')
        if old.get('risk_feedback',False)!=args.joint_risk_feedback and not common_initialization:
            raise ValueError('Changing candidate feedback prediction requires a common initialization.')
        if old.get('coupled_shortlist',0)!=args.joint_coupled_shortlist and not common_initialization:
            raise ValueError('Changing coupled candidate prediction requires a common initialization.')
        if (not same_forecast or not same_proposals) and not common_initialization:
            raise ValueError('New predictor inputs and proposal controls require a common original initialization.')
        if same_forecast:actor.load_state_dict(old['actor'])
        else:
            state=actor.state_dict()
            for key,value in old['actor'].items():
                if state[key].shape==value.shape:state[key]=value
                elif key=='gate_pairs.0.weight':
                    state[key].zero_();state[key][:,:29]=value[:,:29]
                else:raise ValueError(f'Unexpected actor architecture change: {key}')
            actor.load_state_dict(state)
        same_value=(old.get('response_critic',False)==args.joint_response_critic and same_period and same_intent
            and old.get('relational_critic',False)==args.joint_relational_critic
            and old.get('probability_value',False)==args.joint_probability_value
            and old.get('decoded_value',False)==args.joint_decoded_value
            and old.get('invariant_value',False)==args.joint_invariant_value)
        reset_values=(args.joint_decoded_value and not old.get('decoded_value',False)
            or args.joint_invariant_value!=old.get('invariant_value',False))
        if same_value:values.load_state_dict(old['values'])
        elif not common_initialization and not reset_values:raise ValueError('Response features require a common fresh value initialization.')
        if same_forecast and len(aopt.param_groups)==len(old['actor_optimizer']['param_groups']):aopt.load_state_dict(old['actor_optimizer'])
        for group in aopt.param_groups:group['lr']=args.joint_actor_lr
        if args.joint_gate_lr:aopt.param_groups[-1]['lr']=args.joint_gate_lr
        if same_value:vopt.load_state_dict(old['value_optimizer'])
        start=old['iteration'];history=old['history'];total_steps=old['simulation_steps']
        if common_initialization and (old['peer']!=actor.peer or not same_period
                or old.get('coupled_shortlist',0)!=actor.coupled_shortlist):history=[]
        selection_cells=tuple((n,a) for n in (args.selection_teams or args.teams) for a in args.agilities)
        expected={(s,n,a) for s,n,a in bank(args.selection_bank,args.selection_per_cell,selection_cells)}
        if history:
            observed={(r['seed'],r['defenders'],r['agility']) for r in history[0]['rows']}
            if observed!=expected:history=[]
        if history:best=max((h['development']['success'],h['development']['capture']) for h in history)
    if args.warm:
        # Policy iteration may reject an actor update while retaining the
        # critic fitted to its accepted predecessor. The warm checkpoint
        # restores only that actor and optimizer, never its value estimates.
        warm=torch.load(args.warm,map_location=device,weights_only=False)
        for field,current_value in (('peer',actor.peer),('levels',actor.levels),
                ('risk_horizon',actor.risk_horizon),('risk_prior',actor.risk_prior),
                ('risk_clearance',actor.risk_clearance),('forecast_inputs',actor.forecast_inputs),
                ('coupled_shortlist',actor.coupled_shortlist),
                ('risk_feedback',actor.risk_feedback),('frozen_proposal',actor.frozen_proposal)):
            if warm.get(field,0 if field=='coupled_shortlist' else None)!=current_value:raise ValueError(f'Actor warm start differs in {field}.')
        actor.load_state_dict(warm['actor']);aopt.load_state_dict(warm['actor_optimizer'])
        for group in aopt.param_groups:group['lr']=args.joint_actor_lr
        if args.joint_gate_lr:aopt.param_groups[-1]['lr']=args.joint_gate_lr
    current=args.output/'current.pth';selected=args.output/'policy.pth'
    prior_actor_state=copy.deepcopy(actor.state_dict()) if args.joint_direct_prior else None
    def save(path,it):
        torch.save(dict(format='joint-neural-source-ppo-v1',actor={k:v.detach().cpu() for k,v in actor.state_dict().items()},
            values=values.state_dict(),actor_optimizer=aopt.state_dict(),value_optimizer=vopt.state_dict(),
            iteration=it,history=history,peer=actor.peer,graph=actor.graph,levels=actor.levels,
            risk_prior=actor.risk_prior,risk_horizon=actor.risk_horizon,counterfactual=not args.joint_no_cf,
            risk_clearance=actor.risk_clearance,
            response_critic=args.joint_response_critic,
            forecast_inputs=actor.forecast_inputs,frozen_proposal=actor.frozen_proposal,
            risk_feedback=actor.risk_feedback,
            intent_strength=actor.intent_strength,
            coupled_shortlist=actor.coupled_shortlist,
            probability_value=args.joint_probability_value,
            decoded_value=args.joint_decoded_value,
            invariant_value=args.joint_invariant_value,
            direct_value_scale=args.joint_direct_value,
            direct_value_conservative=args.joint_direct_value_conservative,
            direct_value_prior=args.joint_direct_prior,
            value_prior_actor=({key:value.detach().cpu() for key,value in prior_actor_state.items()}
                if prior_actor_state is not None else None),
            relational_critic=args.joint_relational_critic,initial_root=args.joint_initial_root,
            failure_roots=args.joint_failure_roots,
            failure_lookback=args.joint_failure_lookback,
            failure_subjects=args.joint_failure_subjects,
            joint_contexts=args.joint_contexts,
            pair_scale=actor.pair_scale,paired_actor_weight=args.joint_paired_actor,
            proposal_value_weight=args.joint_proposal_value,
            gate_period=actor.gate_period,source_proposal_frozen=actor.gate_period>1 or actor.frozen_proposal,
            terminal_preferences=args.joint_terminal_preferences,
            conditional_improvement=args.joint_conditional_improvement,
            conditional_effect_limit=2. if args.joint_success_objective else .5,
            success_objective=args.joint_success_objective,
            branch_value_data=args.joint_branch_value_data,
            counterfactual_td=args.joint_counterfactual_td,
            branch_grid=args.joint_branch_grid,
            accept_only=args.joint_accept_only,
            fitted_iteration=args.joint_fitted_iteration and not args.joint_no_value_guidance,
            rescue_preferences=args.joint_rescue_preferences,
            empirical_rescues=args.joint_empirical_rescues,
            rescue_cloning=args.joint_rescue_cloning,
            rescue_retention=args.joint_rescue_retention,
            normalized_preferences=args.joint_normalized_preferences,
            greedy_projection=args.joint_greedy_projection,
            continuation_trust=args.joint_continuation_trust,
            joint_value_guidance=not args.joint_no_value_guidance,
            simulation_steps=total_steps,source_sha256=BASELINE_SHA256,source_commit=SOURCE_COMMIT,
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            training_reward=('team terminal +/-1, no time penalty, potential shaping 0.2; gamma 1'
                if args.joint_success_objective else 'team terminal +/-1, step -0.001, potential shaping 0.2; gamma 0.995'),
            execution=(f'unchanged author AdaRes and candidate-interaction fusion at 0.2 s; preference every {actor.gate_period*.2:g} s'
                if actor.intent_strength else f'unchanged author AdaRes at 0.2 s; scalar Adapter decisions every {actor.gate_period*.2:g} s')),path)
    save(current,start)
    if history and (replay or args.start.name=='policy.pth' or args.warm or old.get('accept_only',False)):
        save(selected,start)
    def evaluation(it):
        cells=tuple((n,a) for n in (args.selection_teams or args.teams) for a in args.agilities)
        rows=list(pool.map(joint_episode,[(s,str(current),current.stat().st_mtime_ns,False,0,0,0,0)
            for s in bank(args.selection_bank,args.selection_per_cell,cells)],chunksize=1))
        return dict(iteration=it,development=summarize(rows),rows=rows)
    if not history:
        initial=evaluation(start);history.append(initial);best=(initial['development']['success'],initial['development']['capture'])
        save(current,start);save(selected,start);print(dict(phase='initial_joint',success=best[0],capture=best[1]),flush=True)
    schedule=json.loads(args.matched_schedule.read_text())['history'] if args.matched_schedule else None
    generator=np.random.default_rng(args.seed)
    def plan_values(packet,actions,plans):
        batch,n=actions.shape[:2]
        if plans.ndim==2:plans=plans[None].expand(batch,-1,-1)
        k=plans.shape[1]
        expanded={key:value[:,None].expand(-1,k,*value.shape[1:]).reshape(batch*k,*value.shape[1:]) for key,value in packet.items()}
        controls=actions[:,None].expand(-1,k,-1,-1).clone();controls[...,2]=plans
        return torch.stack([v(expanded,controls.reshape(batch*k,n,3)).reshape(batch,k) for v in values]).mean(0)
    for iteration in range(start,start+args.iterations):
        scenes=bank(args.bank+iteration*100000,args.per_cell);revision=current.stat().st_mtime_ns
        tasks=[(s,str(current),revision,True,0 if args.joint_no_cf else args.roots_per_episode,
            args.repetitions,args.branch_horizon,0) for s in scenes]
        rollouts=replay['rollouts'] if replay else []
        if not replay:
            for k,result in enumerate(pool.map(joint_episode,tasks,chunksize=1),1):
                rollouts.append(result)
                if k%16==0:print(dict(phase='joint_collect',iteration=iteration,episodes=k,total=len(tasks)),flush=True)
        used=sum(r['simulation_steps'] for r in rollouts)
        if schedule:
            budget=next(h['iteration_simulation_steps'] for h in schedule if h['iteration']==iteration+1)
            extra=0
            while used<budget:
                remaining=budget-used;batch=[]
                for j in range(min(args.workers,(remaining+399)//400)):
                    cap=min(400,remaining-j*400)
                    if cap>0:
                        n,a=CELLS[(extra+j)%len(CELLS)]
                        batch.append(((args.bank+iteration*100000+50000+extra+j,n,a),str(current),revision,True,0,0,0,cap))
                additional=list(pool.map(joint_episode,batch,chunksize=1));rollouts.extend(additional)
                used+=sum(r['simulation_steps'] for r in additional);extra+=len(batch)
                if extra%32==0:print(dict(phase='matched_budget',iteration=iteration,
                    simulation_steps=used,budget=budget,additional_episodes=extra),flush=True)
        if args.joint_save_data and not replay:
            torch.save(dict(format='source-joint-policy-improvement-data-v1',
                model=torch.load(current,map_location='cpu',weights_only=False),rollouts=rollouts,
                source_sha256=BASELINE_SHA256,simulation_steps=used),args.output/'training-data.pth')
        if args.joint_decoded_value:
            # Encode each frozen source-policy decoder once, including the
            # continuation states whose original observations were omitted.
            packets={}
            for rollout in rollouts:
                for row in (*rollout['steps'],*rollout['branches'],*rollout.get('value_steps',[])):
                    packets[id(row['packet'])]=row['packet']
            batches={}
            for packet in packets.values():
                batches.setdefault((len(packet['node']),tuple(sorted(packet))),[]).append(packet)
            with torch.no_grad():
                for rows in batches.values():
                    for begin in range(0,len(rows),64):
                        selected_packets=rows[begin:begin+64]
                        encoded=actor.value_encoding(mechanism_stack(selected_packets,device))
                        for key,value in encoded.items():
                            for packet,array in zip(selected_packets,value.cpu().numpy()):packet[key]=array
            print(dict(phase='decoded_value_inputs',states=len(packets)),flush=True)
        if args.joint_direct_prior:
            # The behavior policy is a trust prior, not a measured return.
            # Fit only its learned score correction to counterfactual values.
            prior_actor_state=copy.deepcopy(actor.state_dict())
            with torch.no_grad():
                for rows in batches.values():
                    for begin in range(0,len(rows),64):
                        selected_packets=rows[begin:begin+64]
                        _,logits=joint_factor_advantages(actor,mechanism_stack(selected_packets,device),[],1.)
                        for packet,array in zip(selected_packets,logits.cpu().numpy()):packet['_direct_prior_logits']=array
            print(dict(phase='conditional_value_prior',states=len(packets)),flush=True)
        total_steps+=used
        grouped={n:[] for n in args.teams};contrast={n:[] for n in args.teams};continuations={n:[] for n in args.teams}
        for rollout in rollouts:
            steps=rollout['steps'];n=len(steps[0]['action'])
            with torch.no_grad():
                packet=mechanism_stack([r['packet'] for r in steps],device)
                oldv=torch.stack([v(packet) for v in values]).mean(0).cpu().numpy()
                tail=0. if rollout['tail'] is None else float(torch.stack([v(mechanism_tensor(rollout['tail'],device)) for v in values]).mean())
            gae=0.;mc=tail
            for t in reversed(range(len(steps))):
                row=steps[t];nextv=oldv[t+1] if t+1<len(steps) else tail
                gamma=row.get('discount',.995)
                delta=row['reward']+gamma*nextv-oldv[t]
                gae=delta+gamma*.95*gae;mc=row['reward']+gamma*mc
                row.update(advantage=gae,return_target=mc if args.joint_success_objective else gae+float(oldv[t]),old_value=float(oldv[t]),mc=mc)
            grouped[n].extend(steps);contrast[n].extend(rollout['branches'])
            continuations[n].extend(rollout.get('value_steps',[]))
        def cache_predictions(packet):
            if not actor.frozen_proposal or not (actor.risk_prior or actor.forecast_inputs):return packet
            if '_fusion_risk' in packet:
                packet['_candidate_risk']=packet['_fusion_risk'];return packet
            risks=[]
            with torch.no_grad():
                for begin in range(0,len(packet['node']),64):
                    batch={key:value[begin:begin+64] for key,value in packet.items()}
                    proposals=actor.proposal_distribution(batch['obs'])[0].tanh()
                    risks.append(actor.candidate_risk(batch,proposals))
            packet['_candidate_risk']=torch.cat(risks)
            return packet
        tensors={}
        contrast_groups={}
        for n,rows in grouped.items():
            tensors[n]=dict(packet=cache_predictions(mechanism_stack([r['packet'] for r in rows],device)),
                **{key:torch.tensor(np.asarray([r[key] for r in rows]),device=device,dtype=torch.float32)
                   for key in ('action','latent','logp','advantage','return_target','old_value','mc')})
            groups={}
            for row in contrast[n]:groups.setdefault(row['root'],[]).append(row)
            contrast_groups[n]=list(groups.values())
        print(dict(phase='joint_data',iteration=iteration,simulation_steps=used,transitions=sum(map(len,grouped.values())),
            pairs=sum(map(len,contrast.values())),continuation_values=sum(map(len,continuations.values()))),flush=True)
        continuation_tensors={n:dict(packet=mechanism_stack([r['packet'] for r in rows],device),
            action=torch.tensor(np.stack([r['action'] for r in rows]),device=device),
            mc=torch.tensor([r['mc'] for r in rows],device=device,dtype=torch.float32))
            for n,rows in continuations.items() if rows}
        continuation_policy={}
        if args.joint_continuation_trust:
            with torch.no_grad():
                for n,data_cf in continuation_tensors.items():
                    packet=data_cf['packet']
                    potential=-(packet['motion'][...,:2]-packet['task'][:,:2,None].transpose(1,2)*60.).norm(dim=-1).amin(1)/60.
                    success=data_cf['mc']+.2*potential>0.
                    if not bool(success.any()):continue
                    packet={key:value[success] for key,value in packet.items()}
                    if 'obs' not in packet:
                        obs,_=actor.feedback_observation(packet,packet['motion'][:,None],0.)
                        packet['obs']=obs[:,0];packet['obs'][...,12:14]=packet['node'][...,11:13]
                    index=((data_cf['action'][success,...,2]*(actor.levels-1)).round().long()*
                        actor.levels**torch.arange(n,device=device)).sum(-1)
                    continuation_policy[n]=dict(packet=cache_predictions(packet),index=index)
        def target(entries):
            result=torch.tensor([r['reward'] for r in entries],device=device)
            live=[i for i,r in enumerate(entries) if r['next'] is not None]
            if live:
                pp=mechanism_stack([entries[i]['next'] for i in live],device)
                continuation=torch.stack([v(pp) for v in values]).mean(0)
                discount=torch.tensor([entries[i]['discount'] for i in live],device=device)
                result[live]+=discount*continuation
            return result
        value_updates=args.updates if iteration==0 or reset_values else max(100,args.updates//2)
        if args.joint_direct_value:
            actor_before=copy.deepcopy(actor.state_dict())
            optimizer_before=copy.deepcopy(aopt.state_dict())
        td_values=None
        for update in range(value_updates):
            if args.joint_counterfactual_td and update==value_updates//2:
                td_values=copy.deepcopy(values).requires_grad_(False).eval()
            n=int(generator.choice(args.teams));data=tensors[n]
            ordinary_batch=128 if n in continuation_tensors else 256
            ids=torch.as_tensor(generator.integers(len(grouped[n]),size=ordinary_batch),device=device)
            pp={k:v[ids] for k,v in data['packet'].items()};aa=data['action'][ids]
            if args.joint_direct_value:
                (adv,),logits=joint_factor_advantages(actor,pp,[aa],args.joint_direct_value,reference=aa)
                loss=sum((v(pp)-data['return_target'][ids]).square().mean()+
                    .5*(v(pp)+adv-data['mc'][ids]).square().mean() for v in values)
                index=((aa[...,2]*(actor.levels-1)).round().long()*actor.levels**torch.arange(n,device=device)).sum(-1)
                conservative=(logits.logsumexp(-1)-logits.gather(1,index[:,None]).squeeze(1)).mean()
            else:
                loss=sum((v(pp)-data['return_target'][ids]).square().mean()+.5*(v(pp,aa)-data['mc'][ids]).square().mean() for v in values)
            if n in continuation_tensors:
                data_cf=continuation_tensors[n]
                cf_ids=torch.as_tensor(generator.integers(len(continuations[n]),size=128),device=device)
                cp={key:value[cf_ids] for key,value in data_cf['packet'].items()}
                ca=data_cf['action'][cf_ids];returns=data_cf['mc'][cf_ids]
                if args.joint_direct_value:
                    (adv,),_=joint_factor_advantages(actor,cp,[ca],args.joint_direct_value,reference=ca)
                    loss=.5*loss+.5*sum((v(cp)-returns).square().mean()+.5*(v(cp)+adv-returns).square().mean() for v in values)
                else:
                    loss=.5*loss+.5*sum((v(cp)-returns).square().mean()+.5*(v(cp,ca)-returns).square().mean() for v in values)
            if contrast[n]:
                groups=[contrast_groups[n][i] for i in generator.integers(len(contrast_groups[n]),size=64)]
                rows=[g[0] for g in groups]
                cp=mechanism_stack([r['packet'] for r in rows],device)
                left=torch.tensor(np.stack([r['left'] for r in rows]),device=device)
                right=torch.tensor(np.stack([r['right'] for r in rows]),device=device)
                with torch.no_grad():
                    counts=[len(g) for g in groups]
                    def paired_target(entries):
                        mc=target(entries)
                        if td_values is None:return mc
                        first=[entry['first_step'] for entry in entries]
                        td=torch.tensor([entry['reward'] for entry in first],device=device)
                        live=[i for i,entry in enumerate(first) if entry['next'] is not None]
                        if live:
                            next_packet=mechanism_stack([first[i]['next'] for i in live],device)
                            state_value=torch.stack([v(next_packet) for v in td_values]).mean(0)
                            td[live]+=torch.tensor([first[i]['discount'] for i in live],device=device)*state_value
                        return .25*mc+.75*td
                    before=torch.stack([v.mean() for v in paired_target([r['before'] for g in groups for r in g]).split(counts)])
                    after=torch.stack([v.mean() for v in paired_target([r['after'] for g in groups for r in g]).split(counts)])
                if args.joint_direct_value:
                    reference=torch.tensor(np.stack([r['reference_action'] for r in rows]),device=device)
                    (left_adv,right_adv),_=joint_factor_advantages(actor,cp,[left,right],args.joint_direct_value,reference=reference)
                for value in values:
                    if args.joint_direct_value:
                        state_value=value(cp);qa,qb=state_value+left_adv,state_value+right_adv
                    else:qa,qb=value(cp,left),value(cp,right)
                    loss=loss+.25*((qa-before).square().mean()+(qb-after).square().mean())
                    loss=loss+args.difference_weight*((qa-qb)-(before-after)).square().mean()
            if args.joint_direct_value:
                loss=loss+args.joint_direct_value_conservative*conservative
                aopt.zero_grad(set_to_none=True)
            vopt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(values.parameters(),2.);vopt.step()
            if args.joint_direct_value:
                nn.utils.clip_grad_norm_(actor.parameters(),2.);aopt.step()
        paired_preferences={};preference_counts={}
        if actor.graph and args.joint_paired_actor and not args.joint_no_cf and not args.joint_direct_value:
            # Paired alternatives can have negligible probability under the
            # nominal collision prior. Distill calibrated value preferences
            # directly, rather than waiting for on-policy sampling to find them.
            # Only transfer preferences whose two critics and paired return
            # targets agree in sign; small/uncertain differences carry no loss.
            with torch.no_grad():
                for n,groups in contrast_groups.items():
                    if not groups:continue
                    entries=[]
                    for begin in range(0,len(groups),128):
                        selected_groups=groups[begin:begin+128];rows=[g[0] for g in selected_groups]
                        cp=mechanism_stack([r['packet'] for r in rows],device)
                        left=torch.tensor(np.stack([r['left'] for r in rows]),device=device)
                        right=torch.tensor(np.stack([r['right'] for r in rows]),device=device)
                        differences=torch.stack([v(cp,right)-v(cp,left) for v in values])
                        mean=differences.mean(0);uncertainty=differences.std(0,unbiased=False)
                        counts=[len(g) for g in selected_groups]
                        observed=torch.stack([v.mean() for v in
                            (target([r['after'] for g in selected_groups for r in g])-
                             target([r['before'] for g in selected_groups for r in g])).split(counts)])
                        confidence=(mean.abs()-uncertainty-.05).clamp(0.,1.)
                        confidence*=((mean*observed)>0)&((differences*mean[None])>0).all(0)
                        confidence=torch.minimum(confidence,(observed.abs()-.05).clamp(0.,1.))
                        if args.joint_terminal_preferences:
                            for j,group in enumerate(selected_groups):
                                complete=all(r['before'].get('terminal',0) and r['after'].get('terminal',0) for r in group)
                                if not complete:
                                    confidence[j]=0.;continue
                                changes=np.asarray([r['after']['success']-r['before']['success'] for r in group])
                                effect=float(changes.mean())
                                if not effect or np.any(changes*effect<0.) or float(mean[j])*effect<=0.:
                                    confidence[j]=0.
                                else:confidence[j]*=abs(effect)-.25*float(changes.std())
                        if args.joint_rescue_preferences:
                            reference=torch.tensor(np.stack([r['reference_action'] for r in rows]),device=device)
                            choose_right=observed>0.
                            same_return=observed.abs()<1e-6
                            fewer_changes=(right[...,2]!=reference[...,2]).sum(-1)<(left[...,2]!=reference[...,2]).sum(-1)
                            choose_right=choose_right|(same_return&fewer_changes)
                            winner=torch.where(choose_right[:,None,None],right,left)
                            rescue_values=torch.stack([v(cp,winner)-v(cp,reference) for v in values])
                            rescue_mean=rescue_values.mean(0);rescue_std=rescue_values.std(0,unbiased=False)
                            for j,group in enumerate(selected_groups):
                                key='after' if bool(choose_right[j]) else 'before'
                                improvement=np.asarray([r[key]['success']-r['reference_outcome']['success'] for r in group])
                                if np.any(improvement<0.) or improvement.mean()<=0.:
                                    confidence[j]=0.;continue
                                lower=float(rescue_mean[j]-rescue_std[j])
                                confidence[j]=(float(improvement.mean())**2 if args.joint_empirical_rescues else
                                    max(0.,min(1.,lower-.05))*float(improvement.mean()))
                            for j,row in enumerate(rows):
                                if float(confidence[j])>0.:
                                    entries.append(dict(packet=row['packet'],left=row['reference_action'],
                                        right=winner[j].cpu().numpy(),direction=1.,weight=float(confidence[j]),
                                        key=(*row['root'][:2],*winner[j,:,2].cpu().tolist())))
                            continue
                        for j,row in enumerate(rows):
                            if float(confidence[j])>0.:
                                entries.append(dict(packet=row['packet'],left=row['left'],right=row['right'],
                                    direction=float(mean[j].sign()),weight=float(confidence[j])))
                    if args.joint_rescue_preferences:
                        unique={}
                        for entry in entries:
                            key=entry.pop('key')
                            if key not in unique or entry['weight']>unique[key]['weight']:unique[key]=entry
                        entries=list(unique.values())
                        if args.joint_rescue_retention:
                            # Every measured root contributes a policy target.
                            # Retain its reference when no paired rescue was
                            # observed; otherwise prefer the strongest rescue
                            # with the fewest changed vessel preferences.
                            roots={}
                            for key,entry in unique.items():roots.setdefault(key[:2],[]).append(entry)
                            measured={group[0]['root'][:2]:group[0] for group in contrast_groups[n]}
                            # Successful factual episodes anchor the unchanged
                            # behavior even when counterfactual rollout effort
                            # is concentrated on episodes that actually failed.
                            for rollout in rollouts:
                                first=rollout['steps'][0]
                                if len(first['action'])==n and rollout['row']['success']:
                                    key=(rollout['row']['seed'],int(round(float(first['packet']['node'][0,15])*400)))
                                    measured.setdefault(key,dict(packet=first['packet'],reference_action=first['action']))
                            entries=[]
                            for key,row in measured.items():
                                choices=roots.get(key,[])
                                if not choices:
                                    entries.append(dict(packet=row['packet'],left=row['reference_action'],
                                        right=row['reference_action'],direction=1.,weight=1.))
                                    continue
                                weight=max(entry['weight'] for entry in choices)
                                choices=[entry for entry in choices if entry['weight']==weight]
                                changed=min(int(np.count_nonzero(entry['left'][:,2]!=entry['right'][:,2])) for entry in choices)
                                entries.extend(entry for entry in choices
                                    if np.count_nonzero(entry['left'][:,2]!=entry['right'][:,2])==changed)
                            retained=[entry for entry in entries if np.array_equal(entry['left'],entry['right'])]
                            rescued=[entry for entry in entries if not np.array_equal(entry['left'],entry['right'])]
                            if retained and rescued:
                                retention_weight=sum(entry['weight'] for entry in rescued)/len(retained)
                                for entry in retained:entry['weight']=retention_weight
                    target_masks=None
                    if entries and args.joint_greedy_projection:
                        grouped_targets={}
                        for entry in entries:grouped_targets.setdefault(id(entry['packet']),[]).append(entry)
                        representatives=[];target_masks=[];rescued=[]
                        powers=actor.levels**np.arange(n)
                        for group in grouped_targets.values():
                            entry=dict(group[0]);entry['weight']=max(r['weight'] for r in group)
                            mask=np.zeros(actor.levels**n,dtype=bool)
                            for r in group:
                                winner=r['right'] if r['direction']>0. else r['left']
                                mask[int((np.rint(winner[:,2]*(actor.levels-1)).astype(int)*powers).sum())]=True
                            reference_id=int((np.rint(entry['left'][:,2]*(actor.levels-1)).astype(int)*powers).sum())
                            representatives.append(entry);target_masks.append(mask);rescued.append(not mask[reference_id])
                        positive_weight=sum(r['weight'] for r,positive in zip(representatives,rescued) if positive)
                        retained=len(rescued)-sum(rescued)
                        if positive_weight and retained:
                            for entry,positive in zip(representatives,rescued):
                                if not positive:entry['weight']=positive_weight/retained
                        entries=representatives
                    preference_counts[n]=len(entries)
                    if entries:
                        paired_preferences[n]=dict(packet=cache_predictions(mechanism_stack([r['packet'] for r in entries],device)),
                            **{key:torch.tensor(np.asarray([r[key] for r in entries]),device=device,dtype=torch.float32)
                               for key in ('left','right','direction','weight')})
                        if target_masks is not None:
                            paired_preferences[n]['target_mask']=torch.tensor(np.asarray(target_masks),device=device)
            print(dict(phase='paired_value_preferences',iteration=iteration,counts=preference_counts),flush=True)
        # COMA-style conditional baseline: hold every other sampled coefficient
        # and all proposals fixed; integrate only this vessel's gate distribution.
        for n,data in tensors.items():
            if args.joint_direct_value:continue
            ga=data['advantage'].clone()
            if args.joint_fitted_iteration and not args.joint_no_value_guidance:
                priors=[]
                with torch.no_grad():
                    for begin in range(0,len(ga),64):
                        pp={k:v[begin:begin+64] for k,v in data['packet'].items()}
                        priors.append(actor(pp,latent=data['latent'][begin:begin+64])[4].softmax(-1))
                data['policy_prior']=torch.cat(priors)
            if actor.graph and args.joint_conditional_improvement and not args.joint_no_value_guidance:
                targets=[];weights=[];indices=[]
                with torch.no_grad():
                    for begin in range(0,len(ga),64):
                        pp={k:v[begin:begin+64] for k,v in data['packet'].items()};aa=data['action'][begin:begin+64]
                        _,_,_,_,logits,plans=actor(pp,latent=data['latent'][begin:begin+64])
                        alternatives=joint_conditional_indices(aa[...,2],actor.levels)
                        batch=len(aa);count=n*actor.levels
                        joint=plans[alternatives.flatten(1)]
                        expanded={key:value[:,None].expand(-1,count,*value.shape[1:]).reshape(batch*count,*value.shape[1:]) for key,value in pp.items()}
                        controls=aa[:,None].expand(-1,count,-1,-1).clone();controls[...,2]=joint
                        estimates=torch.stack([value(expanded,controls.reshape(batch*count,n,3)).reshape(batch,n,actor.levels) for value in values])
                        observed=(aa[...,2]*(actor.levels-1)).round().long()
                        effects=estimates-estimates.gather(-1,observed[None,:,:,None].expand(len(values),-1,-1,-1))
                        mean=effects.mean(0);uncertainty=effects.std(0,unbiased=False)
                        # Undiscounted terminal returns are +/-1, so a genuine
                        # rescue can have a conditional effect of two. Capping
                        # it at 0.5 made even a perfect estimate too weak to
                        # overcome the neutral preference's initial logit gap.
                        limit=2. if args.joint_success_objective else .5
                        effect=mean.sign()*(mean.abs()-uncertainty).clamp(0.,limit)
                        old_conditional=logits.gather(1,alternatives.flatten(1)).reshape(batch,n,actor.levels).log_softmax(-1)
                        targets.append((old_conditional+effect/.2).softmax(-1))
                        weights.append((effect.amax(-1)-effect.amin(-1)).clamp(0.,1.))
                        indices.append(alternatives)
                data['conditional_target']=torch.cat(targets)
                data['conditional_weight']=torch.cat(weights)
                data['conditional_indices']=torch.cat(indices)
                # Ordinary factual policy gradients remain the shared control;
                # conditional value learning has its own explicit update below.
                gate_adv=ga[:,None].expand(-1,n)
            elif actor.graph and not args.joint_no_cf:
                baseline=[]
                with torch.no_grad():
                    for begin in range(0,len(ga),64):
                        pp={k:v[begin:begin+64] for k,v in data['packet'].items()};aa=data['action'][begin:begin+64]
                        _,_,_,_,logits,plans=actor(pp,latent=data['latent'][begin:begin+64])
                        if len(plans)>64:
                            sampled=torch.distributions.Categorical(logits=logits).sample((16,)).T
                            baseline.append(plan_values(pp,aa,plans[sampled]).mean(-1))
                        else:baseline.append((logits.softmax(-1)*plan_values(pp,aa,plans)).sum(-1))
                gate_adv=(data['return_target']-torch.cat(baseline))[:,None].expand(-1,n)
            elif not args.joint_no_cf:
                nodes,weights=np.polynomial.hermite.hermgauss(5);weights=weights/np.sqrt(np.pi)
                baseline=[]
                with torch.no_grad():
                    for begin in range(0,len(ga),256):
                        pp={k:v[begin:begin+256] for k,v in data['packet'].items()};aa=data['action'][begin:begin+256]
                        _,_,_,_,mean,std=actor(pp,latent=data['latent'][begin:begin+256])
                        each=[]
                        for i in range(n):
                            marginal=torch.zeros(len(aa),device=device)
                            for z,w in zip(nodes,weights):
                                alt=aa.clone();alt[:,i,2]=torch.sigmoid(mean[:,i]+np.sqrt(2.)*float(z)*std[:,i])
                                marginal+=float(w)*torch.stack([v(pp,alt) for v in values]).mean(0)
                            each.append(marginal)
                        baseline.append(torch.stack(each,-1))
                gate_adv=data['return_target'][:,None]-torch.cat(baseline)
            else:gate_adv=ga[:,None].expand(-1,n)
            pa=(ga-ga.mean())/ga.std().clamp_min(.1)
            gate_adv=(gate_adv-gate_adv.mean())/gate_adv.std().clamp_min(.1)
            data['policy_advantage']=torch.stack((pa[:,None].expand(-1,n),gate_adv),-1)
            if args.joint_proposal_value:
                with torch.no_grad():data['proposal_reference']=actor.proposal_distribution(data['packet']['obs'])[0].detach()
        values.requires_grad_(False)
        retain_actor=args.joint_accept_only or bool(args.joint_projection_scales)
        if not args.joint_direct_value:
            actor_before=copy.deepcopy(actor.state_dict()) if retain_actor else None
            optimizer_before=copy.deepcopy(aopt.state_dict()) if retain_actor else None
        policy_info=[];policy_updates_by_team={n:0 for n in args.teams}
        matched=next((h for h in schedule if h['iteration']==iteration+1),None) if schedule else None
        matched_counts=matched.get('policy_minibatches_by_team') if matched else None
        policy_epochs=matched.get('policy_epochs',args.joint_epochs) if matched else args.joint_epochs
        if args.joint_direct_value:policy_epochs=0
        epoch=-1
        for epoch in range(policy_epochs):
            for n,data in tensors.items():
                if matched_counts:
                    count=int(matched_counts[str(n)])
                    indices=generator.integers(len(grouped[n]),size=count*256)
                else:indices=generator.permutation(len(grouped[n]))
                for begin in range(0,len(indices),256):
                    ids=torch.as_tensor(indices[begin:begin+256],device=device)
                    pp={k:v[ids] for k,v in data['packet'].items()}
                    actions,_,lp,entropy,logits,plans=actor(pp,latent=data['latent'][ids])
                    logratio=(lp-data['logp'][ids]).clamp(-10.,10.);ratio=logratio.exp()
                    adv=data['policy_advantage'][ids]
                    objective=torch.minimum(ratio*adv,ratio.clamp(.85,1.15)*adv).mean()
                    actor_loss=-objective-.001*entropy.mean()
                    if 'policy_prior' in data:
                        prior=data['policy_prior'][ids]
                        # Deterministic fitted policy iteration uses a policy
                        # projection with a trust penalty, not a policy-gradient
                        # claim for actions that were not sampled stochastically.
                        actor_loss=(prior*(prior.clamp_min(1e-12).log()-logits.log_softmax(-1))).sum(-1).mean()
                    if actor.graph and not args.joint_no_value_guidance and not args.joint_conditional_improvement:
                        with torch.no_grad():
                            sampled=(torch.distributions.Categorical(logits=logits[:64]).sample((16,)).T
                                if len(plans)>64 else None)
                            q=plan_values({k:v[:64] for k,v in pp.items()},actions[:64],plans if sampled is None else plans[sampled])
                            q=q-q.mean(-1,keepdim=True)
                        if sampled is None:actor_loss=actor_loss-.5*(logits[:64].softmax(-1)*q).sum(-1).mean()
                        else:actor_loss=actor_loss-.5*(logits[:64].log_softmax(-1).gather(1,sampled)*q).mean()
                    if 'conditional_target' in data:
                        alternatives=data['conditional_indices'][ids]
                        local=logits.gather(1,alternatives.flatten(1)).reshape(-1,n,actor.levels).log_softmax(-1)
                        target=data['conditional_target'][ids]
                        divergence=(target*(target.clamp_min(1e-12).log()-local)).sum(-1)
                        actor_loss=actor_loss+args.joint_conditional_improvement*(data['conditional_weight'][ids]*divergence).mean()
                    if n in paired_preferences:
                        preference=paired_preferences[n]
                        pids=torch.as_tensor(generator.integers(len(preference['weight']),size=32),device=device)
                        cp={key:value[pids] for key,value in preference['packet'].items()}
                        left,right=preference['left'][pids],preference['right'][pids]
                        fixed=torch.cat((left[...,:2].clamp(-.999999,.999999).atanh(),left[...,2:]),-1)
                        plogits=actor(cp,latent=fixed)[4]
                        powers=actor.levels**torch.arange(n,device=device)
                        lid=((left[...,2]*(actor.levels-1)).round().long()*powers).sum(-1)
                        rid=((right[...,2]*(actor.levels-1)).round().long()*powers).sum(-1)
                        delta=plogits.gather(1,rid[:,None]).squeeze(1)-plogits.gather(1,lid[:,None]).squeeze(1)
                        if args.joint_greedy_projection:
                            # Match the deployed argmax. Spreading probability
                            # across several successful actions can leave the
                            # unsuccessful reference as the most likely action.
                            mask=preference['target_mask'][pids]
                            successful=plogits.masked_fill(~mask,-1e9).max(-1).values
                            remaining=plogits.masked_fill(mask,-1e9).max(-1).values
                            per_preference=(.5+remaining-successful).clamp_min(0.)
                        elif args.joint_rescue_cloning:
                            # Project the entire successful joint intervention.
                            # A relative win over the old action alone can also
                            # promote unobserved combinations involving other boats.
                            winner=torch.where(preference['direction'][pids]>0.,rid,lid)
                            per_preference=-plogits.log_softmax(-1).gather(1,winner[:,None]).squeeze(1)
                        else:
                            per_preference=torch.nn.functional.softplus(-preference['direction'][pids]*delta)
                        weights=preference['weight'][pids]
                        preference_loss=((weights*per_preference).sum()/weights.sum().clamp_min(1e-6)
                            if args.joint_normalized_preferences else (weights*per_preference).mean())
                        actor_loss=actor_loss+args.joint_paired_actor*preference_loss
                    if n in continuation_policy:
                        # The measured rescue assumes the old continuation
                        # policy after the intervention ends. Preserve its
                        # successful continuation decisions while learning
                        # where a new intervention should begin.
                        support=continuation_policy[n]
                        cids=torch.as_tensor(generator.integers(len(support['index']),size=64),device=device)
                        cp={key:value[cids] for key,value in support['packet'].items()}
                        cl=actor(cp,deterministic=True)[4];target=support['index'][cids]
                        chosen=cl.gather(1,target[:,None]).squeeze(1)
                        alternate=cl.scatter(1,target[:,None],-1e9).max(-1).values
                        actor_loss=actor_loss+args.joint_continuation_trust*(.25+alternate-chosen).clamp_min(0.).mean()
                    if args.joint_proposal_value and not args.joint_no_value_guidance:
                        # Train the original proposal from a single-vessel
                        # conditional intervention, keeping all other sampled
                        # commands fixed. Activating this candidate supplies a
                        # gradient even when the observed gate preferred Boids.
                        cp={key:value[:16] for key,value in pp.items()}
                        pm=actor.proposal_distribution(cp['obs'])[0]
                        probe=data['action'][ids][:16].clone();batch=len(probe)
                        subjects=torch.as_tensor(generator.integers(n,size=batch),device=device)
                        batch_ids=torch.arange(batch,device=device)
                        probe[batch_ids,subjects,:2]=pm[batch_ids,subjects].tanh()
                        probe[batch_ids,subjects,2]=1.
                        conditional=torch.stack([value(cp,probe) for value in values]).amin(0)
                        reference=data['proposal_reference'][ids][:16]
                        trust=(pm[batch_ids,subjects]-reference[batch_ids,subjects]).square().mean()
                        actor_loss=actor_loss+args.joint_proposal_value*(-conditional.mean()+.05*trust)
                    aopt.zero_grad(set_to_none=True);actor_loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5);aopt.step()
                    policy_updates_by_team[n]+=1
                    policy_info.append(float((ratio-1.-logratio).mean().detach()))
            fitted=args.joint_fitted_iteration and not args.joint_no_value_guidance
            if not fitted and not matched_counts and epoch>=1 and np.mean(policy_info[-max(1,len(indices)//256):])>.025:break
        values.requires_grad_(True)
        save(current,iteration+1);result=evaluation(iteration+1)
        if args.joint_projection_scales:
            proposal=copy.deepcopy(actor.state_dict());chosen=proposal
            trials=[dict(result,projection_scale=1.)]
            for scale in args.joint_projection_scales:
                if not 0.<scale<1.:raise ValueError('Projection scales must be between zero and one.')
                actor.load_state_dict({key:actor_before[key]+scale*(value-actor_before[key]) for key,value in proposal.items()})
                save(current,iteration+1);trial=evaluation(iteration+1);trial['projection_scale']=scale;trials.append(trial)
                if (trial['development']['success'],trial['development']['capture'])>(result['development']['success'],result['development']['capture']):
                    result=trial;chosen=copy.deepcopy(actor.state_dict())
                print(dict(phase='policy_projection',iteration=iteration+1,scale=scale,
                    success=trial['development']['success'],capture=trial['development']['capture']),flush=True)
            actor.load_state_dict(chosen)
            if result.get('projection_scale',1.)<1.:aopt.load_state_dict(optimizer_before)
            result=dict(result,projection_trials=trials)
        result.update(iteration_simulation_steps=used,simulation_steps=total_steps,
            new_simulation_steps=0 if replay else used,reused_simulation_steps=used if replay else 0,
            collection=summarize([r['row'] for r in rollouts if r['steps'][-1]['done']]),
            transitions=sum(map(len,grouped.values())),interventions=sum(map(len,contrast.values())),
            continuation_value_samples=sum(map(len,continuations.values())),
            continuation_policy_anchors={n:len(data['index']) for n,data in continuation_policy.items()},
            transitions_by_team={n:len(rows) for n,rows in grouped.items()},
            policy_epochs=epoch+1,policy_updates_by_team=policy_updates_by_team,
            policy_minibatches_by_team={n:policy_updates_by_team[n]//max(1,epoch+1) for n in args.teams},
            critic_optimizer_steps=value_updates,
            direct_value_policy_updates=value_updates if args.joint_direct_value else 0,
            paired_value_preferences=preference_counts,
            paired_value_weights={n:dict(mean=float(p['weight'].mean()),total=float(p['weight'].sum()))
                for n,p in paired_preferences.items()},
            mean_policy_kl=float(np.mean(policy_info)) if policy_info else None,value_loss=float(loss.detach()))
        score=(result['development']['success'],result['development']['capture'])
        result['accepted']=not args.joint_accept_only or score>=best
        if not result['accepted']:
            actor.load_state_dict(actor_before);aopt.load_state_dict(optimizer_before)
        history.append(result)
        save(current,iteration+1)
        if score>best:best=score;save(selected,iteration+1)
        save_json(args.output/'development.json',dict(history=history,best_success=best[0],best_capture=best[1],
            peer=actor.peer,counterfactual=not args.joint_no_cf,source_sha256=BASELINE_SHA256,
            development_bank=args.selection_bank))
        print(dict(phase='joint_development',iteration=iteration+1,success=score[0],capture=score[1],
            collision=result['development']['collision'],best=best,simulation_steps=total_steps),flush=True)


def main():
    global CELLS
    torch.set_num_threads(1)
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('joint-value-diagnostic','joint-interpolate','joint-risk-search','joint-train','joint-evaluate','stationary-reference','mechanism-train','mechanism-refit','mechanism-evaluate','search','proposal-search','trajectory-search','trajectory-fit','trajectory-observations','trajectory-encode','fit','refine','augment','evaluate'))
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--start',type=Path)
    p.add_argument('--references',type=Path,nargs='+')
    p.add_argument('--data',type=Path)
    p.add_argument('--observations',type=Path)
    p.add_argument('--critic',type=Path)
    p.add_argument('--loss-mode',choices=('conditional','absolute'),default='conditional')
    p.add_argument('--difference-weight',type=float,default=1.)
    p.add_argument('--critic-only',action='store_true')
    p.add_argument('--critic-architecture',choices=('polynomial','centered','additive'),default='polynomial')
    p.add_argument('--actor-update',choices=('joint','conditional'),default='joint')
    p.add_argument('--without-peer',action='store_true')
    p.add_argument('--teams',type=int,nargs='+',default=(3,6))
    p.add_argument('--agilities',type=float,nargs='+',default=(2.,2.5))
    p.add_argument('--bank',type=int,default=4112000000)
    p.add_argument('--per-cell',type=int,default=8)
    p.add_argument('--iterations',type=int,default=32)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--repetitions',type=int,default=4)
    p.add_argument('--updates',type=int,default=1000)
    p.add_argument('--decision-steps',type=int)
    p.add_argument('--angular-features',action='store_true')
    p.add_argument('--node-value',action='store_true')
    p.add_argument('--feedback-response',action='store_true')
    p.add_argument('--anchored-readout',action='store_true')
    p.add_argument('--outcome-value',action='store_true')
    p.add_argument('--value-architecture',choices=('mlp','linear'),default='mlp')
    p.add_argument('--bootstrap',action='store_true')
    p.add_argument('--factorial',action='store_true')
    p.add_argument('--pair-readout',action='store_true')
    p.add_argument('--shared-data',action='store_true')
    p.add_argument('--collect-only',action='store_true')
    p.add_argument('--out-of-bag-select',action='store_true')
    p.add_argument('--cell-floor',type=float,default=0.)
    p.add_argument('--risk-budget-search',action='store_true')
    p.add_argument('--mirror-canonical',action='store_true')
    p.add_argument('--joint-response-features',action='store_true')
    p.add_argument('--selection-bank',type=int,default=4114000000)
    p.add_argument('--selection-per-cell',type=int,default=16)
    p.add_argument('--selection-teams',type=int,nargs='+')
    p.add_argument('--branch-horizon',type=int,default=64)
    p.add_argument('--roots-per-episode',type=int,default=1)
    p.add_argument('--trust-weight',type=float,default=2.)
    p.add_argument('--warm',type=Path)
    p.add_argument('--conditional-actor',action='store_true')
    p.add_argument('--reward-critic',action='store_true')
    p.add_argument('--coupled-adapter',action='store_true')
    p.add_argument('--joint-no-cf',action='store_true')
    p.add_argument('--joint-graph',action='store_true')
    p.add_argument('--joint-levels',type=int,default=2)
    p.add_argument('--joint-risk-prior',type=float,default=0.)
    p.add_argument('--joint-risk-horizon',type=float,default=1.6)
    p.add_argument('--joint-risk-horizons',type=float,nargs='+')
    p.add_argument('--joint-risk-scales',type=float,nargs='+')
    p.add_argument('--joint-risk-clearance',type=float,default=6.)
    p.add_argument('--joint-risk-clearances',type=float,nargs='+')
    p.add_argument('--joint-actor-lr',type=float,default=3e-5)
    p.add_argument('--joint-gate-lr',type=float)
    p.add_argument('--joint-response-critic',action='store_true')
    p.add_argument('--joint-forecast-inputs',action='store_true')
    p.add_argument('--joint-frozen-proposal',action='store_true')
    p.add_argument('--joint-relational-critic',action='store_true')
    p.add_argument('--joint-initial-root',action='store_true')
    p.add_argument('--joint-failure-roots',action='store_true')
    p.add_argument('--joint-failure-lookback',type=float,default=0.)
    p.add_argument('--joint-failure-subjects',action='store_true')
    p.add_argument('--joint-contexts',type=int,default=0)
    p.add_argument('--joint-risk-feedback',action='store_true')
    p.add_argument('--joint-coupled-shortlist',type=int,default=0)
    p.add_argument('--joint-pair-scale',type=float,default=1.)
    p.add_argument('--joint-gate-period',type=int,default=1)
    p.add_argument('--joint-paired-actor',type=float,default=0.)
    p.add_argument('--joint-proposal-value',type=float,default=0.)
    p.add_argument('--joint-terminal-preferences',action='store_true')
    p.add_argument('--joint-conditional-improvement',type=float,default=0.)
    p.add_argument('--joint-success-objective',action='store_true')
    p.add_argument('--joint-branch-value-data',action='store_true')
    p.add_argument('--joint-counterfactual-td',action='store_true')
    p.add_argument('--joint-branch-grid',action='store_true')
    p.add_argument('--joint-accept-only',action='store_true')
    p.add_argument('--joint-intent-strength',type=float,default=0.)
    p.add_argument('--joint-probability-value',action='store_true')
    p.add_argument('--joint-decoded-value',action='store_true')
    p.add_argument('--joint-invariant-value',action='store_true')
    p.add_argument('--joint-direct-value',type=float,default=0.)
    p.add_argument('--joint-direct-value-conservative',type=float,default=.02)
    p.add_argument('--joint-direct-prior',action='store_true')
    p.add_argument('--joint-fitted-iteration',action='store_true')
    p.add_argument('--joint-rescue-preferences',action='store_true')
    p.add_argument('--joint-empirical-rescues',action='store_true')
    p.add_argument('--joint-rescue-cloning',action='store_true')
    p.add_argument('--joint-rescue-retention',action='store_true')
    p.add_argument('--joint-normalized-preferences',action='store_true')
    p.add_argument('--joint-greedy-projection',action='store_true')
    p.add_argument('--joint-continuation-trust',type=float,default=0.)
    p.add_argument('--joint-save-data',action='store_true')
    p.add_argument('--joint-replay',type=Path)
    p.add_argument('--joint-projection-scales',type=float,nargs='+')
    p.add_argument('--joint-no-value-guidance',action='store_true')
    p.add_argument('--joint-epochs',type=int,default=6)
    p.add_argument('--matched-schedule',type=Path)
    args=p.parse_args()
    if args.joint_no_value_guidance:args.joint_no_cf=True
    if args.joint_branch_value_data and args.branch_horizon<400:
        raise ValueError('Branch continuation value training requires terminal rollouts (horizon 400).')
    if args.joint_counterfactual_td and not args.joint_branch_value_data:
        raise ValueError('Conditional TD learning requires continuation value data.')
    if args.joint_branch_grid and not args.joint_graph:
        raise ValueError('Factorial gate interventions require a discrete joint gate policy.')
    if args.joint_intent_strength and (not args.joint_graph or args.joint_response_critic or args.joint_gate_period<2):
        raise ValueError('Fusion preferences require graph gates, option values without constant-thrust response, and a multi-step preference period.')
    if args.joint_probability_value and not args.joint_success_objective:
        raise ValueError('Bounded success-probability values require the undiscounted success objective.')
    if args.joint_direct_value and not (args.joint_decoded_value and args.joint_success_objective and args.joint_levels==3):
        raise ValueError('Shared conditional value factors require decoded, three-level success-return preferences.')
    if args.joint_direct_prior and not args.joint_direct_value:
        raise ValueError('A fixed behavior prior requires shared conditional value factors.')
    if args.joint_coupled_shortlist and args.mode=='joint-train':
        if not args.joint_intent_strength:
            raise ValueError('Training with coupled candidate prediction requires differentiable preference scores.')
        if args.joint_decoded_value and not args.joint_direct_value:
            raise ValueError('The independent physical decoder cannot supervise a coupled-plan action value; use shared conditional factors.')
    if args.joint_decoded_value and not (args.joint_intent_strength and args.joint_frozen_proposal):
        raise ValueError('Decoded option values require frozen source proposals and fusion preferences.')
    if args.joint_fitted_iteration and not args.joint_graph:
        raise ValueError('Fitted conditional policy iteration requires a joint discrete policy.')
    if args.joint_contexts and not args.joint_graph:
        raise ValueError('Joint intervention contexts require a discrete joint gate policy.')
    if args.joint_greedy_projection and not (args.joint_rescue_preferences and args.joint_fitted_iteration):
        raise ValueError('Greedy successful-set projection requires fitted rescue policy improvement.')
    if args.joint_continuation_trust and not (args.joint_branch_value_data and args.joint_success_objective and args.joint_frozen_proposal):
        raise ValueError('Continuation consistency requires terminal branch data, undiscounted returns, and frozen proposals.')
    if args.joint_rescue_retention and not (args.joint_rescue_preferences and args.joint_rescue_cloning):
        raise ValueError('Measured-root retention requires full joint rescue projection.')
    if args.joint_gate_period>1 and args.joint_proposal_value:
        raise ValueError('Temporal gate variants freeze the original neural proposal.')
    CELLS=tuple((n,a) for n in args.teams for a in args.agilities)
    if args.start and args.mode not in ('joint-value-diagnostic','joint-interpolate','joint-train','joint-evaluate','mechanism-train','mechanism-refit','mechanism-evaluate'):
        args.without_peer=args.without_peer or json.loads(args.start.read_text()).get('without_peer',False)
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             initargs=(str(args.source),args.without_peer)) as pool:
        if args.mode=='joint-risk-search': joint_risk_search(args,pool)
        elif args.mode=='joint-value-diagnostic': joint_value_diagnostic(args,pool)
        elif args.mode=='joint-interpolate': joint_interpolate(args,pool)
        elif args.mode=='joint-train': joint_train(args,pool)
        elif args.mode=='joint-evaluate':
            scenes=bank(args.bank,args.per_cell);revision=args.start.stat().st_mtime_ns
            rows=list(pool.map(joint_episode,[(s,str(args.start),revision,False,0,0,0,0) for s in scenes],chunksize=1))
            baseline=evaluate(pool,scenes,None)
            result=dict(model=str(args.start),bank=args.bank,source_sha256=BASELINE_SHA256,
                baseline=summarize(baseline),full=summarize(rows),paired=paired_summary(rows,baseline),
                baseline_rows=baseline,full_rows=rows)
            save_json(args.output,result);print(json.dumps({k:result[k] for k in ('baseline','full','paired')}),flush=True)
        elif args.mode=='mechanism-train': mechanism_train(args,pool)
        elif args.mode=='mechanism-refit': mechanism_refit(args,pool)
        elif args.mode=='stationary-reference':
            rows=list(pool.map(stationary_episode,bank(args.bank,args.per_cell),chunksize=1))
            result=dict(method='constant zero physical thrust; diagnostic reference, not ARBoids',
                bank=args.bank,source_sha256=BASELINE_SHA256,summary=summarize(rows),rows=rows)
            save_json(args.output,result);print(json.dumps(result['summary'],indent=2),flush=True)
        elif args.mode=='mechanism-evaluate':
            if args.start is None:raise ValueError('A frozen mechanism checkpoint is required.')
            scenes=bank(args.bank,args.per_cell);revision=args.start.stat().st_mtime_ns
            rows=list(pool.map(mechanism_episode,[(scene,str(args.start),revision,False,0.,0,0) for scene in scenes],chunksize=1))
            original=evaluate(pool,scenes,None)
            result=dict(bank=args.bank,model=str(args.start),model_sha256=hashlib.sha256(args.start.read_bytes()).hexdigest(),
                source_sha256=BASELINE_SHA256,baseline=summarize(original),full=summarize(rows),
                paired=paired_summary(rows,original),baseline_rows=original,full_rows=rows)
            save_json(args.output,result)
            print(json.dumps({k:result[k] for k in ('baseline','full','paired')},indent=2),flush=True)
        elif args.mode=='search': search(args,pool)
        elif args.mode=='trajectory-search': trajectory_search(args,pool)
        elif args.mode=='proposal-search': proposal_search(args,pool)
        elif args.mode=='trajectory-fit': trajectory_fit(args,pool)
        elif args.mode=='trajectory-observations': collect_trajectory_observations(args,pool)
        elif args.mode=='trajectory-encode': reencode_trajectory_data(args,pool)
        elif args.mode=='fit': fit(args,pool)
        elif args.mode=='refine': refine(args,pool)
        elif args.mode=='augment': augment(args,pool)
        else:
            selected=json.loads(args.start.read_text())
            references=[json.loads(path.read_text()) for path in args.references] if args.references else []
            if references:
                baseline=[row for ref in references for row in ref['baseline_rows']]
                scenes=[(row['seed'],row['defenders'],row['agility']) for row in baseline]
                CELLS=tuple(sorted(set((row['defenders'],row['agility']) for row in baseline)))
            else:
                scenes=bank(args.bank,args.per_cell)
                baseline=evaluate(pool,scenes,None)
            full=evaluate(pool,scenes,controller_spec(selected))
            result=dict(bank=args.bank,policy=str(args.start),source=str(args.source),
                source_commit=SOURCE_COMMIT,source_sha256=BASELINE_SHA256,
                weights=policy_weights(selected).tolist(),features=FEATURES,
                refiner_weights=selected.get('refiner_weights'),
                controller=selected.get('controller'),proposal=selected.get('proposal'),settings=selected.get('settings'),
                baseline=summarize(baseline),full=summarize(full),paired=paired_summary(full,baseline),
                baseline_rows=baseline,full_rows=full)
            if references:
                result['references']=[str(path) for path in args.references]
                result['method']={key:selected.get(key) for key in ('without_peer','loss_mode','actor_update')}
                result['paired_reference_minus_candidate']=paired_summary(
                    [row for ref in references for row in ref['full_rows']],full)
                result['cohorts']={}
                offset=0
                for path,ref in zip(args.references,references):
                    current=full[offset:offset+len(ref['full_rows'])]
                    result['cohorts'][path.stem]=dict(
                        episodes=len(current),reference_success=ref['full']['success'],
                        candidate_success=float(np.mean([row['success'] for row in current])),
                        reference_minus_candidate=paired_summary(ref['full_rows'],current))
                    offset+=len(current)
                result['bank']=[ref['bank'] for ref in references]
            save_json(args.output,result)
            print(json.dumps({k:result[k] for k in
                (('full','paired_reference_minus_candidate','cohorts') if references else ('baseline','full','paired'))},indent=2),flush=True)


if __name__=='__main__': main()
