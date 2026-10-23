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
from torch.nn import functional as F

from source_arboids import SourcePolicy, numerical_source, SOURCE_COMMIT

SOURCE = Path(r'D:\ARBoids\train\experiments\main-seed42-20260921\adares1.pth')
BASELINE_SHA256 = 'fe0c34101379582eac83ec6e89b4090d754308b604a97babf777f5b01d6c6ca2'
REFERENCE_BASELINE_SHA256 = BASELINE_SHA256
REFERENCE_SOURCE = SOURCE
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
EVALUATION_ABLATION = (False, False, False)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def initialize(source, without_peer=False, evaluation_ablation=(False,False,False)):
    global BASE, PEER_CANDIDATES, EVALUATION_ABLATION, SOURCE, BASELINE_SHA256
    PEER_CANDIDATES = not without_peer
    EVALUATION_ABLATION = tuple(evaluation_ablation)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    source=Path(source);identity=hashlib.sha256(source.read_bytes()).hexdigest()
    if source.resolve()==REFERENCE_SOURCE.resolve() and identity!=REFERENCE_BASELINE_SHA256:
        raise ValueError('The author-protocol reproduction baseline is frozen for this study.')
    BASE = SourcePolicy(source)
    SOURCE=source;BASELINE_SHA256=identity
    joint_load.cache_clear()


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
        *(a.velocity_r/5.),a.agility/3.,env.Current_T/env.Total_T],dtype=np.float32)
    if hasattr(env,'_conditional_option_count'):
        packet['_conditional_count']=np.asarray([env._conditional_option_count],dtype=np.float32)
    return packet


def proposal_parameter_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PublicTargetMotion(nn.Module):
    """Predict finite-horizon target motion from public fleet measurements.

    Shared vessel embeddings permit a changing fleet size. Coordinates are
    relative to the target and the protected asset, so no scene identity,
    attacker controller, agility parameter, or realized future is an input.
    """
    def __init__(self,width=48,absolute_frame=False):
        super().__init__()
        self.absolute_frame=absolute_frame
        self.nodes=nn.Sequential(nn.Linear(9,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear(2*width+5+2*absolute_frame,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,4))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)

    @staticmethod
    def encode(packet,absolute_frame=False):
        motion=packet['motion'];c=motion[:,0,2].cos();s=motion[:,0,2].sin()
        def world(vector):
            return torch.stack((c*vector[:,0]-s*vector[:,1],s*vector[:,0]+c*vector[:,1]),-1)
        target=motion[:,0,:2]+world(packet['node'][:,0,2:4]*30.)
        velocity=world(packet['node'][:,0,4:6]*5.)
        return PublicTargetMotion.encode_state(motion,packet['node'][...,11:13],target,velocity,absolute_frame)

    @staticmethod
    def encode_state(motion,boids,target,velocity,absolute_frame=False):
        if (motion.ndim!=3 or motion.shape[-1]!=6 or boids.shape!=(*motion.shape[:2],2)
                or target.shape!=(len(motion),2) or velocity.shape!=target.shape):
            raise ValueError('Target-response inputs must contain public fleet poses, commands and target motion.')
        axis=-target/target.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        def rotate(vector):
            basis=axis[:,None] if vector.ndim==3 else axis
            return torch.stack(((vector*basis).sum(-1),basis[...,0]*vector[...,1]-basis[...,1]*vector[...,0]),-1)
        heading=torch.stack((motion[...,2].cos(),motion[...,2].sin()),-1)
        nodes=torch.cat((rotate(motion[...,:2]-target[:,None])/30.,
            rotate(motion[...,3:5]-velocity[:,None])/5.,rotate(heading),motion[...,5:6],
            boids),-1)
        global_state=torch.cat((target.norm(dim=-1,keepdim=True)/30.,rotate(velocity)/5.,
            target.new_full((len(target),1),motion.shape[1]/6.)),-1)
        if absolute_frame:
            # World bearing is observable. Removing it assumes rotationally
            # covariant target dynamics, which the source reset does not have.
            global_state=torch.cat((global_state,-axis),-1)
        return nodes,global_state,target,velocity,axis

    def forward(self,nodes,global_state,time):
        encoded=self.nodes(nodes)
        inputs=torch.cat((encoded.mean(1),encoded.amax(1),global_state,time[:,None]/3.2),-1)
        residual=6.*self.readout(inputs).tanh()
        return residual[:,:2]*time[:,None],residual[:,2:]*time[:,None].clamp_max(1.)

    def predict(self,packet,time):
        nodes,global_state,target,velocity,axis=self.encode(packet,self.absolute_frame)
        horizon=target.new_full((len(target),),float(time))
        displacement,delta_velocity=self(nodes,global_state,horizon)
        def world(vector):
            return torch.stack((axis[:,0]*vector[:,0]-axis[:,1]*vector[:,1],
                axis[:,1]*vector[:,0]+axis[:,0]*vector[:,1]),-1)
        return target+horizon[:,None]*velocity+world(displacement),velocity+world(delta_velocity)


class PublicTargetFeedback(nn.Module):
    """Advance the public target response separately inside every fleet forecast."""
    def __init__(self,width=48,states=(),absolute_frame=True,control_step=.2,scenario_mode='mean'):
        super().__init__()
        if not states or not np.isclose(control_step,.2):
            raise ValueError('Target feedback needs fitted members at the source control interval.')
        if scenario_mode not in ('mean','members'):
            raise ValueError('Target feedback must use its mean response or complete member trajectories.')
        self.scenario_mode=scenario_mode
        self.control_step=float(control_step);self.absolute_frame=bool(absolute_frame)
        self.models=nn.ModuleList([PublicTargetMotion(width,self.absolute_frame) for _ in states])
        for network,state in zip(self.models,states):network.load_state_dict(state)

    def predict_state(self,motion,boids,target,velocity,component=None):
        nodes,global_state,target,velocity,axis=PublicTargetMotion.encode_state(
            motion,boids,target,velocity,self.absolute_frame)
        horizon=target.new_full((len(target),),self.control_step)
        networks=self.models if component is None else (self.models[component],)
        predictions=[network(nodes,global_state,horizon) for network in networks]
        displacement=torch.stack([value[0] for value in predictions]).mean(0)
        delta_velocity=torch.stack([value[1] for value in predictions]).mean(0)
        def world(vector):
            return torch.stack((axis[:,0]*vector[:,0]-axis[:,1]*vector[:,1],
                axis[:,1]*vector[:,0]+axis[:,0]*vector[:,1]),-1)
        return target+self.control_step*velocity+world(displacement),velocity+world(delta_velocity)

    def component(self,index):
        if not isinstance(index,int) or not 0<=index<len(self.models):
            raise ValueError('A target trajectory component must identify a fitted member.')
        parent=self
        class Component:
            def predict_state(self,motion,boids,target,velocity):
                return parent.predict_state(motion,boids,target,velocity,component=index)
        return Component()


def joint_target_motion(args,pool):
    """Fit M1's motion predictor on factual trajectories, split by layout."""
    torch.set_num_threads(1);torch.set_grad_enabled(True);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    datasets=[torch.load(p,map_location='cpu',weights_only=False) for p in [args.data,*(args.references or [])]]
    rows=[];factual_steps=0
    absolute_frame=getattr(args,'absolute_frame',False)
    balanced_feedback=getattr(args,'balanced_feedback',False)
    for data in datasets:
        for rollout in data['rollouts']:
            steps=rollout['steps']
            if len(steps)<2:continue
            packet={key:torch.tensor(np.stack([s['packet'][key] for s in steps])) for key in ('motion','node')}
            nodes,state,position,velocity,axis=PublicTargetMotion.encode(packet,absolute_frame)
            clock=.2*np.cumsum([0,*[s['physical_steps'] for s in steps[:-1]]])
            factual_steps+=sum(s['physical_steps'] for s in steps)
            for horizon in (.2,.4,.8,1.6,3.2):
                future=np.searchsorted(clock,clock+horizon-1e-6)
                current=np.flatnonzero(future<len(steps));future=future[current]
                if not len(current):continue
                elapsed=torch.tensor(clock[future]-clock[current],dtype=torch.float32)
                def rotate(vector):
                    basis=axis[current]
                    return torch.stack(((vector*basis).sum(-1),basis[:,0]*vector[:,1]-basis[:,1]*vector[:,0]),-1)
                displacement=rotate(position[future]-position[current]-elapsed[:,None]*velocity[current])
                acceleration=rotate(velocity[future]-velocity[current])
                rows.append((nodes[current],state[current],elapsed,displacement,acceleration,
                    torch.full((len(current),),int(rollout['row']['seed'])%5==0,dtype=torch.bool)))
    if not rows:raise ValueError('Target-motion fitting needs public factual trajectory transitions.')
    arrays=[torch.cat([row[i] for row in rows]) for i in range(6)]
    if not arrays[-1].any() or arrays[-1].all():raise ValueError('Both training and held-out layout groups are required.')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    nodes,state,elapsed,displacement,velocity,heldout=[a.to(device) for a in arrays]
    train=torch.nonzero(~heldout).squeeze(-1);validation=torch.nonzero(heldout).squeeze(-1)
    predictor=PublicTargetMotion(absolute_frame=absolute_frame).to(device)
    optimizer=torch.optim.AdamW(predictor.parameters(),lr=8e-4,weight_decay=.001)
    best=float('inf');best_state=None;history=[]
    slow=train[state[train,1:3].norm(dim=-1)<.1]
    for step in range(args.updates):
        ids=train[torch.as_tensor(rng.integers(len(train),size=256),device=device)]
        if balanced_feedback and len(slow):
            ids[:64]=slow[torch.as_tensor(rng.integers(len(slow),size=64),device=device)]
        dp,dv=predictor(nodes[ids],state[ids],elapsed[ids])
        scale=elapsed[ids,None] if balanced_feedback else 1.
        loss=((dp-displacement[ids])/scale).square().mean()+(.75 if balanced_feedback else .15)*(dv-velocity[ids]).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(predictor.parameters(),5.);optimizer.step()
        if (step+1)%200==0 or step+1==args.updates:
            with torch.no_grad():
                errors=[]
                for ids in validation.split(1024):
                    dp,dv=predictor(nodes[ids],state[ids],elapsed[ids])
                    error=(dp-displacement[ids]).square().sum(-1)
                    if balanced_feedback:error=error/elapsed[ids].square()+.75*(dv-velocity[ids]).square().sum(-1)
                    errors.append(error)
                score=float(torch.cat(errors).mean())
            if score<best:
                best=score;best_state={k:v.detach().cpu().clone() for k,v in predictor.state_dict().items()}
            history.append(dict(updates=step+1,heldout_position_mse=score))
    predictor.load_state_dict(best_state);predictor.eval();metrics={}
    with torch.no_grad():
        for horizon in (.2,.4,.8,1.6,3.2):
            ids=torch.nonzero(heldout&((elapsed-horizon).abs()<.05)).squeeze(-1)
            estimated=torch.cat([predictor(nodes[i],state[i],elapsed[i])[0] for i in ids.split(1024)])
            linear=displacement[ids].norm(dim=-1);learned=(estimated-displacement[ids]).norm(dim=-1)
            metrics[str(horizon)]=dict(samples=len(ids),linear_mean_m=float(linear.mean()),
                learned_mean_m=float(learned.mean()),linear_p90_m=float(linear.quantile(.9)),
                learned_p90_m=float(learned.quantile(.9)))
    model=torch.load(args.start,map_location='cpu',weights_only=False)
    model['target_motion']=dict(state=best_state,width=48,heldout_forecast=metrics,
        absolute_frame=absolute_frame,balanced_feedback=balanced_feedback,
        factual_training_steps=factual_steps,updates=args.updates,seed=args.seed,
        training_data=[str(args.data),*[str(p) for p in (args.references or [])]],
        validation='layout seed modulo five; all noise repeats of a layout remain in one group')
    model.pop('selection',None);args.output.mkdir(parents=True,exist_ok=True)
    torch.save(model,args.output/'policy.pth')
    save_json(args.output/'forecast.json',dict(metrics=metrics,history=history,training_samples=len(train),validation_samples=len(validation)))
    print(dict(phase='public_target_motion_fit',metrics=metrics),flush=True)


class JointLearningActor(nn.Module):
    """Continue the original neural proposal and learn its cooperative Adapter."""
    def __init__(self,peer=True,graph=False,levels=2,risk_prior=0.,risk_horizon=1.6,pair_scale=1.,gate_period=1,risk_clearance=6.,forecast_inputs=False,frozen_proposal=False,risk_feedback=False,intent_strength=0.,coupled_shortlist=0,prediction_step=.1,risk_tail=0.,risk_barrier=0.,interception_weight=0.):
        super().__init__();self.peer=peer;self.graph=graph;self.levels=levels
        self.risk_prior=risk_prior;self.risk_horizon=risk_horizon;self.pair_scale=pair_scale
        self.risk_clearance=risk_clearance
        self.forecast_inputs=forecast_inputs;self.frozen_proposal=frozen_proposal
        self.risk_feedback=risk_feedback
        self.intent_strength=intent_strength
        self.coupled_shortlist=coupled_shortlist
        self.adaptive_fusion_grid=False
        self.prediction_step=prediction_step;self.risk_tail=risk_tail
        self.prediction_relaxation=0.
        self.current_marginal_prediction=False
        self.fusion_expression='interaction'
        self.prediction_iterations=0
        self.pair_feedback_prediction=False
        self.physical_prior_weight=0.
        self.risk_barrier=risk_barrier
        self.interception_weight=interception_weight
        self.interception_horizon=0.
        self.prediction_discount=0.
        self.prediction_discount_start=1.6
        self.target_motion=None
        self.target_feedback=None
        self.feedback_fleet_policy=None
        self.terminal_prediction=False
        self.capture_credit=0.
        self.interaction_enabled=True
        self.conditional_commitment=False
        self.immediate_commitment=False
        self.fixed_intervention_clock=False
        self.preference_strength=0.
        self.local_value=None
        self.physical_value=None
        self.physical_values=nn.ModuleList()
        self.physical_spans=[]
        self.physical_reversals=[]
        self.physical_reversal=False
        self.physical_cost=.02
        self.physical_span=10
        self.response_values=nn.ModuleList()
        self.response_span=2
        self.response_margin=.15
        self.response_heads=1
        self.response_paired_gain=False
        self.response_anchor=None
        self.response_effect_support=None
        self.response_action_support=None
        self.response_base=None
        self.joint_intervention_value=None
        self.joint_intervention_values=nn.ModuleList()
        self.midpoint_intervention_values=None
        self.option_refinement_fraction=.5
        self.option_refinement_steps=0
        self.option_refinement_checkpoints=False
        self.phase_preserving_refinement=False
        self.shared_intervention_prior=False
        self.initial_role_credit=None
        self.feedback_role_credit=None
        self.cooperative_role_credit=None
        self.finite_temporal_credit=None
        self.continuous_program_value=None
        self.successor_task_value=None
        self.response_feedback_value=None
        self.generic_fusion_adapter=None
        self.conditional_enabled=True
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
        local=edge(False)
        if not self.interaction_enabled:local[...,17:]=0.
        relational=edge(True) if self.interaction_enabled else local
        expression=getattr(self,'fusion_expression','interaction')
        if expression!='interaction':
            # Generic learned alternatives consume the same source commands
            # without the analytic candidate-response relation. The state
            # control aggregates neighbor states; the candidate control also
            # receives both of each neighbor's source commands. Both use the
            # same MLP capacity, pooling, discrete gates and ordinary PPO loss.
            relational=relational.clone();local=local.clone()
            relational[...,17:]=0.;local[...,17:]=0.
            if expression=='state':relational[...,13:17]=0.
        return dict(node=node,edge=relational,edge_no_peer=local,prior=prior)

    @torch.no_grad()
    def candidate_risk(self,packet,proposals,return_distance=False,pairwise=False,reversal_at=None,
                       candidate_grid=None,proposal_offsets=None,return_paths=False):
        motion=packet['motion'];b,n=motion.shape[:2]
        actuator_grid=candidate_grid is not None and candidate_grid.ndim==4
        k=self.levels if candidate_grid is None else candidate_grid.shape[-2 if actuator_grid else -1]
        if not self.interaction_enabled:
            if return_paths:return motion.new_zeros((b,n,k,round(self.risk_horizon/self.prediction_step),2))
            return motion.new_zeros((b,n,n,k,k) if self.peer or pairwise else (b,n,k))
        if self.pair_feedback_prediction and not return_paths:
            if candidate_grid is not None or proposal_offsets is not None or reversal_at is not None:
                raise ValueError('Pair feedback prediction requires the common fixed blending candidates.')
            return self.pair_feedback_risk(packet,proposals,return_distance)
        iterations=packet.get('_forecast_iteration',self.prediction_iterations)
        if iterations:
            # Recompute candidate responses against predicted peer motion.
            # The reference averages the same candidates under the source
            # preference, so joint and independent selectors share it exactly.
            previous=dict(packet,_forecast_iteration=iterations-1)
            trajectories=self.candidate_risk(previous,proposals,reversal_at=reversal_at,
                candidate_grid=candidate_grid,proposal_offsets=proposal_offsets,return_paths=True)
            hidden=self.proposal_distribution(packet['obs'])[2]
            own=self.base.activation(self.base.a1(torch.cat((proposals,packet['node'][...,11:13]),-1)))
            prior=.02+.96*(self.base.adap_layer(torch.cat((own,hidden),-1)).tanh().squeeze(-1)*.5+.5)
            center=(2.*torch.tanh(torch.logit(prior)/2.)).sigmoid()
            reference_grid=(torch.linspace(0.,1.,k,device=motion.device) if candidate_grid is None else candidate_grid)
            reference_grid=reference_grid[None,None] if reference_grid.ndim==1 else reference_grid
            departure=((reference_grid-center[...,None,None])/.4).square().mean(-1) if actuator_grid else (
                (reference_grid-center[...,None])/.4).square()
            weights=(-departure).softmax(-1)
            positions=(trajectories*weights[...,None,None]).sum(2)
            velocities=((positions[:,:,1:]-positions[:,:,:-1])/self.prediction_step)
            velocities=(torch.cat((velocities,velocities[:,:,-1:]),2) if velocities.shape[2]
                else motion[:,:,None,3:5])
            packet['_candidate_peer_reference']=(positions,velocities)
        grid=(torch.linspace(0.,1.,k,device=motion.device) if candidate_grid is None else candidate_grid)
        grid=grid[None,None] if grid.ndim==1 else grid
        candidate=(proposals[:,:,None] if proposal_offsets is None else
            (proposals[:,:,None]+proposal_offsets[None,None]).clamp(-1.,1.))
        mixture=grid if actuator_grid else grid[...,None]
        force=750.*(mixture*candidate+
            (1.-mixture)*packet['node'][:,:,None,11:13])+250.
        predicted_motion=self.prediction_motion(packet)
        x,y,heading,vx,vy,r=[predicted_motion[...,i,None].expand(-1,-1,k).clone() for i in range(6)]
        paths=[];dt=self.prediction_step
        feedback_steps=max(1,round(.2/dt))
        for step in range(round(self.risk_horizon/dt)):
            if self.risk_feedback and step and step%feedback_steps==0:
                state=torch.stack((x,y,heading,vx,vy,r),-1).transpose(1,2)
                forecast_obs,boids=self.feedback_observation(packet,state,step*dt)
                mean,_,hidden=self.proposal_distribution(forecast_obs.reshape(b*k,n,-1))
                future=mean.tanh().reshape(b,k,n,2).transpose(1,2)
                if self.conditional_enabled and '_proposal_offset' in packet:
                    future=(future+packet['_proposal_offset'][:,:,None]).clamp(-1.,1.)
                if proposal_offsets is not None:future=(future+proposal_offsets[None,None]).clamp(-1.,1.)
                current_grid=1.-grid if reversal_at is not None and step*dt>=reversal_at else grid
                mixture=current_grid if actuator_grid else current_grid[...,None]
                relaxation=getattr(self,'prediction_relaxation',0.)
                if relaxation>0.:
                    # A receding fusion decision is not a promise to hold its
                    # coefficient for the entire forecast. Continue toward the
                    # shared source Adapter as feedback observations change.
                    own=self.base.activation(self.base.a1(torch.cat((future.transpose(1,2),boids),-1)))
                    native=self.base.adap_layer(torch.cat((own,hidden.reshape(b,k,n,-1)),-1)).tanh()*.5+.5
                    continuation=(2.*torch.tanh(torch.logit(.02+.96*native)/2.)).sigmoid().transpose(1,2)
                    weight=1.-np.exp(-step*dt/relaxation)
                    relaxed=(1.-weight)*mixture+weight*continuation
                    committed=packet.get('_committed_gates')
                    mixture=(torch.where((committed>=0.)[:,:,None,None],mixture,relaxed)
                        if committed is not None else relaxed)
                force=750.*(mixture*future+(1.-mixture)*boids.transpose(1,2))+250.
            x=x+dt*vx;y=y+dt*vy;heading=heading+dt*r
            c,s=heading.cos(),heading.sin();u=c*vx+s*vy;v=-s*vx+c*vy
            du=(400.*r*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
            dv=(-380.*r*u-(100.+150.*v.abs())*v)/400.
            dr=(-20.*u*v-(980.+950.*r.abs())*r+1.25*(force[...,0]-force[...,1]))/1430.
            u=u+dt*du;v=v+dt*dv;vx=c*u-s*v;vy=s*u+c*v;r=r+dt*dr
            paths.append(torch.stack((x,y),-1))
        paths=torch.stack(paths,-2)
        if packet.get('_retain_candidate_paths', False):
            packet['_retained_candidate_paths'] = paths
        if return_paths:return paths
        if self.terminal_prediction:
            target,target_velocity=self.public_target_motion(packet)
            times=torch.arange(1,paths.shape[-2]+1,device=motion.device)*dt
            target_path=(torch.stack([self.target_motion.predict(packet,float(t))[0] for t in times],1)
                if self.target_motion is not None else target[:,None]+times[None,:,None]*target_velocity[:,None])
            packet['_terminal_paths']=(paths,target_path)
        if self.interception_weight:
            target,target_velocity=self.public_target_motion(packet)
            times=torch.arange(1,paths.shape[-2]+1,device=motion.device)*dt
            target_path=(torch.stack([self.target_motion.predict(packet,float(t))[0] for t in times],1)
                if self.target_motion is not None else target[:,None]+times[None,:,None]*target_velocity[:,None])
            # The closest eligible interceptor determines capture; summing
            # every vessel's pursuit cost would spuriously reward crowding.
            reach=(paths-target_path[:,None,None]).norm(dim=-1)
            if self.interception_horizon:
                reach=reach[...,:max(1,round(self.interception_horizon/dt))]
            packet['_candidate_interception']=reach.amin(-1).clamp_min(5.)
        if self.peer or pairwise:
            delta=paths[:,:,None,:,None]-paths[:,None,:,None]
            distance=delta.norm(dim=-1);minimum=distance.amin(-1)
            if return_distance:return minimum
            if self.prediction_discount:
                time=torch.arange(1,distance.shape[-1]+1,device=motion.device)*dt
                weight=torch.exp(-(time-self.prediction_discount_start).clamp_min(0.)/self.prediction_discount)
                return (self.distance_risk(distance)*weight).amax(-1)
            return self.distance_risk(minimum)
        time=torch.arange(1,len(paths[0,0,0])+1,device=motion.device)*dt
        constant=motion[:,:,:2][:,:,None]+time[None,None,:,None]*motion[:,:,3:5][:,:,None]
        delta=paths[:,:,None]-constant[:,None,:,None]
        distance=delta.norm(dim=-1);minimum=distance.amin(-1)
        mask=~torch.eye(n,dtype=torch.bool,device=motion.device)[None,:,:,None]
        if self.prediction_discount:
            weight=torch.exp(-(time-self.prediction_discount_start).clamp_min(0.)/self.prediction_discount)
            return ((self.distance_risk(distance)*weight).amax(-1)*mask).amax(2)
        return (self.distance_risk(minimum)*mask).amax(2)

    @torch.no_grad()
    def pair_feedback_risk(self,packet,proposals,return_distance=False):
        """Predict each candidate pair with reciprocal feedback responses.

        Both members respond to the same evolving encounter. Other vessels
        supply a common nominal continuation, keeping the pair costs and the
        complete candidate library identical for joint and independent choice.
        """
        motion=packet['motion'];b,n=motion.shape[:2];k=self.levels
        device=motion.device;dt=self.prediction_step
        pairs=torch.triu_indices(n,n,1,device=device);p=pairs.shape[1]
        i=pairs[0].repeat_interleave(k*k);j=pairs[1].repeat_interleave(k*k)
        code=torch.arange(k*k,device=device).repeat(p)
        left=code//k;right=code%k;m=len(code)
        index=torch.arange(m,device=device)
        gates=motion.new_zeros((m,n));gates[index,i]=left/(k-1);gates[index,j]=right/(k-1)
        active=torch.zeros((m,n),dtype=torch.bool,device=device)
        active[index,i]=True;active[index,j]=True
        state=motion[:,None].expand(-1,m,-1,-1).clone()
        force=750.*(gates[None,...,None]*proposals[:,None]
            +(1.-gates[None,...,None])*packet['node'][:,None,:,11:13])+250.
        minimum=motion.new_full((b,m),float('inf'))
        feedback_steps=max(1,round(.2/dt))
        for step in range(round(self.risk_horizon/dt)):
            if self.risk_feedback and step and step%feedback_steps==0:
                obs,boids=self.feedback_observation(packet,state,step*dt,joint=True)
                future=self.proposal_distribution(obs.reshape(b*m,n,-1))[0].tanh().reshape(b,m,n,2)
                if self.conditional_enabled and '_proposal_offset' in packet:
                    future=(future+packet['_proposal_offset'][:,None]).clamp(-1.,1.)
                force=750.*(gates[None,...,None]*future+(1.-gates[None,...,None])*boids)+250.
            x,y,heading,vx,vy,r=state.unbind(-1)
            x=x+dt*vx;y=y+dt*vy;heading=heading+dt*r
            c,s=heading.cos(),heading.sin();u=c*vx+s*vy;v=-s*vx+c*vy
            du=(400.*r*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
            dv=(-380.*r*u-(100.+150.*v.abs())*v)/400.
            dr=(-20.*u*v-(980.+950.*r.abs())*r+1.25*(force[...,0]-force[...,1]))/1430.
            u=u+dt*du;v=v+dt*dv
            predicted=torch.stack((x,y,heading,c*u-s*v,s*u+c*v,r+dt*dr),-1)
            nominal=motion.clone();nominal[...,:2]=motion[...,:2]+(step+1)*dt*motion[...,3:5]
            state=torch.where(active[None,...,None],predicted,nominal[:,None])
            distance=(state[:,index,i,:2]-state[:,index,j,:2]).norm(dim=-1)
            minimum=torch.minimum(minimum,distance)
        values=minimum if return_distance else self.distance_risk(minimum)
        table=motion.new_zeros((b,n,n,k,k))
        table[:,i,j,left,right]=values;table[:,j,i,right,left]=values
        return table

    @torch.no_grad()
    def conditional_response_features(self,packet,action,proposal=True,return_decisions=False,all_pairs=False):
        """Share M1 responses across M2 alternatives, then expose their effects.

        Full M1 removal returns zero features before any prediction executes.
        The candidate library reuses the same feedback dynamics and joint risk
        decoder, with each changed neural proposal carried through prediction.
        """
        from conditional_fusion_value import conditional_action_programs
        programs,offsets=conditional_action_programs(action.shape[1],action.device,proposal,all_pairs)
        b,n=action.shape[:2];o=len(programs)
        if not self.interaction_enabled:
            zero=action.new_zeros((b,o,16))
            if return_decisions:
                precision=self.preference_strength*.4**2
                gate=(action[:,None,:,2]+precision*programs[None])/(1.+precision)
                return zero,torch.where(programs[None]>=0.,gate,action[:,None,:,2])
            return zero
        if self.levels!=3 or self.coupled_shortlist or self.terminal_prediction or self.interception_weight:
            raise ValueError('Response credit requires its matching three-level M1 decoder.')
        if not getattr(self,'_response_decoder_checked',False):
            if any(torch.count_nonzero(p).item() for layer in (self.adapter.gate[-1],self.gate_pairs[-1])
                   for p in layer.parameters()):
                raise ValueError('Response credit must include learned gate factors before using this decoder.')
            self._response_decoder_checked=True
        changes,variants=torch.unique(offsets,dim=0,return_inverse=True)
        u=len(changes)
        expanded={key:value[:,None].expand(-1,u,*value.shape[1:]).reshape(b*u,*value.shape[1:])
            for key,value in packet.items() if key in ('node','edge','motion','obs','water','task')}
        expanded['_proposal_offset']=changes[None].expand(b,-1,-1,-1).reshape(b*u,n,2)
        proposed=(action[:,None,:,:2]+changes[None]).clamp(-1.,1.)
        paths=self.candidate_risk(expanded,proposed.reshape(b*u,n,2),return_paths=True)
        paths=paths.reshape(b,u,n,3,-1,2)
        separation=(paths[:,:,:,None,:,None]-paths[:,:,None,:,None]).norm(dim=-1).amin(-1)
        risk=self.distance_risk(separation)
        _,_,hidden=self.proposal_distribution(packet['obs'])
        rule=packet['node'][:,None,:,11:13].expand(-1,u,-1,-1)
        own=self.base.activation(self.base.a1(torch.cat((proposed,rule),-1)))
        raw=self.base.adap_layer(torch.cat((own,hidden[:,None].expand(-1,u,-1,-1)),-1)).tanh().squeeze(-1)*.5+.5
        center=(2.*torch.tanh(torch.logit(.02+.96*raw)/2.)).sigmoid()
        center=center[:,variants]
        ids=torch.arange(3**n,device=action.device)
        choices=(ids[:,None]//(3**torch.arange(n,device=action.device)))%3
        plans=choices.to(action.dtype)*.5
        scores=-((plans[None,None]-center[:,:,None])/.4).square().sum(-1)
        scores=scores-self.preference_strength*((plans[None]-programs[:,None]).square()
            *(programs[:,None]>=0.)).sum(-1)[None]
        worst=torch.zeros_like(scores)
        for i in range(n):
            for j in range(i+1,n):
                table=risk[:,variants,i,j]
                term=table[:,:,choices[:,i],choices[:,j]]
                scores=scores-self.risk_prior*term;worst=torch.maximum(worst,term)
        scores=scores-self.risk_prior*self.risk_tail*worst
        digits=choices[scores.argmax(-1)]
        selected=paths[torch.arange(b,device=action.device)[:,None,None],
            variants[None,:,None],torch.arange(n,device=action.device)[None,None],digits]
        pair_distance=torch.stack([(selected[:,:,i]-selected[:,:,j]).norm(dim=-1).amin(-1)
            for i in range(n) for j in range(i+1,n)],-1)
        pair_risk=self.distance_risk(pair_distance)
        target,velocity=self.public_target_motion(packet)
        time=torch.arange(1,paths.shape[-2]+1,device=action.device)*self.prediction_step
        future=target[:,None]+time[None,:,None]*velocity[:,None]
        relative=selected-future[:,None,None]
        distance=relative.norm(dim=-1)
        ticks=[max(0,paths.shape[-2]//4-1),max(0,paths.shape[-2]//2-1),paths.shape[-2]-1]
        nearest=distance[:,:,:,ticks].sort(dim=2).values[:,:,:2].transpose(-1,-2).flatten(-2)/30.
        snapshots=relative[:,:,:,ticks]
        angles=torch.atan2(snapshots[...,1],snapshots[...,0]).sort(dim=2).values
        gaps=torch.cat((angles[:,:,1:]-angles[:,:,:-1],2.*np.pi+angles[:,:,:1]-angles[:,:,-1:]),2)
        enclosure=gaps.amax(2)/(2.*np.pi)
        direction=-future[:,ticks]/future[:,ticks].norm(dim=-1,keepdim=True).clamp_min(1.)
        along=(snapshots*direction[:,None,None]).sum(-1)
        across=(snapshots-along[...,None]*direction[:,None,None]).norm(dim=-1)
        blocking=(across+(-along).clamp_min(0.)).amin(2)/30.
        features=torch.cat((pair_distance.amin(-1,keepdim=True)/10.,
            pair_risk.amax(-1,keepdim=True).clamp_max(500.)/25.,
            pair_risk.mean(-1,keepdim=True).clamp_max(500.)/25.,nearest,
            distance.amin((-1,-2))[...,None]/30.,enclosure,blocking),-1).clamp(-20.,20.)
        return (features,digits.to(action.dtype)*.5) if return_decisions else features

    def distance_risk(self,minimum):
        risk=(self.risk_clearance-minimum).clamp_min(0.).square()
        if self.risk_barrier:
            # Separate an imminent collision from several benign clearances.
            # The broad interaction cost still shapes early coordination;
            # this smooth contact term prevents exchanging one unsafe pair
            # for many small reductions in already safe pair costs.
            risk=risk+self.risk_barrier*F.softplus((5.5-minimum)/.3).square()
        return risk

    def public_target_motion(self,packet):
        motion=packet['motion'];c=motion[:,0,2].cos();s=motion[:,0,2].sin()
        def rotate(vector):
            return torch.stack((c*vector[:,0]-s*vector[:,1],s*vector[:,0]+c*vector[:,1]),-1)
        target=motion[:,0,:2]+rotate(packet['node'][:,0,2:4]*30.)
        return target,rotate(packet['node'][:,0,4:6]*5.)

    def risk_scores(self,scores,risk,choices,interception=None,terminal=None):
        """Credit joint choices for both aggregate and worst-encounter risk."""
        n=choices.shape[-1]
        selection=getattr(self,'fusion_selection','joint')
        if selection in ('independent','single_reference') and not self.coupled_shortlist:
            # Both controls use exactly the full method's candidate paths,
            # pair costs and source preference. Only the decision dependency
            # changes: each boat averages over the others' local preference,
            # or conditions on their single preferred reference candidate.
            k=risk.shape[-1]
            unary=torch.stack([torch.stack([
                scores[:,choices[:,i]==g].logsumexp(-1) for g in range(k)],-1)
                for i in range(n)],1)
            reference=unary.softmax(-1)
            if selection=='single_reference':
                reference=torch.nn.functional.one_hot(unary.argmax(-1),k).to(scores.dtype)
            mask=~torch.eye(n,dtype=torch.bool,device=scores.device)
            # candidate_risk retains self pairs, so remove those before the
            # expectation instead of interpreting them as collision costs.
            expected=torch.einsum('bijgh,bjh,ij->big',risk,reference,mask.to(scores.dtype))
            local=unary-self.risk_prior*expected
            return sum(local[:,i,choices[:,i]] for i in range(n))
        if self.terminal_prediction and terminal is not None and self.peer:
            paths,target=terminal
            # Capture terminates the task. A collision at the same instant
            # still takes precedence, but later encounters have no task cost.
            # Keep the stopping time joint: any one vessel can intercept.
            selected=torch.stack([paths[:,i,choices[:,i]] for i in range(n)],2)
            capture=(selected-target[:,None,None]).norm(dim=-1).amin(2)<5.
            ticks=torch.arange(paths.shape[-2],device=paths.device)
            stop=torch.where(capture,ticks,paths.shape[-2]).amin(-1)
            alive=ticks<=stop[...,None]
            terms=[]
            for i in range(n):
                for j in range(i+1,n):
                    distance=(selected[:,:,i]-selected[:,:,j]).norm(dim=-1)
                    minimum=distance.masked_fill(~alive,float('inf')).amin(-1)
                    terms.append(self.distance_risk(minimum))
            combined=torch.stack(terms,-1)
            scores=scores-self.risk_prior*(combined.sum(-1)+self.risk_tail*combined.amax(-1))
            scores=scores+self.capture_credit*(stop<paths.shape[-2])*(1.-stop/(paths.shape[-2]+1.))
            if self.interception_weight and interception is not None:
                reach=torch.stack([interception[:,i,choices[:,i]] for i in range(n)],-1).amin(-1)
                scores=scores-self.interception_weight*reach
            return scores
        terms=([risk[:,i,j,choices[:,i],choices[:,j]] for i in range(n) for j in range(i+1,n)]
            if self.peer else [risk[:,i,choices[:,i]] for i in range(n)])
        if self.interception_weight and interception is not None:
            reach=torch.stack([interception[:,i,choices[:,i]] for i in range(n)],-1).amin(-1)
            scores=scores-self.interception_weight*reach
        if not self.risk_tail:
            for term in terms:scores=scores-self.risk_prior*term
            return scores
        terms=torch.stack(terms,-1)
        return scores-self.risk_prior*(terms.sum(-1)+self.risk_tail*terms.amax(-1))

    def prediction_motion(self,packet):
        """Predict new-step motion after marginalizing the observed current.

        The source task draws a new zero-mean current at each control step.
        Water-relative velocity is persistent; the last ground-relative
        velocity also contains the previous transient disturbance.
        """
        motion=packet['motion']
        if not self.current_marginal_prediction:return motion
        return torch.cat((motion[...,:3],5.*packet['water']),-1)

    def feedback_observation(self,packet,state,time,joint=False,target_override=None):
        """Original source observations at nominal candidate poses.

        Each candidate refreshes the unchanged source neural and Boids commands;
        neighbors follow public constant-velocity forecasts. These are predictor
        inputs only. No new controller or simulator-private state is executed.
        """
        motion=packet['motion'];b,k,n=state.shape[:3]
        position=state[...,:2];velocity=state[...,3:5];heading=state[...,2]
        peers=motion[:,:,:2]+time*motion[:,:,3:5]
        peer_velocity=motion[:,:,3:5]
        reference=packet.get('_candidate_peer_reference')
        if reference is not None and not joint:
            index=min(max(round(time/self.prediction_step)-1,0),reference[0].shape[2]-1)
            peers=reference[0][:,:,index]
            peer_velocity=reference[1][:,:,index]
        relative=(position[:,:,None]-position[:,:,:,None] if joint else
            peers[:,None,None]-position[:,:,:,None])
        distance=relative.norm(dim=-1)
        mask=~torch.eye(n,dtype=torch.bool,device=state.device)[None,None]
        separation=(-relative/(distance-3.5).clamp_min(.01)[...,None].square()
            *((distance<15.)&mask)[...,None]).sum(-2)
        alignment=(velocity.mean(2,keepdim=True).expand_as(velocity) if joint else
            (peer_velocity.sum(1)[:,None,None]-peer_velocity[:,None]+velocity)/n)
        cohesion=(position.mean(2,keepdim=True)-position if joint else
            (peers.sum(1)[:,None,None]-peers[:,None]+position)/n-position)
        h0=motion[:,0,2];c0,s0=h0.cos(),h0.sin()
        def world(vector):
            return torch.stack((c0*vector[:,0]-s0*vector[:,1],s0*vector[:,0]+c0*vector[:,1]),-1)
        attacker=motion[:,0,:2]+world(packet['node'][:,0,2:4]*30.)
        att_velocity=world(packet['node'][:,0,4:6]*5.)
        target=attacker+time*att_velocity
        if self.target_motion is not None and self.interaction_enabled and time:
            target,att_velocity=self.target_motion.predict(packet,time)
        if target_override is not None:
            target,att_velocity=target_override
            if target.shape not in ((b,2),(b,k,2)) or att_velocity.shape!=target.shape:
                raise ValueError('A public feedback scenario needs one target state per batch or fleet plan.')
        target_position=target[:,None,None] if target.ndim==2 else target[:,:,None]
        target_velocity=att_velocity[:,None,None] if att_velocity.ndim==2 else att_velocity[:,:,None]
        attraction=target_position-position
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
            polar(target_velocity.expand_as(position)),polar(separation),
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
        b,n=proposals.shape[:2];device=proposals.device
        k=round(plans.shape[-2]**(1./n)) if plans.ndim==3 else self.levels
        powers=k**torch.arange(n,device=device)
        plan_count=plans.shape[-2]
        physical_plans=plans[None] if plans.ndim==2 else plans
        committed=packet.get('_committed_gates')
        shortlist=min(self.coupled_shortlist,plan_count)
        if committed is not None:
            feasible=((committed[:,None]<0.)|((physical_plans-committed[:,None]).abs()<1e-5)).all(-1)
            if not bool(feasible.any(-1).all()):
                raise ValueError('Committed gates must be represented in the joint fusion grid.')
            short_scores=short_scores.masked_fill(~feasible,-torch.finfo(short_scores.dtype).max)
            shortlist=min(shortlist,int(feasible.sum(-1).max()))
        best=short_scores.argmax(-1);digits=(best[:,None]//powers)%k
        neighbors=best[:,None,None]+(torch.arange(k,device=device)[None,None]-digits[...,None])*powers[None,:,None]
        uniform=torch.arange(k,device=device)*powers.sum()
        indices=torch.cat((short_scores.topk(shortlist,dim=-1).indices,
            neighbors.flatten(1),uniform[None].expand(b,-1)),1)
        if shortlist==plan_count and committed is None:
            indices=torch.arange(plan_count,device=device)[None].expand(b,-1)
        if committed is not None:
            # Form every prediction under the intervention that will actually
            # execute. Project exploratory neighbors as well as the shortlist;
            # masking only after prediction can leave no evaluated feasible plan.
            candidate_digits=(indices[...,None]//powers)%k
            fixed_digits=((committed.clamp_min(0.)*(k-1)).round().long() if plans.ndim==2 else
                (physical_plans[:,uniform].transpose(1,2)-committed[...,None]).abs().argmin(-1))
            candidate_digits=torch.where(committed[:,None]>=0.,fixed_digits[:,None],candidate_digits)
            indices=(candidate_digits*powers).sum(-1)
            # Eliminate repeated feasible decisions after fixing variables.
            # With one free vessel there are only three distinct programs;
            # predicting every projected neighbor would repeat these rollouts.
            distinct=[row.unique() for row in indices]
            count=max(len(row) for row in distinct)
            indices=torch.stack([torch.cat((row,row[:1].expand(count-len(row)))) for row in distinct])
        # Duplicate plans are harmless and preserve a rectangular batched
        # rollout without changing the set of candidate combinations.
        gates=(plans[indices] if plans.ndim==2 else plans.gather(1,indices[...,None].expand(-1,-1,n)))
        m=gates.shape[1]
        state=self.prediction_motion(packet)[:,None].expand(-1,m,-1,-1).clone()
        force=750.*(gates[...,None]*proposals[:,None]+(1.-gates[...,None])*packet['node'][:,None,:,11:13])+250.
        minimum=None;discounted_risk=None;dt=self.prediction_step
        alive=torch.ones((b,m),dtype=torch.bool,device=device)
        capture_at=proposals.new_full((b,m),self.risk_horizon+dt)
        target,target_velocity=self.public_target_motion(packet)
        feedback_steps=max(1,round(.2/dt))
        for step in range(round(self.risk_horizon/dt)):
            if self.risk_feedback and step and step%feedback_steps==0:
                obs,boids=self.feedback_observation(packet,state,step*dt,joint=self.peer)
                mean,_,hidden=self.proposal_distribution(obs.reshape(b*m,n,-1))
                future=mean.tanh().reshape(b,m,n,2)
                if self.conditional_enabled and '_proposal_offset' in packet:
                    future=(future+packet['_proposal_offset'][:,None]).clamp(-1.,1.)
                mixture=gates[...,None]
                relaxation=getattr(self,'prediction_relaxation',0.)
                if relaxation>0.:
                    # A fusion candidate fixes the present action, while later
                    # uncommitted decisions respond to the candidate's evolving
                    # joint state. Retain a genuinely held intervention, but
                    # continue free decisions toward the predicted local prior.
                    own=self.base.activation(self.base.a1(torch.cat((future,boids),-1)))
                    native=self.base.adap_layer(torch.cat((own,hidden.reshape(b,m,n,-1)),-1)).tanh()*.5+.5
                    continuation=(2.*torch.tanh(torch.logit(.02+.96*native)/2.)).sigmoid()
                    weight=1.-np.exp(-step*dt/relaxation)
                    relaxed=(1.-weight)*mixture+weight*continuation
                    mixture=(torch.where((committed>=0.)[:,None,:,None],mixture,relaxed)
                        if committed is not None else relaxed)
                force=750.*(mixture*future+(1.-mixture)*boids)+250.
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
            if self.terminal_prediction:
                distance=distance.masked_fill(~alive[:,:,None,None],float('inf'))
                target_position=(self.target_motion.predict(packet,(step+1)*dt)[0]
                    if self.target_motion is not None else target+(step+1)*dt*target_velocity)
                captured=(position-target_position[:,None,None]).norm(dim=-1).amin(-1)<5.
                capture_at=torch.where(alive&captured,(step+1)*dt,capture_at)
                alive=alive&~captured
            minimum=distance if minimum is None else torch.minimum(minimum,distance)
            if self.prediction_discount:
                weight=np.exp(-max(0.,(step+1)*dt-self.prediction_discount_start)/self.prediction_discount)
                current=self.distance_risk(distance)*weight
                discounted_risk=current if discounted_risk is None else torch.maximum(discounted_risk,current)
        risk=discounted_risk if discounted_risk is not None else self.distance_risk(minimum)
        mask=~torch.eye(n,dtype=torch.bool,device=device)
        cost=((risk*torch.triu(mask,diagonal=1)).sum((-1,-2)) if self.peer else
            (risk*mask).amax(-1).sum(-1))
        cost=cost+self.risk_tail*(risk*mask).amax((-1,-2))
        rescored=base_scores.gather(1,indices)-self.risk_prior*cost
        if self.terminal_prediction:
            rescored=rescored+self.capture_credit*(~alive)*(1.-capture_at/(self.risk_horizon+2.*dt))
        if self.interception_weight:
            target,target_velocity=self.public_target_motion(packet)
            endpoint=(self.target_motion.predict(packet,self.risk_horizon)[0] if self.target_motion is not None
                else target+self.risk_horizon*target_velocity)
            distance=(position-endpoint[:,None,None]).norm(dim=-1)
            rescored=rescored-self.interception_weight*distance.amin(-1).clamp_min(5.)
        if getattr(self,'coupled_robust_prediction',False):
            # The nominal-neighbor and mutually responding forecasts are two
            # candidate-conditioned continuations. Optimize their lower value
            # envelope so a joint plan cannot win solely by exploiting a large
            # optimistic change in one prediction model.
            rescored=torch.minimum(rescored,short_scores.gather(1,indices))
        result=torch.full_like(short_scores,-1e6)
        result.scatter_(1,indices,rescored)
        selection=getattr(self,'fusion_selection','joint')
        if selection in ('independent','single_reference'):
            if self.coupled_shortlist<plan_count:
                raise ValueError('Independent coupled-response controls require the complete shared candidate library.')
            # Marginalize the SAME coupled-response objective under the SAME
            # source preference. Independent choices must not inherit a joint
            # argmax from the response-refinement step.
            choices=(torch.arange(plan_count,device=device)[:,None]//powers)%k
            probability=base_scores.softmax(-1)
            response=result-base_scores
            local=[]
            reference=base_scores.argmax(-1)
            reference_digits=choices[reference]
            for i in range(n):
                terms=[]
                for g in range(k):
                    mask=choices[:,i]==g
                    preference=base_scores[:,mask].logsumexp(-1)
                    if selection=='independent':
                        mass=probability[:,mask]
                        effect=(mass*response[:,mask]).sum(-1)/mass.sum(-1).clamp_min(1e-30)
                    else:
                        candidate=reference+(g-reference_digits[:,i])*powers[i]
                        effect=response.gather(1,candidate[:,None]).squeeze(1)
                    terms.append(preference+effect)
                local.append(torch.stack(terms,-1))
            result=sum(local[i][:,choices[:,i]] for i in range(n))
        return result

    def proposal_distribution(self,obs):
        b,n,d=obs.shape;net=self.base;act=net.activation
        x=torch.cat((act(net.f1(obs[...,:6])),act(net.f2(obs[...,6:14])),
            act(net.t1(obs[...,14:].reshape(b,n,-1,2))).mean(-2)),-1)
        h=act(net.l2(act(net.l1(x))))
        pm=net.mean_layer(h);ps=net.log_std_layer(h).clamp(-3.,-.3).exp()
        return pm,ps,h

    def task_decision(self,packet,action):
        value=self.successor_task_value
        if hasattr(value,'prepare_packet'):packet=value.prepare_packet(self,packet,action)
        return value(packet,action)

    def fusion_prior_scores(self,packet,proposals,center,plans,width=.4):
        physical=plans[None] if plans.ndim==2 else plans
        deviation=((physical-center[:,None])/width).square()
        if self.physical_prior_weight:
            # A coefficient displacement has a larger physical effect when
            # the learned and Boids commands disagree. Use that command-space
            # metric in the same local preference term, without a new reward.
            separation=(proposals-packet['node'][...,11:13]).square().mean(-1)
            precision=1.-self.physical_prior_weight+self.physical_prior_weight*separation
            deviation=deviation*precision[:,None]
        return -deviation.sum(-1)

    @torch.no_grad()
    def stationary_fusion_candidates(self,packet,proposals,center,width=.4):
        """Find continuous best responses before evaluating joint combinations.

        Interpolate each candidate-pair cost only to propose an interior
        coefficient. Its trajectory is then predicted afresh alongside every
        original candidate. All selectors receive this same candidate library;
        the proposal uses the source distribution, never a joint winner.
        """
        b,n=center.shape;k=self.levels
        if k<2 or not self.peer:
            return center
        grid=torch.linspace(0.,1.,k,device=center.device)
        forecast=dict(packet)
        risk=self.candidate_risk(forecast,proposals,pairwise=True)
        probability=(-((grid-center[...,None])/width).square()).softmax(-1)
        mask=~torch.eye(n,dtype=torch.bool,device=center.device)
        expected=self.risk_prior*torch.einsum('bijgh,bjh,ij->big',risk,probability,mask.to(risk.dtype))
        precision=torch.ones_like(center)
        if self.physical_prior_weight:
            separation=(proposals-packet['node'][...,11:13]).square().mean(-1)
            precision=1.-self.physical_prior_weight+self.physical_prior_weight*separation
        slope=(expected[...,1:]-expected[...,:-1])*(k-1)
        optimum=center[...,None]-.5*width**2*slope/precision[...,None].clamp_min(1e-6)
        optimum=optimum.maximum(grid[:-1]).minimum(grid[1:])
        cost=precision[...,None]*((optimum-center[...,None])/width).square()
        cost=cost+expected[...,:-1]+slope*(optimum-grid[:-1])
        return optimum.gather(-1,cost.argmin(-1,keepdim=True)).squeeze(-1)

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
        scores=self.fusion_prior_scores(packet,proposals,center,plans)
        scores=scores+self.intent_strength*torch.einsum('bn,kn->bk',2.*intent[...,2]-1.,plans)
        base_scores=scores.clone() if self.coupled_shortlist else None
        if self.coupled_shortlist<len(plans):
            risk=self.candidate_risk(packet,proposals)
            scores=self.risk_scores(scores,risk,choices,packet.get('_candidate_interception'),packet.get('_terminal_paths'))
        if self.coupled_shortlist:scores=self.refine_joint_scores(packet,proposals,plans,scores,base_scores)
        gates=plans[scores.argmax(-1)]
        return torch.cat((proposals,gates[...,None]),-1)

    @torch.no_grad()
    def value_encoding(self,packet,reversal_at=None):
        """Expose the public fusion decoder to the training-only option critic."""
        obs=packet.get('obs')
        if obs is None:
            obs,_=self.feedback_observation(packet,packet['motion'][:,None],0.)
            obs=obs[:,0]
            obs[...,12:14]=packet['node'][...,11:13]
        pm,_,h=self.proposal_distribution(obs);proposals=pm.tanh();b,n=proposals.shape[:2];k=self.levels
        if self.conditional_enabled and '_proposal_offset' in packet:
            proposals=(proposals+packet['_proposal_offset']).clamp(-1.,1.)
        own=self.base.activation(self.base.a1(torch.cat((proposals,packet['node'][...,11:13]),-1)))
        prior=.02+.96*(self.base.adap_layer(torch.cat((own,h),-1)).tanh().squeeze(-1)*.5+.5)
        center=(2.*torch.tanh(torch.logit(prior)/2.)).sigmoid()
        indices=torch.arange(k**n,device=proposals.device)
        choices=((indices[:,None]//(k**torch.arange(n,device=proposals.device)))%k).long()
        plans=choices.to(proposals.dtype)/(k-1)
        scores=self.fusion_prior_scores(packet,proposals,center,plans)
        risk=packet.get('_candidate_risk')
        prediction_packet=dict(packet,obs=obs)
        if risk is None:risk=self.candidate_risk(prediction_packet,proposals,reversal_at=reversal_at)
        scores=self.risk_scores(scores,risk,choices,prediction_packet.get('_candidate_interception'),prediction_packet.get('_terminal_paths'))
        return dict(_fusion_scores=scores,_fusion_proposals=proposals,_fusion_risk=risk)

    def choose_joint_value_option(self,packet,action):
        # Alternative windows compare effects against one frozen predecessor.
        # Evaluate its decision once before scoring the alternative programs.
        if self.shared_intervention_prior:
            action=self.joint_intervention_values[0].prior(packet,action)
        alternatives=[value(packet,action,return_gain=True,apply_prior=not self.shared_intervention_prior)
            for value in self.joint_intervention_values]
        gains=torch.stack([gain.masked_fill(gain<=value.margin,float('-inf'))
            for value,(_,gain) in zip(self.joint_intervention_values,alternatives)],1)
        best=gains.argmax(-1)
        return torch.stack([candidate for candidate,_ in alternatives],1)[torch.arange(len(action),device=action.device),best]

    def forward(self,packet,latent=None,deterministic=False,gate_only=False):
        obs=packet['obs'];b,n,d=obs.shape;net=self.base;act=net.activation
        pm,ps,h=self.proposal_distribution(obs)
        pz=(pm if deterministic or self.gate_period>1 or self.frozen_proposal else pm+ps*torch.randn_like(pm)) if latent is None else latent[...,:2]
        proposals=pz.tanh();boids=packet['node'][...,11:13]
        if self.conditional_enabled and '_proposal_offset' in packet:
            proposals=(proposals+packet['_proposal_offset']).clamp(-1.,1.)
        own=act(net.a1(torch.cat((proposals,boids),-1)))
        native_gate_logit=net.adap_layer(torch.cat((own,h),-1)).squeeze(-1)
        raw_prior=native_gate_logit.tanh()*.5+.5
        committed=packet.get('_committed_gates')
        held=packet.get('_held_gates',committed)
        if gate_only and deterministic and held is not None:
            fixed=(held>=0.).all(-1)
            if bool(fixed.any()):
                # Eliminate fully fixed decisions from the constrained joint
                # problem. Their source controls still update at every tick;
                # only encounters with free fusion variables need optimization.
                action=torch.cat((proposals,held[...,None]),-1)
                latent_out=torch.cat((pz,held[...,None]),-1)
                distribution=torch.distributions.Normal(pm,ps)
                zero=torch.zeros_like(raw_prior)
                logp=torch.stack((distribution.log_prob(pz).sum(-1),zero),-1)
                entropy=torch.stack((distribution.entropy().sum(-1),zero),-1)
                free=~fixed
                if bool(free.any()):
                    subset={key:value[free] for key,value in packet.items()}
                    selected=self.forward(subset,deterministic=True,gate_only=True)
                    action[free]=selected[0];latent_out[free]=selected[1]
                    logp[free]=selected[2];entropy[free]=selected[3]
                return action,latent_out,logp,entropy,None,None
        if not self.interaction_enabled and getattr(self,'continuous_gate_learning',False):
            # The independently trained no-M1 policy does not execute an
            # interaction encoder, candidate predictor or joint gate search.
            gate_mean=2.*native_gate_logit
            gate_std=self.gate_logstd.clamp(-4.,-.3).exp()
            gate_raw=gate_mean if deterministic else gate_mean+gate_std*torch.randn_like(gate_mean)
            if latent is not None:gate_raw=torch.logit(latent[...,2].clamp(1e-5,1.-1e-5))
            gates=raw_prior if deterministic and latent is None else gate_raw.sigmoid()
            density=torch.distributions.Normal(gate_mean,gate_std).log_prob(gate_raw)
            density=density-torch.log(gates.clamp_min(1e-7))-torch.log((1.-gates).clamp_min(1e-7))
            proposal_distribution=torch.distributions.Normal(pm,ps)
            logp=torch.stack((proposal_distribution.log_prob(pz).sum(-1),density),-1)
            entropy=torch.stack((proposal_distribution.entropy().sum(-1),-density),-1)
            action=torch.cat((proposals,gates[...,None]),-1)
            if not gate_only and self.cooperative_role_credit is not None and bool(self.cooperative_role_credit.eligible(packet).any()):
                if self.cooperative_role_credit.response_features:
                    raise ValueError('No-M1 execution requires a credit estimator trained without candidate interaction forecasts.')
                action=self.cooperative_role_credit(packet,action)
            if not gate_only and getattr(self,'finite_temporal_credit',None) is not None:
                action=self.finite_temporal_credit(packet,action)
            if not gate_only and self.continuous_program_value is not None:
                action=self.continuous_program_value(packet,action)
            if not gate_only and self.successor_task_value is not None:
                action=self.task_decision(packet,action)
            return action,torch.cat((pz,gates[...,None]),-1),logp,entropy,None,None
        actuator_refinement=getattr(self,'actuator_fusion_refinement',None)
        native_actuator=(self.interaction_enabled and actuator_refinement is not None
            and actuator_refinement.get('source_preference',False))
        if ((not self.interaction_enabled or native_actuator or self.generic_fusion_adapter is not None) and not self.intent_strength
                and not len(self.physical_values) and not len(self.response_values)
                and all(value is None for value in (self.local_value,self.physical_value,self.response_base,
                    self.initial_role_credit,self.feedback_role_credit,self.cooperative_role_credit,
                    self.finite_temporal_credit))):
            # The actuator selector must receive the same native preference
            # in both joint and independent comparisons. Running the legacy
            # joint selector first would hide a coupled decision in the
            # independent arm and change its candidate set. Conditional
            # preferences are applied before the single actuator selector.
            gates=raw_prior
            if native_actuator and actuator_refinement.get('prior_map')=='bounded_logit':
                gates=(2.*torch.tanh(torch.logit(.02+.96*raw_prior)/2.)).sigmoid()
            preference=packet.get('_option_preference')
            if preference is not None and self.preference_strength:
                precision=self.preference_strength*.4**2
                regularized=(raw_prior+precision*preference)/(1.+precision)
                gates=torch.where(preference>=0.,regularized,raw_prior)
            action=torch.cat((proposals,gates[...,None]),-1)
            if not gate_only:
                if self.joint_intervention_value is not None:
                    action=self.joint_intervention_value(packet,action)
                if len(self.joint_intervention_values):
                    action=self.choose_joint_value_option(packet,action)
                if self.continuous_program_value is not None:
                    action=self.continuous_program_value(packet,action)
                if self.successor_task_value is not None:
                    action=self.task_decision(packet,action)
            distribution=torch.distributions.Normal(pm,ps)
            zero=torch.zeros_like(gates)
            logp=torch.stack((distribution.log_prob(pz).sum(-1),zero),-1)
            entropy=torch.stack((distribution.entropy().sum(-1),zero),-1)
            return action,torch.cat((pz,gates[...,None]),-1),logp,entropy,None,None
        prior=.02+.96*raw_prior
        context=self.context(packet,proposals,prior)
        gm=torch.logit(self.adapter(context).clamp(1e-5,1.-1e-5))
        if self.graph:
            # Retain the full joint distribution. Averaging independent gates
            # would discard mutually exclusive, coordinated combinations.
            augmented=self.adaptive_fusion_grid in ('augmented','risk_stationary') and self.interaction_enabled
            k=self.levels+int(augmented);indices=torch.arange(k**n,device=obs.device)
            choices=((indices[:,None]//(k**torch.arange(n,device=obs.device)))%k).long()
            bits=choices.to(obs.dtype)/(k-1)
            unary=2.*torch.tanh(gm/2.)
            adaptive=bool(self.adaptive_fusion_grid and self.interaction_enabled)
            candidate_grid=None;physical_plans=bits
            if adaptive:
                if self.levels!=3 or self.intent_strength or latent is not None or self.forecast_inputs:
                    raise ValueError('Source-centered candidates use the frozen three-candidate joint controller.')
                # Keep two pure policies and the current source preference.
                # A held intervention replaces the middle candidate only for
                # that boat, so free boats predict its actual pending motion.
                middle=unary.sigmoid()
                if self.adaptive_fusion_grid=='risk_stationary':
                    middle=self.stationary_fusion_candidates(packet,proposals,middle)
                held=packet.get('_committed_gates')
                if held is not None:middle=torch.where(held>=0.,held,middle)
                entries=(torch.zeros_like(middle),middle,torch.ones_like(middle))
                if augmented:entries=(entries[0],torch.full_like(middle,.5),middle,entries[-1])
                candidate_grid=torch.stack(entries,-1)
                physical_plans=torch.stack([candidate_grid[:,i,choices[:,i]] for i in range(n)],-1)
            scores=(torch.einsum('bn,kn->bk',unary,bits) if k==2 else
                self.fusion_prior_scores(packet,proposals,unary.sigmoid(),physical_plans,.3 if self.intent_strength else .4))
            preference=packet.get('_option_preference')
            if preference is not None and self.preference_strength:
                selected=preference[:,None]>=0.
                scores=scores-self.preference_strength*(
                    ((physical_plans if adaptive else bits[None])-preference[:,None]).square()*selected).sum(-1)
            complete_response=(self.coupled_shortlist>=len(bits) and not self.forecast_inputs
                and self.risk_prior and not self.intent_strength)
            risk=None if adaptive else packet.get('_candidate_risk')
            if risk is None and (self.risk_prior or self.forecast_inputs) and not complete_response:
                risk=self.candidate_risk(packet,proposals,candidate_grid=candidate_grid)
            pair_context=context['edge' if self.peer else 'edge_no_peer']
            if self.forecast_inputs:
                forecast=(risk if self.peer else risk[:,:,None,:,None].expand(-1,-1,n,-1,k))
                pair_context=torch.cat((pair_context,forecast.flatten(-2).clamp_max(36.)/9.),-1)
            if self.peer:
                factors=self.gate_pairs(pair_context).reshape(b,n,n,self.levels,self.levels)
                def pair_scores(i,j):
                    if not adaptive:return factors[:,i,j,choices[:,i],choices[:,j]]
                    # Interpolate any learned pair factor at the actual source
                    # candidate. The physical prediction uses that candidate
                    # directly, without replacing the original grid points.
                    x=physical_plans[:,:,i]*(self.levels-1)
                    y=physical_plans[:,:,j]*(self.levels-1)
                    lo_x=x.floor().long();hi_x=(lo_x+1).clamp_max(self.levels-1)
                    lo_y=y.floor().long();hi_y=(lo_y+1).clamp_max(self.levels-1)
                    wx=x-lo_x;wy=y-lo_y;batch=torch.arange(b,device=obs.device)[:,None]
                    return (factors[batch,i,j,lo_x,lo_y]*(1.-wx)*(1.-wy)
                        +factors[batch,i,j,hi_x,lo_y]*wx*(1.-wy)
                        +factors[batch,i,j,lo_x,hi_y]*(1.-wx)*wy
                        +factors[batch,i,j,hi_x,hi_y]*wx*wy)
                for i in range(n):
                    for j in range(i+1,n):
                        pair=pair_scores(i,j)
                        # Preserve the older binary prototypes on reload. The
                        # mixed-action graph symmetrizes both receiver frames.
                        if k>2:pair=.5*(pair+pair_scores(j,i))
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
                if not complete_response:
                    scores=self.risk_scores(scores,risk,choices,packet.get('_candidate_interception'),packet.get('_terminal_paths'))
                if self.coupled_shortlist:scores=self.refine_joint_scores(packet,proposals,physical_plans,scores,base_scores)
            if self.local_value is not None:
                from conditional_fusion_value import local_fusion_context
                for i in range(n):
                    for j in range(i+1,n):
                        context=local_fusion_context(packet,proposals,i,j,self.peer)
                        factor=self.local_value(context)
                        # Conditional value learning remains joint in the
                        # no-peer-input ablation; only its candidate inputs
                        # and the first mechanism's predictor are removed.
                        scores=scores+factor[:,choices[:,i],choices[:,j]]
            committed=packet.get('_committed_gates')
            if committed is not None and self.interaction_enabled:
                # Free boats must plan against the gates that their partners
                # actually committed to, not a fictitious freely reselected
                # joint plan which is subsequently overwritten by the option.
                feasible=((committed[:,None]<0.)|(((physical_plans if adaptive else bits[None])-committed[:,None]).abs()<1e-5)).all(-1)
                if not bool(feasible.any(-1).all()):
                    raise ValueError('Committed gates must be represented in the joint fusion grid.')
                scores=scores.masked_fill(~feasible,-torch.finfo(scores.dtype).max)
            dist=torch.distributions.Categorical(logits=scores)
            if latent is None:choice=scores.argmax(-1) if deterministic else dist.sample()
            else:choice=((latent[...,2]*(k-1)).round().long()*(k**torch.arange(n,device=obs.device))).sum(-1)
            gates=physical_plans[torch.arange(b,device=obs.device),choice] if adaptive else bits[choice]
            if not self.interaction_enabled:
                # Complete M1 removal also removes its discrete joint gate
                # selection. Restore the original continuous scalar Adapter;
                # M2 may still learn interventions on this genuine baseline.
                gates=raw_prior
                if getattr(self,'continuous_gate_learning',False):
                    gate_mean=torch.logit(raw_prior.clamp(1e-5,1.-1e-5))
                    gate_std=self.gate_logstd.clamp(-4.,-.3).exp()
                    gate_raw=(gate_mean if deterministic else gate_mean+gate_std*torch.randn_like(gate_mean))
                    if latent is not None:gate_raw=torch.logit(latent[...,2].clamp(1e-5,1.-1e-5))
                    gates=raw_prior if deterministic and latent is None else gate_raw.sigmoid()
                if preference is not None and self.preference_strength:
                    # With M1 removed, retain only M2's learned temporal
                    # preference and the original continuous scalar Adapter.
                    precision=self.preference_strength*(.3 if self.intent_strength else .4)**2
                    regularized=(raw_prior+precision*preference)/(1.+precision)
                    gates=torch.where(preference>=0.,regularized,raw_prior)
            logp=torch.stack((torch.distributions.Normal(pm,ps).log_prob(pz).sum(-1),
                dist.log_prob(choice)[:,None].expand(-1,n)),-1)
            entropy=torch.stack((torch.distributions.Normal(pm,ps).entropy().sum(-1),
                (dist.entropy()/n)[:,None].expand(-1,n)),-1)
            if not self.interaction_enabled and getattr(self,'continuous_gate_learning',False):
                density=torch.distributions.Normal(gate_mean,gate_std).log_prob(gate_raw)
                density=density-torch.log(gates.clamp_min(1e-7))-torch.log((1.-gates).clamp_min(1e-7))
                logp=torch.stack((logp[...,0],density),-1)
                entropy=torch.stack((entropy[...,0],-density),-1)
            action=torch.cat((proposals,gates[...,None]),-1)
            if gate_only:
                return action,torch.cat((pz,gates[...,None]),-1),logp,entropy,scores,bits
            if self.physical_value is not None or len(self.physical_values):
                from conditional_fusion_value import local_fusion_context, response_fusion_context
                option_indices=torch.arange(4**n,device=obs.device)
                options=((option_indices[:,None]//(4**torch.arange(n,device=obs.device)))%4).long()
                context={(i,j):local_fusion_context(packet,proposals,i,j,self.peer)
                    for i in range(n) for j in range(i+1,n)}
                predictors=list(zip(self.physical_values,self.physical_spans,self.physical_reversals))
                if self.physical_value is not None:predictors.insert(0,(self.physical_value,self.physical_span,self.physical_reversal))
                scores_by_span=[]
                for predictor,span,reversal in predictors:
                    option_scores=-self.physical_cost*(options>0).sum(-1)[None].expand(b,-1).to(obs.dtype)
                    for (i,j),inputs in context.items():
                        if predictor.encoder=='response':
                            inputs=response_fusion_context(packet,proposals,risk,i,j,self.peer)
                        factor=predictor(inputs)
                        option_scores=option_scores+factor[:,options[:,i],options[:,j]]
                    scores_by_span.append(option_scores)
                selected_index=torch.stack(scores_by_span,1).flatten(1).argmax(-1)
                selected=options[selected_index%len(options)]
                spans=obs.new_tensor([span for _,span,_ in predictors])[selected_index//len(options)]
                reversal=obs.new_tensor([float(rev) for _,_,rev in predictors])[selected_index//len(options)]
                physical=torch.where(selected>0,(selected.to(obs.dtype)-1.)*.5,-1.)
                second=torch.where((selected>0)&(reversal[:,None]>0.),1.-physical,physical)
                action=torch.cat((action,physical[...,None],spans[:,None,None].expand(-1,n,-1),second[...,None]),-1)
            if self.response_base is not None:
                action=self.response_base(packet,deterministic=True)[0]
            graph_value=getattr(self,'response_aggregation','additive')=='graph'
            response_factual=response_support=None;response_active=True
            if len(self.response_values) and self.response_anchor is not None:
                from conditional_fusion_value import public_return_context
                response_factual,response_support=self.response_anchor(public_return_context(packet),set_context=graph_value)
                # The bounded success value cannot improve by more than this.
                # Rejecting the entire intervention set here is exact, before
                # constructing costly candidate-dependent response features.
                response_active=bool(((1.-response_factual)*response_support>self.response_margin).any())
            if (len(self.response_values) and response_active and self.response_action_support is not None
                    and self.response_action_support.transport):
                from conditional_fusion_value import public_return_context
                response_context=public_return_context(packet)
                maximum=torch.stack([self.response_action_support.support_upper_bound(response_context,span)
                    for span in getattr(self,'response_spans',(self.response_span,))]).amax(0)
                response_active=bool((maximum>self.response_margin).any())
            if len(self.response_values) and response_active:
                from conditional_fusion_value import fusion_response_factors,intervention_programs
                if action.shape[-1]==3:
                    action=torch.cat((action,torch.full_like(action[...,:1],-1.),
                        torch.full_like(action[...,:1],self.physical_span),torch.full_like(action[...,:1],-1.)),-1)
                responses=[];gains=[]
                for span in getattr(self,'response_spans',(self.response_span,)):
                    features=fusion_response_factors(self,packet,action,span)
                    tables=torch.stack([value(features) for value in self.response_values])
                    if graph_value:
                        options=intervention_programs(action)
                        if self.response_action_support is not None:
                            supported=self.response_action_support.candidate_programs(packet,span)
                            if supported is not None and len(supported):options=torch.cat((options,supported)).unique(dim=0)
                    else:
                        indices=torch.arange(4**n,device=obs.device)
                        options=((indices[:,None]//(4**torch.arange(n,device=obs.device)))%4).long()
                    if getattr(self,'response_aggregation','additive') in ('joint','graph'):
                        pairs=torch.stack([options[:,i]*4+options[:,j] for i in range(n) for j in range(i+1,n)],-1)
                        indices=pairs[None].expand(b,-1,-1)
                        prediction=torch.stack([value.joint_actions(table,indices) for value,table in zip(self.response_values,tables)])
                        if self.response_heads==1:prediction=prediction.squeeze(-1)
                    else:
                        prediction=obs.new_zeros((len(self.response_values),b,len(options))+
                            (() if self.response_heads==1 else (self.response_heads,)))
                        bottleneck=[]
                        pair=0
                        for i in range(n):
                            for j in range(i+1,n):
                                selected_pair=tables[:,:,pair,options[:,i],options[:,j]]
                                prediction+=selected_pair;bottleneck.append(selected_pair);pair+=1
                        prediction/=np.sqrt(pair)
                        if getattr(self,'response_aggregation','additive')=='bottleneck':
                            prediction=-torch.logsumexp(-torch.stack(bottleneck,-1),-1)+np.log(pair)
                    states=torch.stack([value.state_value(features) for value in self.response_values])
                    probability=prediction+(states[...,None] if self.response_heads==1 else states[:,:,None])
                    if self.response_paired_gain:probability=prediction
                    elif getattr(self,'response_link','sigmoid')=='sigmoid':probability=probability.sigmoid()
                    else:probability=probability.clamp(0.,1.)
                    if self.response_heads==2:
                        factual=response_factual
                        probability=(1.-factual[None,:,None])*probability[...,0]+factual[None,:,None]*probability[...,1]
                    baseline=(torch.where(action[...,3]>=0.,1.+2.*action[...,3],0.).long()
                        if action.shape[-1]>3 else torch.zeros((b,n),dtype=torch.long,device=obs.device))
                    reference=((options[None]==baseline[:,None]).all(-1).long().argmax(-1) if graph_value else
                        (baseline*(4**torch.arange(n,device=obs.device))).sum(-1))
                    advantage=probability-probability.gather(2,reference[None,:,None].expand(len(self.response_values),-1,-1))
                    if self.response_paired_gain:advantage=advantage.clamp(-1.,1.)
                    if (getattr(self,'response_duration_reference',False) and self.response_anchor is not None
                            and not self.response_paired_gain):
                        factual=response_factual
                        matched=(action[...,3]<0.).all(-1)|(action[:,0,4]==span)
                        # Projecting a two-second reference onto a short pulse
                        # changes the reference intervention itself. Compare
                        # with its observed continuation value in that case.
                        advantage=torch.where(matched[None,:,None],advantage,probability-factual[None,:,None])
                    lower=advantage.mean(0)-advantage.std(0,unbiased=False)
                    if self.response_anchor is not None:
                        factual,support=response_factual,response_support
                        ceiling=(1.-factual[:,None] if self.response_paired_gain else probability.mean(0)-factual[:,None])
                        lower=torch.minimum(lower,ceiling)*support[:,None]
                    if self.response_effect_support is not None:
                        from conditional_fusion_value import public_return_context
                        inputs=public_return_context(packet)
                        inputs=torch.cat((inputs,inputs.new_full((b,1),span/10.)),-1)
                        lower=torch.minimum(lower,self.response_effect_support(inputs)[:,None])
                    if self.response_action_support is not None:
                        lower=torch.minimum(lower,self.response_action_support(packet,options,span))
                    changed=(options[None]!=baseline[:,None]).sum(-1)
                    gain=lower-.005*changed
                    choice=gain.argmax(-1);use=gain.gather(1,choice[:,None]).squeeze(1)>self.response_margin
                    selected=options[choice]
                    physical=torch.where(selected>0,(selected.to(obs.dtype)-1.)*.5,-1.)
                    replacement=torch.cat((action[...,:3],physical[...,None],
                        torch.full_like(physical[...,None],span),physical[...,None]),-1)
                    responses.append(replacement)
                    gains.append(gain.gather(1,choice[:,None]).squeeze(1))
                gain=torch.stack(gains,1);choice=gain.argmax(-1)
                replacement=torch.stack(responses,1)[torch.arange(b,device=obs.device),choice]
                use=gain.gather(1,choice[:,None]).squeeze(1)>self.response_margin
                action=torch.where(use[:,None,None],replacement,action)
            if self.joint_intervention_value is not None:
                action=self.joint_intervention_value(packet,action)
            if self.initial_role_credit is not None:
                action=self.initial_role_credit(packet,action)
            if self.feedback_role_credit is not None:
                action=self.feedback_role_credit(packet,action)
            if self.cooperative_role_credit is not None and bool(self.cooperative_role_credit.eligible(packet).any()):
                if self.interaction_enabled and self.peer and risk is not None:
                    packet['_joint_candidate_risk']=risk
                if self.cooperative_role_credit.response_features:
                    response=self.conditional_response_features(
                        packet,action[...,:3],proposal=self.cooperative_role_credit.proposal,
                        all_pairs=self.cooperative_role_credit.all_pairs,
                        return_decisions=self.cooperative_role_credit.task_frame)
                    if self.cooperative_role_credit.task_frame:
                        packet['_response_features'],packet['_response_gates']=response
                    else:packet['_response_features']=response
                action=self.cooperative_role_credit(packet,action)
            if len(self.joint_intervention_values):
                action=self.choose_joint_value_option(packet,action)
            if getattr(self,'finite_temporal_credit',None) is not None:
                action=self.finite_temporal_credit(packet,action)
            if self.continuous_program_value is not None:
                action=self.continuous_program_value(packet,action)
            if self.successor_task_value is not None:
                action=self.task_decision(packet,action)
            return action,torch.cat((pz,gates[...,None]),-1),logp,entropy,scores,bits
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
            decoded=False,levels=3,intent_strength=0.,invariant=False,temporal_option=False):
        super().__init__();self.response=response;self.options=options;self.relational=relational;self.intents=intents
        self.probability=probability
        self.decoded=decoded;self.levels=levels;self.intent_strength=intent_strength
        self.invariant=invariant
        self.temporal_option=temporal_option
        self.node=nn.Sequential(nn.Linear(14,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.pair=nn.Sequential(nn.Linear(9,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.value=nn.Sequential(nn.Linear(width*4+9,192),nn.SiLU(),nn.Linear(192,1))
        self.control_node=nn.Sequential(nn.Linear((21 if response else 16)+(5 if options else 0)+(3 if decoded else 0)+(1 if temporal_option else 0),width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.control_pair=nn.Sequential(nn.Linear((16 if response else 13)+(10 if options else 0)+(7 if decoded else 0)+(2 if temporal_option else 0),width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
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
        if getattr(self,'temporal_option',False):
            intervention=packet.get('_intervened',action.new_zeros(action.shape[:2]))[...,None]
            ni.append(intervention)
            ei.extend((intervention[:,:,None].expand(-1,-1,n,-1),intervention[:,None].expand(-1,n,-1,-1)))
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


def joint_task_step(env,action,gamma=.995,step_cost=.001,attacker_action=None):
    # Common training objective for every new-method ablation. Author dynamics,
    # rewards and reported terminal classes remain untouched.
    potential=-float(env.def_att_dists.min())/60.
    if attacker_action is None:
        obs,source_reward,done,_=env.step(action,'AdaRes')
    else:
        obs,source_reward,done,_=env.step(action,'AdaRes',att_action=attacker_action)
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


def ablate_joint_mechanisms(actor, without_peer=False, without_conditional=False, without_interaction=False):
    """Remove mechanism paths from the executing model, including ancestors."""
    if without_interaction:
        without_peer=True
        actor.interaction_enabled=False
        actor.risk_prior=0.;actor.risk_feedback=False;actor.coupled_shortlist=0
        actor.risk_tail=0.;actor.risk_barrier=0.;actor.interception_weight=0.
        actor.interception_horizon=0.;actor.prediction_discount=0.
        actor.target_motion=None
        actor.terminal_prediction=False;actor.capture_credit=0.
        actor.conditional_commitment=False
        actor.immediate_commitment=False
        actor.pair_scale=0.
        actor.prediction_relaxation=0.
        actor.coupled_robust_prediction=False
        actor.adaptive_fusion_grid=False
        actor.prediction_iterations=0
        actor.current_marginal_prediction=False
        actor.pair_feedback_prediction=False
        actor.physical_prior_weight=0.
        actor.actuator_fusion_refinement=None
        actor.generic_fusion_adapter=None
        if actor.response_feedback_value is not None:
            for response in actor.response_feedback_value.modules():
                if hasattr(response,'joint_response'):response.joint_response=False
    if without_peer:
        actor.peer=False
        actor.adapter.peer=False
        for value in (actor.local_value,actor.physical_value,*actor.physical_values):
            while value is not None:
                value.peer=False
                if value.encoder!='state':
                    raise ValueError('A response-encoded value must be refitted without peer responses before ablation.')
                # Stored candidate coordinates are part of the representation,
                # too; masking only the live query leaves a hidden metric term.
                value.centers[...,22:26]=0.
                value=value.prior
    if actor.response_base is not None:
        ablate_joint_mechanisms(actor.response_base,without_peer,without_conditional,without_interaction)
    if without_conditional:
        actor.conditional_enabled=False
        # A checkpoint with a learned gate head cannot be converted into a
        # training ablation by just removing the explicit value modules.
        heads=[actor.adapter.gate[-1]]
        if actor.graph:heads.append(actor.gate_pairs[-1])
        ordinary=getattr(actor,'trained_without_conditional',False)
        if not ordinary and any(torch.count_nonzero(p).item() for head in heads for p in head.parameters()):
            raise ValueError('Conditional learning is embedded in the gate weights; use a fresh ordinary-learning initialization.')
        initialized=getattr(actor,'proposal_initialization_sha256',None)
        changed=(proposal_parameter_digest(actor.base.state_dict())!=initialized if initialized else
            BASE is not None and any(not torch.equal(value,BASE.actor.state_dict()[key])
                for key,value in actor.base.state_dict().items()))
        if changed and not ordinary:
            raise ValueError('A conditional ablation requires the unchanged source proposal and fusion prior.')
        actor.local_value=None;actor.physical_value=None
        actor.physical_values=nn.ModuleList()
        actor.response_values=nn.ModuleList();actor.response_base=None
        actor.response_anchor=None;actor.response_effect_support=None;actor.response_action_support=None
        actor.conditional_commitment=False
        actor.immediate_commitment=False
        actor.preference_strength=0.
        actor.initial_role_credit=None
        actor.feedback_role_credit=None
        actor.cooperative_role_credit=None
        actor.finite_temporal_credit=None
        actor.continuous_program_value=None
        actor.successor_task_value=None
        actor.response_feedback_value=None
        actor.joint_intervention_value=None;actor.joint_intervention_values=nn.ModuleList()
        actor.midpoint_intervention_values=None
        actor.option_refinement_fraction=.5
        actor.option_refinement_steps=0
        actor.option_refinement_checkpoints=False
        actor.phase_preserving_refinement=False
        actor.fixed_intervention_clock=False
    return actor


def joint_interaction_identity(model):
    """Execution settings of the predecessor that generated a branch outcome."""
    removal=tuple(model.get('mechanism_ablation',(False,False,False)))+(False,)*3
    if removal[2]:return {'interaction_removed':True}
    defaults=dict(peer=True,graph=False,levels=2,risk_prior=0.,risk_horizon=1.6,
        risk_clearance=6.,risk_feedback=False,coupled_shortlist=0,pair_scale=1.,
        fusion_selection='joint',fusion_expression='interaction',adaptive_fusion_grid=False,
        prediction_step=.1,prediction_relaxation=0.,current_marginal_prediction=False,coupled_robust_prediction=False,
        prediction_iterations=0,pair_feedback_prediction=False,prediction_discount=0.,
        prediction_discount_start=1.6,risk_tail=0.,risk_barrier=0.,
        physical_prior_weight=0.,terminal_prediction=False,capture_credit=0.,
        interception_weight=0.,interception_horizon=0.,actuator_fusion_refinement=None)
    result={key:model.get(key,default) for key,default in defaults.items()}
    if model.get('generic_fusion_adapter') is not None:
        record=model['generic_fusion_adapter'];digest=hashlib.sha256()
        for key,value in sorted(record.items()):
            digest.update(key.encode())
            if key=='state':
                for name,tensor in sorted(value.items()):
                    digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
            else:digest.update(str(value).encode())
        result['generic_fusion_adapter']=digest.hexdigest()
    if removal[0]:result['peer']=False
    return result


def joint_continuation_identity(model):
    """Hash executing controls, excluding collection and fit bookkeeping."""
    digest=hashlib.sha256()
    def add(value):
        if torch.is_tensor(value):value=value.detach().cpu().numpy()
        if isinstance(value,np.ndarray):
            digest.update(str((value.shape,value.dtype)).encode());digest.update(value.tobytes())
        elif isinstance(value,dict):
            for key in sorted(value):
                if key=='learning_objective':continue
                digest.update(str(key).encode());add(value[key])
        elif isinstance(value,(tuple,list)):
            digest.update(str(len(value)).encode())
            for child in value:add(child)
        else:digest.update(repr(value).encode())
    add(joint_interaction_identity(model));add(model.get('actor'))
    removal=tuple(model.get('mechanism_ablation',(False,False,False)))+(False,)*3
    for key in ('joint_intervention_value','joint_intervention_values','midpoint_intervention_values',
                'local_conditional_value','physical_conditional_value','physical_conditional_values',
                'shared_response_value','factual_return_support','conditional_effect_support',
                'response_action_support','response_base','initial_role_credit','feedback_role_credit',
                'cooperative_role_credit','finite_temporal_credit','continuous_program_value','response_feedback_value','successor_task_value'):
        value=None if removal[1] else model.get(key)
        add(None if isinstance(value,(dict,list,tuple)) and not len(value) else value)
    for key,default in (('conditional_commitment',False),('immediate_commitment',False),
                        ('fixed_intervention_clock',False),('preference_strength',0.),
                        ('option_refinement_fraction',.5),('option_refinement_steps',0),
                        ('option_refinement_checkpoints',False),('phase_preserving_refinement',False),
                        ('continuous_gate_learning',False)):
        add(model.get(key,default))
    return digest.hexdigest()


def joint_value_objectives(model):
    """Identify and preserve the objective of each retained value/proposer."""
    objectives=set()
    def visit(record,fallback=None):
        if isinstance(record,(list,tuple)):
            for child in record:visit(child,fallback)
        elif isinstance(record,dict):
            objective=record.get('learning_objective',fallback)
            if objective is None and record.get('kind') in ('plain_joint_return','plain_mlp_return'):
                objective='absolute'
            if objective is None and (record.get('conditional_strata') or record.get('paired_advantage')):
                objective='conditional'
            objectives.add(objective or 'unclassified')
            if 'contexts' in record and 'values' in record:
                record.setdefault('learning_objective',objective or 'unclassified')
            if record.get('prior') is not None:visit(record['prior'])
    fit=model.get('joint_intervention_fit',{})
    fallback=('conditional' if fit['conditional'] else 'absolute') if 'conditional' in fit else None
    for key in ('joint_intervention_value','joint_intervention_values'):
        if model.get(key) is not None:visit(model[key],fallback)
    if model.get('midpoint_intervention_values'):visit(model['midpoint_intervention_values'])
    if model.get('program_proposer',{}).get('values'):visit(model['program_proposer']['values'])
    if model.get('successor_task_value') is not None:visit(model['successor_task_value'])
    for key in ('local_conditional_value','physical_conditional_value','physical_conditional_values',
                'conditional_effect_support','response_base','initial_role_credit','feedback_role_credit',
                'cooperative_role_credit','finite_temporal_credit','continuous_program_value','response_feedback_value'):
        if model.get(key) is not None:objectives.add('conditional')
    if model.get('shared_response_value'):
        objectives.add('conditional' if model['shared_response_value'].get('conditional',True) else 'absolute')
    return objectives


def require_clean_mechanism_start(model,args):
    """Keep removed learning signals out of the ablation's initialization."""
    inherited=[key for key in ('local_conditional_value','physical_conditional_value',
        'physical_conditional_values','joint_intervention_value','joint_intervention_values',
        'conditional_effect_support','response_base','initial_role_credit','feedback_role_credit',
        'cooperative_role_credit','finite_temporal_credit','continuous_program_value','response_feedback_value','successor_task_value',
        'midpoint_intervention_values') if model.get(key) is not None]
    shared=model.get('shared_response_value')
    if shared and shared.get('conditional',True):inherited.append('shared_response_value')
    if args.joint_no_cf and inherited:
        raise ValueError('Ordinary learning cannot inherit conditional modules: '+', '.join(inherited))
    if args.without_peer and model.get('peer',True) and inherited:
        raise ValueError('Retrain the no-peer model from the common initialization, not an interaction-trained ancestor.')
    if args.joint_no_interaction:
        model.update(peer=False,risk_prior=0.,risk_feedback=False,coupled_shortlist=0,
            pair_scale=0.,mechanism_ablation=(True,False,True))
    # A value-fitting entry point installs a new learner. An F00/F10 execution
    # ablation must not silently delete that learner when the checkpoint reloads.
    # Preserve the separately selected removal of candidate interactions.
    removal=list(model.get('mechanism_ablation',(False,False,False)))
    removal+=(3-len(removal))*[False]
    removal[1]=False
    model['mechanism_ablation']=tuple(removal)


@lru_cache(maxsize=6)
def joint_load(path,revision,ablation=(False,False,False)):
    data=torch.load(path,map_location='cpu',weights_only=False)
    for label,flags in (('checkpoint',data.get('mechanism_ablation',(False,False,False))),
                        ('evaluation',EVALUATION_ABLATION),('requested',ablation)):
        if (not isinstance(flags,(tuple,list)) or len(flags)>3
                or any(not isinstance(flag,(bool,np.bool_)) for flag in flags)):
            raise ValueError(f'{label} mechanism_ablation must contain only Boolean removal flags.')
    if (data.get('successor_task_value') or {}).get('kind')=='plain_mlp_return':
        from plain_return import require_plain_predecessor
        require_plain_predecessor(data,allow_plain_return=True)
    actor=JointLearningActor(data['peer'],graph=data.get('graph',False),levels=data.get('levels',2),
        risk_prior=data.get('risk_prior',0.),risk_horizon=data.get('risk_horizon',1.6),
        pair_scale=data.get('pair_scale',1.),gate_period=data.get('gate_period',1),
        risk_clearance=data.get('risk_clearance',6.),forecast_inputs=data.get('forecast_inputs',False),
        frozen_proposal=data.get('frozen_proposal',False),risk_feedback=data.get('risk_feedback',False),
        intent_strength=data.get('intent_strength',0.),coupled_shortlist=data.get('coupled_shortlist',0),
        prediction_step=data.get('prediction_step',.1),risk_tail=data.get('risk_tail',0.),
        risk_barrier=data.get('risk_barrier',0.),interception_weight=data.get('interception_weight',0.))
    actor.load_state_dict(data['actor']);actor.eval()
    actor.environment_protocol=data.get('environment_protocol','source')
    actor.proposal_initialization_sha256=data.get('proposal_initialization_sha256')
    actor.continuous_gate_learning=data.get('continuous_gate_learning',False)
    actor.trained_without_conditional=(data.get('conditional_training') is False
        and data.get('fusion_learning')=='paired-conditional-soft-policy-v1')
    actor.conditional_commitment=data.get('conditional_commitment',False)
    actor.immediate_commitment=data.get('immediate_commitment',False)
    actor.fixed_intervention_clock=data.get('fixed_intervention_clock',False)
    actor.fusion_selection=data.get('fusion_selection','joint')
    actor.fusion_expression=data.get('fusion_expression','interaction')
    actor.prediction_relaxation=data.get('prediction_relaxation',0.)
    actor.coupled_robust_prediction=data.get('coupled_robust_prediction',False)
    actor.adaptive_fusion_grid=data.get('adaptive_fusion_grid',False)
    actor.prediction_iterations=data.get('prediction_iterations',0)
    actor.current_marginal_prediction=data.get('current_marginal_prediction',False)
    actor.pair_feedback_prediction=data.get('pair_feedback_prediction',False)
    actor.physical_prior_weight=data.get('physical_prior_weight',0.)
    actor.actuator_fusion_refinement=data.get('actuator_fusion_refinement')
    if data.get('generic_fusion_adapter') is not None:
        from mechanism_controls import GenericFusionAdapter
        if actor.actuator_fusion_refinement is not None:
            raise ValueError('A generic Adapter cannot be followed by the analytic fusion selector.')
        actor.generic_fusion_adapter=GenericFusionAdapter(**data['generic_fusion_adapter']).eval()
    actor.interception_horizon=data.get('interception_horizon',0.)
    actor.prediction_discount=data.get('prediction_discount',0.)
    actor.prediction_discount_start=data.get('prediction_discount_start',1.6)
    if data.get('finite_temporal_credit') is not None:
        from conditional_fusion_value import FiniteTemporalCredit
        actor.finite_temporal_credit=FiniteTemporalCredit(**data['finite_temporal_credit'])
    if data.get('continuous_program_value') is not None:
        from conditional_fusion_value import ContinuousProgramValue
        actor.continuous_program_value=ContinuousProgramValue(**data['continuous_program_value'])
    if data.get('successor_task_value') is not None:
        from conditional_fusion_value import OrdinaryJointReturnDecision,SuccessorTaskDecision
        record=dict(data['successor_task_value'])
        kind=record.pop('kind','conditional_successor')
        if kind=='local_difference_joint':
            from local_intervention_value import LocalDifferenceDecision
            actor.successor_task_value=LocalDifferenceDecision(**record)
        elif kind=='plain_joint_return':
            from local_intervention_value import PlainJointReturnDecision
            actor.successor_task_value=PlainJointReturnDecision(**record)
        elif kind=='plain_mlp_return':
            from plain_return import PlainReturnDecision
            actor.successor_task_value=PlainReturnDecision(**record)
        elif kind=='shared_paired_effect':
            from conditional_fusion_value import SharedPairedEffectDecision
            actor.successor_task_value=SharedPairedEffectDecision(**record)
        elif kind=='paired_option_task':
            from conditional_fusion_value import PairedOptionTaskDecision
            actor.successor_task_value=PairedOptionTaskDecision(**record)
        elif kind=='ordinary_joint_return':
            actor.successor_task_value=OrdinaryJointReturnDecision(**record)
        elif kind=='coma_joint_return':
            from mechanism_controls import COMAReturnDecision
            actor.successor_task_value=COMAReturnDecision(**record)
        elif kind=='conditional_successor':
            actor.successor_task_value=SuccessorTaskDecision(**record)
        else:
            raise ValueError(f'Unknown task-value execution model: {kind}')
    if data.get('response_feedback_value') is not None:
        from conditional_fusion_value import ResponseFeedbackValue
        actor.response_feedback_value=ResponseFeedbackValue(**data['response_feedback_value'])
    if data.get('feedback_fleet_policy') is not None:
        from conditional_fusion_value import FeedbackFleetPolicy
        if not (actor.actuator_fusion_refinement or {}).get('task_feedback_prediction'):
            raise ValueError('A learned fleet policy requires complete candidate feedback forecasts.')
        actor.feedback_fleet_policy=FeedbackFleetPolicy(**data['feedback_fleet_policy']).eval()
    actor.preference_strength=data.get('preference_strength',0.)
    actor.terminal_prediction=data.get('terminal_prediction',False)
    actor.capture_credit=data.get('capture_credit',0.)
    if data.get('target_motion') is not None:
        record=data['target_motion'];actor.target_motion=PublicTargetMotion(record.get('width',48),record.get('absolute_frame',False))
        actor.target_motion.load_state_dict(record['state']);actor.target_motion.eval()
    if data.get('target_feedback') is not None:
        actor.target_feedback=PublicTargetFeedback(**data['target_feedback']).eval()
    if data.get('local_conditional_value') is not None:
        from conditional_fusion_value import LocalConditionalValue
        actor.local_value=LocalConditionalValue(**data['local_conditional_value'])
    if data.get('physical_conditional_value') is not None:
        from conditional_fusion_value import LocalConditionalValue
        actor.physical_value=LocalConditionalValue(**data['physical_conditional_value'])
    actor.physical_cost=data.get('physical_cost',.02)
    if data.get('response_base') is not None:
        parent=Path(data['response_base'])
        actor.response_base=copy.deepcopy(joint_load(str(parent),parent.stat().st_mtime_ns))
    actor.physical_span=data.get('physical_span',actor.gate_period)
    actor.physical_reversal=data.get('physical_reversal',False)
    for record in data.get('physical_conditional_values',[]):
        from conditional_fusion_value import LocalConditionalValue
        actor.physical_values.append(LocalConditionalValue(**record['value']))
        actor.physical_spans.append(record['span'])
        actor.physical_reversals.append(record.get('reversal',False))
    actor.physical_intervention_span=data.get('physical_intervention_span',actor.physical_span)
    if data.get('shared_response_value') is not None:
        from conditional_fusion_value import SharedResponseValue
        record=data['shared_response_value']
        for state in record['states']:
            value=SharedResponseValue(record['features'],record['width'],record.get('heads',1),record.get('aggregation','bottleneck'))
            value.load_state_dict(state);actor.response_values.append(value.eval())
        actor.response_span=record['span'];actor.response_margin=record['margin']
        actor.response_heads=record.get('heads',1)
        actor.response_paired_gain=record.get('paired_gain',False)
        actor.response_feature_version=record.get('feature_version',1)
        actor.response_aggregation=record.get('aggregation','additive')
        actor.response_link=record.get('link','sigmoid')
        actor.response_spans=record.get('spans',(record['span'],))
        actor.response_duration_reference=record.get('duration_reference',False)
    if data.get('factual_return_support') is not None:
        from conditional_fusion_value import FactualReturnSupport
        actor.response_anchor=FactualReturnSupport(**data['factual_return_support'])
    if data.get('conditional_effect_support') is not None:
        from conditional_fusion_value import ConditionalEffectSupport
        actor.response_effect_support=ConditionalEffectSupport(**data['conditional_effect_support'])
    if data.get('response_action_support') is not None:
        from conditional_fusion_value import ResponseActionSupport
        actor.response_action_support=ResponseActionSupport(**data['response_action_support'])
    if data.get('joint_intervention_value') is not None:
        from conditional_fusion_value import JointInterventionValue
        actor.joint_intervention_value=JointInterventionValue(**data['joint_intervention_value'])
    if data.get('initial_role_credit') is not None:
        from conditional_fusion_value import InitialRoleCredit
        actor.initial_role_credit=InitialRoleCredit(**data['initial_role_credit'])
    if data.get('feedback_role_credit') is not None:
        from conditional_fusion_value import FeedbackRoleCredit
        actor.feedback_role_credit=FeedbackRoleCredit(**data['feedback_role_credit'])
    if data.get('cooperative_role_credit') is not None:
        from conditional_fusion_value import CooperativeRoleCredit,LocalRoleEffectCredit
        credit=data['cooperative_role_credit']
        constructor=LocalRoleEffectCredit if credit.get('estimator')=='local-conditional-effects' else CooperativeRoleCredit
        actor.cooperative_role_credit=constructor(**credit)
    if data.get('joint_intervention_values'):
        from conditional_fusion_value import JointInterventionOptions
        options=JointInterventionOptions(data['joint_intervention_values'])
        actor.joint_intervention_values=options.values
        actor.shared_intervention_prior=options.shared_prior
    if data.get('midpoint_intervention_values'):
        from conditional_fusion_value import JointInterventionOptions
        actor.midpoint_intervention_values=JointInterventionOptions(data['midpoint_intervention_values'])
        actor.option_refinement_fraction=data.get('option_refinement_fraction',.5)
        actor.option_refinement_steps=data.get('option_refinement_steps',0)
        actor.option_refinement_checkpoints=data.get('option_refinement_checkpoints',False)
        actor.phase_preserving_refinement=data.get('phase_preserving_refinement',False)
    actor.physical_intervention_reversal=data.get('physical_intervention_reversal',False)
    actor.physical_singletons=data.get('physical_singletons',False)
    actor.physical_interventions=data.get('physical_interventions',False)
    actor.initial_root=data.get('initial_root',False)
    actor.failure_roots=data.get('failure_roots',False)
    actor.success_roots=data.get('success_roots',False)
    actor.failure_lookback=data.get('failure_lookback',0.)
    actor.failure_duration=data.get('failure_duration',0.)
    actor.credit_lookbacks=tuple(data.get('credit_lookbacks',()))
    actor.paired_advantage=data.get('paired_advantage',False)
    actor.failure_subjects=data.get('failure_subjects',False)
    actor.observed_future=data.get('observed_future',False)
    actor.future_start=data.get('future_start',0)
    actor.joint_contexts=data.get('joint_contexts',0)
    actor.exhaustive_contexts=data.get('exhaustive_contexts',False)
    actor.temporal_programs=data.get('temporal_programs',False)
    actor.temporal_blending_levels=tuple(data.get('temporal_blending_levels',(0.,.5,1.)))
    actor.prefix_refinement=data.get('prefix_refinement',False)
    from conditional_fusion_value import JointInterventionValue
    actor.program_proposer=nn.ModuleList([JointInterventionValue(**record)
        for record in data.get('program_proposer',{}).get('values',[])])
    actor.max_held=data.get('max_held',0)
    actor.edge_futures=data.get('edge_futures',False)
    actor.context_interceptors=data.get('context_interceptors',False)
    actor.context_prediction=data.get('context_prediction',False)
    actor.training_gamma=1. if data.get('success_objective',False) else .995
    actor.training_step_cost=0. if data.get('success_objective',False) else .001
    actor.branch_value_data=data.get('branch_value_data',False)
    actor.branch_grid=data.get('branch_grid',False)
    actor.fitted_iteration=data.get('fitted_iteration',False)
    removal=tuple(data.get('mechanism_ablation',(False,False,False)))
    removal=removal+(False,)*(3-len(removal))
    ablation=tuple(ablation)+(False,)*(3-len(ablation))
    removal=tuple(bool(a or b or c) for a,b,c in zip(removal,EVALUATION_ABLATION,ablation))
    return ablate_joint_mechanisms(actor,*removal)


def joint_phase_boundary(action):
    """Switch time within a complete blending program, in source ticks."""
    duration=int(action[0,4])
    return int(action[0,8]) if action.shape[-1]>=9 else duration//2


def joint_program_gates(action,step):
    """Execute each vessel's own learned phase boundary in a joint program."""
    boundary=action[:,8] if action.shape[-1]>=9 else int(action[0,4])//2
    return np.where(step>=boundary,action[:,5],action[:,3])


def joint_refinement_step(duration,fraction=.5,steps=0):
    """Schedule a new value decision after a measured motion response.

    A fixed observation delay permits long commitments to be reconsidered
    before a quarter of their window has elapsed. Zero retains the original
    fractional schedule.
    """
    return max(1,int(steps)) if steps else max(1,int(duration*fraction))


def joint_refinement_due(duration,step,fraction=.5,steps=0,checkpoints=False):
    """Value the choice to intervene now or retain the later observation.

    Declining an early correction leaves the fractional observation available.
    Once a correction is accepted, its measured continuation runs to the
    original boundary without introducing an untrained second correction.
    """
    return (step==joint_refinement_step(duration,fraction,steps)
        or (checkpoints and step==joint_refinement_step(duration,fraction)))


def joint_remaining_program(control,action,elapsed,preserve_phase=False):
    """Expose a suffix without moving any teammate's scheduled phase change."""
    remaining=int(action[0,4])-elapsed
    boundary=action[:,8:9] if action.shape[-1]>=9 else np.full_like(action[:,:1],joint_phase_boundary(action))
    first=np.where(elapsed<boundary,action[:,3:4],action[:,5:6]) if preserve_phase else action[:,5:6]
    suffix=np.concatenate((control[:,:3],first,np.full_like(action[:,4:5],remaining),action[:,5:6]),-1)
    if preserve_phase:
        correction=action[:,6:8] if action.shape[-1]>=8 else np.zeros_like(action[:,:2])
        switch=np.where(elapsed<boundary,boundary-elapsed,remaining//2)
        suffix=np.concatenate((suffix,correction,switch),-1)
    return suffix


def joint_advance(env,obs,actor,action,maximum=0,refine_midpoint=True,observe_step=None):
    commands=joint_commands(env,obs,actor,action,maximum,refine_midpoint,observe_step)
    try:
        physical=next(commands)
        while True:
            physical=commands.send(joint_task_step(env,physical,actor.training_gamma,actor.training_step_cost))
    except StopIteration as completed:
        return completed.value
    finally:
        commands.close()


def joint_commands(env,obs,actor,action,maximum=0,refine_midpoint=True,observe_step=None):
    """One coordinated gate decision; both source proposals still update at 5 Hz."""
    if action.shape[-1]==14:
        # A delayed intervention is preceded by the actual predecessor policy.
        # Column 13 names that policy relative to the actor's value chain;
        # zero is used while sampling a new option on the current predecessor.
        delay=int(action[0,8]);depth=int(action[0,13])
        if delay>0 and depth>=0:
            original=actor.successor_task_value;predecessor=original
            for _ in range(depth):
                if predecessor is None:raise ValueError('Missing temporal-option predecessor.')
                predecessor=predecessor.prior
            count=0;reward=0.;source_return=0.;done=0
            limit=min(delay,maximum) if maximum else delay
            try:
                actor.successor_task_value=predecessor
                while count<limit and not done:
                    with torch.no_grad():
                        prefix=actor(mechanism_tensor(joint_packet(env,obs)),deterministic=True)[0][0].numpy()
                    observer=(None if observe_step is None else
                        lambda current,observation,elapsed,terminal:observe_step(current,observation,count+elapsed,terminal))
                    obs,r,done,sr,steps=yield from joint_commands(env,obs,actor,prefix,limit-count,refine_midpoint,observer)
                    reward+=(actor.training_gamma**count)*r;source_return+=(.99**count)*sr;count+=steps
            finally:
                actor.successor_task_value=original
            if done or (maximum and count>=maximum):return obs,reward,done,source_return,count
            with torch.no_grad():
                control=actor(mechanism_tensor(joint_packet(env,obs)),deterministic=True,gate_only=True)[0][0].numpy()
            suffix=action[:,:13].copy();suffix[:,:3]=control[:,:3]
            suffix[:,4]-=delay;suffix[:,8]=0;suffix[:,9:11]=suffix[:,11:13]
            observer=(None if observe_step is None else
                lambda current,observation,elapsed,terminal:observe_step(current,observation,count+elapsed,terminal))
            obs,r,done,sr,steps=yield from joint_commands(env,obs,actor,suffix,
                maximum-count if maximum else 0,refine_midpoint,observer)
            return obs,reward+(actor.training_gamma**count)*r,done,source_return+(.99**count)*sr,count+steps
        action=action[:,:13]
    duration=int(action[0,4]) if action.shape[-1]>4 else actor.physical_span
    waiting=(actor.cooperative_role_credit is not None and actor.cooperative_role_credit.wait_span)
    waiting=waiting or (action.shape[-1]>5 and (action[:,3]<0.).all() and (action[:,5]==-2.).all())
    # A phase-preserving suffix owns its remaining decision window even when
    # its optimal correction releases every held coefficient. Releasing the
    # controls must not silently restart the outer intervention policy early.
    waiting=waiting or action.shape[-1]==9
    active=(action.shape[-1]>3 and ((action[:,3]>=0.).any()
        or (action.shape[-1]>5 and (action[:,5]>=0.).any())))
    active=active or (action.shape[-1]==13 and (action[:,9:13]>=0.).any())
    period=max(actor.gate_period,duration) if action.shape[-1]>3 and (active or waiting) else actor.gate_period
    period=min(period,maximum) if maximum else period
    reward=0.;source_return=0.;done=0;control=action[:,:3].copy()
    intent=action
    soft=bool(actor.preference_strength and active)
    immediate=bool(actor.immediate_commitment
        and action.shape[-1]>3 and (action[:,3]>=0.).any())
    correction=(action[:,6:8] if action.shape[-1]>6 and actor.conditional_enabled else None)
    if (actor.cooperative_role_credit is not None or getattr(actor,'finite_temporal_credit',None) is not None
            or getattr(actor,'successor_task_value',None) is not None) and active:
        env._conditional_option_count=getattr(env,'_conditional_option_count',0)+1
    for step in range(period):
        scheduled=joint_program_gates(action,step) if action.shape[-1]>5 else action[:,3] if action.shape[-1]>3 else None
        committed=(action.shape[-1]>3 and (scheduled>=0.).all()
            and not actor.intent_strength and actor.response_base is None and not actor.preference_strength)
        if ((step and step%actor.gate_period==0 and not committed) or (soft and step==0)
                or (immediate and step==0 and not committed)):
            with torch.no_grad():
                # A committed physical intervention only needs the free boats'
                # current fusion gates here, not another ignored option choice.
                gate_only=not actor.intent_strength and actor.response_base is None
                packet=mechanism_tensor(joint_packet(env,obs))
                if soft and step<duration:
                    packet['_option_preference']=torch.tensor(scheduled[None],dtype=torch.float32)
                elif actor.conditional_commitment and action.shape[-1]>3 and step<duration:
                    packet['_committed_gates']=torch.tensor(scheduled[None],dtype=torch.float32)
                if correction is not None and step<duration:
                    packet['_proposal_offset']=torch.tensor(correction[None],dtype=torch.float32)
                intent=actor(packet,deterministic=True,gate_only=gate_only)[0][0].numpy()
            if not actor.intent_strength:control=intent[:,:3].copy()
        if actor.intent_strength:
            with torch.no_grad():
                control=actor.fast_action(mechanism_tensor(joint_packet(env,obs)),torch.tensor(intent[None]))[0].numpy()
        elif step:
            tensor=torch.tensor(obs[None],dtype=torch.float32)
            control[:,:2]=actor.proposal_distribution(tensor)[0][0].tanh().numpy()
            if correction is not None and step<duration:
                control[:,:2]=np.clip(control[:,:2]+correction,-1.,1.)
        if (refine_midpoint and actor.midpoint_intervention_values is not None
                and not soft and action.shape[-1] in (6,9)
                and joint_refinement_due(duration,step,actor.option_refinement_fraction,
                    actor.option_refinement_steps,actor.option_refinement_checkpoints) and step>0
                and (action[:,5]>=0.).any()
                and (actor.phase_preserving_refinement or actor.option_refinement_fraction==.5
                     or np.array_equal(action[:,3],action[:,5]))):
            # Reassess the planned suffix after observing the first phase.
            # Its end time is unchanged. Only the measured prefix refinement
            # may change; the frozen predecessor resumes at the same boundary.
            suffix=joint_remaining_program(control,action,step,actor.phase_preserving_refinement)
            with torch.no_grad():
                revised=actor.midpoint_intervention_values(mechanism_tensor(joint_packet(env,obs)),
                    torch.as_tensor(suffix[None],dtype=torch.float32))[0].numpy()
            if not np.array_equal(revised[:,3:],suffix[:,3:]):
                observer=(None if observe_step is None else
                    lambda current,observation,elapsed,terminal:observe_step(current,observation,step+elapsed,terminal))
                obs,r,done,sr,count=yield from joint_commands(env,obs,actor,revised,period-step,False,observer)
                return obs,reward+(actor.training_gamma**step)*r,done,source_return+(.99**step)*sr,step+count
        if action.shape[-1]>3 and step<duration and not soft:
            fixed=scheduled>=0.
            control[fixed,2]=scheduled[fixed]
        physical=control
        if actor.generic_fusion_adapter is not None:
            with torch.no_grad():
                physical=actor.generic_fusion_adapter.control(actor,mechanism_tensor(joint_packet(env,obs)),
                    torch.as_tensor(control[None],dtype=torch.float32))[0].numpy()
        if actor.response_feedback_value is not None:
            with torch.no_grad():
                physical=actor.response_feedback_value(mechanism_tensor(joint_packet(env,obs)),
                    torch.as_tensor(control[None],dtype=torch.float32))[0].numpy()
        if action.shape[-1]==13 and step<duration:
            from conditional_fusion_value import apply_actuator_preferences
            preferences=np.where((step>=action[:,8])[:,None],action[:,11:13],action[:,9:11])
            physical=apply_actuator_preferences(mechanism_tensor(joint_packet(env,obs)),
                torch.as_tensor(control[None],dtype=torch.float32),torch.as_tensor(physical[None],dtype=torch.float32),
                torch.as_tensor(preferences[None],dtype=torch.float32))[0].numpy()
        if getattr(actor,'actuator_fusion_refinement',None):
            from conditional_fusion_value import refine_actuator_interactions
            refinement_packet=mechanism_tensor(joint_packet(env,obs))
            if action.shape[-1]==13 and step<duration:
                refinement_packet['_committed_actuator_preferences']=torch.as_tensor(
                    preferences[None],dtype=torch.float32)
            elif scheduled is not None and step<duration:
                refinement_packet['_committed_actuator_preferences']=torch.as_tensor(
                    np.repeat(scheduled[None,:,None],2,axis=-1),dtype=torch.float32)
            if actor.response_feedback_value is None:
                native=np.repeat(control[:,2:3],2,axis=-1)
                coefficients=(np.where(preferences>=0.,preferences,native)
                    if action.shape[-1]==13 and step<duration else native)
                refinement_packet['_preferred_actuator_coefficients']=torch.as_tensor(
                    coefficients[None],dtype=torch.float32)
            physical=refine_actuator_interactions(actor,refinement_packet,
                torch.as_tensor(control[None],dtype=torch.float32),
                torch.as_tensor(physical[None],dtype=torch.float32),**actor.actuator_fusion_refinement)[0].numpy()
        obs,r,done,sr=yield physical
        if observe_step is not None:observe_step(env,obs,step+1,done)
        reward+=(actor.training_gamma**step)*r;source_return+=(.99**step)*sr
        if done:break
    return obs,reward,done,source_return,step+1


class JointMeasuredController:
    """Execute the trained fusion policy on successive public measurements.

    The numerical evaluator and physical deployment share joint_commands;
    this adapter supplies measurements in place of stepping a simulator.
    """
    def __init__(self,checkpoint,source,inference_device='cpu'):
        from types import SimpleNamespace
        initialize(source)
        checkpoint=Path(checkpoint)
        self.actor=joint_load(str(checkpoint),checkpoint.stat().st_mtime_ns).cpu()
        # Deployment selects its available inference device; a training host's
        # CUDA placement is not part of the learned intervention rule.
        for module in self.actor.modules():
            if hasattr(module,'inference_device'):
                module.inference_device=inference_device
        self.environment=SimpleNamespace()
        self.commands=None
        self.steps=0

    def reset(self):
        if self.commands is not None:self.commands.close()
        self.commands=None;self.steps=0
        if hasattr(self.environment,'_conditional_option_count'):
            del self.environment._conditional_option_count

    def control(self,defenders,attacker,observations,boids,elapsed,total_time=80.):
        from types import SimpleNamespace
        from source_arboids import source_mixture
        def boat(state):
            # Relative-water velocity is unavailable in ROS feedback. The
            # nominal predictor uses measured ground velocity and zero current.
            return SimpleNamespace(x=float(state[0]),y=float(state[1]),theta=float(state[2]),
                pos=np.asarray(state[:2]),vel=np.asarray(state[3:5]),velocity=np.asarray(state[3:6]),
                velocity_r=np.asarray(state[3:6]),agility=1.)
        env=self.environment
        env.defender_list=[boat(state) for state in defenders];env.attacker=boat(attacker)
        env.boids_actions=np.asarray(boids);env.Current_T=float(elapsed);env.Total_T=float(total_time)
        observations=np.asarray(observations,dtype=np.float32)
        with torch.no_grad():
            if self.commands is not None:
                try:physical=self.commands.send((observations,0.,0,0.))
                except StopIteration:self.commands=None
            if self.commands is None:
                action=self.actor(mechanism_tensor(joint_packet(env,observations)),deterministic=True)[0][0].numpy()
                self.commands=joint_commands(env,observations,self.actor,action)
                physical=next(self.commands)
        self.steps+=1
        return source_mixture(physical,env.boids_actions),dict(
            conditional_interventions=int(getattr(env,'_conditional_option_count',0)),
            control_steps=self.steps,physical_action=physical.copy())


def joint_episode(task):
    scene,path,revision,collect,roots_count,repeats,horizon,limit=task[:8]
    future_seed=task[8] if len(task)>8 else None
    seed,n,agility=scene;np.random.seed(seed)
    actor=joint_load(path,revision,task[9] if len(task)>9 else (False,False))
    torch.manual_seed(seed)
    if actor.environment_protocol=='source':env=numerical_source().TADEnv(n)
    else:
        from envs.TADgame import TADEnv
        env=TADEnv(n,protocol=actor.environment_protocol)
    obs,_=env.reset(agility=agility,noisy_agility=False)
    if future_seed is not None:np.random.seed(future_seed)
    done=0;steps=[];roots=[];rng=np.random.default_rng(seed^917803);total=0.;discount=1.;physical_steps=0
    while not done and (not limit or physical_steps<limit):
        packet=joint_packet(env,obs)
        with torch.no_grad():action,latent,lp,_,_,_=actor(mechanism_tensor(packet),deterministic=not collect or actor.fitted_iteration)
        action=action[0].numpy()
        if collect and roots_count:
            weight=1.+np.exp(-((env.def_def_dists.min()-7.)/3.)**2)
            hindsight=actor.failure_roots and (actor.failure_lookback>0. or actor.credit_lookbacks)
            key=(float(env.Current_T) if hindsight else
                1. if not physical_steps and getattr(actor,'initial_root',False) else
                float(np.log(max(rng.random(),1e-12))/weight))
            lookback=max(actor.credit_lookbacks) if actor.credit_lookbacks else actor.failure_lookback*roots_count
            capacity=(int(np.ceil(lookback/(.2*actor.gate_period)))+2
                if hindsight else roots_count)
            if len(roots)<capacity or key>min(r[0] for r in roots):
                root=copy.deepcopy(env)
                if actor.observed_future:
                    root._intervention_random_state=np.random.get_state()
                    root._intervention_torch_state=torch.get_rng_state()
                for name in ('Pos_Att','Phi_Att','Pos_Def','Phi_Def','Rewards'):
                    value=getattr(root,name)
                    if np.ndim(value)>1:setattr(root,name,value[-1:].copy())
                roots.append((key,root,obs.copy(),packet,action.copy()))
                if len(roots)>capacity:roots.pop(min(range(len(roots)),key=lambda i:roots[i][0]))
        obs,reward,done,sr,span=joint_advance(env,obs,actor,action,limit-physical_steps if limit else 0)
        total+=discount*sr;discount*=.99**span;physical_steps+=span
        if collect:steps.append(dict(packet=packet,action=action,latent=latent[0].numpy(),
            logp=lp[0].numpy(),reward=reward,done=done,discount=actor.training_gamma**span,physical_steps=span))
    row=dict(seed=int(seed),defenders=n,agility=agility,**outcome(env,done),discounted_return=total,
        environment_protocol=actor.environment_protocol,
        conditional_interventions=int(getattr(env,'_conditional_option_count',0)))
    if future_seed is not None:row['future_seed']=int(future_seed)
    if not collect:return row
    tail=None if done else joint_packet(env,obs)
    if actor.success_roots and done<=2:roots=[]
    elif actor.failure_roots and done>2:roots=[]
    elif roots and actor.failure_roots and (actor.failure_lookback>0. or actor.credit_lookbacks):
        # Hindsight chooses training states only. Deployed decisions never
        # receive the eventual failure time or any realized future noise.
        selected={}
        # Resolve short collision credit and delayed interception credit on the
        # same factual trajectory. Selection changes training roots only.
        lookbacks=actor.credit_lookbacks or tuple(actor.failure_lookback*(j+1) for j in range(roots_count))
        for lookback in lookbacks:
            target=env.Current_T-lookback
            eligible=[root for root in roots if root[0]<=target+1e-6]
            root=max(eligible,key=lambda r:r[0]) if eligible else min(roots,key=lambda r:r[0])
            selected[root[0]]=root
        roots=list(selected.values())
    if actor.failure_duration and env.Current_T>actor.failure_duration:roots=[]
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
        if actor.prefix_refinement and (reference.shape[-1]<6
                or int(reference[0,4])!=actor.physical_intervention_span
                or not (reference[:,[3,5]]>=0.).any()):continue
        p=packet['motion'][:,:2];distance=np.linalg.norm(p[:,None]-p[None],axis=-1)+1e6*np.eye(n)
        subjects=np.unravel_index(distance.argmin(),distance.shape)
        if actor.physical_interventions:
            if event_subjects is not None:subjects=event_subjects
            if reference.shape[-1]==3:
                reference=np.concatenate((reference,np.full((n,1),-1.,dtype=reference.dtype)),-1)
            if reference.shape[-1]==4:
                reference=np.concatenate((reference,np.full((n,1),actor.physical_span,dtype=reference.dtype)),-1)
            if reference.shape[-1]==5:
                reference=np.concatenate((reference,reference[:,3:4]),-1)
            arms=[];edges=[];physical_grid=(-1.,0.,.5,1.)
            if actor.physical_singletons:
                # Early assignment credit must cover every vessel. The closest
                # initial pair need not contain the future interceptor.
                if (reference[:,3]>=0.).any():
                    raise ValueError('Single-vessel assignment collection requires an all-feedback preceding policy.')
                arms.append(reference.copy());reference_index=0
                for subject in range(n):
                    local=[0]
                    for gate in physical_grid[1:]:
                        arm=reference.copy();arm[:,4]=actor.physical_intervention_span
                        arm[subject,3]=gate
                        arm[:,5]=np.where((arm[:,3]>=0.)&actor.physical_intervention_reversal,1.-arm[:,3],arm[:,3])
                        local.append(len(arms));arms.append(arm)
                    edges.extend((local[i],local[j]) for i in range(4) for j in range(i+1,4))
            else:
                for x in physical_grid:
                    for y in physical_grid:
                        arm=reference.copy();arm[:,4]=actor.physical_intervention_span
                        arm[subjects[0],3]=x;arm[subjects[1],3]=y
                        arm[:,5]=np.where((arm[:,3]>=0.)&actor.physical_intervention_reversal,1.-arm[:,3],arm[:,3])
                        arms.append(arm)
                for fixed in range(4):
                    for left in range(4):
                        for right in range(left+1,4):
                            edges.extend(((left*4+fixed,right*4+fixed),(fixed*4+left,fixed*4+right)))
                reference_index=physical_grid.index(float(reference[subjects[0],3]))*4+physical_grid.index(float(reference[subjects[1],3]))
        elif actor.branch_grid:
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
        if (actor.joint_contexts or actor.exhaustive_contexts or actor.max_held) and not actor.temporal_programs:
            # Explore coordinated contexts before assigning single-vessel
            # credit. A pair-only neighborhood of an unsuccessful joint action
            # can be flat even when several coordinated changes would succeed.
            intervention_column=3 if actor.physical_interventions else 2
            edges=list(edges);known={tuple(a[:,intervention_column]):j for j,a in enumerate(arms)}
            def add_arm(action):
                key=tuple(action[:,intervention_column])
                if key not in known:known[key]=len(arms);arms.append(action)
                return known[key]
            predicted_contexts=None
            if actor.max_held:
                if not actor.physical_interventions:
                    raise ValueError('Sparse joint programs require explicit feedback and held-gate choices.')
                ids=np.arange(4**n)
                choices=(ids[:,None]//(4**np.arange(n)))%4
                choices=choices[(choices>0).sum(-1)<=actor.max_held]
                predicted_contexts=np.where(choices>0,(choices-1)*.5,-1.)
            elif actor.exhaustive_contexts:
                ids=np.arange(actor.levels**n)
                predicted_contexts=((ids[:,None]//(actor.levels**np.arange(n)))%actor.levels)/(actor.levels-1)
            elif actor.context_interceptors and actor.physical_interventions:
                # Cover coordinated pursuit roles explicitly. Each complete
                # program still gets paired single-vessel deletion edges;
                # terminal outcomes, rather than these templates, determine
                # which role assignments may later be executed.
                library=[np.zeros(n),np.ones(n)]
                target_distance=np.linalg.norm(packet['node'][:,2:4],axis=-1)
                for subject in np.argsort(target_distance):
                    pursuit=np.ones(n);pursuit[subject]=0.
                    library.extend((pursuit,1.-pursuit))
                predicted_contexts=np.asarray(library,dtype=np.float32)
            elif actor.context_prediction and actor.physical_interventions:
                with torch.no_grad():
                    encoding=actor.value_encoding(mechanism_tensor(packet),
                        reversal_at=.1*actor.physical_intervention_span if actor.physical_intervention_reversal else None)
                    scores=encoding['_fusion_scores'][0]
                    index=scores.argsort(descending=True)[:actor.joint_contexts]
                    digits=((index[:,None]//(actor.levels**torch.arange(n)))%actor.levels).float()
                    predicted_contexts=(digits/(actor.levels-1)).numpy()
            count=len(predicted_contexts) if actor.exhaustive_contexts or actor.max_held else actor.joint_contexts
            for context_id in range(count):
                context=reference.copy()
                if actor.physical_interventions:context[:,4]=actor.physical_intervention_span
                if predicted_contexts is not None and context_id<len(predicted_contexts):
                    context[:,intervention_column]=predicted_contexts[context_id]
                elif context_id<2:context[:,intervention_column]=float(context_id)
                elif context_id%2:
                    context[:,intervention_column]=rng.integers(0,2,size=n)
                else:
                    probability=np.asarray([.25,.5,.25]) if actor.levels==3 else None
                    context[:,intervention_column]=rng.choice(actor.levels,size=n,p=probability)/(actor.levels-1)
                if actor.physical_interventions:
                    context[:,5]=np.where((context[:,3]>=0.)&actor.physical_intervention_reversal,1.-context[:,3],context[:,3])
                changed=np.flatnonzero(context[:,intervention_column]!=reference[:,intervention_column])
                if not len(changed):continue
                subject=int(rng.choice(changed));deletion=context.copy()
                deletion[subject,intervention_column]=reference[subject,intervention_column]
                if actor.physical_interventions:deletion[subject,5]=reference[subject,5]
                edge=(add_arm(context),add_arm(deletion))
                if edge not in edges and edge[::-1] not in edges:edges.append(edge)
            if actor.exhaustive_contexts or actor.max_held:
                # A flat sampled neighborhood calls for joint coverage. Connect
                # the complete fixed-gate library by single-vessel changes so
                # higher-order successful combinations remain value nodes, while
                # every comparison still changes exactly one vessel.
                existing={tuple(sorted(edge)) for edge in edges}
                for gates in predicted_contexts:
                    left=known[tuple(gates)]
                    for subject in range(n):
                        for gate in (physical_grid if actor.max_held else np.arange(actor.levels)/(actor.levels-1)):
                            if gate<=gates[subject]:continue
                            alternate=gates.copy();alternate[subject]=gate
                            if tuple(alternate) not in known:continue
                            edge=tuple(sorted((left,known[tuple(alternate)])))
                            if edge not in existing:edges.append(edge);existing.add(edge)
        if actor.temporal_programs:
            from conditional_fusion_value import temporal_program_graph
            with torch.no_grad():
                seed_actions=[value(mechanism_tensor(packet),torch.as_tensor(reference[None]))[0].numpy()
                    for value in actor.program_proposer]
            arms,edges,reference_index=temporal_program_graph(reference,subjects,
                actor.physical_intervention_span,actor.joint_contexts,rng,seed_actions,actor.temporal_blending_levels)
        if actor.prefix_refinement:
            from conditional_fusion_value import prefix_intervention_graph
            with torch.no_grad():
                seed_actions=[value(mechanism_tensor(packet),torch.as_tensor(reference[None]))[0].numpy()
                    for value in actor.program_proposer]
            arms,edges,reference_index=prefix_intervention_graph(reference,subjects,actor.joint_contexts,rng,seed_actions)
        # The factual arm preserves the entire preceding option, including its
        # decision duration. It is a value anchor, never a single-vessel edge.
        if actor.fixed_intervention_clock and actor.physical_interventions and not actor.prefix_refinement:
            for arm in arms:
                if (arm[:,3]<0.).all():arm[:,5]=-2.
        factual_index=len(arms)
        if actor.paired_advantage:arms.append(reference.copy())
        def paired_continuations(selected,future,observed_future):
            nonlocal simulation_steps
            out={}
            for arm_index in selected:
                action=arms[arm_index]
                np.random.seed(future);torch.manual_seed(future)
                if observed_future:
                    # Retain the factual future as one common-random-number
                    # replicate; additional futures still estimate robustness.
                    # Otherwise resampling can erase the very failure whose
                    # conditional control effect is being learned.
                    np.random.set_state(root._intervention_random_state)
                    torch.set_rng_state(root._intervention_torch_state)
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
                out[arm_index]=dict(reward=reward,discount=gamma,next=None if bd else joint_packet(branch,bo),
                    terminal=bd,success=int(bd>2) if bd else None,first_step=first_step)
            return out
        for repetition in range(repeats+actor.future_start):
            future=int(rng.integers(1,2**31-1))
            if repetition<actor.future_start:continue
            observed_future=bool(actor.observed_future and repetition==0)
            # Each new local comparison gets a fresh common-noise continuation.
            # Reusing only two futures for every node makes edge differences
            # nearly identical to differences of the absolute sample means.
            # Independent paired futures retain their covariance while covering
            # more disturbance realizations within the comparison graph.
            groups=([[(a,b)] for a,b in edges] if actor.edge_futures and not observed_future else [edges])
            for group in groups:
                if actor.edge_futures and not observed_future:future=int(rng.integers(1,2**31-1))
                selected={reference_index}
                if actor.paired_advantage:selected.add(factual_index)
                for a,b in group:selected.update((a,b))
                out=paired_continuations(sorted(selected),future,observed_future)
                for a,b in group:
                    branches.append(dict(packet=packet,left=arms[a],right=arms[b],before=out[a],after=out[b],
                        reference_action=reference,reference_outcome=out[reference_index],
                        factual_outcome=out[factual_index] if actor.paired_advantage else None,
                        observed_future=observed_future,future_repetition=repetition,
                        continuation_seed=None if observed_future else future,
                        root=(int(seed),int(round(root.Current_T/.2)),a,b)))
    return dict(row=row,steps=steps,tail=tail,branches=branches,value_steps=value_steps,simulation_steps=simulation_steps)


def collect_initial_roles_batched(path,scenes,output,repetitions=4,future_start=0,
                                  programs=None,batch_size=64,device='cuda:0',root_records=None,
                                  proposal_offsets=None):
    """Paired role rollouts with batched policy prediction and private RNGs.

    Dynamics and termination still execute through the original environment.
    Only the mechanism's inference is batched. A second call can evaluate a
    shortlist on disjoint futures, separating discovery from credit fitting.
    """
    from envs.TADgame import TADEnv
    torch.set_num_threads(1)
    model=torch.load(path,map_location='cpu',weights_only=False)
    if any(model.get(key) is not None for key in ('initial_role_credit','feedback_role_credit','cooperative_role_credit','joint_intervention_value',
            'joint_intervention_values','physical_conditional_value','shared_response_value','response_base')):
        raise ValueError('Initial role experiments require the same unmodified continuation policy.')
    actor=joint_load(str(path),Path(path).stat().st_mtime_ns).to(device)
    if actor.intent_strength or actor.gate_period!=1:
        raise ValueError('Batched initial roles require a per-step fusion policy.')
    torch.backends.cuda.matmul.allow_tf32=False
    roots=[];jobs=[];results={};factual_steps={};factual_rows={};root_steps=[];future_seeds={}
    span=actor.physical_intervention_span
    for scene_index,(seed,n,agility) in enumerate(scenes):
        if root_records is None:
            np.random.seed(seed)
            env=(numerical_source().TADEnv(n) if actor.environment_protocol=='source'
                else TADEnv(n,protocol=actor.environment_protocol))
            obs,_=env.reset(agility=agility,noisy_agility=False)
            random_state=np.random.get_state()
        else:
            record=root_records[scene_index]
            env=copy.deepcopy(record['environment']);obs=record['observation'].copy()
            random_state=record['random_state']
        root_steps.append(int(round(env.Current_T/.2)))
        packet=joint_packet(env,obs)
        with torch.no_grad():
            reference=actor({key:torch.as_tensor(value[None],device=device) for key,value in packet.items()},
                deterministic=True,gate_only=True)[0][0].cpu().numpy()
        if programs is None:
            choices=[np.full(n,-1.,dtype=np.float32)]
            for subject in range(n):
                for gate in (0.,.5,1.):
                    choice=choices[0].copy();choice[subject]=gate;choices.append(choice)
        else:choices=[np.asarray(gates,dtype=np.float32) for gates in programs[scene_index]]
        if np.any(choices[0]>=0.):raise ValueError('The first program must be the all-feedback reference.')
        arms=[np.concatenate((reference,gates[:,None],np.full((n,1),span,dtype=np.float32),
            np.where((gates>=0.)&actor.physical_reversal,1.-gates,gates)[:,None]),-1) for gates in choices]
        if proposal_offsets is not None:
            offsets=proposal_offsets[scene_index]
            if len(offsets)!=len(arms) or np.any(np.asarray(offsets[0])):
                raise ValueError('Proposal credit requires a zero-correction factual program.')
            arms=[np.concatenate((arm,np.asarray(offset,dtype=np.float32)),-1) for arm,offset in zip(arms,offsets)]
        roots.append((env,obs,packet,arms,random_state))
        for repetition in range(future_start,future_start+repetitions):
            entropy=([seed,repetition,8191441] if root_records is None else
                [seed,root_records[scene_index]['id'],root_steps[-1],repetition,8191559])
            future=int(np.random.SeedSequence(entropy).generate_state(1)[0])
            future_seeds[scene_index,repetition]=future
            for arm in range(len(arms)):jobs.append((scene_index,repetition,arm,future))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.save(model,output/'current.pth')
    signature=(proposal_parameter_digest(model['actor']),tuple((key,str(model.get(key))) for key in
        ('environment_protocol','peer','levels','risk_prior','risk_horizon','risk_clearance','risk_barrier',
         'risk_feedback','prediction_step','conditional_commitment','preference_strength','mechanism_ablation',
         'physical_intervention_span','physical_intervention_reversal')),
        tuple(tuple(tuple(float(g) for g in arm[:,3]) for arm in root[3]) for root in roots))
    if model.get('prediction_relaxation',0.):signature=(*signature,('prediction_relaxation',model['prediction_relaxation']))
    if root_records is not None:
        signature=(*signature,tuple((root_steps[i],hashlib.sha256(
            b''.join(value.tobytes() for _,value in sorted(root[2].items()))).hexdigest())
            for i,root in enumerate(roots)))
    if proposal_offsets is not None:
        signature=(*signature,tuple(tuple(tuple(float(x) for x in arm[:,6:8].flatten()) for arm in root[3])
            for root in roots))
    evidence_path=output/'training-data.pth'
    if evidence_path.exists():
        previous=torch.load(evidence_path,map_location='cpu',weights_only=False)
        if previous.get('sampling_signature')!=signature or previous.get('sampling_jobs')!=jobs:
            raise ValueError('Existing role evidence belongs to a different policy or sampling plan.')
        if previous.get('completed',False):
            save_json(output/'development.json',dict(collection=summarize([row['row'] for row in previous['rollouts']]),
                rows=[row['row'] for row in previous['rollouts']],simulation_steps=previous['simulation_steps'],
                repetitions=repetitions,future_start=future_start,inference_device=device))
            actor.cpu();return previous
        results=previous['partial_results'];factual_steps=previous['factual_steps'];factual_rows=previous['factual_rows']
    remaining=[job for job in jobs if tuple(job[:3]) not in results]
    for offset in range(0,len(remaining),batch_size):
        current=remaining[offset:offset+batch_size];environments=[];observations=[];states=[]
        rewards=np.zeros(len(current));returns=np.zeros(len(current));active=list(range(len(current)))
        for scene_index,repetition,arm,future in current:
            root,obs,_,_,random_state=roots[scene_index]
            environments.append(copy.deepcopy(root));observations.append(obs.copy())
            states.append(random_state if repetition==0 else np.random.RandomState(future).get_state())
        for step in range(400):
            if not active:break
            packets=[joint_packet(environments[j],observations[j]) for j in active]
            tensors={key:torch.as_tensor(np.stack([packet[key] for packet in packets]),device=device)
                for key in packets[0]}
            options=np.stack([roots[current[j][0]][3][current[j][2]] for j in active])
            column=5 if step>=span//2 else 3
            if step<span:
                gates=torch.as_tensor(options[:,:,column],device=device)
                if actor.preference_strength:tensors['_option_preference']=gates
                elif actor.conditional_commitment and step:tensors['_committed_gates']=gates
                if options.shape[-1]>6:tensors['_proposal_offset']=torch.as_tensor(options[:,:,6:8],device=device)
            with torch.no_grad():
                action,latent,logp,_,_,_=actor(tensors,deterministic=True,gate_only=True)
                action=action.cpu().numpy();latent=latent.cpu().numpy();logp=logp.cpu().numpy()
            if step<span and not actor.preference_strength:
                if not step:action=options[:,:,:3].copy()
                action[:,:,2]=np.where(options[:,:,3]>=0.,options[:,:,column],action[:,:,2])
            following=[]
            for local,j in enumerate(active):
                scene_index,repetition,arm,future=current[j]
                np.random.set_state(states[j]);env=environments[j]
                obs,reward,done,sr=joint_task_step(env,action[local],1.,0.)
                states[j]=np.random.get_state();observations[j]=obs
                rewards[j]+=reward;returns[j]+=(.99**step)*sr
                if repetition==future_start and arm==0:
                    factual_steps.setdefault(scene_index,[]).append(dict(packet=packets[local],action=action[local],
                        latent=latent[local],logp=logp[local],reward=reward,done=done,discount=1.,physical_steps=1))
                if done:
                    results[scene_index,repetition,arm]=dict(reward=float(rewards[j]),discount=1.,next=None,
                        terminal=int(done),success=int(done>2),first_step=None,physical_steps=step+1,duration=env.Current_T)
                    if repetition==future_start and arm==0:
                        seed,n,agility=scenes[scene_index]
                        factual_rows[scene_index]=dict(seed=seed,defenders=n,agility=agility,**outcome(env,done),
                            discounted_return=float(returns[j]),environment_protocol=actor.environment_protocol)
                        if repetition:factual_rows[scene_index]['future_seed']=future
                        if root_records is not None:
                            factual_rows[scene_index].update(root_id=root_records[scene_index]['id'],
                                root_step=root_steps[scene_index])
                else:following.append(j)
            active=following
        if active:raise ValueError('Counterfactual roles must finish at a true task terminal.')
        temporary=output/'training-data.tmp'
        torch.save(dict(model=model,completed=False,sampling_signature=signature,sampling_jobs=jobs,
            partial_results=results,factual_steps=factual_steps,factual_rows=factual_rows),temporary)
        temporary.replace(evidence_path)
        print(dict(phase='batched_role_continuations',completed=len(results),total=len(jobs),
            simulation_steps=sum(value['physical_steps'] for value in results.values())),flush=True)
    rollouts=[]
    for scene_index,(_,_,packet,arms,_) in enumerate(roots):
        branches=[]
        edges=[(i,j) for i in range(len(arms)) for j in range(i+1,len(arms))
            if np.count_nonzero(np.any(arms[i][:,3:]!=arms[j][:,3:],axis=-1))==1]
        for repetition in range(future_start,future_start+repetitions):
            future=future_seeds[scene_index,repetition]
            reference=results[scene_index,repetition,0]
            for left,right in edges:
                branches.append(dict(packet=packet,left=arms[left],right=arms[right],
                    before=results[scene_index,repetition,left],after=results[scene_index,repetition,right],
                    reference_action=arms[0],reference_outcome=reference,factual_outcome=reference,
                    observed_future=repetition==0,future_repetition=repetition,
                    continuation_seed=None if repetition==0 else future,
                    root=(scenes[scene_index][0],root_steps[scene_index],left,right)))
        steps=sum(value['physical_steps'] for key,value in results.items() if key[0]==scene_index)
        rollouts.append(dict(row=factual_rows[scene_index],steps=factual_steps[scene_index],tail=None,
            branches=branches,value_steps=[],simulation_steps=steps))
    data=dict(model=model,rollouts=rollouts,simulation_steps=sum(row['simulation_steps'] for row in rollouts),
        source_sha256=model.get('source_sha256'),completed=True,future_start=future_start,
        sampling_signature=signature,sampling_jobs=jobs,
        use='initial-role counterfactual development; disjoint future seeds for shortlist confirmation')
    torch.save(data,output/'training-data.pth')
    save_json(output/'development.json',dict(collection=summarize([row['row'] for row in rollouts]),
        rows=[row['row'] for row in rollouts],simulation_steps=data['simulation_steps'],
        repetitions=repetitions,future_start=future_start,inference_device=device))
    actor.cpu()
    return data


@torch.no_grad()
def collect_bootstrapped_roles(path,root_records,value_record,output,span=10,repetitions=4,
                              future_start=40,batch_size=128,device='cuda:0'):
    """Short paired interventions with a cross-fitted factual continuation.

    Only training uses copied environments. Terminal success and estimated
    continuation value are stored separately, so estimates cannot become
    reported task successes.
    """
    from conditional_fusion_value import FactualContinuationValue
    model=torch.load(path,map_location='cpu',weights_only=False)
    actor=joint_load(str(path),Path(path).stat().st_mtime_ns).to(device)
    if actor.cooperative_role_credit is not None or actor.gate_period!=1:
        raise ValueError('Bootstrap collection needs the declared unmodified continuation policy.')
    values=[]
    for state in value_record['states']:
        value=FactualContinuationValue(value_record['width']).to(device)
        value.load_state_dict(state);values.append(value.eval())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    evidence=output/'training-data.pth'
    signature=(proposal_parameter_digest(model['actor']),span,repetitions,future_start,
        tuple((r['scene'],r['id'],round(r['environment'].Current_T,6)) for r in root_records))
    previous=torch.load(evidence,map_location='cpu',weights_only=False) if evidence.exists() else None
    if previous and previous['sampling_signature']!=signature:raise ValueError('The bootstrap sampling plan changed.')
    if previous and previous.get('completed'):return previous
    results={} if previous is None else previous['partial_results']
    roots=[];jobs=[]
    for ri,record in enumerate(root_records):
        seed,n,_=record['scene'];packet=joint_packet(record['environment'],record['observation'])
        reference=actor({k:torch.as_tensor(v[None],device=device) for k,v in packet.items()},
            deterministic=True,gate_only=True)[0][0].cpu().numpy()
        programs=[np.full(n,-1.,np.float32)]
        for subject in range(n):
            for gate in (0.,.5,1.):
                program=programs[0].copy();program[subject]=gate;programs.append(program)
        arms=[np.concatenate((reference,p[:,None],np.full((n,1),span,np.float32),p[:,None]),-1) for p in programs]
        roots.append((packet,arms))
        for repetition in range(future_start,future_start+repetitions):
            for arm in range(len(arms)):
                if (ri,repetition,arm) not in results:jobs.append((ri,repetition,arm))
    for begin in range(0,len(jobs),batch_size):
        current=jobs[begin:begin+batch_size];envs=[];observations=[];random_states=[]
        active=list(range(len(current)));endings={}
        for ri,repetition,arm in current:
            record=root_records[ri]
            envs.append(copy.deepcopy(record['environment']));observations.append(record['observation'].copy())
            future=int(np.random.SeedSequence([record['scene'][0],record['id'],repetition,8191771]).generate_state(1)[0])
            random_states.append(np.random.RandomState(future).get_state())
        for step in range(span):
            if not active:break
            records=[joint_packet(envs[j],observations[j]) for j in active]
            packet={k:torch.as_tensor(np.stack([r[k] for r in records]),device=device) for k in records[0]}
            packet['_option_preference']=torch.as_tensor(np.stack([roots[current[j][0]][1][current[j][2]][:,3]
                for j in active]),device=device)
            actions=actor(packet,deterministic=True,gate_only=True)[0].cpu().numpy();following=[]
            for j,action in zip(active,actions):
                np.random.set_state(random_states[j]);obs,_,done,_=joint_task_step(envs[j],action,1.,0.)
                observations[j]=obs;random_states[j]=np.random.get_state()
                if done:endings[j]=dict(success=int(done>2),value_target=float(done>2),terminal=int(done),
                    physical_steps=step+1,target_source='task terminal')
                else:following.append(j)
            active=following
        if active:
            records=[joint_packet(envs[j],observations[j]) for j in active]
            packet={k:torch.as_tensor(np.stack([r[k] for r in records]),device=device) for k in records[0]}
            action=actor(packet,deterministic=True,gate_only=True)[0]
            estimates=torch.stack([value(packet,action).softmax(-1) for value in values])
            for local,j in enumerate(active):
                seed=root_records[current[j][0]]['scene'][0];fold=value_record['layout_fold'][int(seed)]
                probability=estimates[fold,local]
                endings[j]=dict(success=None,value_target=float(probability[0]+probability[3]),terminal=0,
                    physical_steps=span,target_source='cross-fitted factual continuation',value_fold=fold)
        for j,key in enumerate(current):results[key]=endings[j]
        temporary=output/'training-data.tmp'
        torch.save(dict(model=model,completed=False,sampling_signature=signature,partial_results=results),temporary)
        temporary.replace(evidence)
        print(dict(phase='short_conditional_continuations',completed=len(results),total=len(results)+len(jobs)-begin-len(current)),flush=True)
    rollouts=[]
    for ri,(packet,arms) in enumerate(roots):
        record=root_records[ri];seed,n,agility=record['scene'];branches=[]
        for repetition in range(future_start,future_start+repetitions):
            reference=results[ri,repetition,0]
            for arm in range(1,len(arms)):
                branches.append(dict(packet=packet,left=arms[0],right=arms[arm],before=reference,
                    after=results[ri,repetition,arm],reference_action=arms[0],factual_outcome=reference,
                    observed_future=False,continuation_seed=repetition,root=(seed,record['id'],0,arm)))
        rollouts.append(dict(row=dict(seed=seed,defenders=n,agility=agility,root_id=record['id']),steps=[],branches=branches,
            simulation_steps=sum(results[ri,repetition,arm]['physical_steps'] for repetition in range(future_start,future_start+repetitions)
                for arm in range(len(arms)))))
    data=dict(model=dict(model,physical_intervention_span=span),rollouts=rollouts,completed=True,sampling_signature=signature,
        simulation_steps=sum(r['simulation_steps'] for r in rollouts),
        terminal_branches=sum(v['terminal']!=0 for v in results.values()),branches=len(results),
        target='Paired short intervention plus cross-fitted factual terminal continuation value.')
    torch.save(data,evidence);actor.cpu();return data


@torch.no_grad()
def collect_fusion_td_pairs(path,root_records,output,repetitions=8,horizon=10,batch_size=128,device='cuda:0',
        intervention_steps=1,future_start=0,resume_conditional=False,intervention_mode='reverse'):
    """Intervene on one boat's fusion decision, then resume the policy."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True);destination=output/'training-data.pth'
    if intervention_mode not in ('reverse','maintain','blend','grid'):
        raise ValueError('Unknown conditional fusion intervention.')
    if destination.exists():
        existing=torch.load(destination,map_location='cpu',weights_only=False)
        if existing.get('intervention_mode','reverse')!=intervention_mode:
            raise ValueError('Existing conditional data uses a different intervention.')
        return existing
    actor=joint_load(str(path),Path(path).stat().st_mtime_ns).to(device)
    if resume_conditional and actor.cooperative_role_credit is not None:
        credit=actor.cooperative_role_credit
        if not credit.wait_span or tuple(credit.spans)!=(intervention_steps,) or credit.max_interventions:
            raise ValueError('Repeated conditional continuation requires the same fixed decision period and no inherited intervention-count limit.')
    roots=[];jobs=[];results={}
    for ri,root in enumerate(root_records):
        packet=joint_packet(root['environment'],root['observation'])
        reference=actor({k:torch.as_tensor(v[None],device=device) for k,v in packet.items()},
            deterministic=True,gate_only=True)[0][0].cpu().numpy()
        alternatives=[];subjects=[]
        candidates=([int(root['credit_subject'])] if 'credit_subject' in root else range(len(reference)))
        for subject in candidates:
            values=((0.,.5,1.) if intervention_mode=='grid' else
                (0. if reference[subject,2]>.5 else 1.,) if intervention_mode=='reverse' else
                (.5,) if intervention_mode=='blend' else (float(reference[subject,2]),))
            for gate in values:
                alternate=reference.copy();alternate[subject,2]=gate
                # Holding the present value is distinct from the factual
                # policy, which remains free to replan throughout the option.
                alternatives.append(alternate);subjects.append(subject)
        roots.append((packet,reference,alternatives,list(subjects)))
        for repeat in range(repetitions):
            for arm in range(1+len(alternatives)):jobs.append((ri,repeat,arm))
    for begin in range(0,len(jobs),batch_size):
        current=jobs[begin:begin+batch_size];envs=[];observations=[];random_states=[]
        active=list(range(len(current)));returns=np.zeros(len(current))
        continued_roles=np.full((len(current),len(roots[0][1])),-1.,dtype=np.float32)
        for ri,repeat,arm in current:
            root=root_records[ri];envs.append(copy.deepcopy(root['environment']));observations.append(root['observation'].copy())
            future=int(np.random.SeedSequence([root['scene'][0],root['id'],repeat+future_start,8311771]).generate_state(1)[0])
            random_states.append(np.random.RandomState(future).get_state())
        for step in range(horizon):
            if not active:break
            if step==0:
                actions=np.stack([roots[current[j][0]][1] if current[j][2]==0 else
                    roots[current[j][0]][2][current[j][2]-1] for j in active])
            else:
                records=[joint_packet(envs[j],observations[j]) for j in active]
                packet={k:torch.as_tensor(np.stack([record[k] for record in records]),device=device) for k in records[0]}
                continuation_boundary=resume_conditional and step>=intervention_steps and step%intervention_steps==0
                if actor.conditional_commitment and step<intervention_steps:
                    committed=np.full((len(active),len(roots[0][1])),-1.,dtype=np.float32)
                    for index,j in enumerate(active):
                        ri,_,arm=current[j]
                        if arm:
                            subject=roots[ri][3][arm-1]
                            committed[index,subject]=roots[ri][2][arm-1][subject,2]
                    packet['_committed_gates']=torch.as_tensor(committed,device=device)
                elif actor.conditional_commitment and resume_conditional and step>=intervention_steps and not continuation_boundary:
                    packet['_committed_gates']=torch.as_tensor(continued_roles[active],device=device)
                decisions=actor(packet,deterministic=True,gate_only=not continuation_boundary)[0].cpu().numpy()
                if continuation_boundary:
                    continued_roles[active]=decisions[:,:,3] if decisions.shape[-1]>3 else -1.
                actions=decisions[:,:,:3].copy()
                if step<intervention_steps:
                    for index,j in enumerate(active):
                        ri,_,arm=current[j]
                        if arm:
                            subject=roots[ri][3][arm-1]
                            actions[index,subject,2]=roots[ri][2][arm-1][subject,2]
                elif resume_conditional:
                    held=continued_roles[active]
                    actions[:,:,2]=np.where(held>=0.,held,actions[:,:,2])
            following=[]
            for j,action in zip(active,actions):
                np.random.set_state(random_states[j]);obs,reward,done,_=joint_task_step(envs[j],action,.995,.001)
                observations[j]=obs;random_states[j]=np.random.get_state();returns[j]+=(.995**step)*reward
                if done or step+1==horizon:
                    results[current[j]]=dict(reward=float(returns[j]),discount=.995**(step+1),
                        next=None if done else joint_packet(envs[j],obs),physical_steps=step+1,
                        terminal=int(done),success=int(done>2) if done else None)
                else:following.append(j)
            active=following
    pairs=[]
    for ri,(packet,left,alternatives,subjects) in enumerate(roots):
        for arm,(subject,right) in enumerate(zip(subjects,alternatives),1):
            for repeat in range(repetitions):pairs.append(dict(packet=packet,left=left,right=right,
                before=results[ri,repeat,0],after=results[ri,repeat,arm],scene=root_records[ri]['scene'],
                subject=subject,root_id=root_records[ri]['id'],future_repetition=repeat+future_start))
    data=dict(pairs=pairs,simulation_steps=sum(result['physical_steps'] for result in results.values()),
        intervention_steps=intervention_steps,continuation_steps=horizon,
        intervention_mode=intervention_mode,
        resume_conditional=resume_conditional,
        use='Matched single-vessel interventions followed by the same repeated conditional policy.' if resume_conditional else
            'Matched current-action interventions; all other coefficients are fixed at the intervention, and both arms resume the same policy.')
    torch.save(data,destination);actor.cpu();return data


def feedback_role_roots(datasets,lookbacks=(2.,6.),scene_seeds=None,decision_period=1):
    """Recover exact public decision states along recorded factual trajectories.

    Event-relative sampling is confined to training. Execution sees only the
    current measurement, and every root is labelled with independently drawn
    future noise. Successful paths provide the same negative-control states.
    """
    from envs.TADgame import TADEnv
    records=[]
    for dataset in datasets:
        for rollout in dataset['rollouts']:
            row=rollout['row'];steps=rollout['steps']
            if scene_seeds is not None and row['seed'] not in scene_seeds:continue
            if any(step['physical_steps']!=1 for step in steps):
                raise ValueError('Feedback credit roots require per-step factual actions.')
            indices={max(1,len(steps)-int(round(seconds/.2))) for seconds in lookbacks}
            if decision_period>1:indices={index//decision_period*decision_period for index in indices}
            indices={index for index in indices if index<len(steps)}
            np.random.seed(row['seed'])
            env=TADEnv(row['defenders'],protocol=dataset['model']['environment_protocol'])
            obs,_=env.reset(agility=row['agility'],noisy_agility=False)
            if row.get('future_seed') is not None:np.random.seed(row['future_seed'])
            for index,step in enumerate(steps):
                if index in indices:
                    snapshot=copy.deepcopy(env)
                    for name in ('Pos_Att','Phi_Att','Pos_Def','Phi_Def','Rewards'):
                        value=getattr(snapshot,name)
                        if np.ndim(value)>1:setattr(snapshot,name,value[-1:].copy())
                    records.append(dict(id=len(records),scene=(row['seed'],row['defenders'],row['agility']),
                        environment=snapshot,observation=obs.copy(),random_state=np.random.get_state()))
                obs,_,done,_=joint_task_step(env,step['action'][:,:3],1.,0.)
                if bool(done)!=bool(step['done']):
                    raise ValueError('Recorded factual physics did not reproduce at the requested credit root.')
    return records


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


def joint_effect_support(datasets):
    from conditional_fusion_value import public_return_context
    effects={}
    for dataset in datasets:
        for rollout in dataset['rollouts']:
            for branch in rollout['branches']:
                if branch['before']['success'] is None or branch['after']['success'] is None:continue
                span=float(branch['left'][0,4])
                context=np.append(public_return_context(mechanism_tensor(branch['packet']))[0].numpy(),span/10.)
                key=np.round(context,5).tobytes()
                if key not in effects:effects[key]=(context,{})
                edge=(tuple(branch['left'][:,3]),tuple(branch['right'][:,3]))
                effects[key][1].setdefault(edge,[]).append(branch['after']['success']-branch['before']['success'])
    return dict(contexts=np.asarray([x[0] for x in effects.values()]),
        limits=[float(max(abs(np.mean(y)) for y in x[1].values())>.01) for x in effects.values()],bandwidth=.3)


def joint_action_support(datasets):
    from conditional_fusion_value import public_return_context
    groups={}
    for dataset in datasets:
        for rollout in dataset['rollouts']:
            roots={}
            for branch in rollout['branches']:roots.setdefault(tuple(branch['root'][:2]),[]).append(branch)
            for branches in roots.values():
                first=branches[0];packet=mechanism_tensor(first['packet'])
                context,order=public_return_context(packet,True);order=order[0].numpy()
                span=float(first['left'][0,4]);context=np.append(context[0].numpy(),span/10.)
                key=np.round(context,5).tobytes();n=len(order)
                if key not in groups:groups[key]=(context,np.zeros(4**n,dtype=bool))
                for branch in branches:
                    for side in ('left','right'):
                        gates=branch[side][order,3]
                        codes=np.where(gates<0.,0,1+2*gates).astype(int)
                        groups[key][1][int((codes*4**np.arange(n)).sum())]=True
    return dict(contexts=np.asarray([v[0] for v in groups.values()]),
        programs=np.asarray([v[1] for v in groups.values()]),bandwidth=.3)


def joint_supported_value(args,pool):
    """Calibrate conditional improvement against observed continuation value."""
    from conditional_fusion_value import public_return_context
    model=torch.load(args.start,map_location='cpu',weights_only=False)
    data=torch.load(args.data,map_location='cpu',weights_only=False)
    contexts=[];trajectories=[];returns=[]
    for rollout in data['rollouts']:
        if not rollout['steps']:continue
        index=len(returns);returns.append(float(rollout['row']['success']))
        for step in rollout['steps']:
            contexts.append(public_return_context(mechanism_tensor(step['packet']))[0].numpy())
            trajectories.append(index)
    args.output.mkdir(parents=True,exist_ok=True);history=[];best=(-1.,-1.)
    if args.joint_duration_reference:model['shared_response_value']['duration_reference']=True
    if args.joint_contrast_support:
        model['conditional_effect_support']=joint_effect_support([data,
            *[torch.load(path,map_location='cpu',weights_only=False) for path in (args.references or [])]])
    if args.joint_action_support:
        model['response_action_support']=joint_action_support([data,
            *[torch.load(path,map_location='cpu',weights_only=False) for path in (args.references or [])]])
    scenes=bank(args.selection_bank,args.selection_per_cell,tuple((n,a) for n in args.teams for a in args.agilities))
    for bandwidth in args.joint_local_bandwidths:
        model['factual_return_support']=dict(contexts=np.asarray(contexts),trajectories=trajectories,
            returns=returns,bandwidth=bandwidth)
        model['factual_support_data']=str(args.data)
        for margin in args.joint_response_margins:
            model['shared_response_value']['margin']=margin
            path=args.output/'current.pth';torch.save(model,path)
            tasks=[(s,str(path),path.stat().st_mtime_ns,False,0,0,0,0,
                None if repeat==0 else int(np.random.SeedSequence([s[0],repeat,20261014]).generate_state(1)[0]))
                for repeat in range(args.joint_future_repetitions) for s in scenes]
            rows=list(pool.map(joint_episode,tasks,chunksize=1));summary=summarize(rows)
            trial=dict(bandwidth=bandwidth,margin=margin,development=summary,rows=rows);history.append(trial)
            score=(summary['success'],summary['capture'])
            if score>best:best=score;model['selection']=trial;torch.save(model,args.output/'policy.pth')
            save_json(args.output/'development.json',dict(history=history,best_success=best[0],
                continuation_trajectories=len(returns),continuation_states=len(contexts)))
            print(dict(phase='supported_response_development',bandwidth=bandwidth,margin=margin,
                success=score[0],collision=summary['collision']),flush=True)


def joint_shared_value(args,pool):
    """Learn shared response factors directly from paired terminal outcomes."""
    from conditional_fusion_value import SharedResponseValue, fusion_response_factors, FactualReturnSupport, public_return_context
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    model=torch.load(args.start,map_location='cpu',weights_only=False)
    require_clean_mechanism_start(model,args)
    paired_gain=args.joint_paired_advantage
    if args.joint_global_response and (not paired_gain or args.joint_action_support):
        raise ValueError('Global paired advantages use their shared critic without a state-neighborhood action gate.')
    if paired_gain and (args.joint_no_cf or args.joint_outcome_strata or args.joint_project_returns
            or args.joint_response_aggregation!='graph'):
        raise ValueError('Paired graph advantages use a centered graph critic and their original signed targets.')
    if paired_gain and any(model.get(key) is not None for key in
            ('shared_response_value','joint_intervention_value','joint_intervention_values','response_base',
             'physical_conditional_value','physical_conditional_values')):
        raise ValueError('A shared paired-gain critic currently requires a free-feedback continuation without inherited M2 options.')
    model.pop('shared_response_value',None)
    actor=copy.deepcopy(joint_load(str(args.start),args.start.stat().st_mtime_ns))
    if args.joint_no_interaction:ablate_joint_mechanisms(actor,True,False,True)
    if args.joint_residual_response:
        model['response_base']=str(args.start)
        model.pop('factual_return_support',None)
        model.pop('conditional_effect_support',None)
        actor.response_base=joint_load(str(args.start),args.start.stat().st_mtime_ns)
    actor.response_values=nn.ModuleList();actor.eval()
    actor.response_feature_version=args.joint_response_feature_version
    if args.without_peer:actor.peer=False;model['peer']=False
    datasets=[torch.load(path,map_location='cpu',weights_only=False) for path in [args.data,*(args.references or [])]]
    if args.joint_global_response:
        model.pop('factual_return_support',None)
        model.pop('response_action_support',None)
        model.pop('conditional_effect_support',None)
    if model.get('factual_return_support') is None and not args.joint_global_response:
        contexts=[];trajectories=[];returns=[]
        for dataset in datasets:
            for rollout in dataset['rollouts']:
                if not rollout['steps']:continue
                index=len(returns);returns.append(float(rollout['row']['success']))
                for step in rollout['steps']:
                    contexts.append(public_return_context(mechanism_tensor(step['packet']))[0].numpy());trajectories.append(index)
        model['factual_return_support']=dict(contexts=np.asarray(contexts),trajectories=trajectories,returns=returns,bandwidth=.5)
        model['factual_support_data']=str(args.data)
    actor.response_anchor=(FactualReturnSupport(**model['factual_return_support'])
        if model.get('factual_return_support') is not None else None)
    stratified=args.joint_outcome_strata and not args.joint_no_cf
    heads=2 if stratified else 1
    features=[];records=[];priors=[];observed_edges_excluded=0
    def encode_context(packet,intent,span):
        features.append(fusion_response_factors(actor,packet,intent,span)[0].numpy())
        priors.append(float(actor.response_anchor(public_return_context(packet))[0][0])
            if actor.response_anchor is not None else 0.)
    for data in datasets:
        span=data['model'].get('physical_intervention_span',data['model'].get('physical_span',10))
        if span not in (args.joint_response_spans or [args.joint_physical_span]):continue
        for rollout in data['rollouts']:
            groups={}
            for branch in rollout['branches']:groups.setdefault(tuple(branch['root'][:2]),[]).append(branch)
            for group in groups.values():
                first=group[0];packet=mechanism_tensor(first['packet']);intent=torch.tensor(first['reference_action'][None])
                encode_context(packet,intent,span)
                observations={};seen_edges={}
                reference=first['reference_action']
                matched_reference=not (reference[:,3]>=0.).any() or int(reference[0,4])==span
                for branch in group:
                    if branch['before']['success'] is None or branch['after']['success'] is None:continue
                    left=np.where(branch['left'][:,3]<0.,0,1+2*branch['left'][:,3]).astype(int)
                    right=np.where(branch['right'][:,3]<0.,0,1+2*branch['right'][:,3]).astype(int)
                    edge=tuple(branch['root'][2:]);repetition=seen_edges.get(edge,0);seen_edges[edge]=repetition+1
                    observed_future=branch.get('observed_future',data['model'].get('observed_future',False) and repetition==0)
                    if args.joint_independent_futures and observed_future:
                        # Hindsight chose this root because of its factual
                        # outcome. Reusing that same future biases a rescue
                        # estimate; fresh common-noise continuations estimate
                        # the conditional effect without that selection bias.
                        observed_edges_excluded+=1
                        continue
                    mode=-1
                    if stratified:
                        if observed_future:mode=int(rollout['row']['success'])
                        elif matched_reference:mode=int(branch['reference_outcome']['success'])
                    key=(tuple(left),tuple(right),mode)
                    factual=0.
                    if paired_gain:
                        outcome=branch.get('factual_outcome') or branch.get('reference_outcome')
                        if outcome is None or outcome['success'] is None:
                            raise ValueError('Paired gain fitting requires the same-future factual continuation.')
                        factual=float(outcome['success'])
                    observations.setdefault(key,[]).append((branch['before']['success']-factual,branch['after']['success']-factual))
                rows=[(np.asarray(a),np.asarray(b),*np.mean(y,axis=0),mode) for (a,b,mode),y in observations.items()]
                if not rows:
                    features.pop();priors.pop();continue
                records.append(rows)
            # Factual returns anchor the same shared value in successful
            # encounters. They prevent a failure-root-only data set from
            # treating every encountered state as needing an intervention.
            steps=rollout.get('steps',[])
            if steps and rollout['row'].get('success') is not None:
                def proximity(entry):
                    position=entry['packet']['motion'][:,:2]
                    return (np.linalg.norm(position[:,None]-position[None],axis=-1)+1e6*np.eye(len(position))).min()
                anchors={0,min(range(len(steps)),key=lambda i:proximity(steps[i]))}
                for index in anchors:
                    entry=steps[index];action=entry['action']
                    if action.shape[-1]>3 and (action[:,3]>=0.).any() and int(action[0,4])!=span:continue
                    packet=mechanism_tensor(entry['packet']);intent=torch.tensor(action[None])
                    encode_context(packet,intent,span)
                    choices=(np.where(action[:,3]<0.,0,1+2*action[:,3]).astype(int)
                        if action.shape[-1]>3 else np.zeros(len(action),dtype=int))
                    y=0. if paired_gain else float(rollout['row']['success'])
                    records.append([(choices,choices,y,y,int(y) if stratified else -1)])
    if not records:raise ValueError('No execution-matched physical intervention rows.')
    # Pool repeated public contexts before weighting effects. Weighting each
    # realized success separately would bias a noisy, rare improvement upward.
    # This averages common-noise return differences, not scene-specific actions.
    grouped_contexts={}
    for feature,rows,prior in zip(features,records,priors):
        key=hashlib.sha256(np.round(feature,5).tobytes()).digest()
        if key not in grouped_contexts:grouped_contexts[key]=(feature,{},prior)
        entries=grouped_contexts[key][1]
        for a,b,ya,yb,mode in rows:entries.setdefault((tuple(a),tuple(b),mode),[]).append((ya,yb))
    original_contexts=len(records);features=[];records=[];priors=[]
    for feature,entries,prior in grouped_contexts.values():
        features.append(feature);priors.append(prior)
        records.append([(np.asarray(a),np.asarray(b),*np.mean(y,axis=0),mode) for (a,b,mode),y in entries.items()])
    projection=None
    if args.joint_project_returns:
        from conditional_fusion_value import project_intervention_returns
        records,projection=project_intervention_returns(records,0. if args.joint_no_cf else args.difference_weight)
        print(dict(phase='conditional_cycle_projection',**projection),flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    encoded=torch.tensor(np.asarray(features),device=device)
    n=records[0][0][0].shape[0];pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
    left=[];right=[];targets=[];root=[];modes=[]
    for r,rows in enumerate(records):
        for a,b,ya,yb,mode in rows:
            root.append(r);left.append([a[i]*4+a[j] for i,j in pairs]);right.append([b[i]*4+b[j] for i,j in pairs]);targets.append((ya,yb))
            modes.append(mode)
    left=torch.tensor(left,device=device);right=torch.tensor(right,device=device)
    targets=torch.tensor(targets,dtype=torch.float32,device=device);root=torch.tensor(root,device=device)
    modes=torch.tensor(modes,device=device);priors=torch.tensor(priors,device=device)
    informative=torch.nonzero((targets.abs() if paired_gain else targets).amax(-1)>0.).squeeze(-1)
    changed=torch.nonzero((targets[:,1]-targets[:,0]).abs()>0.).squeeze(-1)
    saved=[];losses=[]
    print(dict(phase='shared_response_fit',roots=len(records),edges=len(root),features=encoded.shape[-1],conditional=not args.joint_no_cf),flush=True)
    for member in range(3):
        torch.manual_seed(args.seed+member)
        value=SharedResponseValue(encoded.shape[-1],args.joint_response_width,heads,args.joint_response_aggregation).to(device)
        opt=torch.optim.AdamW(value.parameters(),lr=8e-4,weight_decay=.005)
        for step in range(args.updates):
            ids=torch.as_tensor(rng.integers(len(root),size=256),device=device)
            if len(informative):
                ids[128:]=informative[torch.as_tensor(rng.integers(len(informative),size=128),device=device)]
            if len(changed) and not args.joint_no_cf:
                ids[192:]=changed[torch.as_tensor(rng.integers(len(changed),size=64),device=device)]
            selected,inverse=torch.unique(root[ids],return_inverse=True)
            table=value(encoded[selected])
            state=(targets.new_zeros((len(ids),heads)) if paired_gain else value.state_value(encoded[selected])[inverse])
            if heads==1 and state.ndim==1:state=state[:,None]
            if args.joint_response_aggregation in ('joint','graph'):
                indices=torch.stack((left[ids],right[ids]),1)
                if paired_gain:
                    indices=torch.cat((indices,torch.zeros_like(indices[:,:1])),1)
                    joint=value.joint_actions(table[inverse],indices)
                    conditional=joint[:,:2]-joint[:,2:3]
                else:conditional=value.joint_actions(table[inverse],indices)+state[:,None]
            else:
                if heads==1:table=table[...,None]
                table=table.flatten(2,3)[inverse]
                def aggregate(index):
                    pair=table.gather(2,index[:,:,None,None].expand(-1,-1,1,heads)).squeeze(2)
                    return (-torch.logsumexp(-pair,1)+np.log(len(pairs))
                        if args.joint_response_aggregation=='bottleneck' else pair.sum(1)/np.sqrt(len(pairs)))
                qa=aggregate(left[ids]);qb=aggregate(right[ids])
                conditional=torch.stack((qa,qb),1)+state[:,None]
            if heads==2:
                known=conditional.gather(2,modes[ids].clamp_min(0)[:,None,None].expand(-1,2,1)).squeeze(-1)
                weight=priors[root[ids],None]
                marginal=(1.-weight)*conditional[...,0]+weight*conditional[...,1]
                prediction=torch.where(modes[ids,None]>=0,known,marginal)
            else:prediction=conditional[...,0]
            loss=(prediction-targets[ids]).square().mean()
            if not args.joint_no_cf:
                delta=targets[ids,1]-targets[ids,0]
                predicted=prediction[:,1]-prediction[:,0]
                # Paired noise isolates the altered vessel's effect; retain
                # ties while increasing the mass of informative interventions.
                weight=torch.ones_like(delta) if paired_gain else 1.+4.*delta.abs()
                loss=loss+args.difference_weight*(weight*(predicted-delta).square()).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(value.parameters(),3.);opt.step()
        saved.append({k:v.detach().cpu() for k,v in value.state_dict().items()});losses.append(float(loss.detach()))
        print(dict(phase='shared_response_member',member=member,loss=losses[-1]),flush=True)
    model['shared_response_value']=dict(states=saved,features=encoded.shape[-1],width=args.joint_response_width,heads=heads,
        outcome_strata=stratified,paired_gain=paired_gain,
        global_response=args.joint_global_response,
        independent_futures=args.joint_independent_futures,observed_edges_excluded=observed_edges_excluded,
        duration_reference=True,
        feature_version=args.joint_response_feature_version,aggregation=args.joint_response_aggregation,link='identity',
        span=args.joint_physical_span,spans=args.joint_response_spans or [args.joint_physical_span],margin=.15,conditional=not args.joint_no_cf,
        roots=len(records),original_contexts=original_contexts,edges=len(root),losses=losses,
        return_projection=projection,
        difference_weight=0. if args.joint_no_cf else args.difference_weight,
        training_simulation_steps=sum(data.get('simulation_steps',0) for data in datasets),
        updates=args.updates,seed=args.seed,
        training_data=[str(args.data),*[str(p) for p in (args.references or [])]])
    if args.joint_contrast_support:model['conditional_effect_support']=joint_effect_support(datasets)
    elif args.joint_no_cf:model.pop('conditional_effect_support',None)
    if args.joint_action_support:
        model['response_action_support']=joint_action_support(datasets)
        model['response_action_support']['transport']=args.joint_response_aggregation=='graph'
    args.output.mkdir(parents=True,exist_ok=True);best=(-1.,-1.);history=[]
    if args.critic_only:
        model.pop('selection',None)
        model['response_training_only']=True
        torch.save(model,args.output/'policy.pth')
        print(dict(phase='response_fit_complete',model=str(args.output/'policy.pth'),losses=losses),flush=True)
        return
    scenes=bank(args.selection_bank,args.selection_per_cell,tuple((n,a) for n in (args.selection_teams or args.teams) for a in args.agilities))
    for margin in args.joint_response_margins:
        model['shared_response_value']['margin']=margin
        path=args.output/'current.pth';torch.save(model,path)
        tasks=[(s,str(path),path.stat().st_mtime_ns,False,0,0,0,0,
            None if repeat==0 else int(np.random.SeedSequence([s[0],repeat,20261014]).generate_state(1)[0]))
            for repeat in range(args.joint_future_repetitions) for s in scenes]
        rows=list(pool.map(joint_episode,tasks,chunksize=1));summary=summarize(rows)
        trial=dict(margin=margin,development=summary,rows=rows);history.append(trial)
        score=(summary['success'],summary['capture'])
        if score>best:best=score;model['selection']=trial;torch.save(model,args.output/'policy.pth')
        save_json(args.output/'development.json',dict(history=history,best_success=best[0],roots=len(records),edges=len(root),losses=losses))
        print(dict(phase='shared_response_development',margin=margin,success=score[0],collision=summary['collision']),flush=True)


def joint_graph_value(args,pool):
    """Fit full joint potentials without projecting a rescue onto pair factors."""
    from conditional_fusion_value import public_return_context,FactualReturnSupport
    torch.set_num_threads(1)
    datasets=[torch.load(path,map_location='cpu',weights_only=False)
        for path in [args.data,*(args.references or [])]]
    if any(data.get('completed') is False for data in datasets):
        raise ValueError('Intervention collection is still in progress; fit after its completed checkpoint is saved.')
    model=torch.load(args.start,map_location='cpu',weights_only=False)
    joint_value_objectives(model)
    require_clean_mechanism_start(model,args)
    if any(any(data['model'].get(key)!=model.get(key) for key in
               ('source_sha256','environment_protocol','proposal_initialization_sha256')) for data in datasets):
        raise ValueError('Conditional returns must belong to the same base policy and task protocol.')
    if any(data['model'].get('preference_strength',0.)!=model.get('preference_strength',0.) for data in datasets):
        raise ValueError('Conditional returns must use the deployed temporal preference decoder.')
    interaction_matches=[joint_interaction_identity(data['model'])==joint_interaction_identity(model)
        for data in datasets]
    if not all(interaction_matches) and not args.shared_data:
        raise ValueError('Branch outcomes use a different candidate-interaction predecessor. '
            'Collect under the selected predecessor, or explicitly select --shared-data for a fixed-data objective comparison.')
    continuation_identity=joint_continuation_identity(model)
    continuation_matches=[joint_continuation_identity(data['model'])==continuation_identity for data in datasets]
    if (args.joint_residual_response and not all(continuation_matches) and not args.shared_data
            and not (args.joint_no_cf or args.loss_mode=='absolute')):
        raise ValueError('Paired conditional updates require outcomes from their executing predecessor. '
            'Resample that predecessor, or explicitly use --shared-data for a fixed-data learning comparison.')
    if args.without_peer:model['peer']=False
    contexts=[];trajectories=[];returns=[]
    for rollout in datasets[0]['rollouts']:
        if not rollout['steps']:continue
        index=len(returns);returns.append(float(rollout['row']['success']))
        for step in rollout['steps']:
            contexts.append(public_return_context(mechanism_tensor(step['packet']),version=args.joint_credit_context,
                reference_action=torch.as_tensor(step['action'][None],dtype=torch.float32))[0].numpy())
            trajectories.append(index)
    anchor=dict(contexts=np.asarray(contexts),trajectories=trajectories,returns=returns,bandwidth=.5)
    reference=FactualReturnSupport(**anchor)
    grouped={};timelines={};factual_support={};clock_modes={};opportunity_samples={};factual_samples={};measured_support={};root_count=0;edge_count=0;clock_filtered=0
    absolute_regression=args.loss_mode=='absolute'
    if args.joint_coma_credit and not absolute_regression:
        raise ValueError('The counterfactual-critic replacement must fit ordinary returns at every stage.')
    from conditional_fusion_value import temporal_program_options
    program_options=(temporal_program_options(args.joint_temporal_blending_levels)
        if args.joint_temporal_programs else None)
    if args.joint_prefix_refinement:
        if args.joint_temporal_programs or not args.joint_residual_response:
            raise ValueError('Relative prefix values refine one fixed predecessor and preserve its planned suffix.')
        program_options=np.asarray([[-2.,-2.],[-1.,-2.],[0.,-2.],[.5,-2.],[1.,-2.]])
        if args.joint_remainder_refinement:
            program_options=np.asarray([[-2.,-2.],[-1.,-1.],[0.,0.],[.5,.5],[1.,1.]])
    if args.joint_midpoint_refinement and (not args.joint_prefix_refinement or args.joint_initial_root):
        raise ValueError('Midpoint values condition on the observed suffix state, not the initial state.')
    if args.joint_remainder_refinement and not args.joint_midpoint_refinement:
        raise ValueError('Whole-remainder values replan an active option at its intermediate observation.')
    radix=4 if program_options is None else len(program_options)
    if args.joint_temporal_programs and (args.joint_role_transport or args.joint_credit_locality):
        raise ValueError('Temporal programs use full-plan conditional potentials in the current fleet.')
    stratified=args.joint_outcome_strata and not args.joint_no_cf and not absolute_regression
    paired=(args.joint_paired_advantage and not absolute_regression and not stratified
        and not args.joint_binomial_value and not args.joint_success_potentials and not args.joint_paired_nuisance
        and not args.joint_conditional_calibration)
    anchored=args.joint_anchored_value
    shared_examples=[];shared_models={};shared_training={};context_metrics={};metric_training={};program_offsets={}
    if args.joint_context_metric and (stratified or args.joint_role_transport or args.joint_credit_locality):
        raise ValueError('A learned context metric uses whole-program, unstratified observations.')
    if args.joint_conditional_calibration and (stratified or anchored or absolute_regression
            or args.joint_paired_nuisance or args.joint_binomial_value or not args.joint_factual_calibration
            or args.joint_unary_value or args.joint_factor_value or args.joint_program_kernel):
        raise ValueError('Conditional calibration needs unstratified paired outcomes and their distinct factual futures.')
    if (args.joint_factor_value or args.joint_program_kernel or args.joint_unary_value) and (anchored or args.joint_credit_confidence
            or args.joint_context_regularization or args.joint_shared_potentials
            or args.joint_binomial_value or args.joint_paired_nuisance):
        raise ValueError('Shared vessel/pair values use their own coefficient fit and compositional support.')
    if sum((args.joint_factor_value,args.joint_program_kernel,args.joint_unary_value))>1:
        raise ValueError('Choose one shared representation for complete-program values.')
    if args.joint_paired_nuisance and (stratified or anchored or args.joint_credit_confidence
            or args.joint_context_regularization or args.joint_shared_potentials or absolute_regression):
        raise ValueError('Paired nuisance estimation fits absolute program values and local contrasts jointly.')
    if args.joint_binomial_value and (stratified or anchored or args.joint_credit_confidence
            or args.joint_context_regularization or args.joint_shared_potentials):
        raise ValueError('Binomial program values use absolute outcomes and paired conditional odds.')
    if args.joint_shared_potentials and (stratified or anchored or args.joint_credit_confidence
            or args.joint_context_regularization or args.joint_credit_locality or args.joint_role_transport):
        raise ValueError('Shared potentials use direct, unstratified complete-program values.')
    if args.joint_program_completion and not args.joint_shared_potentials:
        raise ValueError('Compositional completion requires a learned shared program value.')
    if args.joint_paired_likelihood and (not args.joint_shared_potentials or paired):
        raise ValueError('Paired likelihood requires shared absolute success potentials.')
    if args.joint_block_likelihood and (not args.joint_shared_potentials or paired or absolute_regression
            or args.joint_no_cf or not args.joint_unique_futures):
        raise ValueError('Joint outcome likelihood requires unique paired futures and conditional success potentials.')
    if args.joint_response_precision and not args.joint_block_likelihood:
        raise ValueError('Response precision requires the paired outcome matrix.')
    if anchored and (not args.joint_paired_advantage or stratified or args.joint_credit_confidence or args.joint_context_regularization):
        raise ValueError('Anchored success learning requires unstratified paired outcomes and its local joint fit.')
    if args.joint_credit_confidence and (not paired or stratified):
        raise ValueError('Posterior conditional credit requires unstratified paired continuation outcomes.')
    if args.joint_independent_futures and args.joint_factual_support and not args.joint_paired_support:
        raise ValueError('Unbiased future fitting cannot gate actions on a hindsight-factual rescue.')
    if args.joint_paired_support and not (args.joint_paired_advantage and args.joint_unique_futures):
        raise ValueError('Paired support requires distinct futures and their unchanged-policy outcomes.')
    if args.joint_factual_calibration and not args.joint_unique_futures:
        raise ValueError('Factual calibration counts distinct continuation futures, not repeated graph edges.')
    if paired and stratified:raise ValueError('Paired advantages already remove factual future outcomes.')
    for dataset_index,data in enumerate(datasets):
        span=data['model'].get('physical_intervention_span',data['model'].get('physical_span',10))
        if span not in (args.joint_response_spans or [args.joint_physical_span]):continue
        if args.joint_factual_support and (not args.joint_paired_advantage or not data['model'].get('observed_future',False)):
            raise ValueError('Factual rescue support requires paired, observed-future interventions.')
        reversal=data['model'].get('physical_intervention_reversal',data['model'].get('physical_reversal',False))
        if bool(data['model'].get('prefix_refinement',False))!=args.joint_prefix_refinement:
            raise ValueError('Relative prefix interventions and absolute whole-window programs have different value objects.')
        if bool(data['model'].get('midpoint_refinement',False))!=args.joint_midpoint_refinement:
            raise ValueError('Midpoint and initial-boundary intervention values have different decision contexts.')
        if bool(data['model'].get('remainder_refinement',False))!=args.joint_remainder_refinement:
            raise ValueError('Prefix and whole-remainder interventions must retain their sampled control programs.')
        if args.joint_midpoint_refinement and data['model'].get('option_refinement_fraction',.5)!=args.joint_refinement_fraction:
            raise ValueError('Intermediate control values must retain their sampled decision time.')
        allowed_delays=({args.joint_refinement_steps,0} if args.joint_refinement_checkpoints
            else {args.joint_refinement_steps})
        if args.joint_midpoint_refinement and data['model'].get('option_refinement_steps',0) not in allowed_delays:
            raise ValueError('Intermediate values must retain their sampled observation delay.')
        if args.joint_midpoint_refinement and bool(data['model'].get('phase_preserving_refinement',False))!=args.joint_phase_preserving_refinement:
            raise ValueError('Intermediate values must retain the sampled predecessor phase schedule.')
        if args.joint_temporal_programs:reversal=False
        continuation_has_options=any(data['model'].get(key) is not None for key in
            ('physical_conditional_value','physical_conditional_values','shared_response_value',
             'joint_intervention_value','joint_intervention_values','response_base'))
        clock=bool(data['model'].get('fixed_intervention_clock',False))
        if args.joint_source_reference:
            if continuation_has_options or data['model'].get('gate_period',1)!=1:
                raise ValueError('A common source reference must update each tick without learned options.')
            # At the common initialization, an all-free window executes the
            # same source controller at every tick regardless of its boundary.
            # Its terminal outcome is therefore a factual source observation.
            clock=True
        if (span,reversal) in clock_modes and clock_modes[span,reversal]!=clock:
            raise ValueError('A duration must retain one intervention-clock definition across its samples.')
        clock_modes[span,reversal]=clock
        def option_clock(action):
            period=data['model'].get('gate_period',1)
            duration=int(action[0,4]) if action.shape[-1]>4 else span
            if data['model'].get('prefix_refinement',False) or action.shape[-1]==9:return duration
            wait=action.shape[-1]>5 and (action[:,3]<0.).all() and (action[:,5]==-2.).all()
            active=(action[:,3]>=0.).any() or (action.shape[-1]>5 and (action[:,5]>=0.).any())
            return max(period,duration) if active or wait else period
        for rollout_index,rollout in enumerate(data['rollouts']):
            roots={}
            for branch in rollout['branches']:roots.setdefault(tuple(branch['root'][:2]),[]).append(branch)
            for branches in roots.values():
                context,order=public_return_context(mechanism_tensor(branches[0]['packet']),True,args.joint_credit_context,
                    reference_action=torch.as_tensor(branches[0]['reference_action'][None],dtype=torch.float32))
                context=context[0].numpy();order=order[0].numpy();n=len(order)
                key=(span,reversal,np.round(context,5).tobytes())
                trajectory=(int(rollout['row']['seed']),rollout['row'].get('future_seed'))
                timelines.setdefault(trajectory,{})[key]=.2*branches[0]['root'][1]
                if key not in grouped:grouped[key]=(context,{})
                _,strata=grouped[key]
                seen_edges={};observed={}
                for branch in branches:
                    if branch['before']['success'] is None or branch['after']['success'] is None:continue
                    factual_outcome=branch.get('factual_outcome')
                    if (paired or args.joint_source_reference) and factual_outcome is None and not continuation_has_options:
                        # Without a learned option, the all-free reference arm
                        # executes the original per-step controller exactly.
                        # A learned option instead needs its own factual arm
                        # because its commitment duration must be preserved.
                        if (branch['reference_action'][:,3]<0.).all():
                            factual_outcome=branch['reference_outcome']
                    if paired and (factual_outcome is None or factual_outcome['success'] is None):
                        raise ValueError('Paired advantages require same-future factual continuation rollouts.')
                    if continuation_has_options and option_clock(branch['left'])!=option_clock(branch['right']):
                        # An all-free option can release the next learned joint
                        # decision earlier. Its contrast would then change the
                        # teammates' decision clock as well as one vessel's gate.
                        # Keep fixed-clock edges for individual causal credit;
                        # factual rollouts separately anchor the unchanged policy.
                        clock_filtered+=1
                        continue
                    # A stratum describes the unchanged predecessor's task
                    # outcome. A timed all-feedback intervention is a different
                    # program and must not replace that factual condition.
                    mode=(int((factual_outcome if factual_outcome is not None else
                        branch['reference_outcome'])['success']) if stratified else -1)
                    if mode not in strata:strata[mode]=({}, {})
                    observations,contrasts=strata[mode]
                    edge=tuple(branch['root'][2:]);repeat=seen_edges.get(edge,0);seen_edges[edge]=repeat+1
                    observed_future=branch.get('observed_future',data['model'].get('observed_future',False) and repeat==0)
                    if args.joint_independent_futures and observed_future:
                        continue
                    if args.joint_independent_futures:
                        continuation_seed=branch.get('continuation_seed')
                        # Older collections did not retain their random key.
                        # Identical public roots reached by two factual paths
                        # may reuse that continuation. Conservatively cluster
                        # matching legacy ordinals instead of doubling their
                        # evidence merely because the root was visited twice.
                        sample_id=(('fresh',int(continuation_seed)) if continuation_seed is not None else
                            ('legacy',int(rollout['row']['seed']),int(branch['root'][1]),
                             int(branch.get('future_repetition',repeat+data['model'].get('future_start',0)))))
                    elif args.joint_unique_futures:
                        continuation_seed=branch.get('continuation_seed')
                        if observed_future:
                            sample_id=('observed',int(rollout['row']['seed']),rollout['row'].get('future_seed'))
                        elif continuation_seed is not None:sample_id=('fresh',int(continuation_seed))
                        else:sample_id=('legacy',int(rollout['row']['seed']),int(branch['root'][1]),repeat)
                    else:sample_id=(dataset_index,rollout_index,int(branch['root'][1]),repeat)
                    codes=[];outcomes=[]
                    for side,outcome in (('left','before'),('right','after')):
                        gates=branch[side][order,3]
                        if args.joint_prefix_refinement:
                            reference_gates=branch['reference_action'][order,3]
                            choices=np.where(np.isclose(gates,reference_gates),0,np.where(gates<0.,1,2+2*gates)).astype(int)
                        elif program_options is None:choices=np.where(gates<0.,0,1+2*gates).astype(int)
                        else:
                            phases=branch[side][order][:,[3,5]].copy();phases[gates<0.]=-1.
                            matches=np.abs(phases[:,None]-program_options[None]).max(-1)<1e-6
                            if not matches.any(-1).all():raise ValueError('Observed blending phases are outside the temporal alphabet.')
                            choices=matches.argmax(-1)
                        code=int((choices*radix**np.arange(n)).sum());y=float(branch[outcome]['success'])
                        factual_gain=(y-float(factual_outcome['success']) if factual_outcome is not None else 0.)
                        if args.joint_wait_crossfit or anchored or args.joint_factual_calibration or args.joint_paired_support:
                            if factual_outcome is None:
                                raise ValueError('Selectable waiting values require paired factual continuations.')
                            if args.joint_wait_crossfit or anchored or args.joint_paired_support:
                                opportunity_samples.setdefault(key,{}).setdefault(sample_id,{})[code]=factual_gain
                            factual_samples.setdefault(key,{})[sample_id]=float(factual_outcome['success'])
                        if paired and not anchored:y-=float(factual_outcome['success'])
                        if args.joint_paired_advantage and observed_future and factual_gain>0.:
                            factual_support.setdefault(key,set()).add(code)
                        codes.append(code);outcomes.append(y)
                        observed[mode,sample_id,code]=y
                    a,b=codes;delta=outcomes[1]-outcomes[0]
                    if a>b:a,b=b,a;delta=-delta
                    contrasts.setdefault((a,b),{}).setdefault(sample_id,[]).append(delta);edge_count+=1
                for (mode,sample_id,code),y in observed.items():
                    strata[mode][0].setdefault(code,{}).setdefault(sample_id,[]).append(y)
                root_count+=1
    if args.joint_factual_calibration and factual_samples:
        root_pool={}
        for key,samples in factual_samples.items():
            context=grouped[key][0];identity=np.round(context,5).tobytes()
            _,pooled=root_pool.setdefault(identity,(context,{}))
            for sample_id,outcome in samples.items():pooled.setdefault(sample_id,[]).append(outcome)
        # The same factual future may anchor several intervention windows.
        # Count it once when those windows share the same public context.
        roots=list(root_pool.values())
        anchor.update(root_contexts=np.asarray([context for context,_ in roots]),
            root_successes=np.asarray([sum(np.mean(values) for values in samples.values()) for _,samples in roots]),
            root_trials=np.asarray([len(samples) for _,samples in roots]),
            root_bandwidth=args.joint_context_bandwidth)
        reference=FactualReturnSupport(**anchor)
    fitted={};peaks={};context_systems=[]
    for key,(context,strata) in grouped.items():
        span,reversal=key[:2]
        factual=float(reference(torch.tensor(context[None]))[0][0])
        full=(np.broadcast_to(np.asarray([0.,1.],dtype=np.float32),(radix**n,2)).copy()
            if stratified else np.zeros(radix**n,dtype=np.float32))
        known=np.zeros(radix**n,dtype=bool)
        uncertainty=np.zeros_like(full)
        for mode,(observations,contrasts) in strata.items():
            paired_observations,paired_contrasts=observations,contrasts
            observations={code:np.asarray([np.mean(values) for values in samples.values()])
                for code,samples in observations.items()}
            contrasts={edge:np.asarray([np.mean(values) for values in samples.values()])
                for edge,samples in contrasts.items()}
            programs=sorted(observations);index={code:i for i,code in enumerate(programs)}
            absolute=np.asarray([np.mean(observations[code]) for code in programs])
            if args.joint_shared_potentials or args.joint_context_metric:
                shared_examples.append(dict(key=key,context=context,programs=programs,targets=absolute.copy(),
                    outcomes=[(index[code],future,float(np.mean(samples)))
                        for code,futures in paired_observations.items() for future,samples in futures.items()]
                        if args.joint_block_likelihood else [],
                    edges=[(index[a],index[b],float(samples.mean()),
                        float(len(samples)/(1.+4.*samples.var())) if args.joint_contrast_precision else
                        float(1.+4.*abs(samples.mean())),float((samples>0.).sum()),float((samples<0.).sum()))
                        for (a,b),samples in contrasts.items()]))
            if args.joint_credit_confidence:
                # A symmetric rescue/neutral/harm prior prevents a lone
                # favorable continuation from acting like a certain gain.
                # Only distinct paired futures at this state supply evidence;
                # repeated single-vessel edges do not add sample count.
                samples=[np.asarray(observations[code],dtype=float) for code in programs]
                absolute=np.asarray([values.sum()/(len(values)+2.) for values in samples])
                variance=np.asarray([((np.square(values).sum()+1.)/(len(values)+2.)-mean**2)/(len(values)+3.)
                    for values,mean in zip(samples,absolute)])
                uncertainty[programs]=variance
            # The factual-outcome stratum supplies the zero-effect boundary.
            # Rescue and harm have different conditional distributions; mixing
            # them before credit assignment can cancel the useful intervention.
            null=0. if paired and not anchored else float(mode) if stratified else factual
            matrix=1.25*np.eye(len(programs));target=absolute+.25*null
            if not args.joint_no_cf and not absolute_regression and not args.joint_block_likelihood:
                for (a,b),samples in contrasts.items():
                    delta=float(np.mean(samples));weight=args.difference_weight*(1.+4.*abs(delta))
                    if args.joint_contrast_precision:
                        weight=args.difference_weight*len(samples)/(1.+4.*float(np.var(samples)))
                    if args.joint_credit_confidence:
                        delta=float(np.sum(samples)/(len(samples)+2.));weight=args.difference_weight
                    i,j=index[a],index[b]
                    matrix[i,i]+=weight;matrix[j,j]+=weight;matrix[i,j]-=weight;matrix[j,i]-=weight
                    target[i]-=weight*delta;target[j]+=weight*delta
            if args.joint_factor_value or args.joint_program_kernel or args.joint_unary_value:
                from conditional_fusion_value import fit_factor_program_values,fit_kernel_program_values,fit_unary_program_values
                estimator=(fit_unary_program_values if args.joint_unary_value else
                    fit_kernel_program_values if args.joint_program_kernel else fit_factor_program_values)
                potential,component_support=estimator(observations,contrasts,null,n,radix,
                    0. if args.joint_no_cf or absolute_regression else args.difference_weight,
                    args.joint_factor_ridge)
                potential=potential.clip(-1. if paired else 0.,1.)
            elif anchored:
                # Retain stable absolute outcomes while using paired effects
                # to relate each full program to one shared factual value.
                # Directly regressing Y(q)-Y(factual) can inject factual noise
                # into a program whose absolute outcome is nearly constant.
                joint=np.zeros((len(programs)+1,len(programs)+1));joint[:-1,:-1]=matrix
                factual_values=list(factual_samples[key].values())
                rhs=np.r_[target,np.sum(factual_values)+.25*factual]
                joint[-1,-1]=len(factual_values)+.25
                for code,i in index.items():
                    additional=max(0,len(observations[code])-1)
                    joint[i,i]+=additional;rhs[i]+=additional*absolute[i]
                if not absolute_regression and not args.joint_no_cf:
                    for code,i in index.items():
                        selected=[sample_id for sample_id,sample in opportunity_samples[key].items() if code in sample]
                        factual_values=np.asarray([factual_samples[key][sample_id] for sample_id in selected])
                        gains=np.asarray([opportunity_samples[key][sample_id][code] for sample_id in selected])
                        outcomes=gains+factual_values
                        covariance=np.mean((outcomes-outcomes.mean())*(factual_values-factual_values.mean()))
                        coefficient=float(np.clip(covariance/(factual_values.var()+.05/(len(selected)+1.)),0.,1.))
                        delta=float(gains.mean())
                        weight=args.difference_weight*coefficient*(1.+4.*abs(delta))
                        if args.joint_contrast_precision:
                            weight=args.difference_weight*len(gains)*coefficient/(1.+4.*float(gains.var()))
                        joint[i,i]+=weight;joint[-1,-1]+=weight;joint[i,-1]-=weight;joint[-1,i]-=weight
                        rhs[i]+=weight*delta;rhs[-1]-=weight*delta
                estimated=np.linalg.solve(joint,rhs).clip(0.,1.)
                potential=estimated[:-1]-estimated[-1]
            elif args.joint_conditional_calibration:
                from conditional_fusion_value import fit_calibrated_program_values
                potential=fit_calibrated_program_values(paired_observations,paired_contrasts,
                    factual_samples[key],factual,args.difference_weight)
            elif args.joint_paired_nuisance:
                from conditional_fusion_value import fit_paired_nuisance_values
                potential=fit_paired_nuisance_values(paired_observations,paired_contrasts,factual,
                    args.difference_weight,args.joint_nuisance_ridge)
            elif args.joint_binomial_value:
                from conditional_fusion_value import fit_binomial_program_values
                potential=fit_binomial_program_values(observations,contrasts,factual,
                    0. if absolute_regression or args.joint_no_cf else args.difference_weight)
            elif args.joint_bounded_value:
                from conditional_fusion_value import fit_bounded_graph_values
                potential=fit_bounded_graph_values(matrix,target,-1. if paired else 0.,1.)
            else:potential=np.linalg.solve(matrix,target).clip(-1. if paired else 0.,1.)
            if args.joint_factor_value or args.joint_program_kernel or args.joint_unary_value:
                if stratified:full[:,mode]=potential
                else:full[:]=potential
                known|=component_support
            elif stratified:full[programs,mode]=potential
            else:full[programs]=potential
            if args.joint_context_regularization:
                context_systems.append(dict(key=key,context=context,mode=mode,programs=programs,
                    matrix=matrix,target=target,values=full))
            known[programs]=True
        # The all-feedback program can have a nonzero effect when it waits for
        # a fixed window or releases a predecessor's committed intervention.
        # Its value and uncertainty therefore follow its measured outcomes.
        measured_support[key]=known.copy()
        if args.joint_paired_support:
            from scipy.special import betainc
            by_program={}
            for sample in opportunity_samples.get(key,{}).values():
                for code,gain in sample.items():by_program.setdefault(code,[]).append(gain)
            eligible=np.zeros_like(known)
            for code,gains in by_program.items():
                gains=np.asarray(gains);rescues=int((gains>0.).sum());harms=int((gains<0.).sum())
                # Among discordant paired outcomes, a positive expected effect
                # means rescue is more likely than harm. Neutral outcomes do
                # not establish its sign. Count each sampled future once.
                positive=1.-betainc(rescues+.5,harms+.5,.5)
                eligible[code]=len(gains)>=2 and positive>=args.joint_support_probability
            known&=eligible
        elif args.joint_factual_support:
            # Fresh futures estimate robustness and harm, but cannot by
            # themselves authorize an action that never repaired an observed
            # factual failure. Keep all edges in the value fit; restrict only
            # which whole programs may be deployed.
            if args.joint_factor_value or args.joint_program_kernel:
                from conditional_fusion_value import factor_program_support
                # An inferred plan must retain vessel choices found in factual
                # repairs, in addition to the pairwise sampling support above.
                eligible=factor_program_support(list(factual_support.get(key,set())),n,radix,pairwise=False)
            else:
                eligible=np.zeros_like(known)
                eligible[list(factual_support.get(key,set()))]=True
            known&=eligible
        if stratified:
            conditional=(1.-factual)*full[known,0]+factual*full[known,1]
            peaks[key]=max(0.,float(conditional.max())-factual) if known.any() else 0.
        else:
            selectable=full[known]-args.joint_credit_confidence*np.sqrt(np.maximum(uncertainty[known],0.))
            peaks[key]=max(0.,float(selectable.max())-(factual if not paired and not anchored else 0.)) if known.any() else 0.
        if (span,reversal) not in fitted:fitted[span,reversal]=([],[],[],[],[])
        centers,values,support,keys,variances=fitted[span,reversal]
        centers.append(context);values.append(full);support.append(known);keys.append(key)
        variances.append(uncertainty)
    if not fitted:raise ValueError('No joint intervention contexts match the selected spans.')
    if args.joint_context_metric:
        from conditional_fusion_value import fit_intervention_context_metric
        for configuration in fitted:
            selected=[example for example in shared_examples if example['key'][:2]==configuration]
            metric,training=fit_intervention_context_metric(selected,
                conditional=not args.joint_no_cf and not absolute_regression,
                difference_weight=args.difference_weight,bandwidth=args.joint_metric_bandwidth,
                updates=args.joint_metric_updates,seed=args.seed)
            context_metrics[configuration]=metric;metric_training[str(configuration)]=training
            print(dict(phase='intervention_context_metric',configuration=configuration,**training),flush=True)
    if args.joint_shared_potentials:
        from conditional_fusion_value import fit_shared_program_potential,fit_block_program_potential,fit_response_precision_potential
        for configuration,(centers,values,support,keys,_) in fitted.items():
            selected=[example for example in shared_examples if example['key'][:2]==configuration]
            if args.joint_block_likelihood:
                fit_conditional=(fit_response_precision_potential if args.joint_response_precision else fit_block_program_potential)
                extra={} if args.joint_response_precision else dict(ordering_weight=args.difference_weight)
                network,state,training=fit_conditional(selected,width=args.joint_program_width,
                    updates=args.joint_program_updates,seed=args.seed,**extra,
                    node_width=15 if args.joint_credit_context>=3 else 8,
                    global_width=5 if args.joint_credit_context>=5 else 4 if args.joint_credit_context>=3 else 3,
                    program_options=None if program_options is None else program_options.tolist(),
                    device='cuda' if torch.cuda.is_available() else 'cpu')
            else:
                network,state,training=fit_shared_program_potential(selected,width=args.joint_program_width,
                    updates=args.joint_program_updates,paired=paired,
                    conditional=not args.joint_no_cf and not absolute_regression,
                    difference_weight=args.difference_weight,seed=args.seed,
                    node_width=15 if args.joint_credit_context>=3 else 8,
                    global_width=5 if args.joint_credit_context>=5 else 4 if args.joint_credit_context>=3 else 3,
                    program_options=None if program_options is None else program_options.tolist(),
                    paired_likelihood=args.joint_paired_likelihood)
            shared_models[configuration]=state
            shared_training[str(configuration)]=training
            by_key={example['key']:example for example in selected}
            offsets=[]
            with torch.no_grad():
                for context,value,known,key in zip(centers,values,support,keys):
                    measured_codes=torch.as_tensor(by_key[key]['programs'],dtype=torch.long)
                    if args.joint_program_completion:
                        from conditional_fusion_value import factor_program_support
                        # Infer a complete combination only when every vessel
                        # choice and every pair of choices has been observed.
                        # Eligibility uses sampling support, not outcome labels;
                        # the learned value must decide whether it is useful.
                        known[:]=factor_program_support(measured_codes.numpy(),n,radix)
                    codes=(torch.as_tensor(np.flatnonzero(known),dtype=torch.long)
                        if args.joint_program_completion else measured_codes)
                    if not len(codes):peaks[key]=0.;continue
                    inputs=torch.as_tensor(context[None],dtype=torch.float32)
                    prediction=torch.cat([network(inputs.expand(len(part),-1),part) for part in codes.split(256)]).clamp(-1. if paired else 0.,1.)
                    if args.joint_program_calibration:
                        measured_prediction=torch.cat([network(inputs.expand(len(part),-1),part)
                            for part in measured_codes.split(256)]).clamp(-1. if paired else 0.,1.)
                        offset=float(np.mean(by_key[key]['targets']-measured_prediction.numpy()))
                    else:offset=0.
                    offsets.append(offset)
                    prediction=(prediction+offset).clamp(-1. if paired else 0.,1.)
                    value[codes.numpy()]=prediction.numpy()
                    factual=float(reference(inputs)[0][0]) if not paired else 0.
                    peaks[key]=max(0.,float(value[known].max())-factual) if known.any() else 0.
            if args.joint_program_calibration:program_offsets[configuration]=np.asarray(offsets,dtype=np.float32)
    if context_systems:
        # Estimate neighboring conditional values jointly. Fitting every root
        # independently can make a same-future difference nearly redundant
        # with its two outcome labels. Shared context estimates let the paired
        # contrasts constrain which decision effects persist across roots.
        from scipy.sparse import block_diag,coo_matrix
        from scipy.sparse.linalg import spsolve
        if args.joint_credit_confidence:
            raise ValueError('Joint context estimation requires its own posterior covariance.')
        offsets=np.cumsum([0]+[len(system['programs']) for system in context_systems])
        matrix=block_diag([system['matrix'] for system in context_systems],format='csr')
        target=np.concatenate([system['target'] for system in context_systems])
        centers=np.stack([system['context'] for system in context_systems])
        distance=np.square(centers[:,None]-centers[None]).sum(-1)
        affinity=np.exp(-distance/(2.*args.joint_context_bandwidth**2))
        compatible=np.asarray([[a['key'][:2]==b['key'][:2] and a['mode']==b['mode']
            for b in context_systems] for a in context_systems])
        affinity*=compatible;np.fill_diagonal(affinity,0.)
        affinity[affinity<.01]=0.
        degree=affinity.sum(-1).clip(1.,None)
        rows=[];columns=[];coefficients=[];links=0
        for i,a in enumerate(context_systems):
            for j in np.flatnonzero(affinity[i,i+1:])+i+1:
                b=context_systems[j]
                _,left,right=np.intersect1d(a['programs'],b['programs'],return_indices=True)
                if not len(left):continue
                weight=args.joint_context_regularization*affinity[i,j]/max(degree[i],degree[j])
                left=left+offsets[i];right=right+offsets[j]
                rows.extend(np.concatenate((left,right,left,right)))
                columns.extend(np.concatenate((left,right,right,left)))
                coefficients.extend(np.concatenate((np.full(2*len(left),weight),np.full(2*len(left),-weight))))
                links+=len(left)
        if links:
            matrix=matrix+coo_matrix((coefficients,(rows,columns)),shape=matrix.shape).tocsr()
        potential=spsolve(matrix,target).clip(-1. if paired else 0.,1.)
        for i,system in enumerate(context_systems):
            if stratified:system['values'][system['programs'],system['mode']]=potential[offsets[i]:offsets[i+1]]
            else:system['values'][system['programs']]=potential[offsets[i]:offsets[i+1]]
        for centers,values,support,keys,_ in fitted.values():
            for context,value,known,key in zip(centers,values,support,keys):
                factual=float(reference(torch.tensor(context[None]))[0][0]) if absolute_regression else 0.
                peaks[key]=max(0.,float(value[known].max())-factual) if known.any() else 0.
        print(dict(phase='joint_context_value_fit',variables=len(target),context_links=links,
            regularization=args.joint_context_regularization,bandwidth=args.joint_context_bandwidth),flush=True)
    if args.joint_wait_crossfit:
        # Waiting forfeits the current intervention. Estimate a selectable
        # future gain by choosing on other continuations and scoring on the
        # held continuation, instead of treating the best fitted outcome as
        # an opportunity that will certainly remain available.
        for key,samples in opportunity_samples.items():
            estimates=[];futures=list(samples.values())
            for held_index,held in enumerate(futures):
                training={}
                for index,observations in enumerate(futures):
                    if index==held_index:continue
                    for code,value in observations.items():
                        if code in held:training.setdefault(code,[]).append(value)
                if not training:continue
                eligible={code:np.mean(values) for code,values in training.items()
                    if not args.joint_factual_support or code in factual_support.get(key,set())}
                if not eligible:estimates.append(0.);continue
                selected=max(eligible,key=lambda code:(eligible[code],-code))
                estimates.append(held[selected] if eligible[selected]>0. else 0.)
            peaks[key]=min(peaks[key],max(0.,float(np.mean(estimates)))) if estimates else 0.
    waiting={}
    if args.joint_temporal_credit:
        if not args.joint_paired_advantage or args.joint_credit_context<2:
            raise ValueError('Temporal credit requires paired gains and a time-aware public context.')
        # An early intervention must beat the opportunity to wait for a better
        # later intervention. Propagate that opportunity along factual paths;
        # execution queries only public state, never a future time or seed.
        for timeline in timelines.values():
            for key,t in timeline.items():
                future=[peaks[later]*np.exp(-.02*(time-t)) for later,time in timeline.items() if time>t+1e-6]
                waiting.setdefault(key,[]).append(max(future,default=0.))
    records=[dict(contexts=np.asarray(centers),values=np.asarray(values),observed=np.asarray(support),
        span=span,anchor=anchor,bandwidth=.15,margin=args.joint_credit_margin,conditional_strata=stratified,transport=args.joint_role_transport,
        reversal=reversal,
        effect_locality=args.joint_credit_locality,
        gain_variance=np.asarray(variances),confidence=args.joint_credit_confidence,
        support_uncertainty=args.joint_support_uncertainty,
        decision_tolerance=args.joint_decision_tolerance,
        blending_penalty=args.joint_blending_penalty,
        fixed_intervention_clock=clock_modes[span,reversal],
        initial_only=args.joint_initial_root,
        paired_advantage=paired or anchored,context_version=args.joint_credit_context,
        learning_objective='absolute' if args.joint_no_cf or absolute_regression else 'conditional',
        program_options=None if program_options is None else program_options.tolist(),
        prefix_refinement=args.joint_prefix_refinement,
        context_precision=context_metrics.get((span,reversal)),
        program_model=shared_models.get((span,reversal)),
        program_offsets=program_offsets.get((span,reversal)),
        measured=np.asarray([measured_support[key] for key in keys]) if args.joint_coma_credit else None,
        counterfactual_credit=args.joint_coma_credit,
        wait_values=np.asarray([np.mean(waiting.get(key,[0.])) for key in keys],dtype=np.float32))
        for (span,reversal),(centers,values,support,keys,variances) in sorted(fitted.items())]
    inactive_windows=[record['span'] for record in records if not np.asarray(record['observed']).any()]
    records=[record for record in records if np.asarray(record['observed']).any()]
    if args.joint_residual_response and not args.joint_midpoint_refinement:
        predecessor=model.get('joint_intervention_value') or model.get('joint_intervention_values')
        if predecessor is None:
            raise ValueError('A conditional policy update requires its preceding joint value.')
        if len(records)>1 and not (paired or stratified or ((absolute_regression or args.joint_binomial_value or args.joint_success_potentials or args.joint_paired_nuisance or args.joint_conditional_calibration) and args.joint_paired_advantage)):
            raise ValueError('Competing durations require paired gains against the same preceding policy.')
        # This is a policy-iteration update of the SAME conditional mechanism.
        # Its labels follow the preceding policy, and its factual value is
        # anchored to that policy's new rollouts, not the original baseline.
        # Every duration is evaluated against the same continuation policy.
        # Their paired gains can therefore compete without favoring a duration
        # merely because it was collected in easier factual states.
        for record in records:record['prior']=predecessor
    if args.joint_midpoint_refinement:
        model['midpoint_intervention_values']=records
        model['option_refinement_fraction']=args.joint_refinement_fraction
        model['option_refinement_steps']=args.joint_refinement_steps
        model['option_refinement_checkpoints']=args.joint_refinement_checkpoints
        model['phase_preserving_refinement']=args.joint_phase_preserving_refinement
    elif records or not args.joint_residual_response:
        model.pop('joint_intervention_value',None)
        model.pop('joint_intervention_values',None)
        if len(records)==1:model['joint_intervention_value']=records[0]
        elif records:model['joint_intervention_values']=records
    if records or not args.joint_residual_response:
        model['fixed_intervention_clock']=args.joint_source_reference or bool(datasets[0]['model'].get('fixed_intervention_clock',False))
    context_count=sum(len(record['contexts']) for record in records)
    objectives=joint_value_objectives(model)
    ordinary=args.joint_no_cf or absolute_regression
    # Absolute fitting of one appended head is a local objective comparison
    # whenever any conditional (or unclassified legacy) ancestor remains.
    # Record the scope from the retained algorithm, never from the last loss.
    whole_ordinary=ordinary and objectives=={'absolute'}
    model['whole_mechanism_replacement']=whole_ordinary
    model['comparison_scope']=('whole conditional-value replacement by absolute-return learning'
        if whole_ordinary else 'local learning-objective comparison; predecessor learning retained'
        if ordinary else 'conditional-value learning')
    model['joint_intervention_fit']=dict(conditional=not args.joint_no_cf and not absolute_regression,roots=root_count,
        value_target='joint_outcome_pattern_likelihood' if args.joint_block_likelihood else 'cross_fitted_factual_calibration' if args.joint_conditional_calibration else 'paired_nuisance_program_success' if args.joint_paired_nuisance else 'binomial_success_with_conditional_odds' if args.joint_binomial_value else 'anchored_success_gain' if anchored else 'absolute_terminal_success' if absolute_regression else 'paired_gain' if paired else 'terminal_success_with_local_contrasts',
        independent_futures=args.joint_independent_futures,
        contexts=context_count,edges=edge_count,difference_weight=0. if args.joint_no_cf or absolute_regression or args.joint_block_likelihood else args.difference_weight,
        inactive_windows=inactive_windows,applied_update=bool(records),
        supported_programs=sum(int(np.asarray(record['observed']).sum()) for record in records),
        retained_learning_objectives=sorted(objectives),comparison_scope=model['comparison_scope'],
        same_interaction_predecessor=all(interaction_matches),shared_data_comparison=args.shared_data,
        same_continuation_predecessor=all(continuation_matches),
        data_comparison_scope='matched_predecessor' if all(continuation_matches) else 'fixed_data_learning_comparison',
        mismatched_decision_clock_edges_excluded=clock_filtered,
        context_regularization=args.joint_context_regularization,context_bandwidth=args.joint_context_bandwidth,
        unique_futures=args.joint_unique_futures,wait_crossfit=args.joint_wait_crossfit,
        factual_calibration=args.joint_factual_calibration,
        paired_support=args.joint_paired_support,support_probability=args.joint_support_probability,
        source_reference=args.joint_source_reference,
        anchored_value=anchored,
        adaptive_reference=anchored and not absolute_regression,
        contrast_precision=args.joint_contrast_precision,
        paired_nuisance=args.joint_paired_nuisance,nuisance_ridge=args.joint_nuisance_ridge,
        conditional_calibration=args.joint_conditional_calibration,bounded_value=args.joint_bounded_value,
        counterfactual_credit=args.joint_coma_credit,
        temporal_programs=args.joint_temporal_programs,
        temporal_blending_levels=args.joint_temporal_blending_levels,
        prefix_refinement=args.joint_prefix_refinement,
        shared_program_fit=shared_training,
        paired_likelihood=args.joint_paired_likelihood,
        block_likelihood=args.joint_block_likelihood,
        response_precision=args.joint_response_precision,
        program_completion=args.joint_program_completion,
        context_metric_fit=metric_training,
        factor_value=args.joint_factor_value,unary_value=args.joint_unary_value,factor_ridge=args.joint_factor_ridge,
        program_kernel=args.joint_program_kernel,
        midpoint_refinement=args.joint_midpoint_refinement,
        remainder_refinement=args.joint_remainder_refinement,
        option_refinement_fraction=args.joint_refinement_fraction,
        option_refinement_steps=args.joint_refinement_steps,
        option_refinement_checkpoints=args.joint_refinement_checkpoints,
        phase_preserving_refinement=args.joint_phase_preserving_refinement,
        training_simulation_steps=sum(data.get('simulation_steps',0) for data in datasets),
        training_data=[str(args.data),*[str(p) for p in (args.references or [])]])
    args.output.mkdir(parents=True,exist_ok=True)
    print(dict(phase='joint_graph_fit',conditional=not args.joint_no_cf and not absolute_regression,roots=root_count,
        contexts=context_count,edges=edge_count),flush=True)
    if args.critic_only:
        model.pop('selection',None);torch.save(model,args.output/'policy.pth');return
    history=[];best=(-1.,-1.)
    scenes=bank(args.selection_bank,args.selection_per_cell,tuple((n,a) for n in args.teams for a in args.agilities))
    for bandwidth in (args.joint_local_bandwidths if records else args.joint_local_bandwidths[:1]):
        for record in records:record['bandwidth']=bandwidth
        for margin in (args.joint_response_margins if records else args.joint_response_margins[:1]):
            for record in records:record['margin']=margin
            path=args.output/'current.pth';torch.save(model,path)
            tasks=[(s,str(path),path.stat().st_mtime_ns,False,0,0,0,0,
                None if repeat==0 else int(np.random.SeedSequence([s[0],repeat,20261014]).generate_state(1)[0]))
                for repeat in range(args.joint_future_repetitions) for s in scenes]
            rows=list(pool.map(joint_episode,tasks,chunksize=1));summary=summarize(rows)
            trial=dict(bandwidth=bandwidth,margin=margin,development=summary,rows=rows);history.append(trial)
            score=(summary['success'],summary['capture'])
            if score>best:best=score;model['selection']=trial;torch.save(model,args.output/'policy.pth')
            save_json(args.output/'development.json',dict(history=history,best_success=best[0],
                contexts=context_count,edges=edge_count))
            print(dict(phase='joint_graph_development',bandwidth=bandwidth,margin=margin,
                success=score[0],collision=summary['collision']),flush=True)


def joint_local_value(args,pool):
    """Fit encounter-local conditional values, then select their trust radius."""
    from conditional_fusion_value import LocalConditionalValue, local_fusion_context, response_fusion_context
    if args.collect_only:
        model=torch.load(args.start,map_location='cpu',weights_only=False)
        if args.joint_no_interaction:require_clean_mechanism_start(model,args)
        model.update(failure_roots=args.joint_failure_roots,
            success_roots=args.joint_success_roots,
            initial_root=args.joint_initial_root,
            failure_lookback=args.joint_failure_lookback,failure_subjects=args.joint_failure_subjects,
            failure_duration=args.joint_failure_duration,
            credit_lookbacks=args.joint_credit_lookbacks or (),
            paired_advantage=args.joint_paired_advantage,
            fixed_intervention_clock=args.joint_fixed_intervention_clock or model.get('fixed_intervention_clock',False),
            gate_period=args.joint_gate_period,branch_grid=True,fitted_iteration=True,
            branch_value_data=False,success_objective=True,observed_future=args.joint_observed_future,
            joint_contexts=args.joint_contexts,context_prediction=args.joint_context_prediction)
        model['exhaustive_contexts']=args.joint_exhaustive_contexts
        model['temporal_programs']=args.joint_temporal_programs
        model['temporal_blending_levels']=args.joint_temporal_blending_levels
        model['independent_root_futures']=args.joint_independent_root_futures
        if args.joint_independent_root_futures and not args.joint_batch_collection:
            raise ValueError('Independent root futures use the paired batched collector.')
        model['prefix_refinement']=args.joint_prefix_refinement
        model['midpoint_refinement']=args.joint_midpoint_refinement
        model['remainder_refinement']=args.joint_remainder_refinement
        model['option_refinement_fraction']=args.joint_refinement_fraction
        model['option_refinement_steps']=args.joint_refinement_steps
        model['option_refinement_checkpoints']=args.joint_refinement_checkpoints
        model['phase_preserving_refinement']=args.joint_phase_preserving_refinement
        if args.joint_midpoint_refinement and (not args.joint_prefix_refinement or not args.joint_batch_collection
                or model.get('midpoint_intervention_values')):
            raise ValueError('Midpoint collection refines a fixed predecessor without an existing midpoint update.')
        if args.joint_program_proposer:
            proposer=torch.load(args.joint_program_proposer,map_location='cpu',weights_only=False)
            if any(proposer.get(key)!=model.get(key) for key in ('source_sha256','mechanism_ablation','coupled_shortlist','risk_prior','fusion_selection')):
                raise ValueError('Intervention proposals must come from the same mechanism configuration and source.')
            values=proposer.get('joint_intervention_values') or [proposer['joint_intervention_value']]
            model['program_proposer']=dict(values=values,source=str(args.joint_program_proposer),
                sha256=hashlib.sha256(args.joint_program_proposer.read_bytes()).hexdigest())
        else:model.pop('program_proposer',None)
        model['max_held']=args.joint_max_held
        model['edge_futures']=args.joint_edge_futures
        model['balanced_futures']=args.joint_balanced_futures
        model['context_interceptors']=args.joint_context_interceptors
        model['future_start']=args.joint_future_start
        if args.joint_preference_strength is not None:model['preference_strength']=args.joint_preference_strength
        if args.joint_conditional_commitment or args.joint_immediate_commitment:model['conditional_commitment']=True
        if args.joint_immediate_commitment:model['immediate_commitment']=True
        if args.joint_physical_interventions:
            model.update(physical_interventions=True,physical_intervention_span=args.joint_physical_span,
                physical_intervention_reversal=args.joint_physical_reversal,
                physical_cost=args.joint_physical_cost)
            model['physical_singletons']=args.joint_physical_singletons
            if model.get('physical_conditional_value') is None:
                model['physical_span']=args.joint_physical_span
                model['physical_reversal']=args.joint_physical_reversal
        if args.without_peer:model['peer']=False
        args.output.mkdir(parents=True,exist_ok=True)
        path=args.output/'current.pth';torch.save(model,path)
        tasks=[(s,str(path),path.stat().st_mtime_ns,True,args.roots_per_episode,
                args.repetitions,args.branch_horizon,0,
                None if repeat==0 else int(np.random.SeedSequence([s[0],repeat,20261014]).generate_state(1)[0]))
                for repeat in range(args.joint_future_repetitions) for s in bank(args.bank,args.per_cell)]
        if args.joint_balanced_futures and not (args.joint_batch_collection and (args.joint_initial_root or args.joint_midpoint_refinement)):
            raise ValueError('Outcome-balanced continuation sampling uses shared initial contexts in the batched sampler.')
        if args.joint_batch_collection:
            if not (args.joint_physical_interventions and args.joint_failure_roots
                    and args.joint_failure_subjects and args.branch_horizon>=400):
                raise ValueError('Batched program learning requires terminal physical interventions at factual failures.')
            if args.joint_future_start or args.joint_physical_singletons:
                raise ValueError('This sampling recipe uses the complete local-pair and joint-context library.')
            import run_remote_evidence as runner
            runner.SOURCE=args.source
            runner.collect_joint_credit_batched(path,args.output,[(task[0],task[8]) for task in tasks],
                batch_size=args.joint_batch_collection,repetitions=args.repetitions)
            return
        rollouts=[]
        for result in pool.map(joint_episode,tasks,chunksize=1):
            rollouts.append(result)
            if len(rollouts)%8==0 or len(rollouts)==len(tasks):
                print(dict(phase='local_conditional_collection',episodes=len(rollouts),
                    success=sum(r['row']['success'] for r in rollouts),
                    interventions=sum(len(r['branches']) for r in rollouts),
                    simulation_steps=sum(r['simulation_steps'] for r in rollouts)),flush=True)
        torch.save(dict(model=model,rollouts=rollouts,source_sha256=model.get('source_sha256',BASELINE_SHA256),
            simulation_steps=sum(r['simulation_steps'] for r in rollouts),
            use='development intervention fitting; these scenes are not an independent evaluation'),
            args.output/'training-data.pth')
        save_json(args.output/'development.json',dict(collection=summarize([r['row'] for r in rollouts]),
            rows=[r['row'] for r in rollouts],bank=args.bank,
            use='development intervention fitting',simulation_steps=sum(r['simulation_steps'] for r in rollouts)))
        return
    if args.joint_local_joint:return joint_graph_value(args,pool)
    datasets=[torch.load(path,map_location='cpu',weights_only=False)
              for path in [args.data,*(args.references or [])]]
    model=copy.deepcopy(datasets[0]['model'])
    if args.start:model=torch.load(args.start,map_location='cpu',weights_only=False)
    if args.joint_no_interaction:require_clean_mechanism_start(model,args)
    physical=args.joint_physical_interventions
    value_key='physical_conditional_value' if physical else 'local_conditional_value'
    residual_prior=copy.deepcopy(model.get(value_key))
    intervention_model=datasets[0]['model']
    physical_span=intervention_model.get('physical_intervention_span',intervention_model.get('physical_span',10))
    physical_reversal=intervention_model.get('physical_intervention_reversal',False)
    if physical:
        model.update(physical_intervention_span=physical_span,physical_intervention_reversal=physical_reversal)
    additional_span=physical and (physical_span!=model.get('physical_span',10)
        or physical_reversal!=model.get('physical_reversal',False))
    if additional_span:
        residual_prior=next((copy.deepcopy(record['value']) for record in model.get('physical_conditional_values',[])
            if record['span']==physical_span and record.get('reversal',False)==physical_reversal),None)
    encoder='response' if args.joint_local_response else 'state'
    if residual_prior is not None and residual_prior.get('encoder','state')!=encoder:
        residual_prior=None
    context_actor=joint_load(str(args.start),args.start.stat().st_mtime_ns) if args.joint_local_response else None
    def encode(packet,proposal,left,right):
        if context_actor is None:return local_fusion_context(packet,proposal,left,right,model['peer'])[0].numpy()
        if '_local_value_risk' not in packet:
            packet['_local_value_risk']=context_actor.candidate_risk(packet,proposal)
        return response_fusion_context(packet,proposal,packet['_local_value_risk'],left,right,model['peer'])[0].numpy()
    if model['levels']!=3 or not model.get('intent_strength'):
        raise ValueError('Local conditional values use the three-level fusion preference graph.')
    centers=[];values=[];support=[];root_count=0;informative=0
    for data in datasets:
        for rollout in data['rollouts']:
            groups={}
            for branch in rollout['branches']:
                groups.setdefault(tuple(branch['root'][:2]),[]).append(branch)
            for branches in groups.values():
                column=3 if physical else 2
                gates=np.stack([b[side][:,column] for b in branches for side in ('left','right')])
                subjects=np.flatnonzero(np.ptp(gates,axis=0)>1e-6)
                if physical and len(subjects)==2:
                    i,j=map(int,subjects);design=[];targets=[]
                    for branch in branches:
                        if branch['before']['success'] is None or branch['after']['success'] is None:continue
                        def physical_index(action):
                            values=action[subjects,3]
                            return np.where(values<0.,0,np.rint(values*2).astype(int)+1)
                        left=physical_index(branch['left']);right=physical_index(branch['right'])
                        row=np.zeros(16);row[left[0]*4+left[1]]=1.;row[right[0]*4+right[1]]-=1.
                        design.append(row);targets.append(branch['before']['success']-branch['after']['success'])
                    if not design:continue
                    design=np.asarray(design);targets=np.asarray(targets)
                    reference=physical_index(branches[0]['reference_action'])
                    reference_index=reference[0]*4+reference[1]
                    anchor=np.zeros((1,16));anchor[0,reference_index]=1.
                    matrix=np.concatenate((design,anchor));target=np.append(targets,0.)
                    value=np.linalg.lstsq(matrix,target,rcond=None)[0].reshape(4,4)
                    value-=value[reference[0],reference[1]]
                    packet=mechanism_tensor(branches[0]['packet'])
                    proposal=torch.tensor(branches[0]['left'][None,:,:2])
                    for a,b,table in ((i,j,value),(j,i,value.T)):
                        centers.append(encode(packet,proposal,a,b))
                        values.append(table.copy());support.append(1.)
                    root_count+=1;informative+=int(np.ptp(value)>.1)
                    continue
                if len(subjects)>2:
                    # In a coordinated context a two-vessel neighborhood may
                    # have no successful action. Preserve each single-vessel
                    # edge, estimate the observed joint values, and project
                    # their improvement onto the same local fusion factors.
                    observed={};reference=branches[0]['reference_action'][:,column]
                    reference_returns=[]
                    for branch in branches:
                        reference_returns.append(branch['reference_outcome']['success'])
                        for action_key,outcome_key in (('left','before'),('right','after')):
                            program=tuple(branch[action_key][:,column])
                            observed.setdefault(program,[]).append(branch[outcome_key]['success'])
                    programs=list(observed);index={program:k for k,program in enumerate(programs)}
                    contrasts=[];differences=[]
                    for branch in branches:
                        contrast=np.zeros(len(programs))
                        contrast[index[tuple(branch['left'][:,column])]]=1.
                        contrast[index[tuple(branch['right'][:,column])]]-=1.
                        contrasts.append(contrast)
                        differences.append(branch['before']['success']-branch['after']['success'])
                    contrasts=np.asarray(contrasts)
                    absolute=np.asarray([np.mean(observed[program]) for program in programs])
                    # Absolute anchors connect distinct intervention contexts;
                    # each contrast still changes exactly one vessel's gate.
                    joint_values=np.linalg.solve(contrasts.T@contrasts+np.eye(len(programs)),
                        contrasts.T@np.asarray(differences)+absolute)
                    means=dict(zip(programs,map(float,joint_values)))
                    selected=max(means,key=lambda program:(means[program],
                        -float(np.square(np.asarray(program)-reference).sum())))
                    gain=max(0.,means[selected]-float(np.mean(reference_returns)))
                    target=np.asarray(selected);n=len(target)
                    grid=np.asarray([-1.,0.,.5,1.]) if physical else np.linspace(0.,1.,3)
                    packet=mechanism_tensor(branches[0]['packet'])
                    proposal=torch.tensor(branches[0]['left'][None,:,:2])
                    for i in range(n):
                        for j in range(i+1,n):
                            if physical:
                                value=gain*(float(reference[i]!=target[i])+float(reference[j]!=target[j])
                                    -(grid[:,None]!=target[i]).astype(float)
                                    -(grid[None,:]!=target[j]).astype(float))/max(1,n-1)
                            else:
                                value=gain*((reference[i]-target[i])**2+(reference[j]-target[j])**2
                                    -(grid[:,None]-target[i])**2-(grid[None,:]-target[j])**2)/max(1,n-1)
                            for a,b,table in ((i,j,value),(j,i,value.T)):
                                centers.append(encode(packet,proposal,a,b))
                                values.append(table.copy());support.append(1.)
                    root_count+=1;informative+=int(gain>.01)
                    continue
                if len(subjects)!=2:continue
                i,j=map(int,subjects);design=[];targets=[]
                for branch in branches:
                    if branch['before']['success'] is None or branch['after']['success'] is None:continue
                    left=np.rint(branch['left'][subjects,2]*2).astype(int)
                    right=np.rint(branch['right'][subjects,2]*2).astype(int)
                    row=np.zeros(9);row[left[0]*3+left[1]]=1.;row[right[0]*3+right[1]]-=1.
                    design.append(row);targets.append(branch['before']['success']-branch['after']['success'])
                if not design:continue
                design=np.asarray(design);targets=np.asarray(targets)
                # Q(s, neutral)=0 fixes the graph's unidentifiable state offset.
                anchor=np.zeros((1,9));anchor[0,4]=1.
                matrix=np.concatenate((design,anchor));target=np.append(targets,0.)
                value=np.linalg.lstsq(matrix,target,rcond=None)[0].reshape(3,3)
                value-=value[1,1]
                packet=mechanism_tensor(branches[0]['packet'])
                proposal=torch.tensor(branches[0]['left'][None,:,:2])
                for a,b,table in ((i,j,value),(j,i,value.T)):
                    centers.append(encode(packet,proposal,a,b))
                    values.append(table.copy());support.append(1.)
                root_count+=1;informative+=int(np.ptp(value)>.1)
    if not centers:raise ValueError('No two-vessel conditional intervention graphs were available.')
    model['conditional_value_fit']=dict(
        algorithm=('execution-matched physical fusion intervention contrasts' if physical else
            'anchored single-vessel return contrasts with local joint-value interpolation'),
        training_data=[str(p) for p in [args.data,*(args.references or [])]],
        development_scenes_used_for_fitting=any('development' in d.get('use','') for d in datasets),
        roots=root_count,informative_roots=informative)
    args.output.mkdir(parents=True,exist_ok=True)
    best=(-1.,-1.);history=[]
    previous_path=args.output/'development.json'
    if previous_path.exists():
        previous=json.loads(previous_path.read_text(encoding='utf-8'))
        if (previous.get('training_data')==model['conditional_value_fit']['training_data']
                and previous.get('roots')==root_count and previous.get('history')):
            history=previous['history']
            best=max((h['development']['success'],h['development']['capture']) for h in history)
    if not history and (physical or residual_prior is not None) and model.get('selection'):
        initial=dict(model['selection'],retained_previous_value=True)
        history.append(initial)
        best=(initial['development']['success'],initial['development']['capture'])
        torch.save(model,args.output/'policy.pth')
    cells=tuple((n,a) for n in (args.selection_teams or args.teams) for a in args.agilities)
    scenes=bank(args.selection_bank,args.selection_per_cell,cells)
    print(dict(phase='local_conditional_fit',roots=root_count,informative_roots=informative,
        scenes=len(scenes),period=model.get('gate_period',1)),flush=True)
    authorities=args.joint_local_authorities or (model['intent_strength'],)
    for bandwidth,strength,authority in ((h,s,a) for h in args.joint_local_bandwidths
            for s in args.joint_local_strengths for a in authorities):
        value=LocalConditionalValue(np.asarray(centers),np.asarray(values),support,bandwidth,strength,model['peer'],residual_prior,encoder)
        if additional_span:
            previous=[record for record in model.get('physical_conditional_values',[])
                if (record['span'],record.get('reversal',False))!=(physical_span,physical_reversal)]
            model['physical_conditional_values']=[*previous,dict(span=physical_span,reversal=physical_reversal,value=value.record())]
        else:model[value_key]=value.record()
        model['intent_strength']=authority
        path=args.output/'current.pth';torch.save(model,path)
        rows=list(pool.map(joint_episode,[(s,str(path),path.stat().st_mtime_ns,False,0,0,0,0)
            for s in scenes],chunksize=1))
        summary=summarize(rows);score=(summary['success'],summary['capture'])
        trial=dict(bandwidth=bandwidth,strength=strength,authority=authority,development=summary,rows=rows)
        history.append(trial)
        if score>best:
            best=score;model['selection']=trial;torch.save(model,args.output/'policy.pth')
        save_json(args.output/'development.json',dict(history=history,best_success=best[0],
            roots=root_count,informative_roots=informative,selection_bank=args.selection_bank,
            training_data=[str(p) for p in [args.data,*(args.references or [])]],
            development_scenes_used_for_fitting=model['conditional_value_fit']['development_scenes_used_for_fitting'],
            source_sha256=BASELINE_SHA256))
        print(dict(phase='local_conditional_development',bandwidth=bandwidth,strength=strength,authority=authority,
            success=score[0],collision=summary['collision'],best=best[0]),flush=True)


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
    actor.fusion_expression=args.joint_fusion_expression
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
        if old.get('fusion_expression','interaction')!=actor.fusion_expression and not common_initialization:
            raise ValueError('Fusion-expression alternatives require a common fresh initialization.')
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
            fusion_expression=actor.fusion_expression,
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
        scenes=bank(args.bank+(0 if args.joint_fixed_bank else iteration*100000),args.per_cell);revision=current.stat().st_mtime_ns
        tasks=[(s,str(current),revision,True,0 if args.joint_no_cf else args.roots_per_episode,
            args.repetitions,args.branch_horizon,0) for s in scenes]
        if args.joint_fixed_bank:
            repeat=iteration%4
            tasks=[(*task,None if repeat==0 else int(np.random.SeedSequence(
                [task[0][0],repeat,20261014]).generate_state(1)[0])) for task in tasks]
        rollouts=replay['rollouts'] if replay else []
        if not replay:
            if schedule:
                budget=next(h['iteration_simulation_steps'] for h in schedule if h['iteration']==iteration+1)
                used=0
                for begin in range(0,len(tasks),args.workers):
                    remaining=budget-used;batch=[]
                    for task in tasks[begin:begin+args.workers]:
                        cap=min(400,remaining)
                        if cap<=0:break
                        batch.append((*task[:7],cap,*task[8:]));remaining-=cap
                    if not batch:break
                    additional=list(pool.map(joint_episode,batch,chunksize=1))
                    rollouts.extend(additional);used+=sum(r['simulation_steps'] for r in additional)
                    if len(rollouts)%16==0:print(dict(phase='joint_collect',iteration=iteration,
                        episodes=len(rollouts),total=len(tasks)),flush=True)
            else:
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
                        if args.joint_fixed_bank:
                            scene=scenes[(extra+j)%len(scenes)]
                            repeat=(iteration+(extra+j)//len(scenes))%4
                            future=(None if repeat==0 else int(np.random.SeedSequence(
                                [scene[0],repeat,20261014]).generate_state(1)[0]))
                            batch.append((scene,str(current),revision,True,0,0,0,cap,future))
                        else:
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
    p.add_argument('mode',choices=('joint-target-motion','joint-supported-value','joint-shared-value','joint-local-value','joint-value-diagnostic','joint-interpolate','joint-risk-search','joint-train','joint-evaluate','stationary-reference','mechanism-train','mechanism-refit','mechanism-evaluate','search','proposal-search','trajectory-search','trajectory-fit','trajectory-observations','trajectory-encode','fit','refine','augment','evaluate'))
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
    p.add_argument('--joint-ablation-suite',action='store_true')
    p.add_argument('--joint-no-interaction',action='store_true')
    p.add_argument('--joint-graph',action='store_true')
    p.add_argument('--joint-levels',type=int,default=2)
    p.add_argument('--joint-fusion-expression',choices=('interaction','state','candidate'),default='interaction')
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
    p.add_argument('--joint-success-roots',action='store_true')
    p.add_argument('--joint-failure-lookback',type=float,default=0.)
    p.add_argument('--joint-failure-duration',type=float,default=0.)
    p.add_argument('--joint-credit-lookbacks',type=float,nargs='+')
    p.add_argument('--joint-paired-advantage',action='store_true')
    p.add_argument('--joint-independent-futures',action='store_true')
    p.add_argument('--joint-credit-context',type=int,choices=(1,2,3,4,5),default=1)
    p.add_argument('--joint-role-transport',action='store_true')
    p.add_argument('--joint-credit-locality',action='store_true')
    p.add_argument('--joint-credit-confidence',type=float,default=0.)
    p.add_argument('--joint-context-regularization',type=float,default=0.)
    p.add_argument('--joint-context-bandwidth',type=float,default=.15)
    p.add_argument('--joint-context-metric',action='store_true')
    p.add_argument('--joint-metric-bandwidth',type=float,default=.15)
    p.add_argument('--joint-metric-updates',type=int,default=400)
    p.add_argument('--joint-unique-futures',action='store_true')
    p.add_argument('--joint-factual-calibration',action='store_true')
    p.add_argument('--joint-paired-support',action='store_true')
    p.add_argument('--joint-source-reference',action='store_true')
    p.add_argument('--joint-independent-root-futures',action='store_true')
    p.add_argument('--joint-support-probability',type=float,default=.8)
    p.add_argument('--joint-wait-crossfit',action='store_true')
    p.add_argument('--joint-anchored-value',action='store_true')
    p.add_argument('--joint-contrast-precision',action='store_true')
    p.add_argument('--joint-binomial-value',action='store_true')
    p.add_argument('--joint-bounded-value',action='store_true')
    p.add_argument('--joint-success-potentials',action='store_true')
    p.add_argument('--joint-temporal-programs',action='store_true')
    p.add_argument('--joint-temporal-blending-levels',nargs='+',type=float,default=[0.,.5,1.])
    p.add_argument('--joint-prefix-refinement',action='store_true')
    p.add_argument('--joint-program-proposer',type=Path)
    p.add_argument('--joint-paired-nuisance',action='store_true')
    p.add_argument('--joint-conditional-calibration',action='store_true')
    p.add_argument('--joint-coma-credit',action='store_true')
    p.add_argument('--joint-nuisance-ridge',type=float,default=.25)
    p.add_argument('--joint-shared-potentials',action='store_true')
    p.add_argument('--joint-program-width',type=int,default=48)
    p.add_argument('--joint-program-updates',type=int,default=1500)
    p.add_argument('--joint-program-calibration',action='store_true')
    p.add_argument('--joint-program-completion',action='store_true')
    p.add_argument('--joint-paired-likelihood',action='store_true')
    p.add_argument('--joint-block-likelihood',action='store_true')
    p.add_argument('--joint-response-precision',action='store_true')
    p.add_argument('--joint-credit-margin',type=float,default=.15)
    p.add_argument('--joint-decision-tolerance',type=float,default=0.)
    p.add_argument('--joint-blending-penalty',type=float,default=0.)
    p.add_argument('--joint-fixed-intervention-clock',action='store_true')
    p.add_argument('--joint-batch-collection',type=int,default=0)
    p.add_argument('--joint-support-uncertainty',action='store_true')
    p.add_argument('--joint-temporal-credit',action='store_true')
    p.add_argument('--joint-factual-support',action='store_true')
    p.add_argument('--joint-failure-subjects',action='store_true')
    p.add_argument('--joint-observed-future',action='store_true')
    p.add_argument('--joint-physical-interventions',action='store_true')
    p.add_argument('--joint-physical-singletons',action='store_true')
    p.add_argument('--joint-preference-strength',type=float)
    p.add_argument('--joint-physical-span',type=int,default=10)
    p.add_argument('--joint-physical-reversal',action='store_true')
    p.add_argument('--joint-physical-cost',type=float,default=.02)
    p.add_argument('--joint-context-prediction',action='store_true')
    p.add_argument('--joint-context-interceptors',action='store_true')
    p.add_argument('--joint-conditional-commitment',action='store_true')
    p.add_argument('--joint-immediate-commitment',action='store_true')
    p.add_argument('--joint-contexts',type=int,default=0)
    p.add_argument('--joint-exhaustive-contexts',action='store_true')
    p.add_argument('--joint-max-held',type=int,default=0)
    p.add_argument('--joint-edge-futures',action='store_true')
    p.add_argument('--joint-balanced-futures',action='store_true')
    p.add_argument('--joint-future-repetitions',type=int,default=1)
    p.add_argument('--joint-future-start',type=int,default=0)
    p.add_argument('--joint-risk-feedback',action='store_true')
    p.add_argument('--joint-coupled-shortlist',type=int,default=0)
    p.add_argument('--joint-local-bandwidths',type=float,nargs='+',default=(.5,1.,2.))
    p.add_argument('--joint-local-response',action='store_true')
    p.add_argument('--joint-local-joint',action='store_true')
    p.add_argument('--joint-response-spans',type=int,nargs='+')
    p.add_argument('--joint-response-feature-version',type=int,choices=(2,3,4,5),default=2)
    p.add_argument('--joint-response-width',type=int,default=48)
    p.add_argument('--joint-project-returns',action='store_true')
    p.add_argument('--joint-outcome-strata',action='store_true')
    p.add_argument('--joint-factor-value',action='store_true')
    p.add_argument('--joint-unary-value',action='store_true')
    p.add_argument('--joint-factor-ridge',type=float,default=.25)
    p.add_argument('--joint-program-kernel',action='store_true')
    p.add_argument('--joint-midpoint-refinement',action='store_true')
    p.add_argument('--joint-remainder-refinement',action='store_true')
    p.add_argument('--joint-refinement-fraction',type=float,choices=(.25,.5),default=.5)
    p.add_argument('--joint-refinement-steps',type=int,default=0)
    p.add_argument('--joint-refinement-checkpoints',action='store_true')
    p.add_argument('--joint-phase-preserving-refinement',action='store_true')
    p.add_argument('--joint-residual-response',action='store_true')
    p.add_argument('--joint-action-support',action='store_true')
    p.add_argument('--joint-global-response',action='store_true')
    p.add_argument('--joint-duration-reference',action='store_true')
    p.add_argument('--joint-contrast-support',action='store_true')
    p.add_argument('--joint-response-margins',type=float,nargs='+',default=(.15,.3,.5))
    p.add_argument('--joint-response-aggregation',choices=('additive','bottleneck','joint','graph'),default='bottleneck')
    p.add_argument('--joint-local-strengths',type=float,nargs='+',default=(16.,64.))
    p.add_argument('--joint-local-authorities',type=float,nargs='+')
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
    p.add_argument('--joint-fixed-bank',action='store_true')
    p.add_argument('--joint-replay',type=Path)
    p.add_argument('--joint-projection-scales',type=float,nargs='+')
    p.add_argument('--joint-no-value-guidance',action='store_true')
    p.add_argument('--joint-epochs',type=int,default=6)
    p.add_argument('--matched-schedule',type=Path)
    args=p.parse_args()
    if args.joint_failure_roots and args.joint_success_roots:
        raise ValueError('A collection cannot select only failures and only successes simultaneously.')
    if args.joint_future_start<0:raise ValueError('Future continuation offsets must be nonnegative.')
    if args.joint_no_interaction:
        args.without_peer=True
        if args.mode not in ('joint-evaluate','joint-local-value','joint-shared-value'):
            raise ValueError('Complete interaction removal is supported by the joint conditional-value and evaluation paths.')
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
    if args.start and args.mode not in ('joint-target-motion','joint-supported-value','joint-shared-value','joint-local-value','joint-value-diagnostic','joint-interpolate','joint-train','joint-evaluate','mechanism-train','mechanism-refit','mechanism-evaluate'):
        args.without_peer=args.without_peer or json.loads(args.start.read_text()).get('without_peer',False)
    evaluation_ablation=((args.without_peer or args.joint_no_interaction,args.joint_no_cf,args.joint_no_interaction)
        if args.mode=='joint-evaluate' else (False,False,False))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,
                             initargs=(str(args.source),args.without_peer,evaluation_ablation)) as pool:
        if args.mode=='joint-target-motion': joint_target_motion(args,pool)
        elif args.mode=='joint-supported-value': joint_supported_value(args,pool)
        elif args.mode=='joint-shared-value': joint_shared_value(args,pool)
        elif args.mode=='joint-local-value': joint_local_value(args,pool)
        elif args.mode=='joint-risk-search': joint_risk_search(args,pool)
        elif args.mode=='joint-value-diagnostic': joint_value_diagnostic(args,pool)
        elif args.mode=='joint-interpolate': joint_interpolate(args,pool)
        elif args.mode=='joint-train': joint_train(args,pool)
        elif args.mode=='joint-evaluate' and (args.references or args.joint_ablation_suite or args.repetitions>1):
            scenes=bank(args.bank,args.per_cell)
            models=([(args.start,(False,False,False),'full'),(args.start,(True,False,True),'without_candidate_interaction'),
                     (args.start,(False,True,False),'without_conditional_value')]
                    if args.joint_ablation_suite else [(p,(False,False,False),p.parent.name) for p in [args.start,*(args.references or [])]])
            variants=[]
            for model,ablation,label in models:
                tasks=[]
                for repetition in range(args.repetitions):
                    for scene in scenes:
                        future=(None if repetition==0 else int(np.random.SeedSequence(
                            [scene[0],repetition,20261014]).generate_state(1)[0]))
                        tasks.append((scene,str(model),model.stat().st_mtime_ns,False,0,0,0,0,future,ablation))
                rows=list(pool.map(joint_episode,tasks,chunksize=1))
                configured=tuple(torch.load(model,map_location='cpu',weights_only=False).get('mechanism_ablation',(False,False,False)))
                configured=configured+(False,)*(3-len(configured))
                effective=tuple(a or b or c for a,b,c in zip(ablation,evaluation_ablation,configured))
                variant=dict(model=str(model),label=label,mechanism_removal=list(effective),summary=summarize(rows),
                    original_trajectories=summarize(rows[:len(scenes)]),
                    additional_futures=summarize(rows[len(scenes):]) if args.repetitions>1 else None,rows=rows)
                variants.append(variant)
                save_json(args.output,dict(protocol='same development scenes with paired noise repeats',
                    bank=args.bank,repetitions=args.repetitions,variants=variants))
                print(dict(phase='paired_development_ablation',model=str(model),
                    success=variant['summary']['success'],original=variant['original_trajectories']['success'],
                    additional=None if variant['additional_futures'] is None else variant['additional_futures']['success']),flush=True)
            result=dict(protocol='same development scenes with paired noise repeats',bank=args.bank,
                ablation_type='complete component removal from the same checkpoint' if args.joint_ablation_suite else 'separately supplied checkpoints',
                repetitions=args.repetitions,variants=variants,
                paired=[paired_summary(variants[0]['rows'],variant['rows']) for variant in variants[1:]])
            save_json(args.output,result)
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
