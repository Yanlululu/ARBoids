"""Persistent interception, orbital approach and coordinated closing feedback.

Programs remain state feedback through the terminal transition. Assignment and
waypoints use the currently observed fleet; no sampled future or hidden agility
enters the controller. This module supplements the recorded opening programs.
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from experiments.interception_reserve import tracking_thrust


def arrangement(state, guard, program):
    """Return jointly assigned current goals, roles and source mixing weights."""
    position=state[:-1,:2];enemy=state[-1,:2];n=len(position)
    u=enemy/max(np.linalg.norm(enemy),1e-6);v=np.array([-u[1],u[0]])
    goals=np.zeros((n,2),np.float32);roles=np.zeros(n,np.float32);mix=np.zeros(n,np.float32)
    goals[guard]=12.5*u;roles[guard]=1
    if program==0:return goals,roles,mix
    free=np.array([i for i in range(n) if i!=guard]);m=len(free)
    relative=position[free]-enemy;distance=np.linalg.norm(relative,axis=-1)
    angle=np.arctan2(relative@v,relative@u)
    # Four flank/block positions and one inner striker for the six-boat task.
    # The same construction spreads the available block positions for other N.
    side_count=max(0,m-1)
    if side_count:
        half=(side_count+1)//2
        positive=np.linspace(.25*math.pi,.65*math.pi,half)
        slots=np.sort(np.concatenate((-positive,positive)))[:side_count]
    else:slots=np.empty(0)
    shift=0. if program==1 else (.18*math.pi if program==2 else -.18*math.pi)
    slots=slots+shift
    radius=np.where(abs(slots)<.5*math.pi,14.,18.)
    slot_angle=np.r_[slots,math.pi];slot_radius=np.r_[radius,20.]
    angular=np.arctan2(np.sin(slot_angle[None]-angle[:,None]),np.cos(slot_angle[None]-angle[:,None]))
    next_angle=angle[:,None]+np.clip(angular,-.45,.45)
    approach_radius=np.where(abs(angular)>.45,np.maximum(24.,distance[:,None]-4.),slot_radius[None])
    candidate=enemy+approach_radius[...,None]*(np.cos(next_angle)[...,None]*u+np.sin(next_angle)[...,None]*v)
    candidate[:,-1]=enemy-20.*u
    delta=candidate-position[free,None]
    turning=np.arctan2(delta[...,1],delta[...,0])-state[free,None,2]
    turning=abs(np.arctan2(np.sin(turning),np.cos(turning)))
    cost=abs(angular)*np.minimum(distance[:,None],24.)+abs(distance[:,None]-slot_radius[None])+4*turning
    cost[:,-1]=np.linalg.norm(delta[:,-1],axis=-1)+4*turning[:,-1]
    boats,assigned=linear_sum_assignment(cost)
    blocking=(relative@u>-3.)&(distance<28.)
    ready=int(blocking.sum())>=min(2,side_count)
    for boat,slot in zip(boats,assigned):
        idx=free[boat];roles[idx]=3 if slot==m-1 else 2
        if slot==m-1:
            goals[idx]=enemy+.6*state[-1,3:5] if ready else candidate[boat,slot]
            mix[idx]=.35 if ready else 0.
        else:
            goals[idx]=candidate[boat,slot]
            if ready and abs(angular[boat,slot])<=.45:
                radius_now=max(10.,slot_radius[slot]-4.)
                goals[idx]=enemy+radius_now*(np.cos(slot_angle[slot])*u+np.sin(slot_angle[slot])*v)
    return goals,roles,mix


def parameters(state,guard,program):
    goals,roles,mix=arrangement(state,guard,program)
    return np.column_stack((goals,roles,np.full(len(goals),program),mix)).astype(np.float32)


def forces(state,reference,configuration):
    """Recompute the joint assignment on every real and nominal feedback step."""
    current=[]
    for observed,row in zip(state.detach().cpu().numpy(),configuration.detach().cpu().numpy()):
        guard=int(np.flatnonzero(row[:,2]==1)[0]);program=int(round(row[0,3]))
        current.append(parameters(observed,guard,program))
    current=torch.as_tensor(np.stack(current),dtype=state.dtype,device=state.device)
    tracking=tracking_thrust(state,current[...,:2])
    mix=current[...,4,None]
    combined=mix*reference+(1-mix)*tracking
    return torch.where((current[...,2]>.5)[...,None],combined,reference),current
