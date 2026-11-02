"""Current discrete fusion configuration only. No temporal plan or value head."""
from __future__ import annotations
import itertools
import numpy as np
import torch


def candidate_grid(prior:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    if prior.ndim!=2 or prior.shape[-1]!=2:raise ValueError('prior [N,2]')
    if not torch.isfinite(prior).all() or ((prior<0)|(prior>1)).any():raise ValueError('Bad prior')
    lv=torch.stack((torch.zeros_like(prior),prior,torch.ones_like(prior)),1)
    k=torch.arange(9,device=prior.device);grid=torch.stack((lv[:,k//3,0],lv[:,k%3,1]),-1)
    equal=(grid[:,:,None]-grid[:,None,:]).abs().amax(-1)<1e-7
    duplicate=(equal&torch.ones(9,9,dtype=torch.bool,device=prior.device).tril(-1)).any(-1)
    return grid,~duplicate


def nominal_score(configs,unary,pair,weight=32.):
    configs=np.asarray(configs,dtype=int);n=unary.shape[0];s=np.zeros(len(configs))
    for i in range(n):
        s+=unary[i,configs[:,i]]
        for j in range(i):s+=weight*pair[j,i,configs[:,j],configs[:,i]]
    return s


def nominal_beam(unary,pair,weight=32.,width=128):
    """Independent physical pair table used ONLY for candidate generation."""
    n,k=unary.shape;plans=np.empty((1,0),dtype=int);scores=np.zeros(1)
    for i in range(n):
        joint=np.repeat(plans,k,axis=0);last=np.tile(np.arange(k),len(plans))
        cost=np.repeat(scores,k)+unary[i,last]
        for j in range(i):cost+=weight*pair[j,i,joint[:,j],last]
        ix=np.argsort(cost,kind='stable')[:width];plans=np.column_stack((joint[ix],last[ix]));scores=cost[ix]
    finite=np.isfinite(scores)
    if not finite.any():raise ValueError('No finite nominal configuration')
    return plans[finite],scores[finite]


def whole_configuration_search(initial,evaluate,valid_choices,budget=96,rounds=1):
    """Recompute ALL conditional neural scores after any complete-config change.

    Returns full evaluator trace; duplicates are cached and cannot consume
    extra statistical weight. Fixed-bank evaluation uses rounds=0.
    """
    if budget<1 or rounds<0:raise ValueError('Invalid budget')
    cache={};calls=0
    def score(keys):
        nonlocal calls
        keys=list(dict.fromkeys(tuple(map(int,k)) for k in keys))
        keys=[k for k in keys if k not in cache][:budget-len(cache)]
        if not keys:return
        v=np.asarray(evaluate(np.asarray(keys)),dtype=float)
        if v.shape!=(len(keys),) or not np.isfinite(v).all():raise ValueError('Invalid complete-configuration scores')
        cache.update(zip(keys,v.tolist()));calls+=1
    score(initial)
    if not cache:raise ValueError('No candidate evaluated')
    for _ in range(rounds):
        best=min(cache,key=lambda k:(cache[k],k));neighbors=[]
        for i,values in enumerate(valid_choices):
            for k in values:
                p=list(best);p[i]=int(k);neighbors.append(p)
        score(neighbors)
    best=min(cache,key=lambda k:(cache[k],k))
    return np.asarray(best),dict(cost=cache[best],evaluated_unique=len(cache),forward_batches=calls,
        configurations=[list(k) for k in cache],scores=list(cache.values()))


def source_center_config(grid,valid,prior):
    """Canonical valid index of each source centre, including endpoint deduplication."""
    grid=np.asarray(grid);valid=np.asarray(valid,dtype=bool);prior=np.asarray(prior)
    if grid.ndim!=3 or grid.shape[-1]!=2 or valid.shape!=grid.shape[:2] or prior.shape!=(grid.shape[0],2):
        raise ValueError('Candidate-centre shape mismatch')
    error=np.max(np.abs(grid-prior[:,None,:]),axis=-1)
    error=np.where(valid,error,np.inf)
    centre=np.argmin(error,axis=-1)
    if not np.isfinite(error[np.arange(len(centre)),centre]).all() or (error[np.arange(len(centre)),centre]>1e-6).any():
        raise ValueError('Source centre is absent from the valid candidate grid')
    return centre.astype(np.int64)


def initialize_bank(ranked,grid,valid,prior,bank=12,budget=96):
    """One of the existing slots is reserved for the source centre, without extra budget."""
    if min(bank,budget)<1:raise ValueError('Positive candidate bank and budget required')
    centre=source_center_config(grid,valid,prior)
    initial=[tuple(centre.tolist())]
    for row in np.asarray(ranked,dtype=np.int64):
        key=tuple(row.tolist())
        if key not in initial:initial.append(key)
        if len(initial)>=min(bank,budget):break
    return np.asarray(initial[:min(bank,budget)],dtype=np.int64),centre
