"""Physical trajectory + proper distance likelihood + geometric choice learning."""
import torch
import torch.nn.functional as F
from .risk import branch_minima,rectified_gaussian_nll,bounded_mean


def objective(out,b,clearance=7.,weight=32.,mode='full',distance_weight=.08,risk_weight=.25,rank_weight=.05,temperature=8.):
    if mode not in ('full','predictive_only','motion_only'):raise ValueError('Unknown loss mode')
    real=b['real_positions'].detach();tv=b['time_valid'];ev=b['entity_valid'];cv=b['candidate_valid']
    d,pair_valid=branch_minima(real,tv,ev)
    count=tv.sum(2).clamp_min(1)
    mean=(real*tv[:,:,:,None,:,None]).sum(2)/count[:,:,None,:,None]
    posmask=ev[:,None,:,None,None]&tv.any(2)[:,:,None,:,None]&cv[:,:,None,None,None]
    motion=bounded_mean(F.smooth_l1_loss(out['xy'],mean,reduction='none'),posmask)
    pairmask=pair_valid[:,None,None,:]&cv[:,:,None,None]
    nll=bounded_mean(rectified_gaussian_nll(d,out['mu'][:,:,None],out['sigma'][:,:,None]),pairmask)
    true_risk=(clearance-d).clamp_min(0).square().mean(2)
    # Unweighted bounded mean target. No outcome-dependent likelihood weights.
    risk=bounded_mean(((out['risk']-true_risk)/clearance**2).square(),pair_valid[:,None,:]&cv[:,:,None])
    empirical=b['unary']+weight*(true_risk*pair_valid[:,None]).sum(-1)
    estimated=b['unary']+weight*(out['risk']*pair_valid[:,None]).sum(-1)
    if not cv.any(-1).all():raise ValueError('No feasible candidate')
    target_logits=(-empirical/temperature).masked_fill(~cv,-torch.inf)
    logits=(-estimated/temperature).masked_fill(~cv,-torch.inf)
    q=target_logits.softmax(-1).detach();logp=logits.log_softmax(-1)
    rank=-(q*torch.where(cv,logp,torch.zeros_like(logp))).sum(-1).mean()
    total=motion if mode=='motion_only' else motion+distance_weight*nll
    if mode=='full':total=total+risk_weight*risk+rank_weight*rank
    return dict(total=total,motion=motion,nll=nll,risk=risk,rank=rank,empirical_cost=empirical,estimated_cost=estimated)
