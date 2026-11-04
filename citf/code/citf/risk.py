"""Short-horizon physical clearance scoring. No terminal task value is learned."""
from __future__ import annotations
import math
import torch
from torch import Tensor


def signed_log1p(value: Tensor) -> Tensor:
    """Signed log compression, including its unit derivative at zero."""
    sign = torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
    return sign * torch.log1p(sign * value)


def rectified_gaussian_risk(mu: Tensor, sigma: Tensor, clearance: float = 7.) -> Tensor:
    """E[(r-max(0,Z))_+^2], Z~N(mu,sigma^2), NOT a truncated normal.

    Computation in float64 reduces cancellation. The result has units m^2.
    Variance describes variation across physical future disturbances, not an
    epistemic confidence bound and not a safety certificate.
    """
    if mu.shape != sigma.shape or clearance <= 0:
        raise ValueError('Matching mean/scale tensors and positive clearance required')
    if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('Finite mean and strictly positive scale required')
    dtype=mu.dtype; m=mu.double();s=sigma.double();r=float(clearance)
    b=r-m;z=b/s;z0=-m/s
    cdf=torch.special.ndtr(z);f0=torch.special.ndtr(z0)
    pdf=torch.exp(-.5*z*z)/math.sqrt(2*math.pi)
    p0=torch.exp(-.5*z0*z0)/math.sqrt(2*math.pi)
    score=(b*b+s*s)*cdf+b*s*pdf+(r*r-b*b-s*s)*f0-s*(2*b+m)*p0
    return score.clamp(0.,r*r).to(dtype)


def rectified_gaussian_nll(d: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Unweighted proper likelihood of positive distances + atom at zero."""
    if (d < 0).any() or (sigma <= 0).any():
        raise ValueError('Distances must be nonnegative and scale positive')
    pos=.5*((d-mu)/sigma).square()+sigma.log()+.5*math.log(2*math.pi)
    atom=-torch.special.log_ndtr(-mu/sigma)
    return torch.where(d > 0,pos,atom)


def pair_indices(defenders: int, device=None) -> Tensor:
    if defenders < 2:raise ValueError('At least two defenders required')
    return torch.triu_indices(defenders,defenders,1,device=device)


def branch_minima(xy: Tensor, time_valid: Tensor, entity_valid: Tensor) -> tuple[Tensor,Tensor]:
    """xy[B,C,R,E,T,2], time_valid[B,C,R,T], final entity is attacker.

    Target is the synchronous minimum on the *observed stopped interval*
    min(H,physical task termination). The transition into the event is included.
    Missing later samples are masked, never replaced by zero positions.
    """
    if xy.ndim!=6 or xy.shape[-1]!=2:raise ValueError('Expected [B,C,R,E,T,2]')
    B,C,R,E,T,_=xy.shape
    if time_valid.shape!=(B,C,R,T) or entity_valid.shape!=(B,E):raise ValueError('Mask shape mismatch')
    if time_valid.dtype!=torch.bool or entity_valid.dtype!=torch.bool:raise ValueError('Boolean masks required')
    if not time_valid.any(-1).all():raise ValueError('Each branch must contain one real transition')
    if ((~time_valid[...,:-1])&time_valid[...,1:]).any():raise ValueError('Stopped branch reactivated')
    i,j=pair_indices(E-1,xy.device)
    d=torch.linalg.vector_norm(xy[:,:,:,i]-xy[:,:,:,j],dim=-1)
    d=d.masked_fill(~time_valid[:,:,:,None,:],torch.inf).amin(-1)
    mask=entity_valid[:,i]&entity_valid[:,j]
    return d,mask


def bounded_mean(error:Tensor,mask:Tensor) -> Tensor:
    mask=torch.broadcast_to(mask,error.shape)
    return torch.where(mask,error,torch.zeros_like(error)).sum()/mask.sum().clamp_min(1)
