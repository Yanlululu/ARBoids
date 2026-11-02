"""Candidate-conditioned short stopping intervals for distance prediction.

This is an explicitly versioned development predictor. The auxiliary label is
the observed short-branch interval, not a terminal success/value target. Only
known nominal paths enter the stopped-distance base; learned unobserved tails
are never used to define a distance or risk.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

from citf.model import Config, Predictor
from citf.risk import pair_indices, rectified_gaussian_risk, bounded_mean
from citf.losses import objective
from citf.contracts import canonical_hash, file_hash, tree_hash


@dataclass(frozen=True)
class StoppingConfig(Config):
    revision: str = 'CITF-stopping-graph-v1'
    stop_weight: float = .1


class StoppingPredictor(Predictor):
    def __init__(self, cfg=StoppingConfig()):
        super().__init__(cfg)
        D = cfg.width
        self.interval = nn.Sequential(nn.Linear(2*D+6, D), nn.GELU(), nn.Linear(D, cfg.horizon))
        nn.init.zeros_(self.interval[-1].weight)
        nn.init.zeros_(self.interval[-1].bias)
        with torch.no_grad():
            self.interval[-1].bias[-1] = 10.

    def forward(self, b, history_cache=None):
        nom, candidate, pos, valid = b['nominal'], b['candidate'], b['positions'], b['entity_valid']
        B,C,E,T,_ = nom.shape
        if T != self.cfg.horizon or not valid[:,-1].all():
            raise ValueError('Stopping predictor schema differs')
        known = b['known_time_valid']; D = self.cfg.width
        h = self.encode_history(b) if history_cache is None else history_cache
        displacement = ((nom-pos[:,None,:,None])/10)*known[:,None,None,:,None]
        z = (h[:,None]+self.candidate_embed(torch.cat((candidate,displacement.flatten(-2)),-1))).reshape(B*C,E,D)
        mask = valid[:,None].expand(B,C,E).reshape(B*C,E)
        pn = nom.reshape(B*C,E,T,2)
        cur = pos[:,None].expand(B,C,E,2).reshape(B*C,E,2)
        dd = torch.linalg.vector_norm(pn[:,None]-pn[:,:,None],dim=-1)
        kt = known[:,None].expand(B,C,T).reshape(B*C,T)
        mind, when = dd.masked_fill(~kt[:,None,None,:],torch.inf).min(-1)
        last = (kt.sum(-1)-1)[:,None,None,None].expand(B*C,E,E,1)
        last_distance = dd.gather(-1,last).squeeze(-1)
        delta = cur[:,None]-cur[:,:,None]
        edges = torch.cat((delta/20,mind[...,None]/10,
                           ((last_distance-delta.norm(dim=-1))/10)[...,None],(when/T)[...,None]),-1)
        if self.cfg.fleet_candidate_context:
            if self.cfg.variant != 'candidate_graph':
                raise ValueError('This development identity uses candidate_graph attention')
            for block in self.context:
                z = block(z,mask,edges)
        z = z*mask[...,None]
        ts = torch.arange(1,T+1,device=nom.device,dtype=nom.dtype)*self.cfg.dt
        envelope = (ts-self.cfg.dt).clamp_min(0).square()/3.
        correction = self.cfg.residual_scale*torch.tanh(self.motion(z)).reshape(B,C,E,T,2)*envelope[None,None,None,:,None]
        xy = nom+correction*valid[:,None,:,None,None]
        i,j = pair_indices(E-1,nom.device)
        prefix = (nom[:,:,i]-nom[:,:,j]).norm(dim=-1).cummin(-1).values
        full_distance = prefix.masked_fill(~known[:,None,None,:],torch.inf).amin(-1)
        z = z.reshape(B,C,E,D)
        base = torch.cat((z[:,:,i]+z[:,:,j],(z[:,:,i]-z[:,:,j]).abs()),-1)
        dp = (pos[:,i]-pos[:,j])[:,None].expand(-1,C,-1,-1)/20
        dg = candidate[:,:,i,:2]-candidate[:,:,j,:2]
        one = torch.ones_like(full_distance[...,None]); dist = full_distance[...,None]/10
        left = torch.cat((base,dp,dist,dg,one),-1)
        right = torch.cat((base,-dp,dist,-dg,one),-1)
        q = .5*(self.pair(left)+self.pair(right))
        logits = .5*(self.interval(left)+self.interval(right))
        logits = logits.masked_fill(~known[:,None,None,:],-torch.inf)
        probability = logits.softmax(-1)
        stopped_base = (probability*prefix).sum(-1)
        mu = stopped_base+q[...,0]
        sigma = (F.softplus(q[...,1])+.03).clamp_max(4.)
        pair_valid = valid[:,i]&valid[:,j]
        risk = rectified_gaussian_risk(mu,sigma,self.cfg.clearance)
        mean_distance = mu*torch.special.ndtr(mu/sigma)+sigma*torch.exp(-.5*(mu/sigma).square())/math.sqrt(2*math.pi)
        return dict(xy=xy,mu=mu,sigma=sigma,risk=risk,mean_distance=mean_distance,
                    det_risk=(self.cfg.clearance-mean_distance).clamp_min(0).square(),
                    pair_valid=pair_valid,nominal_mu=stopped_base,stop_logits=logits,
                    mean_stop_steps=(probability*torch.arange(1,T+1,device=nom.device)).sum(-1),
                    stop_weight=self.cfg.stop_weight)


def stopping_objective(out,b,**kwargs):
    result = objective(out,b,**kwargs)
    target = b['time_valid'].sum(-1)-1
    B,C,R = target.shape; P = out['stop_logits'].shape[2]
    logp = out['stop_logits'].log_softmax(-1)[:,:,None].expand(-1,-1,R,-1,-1)
    values = -logp.gather(-1,target[:,:,:,None,None].expand(B,C,R,P,1)).squeeze(-1)
    mask = out['pair_valid'][:,None,None,:]&b['candidate_valid'][:,:,None,None]
    stop = bounded_mean(values,mask)
    result['stop'] = stop
    if kwargs.get('mode','full') != 'motion_only':
        result['total'] = result['total']+out['stop_weight']*stop
    return result


def implementation_hash(path):
    return canonical_hash({'core':tree_hash(path),'stopping_graph':file_hash(__file__)})


def train_main():
    import argparse
    import citf.train as training
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--stop-weight',type=float,default=.1)
    args, remaining = ap.parse_known_args()
    if not math.isfinite(args.stop_weight) or args.stop_weight < 0:
        raise ValueError('Invalid short interval loss weight')
    sys.argv = [sys.argv[0],*remaining]
    training.Config = lambda **kw: StoppingConfig(stop_weight=args.stop_weight,**kw)
    training.Predictor = StoppingPredictor
    training.objective = stopping_objective
    training.tree_hash = implementation_hash
    training.main()


def evaluate_main():
    import citf.collect as collection
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity
    def identity(*args,**kwargs):
        result = original_identity(*args,**kwargs)
        result['method'] = 'CITF-Stopping-development'
        result['runtime_code'] = canonical_hash({'core':result['runtime_code'],
                                                 'stopping_graph':file_hash(__file__)})
        return result
    collection.Config = StoppingConfig
    collection.Predictor = StoppingPredictor
    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='train':
        sys.argv.pop(1); train_main()
    else:
        evaluate_main()
