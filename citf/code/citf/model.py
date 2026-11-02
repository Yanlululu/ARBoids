"""Audited candidate-conditioned interaction Transformer.

This revision adds symmetric pair readout, explicit masks and configurable
horizon/fleet size. It must be RETRAINED: the supplied 46/48 checkpoint is not
assigned this revision's identity. All outputs are physical short-motion
statistics. There is no CV, Bellman target or terminal success head.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import torch
from torch import Tensor,nn
import torch.nn.functional as F
from .risk import pair_indices,rectified_gaussian_risk

@dataclass(frozen=True)
class Config:
    variant:str='candidate_graph'
    width:int=56
    heads:int=4
    history:int=8
    horizon:int=16
    dt:float=.2
    clearance:float=7.
    residual_scale:float=9.
    fleet_candidate_context:bool=True
    revision:str='CITF-stopped-direct-v2'
    def __post_init__(self):
        if self.variant not in ('candidate_graph','transformer','gru','perceiver'):raise ValueError('Unsupported implemented variant')
        if min(self.width,self.heads,self.history,self.horizon)<1 or self.width%self.heads:raise ValueError('Invalid dimensions')
        if min(self.dt,self.clearance)<=0:raise ValueError('Invalid physical scales')

class Attention(nn.Module):
    def __init__(self,d,h,relational=False):
        super().__init__();self.heads=h;self.head_dim=d//h
        self.norm=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d)
        self.edge=nn.Sequential(nn.Linear(5,d),nn.GELU(),nn.Linear(d,h)) if relational else None
        self.out=nn.Linear(d,d);self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
    def forward(self,x,valid,edge=None,causal=False):
        B,E,D=x.shape
        if not valid.any(-1).all():raise ValueError('Attention has no valid token')
        q,k,v=self.qkv(self.norm(x)).reshape(B,E,3,self.heads,self.head_dim).permute(2,0,3,1,4)
        logit=q@k.transpose(-1,-2)/math.sqrt(self.head_dim)
        if self.edge is not None:
            if edge is None:raise ValueError('Relational attention requires geometric edges')
            logit=logit+self.edge(edge).permute(0,3,1,2)
        mask=valid[:,None,None,:].expand(B,1,E,E)
        if causal:
            mask=mask&torch.ones(E,E,device=x.device,dtype=torch.bool).tril()[None,None]
            # Padding queries are not outputs; give them one numerical key.
            empty=~mask.any(-1,keepdim=True)
            mask=mask|(empty&valid[:,None,None,:])
        weight=logit.masked_fill(~mask,-torch.inf).softmax(-1)
        y=x+self.out((weight@v).transpose(1,2).reshape(B,E,D))
        y=y+self.ff(self.norm2(y))
        return y*valid[...,None]

class Predictor(nn.Module):
    def __init__(self,cfg:Config=Config()):
        super().__init__();self.cfg=cfg;D=cfg.width
        self.history_embed=nn.Linear(8,D);self.time_embed=nn.Parameter(torch.randn(cfg.history,D)*.01)
        if cfg.variant=='gru':self.temporal=nn.GRU(D,D,batch_first=True)
        else:self.temporal=nn.ModuleList([Attention(D,cfg.heads),Attention(D,cfg.heads)])
        self.historical_context=Attention(D,cfg.heads)
        self.time_context=nn.Sequential(nn.Linear(2,D),nn.GELU(),nn.Linear(D,D))
        self.candidate_embed=nn.Sequential(nn.Linear(8+2*cfg.horizon,D),nn.GELU(),nn.Linear(D,D))
        if cfg.variant=='candidate_graph':self.context=nn.ModuleList([Attention(D,cfg.heads,True),Attention(D,cfg.heads,True)])
        elif cfg.variant=='perceiver':
            self.latent=nn.Parameter(torch.randn(4,D)*.02);self.read=nn.MultiheadAttention(D,cfg.heads,batch_first=True)
            self.latent_block=Attention(D,cfg.heads);self.write=nn.MultiheadAttention(D,cfg.heads,batch_first=True)
        else:self.context=nn.ModuleList([Attention(D,cfg.heads)])
        self.motion=nn.Sequential(nn.LayerNorm(D),nn.Linear(D,D),nn.GELU(),nn.Linear(D,2*cfg.horizon))
        self.pair=nn.Sequential(nn.Linear(2*D+6,D),nn.GELU(),nn.Linear(D,2))
        for head in (self.motion[-1],self.pair[-1]):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        with torch.no_grad():self.pair[-1].bias[1]=-1.5
    def encode_history(self,b):
        h=b['history'];B,E,L,Ft=h.shape
        if Ft!=8 or L!=self.cfg.history:raise ValueError('Declared history schema mismatch')
        valid=b['entity_valid'];hv=b.get('history_valid',valid[:,:,None].expand(B,E,L)).clone()
        if hv.shape!=(B,E,L):raise ValueError('history_valid shape')
        hv[:,:, -1]|=~valid  # dummy token for padded entities only
        z=self.history_embed(h)+self.time_embed[None,None]
        z=z.reshape(B*E,L,-1);mask=hv.reshape(B*E,L)
        if self.cfg.variant=='gru':
            z=self.temporal(z*mask[...,None])[0][:,-1]
        else:
            for block in self.temporal:z=block(z,mask,causal=True)
            z=z[:,-1]
        z=z.reshape(B,E,-1)*valid[...,None]
        context=b['time_context']
        if context.shape!=(B,2) or not torch.isfinite(context).all():raise ValueError('Known time context [B,2] required')
        return self.historical_context(z,valid)+self.time_context(context)[:,None]*valid[...,None]
    def forward(self,b,history_cache=None):
        nom=b['nominal'];candidate=b['candidate'];pos=b['positions'];valid=b['entity_valid']
        B,C,E,T,_=nom.shape;D=self.cfg.width
        if T!=self.cfg.horizon or candidate.shape!=(B,C,E,8) or pos.shape!=(B,E,2):raise ValueError('Candidate schema mismatch')
        if valid.shape!=(B,E) or not valid[:,-1].all():raise ValueError('Final entity must be valid attacker')
        if not (valid[:,:-1].sum(-1)>=2).all():raise ValueError('Need at least two defenders')
        known=b['known_time_valid']
        if known.shape!=(B,T) or known.dtype!=torch.bool or not known.any(-1).all():raise ValueError('Known interval mask [B,T] required')
        if ((~known[:,:-1])&known[:,1:]).any():raise ValueError('Known time must be a prefix')
        h=self.encode_history(b) if history_cache is None else history_cache
        displacement=((nom-pos[:,None,:,None])/10)*known[:,None,None,:,None]
        z=(h[:,None]+self.candidate_embed(torch.cat((candidate,displacement.flatten(-2)),-1))).reshape(B*C,E,D)
        mask=valid[:,None].expand(B,C,E).reshape(B*C,E)
        pn=nom.reshape(B*C,E,T,2);cur=pos[:,None].expand(B,C,E,2).reshape(B*C,E,2)
        dd=torch.linalg.vector_norm(pn[:,None]-pn[:,:,None],dim=-1)
        kt=known[:,None].expand(B,C,T).reshape(B*C,T)
        mind,when=dd.masked_fill(~kt[:,None,None,:],torch.inf).min(-1)
        last=(kt.sum(-1)-1)[:,None,None,None].expand(B*C,E,E,1)
        last_distance=dd.gather(-1,last).squeeze(-1)
        delta=cur[:,None]-cur[:,:,None];current_d=delta.norm(dim=-1)
        edges=torch.cat((delta/20,mind[...,None]/10,((last_distance-current_d)/10)[...,None],(when/T)[...,None]),-1)
        if self.cfg.fleet_candidate_context:
            if self.cfg.variant=='perceiver':
                latent=self.latent[None].expand(B*C,-1,-1)
                latent=latent+self.read(latent,z,z,key_padding_mask=~mask,need_weights=False)[0]
                latent=self.latent_block(latent,torch.ones(B*C,4,dtype=torch.bool,device=z.device))
                z=z+self.write(z,latent,latent,need_weights=False)[0]
            else:
                for block in self.context:z=block(z,mask,edges if self.cfg.variant=='candidate_graph' else None)
        z=z*mask[...,None]
        ts=torch.arange(1,T+1,device=nom.device,dtype=nom.dtype)*self.cfg.dt
        envelope=(ts-self.cfg.dt).clamp_min(0).square()/3.
        correction=self.cfg.residual_scale*torch.tanh(self.motion(z)).reshape(B,C,E,T,2)*envelope[None,None,None,:,None]
        correction=correction*valid[:,None,:,None,None]
        xy=nom+correction;i,j=pair_indices(E-1,nom.device)
        # Direct stopped-interval distribution head. Auxiliary unobserved trajectory
        # tails never enter mu. Only the known timeout prefix forms a nominal feature.
        nom_d=(nom[:,:,i]-nom[:,:,j]).norm(dim=-1).masked_fill(~known[:,None,None,:],torch.inf).amin(-1)
        z=z.reshape(B,C,E,D);base=torch.cat((z[:,:,i]+z[:,:,j],(z[:,:,i]-z[:,:,j]).abs()),-1)
        dp=(pos[:,i]-pos[:,j])[:,None].expand(-1,C,-1,-1)/20
        dg=candidate[:,:,i,:2]-candidate[:,:,j,:2]
        one=torch.ones_like(nom_d[...,None]);dist=nom_d[...,None]/10
        # Same edge evaluated in both directions. No physical ordering shortcut.
        qij=self.pair(torch.cat((base,dp,dist,dg,one),-1))
        qji=self.pair(torch.cat((base,-dp,dist,-dg,one),-1));q=.5*(qij+qji)
        mu=nom_d+q[...,0]  # unconstrained Gaussian location; physical distance is max(0,Z)
        sigma=(F.softplus(q[...,1])+.03).clamp_max(4.)
        pair_valid=valid[:,i]&valid[:,j]
        risk=rectified_gaussian_risk(mu,sigma,self.cfg.clearance)
        mean_distance=mu*torch.special.ndtr(mu/sigma)+sigma*torch.exp(-.5*(mu/sigma).square())/math.sqrt(2*math.pi)
        return dict(xy=xy,mu=mu,sigma=sigma,risk=risk,mean_distance=mean_distance,det_risk=(self.cfg.clearance-mean_distance).clamp_min(0).square(),
                    pair_valid=pair_valid,nominal_mu=nom_d)
