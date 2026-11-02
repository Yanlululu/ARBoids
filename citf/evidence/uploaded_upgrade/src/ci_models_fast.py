
"""CI-domain predictors. No terminal labels, value heads, actor replacement or CV.
Published architectures are adapted to continuous candidate-control/geometry
inputs, not asserted to reproduce the authors' AIS/SMAC benchmarks.
"""
from dataclasses import dataclass
import math,torch
from torch import nn
import torch.nn.functional as F
VARIANTS=('gru','transformer','dstgt_fixed','mact_perceiver','differential','sga_skip','candidate_graph')
WIDTH={'gru':96,'transformer':64,'dstgt_fixed':60,'mact_perceiver':56,'differential':60,'sga_skip':56,'candidate_graph':56}
@dataclass
class Config:
 variant:str='candidate_graph'
 width:int=56
 heads:int=4
 horizon:int=16
 history:int=8
 residual_scale:float=9.
 causal_first_step:bool=True
class Block(nn.Module):
 def __init__(self,d,h):
  super().__init__();self.n1=nn.LayerNorm(d);self.a=nn.MultiheadAttention(d,h,batch_first=True,dropout=0)
  self.n2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
 def forward(self,x,mask=None):
  z=self.n1(x);x=x+self.a(z,z,z,attn_mask=mask,need_weights=False)[0];return x+self.ff(self.n2(x))
class SAGE(nn.Module):
 def __init__(self,d):super().__init__();self.own=nn.Linear(d,d);self.peer=nn.Linear(d,d,bias=False);self.norm=nn.LayerNorm(d)
 def forward(self,x,adj):
  adj=adj/adj.sum(-1,keepdim=True).clamp_min(1);return self.norm(x+F.gelu(self.own(x)+self.peer(adj@x)))
class DifferentialBlock(nn.Module):
 def __init__(self,d,h):
  super().__init__();self.h=h;self.d=d//h
  self.q=nn.Linear(d,2*d);self.k=nn.Linear(d,2*d);self.v=nn.Linear(d,d)
  self.lam=nn.Parameter(torch.zeros(h));self.out=nn.Linear(d,d);self.norm=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
 def forward(self,x):
  b,n,d=x.shape;z=self.norm(x)
  q=self.q(z).reshape(b,n,2,self.h,self.d).permute(2,0,3,1,4);k=self.k(z).reshape(b,n,2,self.h,self.d).permute(2,0,3,1,4)
  v=self.v(z).reshape(b,n,self.h,self.d).transpose(1,2)
  a=(q@k.transpose(-1,-2)/math.sqrt(self.d)).softmax(-1)
  w=a[0]-self.lam.sigmoid()[None,:,None,None]*a[1]
  x=x+self.out((w@v).transpose(1,2).reshape(b,n,d));return x+self.ff(self.norm(x))
class RelationalBlock(nn.Module):
 def __init__(self,d,h):
  super().__init__();self.h=h;self.d=d//h;self.norm=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.edge=nn.Sequential(nn.Linear(5,d),nn.GELU(),nn.Linear(d,h));self.out=nn.Linear(d,d);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
 def forward(self,x,edges):
  b,n,d=x.shape;q,k,v=self.qkv(self.norm(x)).reshape(b,n,3,self.h,self.d).permute(2,0,3,1,4)
  a=(q@k.transpose(-1,-2)/math.sqrt(self.d)+self.edge(edges).permute(0,3,1,2)).softmax(-1)
  x=x+self.out((a@v).transpose(1,2).reshape(b,n,d));return x+self.ff(self.norm(x))
class Predictor(nn.Module):
 def __init__(self,cfg=Config()):
  super().__init__();self.cfg=cfg;d=cfg.width;h=cfg.heads
  if cfg.variant not in VARIANTS:raise ValueError(cfg.variant)
  self.history_embed=nn.Linear(8,d);self.temporal_pos=nn.Parameter(torch.randn(8,d)*.01)
  if cfg.variant=='gru':self.gru=nn.GRU(d,d,batch_first=True)
  else:self.temporal=Block(d,h)
  if cfg.variant not in ('gru','sga_skip'):self.temporal2=Block(d,h)
  if cfg.variant=='dstgt_fixed':self.sage_history=SAGE(d);self.sage_candidate=SAGE(d)
  if cfg.variant=='sga_skip':
   self.fine=nn.Conv1d(d,d,3,padding=1);self.coarse=nn.Conv1d(d,d,3,padding=1,stride=2)
   self.coarse_attn=Block(d,h);self.scale_fuse=nn.Linear(2*d,d)
  self.candidate_embed=nn.Sequential(nn.Linear(40,d),nn.GELU(),nn.Linear(d,d))
  if cfg.variant in ('transformer','dstgt_fixed','sga_skip'):self.context=Block(d,h)
  if cfg.variant=='mact_perceiver':
   self.latent=nn.Parameter(torch.randn(4,d)*.02);self.read=nn.MultiheadAttention(d,h,batch_first=True)
   self.latent_block=Block(d,h);self.write=nn.MultiheadAttention(d,h,batch_first=True);self.norm=nn.LayerNorm(d)
  if cfg.variant=='differential':self.diff=DifferentialBlock(d,h)
  if cfg.variant=='candidate_graph':self.rel=RelationalBlock(d,h);self.rel2=RelationalBlock(d,h)
  self.motion=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,32))
  self.pair=nn.Sequential(nn.Linear(2*d+6,d),nn.GELU(),nn.Linear(d,2))
  nn.init.zeros_(self.motion[-1].weight);nn.init.zeros_(self.motion[-1].bias)
  nn.init.zeros_(self.pair[-1].weight);nn.init.zeros_(self.pair[-1].bias)
  with torch.no_grad():self.pair[-1].bias[1]=-1.5
 def forward(self,b):
  hist=b['history'];nom=b['nominal'];candidate=b['candidate'];pos=b['positions']
  B,C,E,T,_=nom.shape;L=hist.shape[2];d=self.cfg.width
  h=self.history_embed(hist)+self.temporal_pos[None,None]
  if self.cfg.variant=='dstgt_fixed':
   positions=hist[...,:2]*60;distance=torch.cdist(positions.permute(0,2,1,3).reshape(B*L,E,2),positions.permute(0,2,1,3).reshape(B*L,E,2))
   adj=(distance<20).to(h.dtype);adj.diagonal(dim1=-2,dim2=-1).zero_()
   # Correct: graph aggregation precedes prediction; each batch has its own graph.
   h=self.sage_history(h.permute(0,2,1,3).reshape(B*L,E,d),adj).reshape(B,L,E,d).permute(0,2,1,3)
  h=h.reshape(B*E,L,d);mask=torch.triu(torch.ones(L,L,dtype=torch.bool,device=h.device),1)
  if self.cfg.variant=='gru':h=self.gru(h)[0][:,-1]
  elif self.cfg.variant=='sga_skip':
   fine=self.temporal(self.fine(h.transpose(1,2)).transpose(1,2),mask)
   coarse=self.coarse(h.transpose(1,2)).transpose(1,2)
   coarse=self.coarse_attn(coarse,torch.triu(torch.ones(coarse.shape[1],coarse.shape[1],dtype=torch.bool,device=h.device),1))
   h=self.scale_fuse(torch.cat((fine[:,-1],coarse[:,-1]),-1))
  else:h=self.temporal2(self.temporal(h,mask),mask)[:,-1]
  h=h.reshape(B,E,d)
  displacement=(nom-pos[:,None,:,None])/10
  tokens=(h[:,None]+self.candidate_embed(torch.cat((candidate,displacement.flatten(-2)),-1))).reshape(B*C,E,d)
  pn=nom.reshape(B*C,E,T,2)
  delta=pn[:,None]-pn[:,:,None];dist=torch.linalg.vector_norm(delta,dim=-1)
  mind,timeidx=dist.min(-1)
  cur=pos[:,None].expand(B,C,E,2).reshape(B*C,E,2)
  cdelta=cur[:,None]-cur[:,:,None];c_dist=torch.linalg.vector_norm(cdelta,dim=-1)
  edges=torch.cat((cdelta/20,mind[...,None]/10,((dist[...,-1]-c_dist)/10)[...,None],(timeidx.float()/16)[...,None]),-1)
  if self.cfg.variant=='gru':tokens=tokens+tokens.mean(1,keepdim=True)
  elif self.cfg.variant=='dstgt_fixed':
   adj=(mind<15).to(tokens.dtype);adj.diagonal(dim1=-2,dim2=-1).zero_()
   tokens=self.context(self.sage_candidate(tokens,adj))
  elif self.cfg.variant=='mact_perceiver':
   z=self.latent[None].expand(B*C,-1,-1);z=z+self.read(z,tokens,tokens,need_weights=False)[0];z=self.latent_block(z)
   tokens=self.norm(tokens+self.write(tokens,z,z,need_weights=False)[0])
  elif self.cfg.variant=='differential':tokens=self.diff(tokens)
  elif self.cfg.variant=='candidate_graph':tokens=self.rel2(self.rel(tokens,edges),edges)
  else:tokens=self.context(tokens)
  time=torch.arange(1,T+1,device=nom.device,dtype=nom.dtype)*.2
  tt=(time-.2).clamp_min(0).square()/3.0 if self.cfg.causal_first_step else time.square()/3.2
  correction=self.cfg.residual_scale*torch.tanh(self.motion(tokens)).reshape(B,C,E,T,2)*tt[None,None,None,:,None]
  xy=nom+correction;idx=torch.triu_indices(E-1,E-1,1,device=nom.device);i,j=idx
  pred_d=torch.linalg.vector_norm(xy[:,:,i]-xy[:,:,j],dim=-1).amin(-1)
  nom_d=torch.linalg.vector_norm(nom[:,:,i]-nom[:,:,j],dim=-1).amin(-1)
  z=tokens.reshape(B,C,E,d)
  extra=torch.cat(((pos[:,i]-pos[:,j])[:,None].expand(-1,C,-1,-1)/20,nom_d[...,None]/10,
                   candidate[:,:,i,:2]-candidate[:,:,j,:2],torch.ones_like(nom_d[...,None])),-1)
  raw=self.pair(torch.cat((z[:,:,i]+z[:,:,j],(z[:,:,i]-z[:,:,j]).abs(),extra),-1))
  mu=(pred_d+2*raw[...,0].tanh()).clamp_min(.01);sigma=(F.softplus(raw[...,1])+.03).clamp_max(4)
  gap=7-mu;q=gap/sigma;cdf=.5*(1+torch.erf(q/math.sqrt(2)));pdf=torch.exp(-.5*q*q)/math.sqrt(2*math.pi)
  raw_risk=(gap.square()+sigma.square())*cdf+gap*sigma*pdf
  lower=-mu/sigma;lower_cdf=.5*(1+torch.erf(lower/math.sqrt(2)));lower_pdf=torch.exp(-.5*lower.square())/math.sqrt(2*math.pi)
  # E[(r-max(0,D))_+^2], with nonnegative physical distance and r=7.
  risk=(raw_risk+(49-gap.square()-sigma.square())*lower_cdf-sigma*(2*gap+mu)*lower_pdf).clamp(0,49)
  return dict(xy=xy,mu=mu,sigma=sigma,risk=risk,det_risk=(7-mu).clamp_min(0).square(),nominal_mu=nom_d)
def targets(b):
 real=b['real_positions'];n=real.shape[-3]-1;i,j=torch.triu_indices(n,n,1,device=real.device)
 return torch.linalg.vector_norm(real[:,:,:,i]-real[:,:,:,j],dim=-1).amin(-1)
def loss_fn(out,b,mode='full'):
 real=b['real_positions'];d=targets(b);mean=real.mean(2)
 motion=F.smooth_l1_loss(out['xy'],mean)
 w=1+3*(d<12).float();sigma=out['sigma'][:,:,None];mu=out['mu'][:,:,None]
 nll=(((d-mu).square()/(2*sigma.square())+torch.log(sigma))*w).mean()
 trisk=(7-d).clamp_min(0).square().mean(2);wr=1+2*(d.mean(2)<9).float()
 risk=(F.smooth_l1_loss(out['risk'],trisk,reduction='none')*wr).mean()
 actual=b['unary']+32*trisk.sum(-1);pred=b['unary']+32*out['risk'].sum(-1)
 rank=-(torch.softmax(-actual/8,dim=-1)*torch.log_softmax(-pred/8,dim=-1)).sum(-1).mean()
 return dict(total=motion if mode=='motion_only' else motion+.08*nll+.25*risk+.05*rank,motion=motion,nll=nll,risk=risk,rank=rank)
@torch.no_grad()
def metrics(model,data,ids,batch_size=12):
 model.eval();tot={};count=0;near_sum=0.;near_count=0
 for start in range(0,len(ids),batch_size):
  ix=ids[start:start+batch_size];b={k:v[ix] for k,v in data.items()};o=model(b);d=targets(b)
  true_risk=(7-d).clamp_min(0).square().mean(2);truth=b['unary']+32*true_risk.sum(-1)
  estimate=b['unary']+32*o['risk'].sum(-1);choice=estimate.argmin(-1);nom=b['nominal_cost'].argmin(-1);rows=torch.arange(len(ix))
  vals=dict(motion_mae=(o['xy']-b['real_positions'].mean(2)).abs().mean(),
   min_distance_mae=(o['mu']-d.mean(2)).abs().mean(),risk_mae=(o['risk']-true_risk).abs().mean(),
   regret=(truth[rows,choice]-truth.min(-1).values).mean(),nominal_regret=(truth[rows,nom]-truth.min(-1).values).mean(),
   changed=(choice!=0).float().mean(),geometry_top1_accuracy=(choice==truth.argmin(-1)).float().mean())
  for k,v in vals.items():tot[k]=tot.get(k,0)+float(v)*len(ix)
  near=d.mean(2)<12;near_sum+=float((o['mu']-d.mean(2)).abs()[near].sum());near_count+=int(near.sum());count+=len(ix)
 return {k:v/count for k,v in tot.items()}|{'near_distance_mae':near_sum/max(near_count,1)}
