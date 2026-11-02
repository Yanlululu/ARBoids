
"""CI with a short learned feedback predictor instead of repeated source rollouts.
Current fusion only; no future plan variable, no terminal success supervision.
"""
import argparse,json,time,copy
import numpy as np,torch
from ci_core import *
from ci_data import pool,history
from fast_physics import fast_factorize
from ci_models_fast import Predictor,Config
from eval import evaluate

@torch.no_grad()
def input_plans(envs,f,plans):
 b=len(envs);C=plans.shape[1];n=f['grid'].shape[1]
 ff={k:v[:b] if torch.is_tensor(v) else v for k,v in f.items()}
 gates=physical_gates(ff,plans)
 phyix=plans.gather(2,ff['order'].argsort(-1)[:,None].expand(b,C,n))
 paths=ff['paths'][:,None].expand(-1,C,-1,-1,-1,-1).gather(3,phyix[...,None,None,None].expand(-1,-1,-1,1,32,2)).squeeze(3)[:,:,:,1::2]
 histories=[];positions=[];nominal=[];features=[]
 for r,e in enumerate(envs):
  hist,rot=history(e);actual=np.stack([boat.motion_state() for boat in [*e.defender_list,e.attacker]])
  hist[:,-1,2:4]=(actual[:,3:5]@rot)/5
  pos=actual[:,:2]@rot;ap=pos[-1]+np.arange(1,17)[:,None]*.2*(actual[-1,3:5]@rot)
  nominal.append(np.concatenate((paths[r].numpy()@rot,np.broadcast_to(ap[None,None],(C,1,16,2))),1))
  a=ff['source'][r,:,:2].numpy();rule=np.asarray((e.boids_actions-250)/750,np.float32);g=gates[r].numpy()
  cmd=g*a[None]+(1-g)*rule[None]
  feat=np.concatenate((g,np.broadcast_to(a,(C,n,2)),np.broadcast_to(rule,(C,n,2)),cmd),-1)
  features.append(np.concatenate((feat,np.zeros((C,1,8),np.float32)),1));histories.append(hist);positions.append(pos)
 return dict(history=torch.tensor(np.stack(histories)),positions=torch.tensor(np.stack(positions),dtype=torch.float32),
  nominal=torch.tensor(np.stack(nominal),dtype=torch.float32),candidate=torch.tensor(np.stack(features),dtype=torch.float32),
  gates=gates,unary=score_plans(dict(ff,risk=torch.zeros_like(ff['risk']),weight=0.),plans),
  nominal_cost=score_plans(ff,plans))

class FastController:
 def __init__(self,method='fast_nominal',checkpoint=None,alpha=1.,width=24,risk='expected',expand=1):
  self.actor=CIActor();self.width=width;self.method=method;self.alpha=alpha;self.risk=risk;self.expand=expand;self.models=[]
  self.identity=dict(method=method,alpha=alpha,width=width,risk=risk,expanded_neighborhood_rounds=expand,cv=False,runtime_revision='updated-measured-velocity-consistent-v3',
   nominal='public observed state, held current command, same 0.05s WAMV integrator; no predicted-state SAC calls')
  if checkpoint:
   for ck in str(checkpoint).split(','):
    r=torch.load(ck,map_location='cpu',weights_only=False);m=Predictor(Config(**r['config']));m.load_state_dict(r['state']);m.eval();self.models.append(m)
   self.identity.update(checkpoints=str(checkpoint),training=r['protocol'],selected=r['selected'],ensemble_count=len(self.models))
  if method=='fast_neural' and not self.models:raise ValueError('Missing trained model.')
 @torch.no_grad()
 def score(self,b):
  if not self.models:return b['nominal_cost']
  key='risk' if self.risk=='expected' else 'det_risk'
  risk=torch.stack([m(b)[key] for m in self.models]).mean(0)
  cost=b['unary']+32*risk.sum(-1)
  return b['nominal_cost']+self.alpha*(cost-b['nominal_cost'])
 @torch.no_grad()
 def actions(self,envs,observations):
  result=[];decisions=[]
  for off in range(0,len(envs),self.width):
   es=envs[off:off+self.width];obs=observations[off:off+self.width];m=len(es)
   # Fast branch uses actual batch size; all matching fast comparators share it.
   packet=mechanism_stack([joint_packet(e,o) for e,o in zip(es,obs)],'cpu')
   # The fast model was TRAINED from motion_state() (updated public velocity).
   # Original CI's packet.motion is one-substep-old velocity. Do not mix them.
   packet['motion']=torch.tensor(np.stack([[boat.motion_state() for boat in e.defender_list] for e in es]),dtype=torch.float32)
   f=fast_factorize(self.actor,packet);sol=solve(f);plans=pool(f,sol,12)
   ref=physical_gates(f,sol['reference'][:,None])[:,0]
   b=input_plans(es,f,plans);cost=self.score(b);initial_cost=cost[:,0].clone()
   for rnd in range(self.expand):
    ix=cost.argmin(-1);best=plans[torch.arange(m),ix];neighbors=[plans]
    # Whole configuration reevaluated for every single-coordinate candidate.
    for i in range(6):
     pp=best[:,None].expand(-1,9,-1).clone();pp[:,:,i]=torch.arange(9)[None]
     neighbors.append(pp)
    plans=torch.cat(neighbors,1)
    # Duplicate plans do not change minimization; same fixed budget used by baseline.
    b=input_plans(es,f,plans);cost=self.score(b)
   ix=cost.argmin(-1);g=b['gates'][torch.arange(m),ix]
   cmd=g*f['source'][:,:,:2]+(1-g)*packet['node'][...,11:13]
   result.append(torch.cat((cmd,torch.ones_like(cmd[...,:1])),-1).numpy())
   for i in range(m):decisions.append(dict(choice=int(ix[i]),changed=bool((g[i]-ref[i]).abs().max()>1e-7),
    max_gate_change=float((g[i]-ref[i]).abs().max()),predicted_geometric_gain=float(initial_cost[i]-cost[i,ix[i]])))
  return np.concatenate(result),decisions
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['fast_nominal','fast_neural'],required=True);ap.add_argument('--checkpoint')
 ap.add_argument('--alpha',type=float,default=1.);ap.add_argument('--width',type=int,default=24);ap.add_argument('--expand',type=int,default=1)
 ap.add_argument('--bank',default='development');ap.add_argument('--count',type=int,default=48);ap.add_argument('--risk',default='expected')
 ap.add_argument('--output',required=True);ap.add_argument('--collect',action='store_true');a=ap.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() or path.with_suffix('.jsonl').exists():raise ValueError('Refuse overwrite.')
 tasks=json.loads((ROOT/'configs/tasks.json').read_text())[a.bank][:a.count]
 policy=FastController(a.method,a.checkpoint,a.alpha,a.width,a.risk,a.expand)
 result=evaluate(policy,tasks,path,a.width,a.collect);print({k:v for k,v in result.items() if k not in ('tasks','rows','identity')},flush=True)
if __name__=='__main__':main()
