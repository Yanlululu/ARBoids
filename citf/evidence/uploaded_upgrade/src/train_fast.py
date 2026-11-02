
from pathlib import Path
from dataclasses import asdict
import torch,numpy as np,argparse,json,time,hashlib
from ci_models_fast import *
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=VARIANTS,required=True);ap.add_argument('--seed',type=int,required=True)
 ap.add_argument('--updates',type=int,default=2400);ap.add_argument('--mode',default='full');a=ap.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(a.seed)
 blob=torch.load(ROOT/'datasets/fast_motion_v2.pt',map_location='cpu',weights_only=False,mmap=True);data=blob['tensors']
 train=(data['split']==0).nonzero()[:,0];val=(data['split']==1).nonzero()[:,0]
 cfg=Config(a.variant,WIDTH[a.variant]);m=Predictor(cfg);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.updates,eta_min=3e-5);rng=np.random.default_rng(a.seed)
 out=ROOT/'models'/f'fast_{a.variant}_{a.seed}_{a.mode}';out.mkdir(parents=True,exist_ok=False)
 proto=dict(config=asdict(cfg),seed=a.seed,updates=a.updates,parameters=sum(x.numel() for x in m.parameters()),mode=a.mode,
  model_sha256=hashlib.sha256((ROOT/'src/ci_models_fast.py').read_bytes()).hexdigest(),
  data_protocol='same720scenes/3futures; splitwholelayouts; noCVlabels',
  select_by='validation true geometric candidate regret +5*riskMAE')
 (out/'protocol.json').write_text(json.dumps(proto,indent=2));best=None;start=time.perf_counter()
 def validate(step):
  nonlocal best
  met=metrics(m,data,val);score=met['regret']+5*met['risk_mae'];row=dict(update=step,score=score,metrics=met,seconds=time.perf_counter()-start)
  if best is None or score<best['score']:
   best=row;torch.save(dict(state=m.state_dict(),config=asdict(cfg),protocol=proto,selected=row),out/'best.pt')
  with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(a.variant,a.seed,json.dumps(row),flush=True);m.train()
 validate(0)
 for step in range(1,a.updates+1):
  ids=train[torch.from_numpy(rng.integers(0,len(train),6))]
  b={k:v[ids] for k,v in data.items()};o=m(b);loss=loss_fn(o,b,a.mode)
  if not torch.isfinite(loss['total']):raise FloatingPointError()
  opt.zero_grad(set_to_none=True);loss['total'].backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();sched.step()
  if step%100==0:
   with (out/'learning.jsonl').open('a') as f:f.write(json.dumps({'update':step,**{k:float(v.detach()) for k,v in loss.items()}})+'\n')
  if step%200==0 or step==a.updates:validate(step)
 torch.save(dict(state=m.state_dict(),config=asdict(cfg),protocol=proto),out/'last.pt')
 (out/'completed.json').write_text(json.dumps(dict(completed=True,best=best,seconds=time.perf_counter()-start,protocol=proto),indent=2))
if __name__=='__main__':main()
