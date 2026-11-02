"""Root-grouped training with resume-equivalent RNG/optimizer/selection state."""
from __future__ import annotations
import argparse,json,time,random,hashlib,math
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .model import Config,Predictor
from .losses import objective
from .risk import branch_minima
from .contracts import validate_dataset,SCHEMA,FEATURES,CONTEXT,file_hash,check_split_families,canonical_hash,tree_hash


def atomic_save(value,path):
    path=Path(path);tmp=path.with_suffix('.saving');torch.save(value,tmp);tmp.replace(path)


@torch.no_grad()
def evaluate(model,data,batch_size=8,device='cpu',clearance=7.,weight=32.,loss_kwargs=None):
    model.eval();sums={};N=len(data['history']);groups={'early':{'sum':0.,'n':0},'complete':{'sum':0.,'n':0}}
    for start in range(0,N,batch_size):
        b={k:v[start:start+batch_size].to(device) for k,v in data.items()};o=model(b);l=objective(o,b,clearance=clearance,weight=weight,**(loss_kwargs or {}))
        rows=torch.arange(len(b['history']),device=device)
        choice=l['estimated_cost'].masked_fill(~b['candidate_valid'],torch.inf).argmin(-1)
        truth=l['empirical_cost'].masked_fill(~b['candidate_valid'],torch.inf)
        vals={k:float(l[k]) for k in ('motion','nll','risk','rank')}
        vals['regret']=float((truth[rows,choice]-truth.min(-1).values).mean())
        for k,v in vals.items():sums[k]=sums.get(k,0)+v*len(rows)
        d,pv=branch_minima(b['real_positions'],b['time_valid'],b['entity_valid'])
        err=(o['mean_distance'][:,:,None]-d).abs()
        complete=b['time_valid'].sum(-1)==b['known_time_valid'].sum(-1)[:,None,None]
        for name,test in [('complete',complete),('early',~complete)]:
            mask=test[...,None]&pv[:,None,None]&b['candidate_valid'][:,:,None,None]
            groups[name]['sum']+=float(err[mask].sum());groups[name]['n']+=int(mask.sum())
    return {k:v/N for k,v in sums.items()}|{'distance_mae_'+k:g['sum']/g['n'] if g['n'] else None for k,g in groups.items()}|{'distance_count_'+k:g['n'] for k,g in groups.items()}


def load_shards(paths,allowed):
    blobs=[];root_ids=set()
    for path in paths:
        blob=torch.load(path,map_location='cpu',weights_only=False);validate_dataset(blob)
        for r in blob['roots']:
            if r['split']!=allowed:raise ValueError('Dataset role mismatch')
            if r['root_id'] in root_ids:raise ValueError('Duplicated exact root across shards')
            root_ids.add(r['root_id'])
        blobs.append(blob)
    if not blobs:raise ValueError('Empty data')
    if any(b.get('collection_schema')!=blobs[0].get('collection_schema') or
           set(b['tensors'])!=set(blobs[0]['tensors']) for b in blobs):
        raise ValueError('Mixed collection schemas require explicit separate data versions')
    try:data={k:torch.cat([b['tensors'][k] for b in blobs]) for k in blobs[0]['tensors']}
    except RuntimeError as exc:raise ValueError('Reference loader requires homogeneous fleet/candidate shape; pad only with explicit valid masks') from exc
    return data,[r for b in blobs for r in b['roots']],blobs


def snapshot_steps(requested,total):
    if total<1 or any(x<0 for x in requested):raise ValueError('Invalid snapshot schedule')
    return sorted({0,total,*[int(x) for x in requested if x<=total]})


def parameter_digest(state):
    h=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        h.update(key.encode());value=tensor.detach().cpu().contiguous()
        h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def preserve_snapshot(out,checkpoint):
    """Immutable CURRENT-stage model, not the best-so-far selection."""
    out=Path(out);directory=out/'snapshots';directory.mkdir(exist_ok=True)
    step=int(checkpoint['update']);path=directory/f'update_{step:06d}.pt'
    manifest_path=out/'snapshots.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema':'citf-training-stages-v1','entries':[]}
    existing=next((x for x in manifest['entries'] if x['update']==step),None)
    state_sha=parameter_digest(checkpoint['state'])
    if path.exists():
        record=torch.load(path,map_location='cpu',weights_only=False)
        if record['training_identity']!=checkpoint['training_identity'] or record['update']!=step or parameter_digest(record['state'])!=state_sha:
            raise ValueError('Refuse to overwrite a different stage snapshot')
        if existing and (existing['checkpoint_sha256']!=file_hash(path) or existing['state_sha256']!=state_sha):
            raise ValueError('Immutable stage snapshot changed')
    else:
        if existing:raise ValueError('Registered stage snapshot file is missing')
        atomic_save(checkpoint|{'checkpoint_role':'current-training-stage-not-best-so-far'},path)
    if existing is None:
        manifest['entries'].append({'update':step,'path':str(path.relative_to(out)),
           'checkpoint_sha256':file_hash(path),'state_sha256':state_sha,'training_identity':checkpoint['training_identity'],
           'selection_role':'current-weights-for-fixed-validation-learning-curve'})
        manifest['entries'].sort(key=lambda x:x['update'])
        tmp=manifest_path.with_suffix('.saving');tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(manifest_path)
    return path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',nargs='+',required=True);ap.add_argument('--validation',nargs='+',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--updates',type=int,default=2400)
    ap.add_argument('--batch-roots',type=int,default=6);ap.add_argument('--validate-every',type=int,default=200)
    ap.add_argument('--save-every',type=int,default=100);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--max-updates-this-call',type=int,default=0)
    ap.add_argument('--stop-file',help='Save an exact resumable state at the next update boundary')
    ap.add_argument('--variant',choices=['candidate_graph','transformer','gru','perceiver'],default='candidate_graph')
    ap.add_argument('--width',type=int,default=56)
    ap.add_argument('--mode',choices=['full','predictive_only','motion_only'],default='full');ap.add_argument('--without-fleet-context',action='store_true')
    ap.add_argument('--device',default='cpu');ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--snapshot-updates',type=int,nargs='+',default=[0,400,1200,2400,6000,12000])
    ap.add_argument('--distance-weight',type=float,default=.08)
    ap.add_argument('--risk-loss-weight',type=float,default=.25)
    ap.add_argument('--rank-weight',type=float,default=.05)
    ap.add_argument('--temperature',type=float,default=8.)
    a=ap.parse_args();out=Path(a.out)
    if min(a.updates,a.batch_roots,a.validate_every,a.save_every)<1:raise ValueError('Invalid training counts')
    losses={'distance_weight':a.distance_weight,'risk_weight':a.risk_loss_weight,'rank_weight':a.rank_weight,'temperature':a.temperature}
    if not all(math.isfinite(x) for x in losses.values()) or min(a.distance_weight,a.risk_loss_weight,a.rank_weight)<0 or a.temperature<=0:raise ValueError('Invalid loss contract')
    stages=snapshot_steps(a.snapshot_updates,a.updates)
    train,tm,tb=load_shards(a.train,'train');val,vm,vb=load_shards(a.validation,'validation');check_split_families([('train',tm),('validation',vm)])
    src={r['source_sha256'] for r in tm+vm};env={r['environment_sha256'] for r in tm+vm}
    impl={r['source_implementation_sha256'] for r in tm+vm}
    physical={canonical_hash(b['protocol']) for b in tb+vb}
    if len(src)!=1 or len(env)!=1 or len(impl)!=1 or len(physical)!=1:raise ValueError('One source/environment/physical protocol per training run')
    p=tb[0]['protocol'];torch.set_num_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    cfg=Config(variant=a.variant,width=a.width,fleet_candidate_context=not a.without_fleet_context,history=p['history_steps'],horizon=p['prediction_steps'],dt=p['control_dt'],clearance=p['risk_clearance'])
    model=Predictor(cfg).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.updates,eta_min=min(3e-5,a.lr));rng=np.random.default_rng(a.seed)
    stable_args={k:v for k,v in vars(a).items() if k not in ('resume','max_updates_this_call','out','stop_file')}
    proto=dict(config=asdict(cfg),arguments=stable_args,schema=SCHEMA,history_features=FEATURES,context_features=CONTEXT,
               source_sha256=next(iter(src)),source_implementation_sha256=next(iter(impl)),environment_sha256=next(iter(env)),physical_protocol=p,
               opponent_identities=list({canonical_hash(r.get('opponent_identity',{})):r.get('opponent_identity',{}) for r in tm+vm}.values()),
               data_sha256={str(path):file_hash(path) for path in a.train+a.validation},code_sha256=tree_hash(Path(__file__).parent),
               parameters=sum(p.numel() for p in model.parameters()),loss_contract=losses,stage_updates=stages,selection='validation regret + 5 * normalized risk MSE, zero-update candidate included')
    identity=canonical_hash(proto);best=None;logs=[];start_step=0;start=time.perf_counter()
    if a.resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if ck['training_identity']!=identity:raise ValueError('Resume data/config/code/source identity differs')
        if file_hash(out/'validation.jsonl')!=ck['validation_log_sha256']:raise ValueError('Validation log newer/different from checkpoint; reconcile explicitly')
        if file_hash(out/'best.pt')!=ck['best_checkpoint_sha256']:raise ValueError('Selected checkpoint is inconsistent with resume state')
        model.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_global_rng']);random.setstate(ck['python_rng']);rng.bit_generator.state=ck['sampler_rng']
        if torch.cuda.is_available() and ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
        best=ck['best'];start_step=ck['update']
    else:
        if out.exists():raise ValueError('Refuse overwrite; use --resume with the identical training contract')
        out.mkdir(parents=True);(out/'protocol.json').write_text(json.dumps(proto,ensure_ascii=False,indent=2))
    def validate(step):
        nonlocal best
        met=evaluate(model,val,device=a.device,clearance=p['risk_clearance'],weight=p['risk_weight'],loss_kwargs=losses)
        score=met['regret']+5*met['risk'];row=dict(update=step,score=score,metrics=met)
        with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if best is None or score<best['score']:
            best=row;atomic_save(dict(state=model.state_dict(),config=asdict(cfg),protocol=proto,selected=row),out/'best.pt')
        model.train()
    def save(step):
        record=dict(state=model.state_dict(),config=asdict(cfg),protocol=proto,training_identity=identity,
             optimizer=opt.state_dict(),scheduler=sched.state_dict(),update=step,best=best,torch_rng=torch.get_rng_state(),
             numpy_global_rng=np.random.get_state(),python_rng=random.getstate(),sampler_rng=rng.bit_generator.state,
             cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             validation_log_sha256=file_hash(out/'validation.jsonl'),best_checkpoint_sha256=file_hash(out/'best.pt'))
        atomic_save(record,out/'last.pt')
        if step in stages:preserve_snapshot(out,record)
    if not a.resume:validate(0);save(0)
    elif start_step in stages:preserve_snapshot(out,ck)
    end=min(a.updates,start_step+a.max_updates_this_call) if a.max_updates_this_call else a.updates
    if end<=start_step:raise ValueError('No remaining updates')
    for step in range(start_step+1,end+1):
        if a.stop_file and Path(a.stop_file).exists():
            end=step-1;save(end);break
        ids=rng.integers(0,len(train['history']),a.batch_roots);b={k:v[ids].to(a.device) for k,v in train.items()}
        loss=objective(model(b),b,mode=a.mode,clearance=p['risk_clearance'],weight=p['risk_weight'],**losses)
        if not torch.isfinite(loss['total']):raise FloatingPointError('Nonfinite objective')
        opt.zero_grad(set_to_none=True);loss['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();sched.step()
        if step%100==0:
            with (out/'learning.jsonl').open('a') as f:f.write(json.dumps(dict(update=step,**{k:float(loss[k].detach()) for k in ('total','motion','nll','risk','rank')}))+'\n')
        if step%a.validate_every==0 or step==a.updates:validate(step)
        if step%a.save_every==0 or step%a.validate_every==0 or step==end or step in stages:save(step)
    (out/'status.json').write_text(json.dumps(dict(completed=end==a.updates,update=end,total_updates=a.updates,best=best,saved_stages=[x for x in stages if x<=end],loss_contract=losses,seconds_this_call=time.perf_counter()-start),indent=2))
    print(json.dumps({'update':end,'completed':end==a.updates,'best_update':best['update']}))
if __name__=='__main__':main()
