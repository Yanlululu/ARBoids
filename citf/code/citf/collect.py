"""Stopped short-motion data with known time context and physical layout identity."""
from __future__ import annotations
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .bridge import Source,World,Controller,factorize,inputs
from .model import Config,Predictor
from .contracts import Protocol,SCHEMA,FEATURES,CONTEXT,file_hash,validate_dataset,LayoutRegistry,canonical_hash,validate_predictor_protocol,tree_hash,RUNTIME_REVISION
from .search import nominal_beam,initialize_bank
from .opponents import load_opponent,APFOpponent


def load_behavior(path,source,world,device):
    if not path:return None
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['config'].get('revision')!=Config().revision:raise ValueError('Behavior checkpoint revision mismatch')
    if ck['protocol']['source_sha256']!=source.sha256 or ck['protocol']['environment_sha256']!=world.environment_sha256:
        raise ValueError('Behavior model source/environment mismatch')
    if ck['protocol'].get('source_implementation_sha256')!=source.implementation_sha256:raise ValueError('Behavior source network implementation mismatch')
    transfer=validate_predictor_protocol(ck['protocol']['physical_protocol'],world.protocol,allow_task_transfer=True)
    model=Predictor(Config(**ck['config']));model.load_state_dict(ck['state'])
    model.protocol_transfer=transfer
    return model.to(device).eval()


def sample_root(world,source,plans,future_seeds,root_metadata,opponent=None,stop_file=None):
    from .contracts import COLLECTION_SCHEMA,BRANCH_EVENT_FIELDS
    opponent=opponent or APFOpponent()
    f=factorize(world,source);b,g,R=inputs(world,f,plans)
    C,N=g.shape[:2];T=world.protocol.prediction_steps;E=N+1
    xy=torch.zeros(C,len(future_seeds),E,T,2);valid=torch.zeros(C,len(future_seeds),T,dtype=torch.bool)
    attacker_actions=torch.zeros(C,len(future_seeds),T,2)
    shape=(C,len(future_seeds))
    records={'branch_stop_step':torch.zeros(shape,dtype=torch.int64),
             'branch_outcome':torch.zeros(shape,dtype=torch.int64),
             'branch_terminated':torch.zeros(shape,dtype=torch.bool),
             'branch_prediction_truncated':torch.zeros(shape,dtype=torch.bool),
             'branch_stop_time':torch.zeros(shape,dtype=torch.float64),
             'branch_events':torch.zeros((*shape,len(BRANCH_EVENT_FIELDS)),dtype=torch.bool)}
    physics_steps=0
    for c in range(C):
        for r,seed in enumerate(future_seeds):
            if stop_file and Path(stop_file).exists():
                raise InterruptedError('Resource stop between real candidate branches')
            branch=world.clone();branch.rng=np.random.RandomState(int(seed))
            for q in range(T):
                a,rule,_=source.commands(branch.env);force=(750*(g[c]*a+(1-g[c])*rule)+250).numpy()
                action=opponent.act(branch)
                result=branch.step(force,attacker_action=action);physics_steps+=1
                attacker_actions[c,r,q]=torch.as_tensor(branch.env.att_action,dtype=torch.float32)
                xy[c,r,:,q]=torch.tensor(branch.state()[:,:2]@R,dtype=torch.float32);valid[c,r,q]=True
                if branch.done:break
            records['branch_stop_step'][c,r]=q+1
            records['branch_outcome'][c,r]=result['outcome']
            records['branch_terminated'][c,r]=branch.done
            records['branch_prediction_truncated'][c,r]=not branch.done
            records['branch_stop_time'][c,r]=result['duration']
            records['branch_events'][c,r]=torch.tensor([bool(result[k]) for k in BRANCH_EVENT_FIELDS])
    tensors={k:v[0].clone() for k,v in b.items()};tensors['real_positions']=xy;tensors['time_valid']=valid
    tensors['executed_attacker_actions']=attacker_actions  # supervision provenance only; never a model feature
    tensors.update(records)  # observed branch events are never predictor inputs
    ident=world.initial_identity
    meta={**root_metadata,'collection_schema':COLLECTION_SCHEMA,'branch_event_fields':BRANCH_EVENT_FIELDS,
          'future_seeds':list(map(int,future_seeds)),'source_sha256':source.sha256,
          'source_implementation_sha256':source.implementation_sha256,'environment_sha256':world.environment_sha256,
          'physical_protocol':asdict(world.protocol),
          'capture_stage':f't={world.env.Current_T:.1f}s','elapsed_seconds':float(world.env.Current_T),
          'remaining_seconds':max(0.,world.protocol.time_limit-world.env.Current_T),
          'geometry_identity':ident,'geometry_family_id':ident['fingerprint'],
          'sampling_probability':root_metadata.get('sampling_probability',1.),
          'behavior_identity':root_metadata.get('behavior_identity',{'method':'caller-specified'}),
          'opponent_identity':opponent.identity,
          'opponent_execution':'same frozen feedforward policy on each branch state; APF defaults handled by environment',
          'stopping_target':'min over observed future transitions through first actual task termination or horizon; terminal transition included'}
    return tensors,meta,physics_steps


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--source',required=True)
    ap.add_argument('--tasks',required=True);ap.add_argument('--out',required=True);ap.add_argument('--registry',required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--behavior-checkpoint')
    ap.add_argument('--opponent-spec',help='Frozen APF/TorchScript actor specification; no defender input receives future actions')
    ap.add_argument('--behavior-bank',type=int,default=12);ap.add_argument('--behavior-budget',type=int,default=96)
    ap.add_argument('--behavior-rounds',type=int,default=1)
    ap.add_argument('--root-times',type=float,nargs='+',default=[0.,2.,4.,8.,12.,20.,30.,40.,60.])
    ap.add_argument('--candidates',type=int,default=12);ap.add_argument('--futures',type=int,default=3)
    ap.add_argument('--max-roots',type=int,default=0);ap.add_argument('--max-per-layout',type=int,default=4)
    ap.add_argument('--layout-start',type=int,default=0);ap.add_argument('--layout-count',type=int,default=0)
    ap.add_argument('--stop-file',help='Cooperative resource stop; incomplete shards are not committed')
    args=ap.parse_args();torch.set_num_threads(1);bank=json.loads(Path(args.tasks).read_text())
    if bank['role'] not in ('train','validation','test'):raise ValueError('No final closed-loop bank may enter collection')
    p=Protocol(**bank.get('protocol',{}));out=Path(args.out)
    if out.exists():raise ValueError('Refuse dataset overwrite')
    if args.layout_start<0 or args.layout_count<0:raise ValueError('Invalid layout shard')
    if args.candidates<1 or args.futures<1 or args.max_per_layout<1:raise ValueError('Invalid sample counts')
    if any(t<0 or t>=p.time_limit or abs(t/p.control_dt-round(t/p.control_dt))>1e-6 for t in args.root_times):raise ValueError('Root times must be legal control ticks')
    source=Source(args.project,args.source,args.device);registry=LayoutRegistry(args.registry)
    opponent=load_opponent(args.opponent_spec,args.device)
    items=[];roots=[];cost=0;actual=0;t0=time.perf_counter();seen=set();planned=[]
    selected=bank['tasks'][args.layout_start:args.layout_start+args.layout_count if args.layout_count else None]
    for task in selected:
        world=World(args.project,p,task['defenders']);world.reset(task['scene_seed'],task['future_seed'],task['agility'])
        family=registry.register(world.initial_identity,bank['role'],task['layout_id'])
        if task.get('geometry_family_id') not in (None,family):raise ValueError('Task geometry differs from registered content')
        if family in seen:raise ValueError('One physical layout per collection shard; different agility uses the same split family')
        seen.add(family)
        model=load_behavior(args.behavior_checkpoint,source,world,args.device)
        policy=Controller(source,model,args.device,args.behavior_bank,args.behavior_budget,args.behavior_rounds)
        behavior={'method':'CITF' if model is not None else 'fast_physics_CI',
                  'predictor_sha256':file_hash(args.behavior_checkpoint) if args.behavior_checkpoint else None,
                  'bank':args.behavior_bank,'budget':args.behavior_budget,'rounds':args.behavior_rounds,
                  'source_sha256':source.sha256,'protocol_sha256':p.digest(),'opponent':opponent.identity}
        rng=np.random.default_rng(task['scene_seed']+3329)
        times=sorted(rng.choice(args.root_times,min(args.max_per_layout,len(args.root_times)),replace=False))
        ticks={round(x/p.control_dt) for x in times};step=0
        entry={'layout_id':task['layout_id'],'geometry_family_id':family,'requested_ticks':sorted(ticks),'collected_ticks':[]}
        planned.append(entry);behavior_result=None
        while not world.done:
            if args.stop_file and Path(args.stop_file).exists():raise RuntimeError('Resource stop before committing this collection shard')
            if step in ticks:
                f=factorize(world,source);plan,_=nominal_beam(f['unary'].numpy(),f['pair'].numpy(),p.risk_weight,128)
                count=args.candidates;top=max(1,count//2);choices=[x.nonzero()[:,0].numpy() for x in f['valid']]
                initial,_=initialize_bank(plan,f['grid'].numpy(),f['valid'].numpy(),f['prior'].numpy(),top,top)
                candidates=[tuple(x) for x in initial.tolist()];attempts=0
                while len(candidates)<count and attempts<10000:
                    key=tuple(int(rng.choice(x)) for x in choices);attempts+=1
                    if key not in candidates:candidates.append(key)
                if len(candidates)!=count:raise ValueError('Insufficient unique candidates')
                future=list(map(int,rng.integers(1,2**31-1,size=args.futures)))
                rid=canonical_hash({'geometry':family,'step':step,'behavior':behavior,'future_seeds':future,'agility':task['agility']})
                item,meta,steps=sample_root(world,source,np.asarray(candidates),future,
                    dict(root_id=rid,layout_id=task['layout_id'],split=bank['role'],scene_seed=task['scene_seed'],root_step=step,
                         candidate_indices=[list(x) for x in candidates],behavior_identity=behavior,
                         sampling_probability=min(args.max_per_layout,len(args.root_times))/len(args.root_times)),opponent=opponent,stop_file=args.stop_file)
                items.append(item);roots.append(meta);cost+=steps;entry['collected_ticks'].append(step)
                if args.max_roots and len(roots)>=args.max_roots:break
            force,_=policy.act(world);behavior_result=world.step(force,attacker_action=opponent.act(world));actual+=1;step+=1
        entry['unreached_ticks']=sorted(ticks-set(entry['collected_ticks']))
        entry['reason']='root budget reached' if args.max_roots and len(roots)>=args.max_roots else 'actual task termination'
        entry['task_terminated']=world.done
        entry['actual_terminal_event']=behavior_result if world.done else None
        entry['behavior_stop_time']=float(world.env.Current_T)
        if args.max_roots and len(roots)>=args.max_roots:break
    if not items:raise ValueError('No valid root collected')
    try:tensors={k:torch.stack([x[k] for x in items]) for k in items[0]}
    except RuntimeError as exc:raise ValueError('One fleet size and one candidate count per shard') from exc
    from .contracts import COLLECTION_SCHEMA,BRANCH_EVENT_FIELDS
    blob=dict(schema=SCHEMA,collection_schema=COLLECTION_SCHEMA,branch_event_fields=BRANCH_EVENT_FIELDS,
       history_features=FEATURES,context_features=CONTEXT,tensors=tensors,roots=roots,protocol=p.__dict__,
       provenance=dict(tasks_sha256=file_hash(args.tasks),registry_sha256=file_hash(args.registry),source_sha256=source.sha256,
           environment_sha256=world.environment_sha256,branch_transitions=cost,behavior_transitions=actual,
           planned_roots=planned,opponent_identity=opponent.identity,collection_revision=RUNTIME_REVISION,collection_code_sha256=tree_hash(Path(__file__).parent),seconds=time.perf_counter()-t0,arguments=vars(args)))
    report=validate_dataset(blob);out.parent.mkdir(parents=True,exist_ok=True);torch.save(blob,out)
    out.with_suffix('.json').write_text(json.dumps(report|blob['provenance'],ensure_ascii=False,indent=2));print(report)
if __name__=='__main__':main()
