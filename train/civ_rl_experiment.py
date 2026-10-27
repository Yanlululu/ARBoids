"""Fresh paired-data pilot for the unified CIV-RL deployment interface."""
import argparse
import copy
from dataclasses import asdict,replace
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import civ_rl as c


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_tasks(seed,count,n=3,agilities=(2.,2.5)):
    return [dict(scene=(seed+i,n,float(agilities[i%len(agilities)])),future=seed+1000000+i) for i in range(count)]


def collect_shard(arguments):
    source_path,settings,tasks,future_count,device,include_factual_future=arguments
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    config=c.CIVConfig(**settings);controller=c.CIVController(c.load_source(source_path,device),config)
    rows=c.run_episodes(controller,tasks,collect_roots=True)
    roots=[]
    for row in rows:
        if row['root'] is None:continue
        root=row.pop('root');root['factual']=row
        packet=c.d.mechanism_tensor(c.d.joint_packet(root['environment'],root['observation']),device)
        _,_,features,_=c.fixed_inference(controller.decide,packet,config.inference_width)
        root['features']={key:value[0].cpu().numpy() for key,value in features.items()}
        roots.append(root)
    branch_tasks=[];initial_jobs=[];mapping=[]
    for r,root in enumerate(roots):
        feature=root['features'];valid=np.flatnonzero(feature['valid'])
        root['targets']=np.full((future_count,len(feature['valid'])),-1.,dtype=np.float32)
        root['terminal_outcomes']=np.full_like(root['targets'],-1,dtype=np.int64)
        root['futures']=[int(root['scene'][0]+2000000+future*10000) for future in range(future_count)]
        if include_factual_future:root['futures'][0]=None
        for future,seed in enumerate(root['futures']):
            for option in valid:
                branch_tasks.append(dict(scene=root['scene'],future=seed,program=dict(choice=int(option),
                    gates=feature['gates'][option],duration=int(feature['durations'][option]))))
                initial_jobs.append(root);mapping.append((r,future,int(option)))
    results=c.run_episodes(controller,branch_tasks,initial_jobs=initial_jobs,progress=True)
    for (r,future,option),row in zip(mapping,results):
        roots[r]['targets'][future,option]=row['success']
        roots[r]['terminal_outcomes'][future,option]=row['outcome']
    return dict(roots=roots,factual_rows=rows,simulation_steps=sum(row['simulation_steps'] for row in rows+results),
        branch_count=len(results),runtime=c.runtime_identity())


def collect(args):
    path=Path(args.output)
    if path.exists():raise ValueError('A completed or interrupted dataset must not be silently replaced.')
    source=Path(args.source).resolve();settings=asdict(c.CIVConfig(generator=args.generator))
    tasks=make_tasks(args.seed,args.roots,args.vessels,args.agilities)
    chunks=[tasks[i::args.workers] for i in range(args.workers)]
    work=[(str(source),settings,chunk,args.futures,args.device,args.include_factual_future) for chunk in chunks if chunk]
    if args.workers==1:parts=[collect_shard(work[0])]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:parts=list(pool.map(collect_shard,work))
    if any(part['runtime']!=parts[0]['runtime'] for part in parts):raise RuntimeError('Collector runtime mismatch.')
    roots=sorted([root for part in parts for root in part['roots']],key=lambda root:tuple(root['scene']))
    validation_scenes={tuple(task['scene']) for task in tasks[-args.validation:]}
    for root in roots:root['split']='validation' if tuple(root['scene']) in validation_scenes else 'train'
    record=dict(algorithm='CIV-RL-unified-v1',completed=True,execution_identity=c.CIVConfig(**settings).identity(),
        source_path=str(source),source_sha256=digest(source),config=settings,runtime=parts[0]['runtime'],tasks=tasks,
        roots=roots,factual_rows=[row for part in parts for row in part['factual_rows']],
        include_factual_future=args.include_factual_future,driver_sha256=digest(__file__),
        simulation_steps=sum(part['simulation_steps'] for part in parts),branch_count=sum(part['branch_count'] for part in parts))
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(record,path)
    for split in ('train','validation'):
        selected=[root for root in roots if root['split']==split]
        targets=np.concatenate([root['targets'] for root in selected]);valid=np.concatenate(
            [np.broadcast_to(root['features']['valid'],root['targets'].shape) for root in selected])
        baseline=targets[:,:1]
        print(json.dumps(dict(split=split,roots=len(selected),factual_success=float(baseline.mean()),
            rescue_pairs=int(((baseline==0)&(targets==1)&valid).sum()),
            harm_pairs=int(((baseline==1)&(targets==0)&valid).sum()))),flush=True)
    print(json.dumps(dict(completed=str(path),branches=record['branch_count'],simulation_steps=record['simulation_steps'])),flush=True)


def tensor_features(roots,device):
    return {key:torch.as_tensor(np.stack([root['features'][key] for root in roots]),device=device)
        for key in ('node','edge','global_features','valid')}


def fit_objective(logits,targets,valid,kind,gain_weight=0.):
    if kind=='ordinary':
        if gain_weight:raise ValueError('The ordinary control does not use paired-gain supervision.')
        return c.ordinary_loss(logits,targets,valid)
    loss=c.conditional_loss(logits,targets,valid)
    if gain_weight:
        p0=logits[:,0:1,0].sigmoid()
        gain=(1.-p0)*logits[...,1].sigmoid()-p0*logits[...,2].sigmoid()
        mask=valid.clone();mask[:,0]=False
        paired=((gain-(targets-targets[:,:1])).square()*mask).sum()/mask.sum().clamp_min(1)
        loss=loss+gain_weight*paired
    return loss


@torch.no_grad()
def validation_choices(networks,features,roots,kind,margin,penalty):
    gain,uncertainty=c.ensemble_values(networks,features,kind)
    selected,_=c.choose_options(gain,uncertainty,features['valid'],margin,penalty)
    values=[];rescues=harms=changes=0
    for root,choice in zip(roots,selected.cpu().tolist()):
        y0=root['targets'][:,0];yz=root['targets'][:,choice]
        values.extend(yz-y0);rescues+=int(((y0==0)&(yz==1)).sum());harms+=int(((y0==1)&(yz==0)).sum())
        changes+=int(choice>0)
    return dict(gain=float(np.mean(values)),rescues=rescues,harms=harms,changed_roots=changes,
        margin=margin,choices=selected.cpu().tolist())


def fit(args):
    data=torch.load(args.data,map_location='cpu',weights_only=False)
    if not data['completed'] or data['algorithm']!='CIV-RL-unified-v1':raise ValueError('Fresh completed unified data required.')
    config=c.CIVConfig(**data['config'])
    if config.identity()!=data['execution_identity']:raise ValueError('Dataset execution semantics mismatch.')
    if digest(data['source_path'])!=data['source_sha256']:raise ValueError('Dataset source changed.')
    training=[root for root in data['roots'] if root['split']=='train']
    validation=[root for root in data['roots'] if root['split']=='validation']
    if set(tuple(root['scene']) for root in training)&set(tuple(root['scene']) for root in validation):
        raise ValueError('Whole initial scenes must remain on one side of validation.')
    for root in data['roots']:
        valid=root['features']['valid']
        if not np.isin(root['targets'][:,valid],(0.,1.)).all():raise ValueError('Incomplete closed-loop labels.')
    torch.set_num_threads(1);torch.set_grad_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    train_features=tensor_features(training,args.device);val_features=tensor_features(validation,args.device)
    targets=torch.as_tensor(np.stack([root['targets'] for root in training]),device=args.device)
    networks=[];optimizers=[];randoms=[]
    for member in range(args.members):
        torch.manual_seed(args.seed+member)
        net=c.CIVValueNetwork().to(args.device);networks.append(net)
        optimizers.append(torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4))
        randoms.append(np.random.default_rng(args.seed+member))
    if args.gain_weight<0:raise ValueError('Paired-gain supervision needs a nonnegative weight.')
    history=[];best=None;best_state=None
    checkpoints=sorted(set((max(1,args.updates//4),max(1,args.updates//2),args.updates)))
    for update in range(1,args.updates+1):
        losses=[]
        for net,opt,rng in zip(networks,optimizers,randoms):
            index=torch.as_tensor(rng.integers(0,len(training),min(32,len(training))),device=args.device)
            future=torch.as_tensor(rng.integers(0,targets.shape[1],len(index)),device=args.device)
            features={key:value[index] for key,value in train_features.items()}
            target=targets[index,future].clamp_min(0.)
            loss=fit_objective(net(features),target,features['valid'],args.kind,args.gain_weight)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            losses.append(float(loss.detach()))
        if update in checkpoints:
            for margin in (0.,.05,.1,.2):
                score=validation_choices(networks,val_features,validation,args.kind,margin,config.uncertainty_penalty)
                score.update(update=update,loss=float(np.mean(losses)))
                history.append(score)
                # Identical selection rule for conditional and ordinary value.
                rank=(score['gain'],-score['harms'],-score['changed_roots'],-update,margin)
                if best is None or rank>best:
                    best=rank;best_score=score.copy();best_state=[copy.deepcopy(net.state_dict()) for net in networks]
            print(json.dumps(dict(update=update,loss=float(np.mean(losses)),best=best_score)),flush=True)
    for net,state in zip(networks,best_state):net.load_state_dict(state);net.eval()
    config=replace(config,acceptance_margin=best_score['margin'])
    fit_record=dict(completed=True,training_roots=len(training),validation_roots=len(validation),
        training_fleets=sorted({int(root['scene'][1]) for root in training}),
        member_seeds=[args.seed+i for i in range(args.members)],updates=args.updates,selected=best_score,
        parameter_count=sum(p.numel() for p in networks[0].parameters()),data_path=str(Path(args.data).resolve()),
        data_sha256=digest(args.data),runtime=c.runtime_identity(),validation_history=history,
        paired_gain_weight=args.gain_weight,learning_objective=(
            'BCE(p0,Y0)+(1-Y0)BCE(r,Yz)+Y0*BCE(h,1-Yz)+lambda*MSE(Delta,Yz-Y0)'
            if args.kind=='conditional' else 'Mean of three BCE(Q,Yz) heads'))
    c.save_checkpoint(args.output,data['source_path'],config,networks,args.kind,fit_record)
    print(json.dumps(dict(completed=args.output,fit=best_score)),flush=True)


def evaluate(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if args.model:controller=c.load_controller(args.model,args.device)
    else:controller=c.CIVController(c.load_source(args.source,args.device),c.CIVConfig(generator=args.generator))
    tasks=make_tasks(args.seed,args.tasks,args.vessels,args.agilities)
    rows=c.run_episodes(controller,tasks,progress=True)
    record=dict(completed=True,algorithm='CIV-RL-unified-v1',model=args.model,
        policy_sha256=digest(args.model or args.source),execution_identity=controller.config.identity(),
        config=asdict(controller.config),runtime=c.runtime_identity(),tasks=tasks,rows=rows,
        success=sum(row['success'] for row in rows),collision=sum(row['collision'] for row in rows),
        selected=sum(row['selected']>0 for row in rows),simulation_steps=sum(row['simulation_steps'] for row in rows))
    Path(args.output).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({key:record[key] for key in ('completed','success','collision','selected','simulation_steps')}),flush=True)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    collect_parser=sub.add_parser('collect');collect_parser.set_defaults(function=collect)
    collect_parser.add_argument('--source',required=True);collect_parser.add_argument('--generator',choices=('joint','local'),required=True)
    collect_parser.add_argument('--roots',type=int,default=96);collect_parser.add_argument('--validation',type=int,default=32)
    collect_parser.add_argument('--futures',type=int,default=2);collect_parser.add_argument('--workers',type=int,default=4)
    collect_parser.add_argument('--include-factual-future',action='store_true')
    collect_parser.add_argument('--vessels',type=int,default=6)
    fit_parser=sub.add_parser('fit');fit_parser.set_defaults(function=fit)
    fit_parser.add_argument('--data',required=True);fit_parser.add_argument('--kind',choices=('conditional','ordinary'),required=True)
    fit_parser.add_argument('--updates',type=int,default=1200);fit_parser.add_argument('--members',type=int,default=3)
    fit_parser.add_argument('--gain-weight',type=float,default=0.)
    evaluation=sub.add_parser('evaluate');evaluation.set_defaults(function=evaluate)
    evaluation.add_argument('--source');evaluation.add_argument('--model');evaluation.add_argument('--generator',default='joint',choices=('joint','independent','local'))
    evaluation.add_argument('--tasks',type=int,default=160);evaluation.add_argument('--vessels',type=int,default=6)
    for command in (collect_parser,fit_parser,evaluation):
        command.add_argument('--agilities',type=float,nargs='+',default=[2.,2.5])
        command.add_argument('--seed',type=int,required=True);command.add_argument('--output',required=True)
        command.add_argument('--device',default='cuda:0')
    args=parser.parse_args();args.function(args)


if __name__=='__main__':main()
