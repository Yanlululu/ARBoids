"""Repeated fixed-scene development of the unified conditional value model."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import civ_rl as c
from civ_rl_experiment import digest, tensor_features


def objective(raw, targets, valid, kind, importance, gain_weight, rank_weight):
    weights=valid.to(raw.dtype)*importance[:,None]
    if kind=='ordinary':
        loss=F.binary_cross_entropy_with_logits(raw,targets[...,None].expand_as(raw),reduction='none').mean(-1)
        return (loss*weights).sum()/weights.sum().clamp_min(1.)
    y0=targets[:,:1]
    baseline=(F.binary_cross_entropy_with_logits(raw[:,0,0],y0[:,0],reduction='none')*importance).mean()
    rescue=F.binary_cross_entropy_with_logits(raw[...,1],targets,reduction='none')
    harm=F.binary_cross_entropy_with_logits(raw[...,2],1-targets,reduction='none')
    rm=weights*(1-y0);hm=weights*y0
    loss=baseline+.5*(rescue*rm).sum()/rm.sum().clamp_min(1.)+.5*(harm*hm).sum()/hm.sum().clamp_min(1.)
    p0=raw[:,0:1,0].sigmoid()
    gain=(1-p0)*raw[...,1].sigmoid()-p0*raw[...,2].sigmoid()
    change=targets-y0
    active=weights.clone();active[:,0]=0.
    emphasis=active*(1+3*change.abs())
    loss=loss+gain_weight*((gain-change).square()*emphasis).sum()/emphasis.sum().clamp_min(1.)
    if rank_weight:
        pairs=(targets[:,:,None]>targets[:,None,:])&valid[:,:,None]&valid[:,None,:]
        pair_weights=pairs*importance[:,None,None]
        ranking=F.softplus(-(gain[:,:,None]-gain[:,None,:])/.1)
        loss=loss+rank_weight*(ranking*pair_weights).sum()/pair_weights.sum().clamp_min(1.)
    return loss


@torch.no_grad()
def score(networks,features,roots,kind,margin,penalty,prediction=None):
    gain,uncertainty=c.ensemble_values(networks,features,kind) if prediction is None else prediction
    choices,_=c.choose_options(gain,uncertainty,features['valid'],margin,penalty)
    groups={};rescues=harms=changes=total=success=factual=0
    for root,choice in zip(roots,choices.cpu().tolist()):
        y0=root['targets'][:,0];yz=root['targets'][:,choice]
        r=int(((y0==0)&(yz==1)).sum());h=int(((y0==1)&(yz==0)).sum())
        rescues+=r;harms+=h;changes+=int(choice>0);total+=len(y0)
        success+=int(yz.sum());factual+=int(y0.sum())
        key=root['group'];group=groups.setdefault(key,dict(total=0,rescues=0,harms=0))
        group['total']+=len(y0);group['rescues']+=r;group['harms']+=h
    return dict(gain=(rescues-harms)/total,rescues=rescues,harms=harms,changed_roots=changes,
        success=success,factual_success=factual,total=total,groups=groups,margin=margin,penalty=penalty)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=['conditional','ordinary'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--updates',type=int,default=4000)
    p.add_argument('--members',type=int,default=3);p.add_argument('--width',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',default='cuda:0')
    p.add_argument('--gain-weight',type=float,default=2.);p.add_argument('--rank-weight',type=float,default=1.)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--reuse-validation',action='store_true');p.add_argument('--resume')
    p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--scene-limit',type=int,default=0)
    args=p.parse_args();start=time.monotonic();torch.set_num_threads(1);torch.set_grad_enabled(True)
    torch.backends.cuda.matmul.allow_tf32=False
    training=[];validation=[];manifests=[];reference=None
    for name in args.data:
        data=torch.load(name,map_location='cpu',weights_only=False)
        if not data['completed']:raise ValueError('Completed paired data required.')
        if reference is not None and any(data[k]!=reference[k] for k in ('source_sha256','execution_identity')):
            raise ValueError('A replay bank must use one source and one execution definition.')
        reference=data
        allowed={tuple(t['scene']) for t in data['tasks'][:args.scene_limit]} if args.scene_limit else None
        for root in data['roots']:
            if allowed is not None and tuple(root['scene']) not in allowed:continue
            root=dict(root,group=Path(name).stem+':ag'+str(root['scene'][2]))
            if root['split']=='train' or args.reuse_validation:training.append(root)
            if root['split']=='validation':validation.append(root)
        manifests.append(dict(path=str(Path(name).resolve()),sha256=digest(name)))
    if args.reuse_validation:validation=training
    config=c.CIVConfig(**reference['config'])
    tf=tensor_features(training,args.device);vf=tensor_features(validation,args.device)
    targets=torch.as_tensor(np.stack([r['targets'] for r in training]),device=args.device)
    benefit=np.asarray([np.any((r['targets']-r['targets'][:,:1])[:,r['features']['valid']]>0) for r in training])
    probability=np.ones(len(training),dtype=float)/len(training)
    if benefit.any():probability=.5*probability+.5*benefit/benefit.sum()
    networks=[];opts=[];rngs=[]
    restored=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    for member in range(args.members):
        torch.manual_seed(args.seed+member);net=c.CIVValueNetwork(args.width).to(args.device)
        if restored:net.load_state_dict(restored['networks'][member%len(restored['networks'])])
        networks.append(net);opts.append(torch.optim.AdamW(net.parameters(),lr=args.learning_rate,weight_decay=1e-5))
        rngs.append(np.random.default_rng(args.seed+member))
    best=None;history=[]
    for update in range(1,args.updates+1):
        losses=[]
        for net,opt,rng in zip(networks,opts,rngs):
            indices=rng.choice(len(training),args.batch_size,replace=True,p=probability)
            ix=torch.as_tensor(indices,device=args.device)
            future=torch.as_tensor(rng.integers(0,targets.shape[1],len(ix)),device=args.device)
            importance=torch.as_tensor(1/(len(training)*probability[indices]),device=args.device,dtype=torch.float32)
            feature={k:v[ix] for k,v in tf.items()};target=targets[ix,future].clamp_min(0)
            raw=net(feature)
            loss=objective(raw,target,feature['valid'],args.kind,importance,args.gain_weight,args.rank_weight)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            losses.append(float(loss.detach()))
        if update%args.checkpoint_every==0 or update==args.updates:
            with torch.no_grad():prediction=c.ensemble_values(networks,vf,args.kind)
            for penalty in (0.,.1,.5):
                for margin in (-.02,0.,.02,.05,.1,.2):
                    result=score(networks,vf,validation,args.kind,margin,penalty,prediction)
                    result.update(update=update,loss=float(np.mean(losses)),seconds=time.monotonic()-start)
                    history.append(result)
                    rank=(result['gain'],-result['harms'],-result['changed_roots'],-update,margin,penalty)
                    if best is None or rank>best:
                        best=rank;best_score=copy.deepcopy(result)
                        selected=replace(config,acceptance_margin=margin,uncertainty_penalty=penalty)
                        fit=dict(completed=True,development_search=True,selection_data_reused_for_training=args.reuse_validation,
                            training_roots=len(training),validation_roots=len(validation),training_fleets=sorted({r['scene'][1] for r in training}),
                            member_seeds=[args.seed+i for i in range(args.members)],updates=update,planned_updates=args.updates,
                            parameter_count=sum(v.numel() for v in net.parameters()),data_sources=manifests,
                            selected=best_score,objective='stratified conditional BCE + weighted paired-gain MSE + paired ranking' if args.kind=='conditional' else 'ordinary terminal BCE',
                            gain_weight=args.gain_weight if args.kind=='conditional' else 0.,rank_weight=args.rank_weight if args.kind=='conditional' else 0.,
                            validation_history=history.copy(),runtime=c.runtime_identity())
                        c.save_checkpoint(args.output,reference['source_path'],selected,networks,args.kind,fit)
            print(json.dumps(dict(update=update,loss=float(np.mean(losses)),best=best_score,seconds=time.monotonic()-start)),flush=True)
    finished=torch.load(args.output,map_location='cpu',weights_only=False)
    finished['fit'].update(search_completed=True,completed_updates=args.updates,validation_history=history)
    torch.save(finished,args.output)
    print(json.dumps(dict(completed=args.output,updates=args.updates,seconds=time.monotonic()-start,best=best_score)),flush=True)


if __name__=='__main__':main()
