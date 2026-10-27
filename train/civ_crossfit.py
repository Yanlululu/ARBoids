"""Out-of-fold calibration of the same unified relational CIV decision."""
import argparse
import copy
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import civ_rl as c
from civ_generalization import ARCHITECTURE,RelationalValueNetwork,loss_function
from civ_rl_experiment import digest,tensor_features
from civ_replay_search import score


DRIVER_SHA256=digest(__file__)


def fit(args):
    torch.set_num_threads(1);torch.set_grad_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    roots=[];sources=[];reference=None;start=time.monotonic()
    for number,path in enumerate(args.data):
        data=torch.load(path,map_location='cpu',weights_only=False)
        if not data['completed']:raise ValueError('Completed replay is required.')
        if reference is not None and any(data[k]!=reference[k] for k in ('source_sha256','execution_identity')):
            raise ValueError('A source pipeline must retain its own counterfactual labels.')
        reference=data;sources.append(dict(path=str(Path(path).resolve()),sha256=digest(path)))
        allowed={tuple(t['scene']) for t in data['tasks'][:args.old_scene_limit]} if args.old_scene_limit and number<len(args.data)-1 else None
        roots.extend(dict(root,group=Path(path).stem+':ag'+str(root['scene'][2])) for root in data['roots']
                     if allowed is None or tuple(root['scene']) in allowed)
    scenes=[tuple(root['scene']) for root in roots]
    folds=np.asarray([int(hashlib.sha256(str(scene).encode()).hexdigest()[:8],16)%args.members for scene in scenes])
    if len(set(folds))!=args.members:raise ValueError('Every fold must have held-out scenes.')
    config=c.CIVConfig(**reference['config']);features=tensor_features(roots,args.device)
    count=max(len(r['targets']) for r in roots);options=len(roots[0]['targets'][0])
    targets=np.zeros((len(roots),count,options),np.float32);future_mask=np.zeros((len(roots),count),bool)
    for i,root in enumerate(roots):
        targets[i,:len(root['targets'])]=root['targets'].clip(min=0);future_mask[i,:len(root['targets'])]=True
    targets=torch.as_tensor(targets,device=args.device);future_mask=torch.as_tensor(future_mask,device=args.device)
    benefit=np.asarray([np.any((r['targets']-r['targets'][:,:1])[:,r['features']['valid']]>0) for r in roots])
    networks=[];optimizers=[];randoms=[];probabilities=[];training_counts=[]
    for member in range(args.members):
        torch.manual_seed(args.seed+member);network=RelationalValueNetwork(args.width,args.dropout).to(args.device)
        networks.append(network);optimizers.append(torch.optim.AdamW(network.parameters(),lr=args.learning_rate,weight_decay=1e-3))
        randoms.append(np.random.default_rng(args.seed+member))
        train=folds!=member;positive=train&benefit
        probability=train/train.sum()
        if positive.any():probability=.75*probability+.25*positive/positive.sum()
        probabilities.append(probability);training_counts.append(int(train.sum()))
    best=None;history=[];selected=None;fold_tensor=torch.as_tensor(folds,device=args.device)
    for update in range(1,args.updates+1):
        losses=[]
        for member,(network,optimizer,rng,probability) in enumerate(zip(networks,optimizers,randoms,probabilities)):
            network.train();indices=rng.choice(len(roots),args.batch_size,replace=True,p=probability)
            ix=torch.as_tensor(indices,device=args.device)
            importance=torch.as_tensor(1/(training_counts[member]*probability[indices]),device=args.device,dtype=torch.float32)
            batch={key:value[ix] for key,value in features.items()}
            loss=loss_function(network(batch),targets[ix],future_mask[ix],batch['valid'],importance,args.kind,args.gain_weight,args.rank_weight)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()))
        if update%args.checkpoint_every==0 or update==args.updates:
            gain=torch.zeros_like(features['valid'],dtype=torch.float32)
            with torch.no_grad():
                for member,network in enumerate(networks):
                    network.eval();prediction,_=c.ensemble_values([network],features,args.kind)
                    gain[fold_tensor==member]=prediction[fold_tensor==member]
            prediction=(gain,torch.zeros_like(gain))
            for margin in (-.02,0.,.02,.05,.1,.2):
                result=score(networks,features,roots,args.kind,margin,0.,prediction)
                lower=(result['rescues']-result['harms']-.5*np.sqrt(result['rescues']+result['harms']))/result['total']
                result.update(update=update,lower_gain=float(lower),loss=float(np.mean(losses)),seconds=time.monotonic()-start)
                history.append(result);rank=(lower,result['gain'],-result['harms'],-result['changed_roots'],-update,margin)
                if best is None or rank>best:
                    best=rank;selected=copy.deepcopy(result);source=Path(reference['source_path'])
                    record=dict(algorithm='CIV-RL-unified-v1',value_architecture=ARCHITECTURE,source_path=str(source),
                        source_relative_path=str(source.relative_to(Path(__file__).resolve().parents[1])),source_sha256=reference['source_sha256'],
                        config=asdict(replace(config,acceptance_margin=margin,uncertainty_penalty=0.)),execution_identity=config.identity(),
                        kind=args.kind,width=args.width,dropout=args.dropout,normalize=False,
                        networks=[{k:v.detach().cpu() for k,v in net.state_dict().items()} for net in networks],
                        fit=dict(completed=True,search_completed=False,training_roots=len(roots),validation_roots=len(roots),
                            calibration='out-of-fold-scene-hash',heldout_prediction_used_training_labels=False,
                            member_training_roots=training_counts,fold_by_scene=[dict(scene=scene,fold=int(fold)) for scene,fold in zip(scenes,folds)],
                            training_fleets=sorted({scene[1] for scene in scenes}),data_sources=sources,
                            member_seeds=[args.seed+i for i in range(args.members)],updates=update,planned_updates=args.updates,
                            parameter_count=sum(v.numel() for v in networks[0].parameters()),selected=selected,
                            gain_weight=args.gain_weight if args.kind=='conditional' else 0.,rank_weight=args.rank_weight if args.kind=='conditional' else 0.,
                            objective='conditional stratified BCE and paired mean-gain/ranking' if args.kind=='conditional' else 'ordinary terminal BCE',
                            batch_size=args.batch_size,learning_rate=args.learning_rate,old_scene_limit=args.old_scene_limit,
                            runtime=c.runtime_identity(),driver_sha256=DRIVER_SHA256))
                    torch.save(record,args.output)
            print(json.dumps(dict(update=update,selected=selected,seconds=time.monotonic()-start)),flush=True)
    record=torch.load(args.output,map_location='cpu',weights_only=False)
    record['fit'].update(search_completed=True,completed_updates=args.updates,validation_history=history)
    torch.save(record,args.output);print(json.dumps(dict(completed=args.output,selected=selected)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',nargs='+',required=True);p.add_argument('--kind',choices=['conditional','ordinary'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--updates',type=int,default=3000);p.add_argument('--members',type=int,default=3);p.add_argument('--width',type=int,default=48)
    p.add_argument('--dropout',type=float,default=.08);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--gain-weight',type=float,default=.5);p.add_argument('--rank-weight',type=float,default=.1)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--old-scene-limit',type=int,default=0);fit(p.parse_args())


if __name__=='__main__':main()
