"""Structured conditional-value exploration on the same CI candidate space."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

import civ_rl as c
from civ_rl_experiment import digest,make_tasks,tensor_features
from civ_replay_search import score


def signature(features):
    node=features['node'];edge=features['edge'];n=node.shape[-2]
    mask=~torch.eye(n,device=node.device,dtype=torch.bool)
    edge=edge[:,:,mask]
    def moments(value):return torch.cat((value.mean(-2),value.std(-2,unbiased=False),value.amin(-2),value.amax(-2)),-1)
    centered=node-node.mean(-2,keepdim=True)
    covariance=torch.einsum('bcni,bcnj->bcij',centered,centered)/n
    indices=torch.triu_indices(node.shape[-1],node.shape[-1],1,device=node.device)
    return torch.cat((moments(node),moments(edge),covariance[...,indices[0],indices[1]],features['global_features']),-1)


class ForestMember(torch.nn.Module):
    def __init__(self,models,kind):super().__init__();self.models=models;self.kind=kind
    def forward(self,features):
        x=signature(features).detach().cpu().numpy();b,k,d=x.shape
        flat=x.reshape(-1,d)
        if self.kind=='conditional':
            p0=self.models[0].predict(x[:,0])[:,None].repeat(k,1)
            probabilities=np.stack((p0,self.models[1].predict(flat).reshape(b,k),self.models[2].predict(flat).reshape(b,k)),-1)
        else:probabilities=np.stack([model.predict(flat).reshape(b,k) for model in self.models],-1)
        value=torch.as_tensor(probabilities,device=features['node'].device,dtype=torch.float32).clamp(1e-5,1-1e-5)
        return torch.logit(value)


def load(path,device='cuda:0'):
    record=torch.load(path,map_location='cpu',weights_only=False)
    source=Path(record['source_path'])
    if not source.exists():source=Path(__file__).resolve().parents[1]/record['source_relative_path']
    if digest(source)!=record['source_sha256']:raise ValueError('Source identity changed.')
    return c.CIVController(c.load_source(source,device),c.CIVConfig(**record['config']),
        [ForestMember(record['forests'],record['kind'])],record['kind'])


def fit(args):
    torch.set_num_threads(1);training=[];validation=[];reference=None;sources=[]
    for path in args.data:
        data=torch.load(path,map_location='cpu',weights_only=False)
        if reference is not None and any(data[k]!=reference[k] for k in ('source_sha256','execution_identity')):
            raise ValueError('One candidate mechanism and source are required.')
        reference=data;sources.append(dict(path=str(Path(path).resolve()),sha256=digest(path)))
        for root in data['roots']:
            root=dict(root,group=Path(path).stem+':ag'+str(root['scene'][2]))
            if root['split']=='train' or args.reuse_validation:training.append(root)
            if root['split']=='validation':validation.append(root)
    if args.reuse_validation:validation=training
    tf=tensor_features(training,'cpu');vf=tensor_features(validation,'cpu')
    x=signature(tf).numpy();targets=np.stack([r['targets'] for r in training]);valid=tf['valid'].numpy()
    repeated=np.broadcast_to(x[:,None],(*targets.shape,x.shape[-1]))
    valid_future=np.broadcast_to(valid[:,None],targets.shape)
    baseline=np.broadcast_to(targets[:,:,:1],targets.shape)
    config=c.CIVConfig(**reference['config']);best=None;history=[]
    for leaf in (1,4,12):
        models=[]
        for head in range(3):
            forest=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=leaf,max_features=.5,
                max_depth=24,n_jobs=2,random_state=args.seed+head)
            if args.kind=='ordinary':xx=repeated[valid_future];yy=targets[valid_future]
            elif head==0:xx=x[:,0];yy=targets[:,:,0].mean(1)
            else:
                select=valid_future&(baseline==(0 if head==1 else 1))
                xx=repeated[select];yy=targets[select] if head==1 else 1-targets[select]
            forest.fit(xx,yy);models.append(forest)
        network=ForestMember(models,args.kind)
        with torch.no_grad():prediction=c.ensemble_values([network],vf,args.kind)
        for margin in (-.02,0.,.02,.05,.1,.2):
            result=score([network],vf,validation,args.kind,margin,0.,prediction)
            result.update(leaf=leaf);history.append(result)
            rank=(result['gain'],-result['harms'],-result['changed_roots'],margin)
            if best is None or rank>best:
                best=rank
                record=dict(algorithm='CIV-RL-unified-v1',value_implementation='conditional-extra-trees',
                    source_path=reference['source_path'],source_relative_path=str(Path(reference['source_path']).relative_to(Path(__file__).resolve().parents[1])),
                    source_sha256=reference['source_sha256'],config=asdict(replace(config,acceptance_margin=margin,uncertainty_penalty=0.)),
                    execution_identity=config.identity(),kind=args.kind,forests=models,
                    fit=dict(completed=True,selection_data_reused_for_training=args.reuse_validation,data_sources=sources,
                        training_roots=len(training),validation_roots=len(validation),training_fleets=[6],
                        seed=args.seed,selected=result,tree_count=300,selected_leaf=leaf,
                        tree_nodes=sum(t.tree_.node_count for model in models for t in model.estimators_),
                        runtime=c.runtime_identity(),driver_sha256=digest(__file__),validation_history=history.copy()))
                torch.save(record,args.output)
        print(json.dumps(dict(leaf=leaf,best=best,selected=record['fit']['selected'])),flush=True)
    record['fit'].update(search_completed=True,validation_history=history)
    torch.save(record,args.output)
    print(json.dumps(dict(completed=args.output,selected=record['fit']['selected'])),flush=True)


def evaluate(args):
    torch.set_num_threads(1);controller=load(args.model,args.device)
    tasks=make_tasks(args.seed,args.tasks,args.vessels,args.agilities)
    rows=c.run_episodes(controller,tasks,progress=True)
    record=dict(completed=True,algorithm='CIV-RL-unified-v1',value_implementation='conditional-extra-trees',
        model=args.model,policy_sha256=digest(args.model),execution_identity=controller.config.identity(),
        config=asdict(controller.config),runtime=c.runtime_identity(),driver_sha256=digest(__file__),tasks=tasks,rows=rows,
        success=sum(r['success'] for r in rows),collision=sum(r['collision'] for r in rows),selected=sum(r['selected']>0 for r in rows))
    Path(args.output).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('completed','success','collision','selected')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fit','evaluate'])
    p.add_argument('--data',nargs='+');p.add_argument('--kind',choices=['conditional','ordinary'])
    p.add_argument('--reuse-validation',action='store_true');p.add_argument('--model')
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--tasks',type=int,default=384);p.add_argument('--vessels',type=int,default=6)
    p.add_argument('--agilities',nargs='+',type=float,default=[2.,2.5]);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();(fit if args.command=='fit' else evaluate)(args)


if __name__=='__main__':main()
