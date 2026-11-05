"""Finite full-bank policy improvement from paired complete-task observations.

Only actually queried actions redistribute their collection-prior probability
mass. Unqueried actions receive no invented return labels. All root comparisons
have unit root weight; repeated futures are not treated as independent roots.
"""
import argparse
from dataclasses import asdict
import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from citf.bridge import Source
from citf.contracts import recorded_protocol
from citf.train import atomic_save as save
from experiments.interaction_dynamics import load as load_dynamics
from experiments.interception_fusion import InterceptionFusionController
from experiments.interaction_trajectories import atomic_json
from experiments.joint_macro_policy import MacroConfig,MacroPolicy,candidate_pool,policy_features


def load_examples(directories,planner):
    examples=[];reports=[];families=set();identities=set()
    for name in directories:
        directory=Path(name);report=json.loads((directory/'summary.json').read_text())
        if not report['completed']:raise ValueError('Policy learning uses completed allocated training collections')
        protocol=json.loads((directory/'protocol.json').read_text())
        tasks=json.loads(Path(protocol['arguments']['tasks']).read_text())
        if tasks['role']!='train':raise ValueError('Do not fit on control validation or endpoint layouts')
        reports.append(dict(directory=str(directory),**report))
        for meta in sorted((directory/'roots').glob('*.json')):
            row=json.loads(meta.read_text());identity=(row['task']['task_id'],row['tick'])
            if identity in identities:raise ValueError('Duplicate physical decision root in learning pool')
            identities.add(identity);families.add(row['task']['geometry_family_id'])
            with np.load(meta.with_suffix('.npz')) as raw:
                plans=candidate_pool(raw['queried_plans'][0],raw['all_plans'],raw['queried_plans'][1:])
                queries=[0]+[1+next(i for i,p in enumerate(plans[1:]) if np.array_equal(p,row)) for row in raw['queried_plans'][1:]]
                factor={k:torch.from_numpy(raw[k].copy()) for k in ['grid','prior','lead_seconds','intercept_target']}
                factor['learned']=torch.from_numpy(raw['learned_source'].copy())
                factor['rules']=torch.from_numpy(raw['boids_source'].copy())
                feature,rotation=policy_features(planner,raw['history'],row['elapsed'],factor,plans,recorded_protocol(tasks))
                feature={k:v.cpu() for k,v in feature.items()}
                reward=torch.from_numpy(raw['reward'].copy()).float()
                initial=torch.full((len(plans),),-.25);initial[0]=0
                probability=initial.softmax(0)
                unique=sorted(set(queries))
                mean=torch.stack([reward[[i for i,q in enumerate(queries) if q==index]].mean() for index in unique])
                mass=probability[unique].sum()
                target=probability.clone()
                target[unique]=mass*(probability[unique].log()+mean/.5).softmax(0)
                positions=torch.from_numpy(raw['state'][:,:,1:4,:,:2].copy()).permute(0,1,3,2,4)
                positions=positions@rotation.cpu()
                valid=torch.from_numpy(raw['valid'][:,:,:3].copy())
            examples.append(dict(inputs=feature,queries=torch.tensor(queries),target=target,
                observed_mass=mass,position=positions,valid=valid,root=row['root_id'],family=row['task']['geometry_family_id'],
                returns=reward.mean(-1),initial_probability=probability))
            if len(examples)%32==0:print(json.dumps(dict(prepared_roots=len(examples))),flush=True)
    return examples,dict(collections=reports,roots=len(examples),independent_layouts=len(families),
        complete_branches=sum(r['complete_branches'] for r in reports),
        branch_environment_steps=sum(r['branch_environment_steps'] for r in reports))


def collate(examples,device):
    count=max(len(x['target']) for x in examples)
    batch={}
    for key in examples[0]['inputs']:
        sample=examples[0]['inputs'][key]
        value=torch.zeros((len(examples),count,*sample.shape[1:]),dtype=sample.dtype)
        for i,example in enumerate(examples):value[i,:len(example['target'])]=example['inputs'][key]
        batch[key]=value.to(device)
    target=torch.zeros(len(examples),count)
    for i,example in enumerate(examples):target[i,:len(example['target'])]=example['target']
    labels=dict(target=target.to(device),observed_mass=torch.stack([x['observed_mass'] for x in examples]).to(device),
        queries=torch.stack([x['queries'] for x in examples]).to(device),
        position=torch.stack([x['position'] for x in examples]).to(device),
        valid=torch.stack([x['valid'] for x in examples]).to(device))
    return batch,labels


def objective(model,batch,labels,policy=True):
    output=model(batch,policy=policy);roots=torch.arange(len(labels['queries']),device=output['position'].device)[:,None]
    predicted=output['position'][roots,labels['queries']]
    error=(predicted[:,:,None]-labels['position'])/.25
    mask=labels['valid'][:,:,:,None,:,None]
    physical=(F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')*mask).sum()
    physical=physical/(mask.sum()*predicted.shape[2]*2).clamp_min(1)
    if policy:
        logp=output['logits'].log_softmax(-1)
        safe_logp=torch.where(batch['candidate_valid'],logp,torch.zeros_like(logp))
        target=labels['target'];entropy=torch.where(target>0,target*target.clamp_min(1e-12).log(),torch.zeros_like(target))
        projection=((entropy-target*safe_logp).sum(-1)/labels['observed_mass']).mean()
        loss=projection+.2*physical
    else:
        projection=physical.new_zeros(());loss=physical
    return loss,dict(policy_projection=float(projection.detach()),short_motion=float(physical.detach()))


def main():
    parser=argparse.ArgumentParser()
    for key in ['project','source','teacher','dynamics','out']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--data',nargs='+',required=True)
    parser.add_argument('--seed',type=int,default=9301)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--pretrain',type=int,default=400)
    parser.add_argument('--updates',type=int,default=600)
    parser.add_argument('--stages',type=int,nargs='+',default=[100,300,600])
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--encoder-lr',type=float,default=3e-5)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    source=Source(args.project,args.source,args.device)
    dynamics=load_dynamics(args.dynamics,args.device)
    planner=InterceptionFusionController(source,dynamics,args.device,budget=96,rounds=1)
    examples,data=load_examples(args.data,planner)
    torch.manual_seed(args.seed)
    cfg=MacroConfig(teacher_checkpoint=args.teacher,dynamics_checkpoint=args.dynamics)
    model=MacroPolicy(cfg).to(args.device)
    actor=list(model.actor.parameters());actor_ids={id(x) for x in actor}
    physical=[x for x in model.parameters() if id(x) not in actor_ids]
    optimizer=torch.optim.AdamW([dict(params=physical,lr=args.lr),dict(params=actor,lr=0.)],weight_decay=1e-5)
    rng=np.random.default_rng(args.seed+123);started=time.time()
    protocol=dict(arguments=vars(args),config=asdict(cfg),data=data,
        initial_behavior='exact greedy reference controller; all alternative logits initialize at-0.25',
        response='corrected first3step physical positions explicitly enter policy scoring; bounded correction plus actual short labels',
        score='candidate-wise joint-entity Transformer, independent of other candidate pool members before softmax',
        policy_target='finite KL projection: only actual queried actions redistribute original mass using eight-future mean complete returns/0.5; all other original probabilities retained',
        label_scope='actual complete branch outcomes train policy probabilities only; no return/value/success predictor',
        stages=args.stages,selection='independent256 complete tasks: successes,collisions,capped capture,earlier stage',
        current_scope='fixed-source development; reference teacher contains earlier learning; new-module controls alone do not identify entireCITF',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(protocol,out/'protocol.json')
    def checkpoint(update):
        payload=dict(config=asdict(cfg),state={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),update=update,seed=args.seed,protocol=protocol,
            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
        save(payload,out/f'stage_{update:06d}.pt');save(payload,out/'last.pt')
    with (out/'learning.jsonl').open('w') as log:
        for step in range(1,args.pretrain+args.updates+1):
            policy=step>args.pretrain;update=max(0,step-args.pretrain)
            if step==args.pretrain+1:
                optimizer.param_groups[0]['lr']=args.encoder_lr
                optimizer.param_groups[1]['lr']=args.lr
            indices=rng.integers(len(examples),size=args.batch)
            batch,labels=collate([examples[int(i)] for i in indices],args.device)
            optimizer.zero_grad(set_to_none=True);loss,detail=objective(model,batch,labels,policy)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite joint policy research objective')
            loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            if step%50==0 or step==args.pretrain:
                row=dict(step=step,policy_update=update,phase='policy' if policy else 'physical_warmup',
                    loss=float(loss.detach()),gradient=float(gradient),elapsed_seconds=time.time()-started,**detail)
                log.write(json.dumps(row)+'\n');log.flush();atomic_json(dict(completed=False,**row),out/'status.json')
                print(json.dumps(row),flush=True)
            if step==args.pretrain or (policy and update in args.stages):checkpoint(update)
    report=dict(completed=True,physical_updates=args.pretrain,policy_updates=args.updates,seed=args.seed,
        training_roots=data['roots'],training_layouts=data['independent_layouts'],complete_branches=data['complete_branches'],
        elapsed_seconds=time.time()-started,selection='not yet evaluated in complete closed-loop tasks')
    atomic_json(report,out/'summary.json');atomic_json(report,out/'status.json')


if __name__=='__main__':main()
