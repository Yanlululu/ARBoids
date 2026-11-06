"""Train the all-candidate Transformer on matched, complete feedback outcomes."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from citf.contracts import Protocol, immutable_run_contract, require_same_contract
from citf.evaluate import atomic_json
from citf.train import read_journal
from experiments.fusion_capture_data import save_tensor
from experiments.fusion_capture_train import targets, hurdle_nll
from experiments.fleet_program_policy import FleetConfig, FleetOutcomeTransformer, PROGRAMS, recorded_teacher_identity


def load_data(directory,full_future,cfg,role):
    directory=Path(directory);protocol=json.loads((directory/'protocol.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if protocol['role']!=role or not summary['completed']:raise ValueError('Wrong or incomplete fleet data')
    sampled=FleetConfig(**protocol['config'])
    if (replace(sampled,context=cfg.context,revision=cfg.revision,distance_interval=cfg.distance_interval,
                outcome_readout=cfg.outcome_readout,reselect_seconds=cfg.reselect_seconds,time_prediction=cfg.time_prediction,
                candidate_comparison=cfg.candidate_comparison,candidate_attention=cfg.candidate_attention)!=cfg
            or protocol['programs']!=list(PROGRAMS)):
        raise ValueError('Fleet model configuration differs from the sampled programs')
    full_future=Path(full_future);long_protocol=json.loads((full_future/'protocol.json').read_text())
    long_summary=json.loads((full_future/'summary.json').read_text())
    if (not long_summary['completed'] or long_summary['roots']!=summary['roots']
            or long_protocol['source_data']!=str(directory) or long_protocol['role']!=role):
        raise ValueError('Full-future labels differ from their sampled root bank')
    for key in ('physical_protocol','source_identity'):
        if long_protocol[key]!=protocol[key]:raise ValueError('Full-future label protocol differs: '+key)
    roots=[]
    for path in sorted((directory/'roots').glob('*.pt')):
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if asdict(Protocol(**payload['physical_protocol']))!=asdict(Protocol(**protocol['physical_protocol'])):
            raise ValueError('Mixed physical rules in fleet roots')
        if payload['opponent_identity']!=protocol['opponent_identity']:raise ValueError('Mixed fleet opponents')
        if (payload['batch']['nominal'].shape[-2]!=cfg.horizon
                or payload['batch']['history'].shape[-2]!=cfg.history):
            raise ValueError('Fleet prediction history/horizon differs')
        category=payload['terminal_class'];utility=payload['terminal_utility'];duration=payload['terminal_duration']
        capture=(category==0).float();difference=utility[:,None]-utility[None,:]
        target=targets(payload,cfg)
        remaining=torch.load(full_future/'roots'/path.name,map_location='cpu',weights_only=False)
        if remaining['task']!=payload['task'] or remaining['future_seeds']!=payload['future_seeds']:
            raise ValueError('Full-future root identity differs')
        for key in ('pair','team','radius'):target[key]=remaining[key]
        target.update(outcome_frequency=F.one_hot(category,3).float().mean(1),
            capture_duration=(duration*capture).sum(1)/capture.sum(1).clamp_min(1),capture_count=capture.sum(1),
            utility=utility.mean(1),utility_difference=difference.mean(-1),
            utility_difference_sem2=difference.var(-1,unbiased=True)/utility.shape[1])
        # Full-future distributions use a common replica; motion remains short-range.
        target['xy_samples']=payload['real_positions'];target['valid_samples']=payload['time_valid']
        target['safe_samples']=category!=2
        roots.append(dict(batch=payload['batch'],target=target,root_id=payload['root_id'],
            family=payload['task']['geometry_family_id']))
    if len(roots)!=summary['roots'] or not roots or len({r['root_id'] for r in roots})!=len(roots):
        raise ValueError('Missing or duplicated fleet roots')
    return roots,protocol,summary


def stack(roots,indices,device,stochastic=True,replica_index=0):
    batch={key:torch.stack([roots[i]['batch'][key] for i in indices]).to(device) for key in roots[0]['batch']}
    targets_batch=[]
    for index in indices:
        truth=roots[index]['target'];replica=random.randrange(truth['pair'].shape[1]) if stochastic else replica_index
        value={key:truth[key] for key in ('outcome_frequency','capture_duration','capture_count','utility',
            'utility_difference','utility_difference_sem2')}
        for key in ('pair','team','radius'):value[key]=truth[key][:,replica:replica+1]
        value['xy']=truth['xy_samples'][:,replica];value['xy_valid']=truth['valid_samples'][:,replica]
        value['safe']=truth['safe_samples'][:,replica:replica+1]
        targets_batch.append(value)
    truth={key:torch.stack([t[key] for t in targets_batch]).to(device) for key in targets_batch[0]}
    return batch,truth


def objective(model,batch,truth,output=None):
    if output is None:output=model(batch)
    terminal=-(truth['outcome_frequency']*output['outcome_logits'].log_softmax(-1)).sum(-1).mean()
    predicted_difference=output['utility'][:,:,None]-output['utility'][:,None,:]
    # A continuous paired regression, not a hard sign target for every noisy mean.
    paired=(F.smooth_l1_loss(predicted_difference,truth['utility_difference'],reduction='none')/
            (1+truth['utility_difference_sem2'])).mean()
    pair=hurdle_nll(truth['pair'],output,'pair',5.)
    direct=model.cfg.outcome_readout=='physical_distributions'
    team=hurdle_nll(truth['team'],output,'team',5.,mask=truth['safe'] if direct else None)
    radius=hurdle_nll(truth['radius'],output,'radius',15.)
    mask=truth['xy_valid'][:,:,None,:,None].expand_as(truth['xy'])
    motion=F.smooth_l1_loss(output['xy']/5,truth['xy']/5,reduction='none')[mask].mean()
    captured=truth['capture_count']>0
    timing_error=(output['terminal_capture_time']-truth['capture_duration']).square()
    if direct:
        # Ten seconds is the declared error unit, independent of the old6.4s window.
        timing_error=F.smooth_l1_loss(8*(output['terminal_capture_time']-truth['capture_duration']),
            torch.zeros_like(output['terminal_capture_time']),reduction='none')
    if model.cfg.outcome_readout=='distribution_only':
        timing_error=.5*(timing_error+(output['time_fraction']-truth['capture_duration']).square())
    timing=(timing_error*captured).sum()/captured.sum().clamp_min(1)
    loss=terminal+.2*paired+.05*(pair+.5*team+.2*radius)+.1*motion+.1*timing
    safety=loss.new_zeros(())
    if direct:
        N=batch['positions'].shape[1]-1;_,j=torch.triu_indices(N+1,N+1,1,device=truth['pair'].device)
        minimum=torch.minimum(truth['pair'][...,j<N].amin(-1),truth['radius']/3.)
        safety=hurdle_nll(minimum,output,'safety',5.)
        loss=loss+.2*safety
    chosen=output['utility'].argmax(-1)
    regret=(truth['utility'].max(-1).values-truth['utility'].gather(1,chosen[:,None]).squeeze(1)).mean()
    metrics=dict(loss=loss,terminal_ce=terminal,paired_utility=paired,pair_nll=pair,team_nll=team,radius_nll=radius,
        motion=motion,timing=timing,safety_nll=safety,empirical_regret=regret,
        utility_mae=(output['utility']-truth['utility']).abs().mean(),
        capture_brier=(output['outcome_probabilities'][...,0]-truth['outcome_frequency'][...,0]).square().mean())
    return loss,{k:float(v.detach()) for k,v in metrics.items()}


@torch.no_grad()
def validation_metrics(model,roots,batch_size,device):
    # Match training: roots have equal weight and futures are uniform within a root.
    groups={}
    for index,root in enumerate(roots):
        groups.setdefault(root['target']['pair'].shape[1],[]).append(index)
    values=[]
    for replicas,indices in groups.items():
        for offset in range(0,len(indices),batch_size):
            chosen=indices[offset:offset+batch_size]
            batch,truth=stack(roots,chosen,device,False)
            output=model(batch)
            for replica in range(replicas):
                if replica:_,truth=stack(roots,chosen,device,False,replica)
                _,metrics=objective(model,batch,truth,output)
                values.append((len(chosen)/replicas,metrics))
    return {key:sum(weight*row[key] for weight,row in values)/len(roots) for key in values[0][1]}


def main():
    parser=argparse.ArgumentParser()
    for name in ('train','validation','full-future-train','full-future-validation','teacher','initialize-from','out'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',type=int,default=9801)
    parser.add_argument('--updates',type=int,default=3000)
    parser.add_argument('--batch-roots',type=int,default=16)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--without-context',action='store_true')
    parser.add_argument('--reselect-seconds',type=float,default=0.)
    parser.add_argument('--outcome-readout',choices=['distribution_only','physical_distributions'],default='distribution_only')
    parser.add_argument('--candidate-comparison',action='store_true')
    parser.add_argument('--without-candidate-attention',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if min(args.updates,args.batch_roots)<1 or args.learning_rate<=0:parser.error('Positive training settings required')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    protocol=json.loads((Path(args.train)/'protocol.json').read_text())
    cfg=replace(FleetConfig(**protocol['config']),context=not args.without_context,
        revision='CITF-whole-fleet-interception-full-future-v3',outcome_readout='distribution_only',
        distance_interval='complete remaining stopped task under the candidate feedback program',reselect_seconds=args.reselect_seconds)
    if cfg.feedback_mode=='persistent_interception':
        cfg=replace(cfg,revision='CITF-persistent-joint-interception-policy-switching-v1')
    if args.outcome_readout=='physical_distributions':
        cfg=replace(cfg,revision='CITF-persistent-joint-safety-capture-distributions-v2',
            outcome_readout='physical_distributions',time_prediction='remaining_task')
    if args.candidate_comparison:
        if args.outcome_readout!='physical_distributions':parser.error('Candidate comparison requires direct physical distributions')
        cfg=replace(cfg,revision='CITF-comparative-candidate-transformer-v3',context=False,
            candidate_comparison=True,candidate_attention=not args.without_candidate_attention)
    elif args.without_candidate_attention:parser.error('Candidate-attention ablation requires candidate comparison')
    if cfg.teacher_checkpoint!=args.teacher:raise ValueError('Training teacher differs from the sampled teacher')
    train,tp,ts=load_data(args.train,args.full_future_train,cfg,'train')
    validation,vp,vs=load_data(args.validation,args.full_future_validation,cfg,'validation')
    for key in ('config','physical_protocol','source_identity','opponent_identity'):
        if tp[key]!=vp[key]:raise ValueError('Fleet data roles differ in '+key)
    if set(r['family'] for r in train)&set(r['family'] for r in validation):raise ValueError('Fleet train/validation overlap')
    arguments={k:v for k,v in vars(args).items() if k not in ('resume','out','device')}
    contract=immutable_run_contract(dict(arguments=arguments,config=asdict(cfg)),
        files=[args.initialize_from,args.teacher,*[Path(d)/f for d in (args.train,args.validation,args.full_future_train,args.full_future_validation)
            for f in ('protocol.json','summary.json')]],code=[__file__,Path(__file__).with_name('fleet_program_policy.py')])
    teacher_sha256=contract['artifacts'][str(Path(args.teacher).resolve())]
    for sampled_protocol in (tp,vp):
        expected=recorded_teacher_identity(sampled_protocol,sampled_protocol['config']['teacher_checkpoint'])
        if not expected or expected!=teacher_sha256:
            raise ValueError('Fleet training teacher differs from the teacher that generated its data')
    model=FleetOutcomeTransformer(cfg).to(args.device)
    initial=torch.load(args.initialize_from,map_location='cpu',weights_only=False)
    initial_state=initial['state']
    if cfg.outcome_readout=='physical_distributions':
        initial_state={k:v for k,v in initial_state.items() if not k.startswith('terminal.')}
    missing,unexpected=model.load_state_dict(initial_state,strict=False)
    if unexpected or any(not name.startswith(('terminal.','fleet_safety.','bank_embed.','bank_layers.','bank_output.')) for name in missing):raise ValueError('Incompatible distance initialization')
    if cfg.time_prediction=='remaining_task' and initial.get('config',{}).get('time_prediction')!='remaining_task':
        with torch.no_grad():model.team[-1].weight[10].zero_();model.team[-1].bias[10].zero_()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.updates,eta_min=3e-5)
    out=Path(args.out);start=0
    if args.resume:
        state=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        require_same_contract(state.get('contract'),contract,'Fleet training resume')
        model.load_state_dict(state['state']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['update'];random.setstate(state['random']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])
        read_journal(out/'training.jsonl',committed_update=start)
    out.mkdir(parents=True,exist_ok=args.resume)
    def payload(update):
        return dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},config=asdict(cfg),
            physical_protocol=tp['physical_protocol'],source_identity=tp['source_identity'],
            teacher_sha256=teacher_sha256,
            opponent_identity=tp['opponent_identity'],update=update,seed=args.seed,contract=contract,
            arguments=vars(args),data=dict(training=ts,validation=vs),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            selection_rule='fixed final update; development complete256 and common768 decide retention',
            supervision='all deployed persistent base policies and paired complete futures; terminal probabilities,time,full stopped synchronous distributions,short motion',
            deployment=('periodically select the predicted best persistent base policy; subsequent reselection can change the continuation'
                if cfg.reselect_seconds>0 else 'one complete feedback program committed at the observed root'))
    program_mean=torch.stack([r['target']['utility'] for r in train]).mean(0)
    best_fixed=int(program_mean.argmax())
    if not args.resume:
        checkpoint=payload(0);save_tensor(checkpoint,out/'stage_000000.pt');save_tensor(checkpoint,out/'latest.pt')
        atomic_json(dict(arguments=vars(args),config=asdict(cfg),contract=contract,training=ts,validation=vs,
            fixed_program_from_training=best_fixed,training_program_mean_utility=program_mean.tolist(),
            effective_training_layouts=len({r['family'] for r in train}),
            effective_validation_layouts=len({r['family'] for r in validation})),out/'protocol.json')
    started=time.time()
    with (out/'training.jsonl').open('a',encoding='utf-8') as log:
        for update in range(start+1,args.updates+1):
            model.train();batch,truth=stack(train,[random.randrange(len(train)) for _ in range(args.batch_roots)],args.device)
            optimizer.zero_grad(set_to_none=True);loss,metrics=objective(model,batch,truth)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fleet training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            if update%100==0 or update==args.updates:
                model.eval();validation_result=validation_metrics(model,validation,args.batch_roots,args.device)
                row=dict(update=update,seconds=time.time()-started,gradient_norm=float(norm),training=metrics,validation=validation_result)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                atomic_json(dict(completed=update==args.updates,update=update,**validation_result),out/'status.json')
                save_tensor(payload(update),out/'latest.pt')
            if update in (1000,3000) or update==args.updates:save_tensor(payload(update),out/f'stage_{update:06d}.pt')
    checkpoint=payload(args.updates)
    for program in range(len(PROGRAMS)):
        fixed=dict(checkpoint);fixed['config']=asdict(replace(cfg,fixed_program=program))
        fixed['method_role']='matched computed model with forced complete feedback program'
        save_tensor(fixed,out/f'fixed_program_{program}.pt')
    atomic_json(dict(completed=True,updates=args.updates,seconds=time.time()-started,training_roots=len(train),
        validation_roots=len(validation),fixed_program_from_training=best_fixed,
        scope='trained complete-candidate model; closed-loop results remain necessary'),out/'summary.json')


if __name__=='__main__':main()
