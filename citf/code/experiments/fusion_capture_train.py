"""Train the distance distributions that actually score joint configurations."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from citf.evaluate import atomic_json
from citf.contracts import (Protocol, immutable_run_contract, require_same_contract,
                            source_identity, file_hash)
from citf.risk import rectified_gaussian_nll
from citf.train import read_journal
from citf.opponents import APFOpponent
from experiments.fusion_capture_data import save_tensor
from experiments.fusion_capture_policy import FusionConfig, FusionOutcomeTransformer, control_score, capture_probability


def targets(payload, cfg):
    xy = payload['real_positions']; valid = payload['time_valid']; E = xy.shape[-3]
    i, j = torch.triu_indices(E, E, 1)
    distances = (xy[:, :, i]-xy[:, :, j]).norm(dim=-1).masked_fill(~valid[:, :, None], torch.inf)
    pair = distances.amin(-1)
    team, at = distances[:, :, j == E-1].amin(-2).min(-1)
    radius = xy[:, :, -1].norm(dim=-1).masked_fill(~valid, torch.inf).amin(-1)
    duration = payload['batch']['known_time_valid'].sum()
    fraction = (at.float()+1)/duration
    collision = (5-pair[:, :, j<E-1]).clamp_min(0).square().sum(-1)
    breach = (15-radius).clamp_min(0).square()
    event = (team < 5).float()
    failure = (pair[:, :, j<E-1]<5).float().sum(-1)+(radius<15).float()
    cost = (cfg.collision_weight*collision+cfg.breach_weight*breach+cfg.failure_weight*failure
            +cfg.progress_weight*(team-5).clamp_min(0)-cfg.capture_weight*event+2*event*fraction)
    weights = valid[:, :, None, :, None]
    mean = (xy*weights).sum(1)/weights.sum(1).clamp_min(1)
    return dict(pair=pair, team=team, radius=radius, fraction=fraction, cost=cost,
                xy=mean, xy_valid=valid.any(1), capture=event)


def load_data(directory, cfg, role=None):
    directory = Path(directory)
    protocol = json.loads((directory/'protocol.json').read_text())
    if role is not None and protocol['role'] != role:
        raise ValueError('Fusion dataset role differs from its requested use')
    physical = asdict(Protocol(**protocol['physical_protocol']))
    opponent = protocol.get('opponent_identity',APFOpponent().identity)
    summary = json.loads((directory/'summary.json').read_text())
    if not summary['completed']: raise ValueError('Data collection is incomplete')
    roots = []
    for path in sorted((directory/'roots').glob('*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if asdict(Protocol(**payload['physical_protocol'])) != physical:
            raise ValueError('Fusion root uses a different physical protocol')
        if payload.get('opponent_identity',APFOpponent().identity) != opponent:
            raise ValueError('Fusion root uses a different opponent from its dataset')
        if (payload['batch']['history'].shape[-2] != cfg.history
                or payload['batch']['nominal'].shape[-2] != cfg.horizon):
            raise ValueError('Fusion dataset history/horizon differs from its model')
        roots.append(dict(batch=payload['batch'], target=targets(payload,cfg),
            root_id=payload['root_id'], family=payload['task']['geometry_family_id']))
    if not roots or len(roots) != summary['roots'] or len({r['root_id'] for r in roots}) != len(roots):
        raise ValueError('Fusion dataset has missing, duplicated or unrecorded roots')
    return roots, summary


def load_pools(directories,cfg,role=None):
    roots=[];pools=[];summaries=[]
    for directory in directories:
        group,summary=load_data(directory,cfg,role)
        pools.append(list(range(len(roots),len(roots)+len(group))))
        roots.extend(group);summaries.append(dict(directory=str(directory),**summary))
    return roots,pools,dict(datasets=summaries,roots=len(roots),
        effective_layouts=len(set(r['family'] for r in roots)))


def stack(roots, indices, device):
    return tuple({key:torch.stack([roots[i][section][key] for i in indices]).to(device)
                  for key in roots[0][section]} for section in ['batch','target'])


def hurdle_nll(distance, output, prefix, threshold, mask=None):
    def get(key):return output[prefix+'_'+key][:,:,None]
    below=distance<threshold
    event=F.binary_cross_entropy_with_logits(get('event_logit').expand_as(distance),below.float(),reduction='none')
    x=(distance/threshold).clamp(1e-6,1-1e-6)
    alpha,beta=get('alpha'),get('beta')
    inside=(-(alpha-1)*x.log()-(beta-1)*torch.log1p(-x)
            +torch.lgamma(alpha)+torch.lgamma(beta)-torch.lgamma(alpha+beta)+math.log(threshold))
    excess=(distance-threshold).clamp_min(1e-4);scale=get('excess_sigma')
    outside=(.5*((excess.log()-get('excess_logmu'))/scale).square()
             +scale.log()+excess.log()+.5*math.log(2*math.pi))
    loss=event+torch.where(below,inside,outside)
    if mask is not None:
        weights=mask.to(loss.dtype).expand_as(loss)
        return (loss*weights).sum()/weights.sum().clamp_min(1)
    return loss.mean()


def objective(model, batch, truth, selection_weight):
    output = model(batch)
    probability_target = truth['capture'].mean(-1)
    if model.cfg.distribution=='hurdle':
        pair=hurdle_nll(truth['pair'],output,'pair',5.)
        team=hurdle_nll(truth['team'],output,'team',5.)
        radius=hurdle_nll(truth['radius'],output,'radius',15.)
        calibration=F.binary_cross_entropy_with_logits(output['team_event_logit'],probability_target)
    else:
        pair = rectified_gaussian_nll(truth['pair'],output['mu'][:,:,None],output['sigma'][:,:,None]).mean()
        team = rectified_gaussian_nll(truth['team'],output['team_mu'][:,:,None],output['team_sigma'][:,:,None]).mean()
        radius = rectified_gaussian_nll(truth['radius'],output['radius_mu'][:,:,None],output['radius_sigma'][:,:,None]).mean()
        z = (5-output['team_mu'])/output['team_sigma']
        calibration = -(probability_target*torch.special.log_ndtr(z)
                       +(1-probability_target)*torch.special.log_ndtr(-z)).mean()
    mask = truth['xy_valid'][:,:,None,:,None].expand_as(truth['xy'])
    motion = F.smooth_l1_loss(output['xy']/5,truth['xy']/5,reduction='none')[mask].mean()
    capture_count = truth['capture'].sum(-1)
    capture_time = (truth['fraction']*truth['capture']).sum(-1)/capture_count.clamp_min(1)
    timing = ((output['time_fraction']-capture_time).square()*(capture_count>0)).sum()/(capture_count>0).sum().clamp_min(1)
    score = control_score(output,model.cfg,batch['positions'].shape[1])
    target_cost = truth['cost'].mean(-1)
    distribution = F.softmax(-target_cost/8,dim=-1)
    selection = -(distribution*F.log_softmax(-score/8,dim=-1)).sum(-1).mean()
    loss = pair+.5*team+.2*radius+.2*calibration+.2*motion+.2*timing+selection_weight*selection
    chosen = score.argmin(-1)
    regret = (target_cost.gather(1,chosen[:,None]).squeeze(1)-target_cost.min(-1).values).mean()
    metrics = dict(loss=loss,pair_nll=pair,team_nll=team,radius_nll=radius,capture_calibration=calibration,
        motion=motion,timing=timing,selection=selection,empirical_regret=regret,
        team_mae=(output['team_mu']-truth['team'].mean(-1)).abs().mean(),
        capture_brier=(capture_probability(output)-probability_target).square().mean())
    return loss,{k:float(v.detach()) for k,v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    for name in ['teacher','out']: parser.add_argument('--'+name,required=True)
    for name in ['train','validation']:parser.add_argument('--'+name,nargs='+',required=True)
    parser.add_argument('--seed',type=int,default=9601)
    parser.add_argument('--updates',type=int,default=3000)
    parser.add_argument('--batch-roots',type=int,default=12)
    parser.add_argument('--selection-weight',type=float,default=.1)
    parser.add_argument('--without-context',action='store_true')
    parser.add_argument('--distribution',choices=['rectified_gaussian','hurdle'],default='rectified_gaussian')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--initialize-from')
    parser.add_argument('--balanced-data',action='store_true')
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    args = parser.parse_args(); torch.set_num_threads(1)
    if (min(args.updates,args.batch_roots) < 1 or not math.isfinite(args.selection_weight)
            or args.selection_weight < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('Updates, batch size and learning rate must be positive; selection weight must be nonnegative')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out)
    if out.exists() and not args.resume:
        raise ValueError('Refuse training overwrite; choose a new output or resume its unchanged configuration')
    datasets=[json.loads((Path(folder)/'protocol.json').read_text()) for folder in args.train+args.validation]
    protocol=datasets[0]
    physical=asdict(Protocol(**protocol['physical_protocol']))
    teacher_sha256=file_hash(args.teacher)
    inputs=[]
    for record in datasets:
        if asdict(Protocol(**record['physical_protocol'])) != physical:
            raise ValueError('Fusion datasets use different task rules')
        sampled_teacher=record.get('teacher_sha256') or file_hash(record['config']['teacher_checkpoint'])
        if sampled_teacher != teacher_sha256:
            raise ValueError('Fusion runtime teacher differs from the sampled teacher')
        inputs.append(record.get('source_identity') or source_identity(
            record['arguments']['project'],record['arguments']['source']))
    if any(value != inputs[0] for value in inputs):
        raise ValueError('Fusion datasets use different source controllers or environment implementations')
    cfg = FusionConfig(teacher_checkpoint=args.teacher,history=protocol['config']['history'],
        horizon=protocol['config']['horizon'],context=not args.without_context,distribution=args.distribution)
    if args.distribution=='hurdle':cfg=replace(cfg,revision='CITF-joint-feedback-hurdle-distance-v2')
    train,pools,train_summary = load_pools(args.train,cfg,'train')
    validation,_,val_summary = load_pools(args.validation,cfg,'validation')
    if set(r['family'] for r in train)&set(r['family'] for r in validation):
        raise ValueError('Training and model-validation geometry overlap')
    parameters={k:v for k,v in vars(args).items() if k not in ('resume','out','device')}
    contract=immutable_run_contract(dict(arguments=parameters,config=asdict(cfg),source_identity=inputs[0],
        training_roots=[(r['root_id'],r['family']) for r in train],
        validation_roots=[(r['root_id'],r['family']) for r in validation]),
        files=[args.teacher,*([args.initialize_from] if args.initialize_from else []),
            *[Path(folder)/name for folder in args.train+args.validation
                for name in ('protocol.json','summary.json')]],code=[__file__])
    model = FusionOutcomeTransformer(cfg).to(args.device)
    if args.initialize_from:
        initial=torch.load(args.initialize_from,map_location='cpu',weights_only=False)
        model.load_state_dict(initial['state'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.updates,eta_min=3e-5)
    start = 0
    if args.resume:
        checkpoint = torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        require_same_contract(checkpoint.get('contract'),contract,'Fusion training resume')
        if checkpoint['config']!=asdict(cfg): raise ValueError('Resume configuration changed')
        model.load_state_dict(checkpoint['state']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); start=checkpoint['update']
        random.setstate(checkpoint['random']); np.random.set_state(checkpoint['numpy'])
        torch.set_rng_state(checkpoint['torch']); torch.cuda.set_rng_state_all(checkpoint['cuda'])
        read_journal(out/'training.jsonl',committed_update=start)
    out.mkdir(parents=True,exist_ok=args.resume)
    def payload(update):
        return dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},config=asdict(cfg),
            physical_protocol=protocol['physical_protocol'],update=update,seed=args.seed,contract=contract,
            teacher_sha256=teacher_sha256,source_identity=inputs[0],
            training_opponents=[record.get('opponent_identity',APFOpponent().identity) for record in datasets],
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(),arguments=vars(args),
            data=dict(training=train_summary,validation=val_summary),
            supervision='unweighted proper likelihood on all common futures; stopped distance,collision,capture and breach; actual score distribution loss',
            selection_rule='complete independent256: success,captures,capped_capture_time; stages500/1500/3000')
    if not args.resume:
        initial=payload(0)
        save_tensor(initial,out/'stage_000000.pt');save_tensor(initial,out/'latest.pt')
        nominal=dict(initial);nominal['config']=asdict(replace(cfg,predictor='nominal'))
        nominal['method_role']='matched same candidate/executor nominal predictor; not trained'
        save_tensor(nominal,out/'nominal.pt')
        atomic_json(dict(arguments=vars(args),config=asdict(cfg),contract=contract,training=train_summary,validation=val_summary,
            effective_training_layouts=len(set(r['family'] for r in train)),
            effective_validation_layouts=len(set(r['family'] for r in validation)),
            readout='all bank scores depend directly on predicted physical outcome distributions'),out/'protocol.json')
    started=time.time()
    with (out/'training.jsonl').open('a',encoding='utf-8') as log:
        for update in range(start+1,args.updates+1):
            model.train()
            indices=([random.choice(random.choice(pools)) for _ in range(args.batch_roots)]
                     if args.balanced_data else [random.randrange(len(train)) for _ in range(args.batch_roots)])
            batch,truth=stack(train,indices,args.device)
            optimizer.zero_grad(set_to_none=True);loss,metrics=objective(model,batch,truth,args.selection_weight)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            if update%100==0 or update==args.updates:
                row=dict(update=update,seconds=time.time()-started,gradient_norm=float(norm),training=metrics)
                model.eval(); values=[]
                with torch.no_grad():
                    for offset in range(0,len(validation),args.batch_roots):
                        b,t=stack(validation,list(range(offset,min(offset+args.batch_roots,len(validation)))),args.device)
                        _,value=objective(model,b,t,args.selection_weight);values.append((len(b['positions']),value))
                row['validation']={k:sum(n*v[k] for n,v in values)/sum(n for n,_ in values) for k in values[0][1]}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                atomic_json(dict(completed=update==args.updates,update=update,**row['validation']),out/'status.json')
                save_tensor(payload(update),out/'latest.pt')
            if update in (500,1500,3000) or update==args.updates:
                save_tensor(payload(update),out/f'stage_{update:06d}.pt')
    atomic_json(dict(completed=True,updates=args.updates,seconds=time.time()-started,
        training_roots=len(train),validation_roots=len(validation),
        scope='trained model; no closed-loop benefit established by this summary'),out/'summary.json')


if __name__=='__main__':main()
