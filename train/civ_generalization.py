"""Incremental replay and relational conditional value for unified CIV-RL."""
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import civ_rl as c
from civ_rl_experiment import digest, make_tasks, tensor_features
from civ_replay_search import score


ARCHITECTURE='relational-counterfactual-v1'
DRIVER_SHA256=digest(__file__)


def atomic_save(record,path):
    path=Path(path)
    temporary=path.with_name(path.name+f'.partial-{os.getpid()}')
    try:
        torch.save(record,temporary)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


class RelationalValueNetwork(nn.Module):
    """Whole-plan graph representation, explicitly conditioned on its reference."""
    def __init__(self,width=48,dropout=.08,encounter_features=False,focus_changes=False):
        super().__init__();self.width=width;self.dropout=dropout
        self.encounter_features=bool(encounter_features)
        self.focus_changes=bool(focus_changes)
        self.node=nn.Sequential(nn.Linear(37 if self.encounter_features else 23,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.message=nn.Sequential(nn.Linear(2*width+7,width),nn.SiLU(),nn.Linear(width,width))
        self.attention=nn.Sequential(nn.Linear(2*width+7,width//2),nn.SiLU(),nn.Linear(width//2,1))
        self.update=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear((9 if self.focus_changes else 6)*width+26,128),nn.SiLU(),nn.Dropout(dropout),
            nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))

    def forward(self,features):
        node=features['node'];edge=features['edge'];global_features=features['global_features']
        if self.encounter_features:
            physical=features.get('encounter_node')
            if physical is None:physical=self.encode_encounter(features)
            node=torch.cat((node,physical),-1)
        h=self.node(node);n=h.shape[-2]
        receiver=h.unsqueeze(-2).expand(*h.shape[:-2],n,n,self.width)
        sender=h.unsqueeze(-3).expand_as(receiver)
        pair=torch.cat((receiver,sender,edge),-1)
        mask=torch.eye(n,dtype=torch.bool,device=h.device)
        weight=self.attention(pair).squeeze(-1).masked_fill(mask,float('-inf')).softmax(-1)
        message=(weight[...,None]*self.message(pair)).sum(-2)
        h=self.update(torch.cat((h,message),-1))
        pooled=torch.cat((h.mean(-2),h.amax(-2)),-1)
        reference=pooled[:,:1].expand_as(pooled)
        reference_global=global_features[:,:1].expand_as(global_features)
        parts=[reference,pooled-reference,pooled,reference_global,global_features]
        if self.focus_changes:
            mass=features.get('change_weight')
            if mass is None:mass=features['node'][...,15:17].abs().sum(-1)
            # Focus on the vessels whose coefficients change, while every
            # embedding still contains messages from the complete fleet.
            mass=mass+1e-4;mass=mass/mass.sum(-1,keepdim=True)
            affected=(h*mass[...,None]).sum(-2)
            factual=(h[:,:1]*mass[...,None]).sum(-2)
            parts.extend((factual,affected-factual,affected))
        return self.readout(torch.cat(parts,-1))

    @staticmethod
    def encode_encounter(features):
        """Public relative motion and candidate response, with no future labels."""
        node=features['node'];relative=node[...,:2]
        velocity=node[...,2:4]*(5./30.)
        distance=relative.norm(dim=-1).clamp_min(1e-4)
        projection=(relative*velocity).sum(-1)
        closest_time=(-projection/velocity.square().sum(-1).clamp_min(1e-6)).clamp(0.,6.)
        closest=(relative+velocity*closest_time[...,None]).norm(dim=-1)
        reference_end=relative+node[...,17:19]
        candidate_end=relative+node[...,19:21]
        reference_distance=reference_end.norm(dim=-1)
        candidate_distance=candidate_end.norm(dim=-1)
        shift=node[...,19:21]-node[...,17:19]
        toward=-relative/distance[...,None]
        maximum_risk=features['edge'][...,6].amax(-1)
        return torch.stack((distance,-projection/distance,closest_time/6.,closest,
            reference_distance,candidate_distance,candidate_distance-reference_distance,
            (shift*toward).sum(-1),(node[...,4:6]*toward).sum(-1),
            (velocity*node[...,4:6]).sum(-1),velocity.norm(dim=-1),
            node[...,15:17].norm(dim=-1),maximum_risk,
            node[...,22]-node[:,:1,:,22]),-1).clamp(-10.,10.)


class StandardizedRelationalValueNetwork(RelationalValueNetwork):
    """Replay-fitted feature scales, fixed during validation and deployment."""
    def __init__(self,width=48,dropout=.08,encounter_features=False,focus_changes=False):
        super().__init__(width,dropout,encounter_features,focus_changes)
        for name,size in (('node',23),('edge',7),('global_features',13)):
            self.register_buffer(name+'_mean',torch.zeros(size))
            self.register_buffer(name+'_scale',torch.ones(size))

    @torch.no_grad()
    def standardize_from(self,features):
        for name in ('node','edge','global_features'):
            value=features[name][features['valid']]
            if name=='edge':
                n=value.shape[-2];value=value[:,~torch.eye(n,dtype=torch.bool,device=value.device)]
            value=value.reshape(-1,value.shape[-1])
            getattr(self,name+'_mean').copy_(value.mean(0))
            getattr(self,name+'_scale').copy_(value.std(0,unbiased=False).clamp_min(.1))

    def forward(self,features):
        normalized={name:(features[name]-getattr(self,name+'_mean'))/getattr(self,name+'_scale')
                    for name in ('node','edge','global_features')}
        if self.encounter_features:normalized['encounter_node']=self.encode_encounter(features)
        if self.focus_changes:normalized['change_weight']=features['node'][...,15:17].abs().sum(-1)
        return super().forward(normalized)


def loss_function(raw,targets,future_mask,valid,importance,kind,gain_weight,rank_weight,
                  loss_mode='softplus'):
    if loss_mode not in ('softplus','paired_margin','paired_events','paired_calibrated','paired_value','paired_unbiased','paired_decision','paired_consistent','paired_precision','paired_joint'):
        raise ValueError('Unknown conditional ranking objective.')
    weights=valid.to(raw.dtype)*importance[:,None]
    fm=future_mask.to(raw.dtype)
    count=fm.sum(-1,keepdim=True).clamp_min(1.)
    mean=(targets*fm[:,:,None]).sum(1)/count
    if kind=='ordinary':
        loss=F.binary_cross_entropy_with_logits(raw,mean[...,None].expand_as(raw),reduction='none').mean(-1)
        return (loss*weights).sum()/weights.sum().clamp_min(1.)
    y0=targets[:,:,:1]
    zero=(fm[:,:,None]*(1-y0)).sum(1)
    one=(fm[:,:,None]*y0).sum(1)
    rescue=(fm[:,:,None]*(1-y0)*targets).sum(1)/zero.clamp_min(1.)
    harm=(fm[:,:,None]*y0*(1-targets)).sum(1)/one.clamp_min(1.)
    baseline=(F.binary_cross_entropy_with_logits(raw[:,0,0],mean[:,0],reduction='none')*importance).mean()
    rw=weights*zero/count;hw=weights*one/count
    if loss_mode=='paired_joint':
        # The observed joint-outcome likelihood preserves the factual event
        # frequencies. Outcome-dependent weights would change this estimand.
        active=weights.clone();active[:,0]=0.
        likelihood=(F.binary_cross_entropy_with_logits(raw[...,1],rescue,reduction='none')*zero/count
            +F.binary_cross_entropy_with_logits(raw[...,2],harm,reduction='none')*one/count)
        loss=baseline+(likelihood*active).sum()/active.sum().clamp_min(1.)
        p0=raw[:,0:1,0].sigmoid().detach()
        gain=(1-p0)*raw[...,1].sigmoid()-p0*raw[...,2].sigmoid()
        gain=torch.cat((torch.zeros_like(gain[:,:1]),gain[:,1:]),-1)
        change=mean-mean[:,:1]
        loss=loss+gain_weight*((gain-change).square()*active).sum()/active.sum().clamp_min(1.)
        if rank_weight:
            # Fit every signed contrast, including zero and negative effects.
            # Selecting only observed winners or weighting by their observed
            # variance would couple the regression weights to label noise.
            pairs=valid[:,:,None]&valid[:,None,:]
            pairs=pairs&torch.triu(torch.ones_like(pairs),1)
            pair_weights=pairs*importance[:,None,None]
            error=(gain[:,:,None]-gain[:,None,:])-(change[:,:,None]-change[:,None,:])
            loss=loss+rank_weight*(error.square()*pair_weights).sum()/pair_weights.sum().clamp_min(1.)
        return loss
    loss=baseline+.5*(F.binary_cross_entropy_with_logits(raw[...,1],rescue,reduction='none')*rw).sum()/rw.sum().clamp_min(1.)
    loss=loss+.5*(F.binary_cross_entropy_with_logits(raw[...,2],harm,reduction='none')*hw).sum()/hw.sum().clamp_min(1.)
    p0=raw[:,0:1,0].sigmoid()
    # Preserve the factual likelihood target when fitting candidate gain:
    # candidate ranking must not directly retarget the factual probability head.
    if loss_mode in ('paired_calibrated','paired_unbiased','paired_decision','paired_consistent','paired_precision'):p0=p0.detach()
    gain=(1-p0)*raw[...,1].sigmoid()-p0*raw[...,2].sigmoid()
    if loss_mode=='paired_value':
        # Learn the complete candidate value through its paired difference;
        # the other heads retain conditional rescue and harm supervision.
        gain=raw[...,0].sigmoid()-p0
    if loss_mode=='paired_events':
        # P(rescue)-P(harm) is the same paired conditional gain as
        # (1-p0)*r-p0*h. Direct event likelihood avoids multiplying two
        # separately estimated rare-event probabilities at deployment.
        rescue_mass=(fm[:,:,None]*(1-y0)*targets).sum(1)/count
        harm_mass=(fm[:,:,None]*y0*(1-targets)).sum(1)/count
        event_targets=torch.stack((1-rescue_mass-harm_mass,rescue_mass,harm_mass),-1)
        event_logits=torch.cat((torch.zeros_like(raw[...,:1]),raw[...,1:]),-1)
        event_loss=-(event_targets*event_logits.log_softmax(-1)).sum(-1)
        loss=baseline+(event_loss*weights).sum()/weights.sum().clamp_min(1.)
        events=event_logits.softmax(-1);gain=events[...,1]-events[...,2]
    if loss_mode!='softplus':
        # The unchanged whole configuration has exactly zero causal effect,
        # just as it does in ensemble_values and the deployment selector.
        gain=torch.cat((torch.zeros_like(gain[:,:1]),gain[:,1:]),-1)
    change=mean-mean[:,:1];active=weights.clone();active[:,0]=0.
    # Outcome-dependent weights change the target conditional mean. These
    # variants retain the importance-corrected paired expectation itself.
    emphasis=active if loss_mode in ('paired_unbiased','paired_decision') else active*(1+3*change.abs())
    loss=loss+gain_weight*((gain-change).square()*emphasis).sum()/emphasis.sum().clamp_min(1.)
    if loss_mode=='paired_consistent':
        # Candidate success and its conditional decomposition must describe
        # the same long-term value, in addition to the paired difference.
        candidate=(p0+gain).clamp(1e-6,1.-1e-6)
        value_likelihood=F.binary_cross_entropy(candidate,mean,reduction='none')
        loss=loss+(value_likelihood*active).sum()/active.sum().clamp_min(1.)
    if rank_weight and loss_mode=='paired_precision':
        # Common futures identify the variance of each candidate contrast.
        # Independent terminal marginals do not retain this covariance.
        paired=targets[:,:,:,None]-targets[:,:,None,:]
        contrast=(paired*fm[:,:,None,None]).sum(1)/count[:,:,None]
        variance=((paired-contrast[:,None]).square()*fm[:,:,None,None]).sum(1)/count[:,:,None]
        precision=(variance/count[:,:,None]+.1).reciprocal().clamp_max(10.)
        upper=torch.triu(torch.ones_like(contrast,dtype=torch.bool),1)
        pair_weights=precision*upper*valid[:,:,None]*valid[:,None,:]*importance[:,None,None]
        error=F.smooth_l1_loss(gain[:,:,None]-gain[:,None,:],contrast,reduction='none',beta=.1)
        loss=loss+rank_weight*(error*pair_weights).sum()/pair_weights.sum().clamp_min(1.)
    elif rank_weight and loss_mode=='paired_decision':
        distribution=(gain/.1).masked_fill(~valid,float('-inf')).softmax(-1)
        regret=change.masked_fill(~valid,float('-inf')).amax(-1)-(distribution*change).sum(-1)
        loss=loss+rank_weight*(regret*importance).sum()/importance.sum().clamp_min(1.)
    elif rank_weight:
        differences=change[:,:,None]-change[:,None,:]
        pw=(differences>0)*differences.clamp_min(0)*valid[:,:,None]*valid[:,None,:]*importance[:,None,None]
        predicted_gap=gain[:,:,None]-gain[:,None,:]
        ranking=(F.relu(.5*differences-predicted_gap).square() if loss_mode!='softplus'
                 else F.softplus(-predicted_gap/.1))
        loss=loss+rank_weight*(ranking*pw).sum()/pw.sum().clamp_min(1.)
    return loss


def load(path,device='cuda:0'):
    record=torch.load(path,map_location='cpu',weights_only=False)
    if record.get('execution_backend') and torch.device(device).type!=record['execution_backend']:
        raise ValueError('This paired policy requires its recorded inference backend.')
    if record.get('value_architecture')!=ARCHITECTURE:raise ValueError('Unexpected value architecture.')
    source=Path(record['source_path'])
    if not source.exists():source=Path(__file__).resolve().parents[1]/record['source_relative_path']
    if digest(source)!=record['source_sha256']:raise ValueError('Source identity changed.')
    config=c.CIVConfig(**record['config'])
    if config.identity()!=record['execution_identity']:raise ValueError('Execution identity changed.')
    networks=[]
    for state in record['networks']:
        network_type=StandardizedRelationalValueNetwork if record.get('normalize',False) else RelationalValueNetwork
        network=network_type(record['width'],record['dropout'],record.get('encounter_features',False),
                             record.get('focus_changes',False)).to(device).eval()
        network.paired_event_value=record.get('paired_event_value',False)
        network.paired_value_output=record.get('paired_value_output',False)
        network.load_state_dict(state);networks.append(network)
    return c.CIVController(c.load_source(source,device),config,networks,record['kind'])


def fit(args):
    torch.set_num_threads(1);torch.set_grad_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    training=[];validation=[];manifests=[];reference=None;start=time.monotonic()
    for number,path in enumerate(args.data):
        data=torch.load(path,map_location='cpu',weights_only=False)
        if not data['completed']:raise ValueError('A completed replay bank is required.')
        if reference is not None and any(data[k]!=reference[k] for k in ('source_sha256','execution_identity')):
            raise ValueError('Independent source pipelines must keep their own replay labels.')
        reference=data;manifests.append(dict(path=str(Path(path).resolve()),sha256=digest(path)))
        roots=data['roots']
        for root in roots:
            if not np.isin(root['targets'][:,root['features']['valid']],(0.,1.)).all():
                raise ValueError('Every admissible candidate needs complete terminal labels.')
            if not all(np.isfinite(root['features'][key]).all()
                       for key in ('node','edge','global_features')):
                raise ValueError('Public candidate features must be finite.')
        if args.old_scene_limit and number<len(args.data)-1:
            allowed={tuple(t['scene']) for t in data['tasks'][:args.old_scene_limit]}
            roots=[r for r in roots if tuple(r['scene']) in allowed]
        for root in roots:
            root=dict(root,group=Path(path).stem+':ag'+str(root['scene'][2]))
            if args.hash_split:
                is_validation=int(hashlib.sha256(str(tuple(root['scene'])).encode()).hexdigest()[:8],16)%5==0
            else:is_validation=number==len(args.data)-1 and root['split']=='validation'
            (validation if is_validation else training).append(root)
    if not training or not validation:raise ValueError('Separate training and validation roots are required.')
    training_scenes=sorted({tuple(root['scene']) for root in training})
    validation_scenes=sorted({tuple(root['scene']) for root in validation})
    if set(training_scenes)&set(validation_scenes):raise ValueError('An initial scene cannot occur in both training and validation.')
    config=c.CIVConfig(**reference['config']);tf=tensor_features(training,args.device);vf=tensor_features(validation,args.device)
    max_futures=max(len(r['targets']) for r in training);options=len(training[0]['targets'][0])
    target=np.zeros((len(training),max_futures,options),dtype=np.float32);future_mask=np.zeros((len(training),max_futures),dtype=bool)
    for i,root in enumerate(training):
        target[i,:len(root['targets'])]=root['targets'].clip(min=0);future_mask[i,:len(root['targets'])]=True
    target=torch.as_tensor(target,device=args.device);future_mask=torch.as_tensor(future_mask,device=args.device)
    benefit=np.asarray([np.any((r['targets']-r['targets'][:,:1])[:,r['features']['valid']]>0) for r in training])
    target_probability=np.ones(len(training),dtype=float)/len(training)
    if args.balance_agility:
        groups=[(root['scene'][1],root['scene'][2]) for root in training]
        counts={group:groups.count(group) for group in set(groups)}
        target_probability=np.asarray([1/(len(counts)*counts[group]) for group in groups])
    probability=target_probability.copy()
    if benefit.any():
        positive=target_probability*benefit
        probability=.75*probability+.25*positive/positive.sum()
    networks=[];opts=[];rngs=[];bootstrap=[]
    restored=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if args.heads_only and restored is None:raise ValueError('Frozen representations require a pretrained checkpoint.')
    if args.independent_start and (restored is None or restored['kind']!='ordinary'):
        raise ValueError('An independence initialization requires a fitted ordinary-value checkpoint.')
    if args.independent_start and args.loss_mode in ('paired_events','paired_value'):
        raise ValueError('Ordinary-Q independence initialization requires the conditional probability parameterization.')
    for member in range(args.members):
        network_type=StandardizedRelationalValueNetwork if args.normalize else RelationalValueNetwork
        torch.manual_seed(args.seed+member)
        network=network_type(args.width,args.dropout,args.encounter_features,args.focus_changes).to(args.device)
        network.paired_event_value=args.kind=='conditional' and args.loss_mode=='paired_events'
        network.paired_value_output=args.kind=='conditional' and args.loss_mode=='paired_value'
        if args.normalize:network.standardize_from(tf)
        if restored:network.load_state_dict(restored['networks'][member%len(restored['networks'])])
        if args.independent_start:
            # One shared Q initialization gives r=Q and h=1-Q, so both
            # objectives start with exactly Delta=Q(candidate)-Q(reference).
            with torch.no_grad():
                head=network.readout[-1];weight=head.weight.mean(0,keepdim=True);bias=head.bias.mean().reshape(1)
                head.weight.copy_(weight.expand_as(head.weight));head.bias.copy_(bias.expand_as(head.bias))
                if args.kind=='conditional':head.weight[2].neg_();head.bias[2].neg_()
        if args.heads_only:
            for parameter in network.parameters():parameter.requires_grad_(False)
            for parameter in network.readout[-1].parameters():parameter.requires_grad_(True)
        networks.append(network);opts.append(torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=args.weight_decay))
        rngs.append(np.random.default_rng(args.seed+member))
        bootstrap.append(np.random.default_rng(args.seed+member+10000).poisson(1,len(training)).clip(min=0).astype(float))
    best=None;history=[];selected=None
    for update in range(0 if args.independent_start else 1,args.updates+1):
        losses=[]
        for network,opt,rng,boot in zip(networks,opts,rngs,bootstrap):
            if update==0:break
            network.train();indices=rng.choice(len(training),args.batch_size,replace=True,p=probability)
            ix=torch.as_tensor(indices,device=args.device);importance=torch.as_tensor(boot[indices]*target_probability[indices]/probability[indices],device=args.device,dtype=torch.float32)
            features={k:v[ix] for k,v in tf.items()}
            raw=network(features)
            loss=loss_function(raw,target[ix],future_mask[ix],features['valid'],importance,args.kind,
                               args.gain_weight,args.rank_weight,args.loss_mode)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5.);opt.step();losses.append(float(loss.detach()))
        if update%args.checkpoint_every==0 or update==args.updates:
            for network in networks:network.eval()
            with torch.no_grad():prediction=c.ensemble_values(networks,vf,args.kind)
            current_best=None;current_record=None
            for penalty in ((0.,) if args.zero_uncertainty else (0.,.5,1.,2.)):
                for margin in ((0.,.02,.05,.1,.2) if args.loss_mode!='softplus' else (-.02,0.,.02,.05,.1,.2)):
                    result=score(networks,vf,validation,args.kind,margin,penalty,prediction)
                    lower_gain=(result['rescues']-result['harms']-args.risk_penalty*np.sqrt(result['rescues']+result['harms']))/result['total']
                    result.update(update=update,loss=float(np.mean(losses)) if losses else 0.,lower_gain=float(lower_gain),seconds=time.monotonic()-start)
                    history.append(result)
                    rank=(lower_gain,result['gain'],-result['harms'],-result['changed_roots'],-update,margin,penalty)
                    if current_best is None or rank>current_best:
                        current_best=rank
                        chosen=replace(config,acceptance_margin=margin,uncertainty_penalty=penalty)
                        source=Path(reference['source_path']);relative=str(source.relative_to(Path(__file__).resolve().parents[1]))
                        record=dict(algorithm='CIV-RL-unified-v1',value_architecture=ARCHITECTURE,source_path=str(source),
                            source_relative_path=relative,source_sha256=reference['source_sha256'],config=asdict(chosen),
                            execution_identity=chosen.identity(),kind=args.kind,width=args.width,dropout=args.dropout,normalize=args.normalize,
                            encounter_features=args.encounter_features,
                            focus_changes=args.focus_changes,
                            paired_event_value=args.kind=='conditional' and args.loss_mode=='paired_events',
                            paired_value_output=args.kind=='conditional' and args.loss_mode=='paired_value',
                            networks=[{k:v.detach().cpu() for k,v in network.state_dict().items()} for network in networks],
                            fit=dict(completed=True,search_completed=False,training_roots=len(training),validation_roots=len(validation),
                                training_fleets=sorted({r['scene'][1] for r in training}),data_sources=manifests,member_seeds=[args.seed+i for i in range(args.members)],
                                updates=update,planned_updates=args.updates,parameter_count=sum(v.numel() for v in networks[0].parameters()),
                                total_parameter_updates_at_checkpoint=update*args.members,
                                planned_total_parameter_updates=args.updates*args.members,
                                split_protocol='scene-hash-modulo-five' if args.hash_split else 'latest-bank-validation',
                                old_scene_limit=args.old_scene_limit,training_scenes=training_scenes,validation_scenes=validation_scenes,
                                batch_size=args.batch_size,learning_rate=args.learning_rate,normalize=args.normalize,balance_agility=args.balance_agility,
                                resume_path=args.resume,resume_sha256=digest(args.resume) if args.resume else None,
                                independent_start=args.independent_start,
                                heads_only=args.heads_only,training_device=args.device,
                                trainable_parameter_count=sum(p.numel() for p in networks[0].parameters() if p.requires_grad),
                                selected=copy.deepcopy(result),objective='conditional stratified BCE and paired mean-gain/ranking' if args.kind=='conditional' else 'ordinary terminal BCE',
                                loss_mode=args.loss_mode,zero_uncertainty=args.zero_uncertainty,
                                gain_weight=args.gain_weight if args.kind=='conditional' else 0.,rank_weight=args.rank_weight if args.kind=='conditional' else 0.,
                                weight_decay=args.weight_decay,bootstrap_roots=True,validation_reused_for_training=False,risk_penalty=args.risk_penalty,
                                runtime=c.runtime_identity(),driver_sha256=DRIVER_SHA256))
                        current_record=record
                    if best is None or rank>best:
                        best=rank;selected=copy.deepcopy(result)
                        atomic_save(current_record,args.output)
            if args.save_stages:
                stage=Path(args.output).with_name(Path(args.output).stem+f'-update{update}.pth')
                if stage.exists():raise ValueError('A real training checkpoint must not be overwritten.')
                current_record['fit'].update(stage_checkpoint=True,search_completed=update==args.updates)
                atomic_save(current_record,stage)
            print(json.dumps(dict(update=update,selected=selected,seconds=time.monotonic()-start)),flush=True)
    record=torch.load(args.output,map_location='cpu',weights_only=False)
    record['fit'].update(search_completed=True,completed_updates=args.updates,validation_history=history)
    atomic_save(record,args.output)
    print(json.dumps(dict(completed=args.output,selected=selected,seconds=time.monotonic()-start)),flush=True)


def ensemble(args):
    """Pool every specified independent fit, then select eta on its shared validation roots."""
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    records=[torch.load(path,map_location='cpu',weights_only=False) for path in args.models]
    if len(records)<2:raise ValueError('An ensemble requires at least two completed fits.')
    reference=records[0];seeds=[];networks=[];sources=[]
    for path,record in zip(args.models,records):
        if not record['fit'].get('search_completed'):raise ValueError('Finish each fit before pooling it.')
        for key in ('value_architecture','source_sha256','execution_identity','kind','width','dropout','normalize'):
            if record[key]!=reference[key]:raise ValueError('Ensemble members must share '+key)
        for key in ('encounter_features','focus_changes','paired_event_value','paired_value_output'):
            if record.get(key,False)!=reference.get(key,False):raise ValueError('Ensemble representation changed.')
        for key in ('data_sources','training_scenes','validation_scenes','parameter_count','planned_updates'):
            if record['fit'][key]!=reference['fit'][key]:raise ValueError('Ensemble fits must share '+key)
        member_seeds=record['fit']['member_seeds']
        if set(seeds)&set(member_seeds):raise ValueError('Independent fits must use distinct member seeds.')
        seeds.extend(member_seeds)
        sources.append(dict(path=str(Path(path).resolve()),sha256=digest(path),member_seeds=member_seeds,
                            selected_updates=record['fit']['updates'],completed_updates=record['fit']['completed_updates']))
        for state in record['networks']:
            network_type=StandardizedRelationalValueNetwork if reference['normalize'] else RelationalValueNetwork
            network=network_type(reference['width'],reference['dropout'],reference.get('encounter_features',False),
                reference.get('focus_changes',False)).to(args.device).eval()
            network.paired_event_value=reference.get('paired_event_value',False)
            network.paired_value_output=reference.get('paired_value_output',False)
            network.load_state_dict(state);networks.append(network)
    expected={x['sha256'] for x in reference['fit']['data_sources']}
    if {digest(path) for path in args.data}!=expected:raise ValueError('Use the original replay banks for threshold selection.')
    scenes=set(map(tuple,reference['fit']['validation_scenes']));validation=[]
    for path in args.data:
        data=torch.load(path,map_location='cpu',weights_only=False)
        validation.extend(dict(root,group=Path(path).stem+':ag'+str(root['scene'][2]))
                          for root in data['roots'] if tuple(root['scene']) in scenes)
    if len(validation)!=reference['fit']['validation_roots']:raise ValueError('Validation root coverage changed.')
    features=tensor_features(validation,args.device)
    with torch.no_grad():prediction=c.ensemble_values(networks,features,reference['kind'])
    history=[];best=None;selected=None
    for margin in (0.,.02,.05,.1,.2):
        result=score(networks,features,validation,reference['kind'],margin,0.,prediction)
        lower=(result['rescues']-result['harms']-args.risk_penalty*np.sqrt(result['rescues']+result['harms']))/result['total']
        result['lower_gain']=float(lower);history.append(result)
        rank=(lower,result['gain'],-result['harms'],-result['changed_roots'],margin)
        if best is None or rank>best:best=rank;selected=result
    record=copy.deepcopy(reference)
    record['networks']=[state for member in records for state in member['networks']]
    config=replace(c.CIVConfig(**reference['config']),acceptance_margin=selected['margin'],uncertainty_penalty=0.)
    record.update(config=asdict(config),execution_identity=config.identity())
    fit=record['fit']
    for key in ('updates','completed_updates','resume_path','resume_sha256','stage_checkpoint'):
        fit.pop(key,None)
    fit.update(completed=True,search_completed=True,ensemble_sources=sources,member_seeds=seeds,
        members=len(networks),independent_value_fits=len(records),independent_source_fits=1,
        total_parameter_updates_at_checkpoint=sum(x['selected_updates']*len(x['member_seeds']) for x in sources),
        planned_total_parameter_updates=sum(x['completed_updates']*len(x['member_seeds']) for x in sources),
        selected=selected,validation_history=history,zero_uncertainty=True,runtime=c.runtime_identity(),
        ensemble_driver_sha256=DRIVER_SHA256)
    atomic_save(record,args.output)
    print(json.dumps(dict(completed=args.output,members=len(networks),selected=selected)),flush=True)


def paired_search(args):
    """Fifty matched optimization rounds with a pairing-information control."""
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    bank_path=Path(args.data[0]);bank=torch.load(bank_path,map_location='cpu',weights_only=False)
    if not bank['completed'] or bank.get('execution_backend')!='cuda':
        raise ValueError('Completed paired CUDA acquisition is required.')
    if args.fixed_split and not bank.get('terminal_labels_complete',False):
        raise ValueError('The fixed hard-scene bank needs complete terminal labels.')
    training=[];validation=[]
    for root in bank['roots']:
        item=dict(root,group='contrast50:ag'+str(root['scene'][2]))
        is_validation=(root['split']=='validation' if args.fixed_split else
            int(hashlib.sha256(str(tuple(root['scene'])).encode()).hexdigest()[:8],16)%5==0)
        if not is_validation and args.training_scene_limit and root['split_order']>=args.training_scene_limit:continue
        (validation if is_validation else training).append(item)
    if args.training_scene_limit and len(training)!=args.training_scene_limit:
        raise ValueError('The frozen training scene budget is incomplete.')
    tf=tensor_features(training,args.device);vf=tensor_features(validation,args.device)
    count=max(len(r['targets']) for r in training);options=len(training[0]['targets'][0])
    targets=np.zeros((len(training),count,options),dtype=np.float32);mask=np.zeros((len(training),count),dtype=bool)
    shuffled=targets.copy();shuffle_rng=np.random.default_rng(args.seed+7000)
    changed=0
    for i,root in enumerate(training):
        n=len(root['targets']);targets[i,:n]=root['targets'].clip(min=0);mask[i,:n]=True
        shuffled[i,:n]=targets[i,:n];start=0 if args.shuffle_all else root.get('old_future_count',n)
        for option in range(1,options):
            shuffled[i,start:n,option]=targets[i,shuffle_rng.permutation(np.arange(start,n)),option]
        assert np.array_equal(shuffled[i,:n].sum(0),targets[i,:n].sum(0))
        changed+=int(np.count_nonzero(shuffled[i,:n]!=targets[i,:n]))
    target=torch.as_tensor(targets,device=args.device);permuted=torch.as_tensor(shuffled,device=args.device)
    future_mask=torch.as_tensor(mask,device=args.device)
    groups=[r['scene'][2] for r in training];counts={v:groups.count(v) for v in set(groups)}
    probability=np.asarray([(3. if r.get('old_future_count',len(r['targets']))<len(r['targets']) else 1.)/counts[r['scene'][2]] for r in training])
    probability/=probability.sum()
    names=['conditional','ordinary','shuffled'];networks={};bases={};optimizers={}
    for name in names:
        path=args.control_model if name=='ordinary' else args.model
        bases[name]=torch.load(path,map_location='cpu',weights_only=False)
        controller=load(path,args.device);networks[name]=controller.networks
        if args.fresh_start:
            fresh=[];record=bases[name]
            for member in range(len(networks[name])):
                torch.manual_seed(args.seed+member)
                network_type=StandardizedRelationalValueNetwork if record.get('normalize',False) else RelationalValueNetwork
                net=network_type(record['width'],record['dropout'],record.get('encounter_features',False),record.get('focus_changes',False)).to(args.device)
                if record.get('normalize',False):net.standardize_from(tf)
                # Share seeded representations. Keep the ordinary readouts
                # independently initialized so their symmetry is not forced.
                # Conditional r=Q, h=1-Q starts from an independence prior.
                if name!='ordinary':
                  with torch.no_grad():
                    head=net.readout[-1];weight=head.weight.mean(0,keepdim=True);bias=head.bias.mean().reshape(1)
                    head.weight.copy_(weight.expand_as(head.weight));head.bias.copy_(bias.expand_as(head.bias))
                    head.weight[2].neg_();head.bias[2].neg_()
                net.paired_event_value=False;net.paired_value_output=False;fresh.append(net)
            networks[name]=fresh;bases[name].update(paired_event_value=False,paired_value_output=False)
        optimizers[name]=[torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4) for net in controller.networks]
        if args.fresh_start:optimizers[name]=[torch.optim.AdamW(net.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay) for net in networks[name]]
        del controller
    members=len(networks['conditional'])
    if any(len(networks[k])!=members for k in names):raise ValueError('Matched member counts required.')
    torch.set_grad_enabled(True)
    bootstrap=[np.random.default_rng(args.seed+m+10000).poisson(1,len(training)).astype(float) for m in range(members)]
    rngs=[np.random.default_rng(args.seed+m) for m in range(members)]
    best={name:None for name in names};histories={name:[] for name in names}
    output=Path(args.output);stem=output.stem;start_time=time.monotonic()
    schedule=[(.25,.1),(.5,.1),(.5,.25),(1.,.25),(2.,.25),(.5,.5),(1.,.5),(2.,.5),(.25,1.),(.5,1.)]
    manifest=dict(path=str(bank_path.resolve()),sha256=digest(bank_path))
    for round_index in range(1,args.rounds+1):
        gain_weight,rank_weight=schedule[(round_index-1)%len(schedule)]
        learning_rate=((args.learning_rate if round_index<=40 else args.learning_rate/3) if args.fresh_start else
            (1e-4 if round_index<=10 else 5e-5 if round_index>40 else 1.5e-4))
        for name in names:
            for net,opt in zip(networks[name],optimizers[name]):
                net.train()
                for group in opt.param_groups:group['lr']=learning_rate
        for update in range(args.round_updates):
            for member in range(members):
                ix_np=rngs[member].choice(len(training),args.batch_size,replace=True,p=probability)
                ix=torch.as_tensor(ix_np,device=args.device)
                features={key:value[ix] for key,value in tf.items()}
                importance=torch.as_tensor(bootstrap[member][ix_np],dtype=torch.float32,device=args.device)
                seed=args.seed+round_index*100000+update*10+member
                for name in names:
                    torch.manual_seed(seed)
                    net=networks[name][member];opt=optimizers[name][member]
                    raw=net(features)
                    label=permuted[ix] if name=='shuffled' else target[ix]
                    kind='ordinary' if name=='ordinary' else 'conditional'
                    loss=loss_function(raw,label,future_mask[ix],features['valid'],importance,kind,gain_weight,rank_weight,args.paired_loss_mode)
                    opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        summary={}
        for name in names:
            for net in networks[name]:net.eval()
            kind='ordinary' if name=='ordinary' else 'conditional'
            with torch.no_grad():prediction=c.ensemble_values(networks[name],vf,kind)
            candidates=[]
            for margin in (0.,.02,.05,.075,.1,.15,.2,.3,.4):
                result=score(networks[name],vf,validation,kind,margin,0.,prediction)
                result['lower_gain']=(result['rescues']-result['harms']-.5*np.sqrt(result['rescues']+result['harms']))/result['total']
                candidates.append(result)
            selected=max(candidates,key=lambda r:(r['lower_gain'],r['gain'],-r['harms'],r['margin']))
            selected.update(round=round_index,gain_weight=gain_weight,rank_weight=rank_weight,learning_rate=learning_rate)
            histories[name].append(copy.deepcopy(selected));summary[name]={k:selected[k] for k in ['success','total','rescues','harms','margin']}
            record=copy.deepcopy(bases[name]);config=replace(c.CIVConfig(**bank['config']),acceptance_margin=selected['margin'],uncertainty_penalty=0.)
            record.update(config=asdict(config),execution_identity=config.identity(),execution_backend='cuda',
                networks=[{key:value.detach().cpu() for key,value in net.state_dict().items()} for net in networks[name]])
            record['fit'].update(completed=True,search_completed=False,updates=round_index*args.round_updates,completed_rounds=round_index,
                planned_rounds=args.rounds,updates_per_round=args.round_updates,planned_updates=args.rounds*args.round_updates,
                data_sources=[manifest],training_roots=len(training),validation_roots=len(validation),selected=copy.deepcopy(selected),
                loss_mode='ordinary_terminal' if name=='ordinary' else args.paired_loss_mode,
                gain_weight=gain_weight if name!='ordinary' else 0.,rank_weight=rank_weight if name!='ordinary' else 0.,
                learning_rate=learning_rate,batch_size=args.batch_size,
                training_scenes=sorted({tuple(r['scene']) for r in training}),validation_scenes=sorted({tuple(r['scene']) for r in validation}),
                member_seeds=[args.seed+m for m in range(members)],fresh_start=args.fresh_start,
                initialization='common seeded representations; ordinary independent readouts; conditional independence prior' if args.fresh_start else 'selected previous policy',
                fixed_split=args.fixed_split,training_scene_limit=args.training_scene_limit,shuffle_all_futures=args.shuffle_all,
                pairing_control=name=='shuffled',new_future_labels_permuted=changed if name=='shuffled' else 0,
                candidate_marginals_preserved=True,base_policy=args.control_model if name=='ordinary' else args.model,
                base_policy_sha256=digest(args.control_model if name=='ordinary' else args.model),
                matched_draws_and_dropout=True,balance_agility=True,contrast_root_weight=3.,zero_uncertainty=True,
                total_parameter_updates_at_checkpoint=round_index*args.round_updates*members,
                planned_total_parameter_updates=args.rounds*args.round_updates*members,
                total_additional_parameter_updates=round_index*args.round_updates*members,driver_sha256=DRIVER_SHA256,
                cuda_version=torch.version.cuda,torch_version=torch.__version__,execution_backend='cuda')
            stage=output.with_name(f'{stem}-{name}-round{round_index:02d}.pth');atomic_save(record,stage)
            rank=(selected['lower_gain'],selected['gain'],-selected['harms'],-round_index)
            if best[name] is None or rank>best[name]:
                best[name]=rank;atomic_save(record,output.with_name(f'{stem}-{name}.pth'))
        atomic_save(dict(completed=False,completed_rounds=round_index,planned_rounds=args.rounds,histories=histories,
            pairing_permutations_changed_labels=changed,all_candidate_marginals_preserved=True,
            new_data_source=manifest,seconds=time.monotonic()-start_time),output.with_name(f'{stem}-search.pth'))
        print(json.dumps(dict(round=round_index,results=summary,seconds=time.monotonic()-start_time)),flush=True)
    for name in names:
        path=output.with_name(f'{stem}-{name}.pth');record=torch.load(path,weights_only=False)
        record['fit'].update(search_completed=True,completed_rounds=args.rounds,completed_updates=args.rounds*args.round_updates,
            validation_history=histories[name],completed_total_parameter_updates=args.rounds*args.round_updates*members)
        atomic_save(record,path)
    atomic_save(dict(completed=True,completed_rounds=args.rounds,planned_rounds=args.rounds,histories=histories,
        pairing_permutations_changed_labels=changed,all_candidate_marginals_preserved=True,
        new_data_source=manifest,seconds=time.monotonic()-start_time),output.with_name(f'{stem}-search.pth'))


def calibrate(args):
    """Fit bounded probability-head calibration on the existing validation split."""
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    record=torch.load(args.model,map_location='cpu',weights_only=False)
    if record.get('paired_event_value') or record.get('paired_value_output'):
        raise ValueError('Head calibration requires the probability decomposition or ordinary value.')
    scenes=set(map(tuple,record['fit']['validation_scenes']));roots=[]
    for manifest in record['fit']['data_sources']:
        if digest(manifest['path'])!=manifest['sha256']:raise ValueError('Calibration data identity changed.')
        bank=torch.load(manifest['path'],map_location='cpu',weights_only=False)
        roots.extend(dict(root,group=Path(manifest['path']).stem+':ag'+str(root['scene'][2]))
                     for root in bank['roots'] if tuple(root['scene']) in scenes)
    if len(roots)!=record['fit']['validation_roots']:raise ValueError('Calibration coverage changed.')
    controller=load(args.model,args.device);features=tensor_features(roots,args.device)
    with torch.no_grad():raw=torch.stack([network(features) for network in controller.networks])
    torch.set_grad_enabled(True)
    means=[];rescues=[];harms=[];zeros=[];ones=[]
    for root in roots:
        target=torch.as_tensor(root['targets'].clip(min=0),device=args.device)
        y0=target[:,:1];zero=(1-y0).sum();one=y0.sum()
        means.append(target.mean(0));zeros.append(zero/len(target));ones.append(one/len(target))
        rescues.append(((1-y0)*target).sum(0)/zero.clamp_min(1.))
        harms.append((y0*(1-target)).sum(0)/one.clamp_min(1.))
    mean=torch.stack(means);rescue=torch.stack(rescues);harm=torch.stack(harms)
    valid=features['valid'].float();rw=valid*torch.stack(zeros)[:,None];hw=valid*torch.stack(ones)[:,None]
    log_temperature=nn.Parameter(torch.zeros(3,device=args.device));bias=nn.Parameter(torch.zeros(3,device=args.device))
    optimizer=torch.optim.Adam([log_temperature,bias],lr=.02)
    for _ in range(args.updates):
        probability=(raw/log_temperature.exp()+bias).sigmoid().mean(0).clamp(1e-6,1-1e-6)
        if record['kind']=='ordinary':
            error=F.binary_cross_entropy(probability,mean[...,None].expand_as(probability),reduction='none').mean(-1)
            loss=(error*valid).sum()/valid.sum().clamp_min(1.)
        else:
            loss=F.binary_cross_entropy(probability[:,0,0],mean[:,0])
            loss=loss+.5*(F.binary_cross_entropy(probability[...,1],rescue,reduction='none')*rw).sum()/rw.sum().clamp_min(1.)
            loss=loss+.5*(F.binary_cross_entropy(probability[...,2],harm,reduction='none')*hw).sum()/hw.sum().clamp_min(1.)
        loss=loss+.01*(log_temperature.square().sum()+bias.square().sum())
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():log_temperature.clamp_(-np.log(2.),np.log(2.));bias.clamp_(-2.,2.)
    temperature=log_temperature.detach().cpu().exp();shift=bias.detach().cpu()
    for state in record['networks']:
        state['readout.5.weight']/=temperature[:,None]
        state['readout.5.bias']=state['readout.5.bias']/temperature+shift
    record['fit']['probability_calibration']=dict(base_policy=str(args.model),base_sha256=digest(args.model),
        steps=args.updates,temperature=temperature.tolist(),bias=shift.tolist(),prior_weight=.01,
        scene_split='original validation scenes',roots=len(roots),frozen_representation=True,
        folded_into_existing_head=True,validation_reused_for_calibration=True)
    atomic_save(record,args.output)
    controller=load(args.output,args.device)
    with torch.no_grad():prediction=c.ensemble_values(controller.networks,features,record['kind'])
    history=[]
    for margin in (0.,.02,.05,.075,.1,.15,.2,.3,.4):
        result=score(controller.networks,features,roots,record['kind'],margin,0.,prediction)
        result['lower_gain']=(result['rescues']-result['harms']-.5*np.sqrt(result['rescues']+result['harms']))/result['total']
        history.append(result)
    selected=max(history,key=lambda r:(r['lower_gain'],r['gain'],-r['harms'],r['margin']))
    config=replace(controller.config,acceptance_margin=selected['margin'],uncertainty_penalty=0.)
    record.update(config=asdict(config),execution_identity=config.identity())
    record['fit']['probability_calibration'].update(selected=selected,history=history)
    atomic_save(record,args.output)
    print(json.dumps(dict(completed=args.output,calibration=record['fit']['probability_calibration'])),flush=True)


def evaluate(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    controller=load(args.model,args.device);tasks=make_tasks(args.seed,args.tasks,args.vessels,args.agilities)
    rows=c.run_episodes(controller,tasks,progress=True)
    record=dict(completed=True,algorithm='CIV-RL-unified-v1',value_architecture=ARCHITECTURE,
        model=args.model,policy_sha256=digest(args.model),execution_identity=controller.config.identity(),
        config=asdict(controller.config),runtime=c.runtime_identity(),driver_sha256=DRIVER_SHA256,tasks=tasks,rows=rows,
        success=sum(r['success'] for r in rows),collision=sum(r['collision'] for r in rows),selected=sum(r['selected']>0 for r in rows))
    Path(args.output).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('completed','success','collision','selected')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fit','evaluate','ensemble','calibrate','paired-search'])
    p.add_argument('--data',nargs='+');p.add_argument('--kind',choices=['conditional','ordinary']);p.add_argument('--model')
    p.add_argument('--models',nargs='+')
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--tasks',type=int,default=160);p.add_argument('--vessels',type=int,default=6);p.add_argument('--agilities',type=float,nargs='+',default=[2.,2.5])
    p.add_argument('--updates',type=int,default=3000);p.add_argument('--members',type=int,default=3);p.add_argument('--width',type=int,default=48)
    p.add_argument('--dropout',type=float,default=.08);p.add_argument('--weight-decay',type=float,default=1e-3)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--gain-weight',type=float,default=.5);p.add_argument('--rank-weight',type=float,default=.1)
    p.add_argument('--risk-penalty',type=float,default=.5);p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--old-scene-limit',type=int,default=0);p.add_argument('--hash-split',action='store_true');p.add_argument('--resume')
    p.add_argument('--normalize',action='store_true')
    p.add_argument('--balance-agility',action='store_true')
    p.add_argument('--independent-start',action='store_true')
    p.add_argument('--heads-only',action='store_true')
    p.add_argument('--loss-mode',choices=['softplus','paired_margin','paired_events','paired_calibrated','paired_value','paired_unbiased','paired_decision','paired_consistent','paired_precision','paired_joint'],default='softplus')
    p.add_argument('--paired-loss-mode',choices=['paired_precision','paired_joint'],default='paired_precision')
    p.add_argument('--zero-uncertainty',action='store_true')
    p.add_argument('--save-stages',action='store_true')
    p.add_argument('--encounter-features',action='store_true')
    p.add_argument('--focus-changes',action='store_true')
    p.add_argument('--control-model');p.add_argument('--rounds',type=int,default=50);p.add_argument('--round-updates',type=int,default=100)
    p.add_argument('--fixed-split',action='store_true');p.add_argument('--training-scene-limit',type=int,default=0)
    p.add_argument('--fresh-start',action='store_true');p.add_argument('--shuffle-all',action='store_true')
    args=p.parse_args();{'fit':fit,'evaluate':evaluate,'ensemble':ensemble,'calibrate':calibrate,'paired-search':paired_search}[args.command](args)


if __name__=='__main__':main()
