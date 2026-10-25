"""Matched research controls with explicit information and execution paths."""
from pathlib import Path
import copy
import hashlib
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from conditional_fusion_value import OrdinaryJointReturnDecision


class COMAReturnDecision(OrdinaryJointReturnDecision):
    """Critic counterfactual advantages under the frozen source preference.

    Q is fitted to the same absolute returns as the ordinary control. For
    each joint candidate, the other choices stay fixed while one vessel is
    marginalized under its source-preference distribution. The sum of these
    advantages ranks the common complete program/window candidates.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        codes=self.candidate_codes;n=self.defenders
        powers=10**torch.arange(n)
        digits=(codes[:,None]//powers)%10
        counterfactual=codes[:,None,None]+(torch.arange(10)[None,None]-digits[:,:,None])*powers[None,:,None]
        spans=self.candidate_spans[:,None,None].expand_as(counterfactual)
        all_pairs=torch.cat((torch.stack((codes,self.candidate_spans),-1),
            torch.stack((counterfactual,spans),-1).reshape(-1,2)))
        unique,inverse=torch.unique(all_pairs,dim=0,return_inverse=True)
        self.register_buffer('query_codes',unique[:,0]);self.register_buffer('query_spans',unique[:,1])
        self.register_buffer('value_indices',inverse[:len(codes)])
        self.register_buffer('counterfactual_indices',inverse[len(codes):].reshape(len(codes),n,10))

    def candidate_values(self,value_context,codes,action,order):
        batch=len(action);root=value_context[:,0,:-1];queries=len(self.query_codes);parts=[]
        for begin in range(0,queries,256):
            end=min(begin+256,queries);count=end-begin
            context=torch.cat((root[:,None].expand(-1,count,-1),
                (self.query_spans[begin:end]/20.)[None,:,None].expand(batch,-1,-1)),-1)
            program=self.query_codes[begin:end][None].expand(batch,-1)
            parts.append(self.program_model(context.flatten(0,1),program.flatten()).reshape(batch,count))
        value=torch.cat(parts,-1)
        preference=action[...,2].gather(1,order)
        options=self.program_model.program_options
        coefficients=torch.where(options[None,None]>=0.,options[None,None],preference[...,None,None])
        probability=(-((coefficients-preference[...,None,None])/.4).square().mean(-1)).softmax(-1)
        counterfactual=value[:,self.counterfactual_indices]
        baseline=(counterfactual*probability[:,None]).sum(-1)
        return (value[:,self.value_indices,None]-baseline).sum(-1)


class GenericFusionAdapter(nn.Module):
    """Ordinary attention and additive team-return fitting over fusion choices.

    State mode receives current states and source commands. Candidate mode
    additionally receives the same candidate trajectories as the analytic
    method. Neither mode computes the analytic pair cost or a joint search.
    """
    def __init__(self,mode='state',width=64,feature_width=84,state=None,layers=1,
                 candidate_coordinates='relative_world'):
        super().__init__()
        if mode not in ('state','candidate'):raise ValueError('Unknown generic fusion input.')
        if layers<1 or candidate_coordinates not in ('relative_world','shared_task'):
            raise ValueError('Invalid ordinary attention representation.')
        self.mode=mode;self.width=width;self.feature_width=feature_width
        self.layers=layers;self.candidate_coordinates=candidate_coordinates
        self.embed=nn.Sequential(nn.Linear(feature_width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.attention=nn.MultiheadAttention(width,4,batch_first=True)
        self.extra_attention=nn.ModuleList(nn.MultiheadAttention(width,4,batch_first=True) for _ in range(layers-1))
        self.extra_norm=nn.ModuleList(nn.LayerNorm(width) for _ in range(layers-1))
        self.readout=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,1))
        if state is not None:self.load_state_dict(state)

    def forward(self,features):
        if self.mode=='state':
            features=torch.cat((features[...,:20],torch.zeros_like(features[...,20:])),dim=-1)
        elif self.candidate_coordinates=='shared_task':
            features=generic_shared_task_features(features)
        batch,vessels,choices,_=features.shape
        encoded=self.embed(features).flatten(1,2)
        message=self.attention(encoded,encoded,encoded,need_weights=False)[0]
        for attention,norm in zip(self.extra_attention,self.extra_norm):
            joint=norm(encoded+message)
            message=message+attention(joint,joint,joint,need_weights=False)[0]
        return self.readout(torch.cat((encoded,message),-1)).reshape(batch,vessels,choices)

    @torch.no_grad()
    def control(self,actor,packet,action):
        features,candidates=generic_fusion_features(actor,packet,action,predict=self.mode=='candidate')
        choice=self(features).argmax(-1)
        gates=candidates.gather(2,choice[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        rule=packet['node'][...,11:13]
        return torch.cat((gates*action[...,:2]+(1.-gates)*rule,torch.ones_like(action[...,:1])),-1)


def generic_shared_task_features(features):
    """Express all candidate positions in the same observed task coordinates."""
    if features.shape[-1]!=84:raise ValueError('Shared coordinates require the recorded 32-point candidate paths.')
    displacement=features[...,20:].reshape(*features.shape[:-1],32,2)
    # The public row uses the inward attacker-to-target direction. The
    # stored global bearing is its outward counterpart. Relative paths use
    # 15 m while the public start positions use 30 m.
    direction=-features[...,8:10]
    rotated=torch.stack(((displacement*direction[...,None,:]).sum(-1),
        direction[...,0,None]*displacement[...,1]-direction[...,1,None]*displacement[...,0]),-1)
    positions=features[...,:2,None].transpose(-2,-1)+.5*rotated
    return torch.cat((features[...,:20],positions.flatten(-2)),-1)


@torch.no_grad()
def generic_fusion_features(actor,packet,action,predict=True):
    from conditional_fusion_value import intervention_task_context
    context,order=intervention_task_context(packet,True,True)
    n=action.shape[1]
    rows=context[:,:8*n].reshape(-1,n,8).gather(1,order.argsort(-1)[...,None].expand(-1,-1,8))
    source=action[...,:2];rule=packet['node'][...,11:13]
    raw=action[...,2:3]
    center=(2.*torch.tanh(torch.logit(.02+.96*raw)/2.)).sigmoid()
    entries=torch.stack((torch.zeros_like(center),center,torch.ones_like(center)),-2).expand(-1,-1,-1,2)
    digits=torch.arange(9,device=action.device)
    candidates=torch.stack((entries[:,:,digits//3,0],entries[:,:,digits%3,1]),-1)
    public=torch.cat((rows,context[:,-5:,None].transpose(1,2).expand(-1,n,-1),source,rule,raw),-1)
    if predict:
        paths=actor.candidate_risk(dict(packet),source,candidate_grid=candidates,return_paths=True)
        paths=(paths-packet['motion'][:,:,:2][:,:,None,None])/15.
        if paths.shape[-2]!=32:
            paths=F.interpolate(paths.flatten(0,2).transpose(1,2),size=32,mode='linear',align_corners=False)
            paths=paths.transpose(1,2).reshape(*candidates.shape[:3],32,2)
        trajectory=paths.flatten(-2)
    else:trajectory=source.new_zeros((*candidates.shape[:3],64))
    features=torch.cat((public[:,:,None].expand(-1,-1,9,-1),candidates,trajectory),-1)
    return features,candidates


@torch.no_grad()
def generic_source_action(actor,packet):
    """Return the source Adapter output before the candidate-center map."""
    mean,_,hidden=actor.proposal_distribution(packet['obs'])
    source=mean.tanh();net=actor.base
    own=net.activation(net.a1(torch.cat((source,packet['node'][...,11:13]),-1)))
    gate=net.adap_layer(torch.cat((own,hidden),-1)).tanh()*.5+.5
    return torch.cat((source,gate),-1)


def equivalent_fusion_choices(candidates,source,rule,command,tolerance=2e-5):
    """Identify every candidate producing the measured teacher command."""
    physical=candidates*source[:,:,None]+(1.-candidates)*rule[:,:,None]
    error=(physical-command[:,:,None]).abs().amax(-1)
    matching=error<=tolerance
    if not bool(matching.any(-1).all()):
        raise ValueError('The teacher command is outside the common candidate library.')
    return matching


def generic_distillation_episode(task):
    """Query the analytic selector at teacher or learner-visited public states."""
    import develop_source_fusion as d
    from conditional_fusion_value import paired_effect_controls
    scene,future,teacher_path,behavior_path,cap=task
    teacher=d.joint_load(str(teacher_path),Path(teacher_path).stat().st_mtime_ns)
    learner=(None if behavior_path is None else
        d.joint_load(str(behavior_path),Path(behavior_path).stat().st_mtime_ns))
    if teacher.successor_task_value is not None or not teacher.actuator_fusion_refinement:
        raise ValueError('Distillation requires a joint-fusion teacher without a value intervention.')
    if learner is not None and (learner.generic_fusion_adapter is None
            or learner.actuator_fusion_refinement or learner.successor_task_value is not None):
        raise ValueError('Student behavior must use only the generic fusion adapter.')
    np.random.seed(scene[0]);torch.manual_seed(scene[0])
    env=d.numerical_source().TADEnv(scene[1]);obs,_=env.reset(agility=scene[2],noisy_agility=False)
    if future is not None:np.random.seed(future)
    features=[];labels=[];executed=[];done=0
    while not done and len(labels)<cap:
        packet=d.mechanism_tensor(d.joint_packet(env,obs))
        with torch.no_grad():
            raw=generic_source_action(teacher,packet)
            x,candidates=generic_fusion_features(teacher,packet,raw)
            action=teacher(packet,deterministic=True,gate_only=True)[0]
            factual=paired_effect_controls(teacher,packet,action)[...,4:6]
            valid=equivalent_fusion_choices(candidates,raw[...,:2],packet['node'][...,11:13],factual)
            if learner is None:
                physical=factual
                selected=valid.to(torch.long).argmax(-1)
            else:
                learner_raw=generic_source_action(learner,packet)
                if not torch.allclose(learner_raw,raw,atol=2e-6,rtol=2e-6):
                    raise ValueError('The student and teacher must share their frozen source commands.')
                selected=learner.generic_fusion_adapter(x).argmax(-1)
                gates=candidates.gather(2,selected[...,None,None].expand(-1,-1,1,2)).squeeze(2)
                physical=gates*raw[...,:2]+(1.-gates)*packet['node'][...,11:13]
            control=torch.cat((physical,torch.ones_like(raw[...,2:3])),-1)
        features.append(x[0].numpy());labels.append(valid[0].numpy());executed.append(selected[0].numpy())
        obs,_,done,_=d.joint_task_step(env,control[0].numpy(),1.,0.)
    return dict(features=np.asarray(features,dtype=np.float32),teacher_choices=np.asarray(labels),
        choices=np.asarray(executed),steps=len(labels),terminal=bool(done),
        outcome=d.outcome(env,done) if done else None,scene=scene,future=future,
        behavior=None if behavior_path is None else str(behavior_path))


def collect_generic_distillation(teacher,tasks,output,*,behaviors=(None,),budget=8192,workers=2):
    """Shared teacher queries; partial trajectories contribute no return labels."""
    from concurrent.futures import ProcessPoolExecutor
    import develop_source_fusion as d
    import run_remote_evidence as r
    teacher=Path(teacher);output=Path(output)
    if budget<1 or workers<1 or not tasks or not behaviors:raise ValueError('Empty distillation design.')
    specification=dict(teacher=str(teacher),teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest(),
        tasks=tasks,behaviors=[None if x is None else str(x) for x in behaviors],
        behavior_sha256=[None if x is None else hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in behaviors],
        budget=budget)
    if output.exists():
        previous=torch.load(output,map_location='cpu',weights_only=False)
        if previous.get('specification')!=specification:raise ValueError('Distillation data have a different design.')
        if previous.get('completed'):return previous
        raise ValueError('An incomplete distillation collection requires its recorded partial observations.')
    used=0;episode=0;rows=[]
    if workers==1:r.initialize_source_branches(str(d.SOURCE))
    with (ProcessPoolExecutor(max_workers=workers,initializer=r.initialize_source_branches,initargs=(str(d.SOURCE),))
          if workers>1 else nullcontext(None)) as pool:
        while used<budget:
            remaining=budget-used;jobs=[]
            for k in range(workers):
                cap=min(400,(remaining+workers-k-1)//(workers-k))
                if not cap:break
                scene,future=tasks[(episode+k)%len(tasks)]
                behavior=behaviors[(episode+k)%len(behaviors)]
                jobs.append((scene,future,str(teacher),behavior,cap));remaining-=cap
            batch=list(pool.map(generic_distillation_episode,jobs) if pool is not None else map(generic_distillation_episode,jobs))
            rows.extend(batch);used+=sum(row['steps'] for row in batch);episode+=len(batch)
            if episode%16==0 or used==budget:
                print(dict(phase='generic_teacher_queries',episodes=episode,steps=used,budget=budget),flush=True)
    result=dict(features=np.concatenate([row['features'] for row in rows]),
        teacher_choices=np.concatenate([row['teacher_choices'] for row in rows]),
        choices=np.concatenate([row['choices'] for row in rows]),simulation_steps=used,
        episodes=[{key:value for key,value in row.items() if key not in ('features','teacher_choices','choices')} for row in rows],
        specification=specification,completed=True,
        source='supervised joint-selector queries on teacher and student trajectories; not absolute-return learning')
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(result,output);return result


def fit_distilled_generic_adapter(base,datasets,mode,output,*,updates=12000,seed=42,device='cuda',warm_start=None,
                                 width=64,layers=1,candidate_coordinates='relative_world',balance_changes=False):
    """Approximate the teacher with independent neural candidate choices."""
    import develop_source_fusion as d
    if d.joint_value_objectives(base):raise ValueError('Distillation starts without learned intervention values.')
    if not datasets or any(not data.get('completed') for data in datasets):raise ValueError('Complete query collections are required.')
    torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+104729)
    features=torch.as_tensor(np.concatenate([data['features'] for data in datasets]),device=device)
    labels=torch.as_tensor(np.concatenate([data['teacher_choices'] for data in datasets]),device=device,dtype=torch.bool)
    if labels.shape!=features.shape[:-1] or not bool(labels.any(-1).all()):raise ValueError('Invalid teacher candidate targets.')
    net=GenericFusionAdapter(mode=mode,width=width,feature_width=features.shape[-1],layers=layers,
        candidate_coordinates=candidate_coordinates).to(device)
    change_frequency=float((~labels[...,4]).float().mean())
    if balance_changes and not 0.<change_frequency<1.:
        raise ValueError('Balanced imitation requires changed and retained teacher choices.')
    previous_updates=0
    if warm_start is not None:
        previous=warm_start['generic_fusion_adapter']
        if (previous['mode']!=mode or previous['width']!=net.width or previous['feature_width']!=net.feature_width
                or previous.get('layers',1)!=layers or previous.get('candidate_coordinates','relative_world')!=candidate_coordinates):
            raise ValueError('Warm student dimensions or input mode changed.')
        net.load_state_dict(previous['state']);previous_updates=warm_start['generic_fusion_fit']['updates']
    optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4 if warm_start is not None else 8e-4,weight_decay=.005)
    with torch.enable_grad():
        for step in range(updates):
            index=torch.randint(len(features),(128,),generator=rng).to(device)
            logits=net(features[index]);target=labels[index]
            per_vessel=-logits.log_softmax(-1).masked_fill(~target,-torch.inf).logsumexp(-1)
            weight=(torch.where(target[...,4],.5/(1.-change_frequency),.5/change_frequency)
                    if balance_changes else torch.ones_like(per_vessel))
            loss=(per_vessel*weight).mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%1000==0:
                accuracy=target.gather(-1,logits.argmax(-1,keepdim=True)).float().mean()
                print(dict(phase='generic_distillation_fit',mode=mode,update=step+1,loss=float(loss.detach()),
                    training_command_agreement=float(accuracy)),flush=True)
    model=copy.deepcopy(base);model.pop('actuator_fusion_refinement',None)
    model['generic_fusion_adapter']=dict(mode=mode,width=net.width,feature_width=net.feature_width,
        layers=layers,candidate_coordinates=candidate_coordinates,
        state={key:value.detach().cpu() for key,value in net.state_dict().items()})
    model['generic_fusion_fit']=dict(updates=previous_updates+updates,last_updates=updates,seed=seed,
        simulation_steps=sum(data['simulation_steps'] for data in datasets),observations=len(features),
        parameters=sum(parameter.numel() for parameter in net.parameters()),loss=float(loss.detach()),
        teacher_sha256=[data['specification']['teacher_sha256'] for data in datasets],
        objective='categorical imitation of physically equivalent joint-selector commands',source=datasets[-1]['source'],
        balance_changes=balance_changes,changed_teacher_choice_fraction=change_frequency,
        observation_sha256=hashlib.sha256(features.cpu().numpy().tobytes()+labels.cpu().numpy().tobytes()).hexdigest())
    model['mechanism_ablation']=(False,True,False)
    model['comparison_scope']='joint-selector distillation into ordinary attention with independent deployed choices'
    torch.save(model,output);return model


def generic_replay_episode(task):
    """Shared behavior data uses only original source decisions and exploration."""
    import develop_source_fusion as d
    scene,future,base,cap,exploration_seed=task
    actor=d.joint_load(str(base),Path(base).stat().st_mtime_ns)
    # The base supplies source commands and the common candidate predictor;
    # no learned value, analytic pair selection, or teacher action is read.
    np.random.seed(scene[0]);torch.manual_seed(scene[0])
    env=d.numerical_source().TADEnv(scene[1]);obs,_=env.reset(agility=scene[2],noisy_agility=False)
    if future is not None:np.random.seed(future)
    rng=np.random.default_rng(exploration_seed);features=[];choices=[];done=0
    while not done and len(choices)<cap:
        packet=d.mechanism_tensor(d.joint_packet(env,obs))
        with torch.no_grad():
            source=actor.proposal_distribution(packet['obs'])[0].tanh()
            net=actor.base;hidden=actor.proposal_distribution(packet['obs'])[2]
            own=net.activation(net.a1(torch.cat((source,packet['node'][...,11:13]),-1)))
            gate=net.adap_layer(torch.cat((own,hidden),-1)).tanh()*.5+.5
            action=torch.cat((source,gate),-1)
            x,candidates=generic_fusion_features(actor,packet,action)
            selected=np.where(rng.random(scene[1])<.35,rng.integers(0,9,scene[1]),4)
            gates=candidates[0,torch.arange(scene[1]),torch.as_tensor(selected)]
            physical=torch.cat((gates*source[0]+(1.-gates)*packet['node'][0,:,11:13],torch.ones_like(gate[0])),-1)
        features.append(x[0].numpy());choices.append(selected)
        obs,_,done,_=d.joint_task_step(env,physical.numpy(),1.,0.)
    return dict(features=np.asarray(features,dtype=np.float32),choices=np.asarray(choices),
        success=float(d.outcome(env,done)['success']) if done else None,steps=len(choices),terminal=bool(done))


def collect_generic_replay(base,tasks,output,budget=32768,workers=3):
    from concurrent.futures import ProcessPoolExecutor
    import develop_source_fusion as d
    import run_remote_evidence as r
    used=0;episode=0;features=[];choices=[];targets=[];censored=0
    if workers==1:r.initialize_source_branches(str(d.SOURCE))
    with (ProcessPoolExecutor(max_workers=workers,initializer=r.initialize_source_branches,initargs=(str(d.SOURCE),))
          if workers>1 else nullcontext(None)) as pool:
        while used<budget:
            remaining=budget-used;jobs=[]
            for k in range(workers):
                cap=min(400,remaining)
                if not cap:break
                scene,future=tasks[(episode+k)%len(tasks)]
                jobs.append((scene,future,str(base),cap,73001+episode+k));remaining-=cap
            for row in (pool.map(generic_replay_episode,jobs) if pool is not None else map(generic_replay_episode,jobs)):
                used+=row['steps']
                if row['terminal']:
                    features.append(row['features']);choices.append(row['choices'])
                    targets.extend([row['success']]*row['steps'])
                else:censored+=row['steps']
            episode+=len(jobs)
            if episode%24==0 or used==budget:
                print(dict(phase='generic_shared_replay',episodes=episode,steps=used,budget=budget),flush=True)
    result=dict(features=np.concatenate(features),choices=np.concatenate(choices),targets=np.asarray(targets,dtype=np.float32),
        simulation_steps=used,censored_steps=censored,episodes=episode,base=str(base),
        base_sha256=hashlib.sha256(Path(base).read_bytes()).hexdigest(),completed=True,
        source='shared source-Adapter behavior with independent exploration; no M1 teacher or M2 value')
    torch.save(result,output);return result


def fit_generic_adapter(base,data,mode,output,updates=10000,seed=42,device='cuda',conservative_weight=.1):
    import develop_source_fusion as d
    if d.joint_value_objectives(base):raise ValueError('Generic fusion starts without learned intervention values.')
    torch.manual_seed(seed);rng=torch.Generator().manual_seed(seed+104729)
    net=GenericFusionAdapter(mode=mode,feature_width=data['features'].shape[-1]).to(device)
    features=torch.as_tensor(data['features'],device=device)
    actions=torch.as_tensor(data['choices'],device=device,dtype=torch.long)
    target=torch.as_tensor(data['targets'],device=device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005)
    with torch.enable_grad():
        for step in range(updates):
            index=torch.randint(len(target),(128,),generator=rng).to(device)
            values=net(features[index]);prediction=values.gather(-1,actions[index,...,None]).squeeze(-1).mean(-1)
            return_loss=F.binary_cross_entropy_with_logits(prediction,target[index])
            # Terminal behavior returns do not identify arbitrary unseen
            # joint choices. Penalize unsupported maximization in the usual
            # offline action-value objective, without a deployed support map.
            observed=values.gather(-1,actions[index,...,None]).squeeze(-1)
            conservative=(values.logsumexp(-1)-observed).mean()
            loss=return_loss+conservative_weight*conservative
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%1000==0:print(dict(phase='generic_return_fit',mode=mode,update=step+1,loss=float(loss.detach())),flush=True)
    model=copy.deepcopy(base);model.pop('actuator_fusion_refinement',None)
    model['generic_fusion_adapter']=dict(mode=mode,width=net.width,feature_width=net.feature_width,
        state={k:v.detach().cpu() for k,v in net.state_dict().items()})
    model['generic_fusion_fit']=dict(updates=updates,seed=seed,simulation_steps=data['simulation_steps'],
        observations=len(target),parameters=sum(p.numel() for p in net.parameters()),loss=float(loss.detach()),
        conservative_weight=conservative_weight,
        objective='absolute team task return with conservative action-value fitting; additive action values',source=data['source'])
    model['mechanism_ablation']=(False,True,False)
    model['comparison_scope']='ordinary attention fusion with independent choices; no analytic pair cost or joint search'
    torch.save(model,output);return model


def _batched_model_episodes(task):
    import develop_source_fusion as d
    import run_remote_evidence as r
    path,indices,tasks,device=task[:4]
    fixed_inference_width=task[4] if len(task)>4 else 0
    r.SOURCE=d.SOURCE
    return indices,r.batched_credit_rows(path,tasks,batch_size=len(tasks),device=device,
        fixed_inference_width=fixed_inference_width)


def evaluate_models(models,tasks,workers=3,result_suffix='source-development',evaluation_context=None,
                    opponent_checkpoint=None,inference_batch=0,inference_device='cpu',isolate_inference=True,
                    fixed_inference_width=0):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import develop_source_fusion as d
    import run_remote_evidence as r
    if (not isinstance(fixed_inference_width,int) or isinstance(fixed_inference_width,bool)
            or fixed_inference_width<0 or fixed_inference_width and not inference_batch):
        raise ValueError('Fixed inference requires batched evaluation and a nonnegative integer width.')
    if inference_batch:
        if inference_batch < 1 or workers < 1:
            raise ValueError('Batched evaluation requires a positive batch size and worker count.')
        if opponent_checkpoint is not None:
            raise ValueError('Use the native episode evaluator for a learned opponent.')
        if not isolate_inference and workers!=1:
            raise ValueError('In-process batched evaluation uses exactly one worker.')
        evaluation_context=dict(evaluation_context or {},inference_batch=inference_batch,
            inference_device=inference_device,episode_executor='batched_native_single_value_stage')
        if not isolate_inference:evaluation_context['inference_process']='calling_process'
        if fixed_inference_width:evaluation_context['fixed_inference_width']=fixed_inference_width
    if opponent_checkpoint is not None:
        evaluation_context=dict(evaluation_context or {},
            frozen_opponent_sha256=hashlib.sha256(Path(opponent_checkpoint).read_bytes()).hexdigest())
    if workers==1:r.initialize_source_branches(str(d.SOURCE),opponent_checkpoint)
    for path in map(Path,models):
        output=path.with_name(path.stem+'-'+result_suffix+'.pth')
        previous=None;digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if output.exists():
            previous=torch.load(output,weights_only=False,map_location='cpu')
            if evaluation_context is not None and (
                    previous.get('evaluation_context')!=evaluation_context
                    or previous.get('tasks')!=tasks or previous.get('policy_sha256')!=digest):
                raise ValueError('A resumed evaluation must retain its locked models, tasks and selection context.')
            if previous.get('tasks')!=tasks or previous.get('policy_sha256')!=digest:previous=None
            if previous and previous.get('completed'):continue
        rows=previous['rows'] if previous else [None]*len(tasks)
        start=time.monotonic();revision=path.stat().st_mtime_ns
        if inference_batch:
            pending=[i for i,row in enumerate(rows) if row is None]
            groups=[pending[begin:begin+inference_batch] for begin in range(0,len(pending),inference_batch)]
            jobs=[(str(path),indices,[tasks[index] for index in indices],inference_device,
                fixed_inference_width) for indices in groups]
            # Keep CUDA inside child processes even with one worker, so the
            # controller can subsequently fork a fresh training stage safely.
            # A training process that already initialized CUDA must evaluate
            # inline instead of forking its CUDA runtime. Other pipelines
            # retain isolated inference before their later training forks.
            with (ProcessPoolExecutor(max_workers=workers,initializer=r.initialize_source_branches,
                    initargs=(str(d.SOURCE),)) if isolate_inference else nullcontext(None)) as pool:
                measured_jobs=(pool.map(_batched_model_episodes,jobs,chunksize=1)
                    if pool is not None else map(_batched_model_episodes,jobs))
                for indices,measured in measured_jobs:
                    for index,row in zip(indices,measured):rows[index]=row
                    complete=[row for row in rows if row is not None]
                    record=dict(policy=str(path),policy_sha256=digest,tasks=tasks,rows=rows,
                        evaluated=len(complete),success=sum(row['success'] for row in complete),
                        completed=len(complete)==len(tasks),elapsed_seconds=time.monotonic()-start,
                        evaluation_context=evaluation_context)
                    r.atomic_save(record,output)
                    print(dict(policy=path.name,evaluated=len(complete),success=record['success'],
                        elapsed=round(record['elapsed_seconds'],1),inference_batch=inference_batch),flush=True)
            continue
        with (ProcessPoolExecutor(max_workers=workers,initializer=r.initialize_source_branches,
              initargs=(str(d.SOURCE),opponent_checkpoint))
              if workers>1 else nullcontext(None)) as pool:
            jobs=[(i,(scene,str(path),revision,False,0,0,0,0,future))
                for i,(scene,future) in enumerate(tasks) if rows[i] is None]
            if pool is None:results=((i,d.joint_episode(task)) for i,task in jobs)
            else:
                pending={pool.submit(d.joint_episode,task):i for i,task in jobs}
                results=((pending[job],job.result()) for job in as_completed(pending))
            for index,row in results:
                rows[index]=row;done=[row for row in rows if row is not None]
                if len(done)%32==0 or len(done)==len(tasks):
                    record=dict(policy=str(path),policy_sha256=digest,tasks=tasks,rows=rows,evaluated=len(done),
                        success=sum(row['success'] for row in done),completed=len(done)==len(tasks),elapsed_seconds=time.monotonic()-start)
                    if evaluation_context is not None:record['evaluation_context']=evaluation_context
                    torch.save(record,output)
                    print(dict(policy=path.name,evaluated=len(done),success=record['success'],elapsed=round(record['elapsed_seconds'],1)),flush=True)


def value_learning_checkpoint(model, updates, data=None, *, device='cuda:0'):
    """Materialize an actual optimizer snapshot with its own value calibration."""
    from local_intervention_value import refresh_local_calibration, calibrate_conditional_differences
    snapshots = model['learning_snapshots']
    if updates not in snapshots['states']:
        raise ValueError('The requested learning point was not recorded during optimization.')
    checkpoint = copy.deepcopy({key:value for key,value in model.items() if key != 'learning_snapshots'})
    recipe = checkpoint['successor_task_value']; fit = checkpoint['successor_task_fit']
    if recipe['kind'] == 'plain_mlp_return':
        recipe['state'] = copy.deepcopy(snapshots['states'][updates])
        members = 1
    elif recipe['kind'] == 'local_difference_joint':
        if data is None:
            raise ValueError('A conditional learning point needs its original paired calibration data.')
        recipe['states'] = copy.deepcopy(snapshots['states'][updates]); members = len(recipe['states'])
        if recipe.get('neural_factual_calibration') is not None:
            if 'neural_factual_calibration' not in snapshots:
                raise ValueError('A cross-fitted learning point needs its own recorded factual calibration.')
            recipe['neural_factual_calibration'] = copy.deepcopy(snapshots['neural_factual_calibration'][updates])
            fit['factual_calibration_diagnostics'] = copy.deepcopy(snapshots['factual_calibration_diagnostics'][updates])
        checkpoint = refresh_local_calibration(checkpoint, data, device=device)
        if recipe.get('stratum_member_corrections') is not None:
            checkpoint = calibrate_conditional_differences(checkpoint, data, device=device)
        fit = checkpoint['successor_task_fit']
    else:
        raise ValueError('Learning curves require the matched conditional or ordinary estimator.')
    initial_updates = model['successor_task_fit'].get('total_parameter_updates', snapshots['final_updates'])-snapshots['final_updates']
    fit.update(updates=updates, member_updates=[updates//members]*members,
        total_parameter_updates=initial_updates+updates, last_loss=None,
        learning_curve_final_updates=snapshots['final_updates'],
        learning_curve_rule='actual continuous optimizer path; fixed observations and candidate catalog')
    return checkpoint


def catalog_intervention_action(factual, option, order):
    """Execute a canonical program through its measured frozen predecessor."""
    factual = np.asarray(factual); order = np.asarray(order, dtype=np.int64)
    code, delay, span = map(int, option); defenders = len(factual)
    if not np.array_equal(np.sort(order), np.arange(defenders)):
        raise ValueError('Catalog execution requires the actual vessel permutation.')
    if code == 0:
        if delay or span:
            raise ValueError('The factual catalog entry must have zero timing.')
        return factual.copy()
    if code < 0 or code >= 10**defenders or delay < 0 or span <= 0:
        raise ValueError('Invalid physical intervention program.')
    coefficients = np.asarray([[-1., -1.]] + [[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)],
                              dtype=factual.dtype)
    preference = coefficients[(code//(10**np.arange(defenders))) % 10][order.argsort()]
    one = np.ones((defenders, 1), dtype=factual.dtype)
    first = preference if delay == 0 else -np.ones_like(preference)
    # The continuation actor is the predecessor itself, so chain depth is 0.
    return np.concatenate((factual[:, :3], -one, (delay+span)*one, -2*one,
        np.zeros((defenders, 2), dtype=factual.dtype), delay*one, first, preference, 0*one), -1)


def measure_intervention_choices(base, models, tasks, output, *, repetitions=8, seed=4329,
                                 workers=3, inference_device='cuda:0', runtime_sha256=None,
                                 ranking_candidates=0, branch_inference_batch=0):
    """Measure proposed interventions at unbiased new factual-trajectory states."""
    from concurrent.futures import ProcessPoolExecutor
    import develop_source_fusion as d
    import run_remote_evidence as r

    if workers < 1 or branch_inference_batch < 0:
        raise ValueError('Choice validation needs positive workers and a nonnegative branch batch.')

    base = Path(base); models = {label: Path(path) for label, path in models.items()}
    output = Path(output)
    model = torch.load(base, map_location='cpu', weights_only=False)
    if d.joint_value_objectives(model):
        raise ValueError('Choice validation needs the common frozen predecessor without M2.')
    identity = dict(predecessor_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        models={label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in models.items()},
        tasks=tasks, repetitions=repetitions, seed=seed, inference_device=inference_device,
        root_sampling='one uniformly sampled eligible pre-action state from each complete factual episode',
        runtime_sha256=runtime_sha256)
    if branch_inference_batch:
        identity['branch_execution'] = dict(inference_batch=int(branch_inference_batch),
            inference_device=inference_device,
            pairing='the same root, physical intervention and future seed for every compared decision')
    if ranking_candidates:
        identity['ranking_design'] = dict(random_candidates=int(ranking_candidates),
            selection='uniform nonfactual catalog entries plus each model predicted best and deployed choice',
            outcomes='complete paired continuations under shared future seeds')
    result = (torch.load(output, map_location='cpu', weights_only=False) if output.exists() else
              dict(identity=identity, records=[], simulation_steps=0, completed=False,
                   result_role='development validation of selected intervention values on new layouts'))
    if result['identity'] != identity:
        raise ValueError('Choice validation must retain its models, layouts, root design and futures.')
    if result['completed']:
        return result
    r.initialize_source_branches(str(d.SOURCE))
    predecessor = d.joint_load(str(base), base.stat().st_mtime_ns)
    policies = {label: d.joint_load(str(path), path.stat().st_mtime_ns) for label, path in models.items()}
    for label, actor in policies.items():
        candidate = torch.load(models[label], map_location='cpu', weights_only=False)
        if candidate['successor_task_fit']['predecessor_sha256'] != identity['predecessor_sha256']:
            raise ValueError('Every selected intervention must use the same measured continuation.')
        actor.successor_task_value.record_decisions = True
        actor.successor_task_value.inference_device = inference_device
    periods = {actor.successor_task_value.decision_period for actor in policies.values()}
    if len(periods) != 1:
        raise ValueError('Choice validation requires a shared observation clock.')
    period = periods.pop()
    common_options = None
    if ranking_candidates:
        if ranking_candidates < 1:
            raise ValueError('A ranking design requires a positive random candidate count.')
        for actor in policies.values():
            options = actor.successor_task_value.options.detach().cpu().numpy()
            if common_options is not None and not np.array_equal(common_options, options):
                raise ValueError('Ranking comparisons require exactly the same candidate catalog.')
            common_options = options
    with (ProcessPoolExecutor(max_workers=workers, initializer=r.initialize_source_branches,
          initargs=(str(d.SOURCE),)) if workers > 1 and not branch_inference_batch else nullcontext(None)) as pool:
        # Start fork workers before the parent initializes a CUDA value module.
        if pool is not None:
            list(pool.map(abs, range(-workers, 0)))
        for index in range(len(result['records']), len(tasks)):
            scene, prefix_future = tasks[index]
            np.random.seed(scene[0]); torch.manual_seed(scene[0])
            env = d.numerical_source().TADEnv(scene[1])
            obs, _ = env.reset(agility=scene[2], noisy_agility=False)
            if prefix_future is not None:
                np.random.seed(prefix_future)
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, 973331]))
            elapsed = done = seen = 0; root = None
            while not done:
                if elapsed % period == 0:
                    seen += 1
                    if rng.integers(seen) == 0:
                        root = (copy.deepcopy(env), obs.copy(), np.random.get_state(), elapsed)
                with torch.no_grad():
                    action = predecessor(d.mechanism_tensor(d.joint_packet(env, obs)), deterministic=True)[0][0].numpy()
                obs, _, done, _, steps = d.joint_advance(env, obs, predecessor, action)
                elapsed += steps
            result['simulation_steps'] += elapsed
            root_env, root_obs, random_state, root_step = root
            packet = d.mechanism_tensor(d.joint_packet(root_env, root_obs))
            with torch.no_grad():
                factual = predecessor(packet, deterministic=True)[0][0].numpy()
            actions = {'factual': factual}; decisions = {}; catalogs = {}
            for label, actor in policies.items():
                with torch.no_grad():
                    proposed = actor(dict(packet), deterministic=True)[0][0].numpy()
                decision = actor.successor_task_value.last_decision
                if decision is None:
                    raise ValueError('An eligible validation state did not record its value decision.')
                decisions[label] = decision
                if proposed.shape[-1] == 14:
                    # The branch actor is already the common predecessor.
                    # Rebase the policy-chain pointer without changing the
                    # delay, duration, coefficients or physical execution.
                    proposed = proposed.copy(); proposed[:, 13] = 0.
                actions[label] = proposed if bool(decision['accepted'][0]) else factual
                if ranking_candidates:
                    value = actor.successor_task_value
                    value.record_catalog = True
                    try:
                        # Diagnostic scoring includes options that the deployed
                        # upper bounds can skip; the actual choice above is retained.
                        with torch.no_grad():
                            actor(dict(packet), deterministic=True)
                        catalog = value.last_catalog
                        if catalog is None or not np.array_equal(catalog['options'].numpy(), common_options):
                            raise ValueError('Every value model must score the common physical catalog.')
                        catalogs[label] = catalog
                    finally:
                        value.record_catalog = False
                        value.last_decision = decision
            ranking = None
            if ranking_candidates:
                from conditional_fusion_value import intervention_task_context
                _, order = intervention_task_context(packet, True, True)
                sampling = np.random.default_rng(np.random.SeedSequence([seed, index, 607057]))
                random_indices = sampling.choice(np.arange(1, len(common_options)),
                    size=min(ranking_candidates, len(common_options)-1), replace=False)
                selected = {0, *map(int, random_indices)}
                for label, catalog in catalogs.items():
                    selected.add(int(catalog['predicted_gain'][0].argmax()))
                    chosen = np.asarray(decisions[label]['option'][0])
                    selected.add(int(np.flatnonzero((common_options == chosen).all(-1))[0]))
                indices = np.asarray(sorted(selected), dtype=np.int64)
                labels = ['factual' if index == 0 else f'catalog:{index}' for index in indices]
                for candidate_index, label in zip(indices, labels):
                    if candidate_index:
                        actions[label] = catalog_intervention_action(factual, common_options[candidate_index],
                                                                    order[0].cpu().numpy())
                ranking = dict(indices=indices, options=common_options[indices], labels=labels,
                    random_indices=random_indices, predictions={label:dict(
                        factual_value=catalog['factual_value'].numpy(),
                        predicted_gain=catalog['predicted_gain'][:, indices].numpy(),
                        score=catalog['score'][:, indices].numpy()) for label, catalog in catalogs.items()})
            unique = []; assignment = {}
            for label, action in actions.items():
                matched = next((j for j, other in enumerate(unique)
                                if action.shape == other.shape and np.array_equal(action, other)), None)
                if matched is None:
                    matched = len(unique); unique.append(action)
                assignment[label] = matched
            futures = [int(np.random.SeedSequence([seed, index, repeat, 917805]).generate_state(1)[0])
                       for repeat in range(repetitions)]
            jobs = [(str(base), dict(environment=root_env, observation=root_obs,
                     random_state=random_state, action=action), scene, future, 1)
                    for action in unique for future in futures]
            if branch_inference_batch:
                from local_intervention_value import source_successor_batches
                rows = source_successor_batches(jobs, batch_size=branch_inference_batch,
                    device=inference_device)
            else:
                rows = list(pool.map(r.source_successor_branch, jobs, chunksize=1) if pool is not None
                            else map(r.source_successor_branch, jobs))
            result['simulation_steps'] += sum(row['physical_steps'] for row in rows)
            rows = [{key: value for key, value in row.items() if key != 'response'} for row in rows]
            outcomes = {label: rows[position*repetitions:(position+1)*repetitions]
                        for label, position in assignment.items()}
            record = dict(scene=scene, prefix_future=prefix_future,
                root_step=root_step, eligible_states=seen, factual_episode_outcome=int(done),
                packet={key: value[0].detach().cpu().numpy() for key, value in packet.items()},
                actions=actions, decisions=decisions, futures=futures, outcomes=outcomes)
            if ranking is not None:
                record['ranking'] = ranking
            result['records'].append(record)
            result['completed'] = len(result['records']) == len(tasks)
            r.atomic_save(result, output)
            print(dict(phase='measured_intervention_choices', roots=len(result['records']),
                total=len(tasks), branches=len(rows), simulation_steps=result['simulation_steps']), flush=True)
    return result


def fit_m2_controls(template,base,datasets,directory,prefix,width,updates=10000,device='cuda',
                    modes=('ordinary','coma','shared-reference','no-peer','no-time','no-support')):
    """Fresh, matched fits for the complete dedicated M2 comparison matrix."""
    import conditional_fusion_value as c
    directory=Path(directory);paths=[]
    for mode in modes:
        path=directory/f'{prefix}-{mode}.pth';paths.append(path)
        if path.exists():continue
        if mode in ('ordinary','coma'):
            model=c.fit_ordinary_task_return_policy(base,datasets,path,width=width,updates=updates,seed=42,device=device)
            if mode=='coma':
                model['successor_task_value']['kind']='coma_joint_return'
                model['successor_task_fit']['decision']='sum of critic counterfactual advantages under source preference'
                model['comparison_scope']='same absolute-return critic data and budget; independent COMA-style decision'
        else:
            model=c.fit_shared_stage_task_policy(template,datasets,path,width=width,updates=updates,seed=42,
                device=device,peer_conditioned=mode!='no-peer')
            stage=model['successor_task_value']
            while stage is not None:
                if mode=='no-time':stage['explicit_wait']=False
                if mode=='no-support':
                    stage.update(use_factual_support=False,contexts=[],candidate_contexts=None,
                        candidate_support=None,candidate_variance=None,support_weighted=False,
                        confidence_scale=0.,time_bandwidth=None)
                stage=stage.get('prior')
            model['successor_task_fit']['dedicated_control']=mode
        model['successor_task_fit'].update(matched_updates=updates,matched_width=width,
            dataset_roots=[len(c.successor_task_examples(data,absolute_frame=True)) for data in datasets])
        torch.save(model,path)
        print(dict(phase='m2_control_fit_complete',mode=mode,path=str(path)),flush=True)
    return paths


def retrain_local_fusion_control(base, reference, blueprint, references, output, *, workers=4,
                               device='cuda:0', inference_batch=0, sampling_device='cpu'):
    """Fit current single-vessel difference learning on an M1 replacement.

    The reference supplies the local experiment design, sampling cap and
    optimizer recipe. All states, actions, terminal outcomes and calibration
    data are generated by the replacement predecessor itself.
    """
    import develop_source_fusion as d
    import local_intervention_value as l
    base = Path(base); reference = Path(reference); output = Path(output)
    ref = torch.load(reference, map_location='cpu', weights_only=False)
    source = torch.load(blueprint, map_location='cpu', weights_only=False)
    fit = ref['successor_task_fit']
    if ref['successor_task_value']['kind'] != 'local_difference_joint':
        raise ValueError('Use the frozen current whole-plan local-difference method.')
    if output.exists():
        model = torch.load(output, map_location='cpu', weights_only=False)
        if model['replacement_retraining']['reference_sha256'] != hashlib.sha256(reference.read_bytes()).hexdigest():
            raise ValueError('The saved fit uses a different frozen reference.')
        return model
    bank_path = output.with_name(output.stem+'-factual-data.pth')
    bank = l.collect_factual_trajectories(base, references, bank_path, workers=workers,
        inference_batch=inference_batch, inference_device=sampling_device)
    data_path = output.with_name(output.stem+'-training-data.pth')
    data = l.recollect_local_design(base, source, data_path,
        simulation_budget=fit['simulation_steps']-bank['simulation_steps'], workers=workers,
        inference_batch=inference_batch, inference_device=sampling_device)
    continuation = l.paired_continuation_bank(data)
    bank['records'] += continuation['records']
    bank['paired_continuation_state_labels'] = continuation['state_labels']
    bank['training_data'] = [str(bank_path), str(data_path)]
    bank['state_labels'] = sum(len(row['values']) for row in bank['records'])
    model = l.fit_joint_potential(data, output, factual_bank=bank, device=device,
        updates=fit['updates'], width=fit['width'], seed=fit['seed'],
        edge_weight=fit['difference_weight'], anchor_weight=fit['factual_contrast_weight'],
        rank_weight=fit['local_rank_weight'], factual_weight=fit['factual_weight'],
        **{key:ref['successor_task_value'][key] for key in
           ['bandwidth','margin','confidence','global_penalty','context_time_scale',
            'interpolation_power','context_conditioned','motion_basis','local_contrast_head',
            'conditional_strata','joint_factual_calibration','residual_potential']})
    model['successor_task_value']['inference_device'] = device
    model['replacement_retraining'] = dict(reference=str(reference),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        predecessor_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
        simulation_budget=fit['simulation_steps'], simulation_steps=model['successor_task_fit']['simulation_steps'],
        updates=fit['updates'], inherited_values=False, inherited_labels=False,
        comparison='same M1 candidate forecasts; fresh local intervention outcomes and whole-plan value fitting')
    torch.save(model, output)
    return model


def intervention_root_design(blueprint):
    """Recover the reference failure strata and physical lookback windows."""
    import conditional_fusion_value as c
    plans={}
    for rollout in blueprint['rollouts']:
        row=rollout['row']
        for branch in rollout['branches']:
            packet={k:torch.as_tensor(v[None]) for k,v in branch['packet'].items()}
            context,_=c.intervention_task_context(packet,True,True)
            key=np.round(context[0].numpy(),6).tobytes()
            if key in plans:continue
            # The fixed source protocol uses 80 seconds and 0.2-second ticks.
            root_step=round(float(context[0,-1])*400)
            finish_step=round(float(row['duration'])/.2)
            plans[key]=dict(outcome=int(row['outcome']),lookback_steps=max(1,finish_step-root_step))
    return [plans[np.round(e['context'],6).tobytes()]
        for e in c.successor_task_examples(blueprint,absolute_frame=True)]


def recollect_blueprint(policy,blueprint,root_tasks,output,initial=False,workers=3):
    """Match sampled programs/futures while replaying this control's own prefix.

    Only the experimental design comes from the blueprint. Root states,
    physical successors, factual outcomes, and terminal labels are all newly
    generated under the replacement fusion controller and its own M2 prefix.
    """
    from concurrent.futures import ProcessPoolExecutor
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    r.initialize_source_branches(str(d.SOURCE))
    reference=c.successor_task_examples(blueprint,absolute_frame=True)
    root_design=intervention_root_design(blueprint)
    if len(reference)!=len(root_tasks):raise ValueError('Match every planned intervention root.')
    actor=d.joint_load(str(policy),Path(policy).stat().st_mtime_ns)
    model=torch.load(policy,weights_only=False,map_location='cpu')
    span=int(blueprint['span']);rollouts=[];steps=0;reached=[]
    with (ProcessPoolExecutor(max_workers=workers,initializer=r.initialize_source_branches,initargs=(str(d.SOURCE),))
          if workers>1 else nullcontext(None)) as pool:
        for index,(example,(scene,root_future)) in enumerate(zip(reference,root_tasks)):
            np.random.seed(scene[0]);torch.manual_seed(scene[0])
            env=d.numerical_source().TADEnv(scene[1]);obs,_=env.reset(agility=scene[2],noisy_agility=False)
            if root_future is not None:np.random.seed(root_future)
            elapsed=0
            if not initial:
                history=[];done=0
                while not done:
                    history.append((elapsed,copy.deepcopy(env),obs.copy(),np.random.get_state()))
                    with torch.no_grad():action=actor(d.mechanism_tensor(d.joint_packet(env,obs)),deterministic=True)[0][0].numpy()
                    obs,_,done,_,count=d.joint_advance(env,obs,actor,action)
                    elapsed+=count;steps+=count
                cutoff=max(0,elapsed-root_design[index]['lookback_steps'])
                elapsed,env,obs,random_state=max((item for item in history if item[0]<=cutoff),key=lambda item:item[0])
                np.random.set_state(random_state)
            packet=d.joint_packet(env,obs);tensor=d.mechanism_tensor(packet)
            with torch.no_grad():source=actor(tensor,deterministic=True,gate_only=True)[0][0].numpy()[:,:3]
            _,order=c.intervention_task_context(tensor,True,True);inverse=order[0].argsort().numpy()
            options=np.asarray(blueprint['program_options']);actions={}
            for code,_ in example['samples']:
                if code in actions:continue
                digits=(code//(10**np.arange(scene[1])))%10
                preference=options[digits][inverse];one=np.ones((scene[1],1),dtype=np.float32)
                actions[code]=np.concatenate((source,-one,span*one,-2*one,np.zeros((scene[1],2)),
                    span*one,preference,preference),-1).astype(np.float32)
            keys=list(example['samples']);random_state=np.random.get_state()
            jobs=[(str(policy),dict(environment=env,observation=obs,random_state=random_state,action=actions[code]),
                scene,future,span) for code,future in keys]
            outcomes=dict(zip(keys,pool.map(r.source_successor_branch,jobs,chunksize=1)
                if pool is not None else map(r.source_successor_branch,jobs)))
            steps+=sum(row['physical_steps'] for row in outcomes.values())
            def target(row):
                return dict(success=row['success'],terminal=row['terminal'],reward=row['task_return'],
                    discount=row['task_discount'],next=None,first_step=dict(next=row['response']['packet'],
                    terminal=row['response']['terminal'],steps=row['response']['steps']))
            branches=[]
            for left,right in example['pairs']:
                if left[1]!=right[1]:raise ValueError('A paired branch must use one future seed.')
                factual=outcomes.get((0,left[1]));factual_target=target(factual) if factual is not None else None
                branches.append(dict(packet=packet,left=actions[left[0]],right=actions[right[0]],
                    before=target(outcomes[left]),after=target(outcomes[right]),reference_action=actions.get(0),
                    reference_outcome=factual_target,factual_outcome=factual_target,observed_future=False,
                    continuation_seed=left[1],root=(scene[0],elapsed,left[0],right[0])))
            factual=[row for (code,_),row in outcomes.items() if code==0]
            rollouts.append(dict(row={k:v for k,v in factual[0].items() if k!='response'},steps=[],branches=branches,
                factual_success=float(np.mean([row['success'] for row in factual])),subjects=list(range(scene[1]))))
            reached.append(elapsed)
            print(dict(phase='replacement_m2_data',policy=Path(policy).name,root=index+1,roots=len(reference),
                outcomes=len(outcomes),prefix_steps=elapsed),flush=True)
    model.update(fixed_intervention_clock=True,physical_interventions=True,physical_intervention_span=span,
        physical_intervention_reversal=False,physical_span=span,physical_reversal=False,gate_period=1,
        branch_value_data=True,success_objective=True)
    data=dict(model=model,rollouts=rollouts,span=span,program_options=blueprint['program_options'],
        include_all_vessels=blueprint.get('include_all_vessels',False),completed=True,
        policy_iteration=model.get('successor_task_value') is not None,initial_only=initial,
        simulation_steps=steps,root_tasks=root_tasks,root_steps=reached,
        root_design=root_design,root_sampling='matched_failure_class_and_reference_lookback_v1',
        candidate_programs=[sorted({code for code,_ in example['samples']}) for example in reference],
        use='own-controller terminal resampling with identical reference program/future design')
    actual=c.successor_task_examples(data,absolute_frame=True)
    expected=sum(len(e['samples']) for e in reference)
    observed=sum(len(e['samples']) for e in actual)
    if expected!=observed:raise ValueError(f'Replacement sample budget differs: {observed} versus {expected}.')
    if [set(e['samples']) for e in actual]!=[set(e['samples']) for e in reference]:
        raise ValueError('Replacement observations must retain the planned programs and future seeds.')
    torch.save(data,output);return data


def retrain_m2_on_fusion_control(base,blueprints,tasks,directory,prefix,workers=3,base_results=None,reference=None,restart_from=None,stop_after=None):
    from concurrent.futures import ProcessPoolExecutor
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    r.initialize_source_branches(str(d.SOURCE))
    if reference is None:raise ValueError('Provide the reference M2 recipe for matched replacement training.')
    reference_path=Path(reference);reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    reference_model=torch.load(reference_path,weights_only=False,map_location='cpu')
    guides=[];guide=reference_model['successor_task_value']
    while guide is not None:guides.append(guide);guide=guide.get('prior')
    guides.reverse()
    if len(guides)!=len(blueprints):raise ValueError('Match every reference value stage.')
    corrected=reference_model.get('successor_task_fit',{}).get('refitted_current_stage',{})
    directory=Path(directory);previous=Path(base);data_paths=[];total=0
    for index,blueprint in enumerate(blueprints):
        guide=guides[index]
        root_design=intervention_root_design(blueprint)
        restart=restart_from is not None and index+1>=restart_from
        if int(guide['span'])!=int(blueprint['span']):raise ValueError('Match reference intervention windows.')
        policy=directory/f'{prefix}-stage-{index+1}.pth'
        data_path=directory/f'{prefix}-stage-{index+1}-training-data.pth';data_paths.append(str(data_path))
        if policy.exists() and not restart:
            prior_fit=torch.load(policy,weights_only=False,map_location='cpu').get('replacement_retraining',{})
            if prior_fit.get('reference_sha256')!=reference_sha256:
                raise ValueError('Existing replacement stages were not fitted with this reference recipe.')
            total=prior_fit.get('new_terminal_observations',total)
            previous=policy
            if stop_after is not None and index+1>=stop_after:return previous
            continue
        if data_path.exists() and not restart:
            data=torch.load(data_path,weights_only=False,map_location='cpu')
            if data.get('candidate_programs') is None:
                observed=c.successor_task_examples(data,absolute_frame=True)
                planned=c.successor_task_examples(blueprint,absolute_frame=True)
                if [set(e['samples']) for e in observed]!=[set(e['samples']) for e in planned]:
                    raise ValueError('Saved replacement data differs from the planned intervention design.')
                data['candidate_programs']=[sorted({code for code,_ in e['samples']}) for e in observed]
                torch.save(data,data_path)
        else:
            # Full development futures select the final recovery roots; the
            # preceding stages use the same 64-layout development selection.
            selection=tasks if index in (0,len(blueprints)-1) else [task for task in tasks if task[1] is None]
            result_path=(Path(base_results) if index==0 and base_results is not None else
                previous.with_name(previous.stem+'-source-development.pth'))
            existing=torch.load(result_path,weights_only=False,map_location='cpu') if result_path.exists() else None
            if (existing and existing.get('completed') and existing.get('tasks')==selection
                    and existing.get('policy_sha256')==hashlib.sha256(previous.read_bytes()).hexdigest()):
                rows=existing['rows']
            else:
                jobs=[(scene,str(previous),previous.stat().st_mtime_ns,False,0,0,0,0,future) for scene,future in selection]
                if workers==1:rows=list(map(d.joint_episode,jobs))
                else:
                    with ProcessPoolExecutor(max_workers=workers,initializer=r.initialize_source_branches,initargs=(str(d.SOURCE),)) as pool:
                        rows=list(pool.map(d.joint_episode,jobs,chunksize=1))
            needed=len(c.successor_task_examples(blueprint,absolute_frame=True))
            failures={scene[0]:0 for scene,_ in selection}
            for (scene,_),row in zip(selection,rows):failures[scene[0]]+=1-row['success']
            ranked=sorted(zip(selection,rows),key=lambda pair:(pair[1]['success'],-failures[pair[0][0][0]],pair[0][1] is not None,pair[0][0][0]))
            chosen=[];seen=set()
            for plan in root_design:
                available=[pair for pair in ranked if pair[0][0][0] not in seen]
                matching=[pair for pair in available if pair[1]['outcome']==plan['outcome']]
                task,row=(matching or available)[0]
                seen.add(task[0][0]);chosen.append(task)
            if len(chosen)!=needed:raise ValueError('Provide one own-policy root per reference intervention design.')
            data=recollect_blueprint(previous,blueprint,chosen,data_path,initial=guide['initial_only'],workers=workers)
        wrappers=[];potential=guide['program_model']
        while potential.get('kind')=='calibrated_task_potential':
            wrappers.append(potential);potential=potential['base']
        neural=potential.get('kind')=='paired_successor_task'
        stage_fit=corrected.get('fit',{}) if corrected.get('index')==index+1 else {}
        model=c.fit_successor_task_policy(data,policy,ordinary=False,updates=5000,seed=42,device='cuda',
            bandwidth=guide['bandwidth'],margin=guide['margin'],retain_predecessor=index>0,paired_ordering=not neural,
            paired_calibration=bool(wrappers),task_width=potential.get('width') if neural else None,
            response_width=potential.get('response_width',24),support_weighted=guide.get('support_weighted',False),
            decision_weight=stage_fit.get('decision_weight',2.) if neural else 1.,
            difference_weight=stage_fit.get('terminal_difference_weight',.5) if neural else 10.,absolute_context=True)
        current=model['successor_task_value'];examples=c.successor_task_examples(data,absolute_frame=True)
        for key in ('explicit_wait','time_bandwidth','support_weighted','confidence_scale'):
            current[key]=guide.get(key,False if key in ('explicit_wait','support_weighted') else None if key=='time_bandwidth' else 0.)
        # The vocabulary is part of the fixed experimental design. Its
        # support is rebuilt exclusively from this controller's new outcomes.
        current['candidate_codes']=list(guide['candidate_codes'])
        current['candidate_support']=np.asarray([[q in {sample[0] for sample in example['samples'].values()}
            for q in current['candidate_codes']] for example in examples],dtype=bool)
        for wrapper in reversed(wrappers[:-1]):
            shrink=corrected.get('calibration',{}).get('covariance_shrink',.5)
            estimates,calibration=c.paired_block_task_estimates(examples,covariance_shrink=shrink)
            _,current['program_model']=c.calibrate_program_potential(current['program_model'],estimates,
                bandwidth=wrapper['bandwidth'],time_bandwidth=wrapper.get('time_bandwidth'))
            model['successor_task_fit']['additional_calibration']=calibration
        total+=sum(len(e['samples']) for e in c.successor_task_examples(data,absolute_frame=True))
        model['replacement_retraining']=dict(base=str(base),completed_stages=index+1,
            training_data=list(data_paths),new_terminal_observations=total,neural_updates=5000*(1+int(index>=2)),
            reference=str(reference_path),reference_sha256=reference_sha256,
            inherited_reference_values=False,inherited_reference_support=False,
            root_sampling=data.get('root_sampling'),root_design=data.get('root_design'))
        torch.save(model,policy);previous=policy
        print(dict(phase='replacement_m2_stage_complete',stage=index+1,policy=str(policy),observations=total),flush=True)
        if stop_after is not None and index+1>=stop_after:return previous
    evaluate_models([previous],tasks,workers=workers)
    return previous


def collect_temporal_option_tasks(policy,root_tasks,output,*,codes=None,spans=(10,40,200),
        delays=(0,5,20),repetitions=8,future_start=262144,workers=3,response_tick=20,
        future_pool=64,anchor_count=2,existing_data=None,executed_context=False,root_stride=1,fleet_programs=False):
    """Sample actual delayed programs, keeping one terminal record per future.

    Root tasks are (scene, prefix_future, lookback), with None lookback denoting
    the initial state. The same recipe applies with and without joint fusion.
    Factual futures are actual predecessor continuations, not successful
    intervention trajectories. They are offline conditioning data only.
    """
    from concurrent.futures import ProcessPoolExecutor
    import conditional_fusion_value as c
    import develop_source_fusion as d
    import run_remote_evidence as r
    r.initialize_source_branches(str(d.SOURCE))
    policy=Path(policy);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    model=torch.load(policy,map_location='cpu',weights_only=False)
    actor=d.joint_load(str(policy),policy.stat().st_mtime_ns);roots=[]
    steps=0 if existing_data is None else existing_data['simulation_steps']
    existing_roots={} if existing_data is None else {
        (tuple(root['scene']),root['prefix_future'],root['lookback']):root for root in existing_data['roots']}
    coefficients=np.asarray([[-1.,-1.]]+[[left,right] for left in (0.,.5,1.) for right in (0.,.5,1.)])
    with (ProcessPoolExecutor(max_workers=workers,initializer=r.initialize_source_branches,initargs=(str(d.SOURCE),))
            if workers>1 else nullcontext(None)) as pool:
        for scene,prefix_future,lookback in root_tasks:
            np.random.seed(scene[0]);torch.manual_seed(scene[0])
            env=d.numerical_source().TADEnv(scene[1]);obs,_=env.reset(agility=scene[2],noisy_agility=False)
            if prefix_future is not None:np.random.seed(prefix_future)
            elapsed=0
            if lookback is not None:
                history=[];done=0
                while not done:
                    history.append((elapsed,copy.deepcopy(env),obs.copy(),np.random.get_state()))
                    with torch.no_grad():action=actor(d.mechanism_tensor(d.joint_packet(env,obs)),deterministic=True)[0][0].numpy()
                    obs,_,done,_,count=d.joint_advance(env,obs,actor,action);elapsed+=count;steps+=count
                cutoff=max(0,elapsed-lookback)
                cutoff=(cutoff//root_stride)*root_stride
                elapsed,env,obs,random_state=max((item for item in history if item[0]<=cutoff),key=lambda item:item[0])
                np.random.set_state(random_state)
            packet=d.joint_packet(env,obs);tensor=d.mechanism_tensor(packet)
            with torch.no_grad():
                reference=actor(tensor,deterministic=True)[0][0].numpy()
                control=actor(tensor,deterministic=True,gate_only=True)[0][0].numpy()[:,:3]
            context,order=c.intervention_task_context(tensor,True,True);inverse=order[0].argsort().numpy()
            n=scene[1]
            if codes is None:
                # The common intervention library includes all nine per-boat
                # actuator mixtures and a factorial closest-pair background.
                # Its geometric construction invokes no M1 prediction/solver.
                programs=c.paired_effect_programs(tensor,order,True)[0] if fleet_programs else c.successor_candidate_programs(tensor,order,True,10)[0]
                selected=sorted(set((programs*(10**torch.arange(n))).sum(-1).tolist())-{0})
            else:selected=sorted(set(codes)-{0})
            options=[(0,0,0)]+[(code,delay,span) for code in selected for span in spans for delay in delays]
            population=max(repetitions,future_pool)
            futures=[int(np.random.SeedSequence([scene[0],elapsed,future_start,i,39119]).generate_state(1)[0]) for i in range(population)]
            previous=existing_roots.get((tuple(scene),prefix_future,lookback))
            outcomes={}
            if previous is not None:
                option_positions={option:index for index,option in enumerate(options)}
                if (not all(tuple(option) in option_positions for option in previous['options'])
                        or not np.allclose(previous['context'],context[0].numpy(),atol=1e-6)):
                    raise ValueError('Existing terminal observations require the same predecessor root and option library.')
                futures=(previous['futures']+[f for f in futures if f not in previous['futures']])[:max(population,len(previous['futures']))]
                population=len(futures)
                outcomes={(option_positions[tuple(previous['options'][q])],f):copy.deepcopy(value)
                    for (q,f),value in previous['outcomes'].items()}
                if (existing_data.get('recipe',{}).get('wait_continuation')!='predecessor'
                        and actor.successor_task_value is not None):
                    outcomes={key:value for key,value in outcomes.items() if options[key[0]][1]==0}
            one=np.ones((n,1),np.float32);actions=[reference]
            for code,delay,span in options[1:]:
                preference=coefficients[(code//(10**np.arange(n)))%10][inverse].astype(np.float32)
                first=preference if delay==0 else np.full_like(preference,-1.)
                actions.append(np.concatenate((control,-one,(delay+span)*one,-2*one,
                    np.zeros((n,2),np.float32),delay*one,first,preference,np.zeros_like(one)),-1))
            random_state=np.random.get_state()
            # Every option gets the same branch budget, but different families
            # observe different shared-future blocks. The actual factual pool
            # is broader. Conditional standardization can consequently remove
            # finite-block disturbance imbalance instead of reconstructing an
            # already known complete-grid mean. Local one-vessel neighbours
            # share outcomes, without counting any observation twice.
            option_lookup={option:i for i,option in enumerate(options)}
            keys={(0,future) for future in range(population)}
            for option_index,(code,delay,span) in enumerate(options[1:],1):
                digits=(code//(10**np.arange(n)))%10
                first=int(np.flatnonzero(digits)[0]);block=first*repetitions
                block_futures=[(block+j)%population for j in range(repetitions)]
                neighbours=[option_index]
                for vessel in np.flatnonzero(digits):
                    parent=code-int(digits[vessel])*10**int(vessel)
                    parent_index=option_lookup.get((parent,delay,span))
                    if parent_index is not None:neighbours.append(parent_index)
                keys.update((index,future) for index in neighbours for future in block_futures)
            def sample(missing,refresh_response=False):
                nonlocal steps
                missing=sorted(set(missing) if refresh_response else set(missing)-outcomes.keys())
                jobs=[(str(policy),dict(environment=env,observation=obs,random_state=random_state,action=actions[q]),
                    scene,futures[f],response_tick) for q,f in missing]
                iterator=pool.map(r.source_successor_branch,jobs,chunksize=1) if pool is not None else map(r.source_successor_branch,jobs)
                for key,row in zip(missing,iterator):
                    following={k:torch.as_tensor(v[None]) for k,v in row['response']['packet'].items()}
                    after,after_order=c.intervention_task_context(following,True,True)
                    nodes=after[0,:8*n].reshape(n,8)[after_order[0].argsort()][order[0]]
                    if refresh_response and key in outcomes:
                        old=outcomes[key]
                        if old['success']!=float(row['success']) or old['outcome']!=row['outcome'] or abs(old['duration']-row['duration'])>1e-5:
                            raise ValueError('Response-only resampling changed the terminal task outcome.')
                    outcomes[key]=dict(success=float(row['success']),following=torch.cat((nodes.flatten(),after[0,8*n:])).numpy(),
                        response_steps=row['response']['steps'],response_sampling='physical_tick',
                        duration=row['duration'],outcome=row['outcome'])
                    steps+=row['physical_steps']
            sample(keys)
            # Informative task-fate anchors are observed on the entire actual
            # future population. Other options retain their sparse paired
            # blocks. Anchor observations enter the joint conditioning
            # distribution. Their own prediction masks that anchor's complete
            # response, so no outcome is supplied as its own conditioning input.
            candidates=[]
            for q,(code,delay,span) in enumerate(options[1:],1):
                ys=[item['success'] for (candidate,_),item in outcomes.items() if candidate==q]
                if ys:
                    p=float(np.mean(ys));candidates.append((p*(1.-p),p,-delay,-span,-q,q,code))
            candidates.sort(reverse=True);anchors=[];anchor_codes=set()
            for *_,q,code in candidates:
                if not anchor_count:break
                if code in anchor_codes:continue
                anchors.append(q);anchor_codes.add(code)
                if len(anchors)>=anchor_count:break
            if previous is not None and len(previous.get('anchors',()))==anchor_count:
                anchors=list(previous['anchors'])
            sample((q,f) for q in anchors for f in range(population))
            conditioning=[(q,f) for q in (0,*anchors) for f in range(population)]
            stale=[key for key in conditioning if outcomes[key].get('response_sampling')!='physical_tick']
            sample(stale,refresh_response=True)
            roots.append(dict(context=context[0].numpy(),options=options,futures=futures,outcomes=outcomes,
                controls=np.concatenate((control[:,:2],packet['node'][:,11:13],np.repeat(control[:,2:3],2,axis=-1)),axis=-1)[order[0].numpy()],
                scene=scene,prefix_future=prefix_future,root_steps=elapsed,lookback=lookback,anchors=anchors))
            if executed_context:
                executed=c.paired_effect_controls(actor,tensor,torch.as_tensor(control[None],dtype=torch.float32))
                roots[-1]['effect_controls']=executed[0,order[0]].numpy()
            data=dict(model=model,roots=roots,completed=len(roots)==len(root_tasks),simulation_steps=steps,
                predecessor_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                recipe=dict(spans=spans,delays=delays,repetitions=repetitions,response_tick=response_tick,
                    future_population=population,future_start=future_start,anchor_count=anchor_count,
                    executed_context=executed_context,root_stride=root_stride,fleet_programs=fleet_programs,
                    response_sampling='physical_tick',wait_continuation='predecessor',
                    paired_blocks='single-vessel neighbours and temporal alternatives'),
                terminal_target='true source task success',paired_future_labels='observed predecessor motion and task fate')
            torch.save(data,output)
            print(dict(phase='temporal_option_root',scene=scene[0],root_steps=elapsed,
                options=len(options),observations=len(outcomes),simulation_steps=steps),flush=True)
    return data


def paired_effect_root_tasks(reference,root_count=32,seed=42):
    """One balanced own-controller root rule for both factorial arms."""
    rng=np.random.default_rng(810000+seed);records=list(zip(reference['tasks'],reference['rows']))
    rng.shuffle(records);chosen=[];seen=set()
    for index,category in enumerate((2,1,2,1,3,3,3,3)):
        available=[item for item in records if item[0][0][0] not in seen]
        balanced=[item for item in available if bool(item[1]['success'])==(category==3)] or available
        matching=[item for item in balanced if item[1]['outcome']==category] or balanced
        agility=2. if index%2==0 else 2.5
        matching=[item for item in matching if item[0][0][2]==agility] or matching
        if not matching:break
        (scene,future),_=matching[0];seen.add(scene[0]);chosen.append((scene,future,None if index%2==0 else 20))
        if len(chosen)==root_count:return chosen
    rng=np.random.default_rng(820000+seed);records=list(zip(reference['tasks'],reference['rows']));rng.shuffle(records)
    while len(chosen)<root_count:
        index=len(chosen);success=index%4>=2
        available=[item for item in records if item[0][0][0] not in seen]
        if not available:raise ValueError('Provide enough distinct development layouts for the common root budget.')
        balanced=[item for item in available if bool(item[1]['success'])==success] or available
        agility=2. if index%2==0 else 2.5
        balanced=[item for item in balanced if item[0][0][2]==agility] or balanced
        (scene,future),_=balanced[0];seen.add(scene[0]);chosen.append((scene,future,5 if index%3==2 else None))
    return chosen


def train_shared_paired_effect_control(base,tasks,directory,prefix,*,root_count=32,updates=5000,
        seed=42,workers=1,response_features=False,peer_conditioned=True,use_time=True,training_data=None):
    """Repeat the same paired-effect learning rule on an own-policy base."""
    import conditional_fusion_value as c
    base=Path(base);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    evaluate_models([base],tasks,workers=workers)
    reference=torch.load(base.with_name(base.stem+'-source-development.pth'),map_location='cpu',weights_only=False)
    chosen=paired_effect_root_tasks(reference,root_count,seed)
    path=Path(training_data) if training_data is not None else directory/f'{prefix}-training-data.pth'
    data=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else None
    identity=hashlib.sha256(base.read_bytes()).hexdigest()
    if data is not None and data.get('predecessor_sha256')!=identity:
        raise ValueError('The supplied terminal data belongs to a different executing predecessor.')
    actual=[] if data is None else [(tuple(r['scene']),r['prefix_future'],r['lookback']) for r in data['roots']]
    if data is None or not data.get('completed') or actual!=chosen or not data['recipe'].get('fleet_programs'):
        data=collect_temporal_option_tasks(base,chosen,path,spans=(10,40,200),delays=(0,5),
            repetitions=2,future_pool=16,anchor_count=0,workers=workers,future_start=720000+seed,
            executed_context=True,root_stride=5,fleet_programs=True,existing_data=data)
    data['training_data']=str(path);torch.save(data,path)
    policy=directory/f'{prefix}.pth'
    model=c.fit_shared_paired_effect(data,policy,updates=updates,width=128,ensemble_size=3,
        seed=seed,device='cuda:0',response_features=response_features,balanced_likelihood=True,
        peer_conditioned=peer_conditioned,use_time=use_time)
    model['unified_training_recipe']=dict(root_count=root_count,root_selection='balanced_own_predecessor',
        updates_per_member=updates,ensemble_size=3,width=128,seed=seed,**data['recipe'])
    model['successor_task_fit']['simulation_steps']=data['simulation_steps'];torch.save(model,policy)
    evaluate_models([policy],tasks,workers=workers)
    return policy


def independent_source_initialization(template, source_run, seed, *, source_checkpoint='final'):
    """Instantiate an untrained analytic selector with an independent source actor."""
    import csv
    import develop_source_fusion as d
    from source_arboids import SourcePolicy, SOURCE_COMMIT

    source_run = Path(source_run)
    record = torch.load(source_run/'source-repeat-result.pth', map_location='cpu', weights_only=False)
    if source_checkpoint not in ('final', 'training_development'):
        raise ValueError('Choose the recorded final or training-development source checkpoint.')
    selected = source_checkpoint == 'training_development'
    if (record.get('seed') != seed or not record.get('completed')
            or not record.get('fresh_initialization') or record.get('source_commit') != SOURCE_COMMIT
            or record.get('family', 'arboids') != 'arboids'
            or not selected and record.get('training_defenders', 3) != 3):
        raise ValueError('The selector requires a completed independent original-source ARBoids run.')
    checkpoint = source_run/('best-development.pth' if selected else 'adares1.pth')
    identity = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    configuration = hashlib.sha256((source_run/'config.yaml').read_bytes()).hexdigest()
    checkpoint_key = 'selected_checkpoint_sha256' if selected else 'checkpoint_sha256'
    if identity != record[checkpoint_key] or configuration != record['configuration_sha256']:
        raise ValueError('The source checkpoint or its training configuration has changed.')
    with (source_run/'metrics1.csv').open(newline='') as stream:
        metrics = list(csv.DictReader(stream))
    if not metrics or int(metrics[-1]['step']) < record['training_steps']:
        raise ValueError('Independent source training has not reached its recorded budget.')
    if (template.get('iteration', 0) != 0 or template.get('simulation_steps', 0) != 0
            or d.joint_value_objectives(template) or template.get('generic_fusion_adapter')):
        raise ValueError('Replace the source only in a fresh selector, then recollect its own learning data.')
    if selected:
        _, config = validate_author_source_repeat(source_run)
        weights = AuthorSourceRepeatPolicy(checkpoint, config).actor.state_dict()
    else:
        weights = SourcePolicy(checkpoint).actor.state_dict()
    old = {key[5:]: value for key, value in template['actor'].items() if key.startswith('base.')}
    if old.keys() != weights.keys() or any(old[key].shape != value.shape for key, value in weights.items()):
        raise ValueError('The independent source architecture differs from the selector proposal.')
    model = copy.deepcopy(template)
    model['actor'].update({'base.'+key: value.detach().cpu().clone() for key, value in weights.items()})
    for key in ('actor_optimizer', 'value_optimizer', 'values'):
        model.pop(key, None)
    model.update(seed=seed, source_sha256=identity, history=[],
        proposal_initialization_sha256=d.proposal_parameter_digest(weights),
        initialization='independently trained original source actor; untrained analytic fusion selector',
        independent_source=dict(seed=seed, checkpoint=str(checkpoint), checkpoint_sha256=identity,
            configuration_sha256=configuration, training_steps=record['training_steps'],
            fresh_initialization=True, completed=True, source_commit=SOURCE_COMMIT,
            training_defenders=record.get('training_defenders', 3), source_checkpoint=source_checkpoint,
            repeat_scope='independent source learning; subsequent M2 requires own-policy recollection'))
    return model


def run_author_source_repeat(seed,directory,device='cuda:0',*,defenders=3,family='arboids'):
    """Independently repeat the published source training configuration."""
    import sys
    import train as training
    from source_arboids import numerical_source,SOURCE_ROOT,SOURCE_COMMIT
    from utils.manager import ExperimentManager,set_seed
    from utils.config import load_config
    if family not in ('arboids','rp','sac') or defenders < 2:
        raise ValueError('Choose an original source algorithm and a valid fleet size.')
    directory=Path(directory);result=directory/'source-repeat-result.pth'
    if result.exists():
        recorded=torch.load(result,map_location='cpu',weights_only=False)
        if (recorded['seed']!=seed or recorded.get('family','arboids')!=family
                or recorded.get('training_defenders',3)!=defenders):
            raise ValueError('The source repeat belongs to a different training specification.')
        return recorded
    if (directory/'adares1.pth').exists():
        raise ValueError('A partial actor-only source run cannot be treated as resumed independent training.')
    source=numerical_source()
    training.SAC=source.SAC;training.TADEnv=source.TADEnv
    training.ReplayBuffer=sys.modules[source.SAC.__module__].ReplayBuffer
    torch.set_num_threads(1);torch.set_grad_enabled(True);set_seed(seed)
    config=load_config(str(SOURCE_ROOT/'train/configs/train.yaml'))
    config.agent.defender_num=defenders
    config.agent.adaptive=family=='arboids';config.agent.residual=family!='sac'
    config.training.seed=seed
    class SourceRepeatManager(ExperimentManager):
        def record_metrics(self,**metrics):
            super().record_metrics(**metrics);self.latest=metrics

        def save_model(self,model,filename=None,step=None):
            super().save_model(model,filename,step)
            score=(self.latest['def_sr'],self.latest['reward'])
            if score>getattr(self,'best',(-1.,-float('inf'))):
                self.best=score;self.selection=dict(self.latest)
                model.save(str(Path(self.exp_dir)/'best-development.pth'))
    experiment=SourceRepeatManager(config,base_dir=str(directory.parent),run_id=directory.name,repeat_idx=1)
    training.main(config,experiment,torch.device(device))
    checkpoint=directory/'adares1.pth'
    recorded=dict(seed=seed,family=family,training_defenders=defenders,
        source_commit=SOURCE_COMMIT,training_steps=config.training.total_steps,
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        configuration_sha256=hashlib.sha256((directory/'config.yaml').read_bytes()).hexdigest(),
        selected_checkpoint=str(directory/'best-development.pth'),
        selected_checkpoint_sha256=hashlib.sha256((directory/'best-development.pth').read_bytes()).hexdigest(),
        development_selection=experiment.selection,
        fresh_initialization=True,completed=True)
    torch.save(recorded,result)
    return recorded


class AuthorSourceRepeatPolicy:
    """Pinned source actor with the same per-vessel inference as the controls."""
    def __init__(self, checkpoint, config):
        import sys
        from source_arboids import numerical_source
        source = numerical_source()
        agent = config['agent']
        environment = source.TADEnv(agent['defender_num'], agent['boid_state'], agent['form_reward'])
        adaptive = bool(agent['adaptive'])
        constructor = source.ActorAdap if adaptive else sys.modules[source.SAC.__module__].ActorSAC
        self.actor = constructor(environment.feature1_dim, environment.feature2_dim,
                                 environment.action_dim+int(adaptive), config['rl']['hidden_dim'])
        self.actor.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.actor.eval()
        self.device = torch.device('cpu')
        self.choose_action = source.SAC.choose_action.__get__(self)
        self.controller = 'RL' if not agent['residual'] else ('AdaRes' if adaptive else 'Res')
        self.environment_options = (agent['boid_state'], agent['form_reward'])

    def __call__(self, observations):
        return np.stack([self.choose_action(row[None], deterministic=True) for row in observations])


def validate_author_source_repeat(directory):
    """Verify the completed budget and the predeclared training checkpoint choice."""
    import csv
    import yaml
    from source_arboids import SOURCE_ROOT, SOURCE_COMMIT
    directory = Path(directory)
    record = torch.load(directory/'source-repeat-result.pth', map_location='cpu', weights_only=False)
    family = record.get('family', 'arboids')
    if (family not in ('arboids', 'rp', 'sac') or not record.get('completed')
            or not record.get('fresh_initialization') or record.get('source_commit') != SOURCE_COMMIT):
        raise ValueError('Baseline evaluation requires a completed independent pinned-source run.')
    config_path = directory/'config.yaml'
    config = yaml.safe_load(config_path.read_text())
    expected = yaml.safe_load((SOURCE_ROOT/'train/configs/train.yaml').read_text())
    expected['agent'].update(defender_num=record.get('training_defenders', 3),
                             adaptive=family == 'arboids', residual=family != 'sac')
    # Earlier repeats set the recorded seed directly before constructing the
    # experiment manager; later repeats additionally serialize it in YAML.
    if 'seed' in config['training']:
        expected['training']['seed'] = record['seed']
    if (config != expected or record['training_steps'] != expected['training']['total_steps']
            or hashlib.sha256(config_path.read_bytes()).hexdigest() != record['configuration_sha256']):
        raise ValueError('The baseline configuration differs from its full-budget source recipe.')
    for name, key in [('adares1.pth', 'checkpoint_sha256'),
                      ('best-development.pth', 'selected_checkpoint_sha256')]:
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != record.get(key):
            raise ValueError('The completed baseline checkpoint has changed.')
    with (directory/'metrics1.csv').open(newline='') as stream:
        metrics = list(csv.DictReader(stream))
    if not metrics or int(metrics[-1]['step']) != record['training_steps']:
        raise ValueError('The baseline training stream did not reach the recorded budget.')
    # The manager saves the first lexicographic maximum of success and reward.
    best = max(metrics, key=lambda row: (float(row['def_sr']), float(row['reward'])))
    selected = record['development_selection']
    if (int(best['step']) != selected['step'] or any(not np.isclose(float(best[key]), selected[key],
            atol=1e-12, rtol=1e-12) for key in ('def_sr', 'reward'))):
        raise ValueError('The selected baseline is not the recorded training-development maximum.')
    return record, config


_AUTHOR_REPEAT_POLICIES = {}


def author_source_repeat_episode(task):
    """Use the frozen source dynamics and the common layout/future seed pair."""
    import develop_source_fusion as d
    scene, future, directory, digest = task
    key = (directory, digest)
    if key not in _AUTHOR_REPEAT_POLICIES:
        record, config = validate_author_source_repeat(directory)
        if record['selected_checkpoint_sha256'] != digest:
            raise ValueError('Baseline evaluation is bound to another checkpoint.')
        _AUTHOR_REPEAT_POLICIES[key] = AuthorSourceRepeatPolicy(Path(directory)/'best-development.pth', config)
    policy = _AUTHOR_REPEAT_POLICIES[key]
    seed, defenders, agility = scene
    np.random.seed(int(seed)); torch.manual_seed(int(seed))
    env = d.numerical_source().TADEnv(defenders, *policy.environment_options)
    observations, _ = env.reset(agility=agility, noisy_agility=False)
    if future is not None:
        np.random.seed(int(future))
    done = 0; total = 0.; discount = 1.; steps = 0
    while not done:
        observations, reward, done, _ = env.step(policy(observations), policy.controller)
        total += discount*float(np.mean(reward)); discount *= .99; steps += 1
    row = dict(seed=int(seed), defenders=defenders, agility=agility, **d.outcome(env, done),
               discounted_return=total, environment_protocol='source', conditional_interventions=0,
               physical_steps=steps)
    if future is not None:
        row['future_seed'] = int(future)
    return row


def evaluate_author_source_repeat(directory, tasks, *, workers=2, result_suffix='source-development',
                                  evaluation_context=None):
    """Pair a fully trained ARBoids, RP or SAC endpoint with the mechanism suites."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import run_remote_evidence as r
    directory = Path(directory)
    source, _ = validate_author_source_repeat(directory)
    digest = source['selected_checkpoint_sha256']
    path = directory/(result_suffix+'.pth')
    previous = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
    if previous is not None:
        if (previous['policy_sha256'] != digest or previous['tasks'] != tasks
                or previous.get('evaluation_context') != evaluation_context):
            raise ValueError('A resumed baseline evaluation must retain its checkpoint and paired tasks.')
        if previous['completed']:
            return previous
    rows = previous['rows'] if previous else [None]*len(tasks)
    start = time.monotonic()
    with (ProcessPoolExecutor(max_workers=workers, initializer=torch.set_num_threads, initargs=(1,))
          if workers > 1 else nullcontext(None)) as pool:
        jobs = [(index, (scene, future, str(directory), digest))
                for index, (scene, future) in enumerate(tasks) if rows[index] is None]
        if pool is None:
            results = ((index, author_source_repeat_episode(task)) for index, task in jobs)
        else:
            pending = {pool.submit(author_source_repeat_episode, task): index for index, task in jobs}
            results = ((pending[job], job.result()) for job in as_completed(pending))
        for index, row in results:
            rows[index] = row; completed = [row for row in rows if row is not None]
            if len(completed) % 32 == 0 or len(completed) == len(tasks):
                record = dict(policy=str(directory/'best-development.pth'), policy_sha256=digest,
                    tasks=tasks, rows=rows, evaluated=len(completed), success=sum(row['success'] for row in completed),
                    completed=len(completed) == len(tasks), source_training=source,
                    evaluation_context=evaluation_context, elapsed_seconds=time.monotonic()-start,
                    inference_rule='pinned source deterministic per-vessel calls shared by the mechanism controls')
                r.atomic_save(record, path)
                print(dict(phase='fully_trained_source_evaluation', family=source.get('family', 'arboids'),
                    seed=source['seed'], suite=result_suffix, evaluated=len(completed), success=record['success']), flush=True)
    return record


def train_unified_option_control(base,tasks,directory,prefix,*,rounds=6,roots_per_round=4,
        updates=3000,seed=42,workers=1,paired=True,peer_conditioned=True,use_time=True):
    """One policy-iteration rule for F01, F11 and internal learning deletions.

    All arms receive the same duration/delay vocabulary, root-selection rule,
    future population, optimizer budget and development acceptance rule. Only
    the source controller and explicitly removed learning information differ.
    Each accepted predecessor generates its own subsequent terminal labels.
    """
    import conditional_fusion_value as c
    from collections import Counter
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    previous=Path(base)
    recipe=dict(rounds=rounds,roots_per_round=roots_per_round,spans=(10,40,200),delays=(0,5,20),
        repetitions=2,future_pool=32,anchor_count=2,updates=updates,seed=seed,root_lookbacks=(None,40,120),
        optimizer='AdamW',learning_rate=8e-4,width=48,rank=4,bandwidth=.15,time_bandwidth=.01,
        response_sampling='physical_tick',wait_continuation='predecessor',
        terminal_target='source task success',acceptance='paired development success count strictly improves')
    history=[]
    for stage in range(rounds):
        evaluate_models([previous],tasks,workers=workers)
        reference=torch.load(previous.with_name(previous.stem+'-source-development.pth'),map_location='cpu',weights_only=False)
        failure_counts=Counter(scene[0] for (scene,future),row in zip(tasks,reference['rows']) if not row['success'])
        ranked=sorted(zip(tasks,reference['rows']),key=lambda item:(item[1]['success'],
            -failure_counts[item[0][0][0]],item[0][1] is not None,item[0][0][0]))
        chosen=[];seen=set();lookback=recipe['root_lookbacks'][stage%3]
        for category in (2,1,2,1):
            available=[item for item in ranked if item[0][0][0] not in seen and not item[1]['success']]
            if not available:break
            item=next((item for item in available if item[1]['outcome']==category),available[0])
            (scene,future),_=item;chosen.append((scene,future,lookback));seen.add(scene[0])
            if len(chosen)>=roots_per_round:break
        if not chosen:break
        data_path=directory/f'{prefix}-round-{stage+1}-training-data.pth'
        data=(torch.load(data_path,map_location='cpu',weights_only=False) if data_path.exists() else None)
        identity=hashlib.sha256(previous.read_bytes()).hexdigest()
        if (not data or not data.get('completed') or data.get('predecessor_sha256')!=identity
                or data.get('recipe',{}).get('response_sampling')!='physical_tick'
                or data.get('recipe',{}).get('wait_continuation')!='predecessor'
                or any(len(root.get('anchors',()))!=recipe['anchor_count'] for root in data['roots'])):
            existing=data if data and data.get('predecessor_sha256') in (None,identity) else None
            data=collect_temporal_option_tasks(previous,chosen,data_path,spans=recipe['spans'],delays=recipe['delays'],
                repetitions=recipe['repetitions'],future_pool=recipe['future_pool'],workers=workers,
                future_start=262144+65536*stage+1024*seed,anchor_count=recipe['anchor_count'],existing_data=existing)
            data['predecessor_sha256']=identity;data['unified_training_recipe']=recipe;torch.save(data,data_path)
        candidate=directory/f'{prefix}-round-{stage+1}.pth'
        model=c.fit_paired_option_task_policy(data,candidate,updates=updates,seed=seed,device='cuda',
            paired=paired,peer_conditioned=peer_conditioned,use_time=use_time)
        model['unified_training_recipe']=recipe;model['unified_training_history']=history
        model['successor_task_fit'].update(training_data=str(data_path),predecessor_sha256=identity,
            simulation_steps=data['simulation_steps'],training_round=stage+1)
        torch.save(model,candidate)
        evaluate_models([candidate],tasks,workers=workers)
        result=torch.load(candidate.with_name(candidate.stem+'-source-development.pth'),map_location='cpu',weights_only=False)
        repairs=sum(bool(new['success']) and not bool(old['success']) for old,new in zip(reference['rows'],result['rows']))
        losses=sum(bool(old['success']) and not bool(new['success']) for old,new in zip(reference['rows'],result['rows']))
        accepted=repairs>losses
        history.append(dict(round=stage+1,policy=str(candidate),success=result['success'],repairs=repairs,
            losses=losses,accepted=accepted,terminal_observations=sum(len(root['outcomes']) for root in data['roots']),
            simulation_steps=data['simulation_steps']))
        if accepted:previous=candidate
        print(dict(phase='unified_temporal_policy_iteration',prefix=prefix,**history[-1]),flush=True)
    return previous,history
