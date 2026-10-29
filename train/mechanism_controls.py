"""Matched M1 fusion controls and independently trained source baselines."""
import study_runtime  # Configure numerical runtimes before NumPy and PyTorch.
import copy
import hashlib
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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
    from actuator_fusion import fleet_task_context
    context,order=fleet_task_context(packet,True,True)
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
    import joint_fusion as d
    from actuator_fusion import fusion_controls
    scene,future,teacher_path,behavior_path,cap=task
    teacher=d.joint_load(str(teacher_path),Path(teacher_path).stat().st_mtime_ns)
    learner=(None if behavior_path is None else
        d.joint_load(str(behavior_path),Path(behavior_path).stat().st_mtime_ns))
    if not teacher.actuator_fusion_refinement:
        raise ValueError('Distillation requires a joint-fusion teacher without a value intervention.')
    if learner is not None and (learner.generic_fusion_adapter is None
            or learner.actuator_fusion_refinement):
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
            factual=fusion_controls(teacher,packet,action)[...,4:6]
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
    import joint_fusion as r
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
    if workers==1:r.initialize()
    with (ProcessPoolExecutor(max_workers=workers,initializer=r.initialize)
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
    import joint_fusion as d
    if updates < 1:raise ValueError('Distillation requires a positive update budget.')
    if d.retired_value_fields(base):raise ValueError('Distillation starts without learned intervention values.')
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
    import joint_fusion as d
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
    import joint_fusion as r
    if budget<1 or workers<1 or not tasks:raise ValueError('Empty generic replay design.')
    used=0;episode=0;features=[];choices=[];targets=[];censored=0
    if workers==1:r.initialize()
    with (ProcessPoolExecutor(max_workers=workers,initializer=r.initialize)
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
    if not features:
        raise ValueError('No episode completed within the replay budget; censored trajectories have no return labels.')
    result=dict(features=np.concatenate(features),choices=np.concatenate(choices),targets=np.asarray(targets,dtype=np.float32),
        simulation_steps=used,censored_steps=censored,episodes=episode,base=str(base),
        base_sha256=hashlib.sha256(Path(base).read_bytes()).hexdigest(),completed=True,
        source='shared source-Adapter behavior with independent exploration; no analytic-selector teacher')
    r.atomic_save(result,output);return result


def fit_generic_adapter(base,data,mode,output,updates=10000,seed=42,device='cuda',conservative_weight=.1):
    import joint_fusion as d
    if (updates<1 or not data.get('completed') or not len(data['targets'])
            or not np.isfinite(conservative_weight) or conservative_weight<0):
        raise ValueError('Generic fitting needs completed labeled data and a positive update budget.')
    if d.retired_value_fields(base):raise ValueError('Generic fusion starts without learned intervention values.')
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


def independent_source_initialization(template, source_run, seed, *, source_checkpoint='final'):
    """Instantiate an untrained analytic selector with an independent source actor."""
    import csv
    import joint_fusion as d
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
            or d.retired_value_fields(template) or template.get('generic_fusion_adapter')):
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
            repeat_scope='independent source learning; fixed M1 candidate selector'))
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
    import joint_fusion as d
    scene, future, directory, digest = task
    key = (str(Path(directory).resolve()), digest)
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
               discounted_return=total, environment_protocol='source', physical_steps=steps)
    if future is not None:
        row['future_seed'] = int(future)
    return row


def evaluate_author_source_repeat(directory, tasks, *, workers=2, result_suffix='source-development',
                                  evaluation_context=None):
    """Pair a fully trained ARBoids, RP or SAC endpoint with the mechanism suites."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import joint_fusion as r
    if workers < 1 or not tasks:
        raise ValueError('Baseline evaluation requires paired tasks and a positive worker count.')
    directory = Path(directory)
    source, _ = validate_author_source_repeat(directory)
    digest = source['selected_checkpoint_sha256']
    path = directory/(result_suffix+'.pth')
    previous = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
    if previous is not None:
        if (previous['policy_sha256'] != digest or previous['tasks'] != tasks
                or previous.get('evaluation_context') != evaluation_context):
            raise ValueError('A resumed baseline evaluation must retain its checkpoint and paired tasks.')
        if len(previous['rows']) != len(tasks):
            raise ValueError('A resumed evaluation must retain one row slot per paired task.')
        if previous['completed']:
            if any(row is None for row in previous['rows']):
                raise ValueError('A completed baseline evaluation is missing task outcomes.')
            return previous
    rows = previous['rows'] if previous else [None]*len(tasks)
    if previous is not None and all(row is not None for row in rows):
        record=dict(previous,completed=True,evaluated=len(rows),success=sum(row['success'] for row in rows))
        r.atomic_save(record,path)
        return record
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
