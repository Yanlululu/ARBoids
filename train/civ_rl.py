"""Unified CI candidate generation, paired value learning and one fusion output.

The deployment state is explicit. A scheduled decision commits either to the
CI reference or to one whole-fleet coefficient plan. Source controls continue
to respond to the live state throughout the selected window. Training branches
and deployment use the same executor, including the return to CI afterwards.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import develop_source_fusion as d
from conditional_fusion_value import _refine_actuator_interactions, actuator_candidate_grid


@dataclass(frozen=True)
class CIVConfig:
    generator: str = 'joint'
    candidate_budget: int = 6
    cost_regret_per_vessel: float = .5
    decision_step: int = 20
    durations: tuple = (5, 20)
    delay: int = 0
    uncertainty_penalty: float = .5
    acceptance_margin: float = .1
    inference_width: int = 64
    execution_revision: str = 'civ-unified-whole-plan-v1'

    def __post_init__(self):
        if self.generator not in ('joint', 'independent', 'local'):
            raise ValueError('Unknown candidate generator.')
        if self.candidate_budget < 2 or self.cost_regret_per_vessel < 0:
            raise ValueError('Candidate budget and CI regret must be nonnegative.')
        if self.decision_step < 0 or self.delay < 0 or not self.durations or min(self.durations) < 1:
            raise ValueError('Executable windows need nonnegative delay and positive duration.')
        if self.inference_width < 1:
            raise ValueError('Training and deployment require a fixed inference width.')

    def identity(self):
        # Thresholds are selection settings, not a change in transition labels.
        values = asdict(self)
        values.pop('uncertainty_penalty'); values.pop('acceptance_margin')
        return hashlib.sha256(repr(sorted(values.items())).encode()).hexdigest()


def load_source(path, device='cpu'):
    root = Path(__file__).resolve().parents[1]
    d.initialize(root/'train/experiments/main-seed42-20260921/adares1.pth')
    actor = d.joint_load(str(path), Path(path).stat().st_mtime_ns).to(device).eval()
    if actor.successor_task_value is not None or not actor.actuator_fusion_refinement:
        raise ValueError('Unified learning requires a CI source without a legacy CV controller.')
    if not actor.actuator_fusion_refinement.get('source_preference'):
        raise ValueError('Candidate controls must use the common source preference.')
    legacy=('local_value','physical_value','response_base','initial_role_credit','feedback_role_credit',
            'cooperative_role_credit','finite_temporal_credit','continuous_program_value','joint_intervention_value',
            'feedback_fleet_policy')
    if (any(getattr(actor,name,None) is not None for name in legacy) or actor.gate_period!=1
            or actor.intent_strength or any(len(getattr(actor,name,())) for name in
                ('physical_values','response_values','joint_intervention_values'))):
        raise ValueError('A unified source cannot contain another intervention controller or action clock.')
    torch.backends.cuda.matmul.allow_tf32=False
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def plan_costs(unary, risk, plans, weight):
    b, n, k = unary.shape
    own = torch.stack([unary[:, i].gather(1, plans[:, :, i]) for i in range(n)], -1)
    pair = unary.new_zeros(b, plans.shape[1], n, n)
    for i in range(n):
        for j in range(i+1, n):
            index = plans[:, :, i]*k+plans[:, :, j]
            value = risk[:, i, j].flatten(1).gather(1, index)
            pair[:, :, i, j] = value; pair[:, :, j, i] = value
    return own.sum(-1)+.5*weight*pair.sum((-1, -2)), own, pair


def local_product_plans(unary, budget):
    """Top product configurations under additive local costs, without C_ij."""
    b, n, k = unary.shape
    plans = torch.empty(b, 1, 0, dtype=torch.long, device=unary.device)
    scores = unary.new_zeros(b, 1)
    for i in range(n):
        values = scores[:, :, None]+unary[:, i, None]
        scores, index = values.flatten(1).topk(min(budget, values[0].numel()), largest=False)
        plans = torch.cat((plans.gather(1, (index//k)[..., None].expand(-1, -1, i)),
                           (index % k)[..., None]), -1)
    return plans


@torch.no_grad()
def candidate_catalog(actor, packet, source, config, *, reference_only=False):
    """Expose actual complete CI configurations before applying any control."""
    settings = dict(actor.actuator_fusion_refinement)
    weight = float(settings.get('risk_weight', 32.))
    if config.generator == 'local':
        # System ablation: source-only unary candidates, with no pair-cost
        # calculation, no joint interaction optimizer and no learned CI prior.
        gates = source[..., 2:3].expand(-1, -1, 2)
        grid = actuator_candidate_grid(gates)
        unary = ((grid-gates[:, :, None])/settings.get('preference_width', .4)).square().mean(-1)
        b, n, k = unary.shape
        repeated = (grid[:, :, :, None]-grid[:, :, None, :]).abs().amax(-1)<1e-6
        duplicate = (repeated & torch.tril(torch.ones(k, k, dtype=torch.bool, device=source.device), -1)).any(-1)
        unary = unary.masked_fill(duplicate, float('inf'))
        paths = actor.candidate_risk(dict(packet), source[..., :2], candidate_grid=grid, return_paths=True)
        info = dict(candidates=grid, unary=unary, pair_risk=source.new_zeros(b,n,n,k,k),
            order=torch.arange(n, device=source.device)[None].expand(b,-1), reference=unary.argmin(-1),
            prediction_paths=paths, refined_plans=local_product_plans(unary, settings.get('beam_width',128)))
        weight = 0.
    else:
        settings['selection'] = config.generator
        info = {}
        _refine_actuator_interactions(actor, packet, source, source, candidate_output=info, **settings)
    grid, unary, risk, reference = (info[x] for x in ('candidates','unary','pair_risk','reference'))
    b, n, k = unary.shape
    if reference_only:
        plans = reference[:,None]; valid = torch.ones(b,1,dtype=torch.bool,device=source.device)
    else:
        neighbors = reference[:,None,None,:].expand(-1,n,k,-1).clone()
        for i in range(n):
            neighbors[:,i,:,i] = torch.arange(k,device=source.device)
        pool = torch.cat((reference[:,None],info.get('beam_plans',reference[:,None]),
                          info['refined_plans'],neighbors.flatten(1,2)),1)
        scores, _, _ = plan_costs(unary,risk,pool,weight)
        order = scores.argsort(dim=-1,stable=True).cpu().tolist()
        rows = pool.cpu().tolist(); limits = (scores[:,0]+n*config.cost_regret_per_vessel).cpu().tolist()
        cpu_scores = scores.cpu().tolist(); indices = []; masks = []
        for r, ranking, costs, limit in zip(rows,order,cpu_scores,limits):
            selected = [0]; seen = {tuple(r[0])}
            for index in ranking:
                key = tuple(r[index])
                if key in seen or not np.isfinite(costs[index]) or costs[index]>limit+1e-6:
                    continue
                selected.append(index); seen.add(key)
                if len(selected)==config.candidate_budget: break
            count = len(selected)
            indices.append(selected+[0]*(config.candidate_budget-count))
            masks.append([True]*count+[False]*(config.candidate_budget-count))
        index = torch.tensor(indices,device=source.device)
        plans = pool.gather(1,index[...,None].expand(-1,-1,n))
        valid = torch.tensor(masks,device=source.device)
    score, own, pair = plan_costs(unary,risk,plans,weight)
    inverse = info['order'].argsort(-1)
    gates = grid[:,None].expand(-1,plans.shape[1],-1,-1,-1).gather(
        3,plans[...,None,None].expand(-1,-1,-1,1,2)).squeeze(3)
    gates = gates.gather(2,inverse[:,None,:,None].expand(-1,plans.shape[1],-1,2))
    own = own.gather(2,inverse[:,None].expand(-1,plans.shape[1],-1))
    pair = pair.gather(2,inverse[:,None,:,None].expand_as(pair))
    pair = pair.gather(3,inverse[:,None,None,:].expand_as(pair))
    physical_choices = plans.gather(2,inverse[:,None].expand(-1,plans.shape[1],-1))
    endpoints = info['prediction_paths'][...,-1,:]
    endpoints = endpoints[:,None].expand(-1,plans.shape[1],-1,-1,-1).gather(
        3,physical_choices[...,None,None].expand(-1,-1,-1,1,2)).squeeze(3)
    return dict(gates=gates, valid=valid, costs=score, unary=own, pair_risk=pair,
                endpoints=endpoints, has_ci=float(config.generator!='local'))


def candidate_features(packet, source, catalog, config):
    """Only public state, candidate coefficients, predictions and execution time."""
    nodes, global_state, target, velocity, axis = d.PublicTargetMotion.encode(packet,True)
    b,c,n,_ = catalog['gates'].shape
    ref = catalog['gates'][:,:1].expand(-1,c,-1,-1)
    def rotate(v):
        direction = axis[:,None,None]
        return torch.stack(((v*direction).sum(-1),direction[...,0]*v[...,1]-direction[...,1]*v[...,0]),-1)
    response = rotate(catalog['endpoints']-packet['motion'][:,None,:,:2])/30.
    node = torch.cat((nodes[:,None].expand(-1,c,-1,-1),source[:,None,:,:2].expand(-1,c,-1,-1),
        ref,catalog['gates'],catalog['gates']-ref,response[:,:1].expand_as(response),response,
        catalog['unary'][...,None].clamp_max(30.),catalog['pair_risk'].mean(-1,keepdim=True)),-1)
    position = nodes[...,:2]
    relative = position[:,:,None]-position[:,None,:]
    relative_velocity = nodes[...,2:4][:,:,None]-nodes[...,2:4][:,None,:]
    endpoint = rotate(catalog['endpoints']-target[:,None,None])/30.
    edges = torch.cat((relative[:,None].expand(-1,c,-1,-1,-1),
        relative_velocity[:,None].expand(-1,c,-1,-1,-1),
        endpoint[:,:,:,None]-endpoint[:,:,None,:],catalog['pair_risk'][...,None]),-1)
    clock = packet['node'][:,0,15:16]
    statistics = torch.stack((catalog['unary'].mean(-1),catalog['pair_risk'].mean((-1,-2)),
                             (catalog['costs']-catalog['costs'][:,:1])/n),-1).clamp(-30.,30.)
    public = torch.cat((global_state,clock),-1)[:,None].expand(-1,c,-1)
    base = torch.cat((public,statistics,source.new_full((b,c,1),catalog['has_ci'])),-1)
    # Option zero is factual continuation. Every active slot is a CI plan
    # paired with an explicit hold duration, with a fixed documented delay.
    indices = [0]+[i for i in range(1,c) for _ in config.durations]
    durations = [0]+[span for _ in range(1,c) for span in config.durations]
    timing = source.new_tensor([[0.,0.]]+[[config.delay/20.,span/20.]
        for _ in range(1,c) for span in config.durations])[None].expand(b,-1,-1)
    return dict(node=node[:,indices],edge=edges[:,indices],global_features=torch.cat((base[:,indices],timing),-1),
        valid=catalog['valid'][:,indices],gates=catalog['gates'][:,indices],
        durations=torch.tensor(durations,device=source.device)[None].expand(b,-1))


class CIVValueNetwork(nn.Module):
    """Permutation-invariant whole-configuration value, shared across fleet sizes."""
    def __init__(self,width=48):
        super().__init__(); self.width=width
        self.node = nn.Sequential(nn.Linear(23,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.edge = nn.Sequential(nn.Linear(7,width//2),nn.SiLU(),nn.Linear(width//2,width//2),nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(3*width+13,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,3))

    def forward(self,features):
        nodes = self.node(features['node']); edges = self.edge(features['edge'])
        n = nodes.shape[-2]
        mask = ~torch.eye(n,dtype=torch.bool,device=nodes.device)
        edges = edges[:,:,mask]
        pooled = torch.cat((nodes.mean(-2),nodes.amax(-2),edges.mean(-2),edges.amax(-2),features['global_features']),-1)
        return self.readout(pooled)


def conditional_loss(logits, targets, valid):
    y0 = targets[:,:1]; mask=valid.to(logits.dtype)
    baseline = F.binary_cross_entropy_with_logits(logits[:,0,0],y0[:,0])
    rescue = F.binary_cross_entropy_with_logits(logits[...,1],targets,reduction='none')
    harm = F.binary_cross_entropy_with_logits(logits[...,2],1.-targets,reduction='none')
    return baseline+(((1.-y0)*rescue+y0*harm)*mask).sum()/mask.sum().clamp_min(1.)


def ordinary_loss(logits, targets, valid):
    # Every output is trained on the ordinary terminal success label. Both
    # objectives use exactly the same inputs, architecture and parameter count.
    losses=F.binary_cross_entropy_with_logits(logits,targets[...,None].expand_as(logits),reduction='none').mean(-1)
    return (losses*valid).sum()/valid.sum().clamp_min(1.)


def ensemble_values(networks,features,kind):
    values=[]
    for network in networks:
        raw=network(features)
        if kind=='conditional':
            if getattr(network,'paired_event_value',False):
                events=torch.cat((torch.zeros_like(raw[...,:1]),raw[...,1:]),-1).softmax(-1)
                gain=events[...,1]-events[...,2]
            else:
                p0=raw[:,0:1,0].sigmoid()
                gain=(1.-p0)*raw[...,1].sigmoid()-p0*raw[...,2].sigmoid()
        elif kind=='ordinary':
            q=raw.sigmoid().mean(-1);gain=q-q[:,:1]
        else: raise ValueError('Value training objective must be explicit.')
        gain=gain.clone();gain[:,0]=0.;values.append(gain)
    gains=torch.stack(values)
    return gains.mean(0),gains.std(0,unbiased=False)


def choose_options(gain,uncertainty,valid,margin,penalty):
    scores=gain-penalty*uncertainty-margin
    scores=scores.masked_fill(~valid,float('-inf'));scores=scores.clone();scores[:,0]=0.
    return scores.argmax(-1),scores


class CIVController:
    """Single deployment decision API used by data collection and evaluation."""
    def __init__(self,source,config,networks=(),kind='conditional'):
        self.source=source;self.config=config;self.networks=list(networks);self.kind=kind

    @torch.no_grad()
    def proposals(self,packet,reference_only=False):
        if self.config.generator=='local':
            mean,_,hidden=self.source.proposal_distribution(packet['obs'])
            control=mean.tanh();base=self.source.base
            own=base.activation(base.a1(torch.cat((control,packet['node'][...,11:13]),-1)))
            gate=base.adap_layer(torch.cat((own,hidden),-1)).tanh()*.5+.5
            source=torch.cat((control,gate),-1)
        else:
            source=self.source(packet,deterministic=True,gate_only=True)[0][...,:3]
        catalog=candidate_catalog(self.source,packet,source,self.config,reference_only=reference_only)
        return source,catalog

    @torch.no_grad()
    def decide(self,packet):
        source,catalog=self.proposals(packet)
        features=candidate_features(packet,source,catalog,self.config)
        if self.networks:
            gain,uncertainty=ensemble_values(self.networks,features,self.kind)
            choice,scores=choose_options(gain,uncertainty,features['valid'],
                self.config.acceptance_margin,self.config.uncertainty_penalty)
        else:
            choice=torch.zeros(len(source),dtype=torch.long,device=source.device)
            gain=source.new_zeros(features['valid'].shape);uncertainty=gain.clone();scores=gain.clone()
        return source,catalog,features,dict(choice=choice,gain=gain,uncertainty=uncertainty,scores=scores)

    @staticmethod
    def execute(source,boids,gates):
        if gates.shape!=source[...,:2].shape or not bool(((gates>=0.)&(gates<=1.)).all()):
            raise ValueError('Final control requires one valid two-component fusion configuration.')
        command=gates*source[...,:2]+(1.-gates)*boids
        return torch.cat((command,torch.ones_like(source[...,:1])),-1)


def save_checkpoint(path,source_path,config,networks,kind,fit):
    source_path=Path(source_path)
    if not networks or not fit.get('completed'):raise ValueError('Only a completed value fit is deployable.')
    root=Path(__file__).resolve().parents[1]
    try:relative=str(source_path.resolve().relative_to(root))
    except ValueError:relative=None
    record=dict(algorithm='CIV-RL-unified-v1',source_path=str(source_path),
        source_relative_path=relative,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),config=asdict(config),
        execution_identity=config.identity(),kind=kind,width=networks[0].width,
        networks=[{k:v.detach().cpu() for k,v in net.state_dict().items()} for net in networks],fit=fit)
    torch.save(record,path)
    return record


def load_controller(path,device='cpu'):
    record=torch.load(path,map_location='cpu',weights_only=False)
    if record['algorithm']!='CIV-RL-unified-v1':raise ValueError('This is not a unified CIV-RL checkpoint.')
    if not record['fit'].get('completed'):raise ValueError('Incomplete value fitting cannot be deployed.')
    source=Path(record['source_path'])
    if not source.exists():
        relative=record.get('source_relative_path')
        # The first pilot records predate portable path metadata. Its complete
        # source identity is still checked before accepting a relocated file.
        if relative is None and 'train' in source.parts:
            relative=str(Path(*source.parts[source.parts.index('train'):]))
        if relative is not None:source=Path(__file__).resolve().parents[1]/relative
    if hashlib.sha256(source.read_bytes()).hexdigest()!=record['source_sha256']:
        raise ValueError('The frozen source identity changed.')
    config=CIVConfig(**record['config'])
    if config.identity()!=record['execution_identity']:raise ValueError('Execution semantics changed.')
    networks=[]
    for state in record['networks']:
        network=CIVValueNetwork(record['width']).to(device).eval();network.load_state_dict(state);networks.append(network)
    return CIVController(load_source(source,device),config,networks,record['kind'])


def fixed_inference(function,packet,width):
    """Pad every neural/planner call, including the last surviving episode."""
    count=len(next(iter(packet.values())))
    if not count:raise ValueError('An inference batch must be nonempty.')
    def tree_slice(value,length):
        if torch.is_tensor(value):return value[:length]
        if isinstance(value,dict):return {key:tree_slice(item,length) for key,item in value.items()}
        if isinstance(value,tuple):return tuple(tree_slice(item,length) for item in value)
        return value
    def tree_cat(values):
        first=values[0]
        if torch.is_tensor(first):return torch.cat(values,0)
        if isinstance(first,dict):return {key:tree_cat([value[key] for value in values]) for key in first}
        if isinstance(first,tuple):return tuple(tree_cat([value[i] for value in values]) for i in range(len(first)))
        if any(value!=first for value in values):raise ValueError('Inconsistent fixed inference metadata.')
        return first
    pieces=[]
    for begin in range(0,count,width):
        length=min(width,count-begin)
        batch={}
        for key,value in packet.items():
            part=value[begin:begin+length]
            batch[key]=torch.cat((part,part[-1:].expand(width-length,*part.shape[1:])),0) if length<width else part
        pieces.append(tree_slice(function(batch),length))
    return tree_cat(pieces)


@dataclass
class ExecutionState:
    step:int=0
    decided:bool=False
    choice:int=0
    start:int=0
    end:int=0
    gates:object=None

    def commit(self,choice,gates,duration,delay):
        if self.decided:raise ValueError('One scheduled decision cannot be overwritten.')
        self.decided=True;self.choice=int(choice)
        self.start=self.step+int(delay);self.end=self.start+int(duration)
        self.gates=np.asarray(gates,dtype=np.float32).copy()

    def coefficients(self,reference):
        return self.gates if self.decided and self.choice and self.start<=self.step<self.end else reference


class CIVExecutor:
    """The identical stateful executor for paired branches and deployment."""
    def __init__(self,controller):self.controller=controller

    @torch.no_grad()
    def actions(self,packet,states,forced=None):
        controller=self.controller;config=controller.config
        source,catalog=fixed_inference(lambda p:controller.proposals(p,reference_only=True),packet,config.inference_width)
        due=[i for i,state in enumerate(states) if not state.decided and state.step==config.decision_step]
        decisions={}
        if due:
            part={key:value[due] for key,value in packet.items()}
            _,_,features,decision=fixed_inference(controller.decide,part,config.inference_width)
            for j,i in enumerate(due):
                choice=int(decision['choice'][j]) if forced is None else int(forced[i])
                if not bool(features['valid'][j,choice]):raise ValueError('Cannot execute a padded candidate.')
                states[i].commit(choice,features['gates'][j,choice].cpu().numpy(),
                    int(features['durations'][j,choice]),config.delay)
                decisions[i]=dict(choice=choice,gain=float(decision['gain'][j,choice]),
                    uncertainty=float(decision['uncertainty'][j,choice]),
                    coefficients=states[i].gates.tolist(),duration=int(features['durations'][j,choice]))
        reference=catalog['gates'][:,0].cpu().numpy()
        gates=source.new_tensor(np.stack([state.coefficients(value) for state,value in zip(states,reference)]))
        return controller.execute(source,packet['node'][...,11:13],gates).cpu().numpy(),decisions


def runtime_identity():
    files=('civ_rl.py','conditional_fusion_value.py','develop_source_fusion.py','envs/TADgame.py',
           'envs/modules.py','source_arboids.py','policy/networks.py','policy/SAC.py')
    base=Path(__file__).resolve().parent
    return {name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in files}


def run_episodes(controller,tasks,*,batch_size=64,initial_jobs=None,collect_roots=False,progress=False):
    """Complete closed-loop tasks; no terminal labels enter a controller."""
    from envs.TADgame import TADEnv
    device=next(controller.source.parameters()).device
    executor=CIVExecutor(controller);results=[];saved_random=np.random.get_state()
    try:
        for begin in range(0,len(tasks),batch_size):
            current=tasks[begin:begin+batch_size];jobs=[]
            if len({task['scene'][1] for task in current})!=1:
                raise ValueError('Each inference batch must have one fleet size.')
            for offset,task in enumerate(current):
                scene=tuple(task['scene']);seed,n,agility=scene
                if initial_jobs is None:
                    np.random.seed(seed);env=TADEnv(n,protocol=controller.source.environment_protocol)
                    obs,_=env.reset(agility=agility,noisy_agility=False);state=ExecutionState()
                else:
                    root=initial_jobs[begin+offset]
                    env=copy.deepcopy(root['environment']);obs=root['observation'].copy()
                    np.random.set_state(root['random_state']);state=ExecutionState(step=root['step'])
                    if 'program' in task:
                        program=task['program'];state.commit(program['choice'],program['gates'],program['duration'],controller.config.delay)
                if task.get('future') is not None:np.random.seed(task['future'])
                jobs.append(dict(env=env,obs=obs,rng=np.random.get_state(),state=state,root=None,
                    decision=None,simulation_steps=0,discounted_return=0.))
            active=list(range(len(jobs)))
            while active:
                if collect_roots:
                    for i in active:
                        job=jobs[i]
                        if job['state'].step==controller.config.decision_step:
                            job['root']=dict(environment=copy.deepcopy(job['env']),observation=job['obs'].copy(),
                                random_state=job['rng'],step=job['state'].step,scene=current[i]['scene'])
                packets=[d.joint_packet(jobs[i]['env'],jobs[i]['obs']) for i in active]
                packet=d.mechanism_stack(packets,str(device))
                commands,decisions=executor.actions(packet,[jobs[i]['state'] for i in active])
                following=[]
                for j,i in enumerate(active):
                    job=jobs[i];np.random.set_state(job['rng'])
                    obs,reward,done,_=d.joint_task_step(job['env'],commands[j],
                        controller.source.training_gamma,controller.source.training_step_cost)
                    job['discounted_return']+=controller.source.training_gamma**job['state'].step*reward
                    job['rng']=np.random.get_state();job['obs']=obs;job['state'].step+=1;job['simulation_steps']+=1
                    if j in decisions:job['decision']=decisions[j]
                    if done:
                        row=dict(scene=current[i]['scene'],future=current[i].get('future'),**d.outcome(job['env'],done),
                            simulation_steps=job['simulation_steps'],decision=job['decision'],
                            selected=job['state'].choice,discounted_return=job['discounted_return'])
                        if collect_roots:row['root']=job['root']
                        job['result']=row
                    else:
                        if job['state'].step>int(round(job['env'].Total_T/job['env'].Action_T))+2:
                            raise RuntimeError('Environment failed to emit its terminal event.')
                        following.append(i)
                active=following
            results.extend(job['result'] for job in jobs)
            if progress:print(json.dumps(dict(completed=len(results),total=len(tasks),
                successes=sum(row['success'] for row in results))),flush=True)
    finally:np.random.set_state(saved_random)
    return results
