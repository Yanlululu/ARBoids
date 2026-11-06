"""Whole-task outcome learning for joint live-fusion and coverage roles.

The retained joint controller refreshes every0.2s. A candidate selects which
boat(s) supply inner coverage, or releases coverage; it never freezes the
other boats' fusion coefficients. The Transformer directly predicts terminal
outcomes for selection, alongside synchronous stopped-distance distributions.
"""
from dataclasses import dataclass
import math
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn

from citf.bridge import Controller
from citf.contracts import canonical_hash,file_hash,validate_predictor_protocol
from experiments import batched_feedback as feedback
from experiments.contact_event_policy import retained_teacher
from experiments.fusion_capture_policy import FusionConfig,FusionOutcomeTransformer
from experiments.interception_reserve import tracking_thrust
from experiments.joint_viability_projection import ViabilityController
from experiments.synchronized_configuration_policy import joint


@dataclass(frozen=True)
class RoleConfig(FusionConfig):
    revision: str = 'CITF-live-joint-fusion-role-task-outcomes-v1'
    distribution: str = 'hurdle'
    decision_ticks: int = 32
    temperature: float = .2
    candidate_family: str = 'all_single_guards,no_guard,two_nearest,anticipatory_guard'
    selection_rule: str = 'learned_terminal_utility'


def guard_forces(state,parameters):
    predicted=state[:,-1:,:2]+parameters[...,:1]*state[:,-1:,3:5]
    target=predicted/predicted.norm(dim=-1,keepdim=True).clamp_min(1e-6)*12.5
    return tracking_thrust(state,target)


def role_parameters(world):
    state=torch.as_tensor(world.state(),dtype=torch.float32);n=len(state)-1
    parameters=torch.zeros(n+3,n,5);parameters[...,4]=-1
    for i in range(n):parameters[i,i]=parameters.new_tensor([0,0,0,1,1])
    target=state[-1,:2]/state[-1,:2].norm().clamp_min(1e-6)*12.5
    closest=(state[:-1,:2]-target).norm(dim=-1).argsort()
    parameters[n+1,closest[:2]]=parameters.new_tensor([0,0,0,1,1])
    predicted=state[-1,:2]+state[-1,3:5]
    target=predicted/predicted.norm().clamp_min(1e-6)*12.5
    leading=int((state[:-1,:2]-target).norm(dim=-1).argmin())
    parameters[n+2,leading]=parameters.new_tensor([1,0,0,1,1])
    return parameters


class LiveRoleReference:
    def __init__(self,teacher):
        self.teacher=teacher;self.parameters=None;self.cached=None

    @torch.no_grad()
    def act(self,world):
        if self.cached is None:force,trace=self.teacher.act(world)
        else:
            force,trace=self.cached;self.cached=None
        force=force.copy();trace=dict(trace)
        state=torch.as_tensor(world.state(),dtype=torch.float32)[None]
        guard=guard_forces(state,self.parameters[None])[0].numpy()
        mask=self.parameters[:,3].numpy()>.5
        force[mask]=guard[mask]
        trace.update(live_teacher_gates=trace.get('gates'),active_guards=np.flatnonzero(mask).tolist())
        return force,trace


class RoleExecutor(ViabilityController):
    def __init__(self,source,teacher):
        self.source=source;self.teacher=LiveRoleReference(teacher)
        self.model=SimpleNamespace(cfg=SimpleNamespace(guard_margin=6.,guard_weight=16.))
        self.horizon,self.iterations=10,12

    def configure(self,parameters):self.teacher.parameters=parameters.cpu()


class RoleNominalStep(nn.Module):
    """Cheap known feedback: frozen root SAC and fusion, evolving Boids/roles/APF.

The real executor refreshes the joint teacher. Its response and projection
effects are learned from actual transitions, not silently treated as known.
"""
    def forward(self,state,legacy,gates,parameters,learned,first_force,first):
        _,boids=feedback.source_observations(state,legacy)
        guard=guard_forces(state,parameters)
        rules=torch.where(parameters[...,3:4]>.5,(guard-250)/750,boids)
        force=(750*(gates*learned+(1-gates)*rules)+250).clamp(-500,1000)
        force=torch.where(first>.5,first_force,force)
        thrust=torch.cat((force,feedback.attacker_thrust(state,2.25)[:,None]),1)
        return feedback.integrate(state,thrust)


@torch.no_grad()
def make_role_input(world,source,teacher,cfg,previous_guard):
    reference,trace=teacher.act(world)
    learned,boids,_=source.commands(world.env)
    parameters=role_parameters(world);C,N=parameters.shape[:2]
    gates=torch.as_tensor(trace['gates'],dtype=torch.float32)[None].expand(C,-1,-1).clone()
    gates=torch.where(parameters[...,3:4]>.5,torch.zeros_like(gates),gates)
    state=torch.as_tensor(world.state(),dtype=torch.float32)[None].expand(C,-1,-1).clone()
    guards=guard_forces(state,parameters)
    force=torch.as_tensor(reference,dtype=torch.float32)[None].expand(C,-1,-1).clone()
    force=torch.where(parameters[...,3:4]>.5,guards,force)
    rules=torch.where(parameters[...,3:4]>.5,(guards-250)/750,boids[None].expand(C,-1,-1))
    legacy=state[...,3:5].clone();a=learned[None].expand(C,-1,-1);first=torch.ones(C,1,1)
    if not hasattr(source,'_role_nominal'):source._role_nominal={}
    key=(C,N)
    if key not in source._role_nominal:
        source._role_nominal[key]=torch.jit.freeze(torch.jit.trace(RoleNominalStep().eval(),
            (state,legacy,gates,parameters,a,force,first),check_trace=False))
    step=source._role_nominal[key]
    available=min(cfg.horizon,int(math.ceil((world.protocol.time_limit-world.env.Current_T-1e-8)/.2)))
    paths=[]
    for q in range(available):
        state,legacy=step(state,legacy,gates,parameters,a,force,first if q==0 else torch.zeros_like(first))
        paths.append(state[...,:2])
    nominal=torch.stack(paths,-2)
    if available<cfg.horizon:nominal=torch.cat((nominal,nominal[...,-1:,:].expand(-1,-1,cfg.horizon-available,-1)),-2)
    history,positions,hv,rotation=world.observation()
    candidate=torch.cat((gates,a,rules,(force-250)/750,
        parameters/parameters.new_tensor([4,12,12,1,1])),-1)
    candidate=torch.cat((candidate,candidate.new_zeros(C,1,13)),1)
    batch=dict(history=history[None],positions=positions[None],history_valid=hv[None],
        entity_valid=torch.ones(1,N+1,dtype=torch.bool),candidate=candidate[None],
        nominal=(nominal@torch.as_tensor(rotation,dtype=torch.float32))[None],
        known_time_valid=(torch.arange(cfg.horizon)<available)[None],
        time_context=torch.tensor([[(world.protocol.time_limit-world.env.Current_T)/world.protocol.time_limit,
                                    available/cfg.horizon]],dtype=torch.float32),
        previous_guard=torch.tensor([previous_guard],dtype=torch.long))
    return batch,parameters,rotation,(reference,trace)


class RoleOutcomeTransformer(FusionOutcomeTransformer):
    def __init__(self,cfg):
        super().__init__(cfg);w=cfg.width
        self.outcomes=nn.Sequential(nn.Linear(3*w+3+7,w),nn.GELU(),nn.Linear(w,4))
        self.state_value=nn.Sequential(nn.Linear(w+2,w),nn.Tanh(),nn.Linear(w,1))
        nn.init.zeros_(self.outcomes[-1].weight);nn.init.zeros_(self.outcomes[-1].bias)
        nn.init.zeros_(self.state_value[-1].weight);nn.init.constant_(self.state_value[-1].bias,3.3)

    def forward(self,batch):
        out=super().forward(batch);N=batch['positions'].shape[1]-1
        _,j=torch.triu_indices(N+1,N+1,1,device=out['mu'].device);defenders=j<N
        geometry=torch.stack((out['team_probability'],out['radius_probability'],
            out['team_excess_mean'].clamp_max(200)/20,
            out['mu'][...,defenders].amin(-1)/20,out['sigma'][...,defenders].mean(-1)/5,
            out['pair_probability'][...,defenders].sum(-1),out['time_fraction']),-1)
        correction=self.outcomes(torch.cat((out['configuration_embedding'],geometry),-1))
        guards=batch['candidate'][:,:,:N,11]>.5;count=guards.sum(-1)
        position=batch['positions'];enemy=position[:,-1]
        target=enemy/enemy.norm(dim=-1,keepdim=True).clamp_min(1e-6)*12.5
        distance=(position[:,:N]-target[:,None]).norm(dim=-1)
        minimum=distance.amin(-1)
        guard_distance=distance[:,None].expand_as(guards).masked_fill(~guards,torch.inf).amin(-1)
        previous=batch['previous_guard'];reference=torch.where(previous>=0,previous,distance.argmin(-1))
        same=guards.gather(2,reference[:,None,None].expand(-1,guards.shape[1],1)).squeeze(-1)&(count==1)
        excess=(guard_distance-minimum[:,None]).clamp(0,60)
        failure=.015+.35*(1-torch.exp(-excess/12))
        failure=torch.where(same,torch.full_like(failure,.005),failure)
        failure=torch.where(count==0,torch.full_like(failure,.25),failure)
        capture=.44*(N-count).to(failure.dtype)/(N-1)
        capture=torch.where(count==0,torch.full_like(capture,.58),capture)
        prior=torch.stack((capture,1-capture-failure,failure),-1).clamp_min(1e-5)
        logits=prior.log()+correction[...,:3]
        probability=logits.softmax(-1);capture_time=correction[...,3].sigmoid()
        utility=4*probability[...,0]+3*probability[...,1]-12*probability[...,2]+.2*probability[...,0]*(1-capture_time)
        previous_distance=distance.gather(1,reference[:,None]).squeeze(1)/20
        value=self.state_value(torch.cat((out['state_embedding'],(previous>=0)[:,None].float(),
                                          previous_distance[:,None]),-1)).squeeze(-1)
        out.update(outcome_logits=logits,outcome_probabilities=probability,utility=utility,
                   terminal_capture_time=capture_time,value=value)
        return out


def policy_probabilities(output,cfg,epsilon):
    probability=(output['utility']/cfg.temperature).softmax(-1)
    return (1-epsilon)*probability+epsilon/probability.shape[-1]


def load_role_policy(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=RoleOutcomeTransformer(RoleConfig(**payload['config']))
    model.load_state_dict(payload['state']);model.physical_protocol=payload['physical_protocol']
    return model.to(device).eval()


class RoleTaskController(Controller):
    def __init__(self,*args,epsilon=0.,sampling_seed=None,teacher=None,**kwargs):
        super().__init__(*args,**kwargs);self.cfg=self.model.cfg
        self.teacher=teacher if teacher is not None else joint.JointPolicyController(
            self.source,retained_teacher(self.cfg.teacher_checkpoint),'cpu',12,96,1)
        self.executor=RoleExecutor(self.source,self.teacher)
        self.epsilon=epsilon;self.rng=None if sampling_seed is None else np.random.RandomState(sampling_seed)
        self.ticks=0;self.guard=None;self.parameters=None;self.last_time=None

    def act(self,world):
        started=time.perf_counter();now=world.env.Current_T
        if self.guard is None or self.last_time is not None and now<=self.last_time:
            state=world.state();target=state[-1,:2]/max(np.linalg.norm(state[-1,:2]),1e-6)*12.5
            self.guard=int(np.linalg.norm(state[:-1,:2]-target,axis=-1).argmin());self.ticks=0
        replanned=self.ticks%self.cfg.decision_ticks==0
        if replanned:
            batch,parameters,rotation,cached=make_role_input(world,self.source,self.teacher,self.cfg,self.guard)
            with torch.no_grad():out=self.model({k:v.to(self.device) for k,v in batch.items()})
            probabilities=policy_probabilities(out,self.cfg,self.epsilon)[0].cpu().numpy()
            if self.cfg.selection_rule=='fixed_anticipatory':
                choice=len(parameters)-1
                probabilities=np.zeros_like(probabilities);probabilities[choice]=1.
            elif self.cfg.selection_rule!='learned_terminal_utility':
                raise ValueError('Unknown role selection rule: '+self.cfg.selection_rule)
            elif self.rng is None:choice=int(out['utility'][0].argmax())
            else:choice=int(self.rng.choice(len(parameters),p=(probabilities.astype(np.float64)/probabilities.sum(dtype=np.float64))))
            self.last_batch={k:v[0].cpu() for k,v in batch.items()};self.last_rotation=rotation
            self.last_choice=choice;self.last_probabilities=probabilities;self.last_value=float(out['value'][0])
            self.parameters=parameters[choice];ids=torch.nonzero(self.parameters[:,3]>.5).flatten().tolist()
            self.guard=ids[0] if len(ids)==1 else (-1 if not ids else (self.guard if self.guard in ids else ids[0]))
            self.executor.configure(self.parameters);self.executor.teacher.cached=cached
            self.decision=dict(selected_configuration=choice,active_guards=ids,
                outcome_probabilities=out['outcome_probabilities'][0].cpu().tolist(),utilities=out['utility'][0].cpu().tolist(),
                terminal_capture_time=out['terminal_capture_time'][0].cpu().tolist(),
                predicted_contact_probability=out['team_probability'][0].cpu().tolist(),
                behavior_probabilities=probabilities.tolist(),role_parameters=self.parameters.tolist())
        force,physical=self.executor.act(world)
        role_calls=len(probabilities) if replanned else 0
        teacher_calls=int(physical.get('evaluated_unique',0));teacher_batches=int(physical.get('forward_batches',0))
        trace=dict(self.decision,replanned=replanned,evaluated_unique=role_calls+teacher_calls,
            forward_batches=int(replanned)+teacher_batches,role_configurations=role_calls,
            retained_teacher_configurations=teacher_calls,retained_teacher_forward_batches=teacher_batches,
            selection_rule=self.cfg.selection_rule,physical_thrust=force.tolist(),
            live_teacher_gates=physical.get('live_teacher_gates'),projection_applied=physical['projection_applied'],
            predicted_pair_after=physical['predicted_pair_after'],predicted_coverage_after=physical['predicted_coverage_after'],
            revision=self.cfg.revision,compute_seconds=time.perf_counter()-started)
        self.last_time=now;self.ticks+=1
        return force,trace


def main():
    import citf.evaluate as evaluation
    original=evaluation.runtime_identity
    def identity(*args,**kwargs):
        value=original(*args,**kwargs);path=args[2] if len(args)>2 else kwargs['checkpoint']
        config=torch.load(path,map_location='cpu',weights_only=False)['config']
        value.update(method='CITF-live-fusion-role-task-learning-development',revision=config['revision'])
        rule=config.get('selection_rule','learned_terminal_utility')
        value['controller'].update(configuration=config,
            search='exhaustive joint role programs over live joint-fusion feedback',
            role_candidate_count='N_defenders+3',candidate_counts_include_retained_teacher=True,
            action='joint selection of live teacher fusion with single,dual,anticipatory or released coverage; every0.2s source and projection feedback',
            selection=('always choose the anticipatory guard program; trained role outcomes are computed for matched cost but do not determine the action'
                if rule=='fixed_anticipatory' else 'predicted full-task capture,timeout,failure and capture time; explicit distance-distribution features and candidate Transformer representation'),
            training='complete real episodes,Monte Carlo credit,PPO and terminal-outcome likelihood; actual selected-option synchronous-distance supervision',
            nominal='frozen root SAC/fusion with reactive Boids,roles and fixed2.25 APF; real executor refreshes full joint teacher')
        value['runtime_code']=canonical_hash(dict(base=value['runtime_code'],role_policy=file_hash(__file__)))
        return value
    def load(path,source,world,device):
        model=load_role_policy(path,device);validate_predictor_protocol(model.physical_protocol,world.protocol);return model
    evaluation.runtime_identity=identity;evaluation.load_behavior=load;evaluation.Controller=RoleTaskController
    evaluation.main()


if __name__=='__main__':main()
