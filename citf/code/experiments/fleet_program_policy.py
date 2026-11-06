"""Predict complete fleet-program outcomes and select joint interception feedback.

Labels continue each candidate's state-feedback program to task termination.
The opening controller commits to that program; the persistent controller can
reselect every configured interval, changing the subsequent continuation.
Each program jointly assigns formation goals and closing roles for the fleet.
"""
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from citf.bridge import Controller
from citf.model import Attention
from citf.contracts import canonical_hash, file_hash, validate_predictor_protocol
from experiments import batched_feedback as feedback
from experiments.contact_event_policy import retained_teacher
from experiments.fusion_capture_policy import FusionConfig, FusionOutcomeTransformer, hurdle_parameters, nominal_statistics
from experiments.interception_reserve import tracking_thrust
from experiments.joint_viability_projection import ViabilityController
from experiments.synchronized_configuration_policy import joint


PROGRAMS=('persistent_coverage','symmetric_bracket','left_interception','right_interception')


@dataclass(frozen=True)
class FleetConfig(FusionConfig):
    revision: str = 'CITF-whole-fleet-interception-complete-feedback-v1'
    distribution: str = 'hurdle'
    decision_time: float = 2.
    candidate_family: str = 'coverage,symmetric_bracket,left_interception,right_interception'
    fixed_program: int = -1
    distance_interval: str = 'known short prediction interval'
    outcome_readout: str = 'configuration_and_distribution'
    feedback_mode: str = 'opening_teacher'
    reselect_seconds: float = 0.
    capture_time_weight: float = .2
    time_prediction: str = 'nominal_window'
    selection_rule: str = 'learned'
    handoff_program: int = 1
    candidate_comparison: bool = False
    candidate_attention: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.distribution!='hurdle' or self.predictor!='learned':
            raise ValueError('Fleet outcome readout requires the learned hurdle predictor')
        if not math.isfinite(self.decision_time) or self.decision_time<0:
            raise ValueError('Fleet decision time must be finite and nonnegative')
        if type(self.fixed_program) is not int or not -1<=self.fixed_program<len(PROGRAMS):
            raise ValueError('Fixed fleet program must be -1 or a valid program index')
        if self.outcome_readout not in ('configuration_and_distribution','distribution_only','physical_distributions'):
            raise ValueError('Unknown fleet outcome readout')
        if self.time_prediction not in ('nominal_window','remaining_task'):
            raise ValueError('Unknown fleet time prediction interval')
        if self.outcome_readout=='physical_distributions' and self.time_prediction!='remaining_task':
            raise ValueError('Direct task distributions require remaining-task capture time')
        if self.selection_rule not in ('learned','observable_handoff'):
            raise ValueError('Unknown fleet selection rule')
        if self.candidate_comparison and (self.outcome_readout!='physical_distributions' or self.context):
            raise ValueError('Comparative candidates use direct distributions and replace entity attention')
        if self.selection_rule=='observable_handoff' and (self.feedback_mode!='persistent_interception'
                or self.handoff_program not in (1,2,3) or self.fixed_program!=-1 or self.reselect_seconds<=0):
            raise ValueError('Observable handoff requires a persistent orbital program and periodic decisions')
        if self.feedback_mode not in ('opening_teacher','persistent_interception'):
            raise ValueError('Unknown fleet feedback mode')
        if not math.isfinite(self.reselect_seconds) or self.reselect_seconds<0:
            raise ValueError('Reselection interval must be finite and nonnegative')
        if not math.isfinite(self.capture_time_weight) or self.capture_time_weight<0:
            raise ValueError('Capture time weight must be finite and nonnegative')


def recorded_teacher_identity(record,checkpoint):
    return (record.get('teacher_sha256') or record.get('contract',{}).get('artifacts',{}).get(
        str(Path(checkpoint).resolve()),''))


def load_teacher(source,checkpoint,expected_identity=''):
    model=retained_teacher(checkpoint,expected_identity)
    for key,actual in [('source_sha256',source.sha256),
                       ('source_implementation_sha256',source.implementation_sha256)]:
        if model.checkpoint_protocol.get(key)!=actual:
            raise ValueError('Fleet teacher and source differ: '+key)
    return joint.JointPolicyController(source,model,'cpu',12,96,1)


def initial_guard(world):
    state=world.state();target=state[-1,:2]/max(np.linalg.norm(state[-1,:2]),1e-6)*12.5
    return int(np.linalg.norm(state[:-1,:2]-target,axis=-1).argmin())


def coverage_parameters(world,guard):
    parameters=torch.zeros(world.env.defender_num,5)
    parameters[guard,2]=1.
    return parameters


def program_bank(world,guard,mode='opening_teacher'):
    """Joint minimum travel/turn assignment to separated interception slots."""
    if mode=='persistent_interception':
        from experiments.fleet_feedback import parameters
        return torch.from_numpy(np.stack([parameters(world.state(),guard,p) for p in range(len(PROGRAMS))]))
    state=world.state();n=len(state)-1;free=[i for i in range(n) if i!=guard];m=len(free)
    u=state[-1,:2]/max(np.linalg.norm(state[-1,:2]),1e-6);v=np.array([-u[1],u[0]])
    bank=coverage_parameters(world,guard)[None].repeat(len(PROGRAMS),1,1)
    for program in range(1,len(PROGRAMS)):
        points=[]
        for j in range(m//2):
            if program==1:
                points.extend([(32.-10*j,-14.-6*j),(32.-10*j,14.+6*j)])
            else:
                sign=1. if program==2 else -1.
                points.extend([(36.-8*j,sign*(12.+10*j)),(22.-4*j,-sign*(18.-6*j))])
        if m%2:points.append((3.,0.))
        coordinates=np.asarray(points,dtype=np.float32);goals=coordinates[:,:1]*u+coordinates[:,1:]*v
        delta=goals[None]-state[free,None,:2]
        angle=np.arctan2(delta[...,1],delta[...,0])-state[free,None,2]
        turning=np.abs(np.arctan2(np.sin(angle),np.cos(angle)))
        cost=np.linalg.norm(delta,axis=-1)+4*turning
        boats,slots=linear_sum_assignment(cost)
        for boat,slot in zip(boats,slots):
            bank[program,free[int(boat)],:2]=torch.as_tensor(goals[int(slot)],dtype=torch.float32)
            bank[program,free[int(boat)],2]=2.
        bank[program,:,3]=program
        bank[program,:,4]=1.  # State-triggered joint closing is part of this program.
    return bank


def fleet_forces(state,reference,parameters,closed,mode='opening_teacher'):
    if mode=='persistent_interception':
        from experiments.fleet_feedback import forces
        force,_=forces(state,reference,parameters)
        return force,torch.zeros_like(closed)
    enemy=state[:,-1,:2];direction=enemy/enemy.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    delta=state[:,:-1,:2]-enemy[:,None]
    opening=parameters[...,2]>1.5;guard=(parameters[...,2]>.5)&~opening
    ahead=(delta*direction[:,None]).sum(-1)>-3.
    close=delta.norm(dim=-1)<25.
    closed=closed|((ahead&close&opening).any(-1))
    waypoint=torch.where(guard[...,None],(12.5*direction[:,None]).expand_as(parameters[...,:2]),parameters[...,:2])
    tracking=tracking_thrust(state,waypoint)
    use_tracking=guard|(opening&~closed[:,None])
    return torch.where(use_tracking[...,None],tracking,reference).clamp(-500,1000),closed


class FleetReference:
    def __init__(self,teacher,source=None,mode='opening_teacher'):
        self.teacher=teacher;self.source=source;self.mode=mode
        self.parameters=None;self.closed=False;self.cached=None

    @torch.no_grad()
    def act(self,world):
        state=torch.as_tensor(world.state(),dtype=torch.float32)[None]
        parameters=self.parameters[None]
        if self.mode=='persistent_interception' and int(self.parameters[0,3])>0:
            learned,_,_=self.source.commands(world.env)
            force,_=fleet_forces(state,(750*learned+250)[None],parameters,torch.tensor([False]),self.mode)
            self.cached=None
            return force[0].numpy(),dict(evaluated_unique=1,forward_batches=1,
                formation_closed=False,feedback_phase='persistent_joint_interception_and_closing')
        _,closed=fleet_forces(state,torch.zeros(1,world.env.defender_num,2),parameters,torch.tensor([self.closed]))
        self.closed=bool(closed[0]);opening=bool((self.parameters[:,2]>1.5).any())
        if not opening or self.closed:
            if self.cached is None:reference,trace=self.teacher.act(world)
            else:reference,trace=self.cached
        else:
            reference=np.zeros((world.env.defender_num,2),np.float32)
            trace=dict(evaluated_unique=0,forward_batches=0)
        self.cached=None
        force,_=fleet_forces(state,torch.as_tensor(reference,dtype=torch.float32)[None],parameters,closed)
        return force[0].numpy(),dict(trace,formation_closed=self.closed,
            feedback_phase='joint_pursuit' if self.closed or not opening else 'joint_interception_positions')


class FleetExecutor(ViabilityController):
    def __init__(self,source,teacher,mode='opening_teacher'):
        self.source=source;self.teacher=FleetReference(teacher,source,mode)
        self.model=SimpleNamespace(cfg=SimpleNamespace(guard_margin=6.,guard_weight=16.))
        self.horizon,self.iterations=10,12

    def configure(self,parameters,closed=False):
        self.teacher.parameters=parameters.cpu();self.teacher.closed=closed;self.teacher.cached=None


class FleetNominalStep(nn.Module):
    def __init__(self,source):
        super().__init__();self.net=source.net

    def forward(self,state,legacy,gates,parameters,closed,first_force,first):
        observations,boids=feedback.source_observations(state,legacy)
        learned=feedback.source_actions(self.net,observations)
        reference=750*(gates*learned+(1-gates)*boids)+250
        force,closed=fleet_forces(state,reference,parameters,closed)
        force=torch.where(first>.5,first_force,force)
        thrust=torch.cat((force,feedback.attacker_thrust(state,2.25)[:,None]),1)
        state,legacy=feedback.integrate(state,thrust)
        return state,legacy,closed


class PersistentNominalStep(nn.Module):
    """Compile source control and integration; assignment stays current feedback."""
    def __init__(self,source):
        super().__init__();self.net=source.net

    def forward(self,state,legacy,gates,parameters,first_force,first):
        observations,boids=feedback.source_observations(state,legacy)
        learned=feedback.source_actions(self.net,observations)
        reference=750*(gates*learned+(1-gates)*boids)+250
        reference=torch.where((parameters[:,0,3]>0)[:,None,None],750*learned+250,reference)
        tracked=tracking_thrust(state,parameters[...,:2]);mix=parameters[...,4,None]
        force=torch.where((parameters[...,2]>.5)[...,None],mix*reference+(1-mix)*tracked,reference)
        force=torch.where(first>.5,first_force,force)
        thrust=torch.cat((force,feedback.attacker_thrust(state,2.25)[:,None]),1)
        return feedback.integrate(state,thrust)


@torch.no_grad()
def nominal_paths(world,source,gates,parameters,closed,first_force,horizon,mode='opening_teacher'):
    device=next(source.net.parameters()).device
    C,N=parameters.shape[:2]
    state=torch.as_tensor(world.state(),dtype=torch.float32,device=device)[None].expand(C,-1,-1).clone()
    gates,parameters,closed,first_force=(x.to(device) for x in (gates,parameters,closed,first_force))
    legacy=state[...,3:5].clone();flag=state.new_ones(C,1,1)
    if mode=='persistent_interception':
        from experiments.fleet_feedback import parameters as feedback_parameters
        if not hasattr(source,'_fleet_persistent_nominal'):source._fleet_persistent_nominal={}
        key=(C,N,str(device))
        if key not in source._fleet_persistent_nominal:
            source._fleet_persistent_nominal[key]=torch.jit.freeze(torch.jit.trace(PersistentNominalStep(source).eval(),
                (state,legacy,gates,parameters,first_force,flag),check_trace=False))
        roles=parameters.detach().cpu().numpy()
        guards=[int(np.flatnonzero(row[:,2]==1)[0]) for row in roles]
        programs=[int(round(row[0,3])) for row in roles];paths=[]
        for q in range(horizon):
            current=torch.as_tensor(np.stack([feedback_parameters(observed,guard,program)
                for observed,guard,program in zip(state.detach().cpu().numpy(),guards,programs)]),dtype=state.dtype,device=device)
            state,legacy=source._fleet_persistent_nominal[key](state,legacy,gates,current,first_force,flag if q==0 else torch.zeros_like(flag))
            paths.append(state[...,:2])
        return torch.stack(paths,-2).cpu()
    if not hasattr(source,'_fleet_nominal'):source._fleet_nominal={}
    key=(C,N,str(device))
    if key not in source._fleet_nominal:
        source._fleet_nominal[key]=torch.jit.freeze(torch.jit.trace(FleetNominalStep(source).eval(),
            (state,legacy,gates,parameters,closed,first_force,flag),check_trace=False))
    paths=[]
    for q in range(horizon):
        state,legacy,closed=source._fleet_nominal[key](state,legacy,gates,parameters,
            closed,first_force,flag if q==0 else torch.zeros_like(flag))
        paths.append(state[...,:2])
    return torch.stack(paths,-2).cpu()


@torch.no_grad()
def make_input(world,source,teacher,executor,guard,cfg):
    reference,teacher_trace=teacher.act(world)
    learned,boids,_=source.commands(world.env)
    parameters=program_bank(world,guard,cfg.feedback_mode);C,N=parameters.shape[:2]
    first=[];initial_closed=[];traces=[]
    for row in parameters:
        executor.configure(row);executor.teacher.cached=(reference,teacher_trace)
        force,trace=executor.act(world)
        first.append(torch.as_tensor(force,dtype=torch.float32))
        initial_closed.append(executor.teacher.closed);traces.append(trace)
    first=torch.stack(first);closed=torch.tensor(initial_closed,dtype=torch.bool)
    gates=torch.as_tensor(teacher_trace['gates'],dtype=torch.float32)[None].expand(C,-1,-1).clone()
    state=torch.as_tensor(world.state(),dtype=torch.float32)[None].expand(C,-1,-1).clone()
    references=torch.as_tensor(reference,dtype=torch.float32)[None].expand(C,-1,-1)
    if cfg.feedback_mode=='persistent_interception':
        references=torch.where((parameters[:,0,3]>0)[:,None,None],(750*learned+250)[None],references)
    raw,_=fleet_forces(state,references,parameters,torch.zeros(C,dtype=torch.bool),cfg.feedback_mode)
    tracking=(parameters[...,2]>.5)&~((parameters[...,2]>1.5)&closed[:,None])
    encoded_gates=torch.where(tracking[...,None],torch.zeros_like(gates),gates)
    rules=torch.where(tracking[...,None],(raw-250)/750,boids[None].expand(C,-1,-1))
    if cfg.feedback_mode=='persistent_interception':
        encoded_gates=torch.where((parameters[:,0,3]>0)[:,None,None],parameters[...,4,None].expand(-1,-1,2),encoded_gates)
        rules=torch.where(tracking[...,None],(tracking_thrust(state,parameters[...,:2])-250)/750,boids[None].expand(C,-1,-1))
    available=min(cfg.horizon,int(math.ceil((world.protocol.time_limit-world.env.Current_T-1e-8)/world.protocol.control_dt)))
    if available<1:raise ValueError('Cannot select a fleet program after task termination')
    nominal=nominal_paths(world,source,gates,parameters,closed,first,available,cfg.feedback_mode)
    if available<cfg.horizon:nominal=torch.cat((nominal,nominal[...,-1:,:].expand(-1,-1,cfg.horizon-available,-1)),-2)
    history,positions,hv,rotation=world.observation();R=torch.as_tensor(rotation,dtype=torch.float32)
    encoded_parameters=parameters.clone();encoded_parameters[...,:2]=parameters[...,:2]@R
    encoded_parameters=encoded_parameters/encoded_parameters.new_tensor([40,40,2,3,1])
    candidate=torch.cat((encoded_gates,learned[None].expand(C,-1,-1),rules,(first-250)/750,encoded_parameters),-1)
    candidate=torch.cat((candidate,candidate.new_zeros(C,1,13)),1)
    batch=dict(history=history[None],positions=positions[None],history_valid=hv[None],
        entity_valid=torch.ones(1,N+1,dtype=torch.bool),candidate=candidate[None],nominal=(nominal@R)[None],
        known_time_valid=(torch.arange(cfg.horizon)<available)[None],
        time_context=torch.tensor([[(world.protocol.time_limit-world.env.Current_T)/world.protocol.time_limit,
                                    available/cfg.horizon]],dtype=torch.float32))
    # The four projections reuse one teacher evaluation at this root.
    for c,trace in enumerate(traces):
        trace['evaluated_unique']=int(teacher_trace.get('evaluated_unique',0)) if c==0 else 0
        trace['forward_batches']=int(trace['projection_objective_calls'])
    traces[0]['forward_batches']+=int(teacher_trace.get('forward_batches',0))+available+1
    if cfg.feedback_mode=='persistent_interception':
        for trace in traces[1:]:trace['forward_batches']+=1
    return batch,parameters,first,closed,rotation,traces


class FleetOutcomeTransformer(FusionOutcomeTransformer):
    def __init__(self,cfg):
        super().__init__(cfg);w=cfg.width
        if cfg.outcome_readout=='physical_distributions':
            self.fleet_safety=nn.Sequential(nn.Linear(3*w+3,w),nn.GELU(),nn.Linear(w,5))
            nn.init.zeros_(self.fleet_safety[-1].weight);nn.init.zeros_(self.fleet_safety[-1].bias)
            with torch.no_grad():self.fleet_safety[-1].bias[[1,2,4]]=torch.tensor([9.,.5413,-2.])
            if cfg.candidate_comparison:
                features=2*(3*w+9)
                self.bank_embed=nn.Sequential(nn.LayerNorm(features),nn.Linear(features,w),nn.GELU())
                self.bank_layers=nn.ModuleList([Attention(w,cfg.heads),Attention(w,cfg.heads)])
                self.bank_output=nn.Linear(w,3)
                nn.init.zeros_(self.bank_output.weight);nn.init.zeros_(self.bank_output.bias)
            return
        features=7 if cfg.outcome_readout=='distribution_only' else 3*w+10
        self.terminal=nn.Sequential(nn.Linear(features,w),nn.GELU(),nn.Linear(w,4))
        nn.init.zeros_(self.terminal[-1].weight)
        with torch.no_grad():self.terminal[-1].bias.copy_(torch.tensor([math.log(.45),math.log(.54),math.log(.01),0.]))

    def validate_runtime(self,source,world):
        super().validate_runtime(source,world)
        if world is not None:
            tick=self.cfg.decision_time/world.protocol.control_dt
            if self.cfg.decision_time>=world.protocol.time_limit or abs(tick-round(tick))>1e-6:
                raise ValueError('Fleet decision time must be a legal tick before the task limit')
            interval=self.cfg.reselect_seconds/world.protocol.control_dt
            if abs(interval-round(interval))>1e-6:
                raise ValueError('Fleet reselection must use legal control ticks')

    def forward(self,batch):
        out=super().forward(batch);N=batch['positions'].shape[1]-1
        _,j=torch.triu_indices(N+1,N+1,1,device=out['mu'].device);defenders=j<N
        if self.cfg.outcome_readout=='physical_distributions':
            nominal_pair,nominal_team,nominal_radius,nominal_time,_,_=nominal_statistics(batch)
            nominal_safety=torch.minimum(nominal_pair[...,defenders].amin(-1),nominal_radius/3.)
            safety_parameters=self.fleet_safety(out['configuration_embedding'])
            safety=hurdle_parameters(nominal_safety,safety_parameters,5.)
            if self.cfg.candidate_comparison:
                statistics=torch.stack((nominal_safety/10,nominal_team/20,nominal_time,
                    safety['event_logit']/8,out['team_event_logit']/8,out['time_fraction']),-1)
                absolute=torch.cat((out['configuration_embedding'],statistics),-1)
                centered=absolute-absolute.mean(1,keepdim=True)
                scale=centered.square().mean(1,keepdim=True).sqrt().detach().clamp_min(.05)
                tokens=self.bank_embed(torch.cat((absolute,centered/scale),-1))
                if self.cfg.candidate_attention:
                    available=torch.ones(tokens.shape[:2],dtype=torch.bool,device=tokens.device)
                    for layer in self.bank_layers:tokens=layer(tokens,available)
                correction=self.bank_output(tokens)
                safety_parameters=safety_parameters.clone();safety_parameters[...,0]+=correction[...,0]
                team_parameters=out['team_parameters'].clone();team_parameters[...,0]+=correction[...,1]
                team_parameters[...,10]+=correction[...,2]
                safety=hurdle_parameters(nominal_safety,safety_parameters,5.)
                contact=hurdle_parameters(nominal_team,team_parameters[...,:5],5.)
                out.update({'team_'+key:value for key,value in contact.items()})
                remaining=batch['time_context'][:,0,None]
                out['time_fraction']=1-remaining+remaining*team_parameters[...,10].sigmoid()
                out['candidate_comparison_correction']=correction
            out.update({'safety_'+key:value for key,value in safety.items()})
            # The safety minimum is below5 exactly when collision or breach occurs.
            # Team contact is modeled conditional on avoiding that terminal failure.
            failure=safety['probability'];capture=(1-failure)*out['team_probability']
            probability=torch.stack((capture,(1-failure)*(1-out['team_probability']),failure),-1)
            capture_time=out['time_fraction']
            utility=4*capture+3*probability[...,1]-12*failure+self.cfg.capture_time_weight*capture*(1-capture_time)
            out.update(outcome_logits=probability.clamp_min(1e-8).log(),outcome_probabilities=probability,
                terminal_capture_time=capture_time,utility=utility)
            return out
        geometry=torch.stack((out['team_probability'],out['radius_probability'],
            out['team_excess_mean'].clamp_max(200)/20,out['mu'][...,defenders].amin(-1)/20,
            out['sigma'][...,defenders].mean(-1)/5,out['pair_probability'][...,defenders].sum(-1),out['time_fraction']),-1)
        features=geometry if self.cfg.outcome_readout=='distribution_only' else torch.cat((out['configuration_embedding'],geometry),-1)
        terminal=self.terminal(features)
        probability=terminal[...,:3].softmax(-1);capture_time=terminal[...,3].sigmoid()
        utility=4*probability[...,0]+3*probability[...,1]-12*probability[...,2]+self.cfg.capture_time_weight*probability[...,0]*(1-capture_time)
        out.update(outcome_logits=terminal[...,:3],outcome_probabilities=probability,
            terminal_capture_time=capture_time,utility=utility)
        return out


def load_policy(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=FleetOutcomeTransformer(FleetConfig(**payload['config']));model.load_state_dict(payload['state'])
    model.physical_protocol=payload['physical_protocol']
    model.source_identity=payload.get('source_identity',{})
    model.teacher_sha256=recorded_teacher_identity(payload,model.cfg.teacher_checkpoint)
    return model.to(device).eval()


class FleetController(Controller):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.cfg=self.model.cfg
        self.teacher=load_teacher(self.source,self.cfg.teacher_checkpoint,getattr(self.model,'teacher_sha256',''))
        self.executor=FleetExecutor(self.source,self.teacher,self.cfg.feedback_mode)
        self.guard=None;self.selected=False;self.program=0;self.last_time=None;self.next_decision=self.cfg.decision_time;self.handed_off=False

    def act(self,world):
        self.model.validate_runtime(self.source,world)
        started=time.perf_counter();now=world.env.Current_T
        if self.guard is None or self.last_time is not None and now<=self.last_time:
            self.guard=initial_guard(world);self.selected=False;self.program=0
            self.next_decision=self.cfg.decision_time;self.handed_off=False
            self.executor.configure(coverage_parameters(world,self.guard))
        decision={};root=now+1e-7>=self.next_decision and (not self.selected or self.cfg.reselect_seconds>0)
        if root:
            batch,parameters,first,closed,_,traces=make_input(world,self.source,self.teacher,self.executor,self.guard,self.cfg)
            with torch.no_grad():output=self.model({k:v.to(self.device) for k,v in batch.items()})
            self.program=self.cfg.fixed_program if self.cfg.fixed_program>=0 else int(output['utility'][0].argmax())
            if self.cfg.selection_rule=='observable_handoff':
                state=world.state();enemy=state[-1,:2];direction=enemy/max(np.linalg.norm(enemy),1e-6)
                relative=np.delete(state[:-1,:2],self.guard,axis=0)-enemy
                self.handed_off=self.handed_off or int(((relative@direction>-3.)&(np.linalg.norm(relative,axis=-1)<28.)).sum())>=min(2,len(relative)-1)
                self.program=0 if self.handed_off else self.cfg.handoff_program
            if self.program>=len(parameters):raise ValueError('Fixed fleet program is outside the complete candidate bank')
            self.executor.configure(parameters[self.program],bool(closed[self.program]));self.selected=True
            self.next_decision=now+self.cfg.reselect_seconds if self.cfg.reselect_seconds>0 else float('inf')
            force=first[self.program].numpy();physical=traces[self.program]
            decision=dict(outcome_probabilities=output['outcome_probabilities'][0].cpu().tolist(),
                utilities=output['utility'][0].cpu().tolist(),parameters=parameters[self.program].tolist(),
                predicted_capture_seconds=(world.protocol.time_limit*output['terminal_capture_time'][0]).cpu().tolist(),
                evaluated_unique=len(parameters)+sum(int(t.get('evaluated_unique',0)) for t in traces),
                forward_batches=1+sum(int(t.get('forward_batches',0)) for t in traces))
            if 'safety_probability' in output:
                decision.update(predicted_failure_probabilities=output['safety_probability'][0].cpu().tolist(),
                    predicted_safety_minimum=output['safety_mu'][0].cpu().tolist(),
                    predicted_contact_given_no_failure=output['team_probability'][0].cpu().tolist())
        else:
            force,physical=self.executor.act(world)
            decision.update(evaluated_unique=int(physical.get('evaluated_unique',0)),
                forward_batches=int(physical.get('forward_batches',0)))
        trace=dict(decision,selected_program=self.program,program_name=PROGRAMS[self.program],
            selection_rule=self.cfg.selection_rule,observable_handoff=self.handed_off,
            root_selection=root,selection_committed=self.selected,coverage_guard=self.guard,
            formation_closed=self.executor.teacher.closed,feedback_phase=physical['feedback_phase'],
            physical_thrust=force.tolist(),projection_applied=physical['projection_applied'],
            predicted_pair_after=physical['predicted_pair_after'],predicted_coverage_after=physical['predicted_coverage_after'],
            compute_seconds=time.perf_counter()-started,revision=self.cfg.revision)
        self.last_time=now
        return force,trace


def main():
    import citf.evaluate as evaluation
    original=evaluation.runtime_identity
    def identity(*args,**kwargs):
        value=original(*args,**kwargs);path=args[2] if len(args)>2 else kwargs['checkpoint']
        config=torch.load(path,map_location='cpu',weights_only=False)['config']
        value.update(method='CITF-whole-fleet-interception-development',revision=config['revision'])
        value['controller'].update(configuration=config,search='exhaustive complete four-program bank',
            programs=list(PROGRAMS),selection='Transformer terminal expected utility,or explicitly fixed program',
            execution='same2s coverage prefix; selected joint assignment and feedback program through actual terminal; source/role/projection feedback every0.2s',
            learning='all deployed candidates,paired random futures to actual terminal; full-remaining-task synchronous distances and outcomes;32step motion auxiliary',
            nominal='reactive frozen source actor and Boids; root fusion coefficients; fixed2.25 APF; learned correction for projection and future joint-fusion changes')
        from experiments import fleet_feedback
        value['runtime_code']=canonical_hash(dict(base=value['runtime_code'],fleet_policy=file_hash(__file__),feedback=file_hash(fleet_feedback.__file__)))
        if config.get('feedback_mode')=='persistent_interception':
            value['controller'].update(execution='persistent observed-state joint assignment,orbital approach,blocking and source-fused striker through actual terminal',
                programs=['persistent_coverage','symmetric_orbital_interception','left_orbital_interception','right_orbital_interception'],
                reselection_seconds=config.get('reselect_seconds',0.),
                selection_rule=config.get('selection_rule','learned'),
                nominal_implementation='scripted source,tracking,and known-physics kernel; joint assignment recomputed from each predicted current state')
        return value
    def load(path,source,world,device):
        model=load_policy(path,device);validate_predictor_protocol(model.physical_protocol,world.protocol);return model
    evaluation.runtime_identity=identity;evaluation.load_behavior=load;evaluation.Controller=FleetController
    evaluation.main()


if __name__=='__main__':main()
