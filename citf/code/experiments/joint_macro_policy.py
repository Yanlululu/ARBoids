"""Learned joint fusion decisions throughout a task, with short execution.

An option either continues the retained controller for three ticks or executes
a current joint fusion/feedback role for three ticks. Scores are policy logits,
not predicted terminal values. Corrected short physical trajectories explicitly
enter the policy readout. Complete returns train their relative selection.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import torch
from torch import nn

from citf.bridge import Controller
from citf.contracts import Protocol,canonical_hash,file_hash
from experiments.contact_event_policy import retained_teacher
from experiments.interaction_dynamics import InteractionLayer,load as load_dynamics
from experiments.interception_fusion import InterceptionFusionController
from experiments.synchronized_configuration_policy import joint


@dataclass(frozen=True)
class MacroConfig:
    revision:str='CITF-whole-task-current-joint-fusion-policy-v1'
    teacher_checkpoint:str=''
    dynamics_checkpoint:str=''
    width:int=64
    layers:int=2
    heads:int=4
    feature_dim:int=105
    short_horizon:int=3
    hold_ticks:int=3
    alternative_prior:float=.25
    new_candidate_context:bool=True


def short_geometry(position,valid):
    # [B,C,E,T,2]. Every quantity is a short physical distance, never a value.
    distance=(position[:,:,:-1]-position[:,:,-1:]).norm(dim=-1).amin(2)
    radius=position.norm(dim=-1)
    gap=(radius[:,:,:-1]-radius[:,:,-1:]).amin(2)
    n=position.shape[2]-1;i,j=torch.triu_indices(n,n,1,device=position.device)
    separation=(position[:,:,i]-position[:,:,j]).norm(dim=-1).amin(2)
    return torch.cat((distance/20*valid,gap/20*valid,separation/10*valid),-1)


class MacroPolicy(nn.Module):
    def __init__(self,cfg=MacroConfig()):
        super().__init__();self.cfg=cfg
        self.embedding=nn.Sequential(nn.Linear(cfg.feature_dim,cfg.width),nn.SiLU(),nn.Linear(cfg.width,cfg.width))
        self.layers=nn.ModuleList([InteractionLayer(cfg.width,cfg.heads) for _ in range(cfg.layers)])
        self.motion=nn.Sequential(nn.LayerNorm(cfg.width),nn.Linear(cfg.width,2*cfg.short_horizon))
        self.actor=nn.Sequential(nn.LayerNorm(3*cfg.width+6*cfg.short_horizon),
            nn.Linear(3*cfg.width+6*cfg.short_horizon,cfg.width),nn.SiLU(),nn.Linear(cfg.width,1))
        nn.init.zeros_(self.motion[-1].weight);nn.init.zeros_(self.motion[-1].bias)
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)

    def forward(self,batch,*,policy=True):
        feature=batch['features'];roots,count,entities,_=feature.shape
        value=self.embedding(feature).reshape(roots*count,entities,self.cfg.width)
        relative=batch['relative'].reshape(roots*count,entities,entities,5)
        for layer in self.layers:value=layer(value,relative,self.cfg.new_candidate_context)
        value=value.reshape(roots,count,entities,self.cfg.width)
        residual=.5*self.motion(value).tanh().reshape(roots,count,entities,self.cfg.short_horizon,2)
        position=batch['base_position']+residual
        if not policy:
            return dict(position=position)
        geometric=short_geometry(position,batch['time_valid'])
        pooled=torch.cat((value[:,:,:-1].mean(2),value[:,:,:-1].amax(2),value[:,:,-1],
            geometric,geometric-geometric[:,:1]),-1)
        score=self.actor(pooled).squeeze(-1)
        score=score-score[:,:1]
        prior=torch.full_like(score,-self.cfg.alternative_prior);prior[:,0]=0
        logits=(score+prior).masked_fill(~batch['candidate_valid'],-torch.inf)
        return dict(logits=logits,position=position)


class ObservedRoot:
    """Only saved observations and known elapsed time; no simulator internals."""
    def __init__(self,history,elapsed,horizon=3):
        self.history=history
        self.env=SimpleNamespace(Current_T=elapsed)
        self.protocol=Protocol(prediction_steps=horizon)
    def state(self):return self.history[-1]


def candidate_pool(original,all_plans,additional):
    plans=[]
    for row in [*all_plans,*additional]:
        key=tuple(map(int,row))
        if key not in plans:plans.append(key)
    # Reference is a different short feedback option even if its first command
    # equals an alternative's held configuration.
    return np.asarray([tuple(map(int,original)),*plans],dtype=np.int64)


@torch.no_grad()
def policy_features(planner,history,elapsed,factor,plans):
    device=planner.source.device
    root=ObservedRoot(history,elapsed,3)
    paths,_=planner.rollout(root,factor,plans)
    known=paths.shape[2]
    if known<3:paths=torch.cat((paths,paths[:,:,-1:].expand(-1,-1,3-known,-1)),2)
    current=torch.as_tensor(np.asarray(history[-1]),dtype=torch.float32,device=device)
    raw_history=torch.as_tensor(np.asarray(history),dtype=torch.float32,device=device)[None]
    encoded=planner.model.encode(raw_history,torch.tensor([80-elapsed],device=device))[0]
    angle=torch.atan2(-current[-1,1],-current[-1,0]);c,s=angle.cos(),angle.sin()
    rotation=torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
    position=current[:,:2]@rotation;velocity=current[:,3:5]@rotation;yaw=current[:,2]-angle
    entity_role=current.new_zeros((len(current),1));entity_role[-1]=1
    observed=torch.cat((position/60,velocity/5,yaw.sin()[:,None],yaw.cos()[:,None],current[:,5:6],entity_role),-1)
    paths=paths@rotation
    count,entities,horizon,_=paths.shape;n=entities-1
    tensor_plans=torch.as_tensor(plans,dtype=torch.long,device=device)
    force=planner.forces(current[None].expand(count,-1,-1),current[None,:,3:5].expand(count,-1,-1),factor,tensor_plans,0)
    normalized=(force-250)/750
    difference=normalized-normalized[:1]
    learned=factor['learned'].to(device);boids=factor['rules'].to(device);span=learned-boids
    safe=torch.where(span.abs()>1e-6,span,torch.ones_like(span))
    gates=((normalized-boids[None])/safe[None]).clamp(0,1)
    gates=torch.where(span.abs()[None]>1e-6,gates,factor['prior'].to(device)[None])
    def pad(value):return torch.cat((value,value.new_zeros((count,1,value.shape[-1]))),1)
    mode=torch.stack((tensor_plans<9,tensor_plans==9,tensor_plans==10),-1).to(paths.dtype)
    flag=paths.new_zeros((count,entities,1));flag[0]=1
    relative_attacker=paths-paths[:,-1:]
    radius=paths.norm(dim=-1)
    pair=(paths[:,:,None]-paths[:,None,:]).norm(dim=-1)
    diagonal=torch.eye(entities,dtype=torch.bool,device=device)[None,:,:,None]
    minimum=pair.masked_fill(diagonal,torch.inf).amin(2)
    time_valid=(torch.arange(3,device=device)<known)[None].expand(count,-1)
    future=torch.cat(((paths-paths[:1])/.5,relative_attacker/20,
        ((radius-radius[:,-1:])/20)[...,None],(minimum/10)[...,None],
        (relative_attacker.norm(dim=-1)/20)[...,None]),-1)
    future=(future*time_valid[:,None,:,None]).flatten(2)
    feature=torch.cat((encoded[None].expand(count,-1,-1),observed[None].expand(count,-1,-1),
        pad(normalized),pad(difference),pad(gates),pad(span[None].expand(count,-1,-1)),pad(mode),flag,future),-1)
    final=paths[:,:,-1];delta=final[:,:,None]-final[:,None,:];distance=delta.norm(dim=-1)
    radial=final.norm(dim=-1)
    relative=torch.cat((delta/20,(distance/20)[...,None],(1/(distance+1))[...,None],
        ((radial[:,:,None]-radial[:,None,:])/20)[...,None]),-1)
    return dict(features=feature,relative=relative,base_position=paths,
        time_valid=time_valid,candidate_valid=torch.ones(count,dtype=torch.bool,device=device)),rotation


def load_policy(path,device='cpu'):
    value=torch.load(path,map_location='cpu',weights_only=False)
    model=MacroPolicy(MacroConfig(**value['config'])).to(device)
    model.load_state_dict(value['state']);model.eval()
    model.training_update=value.get('update',-1)
    return model


class JointMacroController(Controller):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        teacher=retained_teacher(self.model.cfg.teacher_checkpoint)
        self.teacher=joint.JointPolicyController(self.source,teacher,self.device,self.bank,self.budget,self.rounds,self.risk_mode)
        dynamics=load_dynamics(self.model.cfg.dynamics_checkpoint,self.device)
        self.planner=InterceptionFusionController(self.source,dynamics,self.device,self.bank,self.budget,self.rounds,self.risk_mode)
        self.active=None

    @torch.no_grad()
    def act(self,world):
        started=time.perf_counter();tick=int(round(world.env.Current_T/.2))
        if self.active is None or tick-self.active['tick']>=self.model.cfg.hold_ticks:
            reference_force,reference_trace=self.teacher.act(world)
            self.planner.cached=None
            _,proposal_trace=self.planner.act(world)
            factor=self.planner.last_factor
            reference=np.asarray(reference_trace['selected_indices'],np.int64)
            original=np.asarray(reference_trace['configurations'],np.int64)
            order=np.argsort(np.asarray(reference_trace['scores']),kind='stable')
            leading=next((original[i] for i in order if not np.array_equal(original[i],reference)),reference)
            plans=candidate_pool(reference,proposal_trace['configurations'],[leading])
            history=np.stack(world.history)
            history=np.concatenate((np.repeat(history[:1],8-len(history),0),history),0)
            batch,_=policy_features(self.planner,history,world.env.Current_T,factor,plans)
            prediction=self.model({k:v[None] for k,v in batch.items()})
            logits=prediction['logits'][0];choice=int(logits.argmax())
            self.active=dict(tick=tick,choice=choice,plan=plans[choice],factor=factor)
            trace=dict(evaluated_unique=len(plans),forward_batches=1,configurations=plans.tolist(),
                scores=(-logits).cpu().tolist(),selected_option=choice,selected_indices=plans[choice].tolist(),
                decision_update='joint_policy',reference_option=0,hold_ticks=self.model.cfg.hold_ticks,
                proposal_prediction_candidates=proposal_trace['evaluated_unique'],
                reference_score_candidates=reference_trace['evaluated_unique'],
                policy_score_candidates=len(plans),
                total_score_invocations=len(plans)+proposal_trace['evaluated_unique']+reference_trace['evaluated_unique'])
            force=reference_force if choice==0 else self._active_force(world)
        elif self.active['choice']==0:
            force,trace=self.teacher.act(world)
            trace.update(selected_option=0,decision_update='reference_feedback_within_current_option',
                joint_policy_candidates=0,hold_ticks=self.model.cfg.hold_ticks)
        else:
            force=self._active_force(world)
            trace=dict(evaluated_unique=0,forward_batches=0,configurations=[],scores=[],
                selected_option=self.active['choice'],selected_indices=self.active['plan'].tolist(),
                decision_update='current_joint_fusion_feedback',hold_ticks=self.model.cfg.hold_ticks)
        trace.update(physical_thrust=force.tolist(),revision=self.model.cfg.revision,
            option_started_tick=self.active['tick'],compute_seconds=time.perf_counter()-started)
        return force,trace

    def _active_force(self,world):
        factor=dict(self.active['factor'])
        factor['learned'],factor['rules'],factor['prior']=self.source.commands(world.env)
        state=torch.as_tensor(world.state(),dtype=torch.float32,device=self.device)[None]
        if int(round(world.env.Current_T/.2))>self.active['tick']:
            factor['intercept_target']=(state[0,-1,:2][None]+factor['lead_seconds'].to(self.device)[:,None]*state[0,-1,3:5][None]).cpu()
        plan=torch.as_tensor(self.active['plan'],device=self.device)[None]
        return self.planner.forces(state,state[...,3:5],factor,plan,0)[0].cpu().numpy()


def main():
    import citf.evaluate as evaluation
    import experiments.interception_fusion as fusion
    import experiments.interception_reserve as reserve
    import experiments.continuous_interception as continuous
    import experiments.interaction_dynamics as dynamics
    import experiments.batched_feedback as feedback
    original=evaluation.runtime_identity
    def identity(*args,**kwargs):
        result=original(*args,**kwargs)
        checkpoint=args[2] if len(args)>2 else kwargs.get('checkpoint')
        value=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=value['config']
        result.update(method='CITF-whole-task-joint-fusion-policy-development',revision=cfg['revision'],
            schema='short_joint_fusion_options_with_observed_interaction_features-v1')
        result['controller'].update(hold_ticks=cfg['hold_ticks'],
            actions='reference feedback or current joint fusion/feedback role, every three control periods throughout the task',
            policy='learned configuration logits; explicit corrected short motion enters readout; no terminal value/outcome head',
            teacher_checkpoint_sha256=file_hash(cfg['teacher_checkpoint']),
            dynamics_checkpoint_sha256=file_hash(cfg['dynamics_checkpoint']),
            training_update=value.get('update'),initialization='zero relative actor; reference is greedy by construction',
            source_domain='original SAC/Boids per-channel convex combinations',
            future_inputs='no future action,current,RNG,hidden agility or actual termination')
        result['runtime_code']=canonical_hash(dict(core=result['runtime_code'],policy=file_hash(__file__),
            fusion=file_hash(fusion.__file__),reserve=file_hash(reserve.__file__),continuous=file_hash(continuous.__file__),
            dynamics=file_hash(dynamics.__file__),feedback=file_hash(feedback.__file__)))
        return result
    evaluation.runtime_identity=identity
    evaluation.load_behavior=lambda checkpoint,source,world,device:load_policy(checkpoint,device)
    evaluation.Controller=JointMacroController
    evaluation.main()


if __name__=='__main__':main()
