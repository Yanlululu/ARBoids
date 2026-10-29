"""Mechanism-one controller shared by numerical evaluation and VRX."""
import study_runtime  # Configure numerical runtimes before NumPy and PyTorch.
import argparse
import hashlib
import json
import inspect
import os
from math import isfinite
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from source_arboids import numerical_source, source_mixture, SOURCE_COMMIT


def outcome(env, done):
    return dict(success=int(done > 2), capture=int(done == 3), collision=int(done == 2),
        source_failure=int(done == 1), timeout=int(done == 4),
        physical_breach=int(np.linalg.norm(env.attacker.pos) < env.Target_R),
        duration=float(env.Current_T), outcome=int(done))


def mechanism_tensor(packet,device='cpu'):
    return {k:torch.as_tensor(value,device=device,dtype=torch.float32).unsqueeze(0) for k,value in packet.items()}


def joint_packet(env, obs):
    """Public measurements used by the source actor and candidate forecasts."""
    motion=np.asarray([[b.x,b.y,b.theta,*b.velocity] for b in env.defender_list])
    p,v,heading=motion[:,:2],motion[:,3:5],motion[:,2]
    c,s=np.cos(heading),np.sin(heading)
    def local(vector):
        return np.stack((c*vector[:,0]+s*vector[:,1],-s*vector[:,0]+c*vector[:,1]),-1)
    n=len(p)
    node=np.concatenate((local(-p)/30.,local(env.attacker.pos-p)/30.,
        local(np.broadcast_to(env.attacker.vel,(n,2)))/5.,local(v)/5.,motion[:,5:6],
        np.zeros((n,2)),(env.boids_actions-250.)/750.,np.zeros((n,1)),
        np.full((n,1),n/6.),np.full((n,1),env.Current_T/80.)),axis=-1)
    return dict(node=np.clip(node,-6.,6.).astype(np.float32),obs=np.asarray(obs,dtype=np.float32),
        motion=motion.astype(np.float32),
        water=np.asarray([b.velocity_r for b in env.defender_list],dtype=np.float32)/5.)


def proposal_parameter_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PublicTargetMotion(nn.Module):
    """Predict finite-horizon target motion from public fleet measurements.

    Shared vessel embeddings permit a changing fleet size. Coordinates are
    relative to the target and the protected asset, so no scene identity,
    attacker controller, agility parameter, or realized future is an input.
    """
    def __init__(self,width=48,absolute_frame=False):
        super().__init__()
        self.absolute_frame=absolute_frame
        self.nodes=nn.Sequential(nn.Linear(9,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear(2*width+5+2*absolute_frame,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,4))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)

    @staticmethod
    def encode(packet,absolute_frame=False):
        motion=packet['motion'];c=motion[:,0,2].cos();s=motion[:,0,2].sin()
        def world(vector):
            return torch.stack((c*vector[:,0]-s*vector[:,1],s*vector[:,0]+c*vector[:,1]),-1)
        target=motion[:,0,:2]+world(packet['node'][:,0,2:4]*30.)
        velocity=world(packet['node'][:,0,4:6]*5.)
        return PublicTargetMotion.encode_state(motion,packet['node'][...,11:13],target,velocity,absolute_frame)

    @staticmethod
    def encode_state(motion,boids,target,velocity,absolute_frame=False):
        if (motion.ndim!=3 or motion.shape[-1]!=6 or boids.shape!=(*motion.shape[:2],2)
                or target.shape!=(len(motion),2) or velocity.shape!=target.shape):
            raise ValueError('Target-response inputs must contain public fleet poses, commands and target motion.')
        axis=-target/target.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        def rotate(vector):
            basis=axis[:,None] if vector.ndim==3 else axis
            return torch.stack(((vector*basis).sum(-1),basis[...,0]*vector[...,1]-basis[...,1]*vector[...,0]),-1)
        heading=torch.stack((motion[...,2].cos(),motion[...,2].sin()),-1)
        nodes=torch.cat((rotate(motion[...,:2]-target[:,None])/30.,
            rotate(motion[...,3:5]-velocity[:,None])/5.,rotate(heading),motion[...,5:6],
            boids),-1)
        global_state=torch.cat((target.norm(dim=-1,keepdim=True)/30.,rotate(velocity)/5.,
            target.new_full((len(target),1),motion.shape[1]/6.)),-1)
        if absolute_frame:
            # World bearing is observable. Removing it assumes rotationally
            # covariant target dynamics, which the source reset does not have.
            global_state=torch.cat((global_state,-axis),-1)
        return nodes,global_state,target,velocity,axis

    def forward(self,nodes,global_state,time):
        encoded=self.nodes(nodes)
        inputs=torch.cat((encoded.mean(1),encoded.amax(1),global_state,time[:,None]/3.2),-1)
        residual=6.*self.readout(inputs).tanh()
        return residual[:,:2]*time[:,None],residual[:,2:]*time[:,None].clamp_max(1.)

    def predict(self,packet,time):
        nodes,global_state,target,velocity,axis=self.encode(packet,self.absolute_frame)
        horizon=target.new_full((len(target),),float(time))
        displacement,delta_velocity=self(nodes,global_state,horizon)
        def world(vector):
            return torch.stack((axis[:,0]*vector[:,0]-axis[:,1]*vector[:,1],
                axis[:,1]*vector[:,0]+axis[:,0]*vector[:,1]),-1)
        return target+horizon[:,None]*velocity+world(displacement),velocity+world(delta_velocity)


class PublicTargetAcceleration(nn.Module):
    """Learn public target acceleration with explicit encounter geometry.

    The response couples displacement to the predicted velocity change while
    allowing a bounded correction for the measured control interval. Vessel
    features remain permutation invariant and contain no target controller,
    hidden heading, agility or realized future.
    """
    def __init__(self,width=48,absolute_frame=True):
        super().__init__()
        self.absolute_frame=bool(absolute_frame)
        self.nodes=nn.Sequential(nn.Linear(18,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.attention=nn.Linear(width,1)
        self.readout=nn.Sequential(nn.Linear(3*width+6+2*self.absolute_frame,64),nn.SiLU(),
            nn.Linear(64,64),nn.SiLU(),nn.Linear(64,4))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)

    def forward(self,nodes,global_state,time):
        relative=30.*nodes[...,:2];relative_velocity=5.*nodes[...,2:4]
        velocity=5.*global_state[:,1:3];speed=velocity.norm(dim=-1,keepdim=True)
        direction=(velocity/speed.clamp_min(.25))[:,None]
        distance=relative.norm(dim=-1,keepdim=True)
        radial=relative/distance.clamp_min(1.)
        def along(vector):return (vector*direction).sum(-1,keepdim=True)
        def across(vector):
            return direction[...,0:1]*vector[...,1:2]-direction[...,1:2]*vector[...,0:1]
        geometry=torch.cat((distance/30.,1./(1.+distance/5.),
            (relative_velocity*radial).sum(-1,keepdim=True)/5.,
            along(relative)/30.,across(relative)/30.,
            along(relative_velocity)/5.,across(relative_velocity)/5.,
            along(nodes[...,4:6]),across(nodes[...,4:6])),-1)
        encoded=self.nodes(torch.cat((nodes,geometry),-1))
        attention=self.attention(encoded).softmax(1)
        inputs=torch.cat((encoded.mean(1),encoded.amax(1),(attention*encoded).sum(1),
            global_state,speed/5.,time[:,None]/3.2),-1)
        response=self.readout(inputs).tanh()
        delta_velocity=20.*time[:,None]*response[:,:2]
        displacement=(.5*time[:,None]+.025)*delta_velocity+.25*(time[:,None]/.2)*response[:,2:]
        return displacement,delta_velocity


def target_feedback_network(width,absolute_frame,representation='pooled_residual'):
    if representation=='pooled_residual':return PublicTargetMotion(width,absolute_frame)
    if representation=='interaction_acceleration':return PublicTargetAcceleration(width,absolute_frame)
    raise ValueError('Unknown public target-response representation.')


class PublicTargetFeedback(nn.Module):
    """Advance the public target response separately inside every fleet forecast."""
    def __init__(self,width=48,states=(),absolute_frame=True,control_step=.2,scenario_mode='mean',
                 kinematic_weight=0.,representation='pooled_residual',scenario_groups=None,
                 scenario_weights=None):
        super().__init__()
        if not states or not np.isclose(control_step,.2):
            raise ValueError('Target feedback needs fitted members at the source control interval.')
        if scenario_mode not in ('mean','members','groups'):
            raise ValueError('Target feedback must use mean, member or grouped complete trajectories.')
        if scenario_mode=='groups':
            groups=tuple(tuple(group) for group in (scenario_groups or ()))
            indices=[index for group in groups for index in group]
            if (not groups or any(not group for group in groups)
                    or any(not isinstance(index,int) or isinstance(index,bool) for index in indices)
                    or sorted(indices)!=list(range(len(states)))):
                raise ValueError('Target scenario groups must partition all fitted members exactly once.')
        else:
            if scenario_groups is not None:
                raise ValueError('Target scenario groups require grouped complete trajectories.')
            groups=None
        count=len(groups) if groups is not None else (len(states) if scenario_mode=='members' else 1)
        if scenario_weights is not None:
            weights=np.asarray(scenario_weights,dtype=float)
            if (weights.shape!=(count,) or not np.isfinite(weights).all()
                    or (weights<=0.).any() or not np.isclose(weights.sum(),1.)):
                raise ValueError('Target scenario weights must be finite probabilities for every trajectory.')
            scenario_weights=tuple(float(weight) for weight in weights)
        if not np.isfinite(kinematic_weight) or not 0. <= kinematic_weight <= 1.:
            raise ValueError('The public kinematic scenario needs a finite weight between zero and one.')
        self.scenario_mode=scenario_mode
        self.scenario_groups=groups
        self.scenario_weights=scenario_weights
        self.kinematic_weight=float(kinematic_weight)
        self.representation=representation
        self.control_step=float(control_step);self.absolute_frame=bool(absolute_frame)
        self.models=nn.ModuleList([target_feedback_network(width,self.absolute_frame,representation) for _ in states])
        for network,state in zip(self.models,states):network.load_state_dict(state)

    def predict_state(self,motion,boids,target,velocity,component=None):
        nodes,global_state,target,velocity,axis=PublicTargetMotion.encode_state(
            motion,boids,target,velocity,self.absolute_frame)
        horizon=target.new_full((len(target),),self.control_step)
        if component is None:networks=self.models
        elif isinstance(component,int):networks=(self.models[component],)
        else:networks=tuple(self.models[index] for index in component)
        predictions=[network(nodes,global_state,horizon) for network in networks]
        displacement=torch.stack([value[0] for value in predictions]).mean(0)
        delta_velocity=torch.stack([value[1] for value in predictions]).mean(0)
        def world(vector):
            return torch.stack((axis[:,0]*vector[:,0]-axis[:,1]*vector[:,1],
                axis[:,1]*vector[:,0]+axis[:,0]*vector[:,1]),-1)
        return target+self.control_step*velocity+world(displacement),velocity+world(delta_velocity)

    def component(self,index):
        indices=(index,) if isinstance(index,int) else tuple(index)
        if (not indices or any(not isinstance(value,int) or isinstance(value,bool)
                or not 0<=value<len(self.models) for value in indices)):
            raise ValueError('A target trajectory component must identify fitted members.')
        parent=self
        class Component:
            def predict_state(self,motion,boids,target,velocity):
                return parent.predict_state(motion,boids,target,velocity,component=indices)
        return Component()

    def trajectory_components(self):
        """Preserve each expert's autoregressive state throughout its fleet future."""
        if self.scenario_mode=='groups':
            return tuple(self.component(group) for group in self.scenario_groups)
        if self.scenario_mode=='members':
            return tuple(self.component(index) for index in range(len(self.models)))
        return (self,)

    def kinematic_component(self):
        """Retain measured target momentum as a separate complete future.

        Averaging an uncertain learned acceleration into every forecast can
        hide a dangerous fleet response. This scenario advances only the
        public position and velocity; each candidate still refreshes the
        unchanged source and Boids feedback against that same future.
        """
        interval=self.control_step
        class Component:
            def predict_state(self,motion,boids,target,velocity):
                return target+interval*velocity,velocity
        return Component()


def joint_task_step(env,action,gamma=.995,step_cost=.001,attacker_action=None):
    # Common training objective for every new-method ablation. Author dynamics,
    # rewards and reported terminal classes remain untouched.
    potential=-float(env.def_att_dists.min())/60.
    if attacker_action is None:
        obs,source_reward,done,_=env.step(action,'AdaRes')
    else:
        obs,source_reward,done,_=env.step(action,'AdaRes',att_action=attacker_action)
    following=0. if done else -float(env.def_att_dists.min())/60.
    reward=-step_cost+(1. if done>2 else -1. if done else 0.)+.2*(gamma*following-potential)
    return obs,float(reward),done,float(np.mean(source_reward))


class JointFusionActor(nn.Module):
    """Frozen source commands with an analytic fleet candidate selector."""
    def __init__(self, peer=True, levels=3, risk_prior=32., risk_horizon=3.2,
                 risk_clearance=7., risk_feedback=True, prediction_step=.1,
                 risk_tail=0., risk_barrier=0., interception_weight=0.):
        super().__init__()
        if (not isfinite(prediction_step) or prediction_step <= 0 or prediction_step > .2
                or not isfinite(risk_horizon) or risk_horizon < prediction_step
                or not isfinite(risk_clearance) or risk_clearance <= 0):
            raise ValueError('Prediction steps, horizon and clearance must be finite and positive.')
        if not np.isclose(.2/prediction_step,round(.2/prediction_step),rtol=0.,atol=1e-9):
            raise ValueError('Prediction steps must divide the 0.2-second physical control interval.')
        if any(not isfinite(value) or value < 0 for value in (risk_prior,risk_tail,risk_barrier,interception_weight)):
            raise ValueError('Prediction risk and interception weights must be finite and nonnegative.')
        if levels < 2:
            raise ValueError('Candidate prediction requires at least two levels.')
        self.peer=peer; self.levels=levels
        self.risk_horizon=risk_horizon
        self.risk_clearance=risk_clearance; self.risk_feedback=risk_feedback
        self.prediction_step=prediction_step; self.risk_tail=risk_tail; self.risk_barrier=risk_barrier
        self.prediction_relaxation=0.; self.prediction_iterations=0
        self.current_marginal_prediction=False; self.prediction_discount=0.; self.prediction_discount_start=1.6
        self.interception_weight=interception_weight; self.interception_horizon=0.
        self.target_motion=None; self.target_feedback=None; self.feedback_fleet_policy=None
        self.terminal_prediction=False; self.interaction_enabled=True
        self.generic_fusion_adapter=None
        self.environment_protocol='source'
        self.training_gamma=.995; self.training_step_cost=.001
        self.fusion_selection='joint'
        self.actuator_fusion_refinement=dict(risk_weight=risk_prior, preference_width=.4, beam_width=128,
            selection='joint', source_preference=True, prior_map='bounded_logit')
        self.base=numerical_source().ActorAdap(6,8,3,512)

    @torch.no_grad()
    def forward(self, packet, latent=None, deterministic=True, gate_only=False):
        if latent is not None or not deterministic:
            raise ValueError('M1 executes deterministic frozen source commands.')
        mean, _, hidden = self.proposal_distribution(packet['obs'])
        proposals = mean.tanh()
        own = self.base.activation(self.base.a1(torch.cat((proposals,packet['node'][...,11:13]),-1)))
        gates = self.base.adap_layer(torch.cat((own,hidden),-1)).squeeze(-1).tanh()*.5+.5
        settings = self.actuator_fusion_refinement
        if settings is not None and settings.get('prior_map') == 'bounded_logit':
            gates = (2.*torch.tanh(torch.logit(.02+.96*gates)/2.)).sigmoid()
        action = torch.cat((proposals,gates[...,None]),-1)
        return action, None, None, None, None, None

    @torch.no_grad()
    def control(self, packet, action=None):
        from actuator_fusion import refine_actuator_interactions
        source = self(packet)[0] if action is None else action
        if source.ndim != 3 or source.shape[-1] != 3:
            raise ValueError('M1 accepts one source action and preference per vessel.')
        if self.generic_fusion_adapter is not None:
            return self.generic_fusion_adapter.control(self, packet, source)
        if self.actuator_fusion_refinement is None:
            raise ValueError('The M1 controller has no candidate selector.')
        query = dict(packet, _preferred_actuator_coefficients=source[...,2:3].expand(-1,-1,2))
        return refine_actuator_interactions(self, query, source, source, **self.actuator_fusion_refinement)

    @torch.no_grad()
    def candidate_risk(self,packet,proposals,return_distance=False,pairwise=False,reversal_at=None,
                       candidate_grid=None,proposal_offsets=None,return_paths=False):
        motion=packet['motion'];b,n=motion.shape[:2]
        actuator_grid=candidate_grid is not None and candidate_grid.ndim==4
        k=self.levels if candidate_grid is None else candidate_grid.shape[-2 if actuator_grid else -1]
        if not self.interaction_enabled:
            if return_paths:return motion.new_zeros((b,n,k,round(self.risk_horizon/self.prediction_step),2))
            return motion.new_zeros((b,n,n,k,k) if self.peer or pairwise else (b,n,k))
        iterations=packet.get('_forecast_iteration',self.prediction_iterations)
        if iterations:
            # Recompute candidate responses against predicted peer motion.
            # The reference averages the same candidates under the source
            # preference, so joint and independent selectors share it exactly.
            previous=dict(packet,_forecast_iteration=iterations-1)
            trajectories=self.candidate_risk(previous,proposals,reversal_at=reversal_at,
                candidate_grid=candidate_grid,proposal_offsets=proposal_offsets,return_paths=True)
            hidden=self.proposal_distribution(packet['obs'])[2]
            own=self.base.activation(self.base.a1(torch.cat((proposals,packet['node'][...,11:13]),-1)))
            prior=.02+.96*(self.base.adap_layer(torch.cat((own,hidden),-1)).tanh().squeeze(-1)*.5+.5)
            center=(2.*torch.tanh(torch.logit(prior)/2.)).sigmoid()
            reference_grid=(torch.linspace(0.,1.,k,device=motion.device) if candidate_grid is None else candidate_grid)
            reference_grid=reference_grid[None,None] if reference_grid.ndim==1 else reference_grid
            departure=((reference_grid-center[...,None,None])/.4).square().mean(-1) if actuator_grid else (
                (reference_grid-center[...,None])/.4).square()
            weights=(-departure).softmax(-1)
            positions=(trajectories*weights[...,None,None]).sum(2)
            velocities=((positions[:,:,1:]-positions[:,:,:-1])/self.prediction_step)
            velocities=(torch.cat((velocities,velocities[:,:,-1:]),2) if velocities.shape[2]
                else motion[:,:,None,3:5])
            packet['_candidate_peer_reference']=(positions,velocities)
        grid=(torch.linspace(0.,1.,k,device=motion.device) if candidate_grid is None else candidate_grid)
        grid=grid[None,None] if grid.ndim==1 else grid
        candidate=(proposals[:,:,None] if proposal_offsets is None else
            (proposals[:,:,None]+proposal_offsets[None,None]).clamp(-1.,1.))
        mixture=grid if actuator_grid else grid[...,None]
        force=750.*(mixture*candidate+
            (1.-mixture)*packet['node'][:,:,None,11:13])+250.
        predicted_motion=self.prediction_motion(packet)
        x,y,heading,vx,vy,r=[predicted_motion[...,i,None].expand(-1,-1,k).clone() for i in range(6)]
        paths=[];dt=self.prediction_step
        feedback_steps=max(1,round(.2/dt))
        for step in range(round(self.risk_horizon/dt)):
            if self.risk_feedback and step and step%feedback_steps==0:
                state=torch.stack((x,y,heading,vx,vy,r),-1).transpose(1,2)
                forecast_obs,boids=self.feedback_observation(packet,state,step*dt)
                mean,_,hidden=self.proposal_distribution(forecast_obs.reshape(b*k,n,-1))
                future=mean.tanh().reshape(b,k,n,2).transpose(1,2)
                if proposal_offsets is not None:future=(future+proposal_offsets[None,None]).clamp(-1.,1.)
                current_grid=1.-grid if reversal_at is not None and step*dt>=reversal_at else grid
                mixture=current_grid if actuator_grid else current_grid[...,None]
                relaxation=getattr(self,'prediction_relaxation',0.)
                if relaxation>0.:
                    # A receding fusion decision is not a promise to hold its
                    # coefficient for the entire forecast. Continue toward the
                    # shared source Adapter as feedback observations change.
                    own=self.base.activation(self.base.a1(torch.cat((future.transpose(1,2),boids),-1)))
                    native=self.base.adap_layer(torch.cat((own,hidden.reshape(b,k,n,-1)),-1)).tanh()*.5+.5
                    continuation=(2.*torch.tanh(torch.logit(.02+.96*native)/2.)).sigmoid().transpose(1,2)
                    weight=1.-np.exp(-step*dt/relaxation)
                    relaxed=(1.-weight)*mixture+weight*continuation
                    mixture=relaxed
                force=750.*(mixture*future+(1.-mixture)*boids.transpose(1,2))+250.
            x=x+dt*vx;y=y+dt*vy;heading=heading+dt*r
            c,s=heading.cos(),heading.sin();u=c*vx+s*vy;v=-s*vx+c*vy
            du=(400.*r*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
            dv=(-380.*r*u-(100.+150.*v.abs())*v)/400.
            dr=(-20.*u*v-(980.+950.*r.abs())*r+1.25*(force[...,0]-force[...,1]))/1430.
            u=u+dt*du;v=v+dt*dv;vx=c*u-s*v;vy=s*u+c*v;r=r+dt*dr
            paths.append(torch.stack((x,y),-1))
        paths=torch.stack(paths,-2)
        if packet.get('_retain_candidate_paths', False):
            packet['_retained_candidate_paths'] = paths
        if return_paths:return paths
        if self.terminal_prediction:
            target,target_velocity=self.public_target_motion(packet)
            times=torch.arange(1,paths.shape[-2]+1,device=motion.device)*dt
            target_path=(torch.stack([self.target_motion.predict(packet,float(t))[0] for t in times],1)
                if self.target_motion is not None else target[:,None]+times[None,:,None]*target_velocity[:,None])
            packet['_terminal_paths']=(paths,target_path)
        if self.interception_weight:
            target,target_velocity=self.public_target_motion(packet)
            times=torch.arange(1,paths.shape[-2]+1,device=motion.device)*dt
            target_path=(torch.stack([self.target_motion.predict(packet,float(t))[0] for t in times],1)
                if self.target_motion is not None else target[:,None]+times[None,:,None]*target_velocity[:,None])
            # The closest eligible interceptor determines capture; summing
            # every vessel's pursuit cost would spuriously reward crowding.
            reach=(paths-target_path[:,None,None]).norm(dim=-1)
            if self.interception_horizon:
                reach=reach[...,:max(1,round(self.interception_horizon/dt))]
            packet['_candidate_interception']=reach.amin(-1).clamp_min(5.)
        if self.peer or pairwise:
            delta=paths[:,:,None,:,None]-paths[:,None,:,None]
            distance=delta.norm(dim=-1);minimum=distance.amin(-1)
            if return_distance:return minimum
            if self.prediction_discount:
                time=torch.arange(1,distance.shape[-1]+1,device=motion.device)*dt
                weight=torch.exp(-(time-self.prediction_discount_start).clamp_min(0.)/self.prediction_discount)
                return (self.distance_risk(distance)*weight).amax(-1)
            return self.distance_risk(minimum)
        time=torch.arange(1,len(paths[0,0,0])+1,device=motion.device)*dt
        constant=motion[:,:,:2][:,:,None]+time[None,None,:,None]*motion[:,:,3:5][:,:,None]
        delta=paths[:,:,None]-constant[:,None,:,None]
        distance=delta.norm(dim=-1);minimum=distance.amin(-1)
        mask=~torch.eye(n,dtype=torch.bool,device=motion.device)[None,:,:,None]
        if self.prediction_discount:
            weight=torch.exp(-(time-self.prediction_discount_start).clamp_min(0.)/self.prediction_discount)
            return ((self.distance_risk(distance)*weight).amax(-1)*mask).amax(2)
        return (self.distance_risk(minimum)*mask).amax(2)


    def distance_risk(self,minimum):
        risk=(self.risk_clearance-minimum).clamp_min(0.).square()
        if self.risk_barrier:
            # Separate an imminent collision from several benign clearances.
            # The broad interaction cost still shapes early coordination;
            # this smooth contact term prevents exchanging one unsafe pair
            # for many small reductions in already safe pair costs.
            risk=risk+self.risk_barrier*F.softplus((5.5-minimum)/.3).square()
        return risk


    def public_target_motion(self,packet):
        motion=packet['motion'];c=motion[:,0,2].cos();s=motion[:,0,2].sin()
        def rotate(vector):
            return torch.stack((c*vector[:,0]-s*vector[:,1],s*vector[:,0]+c*vector[:,1]),-1)
        target=motion[:,0,:2]+rotate(packet['node'][:,0,2:4]*30.)
        return target,rotate(packet['node'][:,0,4:6]*5.)


    def prediction_motion(self,packet):
        """Predict new-step motion after marginalizing the observed current.

        The source task draws a new zero-mean current at each control step.
        Water-relative velocity is persistent; the last ground-relative
        velocity also contains the previous transient disturbance.
        """
        motion=packet['motion']
        if not self.current_marginal_prediction:return motion
        return torch.cat((motion[...,:3],5.*packet['water']),-1)


    def feedback_observation(self,packet,state,time,joint=False,target_override=None):
        """Original source observations at nominal candidate poses.

        Each candidate refreshes the unchanged source neural and Boids commands;
        neighbors follow public constant-velocity forecasts. These are predictor
        inputs only. No new controller or simulator-private state is executed.
        """
        motion=packet['motion'];b,k,n=state.shape[:3]
        position=state[...,:2];velocity=state[...,3:5];heading=state[...,2]
        peers=motion[:,:,:2]+time*motion[:,:,3:5]
        peer_velocity=motion[:,:,3:5]
        reference=packet.get('_candidate_peer_reference')
        if reference is not None and not joint:
            index=min(max(round(time/self.prediction_step)-1,0),reference[0].shape[2]-1)
            peers=reference[0][:,:,index]
            peer_velocity=reference[1][:,:,index]
        relative=(position[:,:,None]-position[:,:,:,None] if joint else
            peers[:,None,None]-position[:,:,:,None])
        distance=relative.norm(dim=-1)
        mask=~torch.eye(n,dtype=torch.bool,device=state.device)[None,None]
        separation=(-relative/(distance-3.5).clamp_min(.01)[...,None].square()
            *((distance<15.)&mask)[...,None]).sum(-2)
        alignment=(velocity.mean(2,keepdim=True).expand_as(velocity) if joint else
            (peer_velocity.sum(1)[:,None,None]-peer_velocity[:,None]+velocity)/n)
        cohesion=(position.mean(2,keepdim=True)-position if joint else
            (peers.sum(1)[:,None,None]-peers[:,None]+position)/n-position)
        h0=motion[:,0,2];c0,s0=h0.cos(),h0.sin()
        def world(vector):
            return torch.stack((c0*vector[:,0]-s0*vector[:,1],s0*vector[:,0]+c0*vector[:,1]),-1)
        attacker=motion[:,0,:2]+world(packet['node'][:,0,2:4]*30.)
        att_velocity=world(packet['node'][:,0,4:6]*5.)
        target=attacker+time*att_velocity
        if self.target_motion is not None and self.interaction_enabled and time:
            target,att_velocity=self.target_motion.predict(packet,time)
        if target_override is not None:
            target,att_velocity=target_override
            if target.shape not in ((b,2),(b,k,2)) or att_velocity.shape!=target.shape:
                raise ValueError('A public feedback scenario needs one target state per batch or fleet plan.')
        target_position=target[:,None,None] if target.ndim==2 else target[:,:,None]
        target_velocity=att_velocity[:,None,None] if att_velocity.ndim==2 else att_velocity[:,:,None]
        attraction=target_position-position
        field=.5*attraction+10.*separation+.1*alignment+.1*cohesion
        field=field/field.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        c,s=heading.cos(),heading.sin()
        forward=c*field[...,0]+s*field[...,1]
        turn=torch.atan2(-s*field[...,0]+c*field[...,1],forward)
        thrust=(1000.*torch.stack((forward+.75*turn,forward-.75*turn),-1)).clamp(-500.,1000.)
        boids=(thrust-250.)/750.
        def polar(vector):
            angle=torch.atan2(vector[...,1],vector[...,0])-heading
            return torch.stack((vector.norm(dim=-1),torch.remainder(angle+np.pi,2*np.pi)-np.pi),-1)
        neighbor_angle=torch.atan2(relative[...,1],relative[...,0])-heading[...,None]
        neighbor_polar=torch.stack((distance,torch.remainder(neighbor_angle+np.pi,2*np.pi)-np.pi),-1)
        neighbors=neighbor_polar[:,:,mask[0,0]].reshape(b,k,n,2*(n-1))
        obs=torch.cat((polar(-position),polar(attraction),
            polar(target_velocity.expand_as(position)),polar(separation),
            polar(alignment),polar(cohesion),boids,neighbors),-1)
        return obs,boids


    def proposal_distribution(self,obs):
        if (obs.ndim != 3 or obs.shape[1] < 2
                or obs.shape[2] != 14+2*(obs.shape[1]-1)):
            raise ValueError('Source observations need exactly one slot per other defender.')
        b,n,d=obs.shape;net=self.base;act=net.activation
        x=torch.cat((act(net.f1(obs[...,:6])),act(net.f2(obs[...,6:14])),
            act(net.t1(obs[...,14:].reshape(b,n,-1,2))).mean(-2)),-1)
        h=act(net.l2(act(net.l1(x))))
        pm=net.mean_layer(h);ps=net.log_std_layer(h).clamp(-3.,-.3).exp()
        return pm,ps,h


RETIRED_VALUE_FIELDS = (
    'successor_task_value', 'finite_temporal_credit', 'continuous_program_value',
    'response_feedback_value', 'local_conditional_value', 'physical_conditional_value',
    'physical_conditional_values', 'response_base', 'shared_response_value',
    'factual_return_support', 'conditional_effect_support', 'response_action_support',
    'joint_intervention_value', 'joint_intervention_values', 'midpoint_intervention_values',
    'initial_role_credit', 'feedback_role_credit', 'cooperative_role_credit', 'program_proposer',
)


def retired_value_fields(record):
    """Identify retired model state without interpreting tensor truth values."""
    present = []
    for key in RETIRED_VALUE_FIELDS:
        value = record.get(key)
        if value is None or isinstance(value, (dict, list, tuple)) and not value:
            continue
        present.append(key)
    return present


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def initialize():
    """Initialize a worker without changing the caller's autograd mode."""
    torch.set_num_threads(1)


def joint_load(path, revision=None, *, source=None, device='cpu'):
    """Load M1 weights and reject checkpoints that still execute mechanism two."""
    from actuator_fusion import FeedbackFleetPolicy, refine_actuator_interactions
    path = Path(path)
    if revision is not None and path.stat().st_mtime_ns != revision:
        raise ValueError('The selected checkpoint changed before it was loaded.')
    data = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(data, dict) or 'actor' not in data:
        raise ValueError('Expected an M1 fusion checkpoint with frozen source weights.')
    retired = retired_value_fields(data)
    if retired:
        raise ValueError('Mechanism two has been removed; use an M1-only checkpoint. Retired fields: '+', '.join(retired))
    if any(data.get(key,False) for key in ('continuous_gate_learning','conditional_commitment',
            'immediate_commitment','fixed_intervention_clock','physical_interventions',
            'physical_singletons','physical_intervention_reversal','preference_strength')):
        raise ValueError('Mechanism two execution options are not supported by the M1 controller.')
    if data.get('source_commit', SOURCE_COMMIT) != SOURCE_COMMIT:
        raise ValueError('The checkpoint uses a different original-source release.')
    flags = data.get('mechanism_ablation', (False, True, False))
    if (not isinstance(flags, (tuple, list)) or len(flags) > 3
            or any(not isinstance(flag, (bool, np.bool_)) for flag in flags)):
        raise ValueError('Mechanism ablation must contain Boolean removal flags.')
    if any(flag for index, flag in enumerate(flags) if index != 1):
        raise ValueError('This checkpoint removes M1; use the original source controller for that baseline.')
    if source is not None:
        identity = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if data.get('source_sha256') != identity:
            raise ValueError('The deployment source does not match the checkpoint source hash.')
    settings = data.get('actuator_fusion_refinement')
    generic = data.get('generic_fusion_adapter')
    if (settings is None) == (generic is None):
        raise ValueError('Choose exactly one analytic selector or matched generic fusion adapter.')
    if settings is not None:
        if not isinstance(settings, dict):
            raise ValueError('Actuator selection settings must be a dictionary.')
        settings = dict(settings)
        if settings.pop('candidate_output', None) is not None or settings.get('return_coefficients', False):
            raise ValueError('Checkpoint control must return physical commands, not diagnostic candidate data.')
        # This historical flag had no effect in a model without interventions.
        settings.pop('commit_interventions', None)
        if not settings.get('source_preference', False):
            raise ValueError('M1 requires the same source preference in joint and independent selection.')
        if settings.get('prior_map') not in (None, 'bounded_logit'):
            raise ValueError('Unknown source preference map.')
        allowed = {key for key,value in inspect.signature(refine_actuator_interactions).parameters.items()
                   if value.kind == inspect.Parameter.KEYWORD_ONLY}
        if set(settings)-allowed:
            raise ValueError('Unknown actuator settings: '+', '.join(sorted(set(settings)-allowed)))
        if settings.get('selection','joint') not in ('joint','independent','single_reference','balanced_independent'):
            raise ValueError('Unknown actuator selection mode.')
        if (not isinstance(settings.get('beam_width',128),int) or settings.get('beam_width',128) < 1
                or not isfinite(settings.get('risk_weight',32.)) or settings.get('risk_weight',32.) < 0
                or not isfinite(settings.get('preference_width',.4)) or settings.get('preference_width',.4) <= 0):
            raise ValueError('Invalid candidate beam, risk weight or preference width.')
    if (data.get('gate_period',1) != 1 or data.get('intent_strength',0.)
            or data.get('coupled_shortlist',0) or data.get('forecast_inputs',False)):
        raise ValueError('This checkpoint uses a retired scalar or multi-step fusion controller.')
    constructor = inspect.signature(JointFusionActor).parameters
    options = {key:data[key] for key in constructor if key in data}
    if data.get('format') == 'joint-neural-source-ppo-v1':
        # Preserve omitted-field semantics in the existing M1 checkpoints.
        for key,value in dict(levels=2,risk_prior=0.,risk_horizon=1.6,risk_clearance=6.,risk_feedback=False).items():
            options.setdefault(key,value)
    actor = JointFusionActor(**options)
    weights = data['actor']
    if not isinstance(weights, dict):
        raise ValueError('The checkpoint actor state must be a parameter dictionary.')
    # Older M1 checkpoints include unused, zero-trained adapter/critic weights.
    # M1 only runs the unchanged source ActorAdap before its analytic selector.
    base = {key[5:]:value for key,value in weights.items() if key.startswith('base.')}
    if not base:
        raise ValueError('The M1 checkpoint is missing its source actor parameters.')
    actor.base.load_state_dict(base, strict=True)
    # Keep the historical linear-kernel dispatch, which also depends on
    # requires_grad under no_grad. Control never records source gradients,
    # and fleet training optimizes only its separate candidate scorer.
    actor.base.requires_grad_(not data.get('frozen_proposal', False))
    actor.actuator_fusion_refinement = settings
    for key in ('prediction_relaxation', 'prediction_iterations', 'current_marginal_prediction',
                'prediction_discount', 'prediction_discount_start', 'terminal_prediction',
                'interception_horizon', 'environment_protocol', 'fusion_selection'):
        if key in data:
            setattr(actor, key, data[key])
    if (not isinstance(actor.prediction_iterations,int) or actor.prediction_iterations < 0
            or not isfinite(actor.prediction_relaxation) or not 0. <= actor.prediction_relaxation <= 1.
            or any(not isfinite(value) or value < 0 for value in
                   (actor.prediction_discount,actor.prediction_discount_start,actor.interception_horizon))):
        raise ValueError('Invalid candidate forecast iteration, relaxation or discount settings.')
    if actor.environment_protocol not in ('source', 'paper-parameters-v1'):
        raise ValueError('Unknown numerical environment protocol.')
    if data.get('pair_feedback_prediction'):
        raise ValueError('The retired scalar pair predictor is incompatible with actuator-level M1.')
    if generic is not None:
        from mechanism_controls import GenericFusionAdapter
        actor.generic_fusion_adapter = GenericFusionAdapter(**generic)
    if data.get('feedback_fleet_policy') is not None:
        if not (settings or {}).get('task_feedback_prediction'):
            raise ValueError('A fleet scoring model requires complete candidate feedback forecasts.')
        actor.feedback_fleet_policy = FeedbackFleetPolicy(**data['feedback_fleet_policy'])
    if data.get('target_motion') is not None:
        record = data['target_motion']
        actor.target_motion = PublicTargetMotion(record.get('width',48), record.get('absolute_frame',False))
        actor.target_motion.load_state_dict(record['state'], strict=True)
    if data.get('target_feedback') is not None:
        actor.target_feedback = PublicTargetFeedback(**data['target_feedback'])
    actor.source_sha256 = data.get('source_sha256')
    actor.proposal_initialization_sha256 = data.get('proposal_initialization_sha256')
    actor.training_gamma = 1. if data.get('success_objective',False) else .995
    actor.training_step_cost = 0. if data.get('success_objective',False) else .001
    return actor.to(device).eval()


class JointMeasuredController:
    """The same M1 control calculation on public ROS measurements."""
    def __init__(self, checkpoint, source=None, inference_device='cpu'):
        self.actor = joint_load(checkpoint, source=source, device=inference_device)
        if self.actor.current_marginal_prediction:
            raise ValueError('This model needs water-relative velocity; the ROS adapter only measures ground velocity.')
        self.reset()

    def reset(self):
        self.steps = 0
        self.last_elapsed = None

    def control(self, defenders, attacker, observations, boids, elapsed, total_time=80.):
        defenders = np.asarray(defenders, dtype=float)
        attacker = np.asarray(attacker, dtype=float)
        observations = np.asarray(observations, dtype=np.float32)
        boids = np.asarray(boids, dtype=float)
        n = len(defenders)
        if n < 2 or defenders.shape != (n,6) or attacker.shape != (6,) or boids.shape != (n,2):
            raise ValueError('M1 needs at least two defender poses/velocities and two Boids thrusts per vessel.')
        # The pinned ROS observer appends an unused zero peer. It must not
        # enter the source actor's peer mean and alter the numerical policy.
        if observations.shape == (n,14+2*n):
            if np.any(observations[:,-2:] != 0.):
                raise ValueError('The ROS observation padding slot must be zero.')
            observations = observations[:,:-2]
        if observations.shape != (n,14+2*(n-1)):
            raise ValueError('Public observations do not match the numerical source policy.')
        if not all(np.isfinite(value).all() for value in (defenders,attacker,observations,boids)):
            raise ValueError('Control measurements must be finite.')
        expected_time = 80. if self.actor.environment_protocol == 'source' else 60.
        if (not isfinite(elapsed) or elapsed < 0. or not isfinite(total_time)
                or abs(total_time-expected_time) > 1e-6):
            raise ValueError('The measured clock must use the numerical task horizon.')
        if self.last_elapsed is not None and elapsed <= self.last_elapsed:
            raise ValueError('Each control call needs a new measurement tick; reset between episodes.')

        def boat(state):
            return SimpleNamespace(x=float(state[0]), y=float(state[1]), theta=float(state[2]),
                pos=state[:2], vel=state[3:5], velocity=state[3:6], velocity_r=state[3:6], agility=1.)

        env = SimpleNamespace(defender_list=[boat(row) for row in defenders], attacker=boat(attacker),
            boids_actions=boids, Current_T=float(elapsed), Total_T=float(total_time))
        packet = mechanism_tensor(joint_packet(env,observations), next(self.actor.parameters()).device)
        with torch.no_grad():
            physical = self.actor.control(packet)[0].cpu().numpy()
        self.steps += 1
        self.last_elapsed = float(elapsed)
        return source_mixture(physical,boids), dict(control_steps=self.steps,physical_action=physical.copy())


def joint_episode(task):
    """Evaluate a complete paired layout/future using the original dynamics."""
    scene, path, future = task
    seed, n, agility = scene
    if n < 2:
        raise ValueError('The source policy requires at least two vessels.')
    np.random.seed(int(seed)); torch.manual_seed(int(seed))
    actor = joint_load(path)
    if actor.environment_protocol == 'source':
        env = numerical_source().TADEnv(n)
    else:
        from envs.TADgame import TADEnv
        env = TADEnv(n, protocol=actor.environment_protocol)
    obs, _ = env.reset(agility=agility,noisy_agility=False)
    if future is not None:
        np.random.seed(int(future))
    done = 0; steps = 0; total = 0.; discount = 1.
    while not done:
        with torch.no_grad():
            action = actor.control(mechanism_tensor(joint_packet(env,obs)))[0].numpy()
        obs, rewards, done, _ = env.step(action,'AdaRes')
        total += discount*float(np.mean(rewards)); discount *= .99; steps += 1
    row = dict(seed=int(seed),defenders=n,agility=agility,**outcome(env,done),
        discounted_return=total,physical_steps=steps,environment_protocol=actor.environment_protocol,
        completed=bool(done))
    if future is not None:
        row['future_seed'] = int(future)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    create = sub.add_parser('create',help='Build an M1 selector from an original source ActorAdap checkpoint.')
    create.add_argument('--source',type=Path,required=True)
    create.add_argument('--output',type=Path,required=True)
    create.add_argument('--selection',choices=('joint','independent'),default='joint')
    create.add_argument('--risk-weight',type=float,default=32.)
    create.add_argument('--preference-width',type=float,default=.4)
    create.add_argument('--horizon',type=float,default=3.2)
    create.add_argument('--clearance',type=float,default=7.)
    create.add_argument('--task-feedback',action='store_true',help='Use complete feedback forecasts for fleet-policy training.')
    train = sub.add_parser('train',help='Learn M1 whole-fleet selection from complete on-policy episodes.')
    train.add_argument('--checkpoint',type=Path,required=True)
    train.add_argument('--output',type=Path,required=True)
    train.add_argument('--iterations',type=int,default=24)
    train.add_argument('--episodes-per-iteration',type=int,default=8)
    train.add_argument('--defenders',type=int,default=6)
    train.add_argument('--batch-size',type=int,default=1)
    train.add_argument('--device',default='cpu')
    train.add_argument('--seed',type=int,default=42)
    collect = sub.add_parser('collect',help='Collect factual M1 trajectories and public prediction roots.')
    collect.add_argument('--checkpoint',type=Path,required=True)
    collect.add_argument('--tasks',type=Path,required=True)
    collect.add_argument('--output',type=Path,required=True)
    collect.add_argument('--batch-size',type=int,default=1)
    collect.add_argument('--device',default='cpu')
    collect.add_argument('--seed',type=int,default=42)
    motion = sub.add_parser('fit-motion',help='Fit public target forecasts on factual M1 trajectories.')
    motion.add_argument('--checkpoint',type=Path,required=True)
    motion.add_argument('--data',type=Path,required=True)
    motion.add_argument('--output',type=Path,required=True)
    motion.add_argument('--updates',type=int,default=6000)
    motion.add_argument('--device',default='cpu')
    motion.add_argument('--seed',type=int,default=42)
    evaluate = sub.add_parser('evaluate',help='Evaluate M1 on paired layout/future seeds.')
    evaluate.add_argument('--checkpoint',type=Path,required=True)
    evaluate.add_argument('--tasks',type=Path,help='JSON array of [[layout_seed, defenders, agility], future_seed].')
    evaluate.add_argument('--episodes',type=int,default=10)
    evaluate.add_argument('--defenders',type=int,default=6)
    evaluate.add_argument('--agility',type=float,default=2.)
    evaluate.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'create':
        if (not isfinite(args.risk_weight) or args.risk_weight < 0
                or not isfinite(args.preference_width) or args.preference_width <= 0):
            parser.error('Risk weight must be finite and nonnegative; preference width must be positive.')
        actor = JointFusionActor(risk_horizon=args.horizon,risk_clearance=args.clearance)
        actor.base.load_state_dict(torch.load(args.source,map_location='cpu',weights_only=True),strict=True)
        settings = dict(actor.actuator_fusion_refinement,selection=args.selection,
            risk_weight=args.risk_weight,preference_width=args.preference_width)
        if args.task_feedback:
            settings.update(task_stopping=True,task_feedback_prediction=True,synchronized_prediction=True)
            if args.selection=='independent':
                settings['selection']='single_reference'
        record = dict(format='joint-actuator-fusion-v1',actor=actor.state_dict(),
            source_commit=SOURCE_COMMIT,source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
            risk_horizon=args.horizon,risk_clearance=args.clearance,risk_feedback=True,
            actuator_fusion_refinement=settings,environment_protocol='source')
        atomic_save(record,args.output)
        print(args.output.resolve())
    elif args.command == 'train':
        from fusion_training import fit_feedback_fleet_policy
        fit_feedback_fleet_policy(args.checkpoint,args.output,seed=args.seed,iterations=args.iterations,
            episodes_per_iteration=args.episodes_per_iteration,batch_size=args.batch_size,
            device=args.device,training_defenders=args.defenders)
    elif args.command == 'collect':
        from fusion_training import collect_factual_data
        tasks=json.loads(args.tasks.read_text(encoding='utf-8'))
        collect_factual_data(args.checkpoint,tasks,args.output,seed=args.seed,
            batch_size=args.batch_size,device=args.device)
    elif args.command == 'fit-motion':
        from fusion_prediction import fit_target_motion
        fit_target_motion(args.checkpoint,args.data,args.output,updates=args.updates,
            seed=args.seed,device=args.device)
    elif args.command == 'evaluate':
        if args.episodes < 1 or args.defenders < 2:
            parser.error('Evaluation needs positive episodes and at least two defenders.')
        tasks = (json.loads(args.tasks.read_text(encoding='utf-8')) if args.tasks else
            [((args.seed+i,args.defenders,args.agility),args.seed+100000+i) for i in range(args.episodes)])
        if not tasks:
            parser.error('The evaluation task list is empty.')
        rows = [joint_episode((scene,str(args.checkpoint),future)) for scene,future in tasks]
        print(json.dumps(dict(completed=all(row['completed'] for row in rows),
            episodes=len(rows),success=sum(row['success'] for row in rows),rows=rows),indent=2))


if __name__ == '__main__':
    main()
