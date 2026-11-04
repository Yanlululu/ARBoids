"""Joint interception and reserve control with explicit tracking authority.

New current actions supplement the original SAC/Boids fusion candidates. They
can turn toward a reachable lead point or retain an inner reserve, which cannot
in general be obtained by mixing two commands that both pursue the attacker.
The source and task remain unchanged; this is a new development identity.
"""
import argparse
import itertools
import sys
import time

import numpy as np
import torch

from citf.physics import forecast
from citf.search import whole_configuration_search
from experiments.batched_feedback import integrate,source_observations,source_actions,attacker_thrust
from experiments.continuous_interception import InterceptionController
from experiments.interaction_dynamics import load as load_dynamics
from experiments.root_response_diagnostic import physical_bank


def tracking_thrust(state,waypoint):
    """Observable velocity/yaw tracking with the known boat inertia and drag."""
    position,yaw,velocity=state[:,:-1,:2],state[:,:-1,2],state[:,:-1,3:6]
    delta=waypoint-position;distance=delta.norm(dim=-1,keepdim=True)
    desired=delta/distance.clamp_min(1e-6)*(distance*.8).clamp_max(3.2)
    pair=position[:,:,None]-position[:,None,:];separation=pair.norm(dim=-1)
    diagonal=torch.eye(position.shape[1],dtype=torch.bool,device=state.device)[None]
    active=(separation<9)&~diagonal
    repel=(pair/separation.clamp_min(1e-3)[...,None]
        *((9-separation).clamp_min(0)/(separation-3.5).clamp_min(.5).square()*3*active)[...,None]).sum(2)
    repel=repel/repel.norm(dim=-1,keepdim=True).clamp_min(1.5)*1.5
    desired=desired+repel
    desired_yaw=torch.atan2(desired[...,1],desired[...,0])
    error=torch.atan2((desired_yaw-yaw).sin(),(desired_yaw-yaw).cos())
    target_u=desired.norm(dim=-1).clamp_max(3.2)*error.cos().clamp_min(0)
    target_r=(1.5*error).clamp(-.8,.8)
    c,s=yaw.cos(),yaw.sin()
    u=c*velocity[...,0]+s*velocity[...,1]
    v=-s*velocity[...,0]+c*velocity[...,1]
    r=velocity[...,2]
    total=380*(target_u-u)/.7+(100+150*u.abs())*u-400*r*v
    difference=(1430*(target_r-r)/.35+(980+950*r.abs())*r+20*u*v)/1.25
    return torch.stack(((total+difference)/2,(total-difference)/2),-1).clamp(-500,1000)


class InterceptionReserveController(InterceptionController):
    revision='CITF-joint-interception-reserve-v2'

    @torch.no_grad()
    def prepare(self,world,factor,center):
        # Use a complete reference branch to locate a feasible lead point. The
        # selected candidate's own response is then re-predicted in full.
        enemy=super().rollout(world,factor,np.asarray([center]))[0][0,-1].cpu()
        state=torch.as_tensor(world.state(),dtype=torch.float32)
        position=state[:-1,:2]
        delta=enemy[None]-position[:,None]
        heading=torch.atan2(delta[...,1],delta[...,0])-state[:-1,2,None]
        turn=torch.atan2(heading.sin(),heading.cos()).abs()/.8
        travel=delta.norm(dim=-1)/3.2
        ticks=torch.arange(1,len(enemy)+1,dtype=torch.float32)*.2
        reach=(travel+.6*turn-ticks[None]).clamp_min(0)+.08*ticks[None]
        at=reach.argmin(-1)
        factor['lead_seconds']=ticks[at]
        factor['intercept_target']=enemy[at]
        # A declared maximum propulsion envelope, never the actual hidden
        # episode agility. It represents possible inward turns before a reserve
        # can recover, including an attacker that is currently moving outward.
        command=state.new_tensor([[3000,3000],[-1500,3000],[3000,-1500],[0,3000],[3000,0],[-1500,-1500]])
        attacker=state[-1:].expand(len(command),-1)
        envelope=forecast(attacker.double(),command.double(),world.protocol.prediction_steps,.2).float()
        factor['inward_radius']=envelope.norm(dim=-1).amin(0)
        return factor

    def forces(self,state,legacy,factor,plans,tick):
        device=state.device;n=plans.shape[1]
        index=plans.clamp_max(8)
        gates=factor['grid'].to(device)[torch.arange(n,device=device)[None],index]
        if tick==0:
            learned,boids=factor['learned'].to(device)[None],factor['rules'].to(device)[None]
        else:
            observations,boids=source_observations(state,legacy)
            learned=source_actions(self.source.net,observations)
        force=750*(gates*learned+(1-gates)*boids)+250
        if tick==0:
            intercept=factor['intercept_target'].to(device)[None].expand(len(plans),-1,-1)
        else:
            intercept=state[:,-1:,:2]+factor['lead_seconds'].to(device)[None,:,None]*state[:,-1:,3:5]
        attacker_position=state[:,-1:,:2]
        radius=attacker_position.norm(dim=-1,keepdim=True)
        reserve=attacker_position/radius.clamp_min(1e-6)*(radius-10).clamp(18,28)
        reserve=reserve.expand(-1,n,-1)
        target=torch.where((plans==10)[...,None],reserve,intercept)
        role_force=tracking_thrust(state,target)
        return torch.where((plans>=9)[...,None],role_force,force)

    @torch.no_grad()
    def rollout(self,world,factor,plans):
        device=self.source.device;plans=torch.as_tensor(plans,dtype=torch.long,device=device)
        count=len(plans);raw=np.stack(world.history)
        length=self.model.cfg.history if self.model is not None else 8
        raw=np.concatenate((np.repeat(raw[:1],max(0,length-len(raw)),0),raw),0)[-length:]
        history=torch.as_tensor(raw,dtype=torch.float32,device=device)[None].expand(count,-1,-1,-1).clone()
        state=history[:,-1];legacy=state[...,3:5];paths=[];sigmas=[]
        remaining=torch.full((count,),world.protocol.time_limit-world.env.Current_T,device=device)
        for tick in range(world.protocol.known_steps(world.env.Current_T)):
            force=self.forces(state,legacy,factor,plans,tick)
            if self.model is not None:
                prediction=self.model(history,force,(remaining-.2*tick).clamp_min(0))
                state,legacy=prediction['state'],prediction['legacy_velocity']
                sigmas.append(prediction['attacker_log_std'].exp().mean(-1)*2500)
            else:
                command=torch.cat((force,attacker_thrust(state,2.25)[:,None]),1)
                state,legacy=integrate(state,command)
            paths.append(state[...,:2]);history=torch.cat((history[:,1:],state[:,None]),1)
        return torch.stack(paths,2),(torch.stack(sigmas,1).mean(-1) if sigmas else None)

    def proposals(self,world,factor,physical_plans,physical_values,reference,center):
        plans=[]
        def add(row):
            row=tuple(map(int,row))
            if row not in plans:plans.append(row)
        add(center);add(reference)
        for i in np.argsort(physical_values,kind='stable')[:6]:add(physical_plans[i])
        for i in range(len(reference)):
            for role in [9,10]:
                row=reference.copy();row[i]=role;add(row)
        for i,j in itertools.permutations(range(len(reference)),2):
            row=reference.copy();row[i]=9;row[j]=10;add(row)
        return np.asarray(plans,dtype=np.int64)

    @torch.no_grad()
    def score_paths(self,world,factor,plans,paths):
        n=paths.shape[1]-1;horizon=paths.shape[2];device=paths.device
        distance=(paths[:,:-1]-paths[:,-1:]).norm(dim=-1)
        radius=paths.norm(dim=-1)
        i,j=torch.triu_indices(n,n,1,device=device)
        pair=(paths[:,i]-paths[:,j]).norm(dim=-1)
        collision=(7-pair.amin(-1)).clamp_min(0).square().sum(-1)
        # Reserve and collision protection remain active over the whole known
        # interval, including after an uncertain predicted capture.
        gap=(radius[:,:-1]-radius[:,-1:]).amin(1).amax(-1)
        inward=torch.minimum(radius[:,-1],factor['inward_radius'][:horizon].to(device)[None])
        inward=inward.clamp_min(world.protocol.target_radius)
        reserve_gap=(radius[:,:-1].amin(1)-inward).amax(-1)
        nearest=distance.amin(1);closest=nearest.amin(-1)
        approach=(nearest-5).clamp_min(0).mean(-1)
        selected=torch.as_tensor(plans,dtype=torch.long,device=device)
        state=torch.as_tensor(world.state(),dtype=torch.float32,device=device)[None].expand(len(plans),-1,-1)
        force=self.forces(state,state[...,3:5],factor,selected,0)
        center=factor['prior'].to(device)
        source_force=750*(center*factor['learned'].to(device)+(1-center)*factor['rules'].to(device))+250
        control_change=((force-source_force)/750).square().mean(-1).sum(-1)
        capture_reward=20*torch.sigmoid((4.5-closest)/.5)
        return (.25*control_change+64*collision+16*(gap+6).clamp_min(0).square()
            +16*(reserve_gap+2).clamp_min(0).square()
            +self.pursuit_weight*(closest-5).clamp_min(0)+approach-capture_reward)

    @torch.no_grad()
    def act(self,world):
        started=time.perf_counter()
        factor,physical,values,reference,center=physical_bank(world,self.source,self.physical_cfg,
            budget=self.budget,bank=self.bank,rounds=self.rounds)
        factor=self.prepare(world,factor,center)
        initial=self.proposals(world,factor,physical,values,reference,center)
        transitions=0
        def score(plans):
            nonlocal transitions
            paths,_=self.rollout(world,factor,plans);transitions+=len(plans)*paths.shape[2]
            return self.score_paths(world,factor,plans,paths).cpu().numpy()
        choices=[np.concatenate((x.nonzero()[:,0].numpy(),[9,10])) for x in factor['valid']]
        selected,trace=whole_configuration_search(initial,score,choices,self.budget,self.rounds)
        state=torch.as_tensor(world.state(),dtype=torch.float32,device=self.source.device)[None]
        force=self.forces(state,state[...,3:5],factor,torch.as_tensor(selected,device=state.device)[None],0)[0].cpu().numpy()
        trace.update(revision=self.revision,selected_indices=selected.tolist(),physical_thrust=force.tolist(),
            roles=['fusion' if x<9 else ('intercept' if x==9 else 'reserve') for x in selected],
            physical_reference_indices=reference.tolist(),source_center_indices=center.tolist(),
            source_center_scored=center.tolist() in trace['configurations'],initial_configurations=initial.tolist(),
            lead_seconds=factor['lead_seconds'].tolist(),prediction_transitions=transitions,
            proposal_search_evaluated=len(physical),budget_limit=self.budget,
            compute_seconds=time.perf_counter()-started)
        return force,trace


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--pursuit-weight',type=float,default=4.)
    args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    import citf.evaluate as evaluation
    original=evaluation.runtime_identity
    def identity(*positional,**kwargs):
        result=original(*positional,**kwargs)
        result.update(method='CITF-interception-reserve-development',revision=InterceptionReserveController.revision,
            schema='observed-interaction-role-and-fusion-v2',
            history_features=['observed joint state; common target rotation; entity role/order'],
            context_features=['known remaining time','current defender commands'])
        result['controller'].update(revision=InterceptionReserveController.revision,
            actions='original9 current SAC/Boids fusions plus physical lead-point tracking and inner reserve tracking',
            search='all primary/reserve assignments plus original physical choices; complete-action coordinate search',
            bank_scope='physical proposal seeds; role proposals added within the final scoring budget',
            replanning='every0.2s; role or fusion held only inside refreshed feedback prediction',
            reserve='radius clip(attacker_radius-10,18,28); known-body velocity and yaw tracking',
            protection='full-horizon collision and radial cover even after predicted capture; inward-turn propulsion envelope[-1500,3000]',
            predictor='learned joint interaction dynamics' if result['checkpoint_sha256'] else 'nominal observable APF, fixed agility2.25',
            pursuit_weight=args.pursuit_weight,future_inputs='no private future current,random state,attacker command,agility or termination')
        return result
    evaluation.runtime_identity=identity
    evaluation.load_behavior=lambda checkpoint,source,world,device:load_dynamics(checkpoint,device,source,world) if checkpoint else None
    evaluation.Controller=lambda *positional,**kwargs:InterceptionReserveController(*positional,pursuit_weight=args.pursuit_weight,**kwargs)
    evaluation.main()


if __name__=='__main__':main()
