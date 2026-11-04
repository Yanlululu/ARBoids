"""Prediction-guided interception/reserve within the original two-source hull.

Tracking targets generate current fusion coefficients, never a third physical
control source. Short configuration persistence matches feedback prediction
more closely; contact or coverage danger wakes the whole joint search early.
"""
import argparse
import sys
import time

import numpy as np
import torch

from citf.contracts import canonical_hash,file_hash
from experiments.batched_feedback import source_observations,source_actions
from experiments.interception_reserve import InterceptionReserveController,tracking_thrust
from experiments.interaction_dynamics import load as load_dynamics


class InterceptionFusionController(InterceptionReserveController):
    revision='CITF-predicted-interception-reserve-fusion-v3'

    def __init__(self,*args,replan_ticks=3,**kwargs):
        super().__init__(*args,**kwargs)
        self.replan_ticks=replan_ticks
        self.cached=None

    def forces(self,state,legacy,factor,plans,tick):
        device=state.device;n=plans.shape[1]
        gates=factor['grid'].to(device)[torch.arange(n,device=device)[None],plans.clamp_max(8)]
        if tick==0:
            learned,boids=factor['learned'].to(device)[None],factor['rules'].to(device)[None]
        else:
            observations,boids=source_observations(state,legacy)
            learned=source_actions(self.source.net,observations)
        if tick==0:
            intercept=factor['intercept_target'].to(device)[None].expand(len(plans),-1,-1)
        else:
            intercept=state[:,-1:,:2]+factor['lead_seconds'].to(device)[None,:,None]*state[:,-1:,3:5]
        position=state[:,-1:,:2];radius=position.norm(dim=-1,keepdim=True)
        reserve=(position/radius.clamp_min(1e-6)*(radius-10).clamp(18,28)).expand(-1,n,-1)
        target=torch.where((plans==10)[...,None],reserve,intercept)
        desired=(tracking_thrust(state,target)-250)/750
        span=learned-boids
        safe_span=torch.where(span.abs()>1e-6,span,torch.ones_like(span))
        projected=((desired-boids)/safe_span).clamp(0,1)
        projected=torch.where(span.abs()>1e-6,projected,gates)
        actual_gate=torch.where((plans>=9)[...,None],projected,gates)
        return 750*(actual_gate*learned+(1-actual_gate)*boids)+250

    def prepare(self,world,factor,center):
        factor=super().prepare(world,factor,center)
        self.last_factor=factor
        return factor

    @torch.no_grad()
    def act(self,world):
        started=time.perf_counter();state=world.state();tick=int(round(world.env.Current_T/.2))
        position=state[:,:2];radius=np.linalg.norm(position,axis=-1)
        nearest=float(np.linalg.norm(position[:-1]-position[-1],axis=-1).min())
        pair=np.linalg.norm(position[:-1,None]-position[None,:-1],axis=-1)
        np.fill_diagonal(pair,np.inf)
        urgent=nearest<12 or float((radius[:-1]-radius[-1]).min())>-12 or float(pair.min())<7
        search=self.cached is None or tick-self.cached['tick']>=self.replan_ticks or urgent
        if search:
            force,trace=super().act(world)
            self.cached=dict(tick=tick,selected=np.asarray(trace['selected_indices']),factor=self.last_factor)
            trace['decision_update']='joint_search'
        else:
            factor=dict(self.cached['factor'])
            factor['learned'],factor['rules'],factor['prior']=self.source.commands(world.env)
            factor['intercept_target']=torch.as_tensor(state[-1,:2],dtype=torch.float32)[None]+factor['lead_seconds'][:,None]*torch.as_tensor(state[-1,3:5],dtype=torch.float32)[None]
            value=torch.as_tensor(state,dtype=torch.float32,device=self.source.device)[None]
            selected=self.cached['selected']
            force=self.forces(value,value[...,3:5],factor,torch.as_tensor(selected,device=value.device)[None],0)[0].cpu().numpy()
            trace=dict(revision=self.revision,selected_indices=selected.tolist(),physical_thrust=force.tolist(),
                evaluated_unique=0,forward_batches=0,configurations=[],scores=[],
                decision_update='held_configuration_with_refreshed_source_feedback',
                held_since_tick=self.cached['tick'],roles=['fusion' if x<9 else ('intercept_fusion' if x==9 else 'reserve_fusion') for x in selected])
        trace.update(revision=self.revision,source_domain='original SAC and Boids only; channel-wise fusion in[0,1]',
            replan_ticks=self.replan_ticks,urgent_replan=urgent,compute_seconds=time.perf_counter()-started)
        return force,trace


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pursuit-weight',type=float,default=4.)
    parser.add_argument('--replan-ticks',type=int,default=3)
    args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    import citf.evaluate as evaluation
    import experiments.interception_reserve as reserve
    import experiments.continuous_interception as continuous
    import experiments.interaction_dynamics as dynamics
    import experiments.batched_feedback as feedback
    original=evaluation.runtime_identity
    def identity(*positional,**kwargs):
        result=original(*positional,**kwargs)
        result.update(method='CITF-interception-reserve-fusion-development',revision=InterceptionFusionController.revision,
            schema='joint-interaction-with-two-source-role-fusion-v3',
            history_features=['observed joint state; common target rotation; role/order'],
            context_features=['known remaining time','current defender commands'])
        result['controller'].update(revision=InterceptionFusionController.revision,
            actions='original9 fusions plus lead-point and reserve tracking projected channel-wise into current SAC/Boids convex hull',
            source_domain='unchanged frozen SAC and original Boids; no third physical command source',
            replan_ticks=args.replan_ticks,urgent_replan='nearest attacker<12m or radial margin<12m or defender pair<7m',
            execution='source feedback every0.2s; fixed selected role or coefficients for at most declared ticks',
            prediction='fixed selected role/fusion with refreshed source feedback inside3.2s; no future reoptimization input',
            reserve='radius clip(attacker_radius-10,18,28); full-horizon physical protection',
            predictor='learned interaction dynamics' if result['checkpoint_sha256'] else 'nominal observable APF,constant2.25 agility',
            pursuit_weight=args.pursuit_weight,future_inputs='none: no private future current,agility,command,RNG or termination')
        result['runtime_code']=canonical_hash(dict(core=result['runtime_code'],fusion=file_hash(__file__),
            reserve=file_hash(reserve.__file__),continuous=file_hash(continuous.__file__),
            dynamics=file_hash(dynamics.__file__),feedback=file_hash(feedback.__file__)))
        return result
    evaluation.runtime_identity=identity
    evaluation.load_behavior=lambda checkpoint,source,world,device:load_dynamics(checkpoint,device) if checkpoint else None
    evaluation.Controller=lambda *positional,**kwargs:InterceptionFusionController(*positional,
        pursuit_weight=args.pursuit_weight,replan_ticks=args.replan_ticks,**kwargs)
    evaluation.main()


if __name__=='__main__':main()
