"""Current joint policy selection subject to an explicit physical separation.

The learned relative preference orders configurations. Short predicted motion
restricts alternatives to the existing seven-metre risk clearance, while the
retained feedback remains an available fallback. A held alternative is checked
again at every actual control period. This is a controller change, not a new
terminal-value head, new thrust source, or a change to task success criteria.
"""
import time

import numpy as np
import torch

from citf.contracts import canonical_hash, file_hash
from experiments.joint_macro_policy import (
    JointMacroController, ObservedRoot, candidate_pool, load_policy, policy_features,
)


def minimum_separation(position,valid=None):
    # [C,E,T,2], target rotation preserves these physical distances.
    n=position.shape[1]-1
    i,j=torch.triu_indices(n,n,1,device=position.device)
    distance=(position[:,i]-position[:,j]).norm(dim=-1)
    if valid is not None:distance=distance.masked_fill(~valid[:,None],torch.inf)
    return distance.amin((1,2))


class JointConstrainedController(JointMacroController):
    revision='CITF-joint-policy-with-continuous-predicted-separation-v1'

    def history(self,world):
        length=self.planner.model.cfg.history
        history=np.stack(list(world.history)[-length:])
        return np.concatenate((np.repeat(history[:1],length-len(history),0),history),0)

    def current_factor(self,world):
        factor=dict(self.active['factor'])
        factor['learned'],factor['rules'],factor['prior']=self.source.commands(world.env)
        state=torch.as_tensor(world.state(),dtype=torch.float32)
        factor['intercept_target']=state[-1,:2][None]+factor['lead_seconds'][:,None]*state[-1,3:5][None]
        return factor

    @torch.no_grad()
    def act(self,world):
        started=time.perf_counter();tick=int(round(world.env.Current_T/.2))
        clearance=world.protocol.risk_clearance
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
            features,_=policy_features(self.planner,self.history(world),world.env.Current_T,factor,plans,world.protocol)
            prediction=self.model({k:v[None] for k,v in features.items()})
            logits=prediction['logits'][0]
            clearance_prediction=minimum_separation(prediction['position'][0],features['time_valid'])
            allowed=clearance_prediction>=clearance
            allowed[0]=True
            choice=int(logits.masked_fill(~allowed,-torch.inf).argmax())
            unconstrained=int(logits.argmax())
            self.active=dict(tick=tick,choice=choice,plan=plans[choice],factor=factor)
            force=reference_force if choice==0 else self._active_force(world)
            trace=dict(evaluated_unique=len(plans),forward_batches=1,configurations=plans.tolist(),
                scores=(-logits).cpu().tolist(),selected_option=choice,selected_indices=plans[choice].tolist(),
                unconstrained_option=unconstrained,decision_update='constrained_joint_policy',
                predicted_minimum_pair_distance=clearance_prediction.cpu().tolist(),
                alternative_allowed=allowed.cpu().tolist(),constraint_changed_choice=choice!=unconstrained,
                reference_option=0,hold_ticks=self.model.cfg.hold_ticks,
                proposal_prediction_candidates=proposal_trace['evaluated_unique'],
                reference_score_candidates=reference_trace['evaluated_unique'],
                policy_score_candidates=len(plans),
                total_score_invocations=len(plans)+proposal_trace['evaluated_unique']+reference_trace['evaluated_unique'])
        elif self.active['choice']==0:
            force,trace=self.teacher.act(world)
            trace.update(selected_option=0,decision_update='reference_feedback_within_current_option',
                joint_policy_candidates=0,hold_ticks=self.model.cfg.hold_ticks)
        else:
            # Reforecast the actual held coefficients, with current sources.
            # No fictitious reference candidate is supplied to the readout.
            factor=self.current_factor(world)
            root=ObservedRoot(self.history(world),world.env.Current_T,3,world.protocol)
            paths,_=self.planner.rollout(root,factor,self.active['plan'][None])
            distance=float(minimum_separation(paths)[0])
            stopped=distance<clearance
            if stopped:
                self.active['choice']=0
                force,trace=self.teacher.act(world)
                trace['evaluated_unique']+=1
                trace['decision_update']='alternative_replaced_by_reference_for_separation'
            else:
                force=self._active_force(world)
                trace=dict(evaluated_unique=1,forward_batches=1,configurations=[],scores=[],
                    selected_indices=self.active['plan'].tolist(),decision_update='held_joint_fusion_with_separation_check')
            trace.update(selected_option=self.active['choice'],predicted_held_minimum_pair_distance=distance,
                continuous_constraint_fallback=stopped,physical_prediction_candidates=1,
                hold_ticks=self.model.cfg.hold_ticks)
        trace.update(physical_thrust=force.tolist(),revision=self.revision,
            option_started_tick=self.active['tick'],risk_clearance=clearance,
            compute_seconds=time.perf_counter()-started)
        return force,trace


def main():
    import citf.evaluate as evaluation
    import experiments.joint_macro_policy as policy
    import experiments.interception_fusion as fusion
    import experiments.interception_reserve as reserve
    import experiments.continuous_interception as continuous
    import experiments.interaction_dynamics as dynamics
    import experiments.batched_feedback as feedback
    original=evaluation.runtime_identity
    def identity(*args,**kwargs):
        result=original(*args,**kwargs)
        checkpoint=args[2] if len(args)>2 else kwargs.get('checkpoint')
        # The base evaluation identity already includes the exact policy file.
        config=None
        if checkpoint:
            config=torch.load(checkpoint,map_location='cpu',weights_only=False)['config']
        result.update(method='CITF-current-joint-constrained-policy-development',
            revision=JointConstrainedController.revision)
        result['controller'].update(revision=JointConstrainedController.revision,
            selection='learned relative preference among alternatives with predicted3step defender-pair clearance>=existing protocol risk_clearance; reference feedback always available',
            execution='up to3ticks; each intervening tick rechecks alternative and falls back to retained feedback if separation constraint fails',
            prediction='candidate Transformer corrected3step motion for joint choice; observed-state dynamics for each held-action recheck; no actual future/RNG/termination input',
            task_collision_radius='unchanged',source_domain='original SAC/Boids convex combinations',
            training='unchanged final600 macro policy weights; single prospectively declared physical constraint',
            reference_has_learning=True)
        if config:
            result['controller'].update(teacher_checkpoint_sha256=file_hash(config['teacher_checkpoint']),
                dynamics_checkpoint_sha256=file_hash(config['dynamics_checkpoint']))
        result['runtime_code']=canonical_hash(dict(core=result['runtime_code'],constrained=file_hash(__file__),
            policy=file_hash(policy.__file__),fusion=file_hash(fusion.__file__),reserve=file_hash(reserve.__file__),
            continuous=file_hash(continuous.__file__),dynamics=file_hash(dynamics.__file__),feedback=file_hash(feedback.__file__)))
        return result
    evaluation.runtime_identity=identity
    evaluation.load_behavior=lambda checkpoint,source,world,device:load_policy(checkpoint,device)
    evaluation.Controller=JointConstrainedController
    evaluation.main()


if __name__=='__main__':main()
