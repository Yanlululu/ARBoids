"""Joint learned preferences with separation and whole-team radial coverage.

Coverage uses the existing strong controller's six-metre reserve and legal
16-step current-configuration prediction, not a future fusion plan. The current
configuration is rechecked during execution. The retained feedback remains the
fallback; actual task rules and the two physical control sources stay fixed.
"""
import time

import numpy as np
import torch

from citf.contracts import canonical_hash, file_hash
from experiments.joint_constrained_policy import JointConstrainedController
from experiments.joint_macro_policy import load_policy


def minimum_radial_margin(paths):
    radius=paths.norm(dim=-1)
    return (radius[:,-1]-radius[:,:-1].amin(1)).amin(-1)


class JointCoverageController(JointConstrainedController):
    revision='CITF-joint-policy-with-separation-and-team-coverage-v1'
    radial_clearance=6.

    @torch.no_grad()
    def act(self,world):
        started=time.perf_counter()
        force,trace=super().act(world)
        original_choice=self.active['choice']
        if original_choice and trace['decision_update']=='constrained_joint_policy':
            plans=np.asarray(trace['configurations'],dtype=np.int64)
            paths,_=self.planner.rollout(world,self.active['factor'],plans)
            margin=minimum_radial_margin(paths)
            coverage=margin>=self.radial_clearance
            coverage[0]=True
            allowed=torch.as_tensor(trace['alternative_allowed'],device=paths.device)&coverage
            scores=torch.as_tensor(trace['scores'],device=paths.device)
            choice=int(scores.masked_fill(~allowed,torch.inf).argmin())
            self.active.update(choice=choice,plan=plans[choice])
            if choice==0:
                force,reference_trace=self.teacher.act(world)
                trace['coverage_fallback_reference_candidates']=reference_trace['evaluated_unique']
                trace['total_score_invocations']+=reference_trace['evaluated_unique']
                trace['forward_batches']+=reference_trace['forward_batches']
            else:
                force=self._active_force(world)
            trace.update(selected_option=choice,selected_indices=plans[choice].tolist(),
                decision_update='joint_policy_with_separation_and_coverage',
                predicted_minimum_radial_margin=margin.cpu().tolist(),
                coverage_allowed=coverage.cpu().tolist(),joint_alternative_allowed=allowed.cpu().tolist(),
                coverage_changed_choice=choice!=original_choice,
                coverage_prediction_candidates=len(plans),coverage_prediction_steps=paths.shape[2])
        elif original_choice:
            factor=self.current_factor(world)
            paths,_=self.planner.rollout(world,factor,self.active['plan'][None])
            margin=float(minimum_radial_margin(paths)[0])
            fallback=margin<self.radial_clearance
            if fallback:
                self.active['choice']=0
                previous_trace=trace
                force,trace=self.teacher.act(world)
                trace.update(selected_option=0,decision_update='alternative_replaced_by_reference_for_coverage',
                    separation_recheck=previous_trace.get('predicted_held_minimum_pair_distance'),
                    separation_prediction_candidates=previous_trace.get('physical_prediction_candidates',0))
            trace.update(predicted_held_minimum_radial_margin=margin,continuous_coverage_fallback=fallback,
                coverage_prediction_candidates=1,coverage_prediction_steps=paths.shape[2])
        trace.update(physical_thrust=force.tolist(),revision=self.revision,
            option_started_tick=self.active['tick'],radial_clearance=self.radial_clearance,
            compute_seconds=time.perf_counter()-started)
        return force,trace


def main():
    import citf.evaluate as evaluation
    import experiments.joint_constrained_policy as separation
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
        value=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=value['config']
        result.update(method='CITF-current-joint-feasible-policy-development',revision=JointCoverageController.revision)
        result['controller'].update(revision=JointCoverageController.revision,
            selection='learned relative preference under corrected3step defender separation>=protocol7m and observed-interaction16step team radial margin>=6m; retained feedback always available',
            execution='current fusion/feedback role up to3ticks, every0.2s recheck physical separation and team coverage',
            coverage_prediction='hold current configuration with refreshed two-source feedback within known legal horizon; no future reoptimization sequence or actual future termination',
            radial_clearance=6.,source_domain='unchanged original SAC/Boids convex combinations',
            teacher_checkpoint_sha256=file_hash(cfg['teacher_checkpoint']),
            dynamics_checkpoint_sha256=file_hash(cfg['dynamics_checkpoint']),
            training_update=value.get('update'),weights_changed=False,
            reference_has_learning=True,claim='development main-control change; complete net gains and whole-method attribution not assumed')
        result['runtime_code']=canonical_hash(dict(core=result['runtime_code'],coverage=file_hash(__file__),
            separation=file_hash(separation.__file__),policy=file_hash(policy.__file__),fusion=file_hash(fusion.__file__),
            reserve=file_hash(reserve.__file__),continuous=file_hash(continuous.__file__),
            dynamics=file_hash(dynamics.__file__),feedback=file_hash(feedback.__file__)))
        return result
    evaluation.runtime_identity=identity
    evaluation.load_behavior=lambda checkpoint,source,world,device:load_policy(checkpoint,device)
    evaluation.Controller=JointCoverageController
    evaluation.main()


if __name__=='__main__':main()
