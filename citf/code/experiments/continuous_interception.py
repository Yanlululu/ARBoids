"""Continuous joint current-fusion search using learned interaction dynamics.

The current fusion configuration is held only inside each prediction branch.
SAC and Boids are refreshed in the predicted state. The real controller executes
one 0.2-second command and replans, from the start of every allocated task.
"""
import argparse
import itertools
import sys
import time

import numpy as np
import torch

from citf.bridge import Controller
from citf.search import whole_configuration_search
from experiments.batched_feedback import integrate, source_observations, source_actions, attacker_thrust
from experiments.interaction_dynamics import load as load_dynamics
from experiments.root_response_diagnostic import physical_bank
from experiments.synchronized_configuration_policy import SynchronizedPolicyConfig


class InterceptionController(Controller):
    revision = 'CITF-continuous-interception-v1'

    def __init__(self, *args, pursuit_weight=4., **kwargs):
        super().__init__(*args, **kwargs)
        self.physical_cfg = SynchronizedPolicyConfig()
        self.pursuit_weight = pursuit_weight

    @torch.no_grad()
    def rollout(self, world, factor, plans):
        plans = torch.as_tensor(plans, dtype=torch.long)
        defenders = plans.shape[1]
        device = self.source.device
        gates = factor['grid'][torch.arange(defenders)[None], plans].to(device)
        def forces(state, legacy, plans, tick):
            if tick == 0:
                learned, boids = factor['learned'].to(device)[None], factor['rules'].to(device)[None]
            else:
                observations, boids = source_observations(state, legacy)
                learned = source_actions(self.source.net, observations)
            return 750*(gates*learned+(1-gates)*boids)+250
        return self._rollout(world, plans, forces)

    @torch.no_grad()
    def _rollout(self, world, plans, forces):
        """Shared feedback rollout; the caller supplies the executed control rule."""
        device = self.source.device
        plans = torch.as_tensor(plans, dtype=torch.long, device=device)
        count = len(plans)
        raw = np.stack(world.history)
        length = self.model.cfg.history if self.model is not None else 8
        raw = np.concatenate((np.repeat(raw[:1], max(0, length-len(raw)), 0), raw), 0)[-length:]
        history = torch.as_tensor(raw, dtype=torch.float32, device=device)[None].expand(count, -1, -1, -1).clone()
        remaining = torch.full((count,), world.protocol.time_limit-world.env.Current_T, device=device)
        paths, uncertainties = [], []
        state = history[:, -1]
        legacy = state[..., 3:5]
        # All candidates start from the same observed history. Later histories
        # diverge and must be encoded separately; this cache lives for one call.
        history_cache = (self.model.encode(history[:1], remaining[:1]).expand(count, -1, -1)
                         if self.model is not None else None)
        for tick in range(world.protocol.known_steps(world.env.Current_T)):
            force = forces(state, legacy, plans, tick)
            if self.model is not None:
                prediction = self.model(history, force, (remaining-.2*tick).clamp_min(0),
                                        history_cache=history_cache if tick == 0 else None)
                state, legacy = prediction['state'], prediction['legacy_velocity']
                uncertainties.append(prediction['attacker_log_std'].exp().mean(-1)*2500)
            else:
                # A strong unlearned same-planner control, using the nominal
                # observable APF feedback model with fixed, non-task-specific agility.
                command = torch.cat((force, attacker_thrust(state, 2.25,
                    world.protocol.apf_obstacle_rule == 'legacy_ordered')[:, None]), 1)
                state, legacy = integrate(state, command)
            paths.append(state[..., :2])
            history = torch.cat((history[:, 1:], state[:, None]), 1)
        return torch.stack(paths, 2), (torch.stack(uncertainties, 1).mean(-1) if uncertainties else None)

    @torch.no_grad()
    def proposals(self, world, factor, physical_plans, physical_values, reference, center):
        enemy = self.rollout(world, factor, np.asarray([center]))[0][0, -1].cpu()
        paths = factor['paths'][..., :len(enemy), :]
        distance = (paths-enemy[None,None]).norm(dim=-1)
        radial = paths.norm(dim=-1)-enemy.norm(dim=-1)[None,None]
        unary = factor['unary']
        primary_cost = distance.amin(-1) + .25*distance[..., -1] + .2*unary
        backup_cost = (distance+1.5*(radial+3).clamp_min(0)).mean(-1)+.2*unary
        primary = primary_cost.argmin(-1).numpy()
        backup = backup_cost.argmin(-1).numpy()
        plans = []
        def add(plan):
            key = tuple(map(int, plan))
            if key not in plans:
                plans.append(key)
        add(center); add(reference)
        for index in np.argsort(physical_values, kind='stable')[:6]:
            add(physical_plans[index])
        # Every defender may initiate interception; every other defender may
        # maintain backup. The full learned prediction scores each joint action.
        for i in range(len(reference)):
            row=reference.copy();row[i]=primary[i];add(row)
        for i,j in itertools.permutations(range(len(reference)),2):
            row=reference.copy();row[i]=primary[i];row[j]=backup[j];add(row)
        return np.asarray(plans,dtype=np.int64)

    @torch.no_grad()
    def score_paths(self, world, factor, plans, paths):
        count, entities, horizon, _ = paths.shape
        n = entities-1
        distance = (paths[:, :-1]-paths[:, -1:]).norm(dim=-1)
        radius = paths.norm(dim=-1)
        gap = radius[:, :-1]-radius[:, -1:]
        i,j = torch.triu_indices(n,n,1,device=paths.device)
        pair = (paths[:,i]-paths[:,j]).norm(dim=-1)
        nearest = distance.amin(1)
        radial = gap.amin(1)
        collision = (pair < world.protocol.collision_radius).any(1)
        loss = collision | (radius[:,-1]<world.protocol.target_radius)
        if world.protocol.radial_early_failure:
            loss = loss | (radial>0)
        capture = nearest < world.protocol.capture_radius
        tick = torch.arange(horizon, device=paths.device)[None]
        first_loss = torch.where(loss,tick,horizon+1).amin(-1)
        first_capture = torch.where(capture,tick,horizon+1).amin(-1)
        first_event = torch.minimum(first_loss,first_capture)
        observed = tick<=first_event[:,None]
        closest = nearest.masked_fill(~observed,torch.inf).amin(-1)
        worst_radial = radial.masked_fill(~observed,-torch.inf).amax(-1)
        minimum_pair = pair.masked_fill(~observed[:,None],torch.inf).amin(-1)
        separation_cost = (world.protocol.risk_clearance-minimum_pair).clamp_min(0).square().sum(-1)
        coverage_cost = (worst_radial+6).clamp_min(0).square()
        approach_cost = (nearest-5).clamp_min(0).masked_fill(~observed,0).sum(-1)/observed.sum(-1)
        backup_curve = (distance+1.5*(gap+3).clamp_min(0)).topk(2,dim=1,largest=False).values[:,1]
        backup_cost = backup_curve.masked_fill(~observed,0).sum(-1)/observed.sum(-1)
        early_capture = ((first_capture<first_loss)&(first_capture<horizon)).float()*(horizon-first_capture).clamp_min(0)/horizon
        predicted_loss = ((first_loss<=first_capture)&(first_loss<horizon)).float()
        selected = torch.as_tensor(plans,device=paths.device)
        unary = factor['unary'].to(paths.device)[torch.arange(n,device=paths.device)[None],selected].sum(-1)
        score = (.25*unary+64*separation_cost+16*coverage_cost
                 +self.pursuit_weight*(closest-5).clamp_min(0)+approach_cost+.5*backup_cost
                 +500*predicted_loss-20*early_capture)
        return score

    @torch.no_grad()
    def act(self,world):
        started=time.perf_counter()
        factor,physical_plans,values,reference,center=physical_bank(world,self.source,self.physical_cfg,
            budget=self.budget,bank=self.bank,rounds=self.rounds)
        initial=self.proposals(world,factor,physical_plans,values,reference,center)
        total_transitions=0; uncertainty=[]
        def score(plans):
            nonlocal total_transitions
            paths,sigma=self.rollout(world,factor,plans)
            total_transitions+=len(plans)*paths.shape[-2]
            if sigma is not None:uncertainty.extend(sigma.cpu().tolist())
            return self.score_paths(world,factor,plans,paths).cpu().numpy()
        selected,trace=whole_configuration_search(initial,score,
            [x.nonzero()[:,0].numpy() for x in factor['valid']],self.budget,self.rounds)
        gate=factor['grid'][torch.arange(len(selected)),torch.as_tensor(selected)]
        force=(750*(gate*factor['learned']+(1-gate)*factor['rules'])+250).numpy()
        trace.update(revision=self.revision,selected_indices=selected.tolist(),gates=gate.tolist(),
            physical_thrust=force.tolist(),source_center_indices=center.tolist(),physical_reference_indices=reference.tolist(),
            initial_configurations=initial.tolist(),source_center_scored=center.tolist() in trace['configurations'],
            prediction_transitions=total_transitions,control_period=.2,
            proposal_search_evaluated=len(physical_plans),budget_limit=self.budget,
            prediction='learned interaction dynamics' if self.model is not None else 'nominal observable APF feedback, fixed agility2.25',
            uncertainty_thrust_mean_N=float(np.mean(uncertainty)) if uncertainty else None,
            compute_seconds=time.perf_counter()-started)
        return force,trace


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pursuit-weight',type=float,default=4.)
    args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]
    import citf.evaluate as evaluation
    original_identity=evaluation.runtime_identity
    def identity(*positional,**kwargs):
        result=original_identity(*positional,**kwargs)
        result['method']='CITF-continuous-interception-development'
        result['revision']=InterceptionController.revision
        result['schema']='observed-joint-state-causal-step-dynamics-v1'
        result['history_features']=['observed x,y,yaw,ground vx,vy,yaw rate; common target rotation; entity role and order']
        result['context_features']=['known remaining time','current defender physical commands']
        result['controller'].update(revision=InterceptionController.revision,pursuit_weight=args.pursuit_weight,
            search='primary/backup joint proposals, strong physical proposals, whole-configuration coordinate refinement',
            bank_scope='physical proposal seeds; role proposals added within the final scoring budget',
            replanning='every0.2s from task start; current fusion held within prediction only',
            predictor='observable interaction dynamics' if result.get('checkpoint_sha256') else 'nominal observable APF feedback',
            selection='full predicted trajectories directly determine the selected complete joint action',
            objective='physical collision/radial coverage, reachable contact, approach and backup coverage',
            nominal_comparator='same search and score, nominal observable APF with constant2.25 agility',
            future_information='no private current,agility,attacker future action,RNG or true termination inputs')
        return result
    evaluation.runtime_identity=identity
    evaluation.load_behavior=lambda checkpoint,source,world,device:load_dynamics(checkpoint,device,source,world) if checkpoint else None
    evaluation.Controller=lambda *positional,**kwargs:InterceptionController(*positional,pursuit_weight=args.pursuit_weight,**kwargs)
    evaluation.main()


if __name__=='__main__':
    main()
