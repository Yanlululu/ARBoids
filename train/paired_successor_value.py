"""Learn whole-fleet action values with measured paired successor supervision.

This module is an independent experiment entry point. It does not alter the
shared collectors or install its policy in another running process.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def measured_joint_gain(effect_logit, neutral_logit):
    """Undo event balancing before computing rescue probability minus harm."""
    probability = torch.stack((-effect_logit, neutral_logit, effect_logit), -1).softmax(-1)
    probability = probability/probability.new_tensor([1., .2, 1.])
    probability = probability/probability.sum(-1, keepdim=True)
    return probability[..., 2]-probability[..., 0]


class PublicMotionHistory:
    """Retain observed states; episode keys only address memory, never the net."""
    def __init__(self):
        self.states = {}

    def previous(self, context, episode_keys):
        from collections import deque
        states = context.detach().cpu()
        previous = []
        for state, key in zip(states, episode_keys.detach().cpu().flatten().tolist()):
            tick = int(round(float(state[-1])*400.))
            history = self.states.setdefault(key, deque(maxlen=8))
            if history and tick < history[-1][0]:
                history.clear()
            if not history or tick > history[-1][0]:
                history.append((tick, state.clone()))
            past = next((value for time, value in reversed(history) if time <= tick-5), history[0][1])
            previous.append(past)
        return torch.stack(previous).to(context.device)


class OnPolicyGateNetwork(nn.Module):
    """Learn whether to execute a frozen proposal from complete trajectories."""

    def __init__(self, defenders, width=64, identity_conditioned=True,
                 action_encoding='physical_delta', feedback_relative_scale=None,
                 state_baseline=False, representation='dense', peer_decision_conditioned=True):
        super().__init__()
        if feedback_relative_scale is not None or action_encoding != 'physical_delta':
            raise ValueError('The on-policy gate uses absolute measured proposals.')
        self.defenders = defenders; self.identity_conditioned = identity_conditioned
        self.feedback_relative_scale = None; self.action_encoding = action_encoding
        if representation not in ('dense', 'dense_history', 'compact_physical', 'compact_physical_only', 'compact_history'):
            raise ValueError('Unknown gate representation.')
        self.representation = representation
        feature_width = {'compact_physical': 45, 'compact_physical_only': 43, 'compact_history': 55,
                         'dense_history': (19+identity_conditioned)*defenders+23}.get(
            representation, (19+identity_conditioned)*defenders+11)
        self.encoder = nn.Sequential(nn.Linear(feature_width, width),
                                     nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.acceptance = nn.Linear(width, 1); self.baseline = nn.Linear(width, 1)
        nn.init.zeros_(self.acceptance.weight); nn.init.zeros_(self.acceptance.bias)
        nn.init.zeros_(self.baseline.weight); nn.init.zeros_(self.baseline.bias)

    def configure_fleet(self, defenders):
        if defenders != self.defenders:
            raise ValueError('The learned gate must retain its trained fleet size.')

    def active_vessels(self, programs):
        return programs > 0

    def features(self, context, controls, programs, timing, statistics):
        local, clock = PairedSuccessorNetwork.action_features(self, programs, timing, controls)
        return torch.cat((context, controls.flatten(1), local.flatten(1), clock, statistics), -1)

    def forward(self, features):
        if self.representation in ('dense_history', 'compact_physical', 'compact_physical_only', 'compact_history'):
            n = self.defenders; context_width = (8+self.identity_conditioned)*n+5
            history_context = features[:, (19+self.identity_conditioned)*n+11:]
            features = features[:, :(19+self.identity_conditioned)*n+11]
            dense_features = features
            context = features[:, :context_width]
            nodes = context[:, :8*n].reshape(-1, n, 8)
            identity = context[:, 8*n:9*n] if self.identity_conditioned else nodes.new_zeros(len(nodes), n)
            global_state = context[:, -5:]
            controls = features[:, context_width:context_width+6*n].reshape(-1, n, 6)
            local = features[:, context_width+6*n:context_width+11*n].reshape(-1, n, 5)
            active = local[..., 0]; weights = active/active.sum(-1, keepdim=True).clamp_min(1.)
            pool = lambda value: (value*weights[..., None]).sum(1)
            distance = nodes[..., :2].norm(dim=-1).clamp_min(.01)
            relative_velocity = .6*nodes[..., 2:4]-global_state[:, None, 2:4]
            closing = -(nodes[..., :2]*relative_velocity).sum(-1)/distance
            tangential = (nodes[..., 0]*relative_velocity[..., 1]-nodes[..., 1]*relative_velocity[..., 0])/distance
            separation = torch.cdist(nodes[..., :2], nodes[..., :2])+100.*torch.eye(n, device=features.device)[None]
            nearest_index = separation.argmin(-1)
            nearest_peer = nodes.gather(1, nearest_index[..., None].expand(-1, -1, 8))
            physical = torch.stack((active.mean(-1), (separation.amin(-1)*weights).sum(-1),
                distance.amin(-1), separation.amin((1, 2)), (closing*weights).sum(-1),
                (tangential*weights).sum(-1), (relative_velocity.norm(dim=-1)*weights).sum(-1),
                ((controls[..., :2]-controls[..., 2:4]).norm(dim=-1)*weights).sum(-1)), -1)
            # Recover the observed target range from the fleet geometry. In
            # this frame |boat|^2-|boat-target|^2 = range^2-2*range*relative_x.
            x = nodes[..., 0]; centered_x = x-x.mean(-1, keepdim=True)
            squared_offset = nodes[..., 7].square()-nodes[..., :2].square().sum(-1)
            target_range = (-(centered_x*squared_offset).mean(-1)/
                            (2.*centered_x.square().mean(-1).clamp_min(1e-6))).clamp(0., 5.)
            target_time = ((30.*target_range-15.).clamp_min(0.)/
                           (5.*global_state[:, 2].clamp_min(.01))/80.).clamp_max(2.)
            intercept_time = (30.*distance/(5.*closing.clamp_min(.01))/80.).amin(-1).clamp_max(2.)
            features = torch.cat((pool(nodes), pool(controls), pool(local[..., 1:]),
                (identity*weights).sum(-1, keepdim=True), global_state,
                features[:, -6:] if self.representation == 'compact_physical' else features[:, -6:-2], physical,
                pool(nearest_peer[..., :4]-nodes[..., :4]),
                torch.stack((target_range, target_time, intercept_time), -1)), -1)
            if self.representation in ('compact_history', 'dense_history'):
                if not self.identity_conditioned or history_context.shape != context.shape:
                    raise ValueError('Motion history needs one preceding observed context with vessel identities.')
                past_nodes = history_context[:, :8*n].reshape(-1, n, 8)
                past_order = (history_context[:, 8*n:9*n]*(n-1)).round().long()
                current_order = (identity*(n-1)).round().long()
                indices = past_order.argsort(-1).gather(1, current_order)
                past_nodes = past_nodes.gather(1, indices[..., None].expand(-1, -1, 8))
                past_global = history_context[:, -5:]
                elapsed = ((global_state[:, -1]-past_global[:, -1])*80.).clamp_min(.2)

                def reframe(vector):
                    old_x, old_y = -past_global[:, 0], -past_global[:, 1]
                    new_x, new_y = -global_state[:, 0], -global_state[:, 1]
                    if vector.ndim == 3:
                        old_x, old_y, new_x, new_y = [value[:, None] for value in (old_x, old_y, new_x, new_y)]
                    world_x = old_x*vector[..., 0]-old_y*vector[..., 1]
                    world_y = old_y*vector[..., 0]+old_x*vector[..., 1]
                    return torch.stack((new_x*world_x+new_y*world_y,
                                        new_x*world_y-new_y*world_x), -1)

                previous_velocity = reframe(past_nodes[..., 2:4])
                acceleration = (nodes[..., 2:4]-previous_velocity)/elapsed[:, None, None]
                previous_target_velocity = reframe(past_global[:, 2:4])
                target_acceleration = (global_state[:, 2:4]-previous_target_velocity)/elapsed[:, None]
                previous_heading = reframe(past_nodes[..., 4:6])
                heading_turn = torch.atan2(previous_heading[..., 0]*nodes[..., 5]-
                    previous_heading[..., 1]*nodes[..., 4], (previous_heading*nodes[..., 4:6]).sum(-1))/elapsed[:, None]/np.pi
                cross = (previous_target_velocity[:, 0]*global_state[:, 3]-
                         previous_target_velocity[:, 1]*global_state[:, 2])
                dot = (previous_target_velocity*global_state[:, 2:4]).sum(-1)
                moving = (previous_target_velocity.norm(dim=-1)*global_state[:, 2:4].norm(dim=-1)) > 1e-4
                target_turn = torch.where(moving, torch.atan2(cross, dot)/elapsed/np.pi, torch.zeros_like(dot))
                history_features = torch.cat((pool(acceleration), target_acceleration,
                    (((nodes[..., 6]-past_nodes[..., 6])/elapsed[:, None])*weights).sum(-1, keepdim=True),
                    (heading_turn*weights).sum(-1, keepdim=True),
                    pool(acceleration.gather(1, nearest_index[..., None].expand(-1, -1, 2))),
                    target_turn[:, None],
                    ((global_state[:, 2:4].norm(dim=-1)-previous_target_velocity.norm(dim=-1))/elapsed)[:, None],
                    acceleration.norm(dim=-1).amax(-1, keepdim=True),
                    ((global_state[:, -1]-past_global[:, -1])*80.).clamp(0., 1.)[:, None]), -1)
                features = torch.cat((dense_features if self.representation == 'dense_history'
                                      else features, history_features), -1)
        encoded = self.encoder(features)
        return self.acceptance(encoded).squeeze(-1), self.baseline(encoded).squeeze(-1)


class PairedSuccessorNetwork(nn.Module):
    """A shared interaction representation for task value and physical change.

    The value is a nonlinear function of the entire fleet. No sum of individual
    vessel values or factual-probability multiplication defines the advantage.
    Both arms of a comparison pass through exactly the same network.
    """

    def __init__(self, defenders, width=48, identity_conditioned=True,
                 action_encoding='physical_delta', feedback_relative_scale=.25,
                 state_baseline=False, representation='dense',
                 peer_decision_conditioned=True):
        super().__init__()
        if defenders < 1 or width < 8 or width % 4:
            raise ValueError('Use a positive fleet size and a width divisible by four.')
        if action_encoding != 'physical_delta':
            raise ValueError('Successor supervision requires physical action features.')
        if feedback_relative_scale is not None and (not np.isfinite(feedback_relative_scale)
                or not 0. < feedback_relative_scale <= 1.):
            raise ValueError('Use a finite relative intervention scale in (0, 1].')
        if state_baseline or representation != 'dense' or not peer_decision_conditioned:
            raise ValueError('This network owns its whole-fleet representation.')
        self.defenders = int(defenders)
        self.width = int(width)
        self.identity_conditioned = bool(identity_conditioned)
        self.action_encoding = action_encoding
        self.feedback_relative_scale = (None if feedback_relative_scale is None
                                        else float(feedback_relative_scale))
        self.state_size = (14 + self.identity_conditioned)*self.defenders + 5
        self.action_size = 5*self.defenders + 4
        self.node_encoder = nn.Sequential(
            nn.Linear(14+self.identity_conditioned, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU())
        self.global_encoder = nn.Linear(5, width)
        self.action_encoder = nn.Linear(5, width, bias=False)
        self.time_encoder = nn.Linear(4, width, bias=False)
        self.interaction = nn.TransformerEncoderLayer(
            width, 4, 2*width, dropout=0., activation='gelu',
            batch_first=True, norm_first=True)
        self.task_value = nn.Sequential(
            nn.Linear(2*width+5, width), nn.SiLU(), nn.Linear(width, 1))
        self.node_response = nn.Linear(width, 8)
        self.global_response = nn.Linear(2*width+5, 5)

    def configure_fleet(self, defenders):
        if defenders != self.defenders:
            raise ValueError('Evaluate the fleet size recorded by this trained model.')

    def active_vessels(self, programs):
        active = programs > 0
        return active if self.feedback_relative_scale is None else active & (programs != 5)

    def action_features(self, programs, timing, controls):
        if programs.shape[-1] != self.defenders or controls.shape[-2:] != (self.defenders, 6):
            raise ValueError('Commands and programs must share the recorded vessel order.')
        if timing.shape[-1] != 2 or bool(((programs < 0) | (programs > 9)).any()):
            raise ValueError('Use valid physical programs and a delay/span pair.')
        active = self.active_vessels(programs)
        coefficients = controls.new_tensor([[0., 0.]] +
            [[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])
        offset = coefficients[programs]
        source, rule, factual = controls.split(2, -1)
        if self.feedback_relative_scale is None:
            command = offset*source+(1.-offset)*rule
        else:
            offset = self.feedback_relative_scale*(2.*offset-1.)
            requested = factual+offset*(source-rule)
            command = torch.maximum(torch.minimum(requested, torch.maximum(source, rule)),
                                    torch.minimum(source, rule))
        delta = torch.where(active[..., None], command-factual, torch.zeros_like(factual))
        local = torch.cat((active[..., None].to(controls.dtype),
                           offset*active[..., None], .25*delta), -1)
        temporal = torch.cat((timing.to(controls.dtype)/400.,
            timing.to(controls.dtype).log1p()/np.log(401.)), -1)
        if self.feedback_relative_scale is not None:
            temporal = temporal*active.any(-1, keepdim=True)
        return local, temporal

    def encode(self, context, controls, programs, timing):
        n = self.defenders
        if context.shape[-1] != (8+self.identity_conditioned)*n+5:
            raise ValueError('State features must retain the observed physical vessel identities.')
        nodes = context[..., :8*n].reshape(-1, n, 8)
        commands = controls.reshape(-1, n, 6)
        pieces = [nodes, commands]
        if self.identity_conditioned:
            pieces.append(context[..., 8*n:9*n].reshape(-1, n, 1))
        shared = context[..., -5:].reshape(-1, 5)
        local, temporal = self.action_features(programs, timing, controls)
        hidden = (self.node_encoder(torch.cat(pieces, -1))
            + self.global_encoder(shared)[:, None]
            + self.action_encoder(local.reshape(-1, n, 5))
            + self.time_encoder(temporal.reshape(-1, 4))[:, None])
        hidden = self.interaction(hidden)
        pooled = torch.cat((hidden.mean(1), hidden.amax(1), shared), -1)
        return hidden, pooled

    def forward_with_response(self, context, controls, programs, timing):
        nodes, pooled = self.encode(context, controls, programs, timing)
        response = torch.cat((self.node_response(nodes).flatten(1),
                              self.global_response(pooled)), -1)
        return self.task_value(pooled).squeeze(-1), response

    def forward(self, context, controls, programs, timing):
        _, pooled = self.encode(context, controls, programs, timing)
        return self.task_value(pooled).squeeze(-1)

    def forward_catalog(self, context, controls, programs, timing, chunk_size=128):
        if chunk_size < 1:
            raise ValueError('The candidate chunk must be positive.')
        batch, n = len(context), self.defenders
        result = []
        for begin in range(0, len(programs), chunk_size):
            action = programs[begin:begin+chunk_size]
            clock = timing[begin:begin+chunk_size]
            count = len(action)
            values = self(context[:, None].expand(-1, count, -1).reshape(batch*count, -1),
                controls[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, 6),
                action[None].expand(batch, -1, -1).reshape(batch*count, n),
                clock[None].expand(batch, -1, -1).reshape(batch*count, 2))
            result.append(values.reshape(batch, count))
        return torch.cat(result, 1)


class PairedOutcomeSuccessorNetwork(PairedSuccessorNetwork):
    """Separate factual success from conditional rescue and harm.

    Both complete-plan values share one factual value. The changed plan is
    valued by the law of total probability, so its difference is integrable
    and bounded even when factual success is close to one. Paired observations
    identify the two conditional consequences without subtracting saturated
    marginal logits or using the realized future during execution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcome_effect = nn.Sequential(nn.Linear(4*self.width+10, self.width),
            nn.SiLU(), nn.Linear(self.width, 2))
        nn.init.constant_(self.outcome_effect[-1].bias, -3.)

    @staticmethod
    def plan_probability(factual_logit, rescue_logit, harm_logit, active):
        factual = factual_logit.sigmoid()
        changed = factual*(1.-harm_logit.sigmoid())+(1.-factual)*rescue_logit.sigmoid()
        return torch.where(active, changed, factual)

    def forward_components(self, context, controls, programs, timing):
        nodes, pooled = self.encode(context, controls, programs, timing)
        _, reference = self.encode(context, controls, torch.zeros_like(programs),
                                   torch.zeros_like(timing))
        factual = self.task_value(reference).squeeze(-1)
        rescue, harm = self.outcome_effect(torch.cat((pooled, pooled-reference), -1)).unbind(-1)
        active = self.active_vessels(programs).any(-1)
        probability = self.plan_probability(factual, rescue, harm, active)
        response = torch.cat((self.node_response(nodes).flatten(1), self.global_response(pooled)), -1)
        return torch.logit(probability.clamp(1e-6, 1.-1e-6)), response, factual, rescue, harm

    def forward_with_response(self, context, controls, programs, timing):
        value, response, _, _, _ = self.forward_components(context, controls, programs, timing)
        return value, response

    def forward(self, context, controls, programs, timing):
        return self.forward_components(context, controls, programs, timing)[0]


def paired_outcome_conditional_loss(rescue_logit, harm_logit, joint_target):
    """Learn each factual-outcome stratum from its measured common futures."""
    if joint_target.shape != (*rescue_logit.shape, 4):
        raise ValueError('Paired outcomes must contain the four measured joint probabilities.')
    if bool((joint_target < 0).any()) or not torch.allclose(
            joint_target.sum(-1), torch.ones_like(rescue_logit), atol=1e-5):
        raise ValueError('Measured joint outcomes must form a probability distribution.')
    failure = joint_target[..., :2].sum(-1)
    success = joint_target[..., 2:].sum(-1)
    rescue = F.binary_cross_entropy_with_logits(rescue_logit,
        joint_target[..., 1]/failure.clamp_min(1e-8), reduction='none')
    harm = F.binary_cross_entropy_with_logits(harm_logit,
        joint_target[..., 2]/success.clamp_min(1e-8), reduction='none')
    # Equal conditional objectives improve the rarely observed failure stratum;
    # the calibrated factual probability restores its population mass at use.
    return .5*((rescue*failure).sum()/failure.sum().clamp_min(1.)
               +(harm*success).sum()/success.sum().clamp_min(1.))


class PhysicalBottleneckSuccessorNetwork(PairedSuccessorNetwork):
    """Predict a physical successor, then value its joint fleet state.

    The success head receives no action code or free action-conditioned latent.
    It is also trained on measured successor states and their terminal outcomes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_value = None
        self.potential_encoder = nn.Sequential(
            nn.Linear(8+self.identity_conditioned, self.width), nn.SiLU(),
            nn.Linear(self.width, self.width), nn.SiLU())
        self.potential_global = nn.Linear(5, self.width)
        self.potential_interaction = nn.TransformerEncoderLayer(
            self.width, 4, 2*self.width, dropout=0., activation='gelu',
            batch_first=True, norm_first=True)
        self.potential = nn.Sequential(nn.Linear(2*self.width+5, self.width),
                                       nn.SiLU(), nn.Linear(self.width, 1))

    def value_state(self, context):
        n = self.defenders
        pieces = [context[..., :8*n].reshape(-1, n, 8)]
        if self.identity_conditioned:
            pieces.append(context[..., 8*n:9*n].reshape(-1, n, 1))
        shared = context[..., -5:]
        hidden = self.potential_interaction(self.potential_encoder(torch.cat(pieces, -1))
                                           + self.potential_global(shared)[:, None])
        return self.potential(torch.cat((hidden.mean(1), hidden.amax(1), shared), -1)).squeeze(-1)

    def successor_context(self, context, response):
        n = self.defenders
        return torch.cat((context[..., :8*n]+response[..., :8*n],
            context[..., 8*n:-5], context[..., -5:]+response[..., -5:]), -1)

    def forward_with_response(self, context, controls, programs, timing):
        nodes, pooled = self.encode(context, controls, programs, timing)
        response = torch.cat((self.node_response(nodes).flatten(1), self.global_response(pooled)), -1)
        return self.value_state(self.successor_context(context, response)), response

    def forward(self, context, controls, programs, timing):
        return self.forward_with_response(context, controls, programs, timing)[0]


class PlannedSuccessorNetwork(PhysicalBottleneckSuccessorNetwork):
    """Value physical state together with an intervention's unfinished hold.

    A delayed or long action is part of the continuation state. Completed
    actions have exactly zero pending features; their value must pass through
    the predicted physical response.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_horizon = 20 if self.feedback_relative_scale is None else 5
        self.pending_encoder = nn.Linear(3, self.width, bias=False)
        self.pending_clock = nn.Linear(4, self.width, bias=False)

    def pending_features(self, programs, timing, dtype):
        delay = (timing[:, 0]-self.response_horizon).clamp_min(0)
        end = (timing.sum(-1)-self.response_horizon).clamp_min(0)
        clock = torch.stack((delay, end-delay), -1).to(dtype)
        active = self.active_vessels(programs) & (clock[:, 1:] > 0)
        coefficients = clock.new_tensor([[0., 0.]]+
            [[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])[programs]
        if self.feedback_relative_scale is not None:
            coefficients = self.feedback_relative_scale*(2.*coefficients-1.)
        local = torch.cat((active[..., None].to(dtype), coefficients*active[..., None]), -1)
        temporal = torch.cat((clock/400., clock.log1p()/np.log(401.)), -1)
        return local, temporal*active.any(-1, keepdim=True)

    def value_state(self, context, programs=None, timing=None):
        n = self.defenders
        pieces = [context[..., :8*n].reshape(-1, n, 8)]
        if self.identity_conditioned:
            pieces.append(context[..., 8*n:9*n].reshape(-1, n, 1))
        shared = context[..., -5:]
        hidden = self.potential_encoder(torch.cat(pieces, -1))+self.potential_global(shared)[:, None]
        if programs is not None:
            local, temporal = self.pending_features(programs, timing, context.dtype)
            hidden = hidden+self.pending_encoder(local)+self.pending_clock(temporal)[:, None]
        hidden = self.potential_interaction(hidden)
        return self.potential(torch.cat((hidden.mean(1), hidden.amax(1), shared), -1)).squeeze(-1)

    def forward_with_response(self, context, controls, programs, timing):
        nodes, pooled = self.encode(context, controls, programs, timing)
        response = torch.cat((self.node_response(nodes).flatten(1), self.global_response(pooled)), -1)
        return self.value_state(self.successor_context(context, response), programs, timing), response


class ConstrainedPlannedSuccessorNetwork(PlannedSuccessorNetwork):
    """Keep the predicted clock and angular coordinates physically valid."""

    def successor_context(self, context, response):
        successor = super().successor_context(context, response)
        n = self.defenders
        nodes = successor[:, :8*n].reshape(-1, n, 8)
        nodes = torch.cat((nodes[..., :4], F.normalize(nodes[..., 4:6], dim=-1),
                           nodes[..., 6:7], nodes[..., 7:8].clamp_min(0.)), -1)
        shared = torch.cat((F.normalize(successor[:, -5:-3], dim=-1), successor[:, -3:-1],
                            (context[:, -1:]+self.response_horizon/400.).clamp_max(1.)), -1)
        return torch.cat((nodes.flatten(1), successor[:, 8*n:-5], shared), -1)

    def forward_with_response(self, context, controls, programs, timing):
        nodes, pooled = self.encode(context, controls, programs, timing)
        response = torch.cat((self.node_response(nodes).flatten(1), self.global_response(pooled)), -1)
        successor = self.successor_context(context, response)
        n = self.defenders
        projected = torch.cat((successor[:, :8*n]-context[:, :8*n],
                               successor[:, -5:]-context[:, -5:]), -1)
        return self.value_state(successor, programs, timing), projected


class LocalQuadraticSuccessorNetwork(PairedSuccessorNetwork):
    """A fleet-conditioned local control response with an exact zero origin.

    The state representation may describe arbitrary fleet interactions, while
    the response to a small single-vessel intervention has a shared quadratic
    form. This limits arbitrary peaks between sparsely observed controls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_coefficients = nn.Linear(self.width, 7)
        nn.init.zeros_(self.local_coefficients.weight)
        nn.init.zeros_(self.local_coefficients.bias)

    def local_value(self, context, controls, programs, timing):
        invalid_clock = (timing[..., 0] != 0) | ((timing[..., 1] != 0) & (timing[..., 1] != 5))
        if bool((self.active_vessels(programs).any(-1) & invalid_clock).any()):
            raise ValueError('The fitted local response uses one immediate decision period.')
        nodes, pooled = self.encode(context, controls, torch.zeros_like(programs), torch.zeros_like(timing))
        action, _ = self.action_features(programs, timing, controls)
        offset = action[..., 1:3]/self.feedback_relative_scale
        left, right = offset.unbind(-1)
        features = torch.cat((offset, offset.square(), (left*right)[..., None],
                              4.*action[..., 3:]/self.feedback_relative_scale), -1)
        response = .25*(self.local_coefficients(nodes)*features).sum((-1, -2))
        return self.task_value(pooled).squeeze(-1)+response

    def forward(self, context, controls, programs, timing):
        return self.local_value(context, controls, programs, timing)

    def forward_with_response(self, context, controls, programs, timing):
        _, physical = super().forward_with_response(context, controls, programs, timing)
        return self.local_value(context, controls, programs, timing), physical


class EventTimeSuccessorNetwork(LocalQuadraticSuccessorNetwork):
    """Learn breach, collision, and capture rates from actual event times.

    Surviving to the episode horizon is a success in this environment. For
    constant conditional rates, success is survival plus capture before any
    competing failure. Event time is a training target, never an input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_value[-1] = nn.Linear(self.width, 3)
        self.local_coefficients = nn.Linear(self.width, 21)
        nn.init.zeros_(self.local_coefficients.weight)
        nn.init.zeros_(self.local_coefficients.bias)

    def event_rates(self, context, controls, programs, timing):
        invalid_clock = (timing[..., 0] != 0) | ((timing[..., 1] != 0) & (timing[..., 1] != 5))
        if bool((self.active_vessels(programs).any(-1) & invalid_clock).any()):
            raise ValueError('The fitted event response uses one immediate decision period.')
        nodes, pooled = self.encode(context, controls, torch.zeros_like(programs), torch.zeros_like(timing))
        action, _ = self.action_features(programs, timing, controls)
        offset = action[..., 1:3]/self.feedback_relative_scale
        left, right = offset.unbind(-1)
        features = torch.cat((offset, offset.square(), (left*right)[..., None],
                              4.*action[..., 3:]/self.feedback_relative_scale), -1)
        coefficients = self.local_coefficients(nodes).reshape(-1, self.defenders, 7, 3)
        log_rates = self.task_value(pooled)+.25*(coefficients*features[..., None]).sum((1, 2))
        return log_rates.clamp(-8., 8.).exp()

    @staticmethod
    def success_logit(context, rates):
        total = rates.sum(-1)
        remaining = (1.-context[..., -1]).clamp(0., 1.)
        survived = (-total*remaining).exp()
        probability = survived+(1.-survived)*rates[..., 2]/total
        return torch.logit(probability.clamp(1e-6, 1.-1e-6))

    def local_value(self, context, controls, programs, timing):
        return self.success_logit(context, self.event_rates(context, controls, programs, timing))

    def forward_with_events(self, context, controls, programs, timing):
        nodes, pooled = self.encode(context, controls, programs, timing)
        physical = torch.cat((self.node_response(nodes).flatten(1), self.global_response(pooled)), -1)
        rates = self.event_rates(context, controls, programs, timing)
        return self.success_logit(context, rates), physical, rates

    def forward_with_response(self, context, controls, programs, timing):
        logit, physical, _ = self.forward_with_events(context, controls, programs, timing)
        return logit, physical


def paired_successor_objective(candidate_logits, reference_logits,
        candidate_response, reference_response, candidate_target, reference_target,
        candidate_physical, reference_physical, physical_mask, response_scale,
        *, paired, physical_weight=1., effect_weight=4., response_difference_weight=1., absolute_weight=1.):
    """Use the same absolute observations in both arms; add measured contrasts.

    Inputs are means over the same recorded future population. In particular,
    the two response means must exclude exactly the same terminal/missing
    futures before their difference is supervised. A predicted value is never
    substituted for a measured terminal target.
    """
    if response_scale.ndim != 1 or bool((response_scale <= 0).any()):
        raise ValueError('Physical normalization must be positive and feature-wise.')
    absolute = .5*(F.binary_cross_entropy_with_logits(candidate_logits, candidate_target)
        + F.binary_cross_entropy_with_logits(reference_logits, reference_target))
    mask = physical_mask.to(candidate_response.dtype)
    denominator = mask.sum().clamp_min(1.)*candidate_response.shape[-1]
    candidate_error = (candidate_response-candidate_physical)/response_scale
    reference_error = (reference_response-reference_physical)/response_scale
    physical = (.5*(candidate_error.square()+reference_error.square())*mask[:, None]).sum()/denominator
    effect = (candidate_logits.sigmoid()-reference_logits.sigmoid()
              -(candidate_target-reference_target)).square().mean()
    contrast = (((candidate_error-reference_error).square())*mask[:, None]).sum()/denominator
    loss = absolute_weight*absolute+physical_weight*physical
    if paired:
        loss = loss+effect_weight*effect+response_difference_weight*contrast
    return loss, dict(absolute=absolute.detach(), physical=physical.detach(),
                     effect=effect.detach(), response_contrast=contrast.detach())


def physical_response(root, outcome, fixed_horizon=False):
    """Reorder the measured successor back to the decision's vessel identities."""
    if fixed_horizon:
        if outcome.get('following') is None or outcome.get('response_steps') != 20:
            return None
        if outcome.get('response_sampling') != 'physical_tick':
            raise ValueError('The fixed-horizon response must be a measured physical state.')
        # Collectors store `following` in the root's order, while the later
        # `continuation_context` is re-sorted and may follow a longer hold.
        state = np.asarray(outcome['following'], dtype=np.float32)
        if state.shape != np.shape(root['context']):
            raise ValueError('The measured response and root must use the same physical features.')
        return state-np.asarray(root['context'], dtype=np.float32)
    if outcome.get('continuation_context') is None:
        return None
    n = len(root['order'])
    order = list(outcome['continuation_order'])
    if sorted(order) != sorted(root['order']):
        raise ValueError('A successor must contain exactly the original fleet.')
    state = np.asarray(outcome['continuation_context'], dtype=np.float32)
    nodes = state[:8*n].reshape(n, 8)
    aligned = np.concatenate((nodes[[order.index(i) for i in root['order']]].ravel(), state[-5:]))
    return aligned-np.asarray(root['context'], dtype=np.float32)


def root_context(root):
    n = len(root['order'])
    context = np.asarray(root['context'], dtype=np.float32)
    return np.concatenate((context[:-5], np.asarray(root['order'], np.float32)/max(n-1, 1), context[-5:]))


def complete_validation_catalog(validation, original):
    """Join supplemental measurements to the original actions by exact root."""
    if not validation.get('completed') or not original.get('completed'):
        raise ValueError('Catalog completion requires two complete measurement collections.')
    def identity(root):
        return (tuple(root['scene']), root['prefix_future'], root['root_steps'])
    indexed = {identity(root): root for root in original['roots']}
    merged = []
    for root in validation['roots']:
        base = indexed[identity(root)]
        for name in ('context', 'effect_controls', 'order', 'futures', 'future_weights'):
            if not np.array_equal(root[name], base[name]):
                raise ValueError('Supplemental actions must share the exact root and future population.')
        options = sorted(set(map(tuple, root['options'])) | set(map(tuple, base['options'])))
        if len(options) != 8*len(root['order'])+1 or options[0] != (0, 0, 0):
            raise ValueError('The original and supplemental actions must complete the deployment catalog.')
        mapping = {option: index for index, option in enumerate(options)}
        outcomes = {}
        for source in (base, root):
            for (action, future), observation in source['outcomes'].items():
                key = (mapping[tuple(source['options'][action])], future)
                if key in outcomes:
                    for field in ('success', 'outcome', 'duration'):
                        if observation[field] != outcomes[key][field]:
                            raise ValueError('Repeated factual outcomes disagree between collections.')
                outcomes[key] = observation
        if len(outcomes) != len(options)*len(root['futures']):
            raise ValueError('Every catalog option needs all recorded terminal futures.')
        merged.append(dict(root, options=options, outcomes=outcomes))
    return dict(validation, roots=merged, catalog_complete=True)


def measured_pairs(datasets, layouts, fixed_horizon=False, all_action_pairs=False):
    """Preserve complete paired terminal observations, including unchanged outcomes."""
    layouts = set(map(tuple, layouts))
    packed = {k: [] for k in ('context', 'controls', 'programs', 'timing', 'reference_programs',
        'reference_timing', 'target', 'reference_target', 'physical', 'reference_physical', 'mask',
        'events', 'reference_events', 'exposure', 'reference_exposure', 'joint_target', 'population_weight')}
    seen = set()
    roots = []
    for data in datasets:
        if not data.get('completed'):
            raise ValueError('Only completed collections may supply training observations.')
        for root in data['roots']:
            if tuple(root['scene']) not in layouts:
                continue
            n = len(root['order'])
            population_weight = float(root.get('population_weight', 1.))
            if not np.isfinite(population_weight) or population_weight <= 0.:
                raise ValueError('Root inclusion weights must be finite and positive.')
            if tuple(root['options'][0]) != (0, 0, 0):
                raise ValueError('The measured zero intervention must be the first option.')
            comparisons = [(a, b) for a in range(1, len(root['options']))
                           for b in (range(a) if all_action_pairs else (0,))]
            for a, reference_action in comparisons:
                futures = list(range(len(root['futures'])))
                if any((b, f) not in root['outcomes'] for b in (reference_action, a) for f in futures):
                    raise ValueError('A paired collection is missing actual terminal outcomes.')
                key = (tuple(root['scene']), root['prefix_future'], root['root_steps'],
                       tuple(root['options'][a]), tuple(root['options'][reference_action]), tuple(root['futures']))
                if key in seen:
                    continue
                seen.add(key)
                weights = np.asarray(root.get('future_weights', np.ones(len(futures))), float)
                if weights.shape != (len(futures),) or np.any(weights < 0) or weights.sum() <= 0:
                    raise ValueError('Recorded future weights must define a positive population.')
                weights /= weights.sum()
                physical = []; reference = []; physical_weights = []
                for f in futures:
                    left = physical_response(root, root['outcomes'][a, f], fixed_horizon)
                    right = physical_response(root, root['outcomes'][reference_action, f], fixed_horizon)
                    if left is not None and right is not None:
                        physical.append(left); reference.append(right); physical_weights.append(weights[f])
                active = bool(physical) and sum(physical_weights) > 0
                option = root['options'][a]
                reference_option = root['options'][reference_action]
                def events(index):
                    observations = [root['outcomes'][index, f] for f in futures]
                    if any(row['outcome'] not in (1, 2, 3, 4) for row in observations):
                        raise ValueError('Event learning requires measured terminal event codes.')
                    counts = [sum(weights[f]*(observations[f]['outcome'] == cause) for f in futures)
                              for cause in (1, 2, 3)]
                    exposure = sum(weights[f]*(observations[f]['duration']-.2*root['root_steps'])/80.
                                   for f in futures)
                    if exposure <= 0 or exposure > 1.-root['root_steps']/400.+1e-5:
                        raise ValueError('Measured event time must follow the root within the episode horizon.')
                    return counts, exposure
                event, exposure = events(a); reference_event, reference_exposure = events(reference_action)
                row = dict(context=root_context(root), controls=root['effect_controls'],
                    programs=[option[0]//10**i % 10 for i in range(n)], timing=option[1:],
                    reference_programs=[reference_option[0]//10**i % 10 for i in range(n)],
                    reference_timing=reference_option[1:],
                    target=sum(weights[f]*root['outcomes'][a, f]['success'] for f in futures),
                    reference_target=sum(weights[f]*root['outcomes'][reference_action, f]['success'] for f in futures),
                    physical=np.average(physical, axis=0, weights=physical_weights) if active else np.zeros(8*n+5),
                    reference_physical=np.average(reference, axis=0, weights=physical_weights) if active else np.zeros(8*n+5),
                    mask=float(active), events=event, reference_events=reference_event,
                    exposure=exposure, reference_exposure=reference_exposure,
                    population_weight=population_weight,
                    joint_target=[sum(weights[f]*(2*int(root['outcomes'][reference_action, f]['success'])
                        +int(root['outcomes'][a, f]['success'])==joint) for f in futures) for joint in range(4)])
                for name, value in row.items():
                    packed[name].append(value)
                roots.append(key[:3])
    if not roots:
        raise ValueError('No complete paired training observations matched the training layouts.')
    tensors = {k: torch.as_tensor(np.asarray(v), dtype=torch.long if 'programs' in k else torch.float32)
               for k, v in packed.items()}
    return tensors, roots


@torch.no_grad()
def catalog_diagnostics(networks, data, layouts, margin=.02, reference=None):
    allowed = set(map(tuple, layouts))
    errors = []; selected = []; predicted = []; choices = []
    for root in data['roots']:
        if tuple(root['scene']) not in allowed:
            continue
        n = len(root['order']); options = np.asarray(root['options'])
        context = torch.as_tensor(root_context(root))[None]
        controls = torch.as_tensor(np.asarray(root['effect_controls']), dtype=torch.float32)[None]
        programs = torch.as_tensor((options[:, :1]//10**np.arange(n)) % 10)
        timing = torch.as_tensor(options[:, 1:])
        probability = torch.stack([net.forward_catalog(context, controls, programs, timing).sigmoid()[0]
                                   for net in networks]).mean(0).numpy()
        weights = np.asarray(root.get('future_weights', np.ones(len(root['futures']))), float)
        weights /= weights.sum()
        target = np.asarray([sum(weights[f]*root['outcomes'][a, f]['success']
            for f in range(len(weights))) for a in range(len(options))])
        reference_index = 0
        if reference is not None:
            reference_choice = int(reference.selected_catalog_indices(context, controls)[0])
            reference_option = tuple(reference.options[reference_choice].tolist())
            mapping = {tuple(option): index for index, option in enumerate(options)}
            if reference_option not in mapping:
                raise ValueError('Policy-improvement diagnostics require the measured reference action.')
            reference_index = mapping[reference_option]
        gain = probability-probability[reference_index]
        choice = int(np.argmax(gain))
        if gain[choice] <= margin:
            choice = reference_index
        active = np.arange(len(options)) != reference_index
        errors.extend(((gain[active]-(target[active]-target[reference_index]))**2).tolist())
        selected.append(float(target[choice]-target[reference_index])); predicted.append(float(gain[choice]))
        choices.append(dict(scene=tuple(root['scene']), option=tuple(map(int, options[choice])),
            reference_option=tuple(map(int, options[reference_index])),
            observed_gain=selected[-1], predicted_gain=predicted[-1]))
    layout_gains = {}
    for row in choices:
        layout_gains.setdefault(row['scene'], []).append(row['observed_gain'])
    return dict(roots=len(selected), effect_mse=float(np.mean(errors)),
        layout_mean_gain=float(np.mean([np.mean(values) for values in layout_gains.values()])),
        comparator='zero_intervention' if reference is None else 'frozen_continuation',
        measured_options=sorted(set(len(root['options']) for root in data['roots']
                                    if tuple(root['scene']) in allowed)),
        selected_measured_gain=float(np.mean(selected)), selected_predicted_gain=float(np.mean(predicted)),
        intervened=sum(row['option'][0] != 0 for row in choices), choices=choices)


def measured_successors(datasets, layouts, factual_banks=(), fixed_horizon=False):
    allowed = set(map(tuple, layouts)); seen = set(); contexts = []; targets = []; programs = []; timings = []
    for data in datasets:
        if not data.get('completed'):
            raise ValueError('Successor values require completed terminal measurements.')
        for root in data['roots']:
            if tuple(root['scene']) not in allowed:
                continue
            for (action, future), outcome in root['outcomes'].items():
                key = (tuple(root['scene']), root['prefix_future'], root['root_steps'],
                       tuple(root['options'][action]), root['futures'][future])
                response = physical_response(root, outcome, fixed_horizon)
                if key in seen or response is None:
                    continue
                seen.add(key); n = len(root['order']); context = root_context(root)
                contexts.append(np.concatenate((context[:8*n]+response[:8*n],
                    context[8*n:-5], context[-5:]+response[-5:])))
                targets.append(float(outcome['success']))
                option = root['options'][action]
                programs.append([option[0]//10**i % 10 for i in range(n)])
                timings.append(option[1:])
    factual_seen = set()
    for bank in factual_banks:
        if not bank.get('completed'):
            raise ValueError('Factual value learning requires complete trajectories.')
        for record in bank['records']:
            if tuple(record['scene']) not in allowed:
                continue
            for context, order, target in zip(record['contexts'], record['orders'], record['values']):
                if target not in (0., 1.):
                    raise ValueError('Factual potential targets must be measured terminal success.')
                key = (tuple(record['scene']), record['future'], round(float(context[-1])*400))
                if key in factual_seen:
                    continue
                factual_seen.add(key); n = len(order)
                contexts.append(np.concatenate((context[:-5], np.asarray(order, np.float32)/max(n-1, 1), context[-5:])))
                targets.append(float(target)); programs.append([0]*n); timings.append([0, 0])
    if not contexts:
        raise ValueError('No measured continuation states matched the training layouts.')
    return (torch.tensor(np.asarray(contexts), dtype=torch.float32), torch.tensor(targets),
            torch.tensor(programs, dtype=torch.long), torch.tensor(timings, dtype=torch.float32))


def measured_factual_values(factual_banks, layouts):
    """Use actual predecessor trajectories without repeating long episodes more.

    Every trajectory has the same sampling mass; its eligible observed states
    divide that mass. No terminal result or future identifier is an input to
    the estimator. Keeping this pool separate prevents outcome-stratified
    intervention acquisition from defining the factual success prior.
    """
    allowed = set(map(tuple, layouts)); seen = set()
    packed = {name: [] for name in ('context', 'controls', 'target', 'sampling_weight')}
    for bank in factual_banks:
        if not bank.get('completed'):
            raise ValueError('Factual calibration needs completed real trajectories.')
        for record in bank['records']:
            key = (tuple(record['scene']), record['future'])
            if key[0] not in allowed or key in seen:
                continue
            seen.add(key)
            count = len(record['contexts'])
            if not count or any(len(record[name]) != count for name in ('controls', 'orders', 'values')):
                raise ValueError('Factual states, commands and terminal labels must align.')
            for context, controls, order, target in zip(record['contexts'], record['controls'],
                                                       record['orders'], record['values']):
                if target not in (0., 1.):
                    raise ValueError('Factual calibration uses measured terminal success.')
                n = len(order)
                packed['context'].append(np.concatenate((context[:-5],
                    np.asarray(order, np.float32)/max(n-1, 1), context[-5:])))
                packed['controls'].append(controls)
                packed['target'].append(float(target))
                packed['sampling_weight'].append(1./count)
    if not packed['target']:
        raise ValueError('No factual trajectories matched the training layouts.')
    return {name: torch.as_tensor(np.asarray(values), dtype=torch.float32)
            for name, values in packed.items()}


def fit_successor(datasets, validation, split_model, *, paired, seed=43, updates=2000,
                  width=48, batch_size=128, physical_weight=.1, effect_weight=4.,
                  response_difference_weight=.1, learning_rate=3e-4, members=1,
                  value_form='potential', complete_catalog=True, factual_banks=(),
                  all_action_pairs=False, select_by_validation=False, root_balanced=False,
                  factual_value_weight=1.):
    """Fit matched architectures and update budgets; reserve validation layouts."""
    if value_form not in ('potential', 'physical_bottleneck', 'planned_successor', 'constrained_planned', 'local_quadratic', 'event_time', 'paired_outcome'):
        raise ValueError('Unknown successor value representation.')
    factory = {'potential': PairedSuccessorNetwork, 'physical_bottleneck': PhysicalBottleneckSuccessorNetwork,
               'planned_successor': PlannedSuccessorNetwork,
               'constrained_planned': ConstrainedPlannedSuccessorNetwork,
               'local_quadratic': LocalQuadraticSuccessorNetwork,
               'event_time': EventTimeSuccessorNetwork,
               'paired_outcome': PairedOutcomeSuccessorNetwork}[value_form]
    if value_form=='paired_outcome' and all_action_pairs:
        raise ValueError('Conditional outcome learning uses the common factual reference.')
    train_layouts = split_model['successor_task_fit']['training_layouts']
    validation_layouts = split_model['successor_task_fit']['validation_layouts']
    execution = split_model['successor_task_value']
    relative_scale = execution.get('feedback_relative_scale')
    if relative_scale is None and value_form not in ('potential', 'physical_bottleneck', 'planned_successor', 'constrained_planned'):
        raise ValueError('Local response models require relative one-period interventions.')
    if not np.isfinite(factual_value_weight) or factual_value_weight < 0.:
        raise ValueError('The factual calibration weight must be finite and nonnegative.')
    if factual_banks and value_form != 'paired_outcome' and (relative_scale is not None
            or value_form not in ('planned_successor', 'constrained_planned')):
        raise ValueError('M1 factual trajectories belong to the M1-continuation planned successor fit.')
    if complete_catalog:
        validation = complete_validation_catalog(validation, datasets[0])
    elif not validation.get('completed'):
        raise ValueError('Validation requires completed measured collections.')
    if set(map(tuple, train_layouts)) & set(map(tuple, validation_layouts)):
        raise ValueError('Training and validation layouts must be disjoint.')
    fixed_horizon = value_form in ('planned_successor', 'constrained_planned') and relative_scale is None
    data, roots = measured_pairs(datasets, train_layouts, fixed_horizon, all_action_pairs)
    sampling_weights = None
    if root_balanced:
        from collections import Counter
        counts = Counter(roots)
        sampling_weights = torch.tensor([1./counts[root] for root in roots], dtype=torch.float64)
    population_weights = data['population_weight'].double()
    if not bool((population_weights == 1.).all()):
        sampling_weights = population_weights if sampling_weights is None else sampling_weights*population_weights
    factual_values = (measured_factual_values(factual_banks, train_layouts)
                      if factual_banks and value_form == 'paired_outcome' else None)
    successor_states = (measured_successors(datasets, train_layouts, factual_banks, fixed_horizon)
                        if value_form in ('physical_bottleneck', 'planned_successor', 'constrained_planned') else None)
    n = data['controls'].shape[1]
    valid = data['mask'].bool()
    if not bool(valid.any()):
        raise ValueError('Physical representation training needs measured successor states.')
    scale = torch.cat((data['physical'][valid], data['reference_physical'][valid])).std(0).clamp_min(.05)
    networks = []; history = []; selected_updates = []; started = time.monotonic()
    torch.set_grad_enabled(True)
    for member in range(members):
        torch.manual_seed(seed+1009*member)
        net = factory(n, width, feedback_relative_scale=relative_scale)
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=.01)
        generator = torch.Generator().manual_seed(seed+1009*member+71)
        best_score = (-float('inf'), -float('inf')); best_state = None; best_step = updates
        for step in range(1, updates+1):
            indices = (torch.randint(len(roots), (batch_size,), generator=generator)
                if sampling_weights is None else
                torch.multinomial(sampling_weights, batch_size, replacement=True, generator=generator))
            b = {k: v[indices] for k, v in data.items()}
            if value_form == 'paired_outcome':
                logits, response, factual, rescue, harm = net.forward_components(
                    b['context'], b['controls'], b['programs'], b['timing'])
                reference_logits, reference_response = net.forward_with_response(
                    b['context'], b['controls'], b['reference_programs'], b['reference_timing'])
            elif value_form == 'event_time':
                logits, response, rates = net.forward_with_events(b['context'], b['controls'], b['programs'], b['timing'])
                reference_logits, reference_response, reference_rates = net.forward_with_events(
                    b['context'], b['controls'], b['reference_programs'], b['reference_timing'])
            else:
                logits, response = net.forward_with_response(b['context'], b['controls'], b['programs'], b['timing'])
                reference_logits, reference_response = net.forward_with_response(
                    b['context'], b['controls'], b['reference_programs'], b['reference_timing'])
            loss, parts = paired_successor_objective(logits, reference_logits, response, reference_response,
                b['target'], b['reference_target'], b['physical'], b['reference_physical'], b['mask'], scale,
                paired=paired, physical_weight=physical_weight, effect_weight=effect_weight,
                response_difference_weight=response_difference_weight, absolute_weight=float(value_form != 'event_time'))
            if value_form == 'paired_outcome' and paired:
                conditional_loss = paired_outcome_conditional_loss(rescue, harm, b['joint_target'])
                loss = loss+conditional_loss
                parts['conditional_outcome'] = conditional_loss.detach()
            if factual_values is not None and factual_value_weight:
                factual_indices = torch.multinomial(factual_values['sampling_weight'], batch_size,
                    replacement=True, generator=generator)
                factual_context = factual_values['context'][factual_indices]
                factual_controls = factual_values['controls'][factual_indices]
                _, factual_pooled = net.encode(factual_context, factual_controls,
                    torch.zeros((batch_size, n), dtype=torch.long), torch.zeros((batch_size, 2)))
                factual_prediction = net.task_value(factual_pooled).squeeze(-1)
                factual_loss = F.binary_cross_entropy_with_logits(factual_prediction,
                    factual_values['target'][factual_indices])
                loss = loss+factual_value_weight*factual_loss
                parts['observed_factual_value'] = factual_loss.detach()
            if value_form == 'event_time':
                event_loss = .5*((rates.sum(-1)*b['exposure']-(b['events']*rates.log()).sum(-1)).mean()
                    +(reference_rates.sum(-1)*b['reference_exposure']
                      -(b['reference_events']*reference_rates.log()).sum(-1)).mean())
                loss = loss+event_loss; parts['event_likelihood'] = event_loss.detach()
            if successor_states is not None:
                successor_indices = torch.randint(len(successor_states[0]), (batch_size,), generator=generator)
                successor = [values[successor_indices] for values in successor_states]
                prediction = (net.value_state(successor[0], successor[2], successor[3])
                              if value_form in ('planned_successor', 'constrained_planned') else net.value_state(successor[0]))
                successor_loss = F.binary_cross_entropy_with_logits(
                    prediction, successor[1])
                loss = loss+successor_loss; parts['observed_successor_value'] = successor_loss.detach()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.); optimizer.step()
            if step % 500 == 0 or step == updates:
                net.eval()
                metrics = catalog_diagnostics([net], validation, validation_layouts)
                row = dict(member=member, step=step, loss=float(loss.detach()),
                    **{k: float(v) for k, v in parts.items()},
                    **{k: v for k, v in metrics.items() if k != 'choices'})
                history.append(row); print(row, flush=True); net.train()
                score = (metrics['layout_mean_gain'], -metrics['effect_mse'])
                if select_by_validation and score > best_score:
                    best_score = score; best_state = copy.deepcopy(net.state_dict()); best_step = step
        if best_state is not None:
            net.load_state_dict(best_state)
        selected_updates.append(best_step)
        networks.append(net.eval())
    reference = execution.get('policy_reference')
    if reference is None and execution.get('intervention_budget', 0) != 1:
        raise ValueError('Recurrent improvement must preserve the frozen continuation used for its labels.')
    from plain_return import PlainReturnDecision, PairedReturnDecision
    reference_decision = None
    if reference is not None:
        recipe = copy.deepcopy(reference); kind = recipe.pop('kind')
        reference_factory = {'plain_mlp_return': PlainReturnDecision, 'paired_mlp_return': PairedReturnDecision}[kind]
        reference_decision = reference_factory(**recipe).eval()
    diagnostics = catalog_diagnostics(networks, validation, validation_layouts,
                                      reference=reference_decision)
    return dict(networks=networks, diagnostics=diagnostics, policy_reference=copy.deepcopy(reference),
        execution={key: execution[key] for key in ('feedback_relative_hold', 'isolate_factual_start',
            'intervention_budget', 'decision_period', 'velocity_reference') if key in execution},
        fit=dict(paired=paired, seed=seed, updates=updates, members=members, width=width, value_form=value_form,
            root_balanced=bool(root_balanced),
            batch_size=batch_size, physical_weight=physical_weight, effect_weight=effect_weight,
            response_difference_weight=response_difference_weight, learning_rate=learning_rate,
            training_layouts=train_layouts, validation_layouts=validation_layouts,
            training_pairs=len(roots), physical_pairs=int(valid.sum()), history=history,
            measured_successor_state_labels=0 if successor_states is None else len(successor_states[0]),
            measured_factual_state_labels=0 if factual_values is None else len(factual_values['target']),
            factual_value_weight=factual_value_weight,
            population_weight_range=[float(population_weights.min()), float(population_weights.max())],
            validation_catalog_complete=complete_catalog,
            all_action_pairs=all_action_pairs, selected_updates=selected_updates,
            checkpoint_selection='validation_layout_mean_gain' if select_by_validation else 'final_update',
            elapsed_seconds=time.monotonic()-started, response_scale=scale,
            parameter_count=sum(p.numel() for p in networks[0].parameters()),
            continuation_context=copy.deepcopy(datasets[0].get('continuation_context'))))


def make_checkpoint(result, base_path, options, *, gain_margin=.02, runtime_sources=None):
    networks = result['networks']; first = networks[0]
    base_path = Path(base_path).resolve()
    checkpoint = dict(format='paired-successor-v1', base_policy=str(base_path),
        base_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
        successor_task_value=dict(kind=('paired_outcome_successor_return' if isinstance(first, PairedOutcomeSuccessorNetwork)
            else 'constrained_planned_successor_return' if isinstance(first, ConstrainedPlannedSuccessorNetwork)
            else 'planned_successor_return' if isinstance(first, PlannedSuccessorNetwork)
            else 'physical_bottleneck_successor_return' if isinstance(first, PhysicalBottleneckSuccessorNetwork)
            else 'event_time_successor_return' if isinstance(first, EventTimeSuccessorNetwork)
            else 'local_quadratic_successor_return' if isinstance(first, LocalQuadraticSuccessorNetwork)
            else 'paired_successor_return'), defenders=first.defenders,
            width=first.width, identity_conditioned=True, action_encoding='physical_delta',
            feedback_relative_scale=first.feedback_relative_scale, feedback_relative_hold=True,
            isolate_factual_start=True, state=first.state_dict(), options=options,
            ensemble_states=[net.state_dict() for net in networks[1:]], decision_period=5,
            gain_margin=gain_margin, intervention_budget=0,
            policy_reference=copy.deepcopy(result.get('policy_reference'))),
        successor_task_fit=result['fit'], validation_diagnostics=result['diagnostics'],
        runtime_sources=runtime_sources or {}, completed=True)
    checkpoint['successor_task_value'].update(result.get('execution', {}))
    return checkpoint


def install_runtime():
    """Install the policy factory only inside this experiment's Python process."""
    import develop_source_fusion as d
    from plain_return import PlainReturnDecision
    from functools import lru_cache
    if getattr(d.joint_load, '_paired_successor', False):
        return
    original = d.joint_load
    from envs.TADgame import TADEnv
    original_reset = TADEnv.reset; original_packet = d.joint_packet
    episode_sequence = [0]

    def reset_with_policy_key(environment, *args, **kwargs):
        result = original_reset(environment, *args, **kwargs)
        initial = np.asarray([[boat.x, boat.y, boat.theta] for boat in
                              [*environment.defender_list, environment.attacker]], dtype=np.float64)
        environment._gate_random_key = int.from_bytes(hashlib.blake2b(
            initial.tobytes(), digest_size=8).digest(), 'little') & (2**63-1)
        environment._gate_episode_key = episode_sequence[0]
        episode_sequence[0] += 1
        return result

    def packet_with_policy_key(environment, observation):
        result = original_packet(environment, observation)
        if hasattr(environment, '_gate_random_key'):
            result['_gate_random_key'] = np.array([environment._gate_random_key], dtype=np.int64)
            result['_gate_episode_key'] = np.array([environment._gate_episode_key], dtype=np.int64)
        return result

    TADEnv.reset = reset_with_policy_key; d.joint_packet = packet_with_policy_key

    class PairedSuccessorDecision(PlainReturnDecision):
        network_type = PairedSuccessorNetwork

    class LocalQuadraticSuccessorDecision(PlainReturnDecision):
        network_type = LocalQuadraticSuccessorNetwork

    class PairedOutcomeSuccessorDecision(PlainReturnDecision):
        network_type = PairedOutcomeSuccessorNetwork

    class PhysicalBottleneckSuccessorDecision(PlainReturnDecision):
        network_type = PhysicalBottleneckSuccessorNetwork

    class PlannedSuccessorDecision(PlainReturnDecision):
        network_type = PlannedSuccessorNetwork

    class ConstrainedPlannedSuccessorDecision(PlainReturnDecision):
        network_type = ConstrainedPlannedSuccessorNetwork

    class ConsensusSuccessorDecision(ConstrainedPlannedSuccessorDecision):
        def __init__(self, *args, consensus_reference, **kwargs):
            super().__init__(*args, **kwargs)
            from plain_return import PairedReturnDecision
            recipe = copy.deepcopy(consensus_reference); kind = recipe.pop('kind')
            if self.policy_reference is not None or self.feedback_relative_scale is not None:
                raise ValueError('Consensus improvement uses a common fixed M1 continuation.')
            if (recipe.get('intervention_budget') != 1 or self.intervention_budget != 1
                    or recipe.get('feedback_relative_scale') is not None):
                raise ValueError('Both critics must describe one intervention followed by M1.')
            recipe['options'] = self.options.tolist()
            factory = {'plain_mlp_return': PlainReturnDecision, 'paired_mlp_return': PairedReturnDecision}[kind]
            self.consensus_reference = factory(**recipe).requires_grad_(False)

        def catalog_estimates(self, context, controls):
            _, physical_value = super().catalog_estimates(context, controls)
            _, direct_value = self.consensus_reference.catalog_estimates(context, controls)
            gain = torch.minimum(physical_value-physical_value[:, :1],
                                 direct_value-direct_value[:, :1])
            return gain, (physical_value[:, :1]+gain).clamp(0., 1.)

    class OnPolicyGateDecision(PlainReturnDecision):
        network_type = OnPolicyGateNetwork

        def __init__(self, *args, proposal_reference, policy_seed=43,
                     score_mode='policy', gate_margin=.02, stochastic_execution=False,
                     latest_proposal_tick=None, **kwargs):
            super().__init__(*args, **kwargs)
            from plain_return import PairedReturnDecision
            recipe = copy.deepcopy(proposal_reference); kind = recipe.pop('kind')
            factory = {'plain_mlp_return': PlainReturnDecision, 'paired_mlp_return': PairedReturnDecision}[kind]
            self.proposal_reference = factory(**recipe).requires_grad_(False)
            if (self.intervention_budget != 1 or self.proposal_reference.intervention_budget != 1
                    or self.feedback_relative_scale is not None or self.policy_reference is not None
                    or not torch.equal(self.options, self.proposal_reference.options)):
                raise ValueError('Gate learning needs a common one-intervention M1 continuation and catalog.')
            self.sampling = bool(stochastic_execution); self.collect = False; self.trajectory = []
            self.policy_generator = torch.Generator().manual_seed(policy_seed)
            self.policy_seed = policy_seed; self._policy_uniform = None
            self._trace_eligible = None; self._trace_episodes = None
            if score_mode not in ('policy', 'measured_gain', 'joint_effect') or not 0. <= gate_margin < 1.:
                raise ValueError('Unknown gate score or invalid probability margin.')
            self.score_mode = score_mode; self.gate_margin = gate_margin
            self.latest_proposal_tick = latest_proposal_tick
            self.motion_history = (PublicMotionHistory() if self.network.representation
                                   in ('compact_history', 'dense_history') else None)

        def eligible(self, packet):
            result = super().eligible(packet)
            if self.network.representation == 'compact_history':
                result &= (packet['node'][:, 0, 15]*400.).round() >= 5
            if self.latest_proposal_tick is not None:
                result &= (packet['node'][:, 0, 15]*400.).round() <= self.latest_proposal_tick
            return result

        def forward(self, packet, action):
            if self.motion_history is not None:
                if '_gate_episode_key' not in packet:
                    raise ValueError('Observed history requires an explicit episode memory key.')
                context, _ = self.public_context(packet)
                self._history_context = self.motion_history.previous(context, packet['_gate_episode_key'])
            if self.sampling:
                if '_gate_random_key' not in packet:
                    raise ValueError('Stochastic execution requires an episode-local random key.')
                key = packet['_gate_random_key'].flatten().detach().cpu().numpy().astype(np.uint64)
                tick = (packet['node'][:, 0, 15]*400.).round().long().detach().cpu().numpy().astype(np.uint64)
                # A counter-based stream gives each scene its own policy noise.
                # The key is sampling metadata, never a network feature.
                value = key ^ (tick*np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(self.policy_seed)
                value = value+np.uint64(0x9E3779B97F4A7C15)
                value = (value ^ (value >> 30))*np.uint64(0xBF58476D1CE4E5B9)
                value = (value ^ (value >> 27))*np.uint64(0x94D049BB133111EB)
                value = value ^ (value >> 31)
                self._policy_uniform = torch.from_numpy((value >> 11).astype(np.float64)/2**53)
            if self.collect:
                if '_paired_episode' not in packet:
                    raise ValueError('Training decisions need an explicit rollout index, kept outside policy features.')
                self._trace_eligible = self.eligible(packet).detach()
                self._trace_episodes = packet['_paired_episode'].flatten().detach()
            return super().forward(packet, action)

        def catalog_estimates(self, context, controls):
            ranking, values = self.proposal_reference.catalog_estimates(context, controls)
            proposal = ranking.argmax(-1)
            score = self.proposal_reference.decision_scores(ranking, values)
            improvement = score.gather(1, proposal[:, None]).flatten()-score[:, 0]
            proposal = torch.where(improvement > self.proposal_reference.gain_margin,
                                   proposal, torch.zeros_like(proposal))
            features = self.network.features(context, controls, self.programs[proposal],
                self.options[proposal, 1:], torch.stack((values[:, 0], improvement), -1))
            if self.motion_history is not None:
                features = torch.cat((features, self._history_context.to(features.device)), -1)
            logits, baseline = self.network(features)
            if self.score_mode in ('measured_gain', 'joint_effect'):
                if self.sampling:
                    raise ValueError('A measured gain model is not a stochastic policy logit.')
                gain = (measured_joint_gain(logits, baseline) if self.score_mode == 'joint_effect'
                        else (baseline+logits).sigmoid()-baseline.sigmoid())
                accepted = gain > self.gate_margin
            else:
                accepted = (self._policy_uniform.to(logits.device)
                            < logits.sigmoid() if self.sampling else logits >= 0.)
            if self.collect:
                valid = self._trace_eligible.to(logits.device) & (proposal != 0)
                if bool(valid.any()):
                    action = accepted.to(logits.dtype)
                    log_probability = -F.binary_cross_entropy_with_logits(logits, action, reduction='none')
                    self.trajectory.append(dict(features=features[valid].detach().cpu(),
                        action=action[valid].detach().cpu(), log_probability=log_probability[valid].detach().cpu(),
                        value=baseline[valid].detach().cpu(),
                        episode=self._trace_episodes.to(logits.device)[valid].detach().cpu().long()))
            choice = torch.where(accepted, proposal, torch.zeros_like(proposal))
            # These scores encode a sampled policy action, not success values.
            chosen = torch.zeros_like(values).scatter_(1, choice[:, None], 1.)
            return chosen, chosen

    class EventTimeSuccessorDecision(PlainReturnDecision):
        network_type = EventTimeSuccessorNetwork

    @lru_cache(maxsize=32)
    def load_policy(path, revision, ablation=(False, False, False)):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != 'paired-successor-v1':
            return original(path, revision, ablation)
        base = Path(checkpoint['base_policy'])
        if hashlib.sha256(base.read_bytes()).hexdigest() != checkpoint['base_sha256']:
            raise ValueError('The fixed mechanism-one policy changed after training.')
        actor = copy.deepcopy(original(str(base), base.stat().st_mtime_ns, ablation))
        recipe = dict(checkpoint['successor_task_value']); kind = recipe.pop('kind')
        factory = (OnPolicyGateDecision if kind == 'on_policy_gate' else
                   PairedOutcomeSuccessorDecision if kind == 'paired_outcome_successor_return' else
                   ConsensusSuccessorDecision if kind == 'consensus_successor_return'
                   else ConstrainedPlannedSuccessorDecision if kind == 'constrained_planned_successor_return'
                   else PlannedSuccessorDecision if kind == 'planned_successor_return'
                   else PhysicalBottleneckSuccessorDecision if kind == 'physical_bottleneck_successor_return'
                   else EventTimeSuccessorDecision if kind == 'event_time_successor_return'
                   else LocalQuadraticSuccessorDecision if kind == 'local_quadratic_successor_return'
                   else PairedSuccessorDecision if kind == 'paired_successor_return' else None)
        if factory is None:
            raise ValueError('Unknown successor policy kind.')
        actor.successor_task_value = factory(**recipe).eval()
        actor.physical_interventions = True; actor.fixed_intervention_clock = True
        actor.training_gamma = 1.; actor.training_step_cost = 0.; actor.gate_period = 1
        removal = tuple(ablation)+(False,)*(3-len(ablation))
        removal = tuple(a or b for a, b in zip(removal, d.EVALUATION_ABLATION))
        return d.ablate_joint_mechanisms(actor, *removal)

    load_policy._paired_successor = True
    d.joint_load = load_policy


def restore_runtime(sources):
    """Replay the exact implementation embedded in an experiment checkpoint."""
    import sys
    import types
    directory = Path(__file__).resolve().parent
    for name, source in sources.items():
        if name == 'paired_successor_value.py':
            continue
        module = types.ModuleType(Path(name).stem)
        module.__file__ = str(directory/name)
        sys.modules[module.__name__] = module
        exec(compile(source, module.__file__, 'exec'), module.__dict__)


def _gate_rollout_worker(request):
    policy, tasks, sources, device, sampling, collect, seed, offset = request
    if not getattr(_gate_rollout_worker, 'initialized', False):
        restore_runtime(sources)
        import run_remote_evidence as r
        r.initialize(); install_runtime(); torch.set_num_threads(1)
        _gate_rollout_worker.initialized = True
    import develop_source_fusion as d
    import run_remote_evidence as r
    from envs.TADgame import TADEnv
    path = Path(policy)
    actor = d.joint_load(str(path), path.stat().st_mtime_ns)
    gate = actor.successor_task_value
    if gate is not None:
        gate.inference_device = device
    if collect:
        gate.sampling = sampling; gate.collect = True; gate.trajectory = []
        gate.policy_generator.manual_seed(seed)
        gate.policy_seed = seed
    reset_original = TADEnv.reset; packet_original = d.joint_packet
    episode_counter = [0]

    def tracked_reset(environment, *args, **kwargs):
        result = reset_original(environment, *args, **kwargs)
        environment._paired_training_episode = episode_counter[0]
        episode_counter[0] += 1
        return result

    def tracked_packet(environment, observation):
        result = packet_original(environment, observation)
        result['_paired_episode'] = np.array([environment._paired_training_episode], dtype=np.int64)
        return result

    if collect:
        TADEnv.reset = tracked_reset; d.joint_packet = tracked_packet
    try:
        rows = r.batched_credit_rows(str(path), tasks, len(tasks), device, actor_override=actor)
    finally:
        TADEnv.reset = reset_original; d.joint_packet = packet_original
        if collect:
            gate.sampling = False; gate.collect = False
    trace = None
    if collect:
        if episode_counter[0] != len(tasks):
            raise ValueError('Incomplete trajectory identities.')
        if gate.trajectory:
            trace = {key: torch.cat([item[key] for item in gate.trajectory]).numpy()
                     for key in gate.trajectory[0]}
            trace['episode'] += offset
        gate.trajectory = []
    actor.cpu()
    return rows, trace


def _history_prefix_worker(request):
    policy, tasks, sources, device, samples = request
    if not getattr(_history_prefix_worker, 'initialized', False):
        restore_runtime(sources)
        import run_remote_evidence as r
        r.initialize(); install_runtime(); torch.set_num_threads(1)
        _history_prefix_worker.initialized = True
    import develop_source_fusion as d
    import run_remote_evidence as r
    from envs.TADgame import TADEnv
    import types
    path = Path(policy); actor = d.joint_load(str(path), path.stat().st_mtime_ns)
    gate = actor.successor_task_value
    original_forward = gate.forward
    original_reset = TADEnv.reset; original_packet = d.joint_packet
    counter = [0]; history = PublicMotionHistory(); captured = {}; errors = []
    indexed = {}
    for index, episode, feature in samples:
        tick = int(round(float(feature[9*gate.network.defenders+4])*400.))
        indexed.setdefault((episode, tick), []).append((index, feature))

    class PrefixComplete(Exception):
        pass

    def reset(environment, *args, **kwargs):
        result = original_reset(environment, *args, **kwargs)
        environment._history_replay_episode = counter[0]; counter[0] += 1
        return result

    def packet(environment, observation):
        result = original_packet(environment, observation)
        result['_history_replay_episode'] = np.array([environment._history_replay_episode], np.int64)
        return result

    def forward(self, packet, action):
        context, _ = self.public_context(packet)
        episodes = packet['_history_replay_episode'].flatten()
        previous = history.previous(context, episodes)
        for row, episode in enumerate(episodes.detach().cpu().tolist()):
            tick = int(round(float(context[row, -1])*400.))
            for index, feature in indexed.get((episode, tick), []):
                if index in captured:
                    continue
                error = float((context[row].detach().cpu()-feature[:context.shape[-1]]).abs().max())
                if error > 1e-4:
                    raise ValueError(f'Observed-prefix replay differs from its saved decision by {error:.6g}.')
                errors.append(error)
                captured[index] = torch.cat((feature, previous[row].detach().cpu()))
        if len(captured) == len(samples):
            raise PrefixComplete()
        return action

    TADEnv.reset = reset; d.joint_packet = packet; gate.forward = types.MethodType(forward, gate)
    try:
        try:
            r.batched_credit_rows(str(path), tasks, len(tasks), device, actor_override=actor)
        except PrefixComplete:
            pass
    finally:
        TADEnv.reset = original_reset; d.joint_packet = original_packet; gate.forward = original_forward
    if len(captured) != len(samples):
        raise ValueError('A saved intervention state was not reached by its factual prefix.')
    return captured, max(errors, default=0.)


def prepare_observed_history(data_paths, output, *, workers=2, device='cuda:0'):
    """Replay public prefixes, retaining the original measured terminal pairs."""
    import io
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import run_remote_evidence as r
    inputs = {}; records = []
    for path in data_paths:
        raw = Path(path).read_bytes(); inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        records.append(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False))
    if any(not record.get('completed') for record in records):
        raise ValueError('Observed history must be attached to completed measurements.')
    template = records[0]
    if any(record['base_sha256'] != template['base_sha256'] or
           record['proposal_sha256'] != template['proposal_sha256'] for record in records):
        raise ValueError('All histories must share the same factual and proposal policies.')
    observations = []; seen = {}; requests = []
    for path, record in zip(data_paths, records):
        for batch in record.get('measured_proposals', []):
            iteration = batch['iteration']
            metrics = record['history'][iteration-1]
            if metrics['episodes'] != 64 or metrics.get('rollout_workers', 4) != 4:
                raise ValueError('Prefix reconstruction needs the original four batches of sixteen.')
            tasks = record['training_tasks'][(iteration-1)*64:iteration*64]
            location = {(tuple(task[0]), task[1]): index for index, task in enumerate(tasks)}
            grouped = [[] for _ in range(4)]
            for index, task in enumerate(batch['tasks']):
                feature = batch['features'][index].contiguous()
                target = (float(batch['actual'][index]), float(batch['factual'][index]))
                key = (tuple(task[0]), task[1], feature.numpy().tobytes(), target)
                if key in seen:
                    continue
                identity = len(observations); seen[key] = identity
                observations.append((task, target))
                episode = location[tuple(task[0]), task[1]]
                grouped[episode//16].append((identity, episode % 16, feature))
            for chunk, selected in enumerate(grouped):
                if selected:
                    requests.append((str(path), tasks[chunk*16:(chunk+1)*16],
                                     record['runtime_sources'], device, selected))
    features = {}; maximum_error = 0.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(_history_prefix_worker, request) for request in requests]
        for number, future in enumerate(futures):
            result, error = future.result(); features.update(result); maximum_error = max(maximum_error, error)
            if (number+1) % 8 == 0 or number+1 == len(futures):
                print({'history_observations': len(features), 'total': len(observations),
                       'maximum_context_error': maximum_error}, flush=True)
    if len(features) != len(observations):
        raise ValueError('Every recorded terminal pair needs its observed prefix.')
    result = {key: copy.deepcopy(template[key]) for key in ('base_policy', 'base_sha256',
        'proposal_policy', 'proposal_sha256', 'successor_task_value', 'runtime_sources')}
    result['runtime_sources']['paired_successor_value.py'] = Path(__file__).read_text(encoding='utf-8-sig')
    result.update(format='paired-observation-history-v1', completed=True, development_only=True,
        input_sha256=inputs, history_replay=dict(window_ticks=5, maximum_context_error=maximum_error,
        observations=len(observations), tasks_replayed=sum(len(request[1]) for request in requests),
        terminal_targets='Unchanged actual outcomes from the source paired trajectories.'),
        measured_proposals=[dict(features=torch.stack([features[index] for index in range(len(observations))]),
            actual=torch.tensor([target[0] for _, target in observations]),
            factual=torch.tensor([target[1] for _, target in observations]),
            tasks=[task for task, _ in observations])])
    r.atomic_save(result, output)


def train_on_policy_gate(base_policy, proposal_model, output_prefix, runtime_sources,
                         *, iterations=16, batch_size=64, seed=43, device='cuda:0', workers=1,
                         warm_start=None, representation='dense', layout_start=None):
    """Fit matched policies to complete trajectories and their actual outcomes.

    The paired arm uses the outcome of the fixed M1 policy under the same
    environment randomness as an action-independent control variate. Its scale
    is estimated from earlier complete observations, never from the current
    sampled actions. It is not exposed to the policy at inference. Both arms use identical initial weights,
    proposal catalogs, rollout tasks, update budgets, and terminal rewards.
    """
    import develop_source_fusion as d
    import run_remote_evidence as r
    from envs.TADgame import TADEnv
    install_runtime()
    base_policy = Path(base_policy); proposal_model = Path(proposal_model)
    proposal = torch.load(proposal_model, map_location='cpu', weights_only=False)
    reference = proposal['successor_task_value']
    if reference['intervention_budget'] != 1 or reference.get('feedback_relative_scale') is not None:
        raise ValueError('The frozen proposal must have a single intervention and fixed M1 continuation.')
    torch.manual_seed(seed)
    initial = OnPolicyGateNetwork(reference['defenders'], representation=representation)
    sources = dict(runtime_sources, **{'paired_successor_value.py': Path(__file__).read_text(encoding='utf-8-sig')})
    base_hash = hashlib.sha256(base_policy.read_bytes()).hexdigest()
    recipe = dict(kind='on_policy_gate', defenders=reference['defenders'], width=64,
        identity_conditioned=True, action_encoding='physical_delta', velocity_reference='ground',
        decision_period=5, inference_device=device, isolate_factual_start=True,
        intervention_budget=1, gain_margin=0., state=copy.deepcopy(initial.state_dict()),
        options=reference['options'], proposal_reference=reference, policy_seed=seed,
        stochastic_execution=True, representation=representation)
    warm_provenance = None
    initial_control_scale = 1.
    if warm_start is not None:
        warm_start = Path(warm_start)
        warm = torch.load(warm_start, map_location='cpu', weights_only=False)
        if (not warm.get('completed') or warm['base_sha256'] != base_hash
                or warm['proposal_sha256'] != hashlib.sha256(proposal_model.read_bytes()).hexdigest()
                or warm['successor_task_value']['kind'] != 'on_policy_gate'
                or warm['successor_task_value'].get('score_mode', 'policy') != 'policy'):
            raise ValueError('Warm start must be a completed policy with the same fixed policies and action law.')
        state = copy.deepcopy(warm['successor_task_value']['state'])
        if representation == 'dense_history' and warm['successor_task_value'].get('representation', 'dense') == 'dense':
            expanded = initial.state_dict()['encoder.0.weight'].clone().zero_()
            expanded[:, :state['encoder.0.weight'].shape[1]] = state['encoder.0.weight']
            state['encoder.0.weight'] = expanded
        initial.load_state_dict(state)
        recipe['state'] = copy.deepcopy(initial.state_dict())
        warm_provenance = dict(path=str(warm_start), sha256=hashlib.sha256(warm_start.read_bytes()).hexdigest(),
                               training_tasks=warm['training_tasks'])
        if warm.get('measured_proposals'):
            warm_actual = torch.cat([item['actual'] for item in warm['measured_proposals']])
            warm_factual = torch.cat([item['factual'] for item in warm['measured_proposals']])
            centered = warm_factual-warm_factual.mean()
            variance = centered.square().mean()
            if variance > 1e-8:
                initial_control_scale = max(0., min(1., float(
                    ((warm_actual-warm_actual.mean())*centered).mean()/variance)))
    task_start = int(layout_start if layout_start is not None else 830043000+seed*100000)
    if warm_provenance is not None and layout_start is None:
        raise ValueError('A continued policy must use an explicit fresh training layout range.')
    if warm_provenance is not None:
        old_layouts = {task[0][0] for task in warm_provenance['training_tasks']}
        if old_layouts.intersection(range(task_start, task_start+iterations*batch_size)):
            raise ValueError('Continued training layouts overlap the warm-start observations.')
    outputs = {arm: Path(f'{output_prefix}-{arm}-seed{seed}.pth') for arm in ('paired', 'ordinary')}
    models = {}; records = {}; optimizers = {}
    for arm, path in outputs.items():
        record = dict(format='paired-successor-v1', base_policy=str(base_policy), base_sha256=base_hash,
            successor_task_value=copy.deepcopy(recipe), runtime_sources=sources, completed=False,
            development_only=True, proposal_policy=str(proposal_model),
            proposal_sha256=hashlib.sha256(proposal_model.read_bytes()).hexdigest(),
            training_objective='paired_terminal_control_variate' if arm == 'paired' else 'terminal_return',
            training_tasks=[], history=[], seed=seed, warm_start=warm_provenance,
            task_layout_start=task_start, factual_critic_centering=warm_provenance is not None,
            initial_factual_control_scale=initial_control_scale)
        if path.exists():
            record = torch.load(path, map_location='cpu', weights_only=False)
            if (record['completed'] or record['base_sha256'] != base_hash
                    or record['proposal_sha256'] != hashlib.sha256(proposal_model.read_bytes()).hexdigest()):
                raise ValueError('Only an unfinished training run with the same frozen policies can resume.')
            record['runtime_sources'] = sources
            record['successor_task_value']['stochastic_execution'] = True
            for previous in record['history']:
                previous.setdefault('factual_control_scale', 1. if arm == 'paired' else 0.)
        else:
            r.atomic_save(record, path)
        records[arm] = record
        actor = d.joint_load(str(path), path.stat().st_mtime_ns).cpu().eval()
        actor.requires_grad_(False); actor.successor_task_value.network.requires_grad_(True)
        models[arm] = actor
        optimizers[arm] = torch.optim.Adam(actor.successor_task_value.network.parameters(), lr=3e-4)
        if 'optimizer' in record:
            optimizers[arm].load_state_dict(record['optimizer'])
    rng = np.random.default_rng(seed+(830043000 if layout_start is None else task_start))
    shuffle = torch.Generator().manual_seed(seed+930043000)
    executor = None
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))

    def parallel_rollout(policy, tasks, *, collect=False, iteration=0):
        chunk_size = (len(tasks)+workers-1)//workers
        futures = [executor.submit(_gate_rollout_worker,
            (str(policy), tasks[begin:begin+chunk_size], sources, device, collect, collect,
             seed+iteration*100000, begin)) for begin in range(0, len(tasks), chunk_size)]
        result = [future.result() for future in futures]
        rows = [row for part, _ in result for row in part]
        traces = [trace for _, trace in result if trace is not None]
        trace = ({key: torch.as_tensor(np.concatenate([part[key] for part in traces]))
                  for key in traces[0]} if traces else None)
        return rows, trace

    started = time.monotonic()
    for iteration in range(iterations):
        tasks = [((task_start+iteration*batch_size+i,
                   reference['defenders'], (2., 2.5, 3., 3.5)[i % 4]),
                  int(rng.integers(1, 2**31-1))) for i in range(batch_size)]
        if all(len(record['history']) > iteration for record in records.values()):
            continue
        factual = (parallel_rollout(base_policy, tasks)[0] if executor is not None else
                   r.batched_credit_rows(str(base_policy), tasks, batch_size, device))
        factual_returns = torch.tensor([row['success'] for row in factual], dtype=torch.float32)
        for arm in ('paired', 'ordinary'):
            if len(records[arm]['history']) > iteration:
                continue
            actor = models[arm]; gate = actor.successor_task_value
            gate.sampling = True; gate.collect = True; gate.trajectory = []
            reset_original = TADEnv.reset; packet_original = d.joint_packet
            episode_counter = [0]

            def tracked_reset(environment, *args, **kwargs):
                result = reset_original(environment, *args, **kwargs)
                environment._paired_training_episode = episode_counter[0]
                episode_counter[0] += 1
                return result

            def tracked_packet(environment, observation):
                result = packet_original(environment, observation)
                result['_paired_episode'] = np.array([environment._paired_training_episode], dtype=np.int64)
                return result

            if executor is not None:
                rows, trace = parallel_rollout(outputs[arm], tasks, collect=True, iteration=iteration)
            else:
                TADEnv.reset = tracked_reset; d.joint_packet = tracked_packet
                try:
                    rows = r.batched_credit_rows(str(outputs[arm]), tasks, batch_size, device, actor_override=actor)
                finally:
                    TADEnv.reset = reset_original; d.joint_packet = packet_original
                    gate.sampling = False; gate.collect = False
                if episode_counter[0] != batch_size or not gate.trajectory:
                    raise ValueError('Incomplete rollout indices or missing training decisions.')
                trace = {key: torch.cat([item[key] for item in gate.trajectory]) for key in gate.trajectory[0]}
            if trace is None:
                raise ValueError('A training rollout produced no eligible decisions.')
            gate.trajectory = []
            if trace['episode'].min() < 0 or trace['episode'].max() >= batch_size:
                raise ValueError('A decision lost its complete-trajectory identity.')
            actual = torch.tensor([row['success'] for row in rows], dtype=torch.float32)
            coefficient = 0.
            if arm == 'paired':
                previous = records[arm].get('measured_proposals', [])
                coefficient = initial_control_scale
                if sum(len(item['actual']) for item in previous) >= 64:
                    past_actual = torch.cat([item['actual'] for item in previous])
                    past_factual = torch.cat([item['factual'] for item in previous])
                    centered = past_factual-past_factual.mean()
                    variance = centered.square().mean()
                    coefficient = (float(((past_actual-past_actual.mean())*centered).mean()/variance)
                                   if variance > 1e-8 else 0.)
                    coefficient = max(0., min(1., coefficient))
            reward = actual-coefficient*factual_returns
            if warm_provenance is not None:
                target = actual[trace['episode']]
                advantage = (target-trace['value']-coefficient*
                             (factual_returns[trace['episode']]-trace['value']))
            else:
                target = reward[trace['episode']]
                advantage = target-trace['value']
            # Keep complete-episode policy gradients, including all wait decisions.
            # Subtracting a batch mean here would bias the variable-length policy.
            network = gate.network.cpu(); optimizer = optimizers[arm]
            shuffle.manual_seed(seed+930043000+iteration*10000)
            measured_kl = 0.; loss_value = 0.
            with torch.enable_grad():
                for epoch in range(64):
                    permutation = torch.randperm(len(target), generator=shuffle)
                    for begin in range(0, len(target), 256):
                        indices = permutation[begin:begin+256]
                        logits, value = network(trace['features'][indices])
                        logp = -F.binary_cross_entropy_with_logits(logits, trace['action'][indices], reduction='none')
                        ratio = (logp-trace['log_probability'][indices]).exp()
                        clipped = torch.minimum(ratio*advantage[indices],
                            ratio.clamp(.8, 1.2)*advantage[indices])
                        probability = logits.sigmoid()
                        entropy = F.binary_cross_entropy_with_logits(logits, probability, reduction='none')
                        loss = -clipped.mean()+.5*F.mse_loss(value, target[indices])-.005*entropy.mean()
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        nn.utils.clip_grad_norm_(network.parameters(), .5); optimizer.step()
                        loss_value = float(loss.detach())
                    with torch.no_grad():
                        new_logits, _ = network(trace['features'])
                        new_logp = -F.binary_cross_entropy_with_logits(new_logits, trace['action'], reduction='none')
                        log_ratio = new_logp-trace['log_probability']
                        measured_kl = float((log_ratio.exp()-1.-log_ratio).mean())
                    if measured_kl > .02:
                        break
            record = records[arm]
            record['successor_task_value']['state'] = copy.deepcopy(network.state_dict())
            record['training_tasks'].extend(tasks)
            accepted = trace['action'] > .5
            accepted_episodes = trace['episode'][accepted]
            if len(accepted_episodes.unique()) != len(accepted_episodes):
                raise ValueError('The one-intervention policy accepted multiple proposals in a trajectory.')
            record.setdefault('measured_proposals', []).append(dict(features=trace['features'][accepted],
                actual=actual[accepted_episodes], factual=factual_returns[accepted_episodes],
                tasks=[tasks[index] for index in accepted_episodes.tolist()], iteration=iteration+1))
            metrics = dict(iteration=iteration+1, episodes=batch_size, success=int(actual.sum()),
                factual_success=int(factual_returns.sum()), rescues=int(((actual == 1)&(factual_returns == 0)).sum()),
                harms=int(((actual == 0)&(factual_returns == 1)).sum()), decisions=len(target),
                accepted=int(trace['action'].sum()), loss=loss_value, policy_kl=measured_kl,
                elapsed_seconds=time.monotonic()-started, inference_device=device, rollout_workers=workers,
                optimization_epochs=epoch+1, factual_control_scale=coefficient,
                factual_critic_centering=warm_provenance is not None,
                actual_return_variance=float(actual.var(unbiased=False)),
                full_difference_variance=float((actual-factual_returns).var(unbiased=False)),
                control_variate_variance=float(reward.var(unbiased=False)))
            record['history'].append(metrics); record['optimizer'] = optimizer.state_dict()
            record['completed'] = iteration+1 == iterations
            r.atomic_save(record, outputs[arm]); print(arm, metrics, flush=True)
            actor.cpu()
    if executor is not None:
        executor.shutdown()
    return {arm: str(path) for arm, path in outputs.items()}


def fit_measured_gate(data_paths, output_prefix, *, seed=43, updates=3000, paired_roots_from=None,
                      joint_effect=False):
    """Learn gains from accepted proposals with observed M1 counterfactuals."""
    import io
    import run_remote_evidence as r
    inputs = {}; records = []
    for path in data_paths:
        raw = Path(path).read_bytes(); inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        records.append(torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False))
    if any(not record.get('completed') for record in records):
        raise ValueError('Measured-gain training requires completed immutable policy collections.')
    template = records[0]
    if any(record['base_sha256'] != template['base_sha256']
           or record['proposal_sha256'] != template['proposal_sha256'] for record in records):
        raise ValueError('Observed pairs must use the same M1 and proposal policies.')
    observations = {}; conflicting = 0; excluded_before_history = 0
    for record in records:
        for batch in record.get('measured_proposals', []):
            for index, task in enumerate(batch['tasks']):
                feature = batch['features'][index].contiguous()
                n = template['successor_task_value']['defenders']
                if feature.numel() == 29*n+16 and int(round(float(feature[9*n+4])*400.)) < 5:
                    excluded_before_history += 1
                    continue
                target = (float(batch['actual'][index]), float(batch['factual'][index]))
                key = (tuple(task[0]), task[1], feature.numpy().tobytes())
                previous = observations.get(key)
                if previous is not None and previous[1] != target:
                    conflicting += 1
                    key += (target,)
                observations[key] = (feature, target, tuple(task[0]))
    if len(observations) < 128:
        raise ValueError('At least 128 measured proposals are needed for a separate validation split.')
    samples = list(observations.values())
    on_policy_count = len(samples)
    joint_targets = [[max(b-a, 0.), 1.-abs(a-b), max(a-b, 0.)] for _, (a, b), _ in samples]
    if paired_roots_from is not None:
        raw = Path(paired_roots_from).read_bytes()
        inputs[str(paired_roots_from)] = hashlib.sha256(raw).hexdigest()
        source = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
        allowed = set(map(tuple, source['successor_task_fit']['training_layouts']))
        extractor = OnPolicyGateNetwork(template['successor_task_value']['defenders'], width=32,
                                        representation='compact_physical_only')
        seen_roots = set()
        for path in source['successor_task_fit']['training_data']:
            raw = Path(path).read_bytes(); inputs[str(path)] = hashlib.sha256(raw).hexdigest()
            data = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
            if not data.get('completed') or data.get('predecessor_sha256') != template['base_sha256']:
                raise ValueError('Supplemental measurements must be complete and share the factual policy.')
            if data.get('continuation_context') or data.get('feedback_relative_scale') is not None:
                raise ValueError('The gate requires absolute interventions followed by the common M1 policy.')
            for root in data['roots']:
                if tuple(root['scene']) not in allowed:
                    continue
                n = len(root['order']); options = np.asarray(root['options'])
                if tuple(options[0]) != (0, 0, 0) or root.get('feedback_relative_hold', False):
                    raise ValueError('Supplemental actions must use the same absolute control semantics.')
                context = torch.as_tensor(root_context(root))[None]
                controls = torch.as_tensor(np.asarray(root['effect_controls']), dtype=torch.float32)[None]
                programs = torch.as_tensor(options[1:, :1]//10**np.arange(n) % 10)
                timing = torch.as_tensor(options[1:, 1:])
                count = len(programs)
                feature = extractor.features(context.expand(count, -1), controls.expand(count, -1, -1),
                    programs, timing, torch.zeros(count, 2))
                weights = np.asarray(root.get('future_weights', np.ones(len(root['futures']))), float)
                if np.any(weights < 0) or weights.sum() <= 0:
                    raise ValueError('Supplemental futures need valid recorded weights.')
                weights = weights/weights.sum()
                for action in range(1, len(options)):
                    key = (tuple(root['scene']), root['prefix_future'], root['root_steps'],
                           tuple(options[action]), tuple(root['futures']))
                    if key in seen_roots:
                        continue
                    seen_roots.add(key)
                    if any((a, f) not in root['outcomes'] for a in (0, action) for f in range(len(weights))):
                        raise ValueError('Every supplemental contrast needs its measured common futures.')
                    target = tuple(sum(weights[f]*root['outcomes'][a, f]['success']
                                       for f in range(len(weights))) for a in (action, 0))
                    samples.append((feature[action-1], target, tuple(root['scene'])))
                    differences = np.asarray([root['outcomes'][action, f]['success']-
                                              root['outcomes'][0, f]['success'] for f in range(len(weights))])
                    joint_targets.append([float(weights[differences == value].sum()) for value in (-1., 0., 1.)])
    features = torch.stack([sample[0] for sample in samples])
    history_conditioned = features.shape[-1] == 29*template['successor_task_value']['defenders']+16
    targets = torch.tensor([sample[1] for sample in samples], dtype=torch.float32)
    joint_targets = torch.tensor(joint_targets, dtype=torch.float32)
    layouts = sorted({sample[2] for sample in samples})
    rng = np.random.default_rng(seed)
    validation_layouts = set()
    for agility in sorted({layout[-1] for layout in layouts}):
        group = [layout for layout in layouts if layout[-1] == agility]
        validation_layouts.update(group[index] for index in rng.permutation(len(group))[:max(1, len(group)//5)])
    validation = torch.tensor([sample[2] in validation_layouts for sample in samples])
    training_indices = torch.where(~validation)[0]
    validation_indices = torch.where(validation)[0]
    on_policy = torch.arange(len(samples)) < on_policy_count
    training_weights = torch.ones(len(training_indices))
    validation_weights = torch.ones(len(validation_indices))
    if paired_roots_from is not None:
        for indices, weights in ((training_indices, training_weights), (validation_indices, validation_weights)):
            current = on_policy[indices]
            if not bool(current.any()) or not bool((~current).any()):
                raise ValueError('Both splits must contain on-policy and supplemental evidence.')
            weights[current] = .5/current.sum()
            weights[~current] = .5/(~current).sum()
    validation_weights /= validation_weights.sum()
    outputs = {}
    for paired in (True, False):
        arm = 'paired' if paired else 'ordinary'
        output = Path(f'{output_prefix}-{arm}-seed{seed}.pth')
        if output.exists():
            raise ValueError('Measured-gain output already exists.')
        torch.manual_seed(seed)
        representation = ('compact_history' if history_conditioned else
                          'compact_physical_only' if paired_roots_from is not None else 'compact_physical')
        network = OnPolicyGateNetwork(template['successor_task_value']['defenders'], width=32,
                                      representation=representation)
        generator = torch.Generator().manual_seed(seed+811)
        optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4, weight_decay=.01)
        best = None; best_score = None; history = []
        with torch.enable_grad():
            for update in range(1, updates+1):
                chosen = (torch.multinomial(training_weights, 128, replacement=True, generator=generator)
                          if paired_roots_from is not None else
                          torch.randint(len(training_indices), (128,), generator=generator))
                index = training_indices[chosen]
                gain, factual = network(features[index]); intervention = factual+gain
                if paired and joint_effect:
                    log_probability = torch.stack((-gain, factual, gain), -1).log_softmax(-1)
                    loss = -(joint_targets[index]*log_probability*log_probability.new_tensor([1., .2, 1.])).sum(-1).mean()
                else:
                    loss = (F.binary_cross_entropy_with_logits(factual, targets[index, 1])+
                            F.binary_cross_entropy_with_logits(intervention, targets[index, 0]))
                    if paired:
                        loss = loss+4.*F.mse_loss(intervention.sigmoid()-factual.sigmoid(),
                                                 targets[index, 0]-targets[index, 1])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 1.); optimizer.step()
                if update % 250 == 0 or update == updates:
                    with torch.no_grad():
                        gain, factual = network(features[validation_indices])
                        predicted = (measured_joint_gain(gain, factual) if paired and joint_effect
                                     else (factual+gain).sigmoid()-factual.sigmoid())
                        actual = targets[validation_indices, 0]-targets[validation_indices, 1]
                        accepted = predicted > .02
                        measured_gain = float((torch.where(accepted, actual, torch.zeros_like(actual))*validation_weights).sum())
                        mse = float(((predicted-actual).square()*validation_weights).sum())
                    metrics = dict(update=update, validation_measured_gain=measured_gain,
                        validation_effect_mse=mse, accepted=int(accepted.sum()),
                        rescues=int((accepted & (actual > 0)).sum()), harms=int((accepted & (actual < 0)).sum()))
                    history.append(metrics)
                    score = (measured_gain, -mse)
                    if best_score is None or score > best_score:
                        best_score = score; best = copy.deepcopy(network.state_dict()); selected = update
        checkpoint = {key: copy.deepcopy(template[key]) for key in (
            'format', 'base_policy', 'base_sha256', 'proposal_policy', 'proposal_sha256', 'runtime_sources')}
        checkpoint['format'] = 'paired-successor-v1'
        recipe = copy.deepcopy(template['successor_task_value'])
        recipe.update(state=best, score_mode='joint_effect' if paired and joint_effect else 'measured_gain', gate_margin=.02,
                      width=32, representation=representation, stochastic_execution=False,
                      latest_proposal_tick=int((features[training_indices, 9*recipe['defenders']+4]*400.).round().max()))
        checkpoint.update(successor_task_value=recipe, completed=True, development_only=True,
            input_sha256=inputs, measured_gain_fit=dict(paired=paired, seed=seed, updates=updates,
                selected_update=selected, observations=len(samples), conflicting_repeats=conflicting,
                on_policy_observations=on_policy_count, supplemental_observations=len(samples)-on_policy_count,
                excluded_before_observed_history=excluded_before_history,
                objective='balanced_joint_effect' if paired and joint_effect else 'paired_mse' if paired else 'absolute_returns',
                training_layouts=sorted(set(layouts)-validation_layouts), validation_layouts=sorted(validation_layouts),
                training_observations=len(training_indices), validation_observations=len(validation_indices),
                history=history, parameters=sum(parameter.numel() for parameter in network.parameters())))
        checkpoint['runtime_sources']['paired_successor_value.py'] = Path(__file__).read_text(encoding='utf-8-sig')
        r.atomic_save(checkpoint, output); outputs[arm] = str(output)
        print(arm, {'observations': len(samples), 'selected_update': selected,
                    'validation_gain': best_score[0], 'validation_mse': -best_score[1]}, flush=True)
    return outputs


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--data', nargs='+', required=True)
    fit.add_argument('--validation', required=True)
    fit.add_argument('--split-model', required=True)
    fit.add_argument('--base-policy', required=True)
    fit.add_argument('--paired', action='store_true')
    fit.add_argument('--seed', type=int, default=43)
    fit.add_argument('--updates', type=int, default=2000)
    fit.add_argument('--members', type=int, default=1)
    fit.add_argument('--value-form', choices=('potential', 'physical_bottleneck', 'planned_successor', 'constrained_planned', 'local_quadratic', 'event_time', 'paired_outcome'), default='potential')
    fit.add_argument('--physical-weight', type=float, default=.1)
    fit.add_argument('--effect-weight', type=float, default=4.)
    fit.add_argument('--response-difference-weight', type=float, default=.1)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--model', required=True)
    evaluate.add_argument('--tasks-from', required=True)
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--batch-size', type=int, default=32)
    evaluate.add_argument('--workers', type=int, default=1)
    gate = commands.add_parser('fit-gate')
    gate.add_argument('--base-policy', required=True)
    gate.add_argument('--proposal-model', required=True)
    gate.add_argument('--iterations', type=int, default=16)
    gate.add_argument('--batch-size', type=int, default=64)
    gate.add_argument('--workers', type=int, default=2)
    gate.add_argument('--seed', type=int, default=43)
    gate.add_argument('--device', default='cuda:0')
    gate.add_argument('--warm-start')
    gate.add_argument('--representation', choices=('dense', 'dense_history'), default='dense')
    gate.add_argument('--layout-start', type=int)
    measured = commands.add_parser('fit-measured-gate')
    measured.add_argument('--data', nargs='+', required=True)
    measured.add_argument('--seed', type=int, default=43)
    measured.add_argument('--updates', type=int, default=3000)
    measured.add_argument('--paired-roots-from')
    measured.add_argument('--joint-effect', action='store_true')
    history = commands.add_parser('prepare-history')
    history.add_argument('--data', nargs='+', required=True)
    history.add_argument('--workers', type=int, default=2)
    history.add_argument('--device', default='cuda:0')
    for command in (fit, evaluate, gate, measured, history):
        command.add_argument('--output', required=True)
        command.add_argument('--runtime-from')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Choose a new experiment output; completed evidence is never overwritten.')
    read = lambda path: torch.load(path, map_location='cpu', weights_only=False)
    runtime_from = args.runtime_from or (args.model if args.command == 'evaluate' else None)
    if runtime_from is not None:
        restore_runtime(read(runtime_from).get('runtime_sources', {}))
    import run_remote_evidence as r
    r.initialize(); install_runtime(); torch.set_num_threads(1)
    if args.command == 'prepare-history':
        prepare_observed_history(args.data, args.output, workers=args.workers, device=args.device)
    elif args.command == 'fit-measured-gate':
        fit_measured_gate(args.data, args.output, seed=args.seed, updates=args.updates,
                          paired_roots_from=args.paired_roots_from, joint_effect=args.joint_effect)
    elif args.command == 'fit-gate':
        if runtime_from is None:
            raise ValueError('On-policy training requires a frozen execution runtime.')
        train_on_policy_gate(args.base_policy, args.proposal_model, args.output,
            read(runtime_from)['runtime_sources'], iterations=args.iterations,
            batch_size=args.batch_size, seed=args.seed, device=args.device, workers=args.workers,
            warm_start=args.warm_start, representation=args.representation, layout_start=args.layout_start)
    elif args.command == 'fit':
        split = read(args.split_model)
        result = fit_successor([read(path) for path in args.data], read(args.validation), split,
            paired=args.paired, seed=args.seed, updates=args.updates, members=args.members,
            physical_weight=args.physical_weight, effect_weight=args.effect_weight,
            response_difference_weight=args.response_difference_weight, value_form=args.value_form)
        names = ('develop_source_fusion.py', 'conditional_fusion_value.py', 'local_intervention_value.py',
                 'plain_return.py', 'mechanism_controls.py', 'run_remote_evidence.py', 'paired_successor_value.py')
        sources = (read(runtime_from)['runtime_sources'] if runtime_from else
                   {name: (Path(__file__).parent/name).read_text(encoding='utf-8-sig') for name in names})
        sources = dict(sources, **{'paired_successor_value.py': Path(__file__).read_text(encoding='utf-8-sig')})
        checkpoint = make_checkpoint(result, args.base_policy, split['successor_task_value']['options'],
                                     runtime_sources=sources)
        checkpoint['input_sha256'] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [*args.data, args.validation, args.split_model]}
        r.atomic_save(checkpoint, output)
    else:
        tasks = read(args.tasks_from)['tasks']; rows = []; started = time.monotonic()
        result = dict(policy=args.model, policy_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
            tasks=tasks, rows=rows, completed=False, development_only=True,
            runtime_sha256={name: hashlib.sha256(source.encode()).hexdigest()
                for name, source in dict(read(runtime_from).get('runtime_sources', {}),
                    **{'paired_successor_value.py': Path(__file__).read_text(encoding='utf-8-sig')}).items()},
            inference_device=args.device, batch_size=args.batch_size, rollout_workers=args.workers)
        executor = None; futures = None
        if args.workers > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
            sources = read(runtime_from)['runtime_sources']
            futures = [executor.submit(_gate_rollout_worker, (args.model,
                tasks[begin:begin+args.batch_size], sources, args.device, False, False, 0, begin))
                for begin in range(0, len(tasks), args.batch_size)]
        for number, begin in enumerate(range(0, len(tasks), args.batch_size)):
            batch = (futures[number].result()[0] if futures is not None else
                     r.batched_credit_rows(args.model, tasks[begin:begin+args.batch_size],
                                          args.batch_size, args.device))
            rows.extend(batch)
            result.update(evaluated=len(rows), success=sum(row['success'] for row in rows),
                          elapsed_seconds=time.monotonic()-started, completed=len(rows) == len(tasks))
            r.atomic_save(result, output)
            print({k: result[k] for k in ('evaluated', 'success', 'elapsed_seconds', 'completed')}, flush=True)
        if executor is not None:
            executor.shutdown()


if __name__ == '__main__':
    main()
