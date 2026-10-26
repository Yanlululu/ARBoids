"""Independent absolute-return regression over the common action catalog."""
import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def root_sampling_probabilities(root_ids, fraction=0., root_weights=None, observation_weights=None):
    """Mix observation-uniform sampling with positive mass per measured state.

    Optional weights follow sorted unique root ids. No state can be removed.
    """
    if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError('Root sampling fraction must lie between zero and one.')
    root_ids = np.asarray(root_ids)
    if root_ids.ndim != 1 or not len(root_ids):
        raise ValueError('Root sampling requires a nonempty one-dimensional population.')
    if fraction == 0. and root_weights is None and observation_weights is None:
        return None
    _, inverse, counts = np.unique(root_ids, return_inverse=True, return_counts=True)
    if root_weights is None:
        probability = (1-fraction)/len(root_ids)+fraction/(len(counts)*counts[inverse])
    else:
        prior = np.asarray(root_weights, np.float64)
        if (fraction == 0. or prior.shape != counts.shape or
                not np.isfinite(prior).all() or np.any(prior <= 0.)):
            raise ValueError('Root weights need positive finite mass for every state and a positive sampling fraction.')
        prior = prior/prior.max()
        prior = prior/prior.sum()
        probability = (1-fraction)/len(root_ids)+fraction*prior[inverse]/counts[inverse]
    if observation_weights is not None:
        observed = np.asarray(observation_weights, np.float64)
        if (observed.shape != root_ids.shape or not np.isfinite(observed).all()
                or np.any(observed <= 0.)):
            raise ValueError('Observation weights need finite positive mass for every measured outcome.')
        # Correct the sampled future distribution within each state while
        # preserving exactly the previously specified mass of that state.
        observed = observed/observed.max()
        means = np.bincount(inverse, weights=observed)/counts
        probability *= observed/means[inverse]
    return probability/probability.sum()


def require_plain_predecessor(model, *, allow_plain_return=False, allow_paired_return=False):
    learned_keys = ('local_conditional_value', 'physical_conditional_value',
        'physical_conditional_values', 'joint_intervention_value', 'joint_intervention_values',
        'conditional_effect_support', 'response_base', 'initial_role_credit',
        'feedback_role_credit', 'cooperative_role_credit', 'finite_temporal_credit',
        'continuous_program_value', 'response_feedback_value', 'midpoint_intervention_values',
        'shared_response_value', 'factual_return_support', 'response_action_support',
        'program_proposer')
    inherited = [key for key in learned_keys if model.get(key) is not None]
    value = model.get('successor_task_value')
    permitted = ((allow_plain_return and value is not None and value.get('kind') == 'plain_mlp_return')
                 or (allow_paired_return and value is not None and value.get('kind') == 'paired_mlp_return'))
    if value is not None and not permitted:
        inherited.append('successor_task_value')
    if inherited:
        raise ValueError('The plain return controller cannot inherit learned decision modules: ' +
                         ', '.join(inherited))


def recurrent_single_vessel_specification(value):
    """Require one complete, nonoverlapping relative-action decision period."""
    get = value.get if isinstance(value, dict) else lambda key, default=None: getattr(value, key, default)
    options = get('options')
    if torch.is_tensor(options):
        options = options.detach().cpu().numpy()
    options = np.asarray(options)
    defenders = get('defenders')
    if defenders is None:
        defenders = getattr(get('network'), 'defenders', None)
    scale = get('feedback_relative_scale')
    if (get('intervention_budget') != 0 or get('decision_period') != 5
            or not get('isolate_factual_start') or not get('feedback_relative_hold')
            or scale is None or not np.isfinite(scale) or not 0. < scale <= 1.
            or get('wait_value') is not None or get('prior') is not None
            or not isinstance(defenders, int) or defenders < 1
            or options.shape != (8*defenders+1, 3)
            or not np.isfinite(options).all() or not np.equal(options, np.rint(options)).all()):
        raise ValueError('Recurrent sampling requires isolated relative single-vessel decisions every five ticks.')
    expected = {(0, 0, 0)} | {(digit*10**boat, 0, 5)
        for boat in range(defenders) for digit in (1, 2, 3, 4, 6, 7, 8, 9)}
    if tuple(options[0]) != (0, 0, 0) or set(map(tuple, options)) != expected:
        raise ValueError('Recurrent sampling needs the complete one-window single-vessel action catalog.')
    specification = dict(decision_period=5, feedback_relative_scale=float(scale), defenders=defenders)
    reference = get('policy_reference')
    if reference is not None and recurrent_single_vessel_specification(reference) != specification:
        raise ValueError('A recurrent policy reference must retain the same fleet, window and relative scale.')
    return specification


def standalone_single_vessel_windows(value):
    """Retain every measured duration when sharing a fleet value across sizes."""
    options = value.options.detach().cpu().numpy()
    windows = sorted({tuple(option[1:]) for option in options[1:]})
    expected = {(0, 0, 0)} | {(digit*10**boat, delay, span)
        for boat in range(value.network.defenders) for digit in (1, 2, 3, 4, 6, 7, 8, 9)
        for delay, span in windows}
    if (not value.isolate_factual_start or not value.feedback_relative_hold
            or value.decision_period != 5 or value.wait_value is not None
            or value.policy_reference is not None or not windows
            or any(delay < 0 or span <= 0 or delay+span > 400 or delay % 5 or span % 5
                   for delay, span in windows)
            or tuple(options[0]) != (0, 0, 0)
            or len(options) != len(expected) or set(map(tuple, options)) != expected):
        raise ValueError('Shared fleet values need complete standalone single-vessel program windows.')
    return tuple(windows)


def single_coordinate_catalog(defenders):
    """Union of one-coordinate neighbors of every single-vessel plan."""
    digits = (1, 2, 3, 4, 6, 7, 8, 9)
    codes = {digit*10**boat for boat in range(defenders) for digit in digits}
    codes.update(a*10**left+b*10**right for left in range(defenders)
                 for right in range(left+1, defenders) for a in digits for b in digits)
    return [(0, 0, 0)]+[(code, 0, 5) for code in sorted(codes)]


def recurrent_decision_specification(value):
    """Validate either the original star or a frozen-policy local neighborhood."""
    get = value.get if isinstance(value, dict) else lambda key, default=None: getattr(value, key, default)
    if not get('policy_neighborhood', False):
        return recurrent_single_vessel_specification(value)
    reference = get('policy_reference')
    if reference is None:
        raise ValueError('Local policy neighborhoods require their actual frozen reference.')
    specification = recurrent_single_vessel_specification(reference)
    defenders = get('defenders', getattr(get('network'), 'defenders', None))
    options = get('options')
    if torch.is_tensor(options):
        options = options.detach().cpu().numpy()
    if (defenders != specification['defenders'] or get('intervention_budget') != 0
            or get('decision_period') != 5 or not get('isolate_factual_start')
            or not get('feedback_relative_hold')
            or get('feedback_relative_scale') != specification['feedback_relative_scale']
            or get('wait_value') is not None or get('prior') is not None
            or not np.array_equal(np.asarray(options), np.asarray(single_coordinate_catalog(defenders)))):
        raise ValueError('Local policy neighborhoods require their complete one-coordinate action union and common clock.')
    return dict(specification, policy_neighborhood=True)


def validate_recurrent_predecessor(model, base):
    """Keep the base controller explicit while freezing a learned continuation."""
    require_plain_predecessor(base)
    value = model.get('successor_task_value') or {}
    if value.get('kind') not in ('plain_mlp_return', 'paired_mlp_return'):
        raise ValueError('Policy iteration requires a native dense continuation policy.')
    require_plain_predecessor(model, allow_plain_return=True, allow_paired_return=True)
    specification = recurrent_decision_specification(value)

    def equal(left, right):
        if torch.is_tensor(left) or torch.is_tensor(right):
            return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
        if isinstance(left, (tuple, list)):
            return isinstance(right, (tuple, list)) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
        return left == right

    metadata = {'successor_task_value', 'successor_task_fit', 'learning_snapshots', 'core_algorithm_context', 'execution_revision',
        'physical_interventions', 'fixed_intervention_clock', 'success_objective', 'mechanism_ablation', 'gate_period'}
    if model.get('gate_period', 1) != base.get('gate_period', 1):
        raise ValueError('A recurrent continuation changed its frozen base controller: gate_period')
    if model.keys()-base.keys()-metadata:
        raise ValueError('A recurrent continuation added unrecognized base-controller settings.')
    for key in base.keys()-metadata:
        if key not in model or not equal(base[key], model[key]):
            raise ValueError('A recurrent continuation changed its frozen base controller: ' + key)
    before = tuple(base.get('mechanism_ablation', (False, False, False)))
    after = tuple(model.get('mechanism_ablation', (False, False, False)))
    if len(before) != 3 or len(after) != 3 or before[0::2] != after[0::2] or after[1]:
        raise ValueError('Policy iteration must retain the base candidate interactions.')
    return specification


class PlainReturnNetwork(nn.Module):
    def __init__(self, defenders, width, identity_conditioned=False, output_width=1,
                 action_encoding='categorical', feedback_relative_scale=None, state_baseline=False,
                 representation='dense', peer_decision_conditioned=True):
        super().__init__()
        if action_encoding not in ('categorical', 'physical_delta'):
            raise ValueError('Unknown dense intervention action representation.')
        if representation not in ('dense', 'fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic'):
            raise ValueError('Unknown intervention state/action representation.')
        if representation in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic') and (not state_baseline or representation != 'fleet_graph_quadratic' and width % 4):
            raise ValueError('Fleet attention requires a state baseline and a width divisible by four.')
        self.representation = representation
        self.defenders = defenders
        self.width = width
        self.identity_conditioned = identity_conditioned
        self.action_encoding = action_encoding
        if state_baseline and (action_encoding != 'physical_delta' or output_width != 1):
            raise ValueError('A separate state baseline requires scalar physical-action values.')
        self.state_baseline = bool(state_baseline)
        if not peer_decision_conditioned and not state_baseline:
            raise ValueError('Removing peer decisions requires a shared state baseline and physical actions.')
        self.peer_decision_conditioned = bool(peer_decision_conditioned)
        if feedback_relative_scale is not None and (action_encoding != 'physical_delta' or
                not np.isfinite(feedback_relative_scale) or not 0. < feedback_relative_scale <= 1.):
            raise ValueError('Relative dense features need a finite scale in (0,1] and physical commands.')
        self.feedback_relative_scale = feedback_relative_scale
        self.state_size = (14 + identity_conditioned) * defenders + 5
        self.action_size = 10*defenders+2 if action_encoding == 'categorical' else 5*defenders+4
        if representation in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic'):
            # All vessels share the same state/action maps. The pooled output
            # is a value of the complete program, without additive boat values.
            self.state_embed = nn.Sequential(nn.Linear(14+identity_conditioned, width), nn.SiLU(),
                nn.Linear(width, width), nn.SiLU())
            if representation == 'fleet_graph_quadratic':
                self.state_message = nn.Sequential(nn.Linear(2*width+8, width), nn.SiLU(),
                    nn.Linear(width, width), nn.SiLU())
                self.state_update = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, width))
                self.state_norm = nn.LayerNorm(width)
            else:
                self.state_attention = nn.TransformerEncoderLayer(width, 4, 2*width, dropout=0.,
                    activation='gelu', batch_first=True, norm_first=True)
            if representation in ('fleet_quadratic', 'fleet_graph_quadratic'):
                self.baseline = nn.Sequential(nn.Linear(2*width+6, width), nn.ReLU(), nn.Linear(width, 1))
                self.layers = nn.Sequential(nn.Linear(3*width+10, width), nn.SiLU(), nn.Linear(width, 14))
                # Conditional coefficients see the whole fleet and duration.
                # Restrict local action interpolation to a quadratic response
                # while retaining interactions in the shared state encoder.
                rows, columns = torch.triu_indices(4, 4)
                self.register_buffer('quadratic_rows', rows, persistent=False)
                self.register_buffer('quadratic_columns', columns, persistent=False)
                nn.init.zeros_(self.layers[-1].weight)
                nn.init.zeros_(self.layers[-1].bias)
            else:
                self.action_embed = nn.Linear(5, width)
                self.action_attention = nn.TransformerEncoderLayer(width, 4, 2*width, dropout=0.,
                    activation='gelu', batch_first=True, norm_first=True)
                self.baseline = nn.Sequential(nn.Linear(2*width+6, width), nn.ReLU(), nn.Linear(width, 1))
                self.layers = nn.Sequential(nn.Linear(2*width+10, width+8), nn.ReLU(), nn.Linear(width+8, 1))
            return
        self.layers = nn.Sequential(
            nn.Linear(self.state_size+self.action_size, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, output_width))
        if self.state_baseline:
            self.baseline = nn.Sequential(
                nn.Linear(self.state_size, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward_features(self, features):
        if not self.peer_decision_conditioned:
            shape = features.shape[:-1]
            flat = features.reshape(-1, features.shape[-1])
            value = self.subject_conditioned_values(flat[:, :self.state_size], flat[:, self.state_size:])
            return value.reshape(*shape, 1)
        if self.representation in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic'):
            shape = features.shape[:-1]
            flat = features.reshape(-1, features.shape[-1])
            state, shared, baseline = self.encode_fleet_state(flat[:, :self.state_size])
            value = self.fleet_action_value(state, shared, baseline, flat[:, self.state_size:])
            return value.reshape(*shape, 1)
        value = self.layers(features)
        if self.state_baseline:
            action = features[..., self.state_size:self.state_size+5*self.defenders]
            active = action.reshape(*action.shape[:-1], self.defenders, 5)[..., 0].bool().any(-1)
            value = self.baseline(features[..., :self.state_size])+active[..., None]*value
        return value

    def subject_conditioned_values(self, state, actions, baseline=None):
        """Retain the intervention subject's commands and remove peer decisions.

        The common factual baseline sees all public states, without command
        conditioning. Each action residual retains the changed vessel's
        source, rule and factual commands as well as its physical action.
        """
        n = self.defenders
        context_size = (8+self.identity_conditioned)*n+5
        active = actions[:, :5*n].reshape(-1, n, 5)[..., :1].bool()
        commands = state[:, context_size:].reshape(-1, n, 6)
        subject_state = torch.cat((state[:, :context_size],
                                   (commands*active).flatten(1)), -1)
        if baseline is None:
            public_state = torch.cat((state[:, :context_size], torch.zeros_like(state[:, context_size:])), -1)
            baseline = (self.encode_fleet_state(public_state)[2]
                        if self.representation in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic') else self.baseline(public_state))
        if self.representation in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic'):
            encoded, shared, _ = self.encode_fleet_state(subject_state)
            return self.fleet_action_value(encoded, shared, baseline, actions)
        residual = self.layers(torch.cat((subject_state, actions), -1))
        return baseline+active.any(1)*residual

    def configure_fleet(self, defenders):
        if defenders == self.defenders:
            return
        if self.representation not in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic') or not isinstance(defenders, int) or defenders < 1:
            raise ValueError('The plain MLP and deployed fleet must have the same size; fleet attention can transfer.')
        self.defenders = defenders
        self.state_size = (14+self.identity_conditioned)*defenders+5
        self.action_size = 5*defenders+4

    @staticmethod
    def fleet_pair_features(rows):
        displacement = rows[:, :, None, :2]-rows[:, None, :, :2]
        velocity = rows[:, :, None, 2:4]-rows[:, None, :, 2:4]
        separation = displacement.norm(dim=-1, keepdim=True)
        radial_speed = (displacement*velocity).sum(-1, keepdim=True)/separation.clamp_min(1e-6)
        heading = F.normalize(rows[..., 4:6], dim=-1)
        cosine = (heading[:, :, None]*heading[:, None]).sum(-1, keepdim=True)
        sine = (heading[:, :, None, 0]*heading[:, None, :, 1]
                -heading[:, :, None, 1]*heading[:, None, :, 0])[..., None]
        return torch.cat((displacement, velocity, 3.*separation, radial_speed, cosine, sine), -1)

    def encode_fleet_state(self, state):
        n = self.defenders
        context_size = (8+self.identity_conditioned)*n+5
        local = [state[:, :8*n].reshape(-1, n, 8)]
        if self.identity_conditioned:
            local.append(state[:, 8*n:9*n, None])
        local.append(state[:, context_size:].reshape(-1, n, 6))
        encoded = self.state_embed(torch.cat(local, -1))
        if self.representation == 'fleet_graph_quadratic':
            # Explicit public relative motion lets sparse task outcomes train
            # fleet interactions without inferring pair geometry from scratch.
            geometry = self.fleet_pair_features(local[0])
            off_diagonal = (~torch.eye(n, dtype=torch.bool, device=state.device))[None, :, :, None]
            for _ in range(2):
                pairs = torch.cat((encoded[:, :, None].expand(-1, -1, n, -1),
                    encoded[:, None].expand(-1, n, -1, -1), geometry), -1)
                messages = (self.state_message(pairs)*off_diagonal).sum(2)/max(n-1, 1)
                encoded = self.state_norm(encoded+self.state_update(torch.cat((encoded, messages), -1)))
        else:
            encoded = self.state_attention(encoded)
        shared = torch.cat((state[:, context_size-5:context_size],
                            state.new_full((len(state), 1), n/6.)), -1)
        baseline = self.baseline(torch.cat((encoded.mean(1), encoded.amax(1), shared), -1))
        return encoded, shared, baseline

    def fleet_action_value(self, state, shared, baseline, actions):
        local = actions[:, :5*self.defenders].reshape(-1, self.defenders, 5)
        if self.representation in ('fleet_quadratic', 'fleet_graph_quadratic'):
            offsets = local[..., 1:]
            basis = torch.cat((offsets, offsets[..., self.quadratic_rows]*
                               offsets[..., self.quadratic_columns]), -1)
            pooled = torch.cat((state.mean(1), state.amax(1), shared, actions[:, -4:]), -1)
            coefficients = self.layers(torch.cat((state, pooled[:, None].expand(-1, self.defenders, -1)), -1))
            residual = (coefficients*basis).sum((-1, -2))
            return baseline+residual[:, None]
        encoded = self.action_attention(state+self.action_embed(local))
        residual = self.layers(torch.cat((encoded.mean(1), encoded.amax(1), shared, actions[:, -4:]), -1))
        active = local[..., 0].bool().any(-1)
        return baseline+active[:, None]*residual

    def action_features(self, programs, timing, controls=None):
        if self.action_encoding == 'categorical':
            return torch.cat((F.one_hot(programs, 10).float().flatten(-2),
                              timing.float() / 400.), -1)
        if controls is None or controls.shape[-2:] != (self.defenders, 6):
            raise ValueError('Physical intervention features require current source, rule and factual commands.')
        active = self.active_vessels(programs)
        coefficients = controls.new_tensor([[0., 0.]]+
            [[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])
        preference = coefficients[programs]
        source, rule, factual = controls.split(2, -1)
        delta = torch.where(active[..., None],
            preference*source+(1.-preference)*rule-factual, torch.zeros_like(factual))
        if self.feedback_relative_scale is not None:
            preference = self.feedback_relative_scale*(2.*preference-1.)
            command = torch.maximum(torch.minimum(factual+preference*(source-rule),
                torch.maximum(source, rule)), torch.minimum(source, rule))
            delta = torch.where(active[..., None], command-factual, torch.zeros_like(factual))
        # The coefficient remains visible when two current commands coincide:
        # a finite hold can still separate their later closed-loop commands.
        local = torch.cat((active[..., None].expand_as(delta[..., :1]),
            (preference*active[..., None]).expand_as(delta), .25*delta), -1).flatten(-2)
        temporal = torch.cat((timing.float()/400., timing.float().log1p()/np.log(401.)), -1)
        if self.feedback_relative_scale is not None:
            temporal = temporal*active.any(-1, keepdim=True)
        return torch.cat((local, temporal.expand(*local.shape[:-1], 4)), -1)

    def active_vessels(self, programs):
        active = programs > 0
        return active if self.feedback_relative_scale is None else active & (programs != 5)

    def forward(self, context, controls, programs, timing):
        features = torch.cat((context, controls.flatten(1),
                              self.action_features(programs, timing, controls)), -1)
        return self.forward_features(features).squeeze(-1)

    def forward_catalog(self, context, controls, programs, timing, chunk_size=1024):
        if not self.peer_decision_conditioned:
            state = torch.cat((context, controls.flatten(1)), -1)
            public_state = torch.cat((context, torch.zeros_like(controls.flatten(1))), -1)
            baseline = (self.encode_fleet_state(public_state)[2]
                        if self.representation in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic') else self.baseline(public_state))
            output = []
            for start in range(0, len(programs), chunk_size):
                actions = self.action_features(programs[None, start:start+chunk_size],
                    timing[None, start:start+chunk_size], controls[:, None])
                count = actions.shape[1]
                value = self.subject_conditioned_values(
                    state[:, None].expand(-1, count, -1).flatten(0, 1), actions.flatten(0, 1),
                    baseline[:, None].expand(-1, count, -1).flatten(0, 1))
                output.append(value.reshape(len(context), count))
            return torch.cat(output, 1)
        if self.representation in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic'):
            state, shared, baseline = self.encode_fleet_state(torch.cat((context, controls.flatten(1)), -1))
            output = []
            for start in range(0, len(programs), chunk_size):
                actions = self.action_features(programs[None, start:start+chunk_size],
                    timing[None, start:start+chunk_size], controls[:, None])
                count = actions.shape[1]
                value = self.fleet_action_value(
                    state[:, None].expand(-1, count, -1, -1).flatten(0, 1),
                    shared[:, None].expand(-1, count, -1).flatten(0, 1),
                    baseline[:, None].expand(-1, count, -1).flatten(0, 1), actions.flatten(0, 1))
                output.append(value.reshape(len(context), count))
            return torch.cat(output, 1)
        # Factor the first affine layer only; this equals ordinary dense MLP
        # evaluation of every state/action pair without duplicating its input.
        first = self.layers[0]
        state = torch.cat((context, controls.flatten(1)), -1)
        state_hidden = F.linear(state, first.weight[:, :self.state_size], first.bias)
        baseline = self.baseline(state)[:, None] if self.state_baseline else None
        output = []
        for start in range(0, len(programs), chunk_size):
            if self.action_encoding == 'physical_delta':
                actions = self.action_features(programs[None, start:start+chunk_size],
                    timing[None, start:start+chunk_size], controls[:, None])
            else:
                actions = self.action_features(programs[start:start+chunk_size],
                                               timing[start:start+chunk_size])
            action_hidden = F.linear(actions, first.weight[:, self.state_size:])
            value = state_hidden[:, None] + (action_hidden if action_hidden.ndim == 3 else action_hidden[None])
            for layer in list(self.layers)[1:]:
                value = layer(value)
            if baseline is not None:
                active = self.active_vessels(programs[start:start+chunk_size]).any(-1)
                value = baseline+active[None, :, None]*value
            output.append(value.squeeze(-1))
        return torch.cat(output, 1)


class PairedReturnNetwork(PlainReturnNetwork):
    """The common dense state/action trunk with factual and signed-event outputs."""
    def __init__(self, defenders, width, identity_conditioned=False, action_encoding='categorical',
                 feedback_relative_scale=None, state_baseline=False, representation='dense',
                 peer_decision_conditioned=True):
        super().__init__(defenders, width, identity_conditioned, output_width=3,
                         action_encoding=action_encoding, feedback_relative_scale=feedback_relative_scale,
                         state_baseline=state_baseline, representation=representation,
                         peer_decision_conditioned=peer_decision_conditioned)

    @staticmethod
    def event_probabilities(raw, hierarchy=False):
        if hierarchy:
            change = raw[..., 1].sigmoid()
            rescue_given_change = raw[..., 2].sigmoid()
            return torch.stack((1.-change, change*rescue_given_change,
                                change*(1.-rescue_given_change)), -1)
        return torch.cat((torch.zeros_like(raw[..., :1]), raw[..., 1:]), -1).softmax(-1)

    @staticmethod
    def joint_outcome_loss(factual_logits, raw, factual_target, outcome_target, active):
        factual_loss = F.binary_cross_entropy_with_logits(factual_logits, factual_target, reduction='none')
        rescue_loss = F.binary_cross_entropy_with_logits(raw[..., 1], outcome_target, reduction='none')
        harm_loss = F.binary_cross_entropy_with_logits(raw[..., 2], 1.-outcome_target, reduction='none')
        conditional_loss = torch.where(factual_target > .5, harm_loss, rescue_loss)
        return (factual_loss+active*conditional_loss).mean()

    @classmethod
    def signed_gain(cls, raw, active, factual_probability=None, *, event_hierarchy=False):
        if event_hierarchy and factual_probability is not None:
            raise ValueError('Change-direction effects do not condition on factual outcome strata.')
        if factual_probability is not None:
            rescue, harm = raw[..., 1:].sigmoid().unbind(-1)
            return ((1.-factual_probability)*rescue-factual_probability*harm)*active.to(raw.dtype)
        probabilities = cls.event_probabilities(raw, event_hierarchy)
        return (probabilities[..., 1]-probabilities[..., 2])*active.to(raw.dtype)


class PlainReturnDecision(nn.Module):
    """Mean dense values and argmax, optionally requiring a fixed predicted gain."""
    network_type = PlainReturnNetwork

    def __init__(self, defenders, width, state, options, decision_period=5,
                 inference_device=None, intervention_budget=0, identity_conditioned=False,
                 gain_margin=0., isolate_factual_start=False, wait_value=None,
                 action_encoding='categorical', ensemble_states=(), velocity_reference='ground',
                 ensemble_gain_penalty=0., feedback_relative_hold=False, feedback_relative_scale=None,
                 policy_reference=None, policy_neighborhood=False, state_baseline=False, representation='dense',
                 peer_decision_conditioned=True, decision_rule='value_difference', counterfactual_width=.4):
        super().__init__()
        if decision_rule not in ('value_difference', 'subject_counterfactual'):
            raise ValueError('Unknown recurrent critic decision rule.')
        if not np.isfinite(counterfactual_width) or counterfactual_width <= 0.:
            raise ValueError('The source-centered counterfactual prior requires a positive finite width.')
        self.decision_rule = decision_rule
        self.counterfactual_width = float(counterfactual_width)
        if velocity_reference not in ('ground', 'water'):
            raise ValueError('Dense values require an observed ground or water velocity reference.')
        self.velocity_reference = velocity_reference
        if (feedback_relative_hold != (feedback_relative_scale is not None)
                or feedback_relative_hold and not isolate_factual_start):
            raise ValueError('Dense relative execution requires its measured scale and isolated factual start.')
        self.feedback_relative_hold = bool(feedback_relative_hold)
        self.feedback_relative_scale = feedback_relative_scale
        self.network = self.network_type(defenders, width, identity_conditioned,
            action_encoding=action_encoding, feedback_relative_scale=feedback_relative_scale,
            state_baseline=state_baseline, representation=representation,
            peer_decision_conditioned=peer_decision_conditioned)
        self.network.load_state_dict(state)
        self.ensemble = nn.ModuleList()
        for member_state in ensemble_states:
            member = self.network_type(defenders, width, identity_conditioned,
                action_encoding=action_encoding, feedback_relative_scale=feedback_relative_scale,
                state_baseline=state_baseline, representation=representation,
                peer_decision_conditioned=peer_decision_conditioned)
            member.load_state_dict(member_state)
            self.ensemble.append(member)
        self.register_buffer('options', torch.as_tensor(options, dtype=torch.long))
        self.register_buffer('programs',
            (self.options[:, :1] // (10 ** torch.arange(defenders))) % 10)
        self.isolate_factual_start = bool(isolate_factual_start)
        self.policy_neighborhood = bool(policy_neighborhood)
        if self.isolate_factual_start and not self.policy_neighborhood and bool(((self.programs > 0).sum(-1) > 1).any()):
            raise ValueError('Isolated plain-return execution requires a single-vessel candidate catalog.')
        self.register_buffer('coefficients', torch.tensor(
            [[-1., -1.]] + [[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)]))
        self.decision_period = decision_period
        if not isinstance(intervention_budget, int) or intervention_budget < 0:
            raise ValueError('The shared execution budget must be a nonnegative integer.')
        self.intervention_budget = intervention_budget
        if not np.isfinite(gain_margin) or not 0. <= gain_margin <= 1.:
            raise ValueError('A fixed success-probability gain margin must lie between zero and one.')
        self.gain_margin = float(gain_margin)
        if (not np.isfinite(ensemble_gain_penalty) or ensemble_gain_penalty < 0.
                or ensemble_gain_penalty and not self.ensemble):
            raise ValueError('A gain disagreement penalty requires multiple members and a finite nonnegative weight.')
        self.ensemble_gain_penalty = float(ensemble_gain_penalty)
        self.inference_device = inference_device
        self.prior = None
        self.ordinary = True
        from local_intervention_value import FactualWaitValue
        self.wait_value = None if wait_value is None else FactualWaitValue(**wait_value)
        self.record_decisions = False; self.last_decision = None
        self.record_catalog = False; self.last_catalog = None
        self.policy_reference = None
        if policy_reference is not None:
            if not self.policy_neighborhood:
                recurrent_single_vessel_specification(self)
            record = copy.deepcopy(policy_reference)
            kind = record.pop('kind', None)
            if kind not in ('plain_mlp_return', 'paired_mlp_return'):
                raise ValueError('A policy reference must be a frozen native dense continuation.')
            factory = PlainReturnDecision if kind == 'plain_mlp_return' else PairedReturnDecision
            reference = factory(**record).requires_grad_(False)
            if (reference.network.identity_conditioned != self.network.identity_conditioned
                    or reference.velocity_reference != self.velocity_reference):
                raise ValueError('A policy reference must use the same catalog and public state representation.')
            if self.policy_neighborhood:
                self.policy_reference = reference
                recurrent_decision_specification(self)
                mapping = {tuple(option): index for index, option in enumerate(self.options.tolist())}
                self.register_buffer('reference_indices', torch.tensor([
                    mapping[tuple(option)] for option in reference.options.tolist()], dtype=torch.long))
            elif (recurrent_single_vessel_specification(reference) != recurrent_single_vessel_specification(self)
                    or not torch.equal(reference.options, self.options)):
                raise ValueError('A policy reference must use the same catalog and public state representation.')
            else:
                self.policy_reference = reference
        elif self.policy_neighborhood:
            raise ValueError('A local policy neighborhood cannot omit its frozen reference.')
        if representation in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic'):
            self.program_windows = standalone_single_vessel_windows(self)
        if decision_rule == 'subject_counterfactual':
            recurrent_single_vessel_specification(self)
            if self.policy_reference is not None or self.wait_value is not None:
                raise ValueError('Subject counterfactual ranking requires a standalone recurrent catalog.')
            self.register_buffer('counterfactual_indices', torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer('counterfactual_subjects', torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer('counterfactual_weights', torch.empty(0), persistent=False)
            self.configure_counterfactual_catalog()

    def configure_counterfactual_catalog(self):
        mapping = {tuple(option): index for index, option in enumerate(self.options.tolist())}
        digits = (1, 2, 3, 4, 6, 7, 8, 9)
        self.counterfactual_indices = torch.tensor([[0]+[
            mapping[(digit*10**boat, 0, 5)] for digit in digits]
            for boat in range(self.network.defenders)], dtype=torch.long, device=self.options.device)
        self.counterfactual_subjects = (self.programs > 0).long().argmax(-1)
        offsets = self.feedback_relative_scale*(2.*self.coefficients[list(digits)]-1.)
        offsets = torch.cat((torch.zeros_like(offsets[:1]), offsets))
        self.counterfactual_weights = (-(offsets/self.counterfactual_width).square().mean(-1)).softmax(-1)

    def counterfactual_advantages(self, estimates):
        """Marginalize only the changed vessel, retaining all peers' feedback.

        This is a critic-based local credit comparator on the measured star
        catalog, not a separately trained decentralized COMA actor. The null
        decision has zero credit; source-centered blending alternatives define
        each subject's counterfactual baseline.
        """
        gain = estimates-estimates[..., :1]
        baseline = (gain[..., self.counterfactual_indices]*self.counterfactual_weights).sum(-1)
        advantages = gain-baseline[..., self.counterfactual_subjects]
        return torch.cat((torch.zeros_like(advantages[..., :1]), advantages[..., 1:]), -1)

    def decision_scores(self, ranking, values):
        return ranking if self.ensemble_gain_penalty or self.decision_rule != 'value_difference' else values

    def configure_fleet(self, defenders):
        if defenders == self.network.defenders:
            return
        windows = getattr(self, 'program_windows', None)
        if windows is None:
            windows = standalone_single_vessel_windows(self)
        self.network.configure_fleet(defenders)
        for member in self.ensemble:
            member.configure_fleet(defenders)
        self.program_windows = windows
        options = [(0, 0, 0)]+sorted((digit*10**boat, delay, span)
            for boat in range(defenders) for digit in (1, 2, 3, 4, 6, 7, 8, 9)
            for delay, span in self.program_windows)
        self.options = torch.as_tensor(options, dtype=self.options.dtype, device=self.options.device)
        self.programs = (self.options[:, :1]//(10**torch.arange(defenders, device=self.options.device))) % 10
        if self.decision_rule == 'subject_counterfactual':
            self.configure_counterfactual_catalog()

    def eligible(self, packet):
        eligible = (packet['node'][:, 0, 15]*400.).round().long() % self.decision_period == 0
        if self.intervention_budget and '_conditional_count' in packet:
            eligible &= packet['_conditional_count'].flatten() < self.intervention_budget
        return eligible

    @torch.no_grad()
    def prepare_packet(self, actor, packet, action):
        from conditional_fusion_value import paired_effect_controls
        result = dict(packet)
        if bool(self.eligible(packet).any()):
            result['_plain_return_controls'] = paired_effect_controls(actor, packet, action)
        return result

    def reference_choices(self, context, controls):
        choice = self.policy_reference.selected_catalog_indices(context, controls)
        return self.reference_indices[choice] if self.policy_neighborhood else choice

    def neighborhood_indices(self, context, controls):
        if not self.policy_neighborhood:
            return None
        reference = self.programs[self.reference_choices(context, controls)]
        valid = (self.programs[None] != reference[:, None]).sum(-1) <= 1
        # Each coordinate has exactly eight alternatives to its present digit.
        return valid.nonzero(as_tuple=False)[:, 1].reshape(len(context), 8*self.network.defenders+1)

    def network_catalog(self, member, context, controls, indices=None):
        if indices is None:
            return member.forward_catalog(context, controls, self.programs, self.options[:, 1:])
        batch, count = indices.shape
        raw = member(context[:, None].expand(-1, count, -1).reshape(-1, context.shape[-1]),
            controls[:, None].expand(-1, count, -1, -1).reshape(-1, self.network.defenders, 6),
            self.programs[indices].reshape(-1, self.network.defenders),
            self.options[indices, 1:].reshape(-1, 2)).reshape(batch, count)
        return raw.new_full((batch, len(self.options)), float('nan')).scatter(1, indices, raw)

    def catalog_estimates(self, context, controls):
        self.configure_fleet(controls.shape[-2])
        indices = self.neighborhood_indices(context, controls)
        logits = self.network_catalog(self.network, context, controls, indices)
        if not self.ensemble and self.decision_rule == 'value_difference':
            return logits.masked_fill(torch.isnan(logits), -torch.inf), logits.sigmoid()
        reference = (self.reference_choices(context, controls)
                     if self.ensemble_gain_penalty and self.policy_reference is not None else None)
        return self.aggregate_estimates([logits.sigmoid()]+[
            self.network_catalog(member, context, controls, indices).sigmoid()
            for member in self.ensemble], reference)

    def aggregate_estimates(self, estimates, reference=None):
        estimates = torch.stack(estimates)
        values = estimates.mean(0)
        if self.decision_rule == 'subject_counterfactual':
            advantage = self.counterfactual_advantages(estimates)
            ranking = advantage.mean(0)-self.ensemble_gain_penalty*advantage.std(0, unbiased=False)
            return ranking, values
        # Penalize disagreement of paired gains, not absolute value offsets.
        # Policy improvement compares each member with the actual frozen
        # reference plan, so disagreement of that reference with itself is zero.
        # This is a decision criterion, not a calibrated confidence interval
        # or a correction to the raw neural success probabilities.
        baseline = (estimates[:, :, :1] if reference is None else
                    estimates.gather(2, reference[None, :, None].expand(len(estimates), -1, 1)))
        gains = estimates-baseline
        ranking = values-self.ensemble_gain_penalty*gains.std(0, unbiased=False)
        return ranking.masked_fill(torch.isnan(ranking), -torch.inf), values

    def selected_catalog_indices(self, context, controls):
        """Recover the exact current choice of a frozen recurrent policy."""
        if self.wait_value is not None or self.intervention_budget:
            raise ValueError('A recurrent policy reference cannot have waiting or episode-budget state.')
        ranking, values = self.catalog_estimates(context, controls)
        choice = ranking.argmax(-1)
        score = self.decision_scores(ranking, values)
        reference = (torch.zeros_like(choice) if self.policy_reference is None else
                     self.reference_choices(context, controls))
        improvement = score.gather(1, choice[:, None]).flatten()-score.gather(1, reference[:, None]).flatten()
        return torch.where(improvement > self.gain_margin, choice, reference)

    def public_context(self, packet):
        from conditional_fusion_value import intervention_task_context, append_vessel_identity
        if self.velocity_reference == 'water':
            packet = dict(packet, motion=torch.cat((packet['motion'][..., :3], 5.*packet['water']), -1))
        context, order = intervention_task_context(packet, True, True)
        if self.network.identity_conditioned:
            context = append_vessel_identity(context, order)
        return context, order

    @torch.no_grad()
    def forward(self, packet, action):
        if self.record_decisions:
            self.last_decision = None
        if self.record_catalog:
            self.last_catalog = None
        eligible = self.eligible(packet)
        if not bool(eligible.any()):
            return action
        return_device = action.device
        if self.inference_device is not None and self.options.device != torch.device(self.inference_device):
            if str(self.inference_device).startswith('cuda'):
                os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
                torch.use_deterministic_algorithms(True)
                torch.backends.cuda.matmul.allow_tf32 = False
            self.to(self.inference_device)
        context, order = self.public_context(packet)
        self.configure_fleet(order.shape[1])
        controls = packet['_plain_return_controls'].gather(
            1, order[..., None].expand(-1, -1, 6))
        device = self.options.device
        context, controls, order, action, eligible = [
            item.to(device) for item in (context, controls, order, action, eligible)]
        ranking, values = self.catalog_estimates(context, controls)
        choice = ranking.argmax(-1)
        accept = eligible & (choice != 0)
        gain = values.gather(1, choice[:, None]).flatten()-values[:, 0]
        score = self.decision_scores(ranking, values)
        decision_gain = score.gather(1, choice[:, None]).flatten()-score[:, 0]
        if self.gain_margin:
            accept &= decision_gain > self.gain_margin
        wait = None
        if self.wait_value is not None:
            wait_margin = self.gain_margin if getattr(self.wait_value, 'margin_adjusted', True) else 0.
            wait = self.wait_value(context, controls)
            accept &= decision_gain-wait_margin > wait
        reference_choice = proposal_choice = improvement = None
        if self.policy_reference is not None:
            reference_choice = self.reference_choices(context, controls)
            proposal_choice = choice
            score = self.decision_scores(ranking, values)
            improvement = (score.gather(1, choice[:, None]).flatten()-
                           score.gather(1, reference_choice[:, None]).flatten())
            choice = torch.where(improvement > self.gain_margin, choice, reference_choice)
            accept = eligible & (choice != 0)
            gain = values.gather(1, choice[:, None]).flatten()-values[:, 0]
            decision_gain = score.gather(1, choice[:, None]).flatten()-score[:, 0]
        if self.record_catalog:
            self.last_catalog = dict(options=self.options.detach().cpu(),
                factual_value=values[:, 0].detach().cpu(),
                predicted_gain=(values-values[:, :1]).detach().cpu(),
                score=self.decision_scores(ranking, values).detach().cpu())
            if wait is not None:
                self.last_catalog['wait_value'] = wait.detach().cpu()
        if self.record_decisions:
            gain = values.gather(1, choice[:, None]).flatten()-values[:, 0]
            self.last_decision = dict(reason='catalog', factual_value=values[:, 0].detach().cpu(),
                option=self.options[choice].detach().cpu(), accepted=accept.detach().cpu(),
                predicted_gain=torch.where(accept, gain, torch.zeros_like(gain)).detach().cpu(),
                candidate_gain=gain.detach().cpu(),
                decision_gain=decision_gain.detach().cpu(),
                score=self.decision_scores(ranking, values).gather(
                    1, choice[:, None]).flatten().detach().cpu())
            if wait is not None:
                self.last_decision['wait_value'] = wait.detach().cpu()
            if reference_choice is not None:
                self.last_decision.update(reference_option=self.options[reference_choice].detach().cpu(),
                    proposal_option=self.options[proposal_choice].detach().cpu(),
                    proposal_improvement=improvement.detach().cpu(),
                    reference_replaced=(eligible & (choice != reference_choice)).detach().cpu())
        if not bool(accept.any()):
            return action.to(return_device)
        preference = self.coefficients[self.programs[choice]]
        preference = preference.gather(1, order.argsort(-1)[..., None].expand(-1, -1, 2))
        delay, span = self.options[choice, 1:].unbind(-1)
        one = torch.ones_like(action[..., :1]); free = -one
        first = torch.where((delay == 0)[:, None, None], preference, free.expand(-1, -1, 2))
        proposed = torch.cat((action[..., :3], free, (delay+span)[:, None, None]*one, 2*free,
            torch.zeros_like(action[..., :2]), delay[:, None, None]*one, first, preference, one), -1)
        factual = torch.cat((action[..., :3], free, one, 2*free, torch.zeros_like(action[..., :2]),
            one, free.expand(-1, -1, 2), free.expand(-1, -1, 2), free), -1)
        return torch.where(accept[:, None, None], proposed, factual).to(return_device)


class PairedReturnDecision(PlainReturnDecision):
    """Rank learned same-future differences with the shared physical executor."""
    network_type = PairedReturnNetwork

    def __init__(self, *args, learning_objective='conditional',
                 conditional_objective='marginal_events', **kwargs):
        if learning_objective != 'conditional':
            raise ValueError('Paired dense returns require the conditional learning objective.')
        if conditional_objective not in ('marginal_events', 'event_hierarchy', 'factual_strata', 'shared_potential', 'joint_likelihood', 'paired_ordering', 'paired_catalog_contrast'):
            raise ValueError('Unknown paired dense conditional objective.')
        self.conditional_objective = conditional_objective
        if conditional_objective in ('shared_potential', 'paired_ordering', 'paired_catalog_contrast'):
            self.network_type = PlainReturnNetwork
        elif kwargs.get('policy_neighborhood', False):
            raise ValueError('Local neighborhoods require complete-plan scalar values.')
        super().__init__(*args, **kwargs)
        if not torch.equal(self.options[0], torch.zeros_like(self.options[0])):
            raise ValueError('The paired dense catalog must start with its factual reference.')
        self.ordinary = False

    def catalog_estimates(self, context, controls):
        self.configure_fleet(controls.shape[-2])
        estimates = []
        indices = self.neighborhood_indices(context, controls)
        for member in [self.network, *self.ensemble]:
            raw = self.network_catalog(member, context, controls, indices)
            if self.conditional_objective in ('shared_potential', 'paired_ordering', 'paired_catalog_contrast'):
                estimates.append(raw.sigmoid())
                continue
            factual = raw[:, :1, 0].sigmoid()
            gain = member.signed_gain(raw, member.active_vessels(self.programs).any(-1)[None],
                factual if self.conditional_objective in ('factual_strata', 'joint_likelihood') else None,
                event_hierarchy=self.conditional_objective == 'event_hierarchy')
            estimates.append(factual+gain)
        # The common factual value cancels before ranking and margin checks.
        # Avoid clipping a beneficial contrast at the factual value boundary.
        reference = (self.reference_choices(context, controls)
                     if self.ensemble_gain_penalty and self.policy_reference is not None else None)
        return self.aggregate_estimates(estimates, reference)


def crossfit_continuation_targets(data, bootstrap):
    """Validate fixed-policy continuation estimates kept separate from observations."""
    scenes = {tuple(root['scene']) for root in data['roots']}
    targets = bootstrap.get('training_targets', {})
    fits = {fit['fold']: fit for fit in bootstrap.get('fits', ())}
    folds = bootstrap.get('fold_by_scene', {})
    if (not bootstrap.get('completed') or bootstrap.get('predecessor_sha256') != data['predecessor_sha256']
            or bootstrap.get('continuation_context') != data.get('continuation_context')
            or not data.get('continuation_context') or set(targets) != scenes or set(folds) != scenes
            or len(fits) < 2 or set(fits) != set(folds.values())
            or any(fit.get('updates', 0) <= 0 for fit in fits.values())):
        raise ValueError('Continuation targets require complete cross-fitting under the same frozen policy.')
    digest = hashlib.sha256()
    for scene in sorted(targets):
        fit = fits[folds[scene]]
        training = {tuple(value) for value in fit['training_scenes']}
        excluded = {tuple(value) for value in fit['excluded_scenes']}
        expected = {value for value in scenes if folds[value] == folds[scene]}
        if excluded != expected or training != scenes-excluded or training & excluded:
            raise ValueError('A continuation estimate must exclude the entire queried scene and all its futures.')
        digest.update(repr(scene).encode())
        for key, value in sorted(targets[scene].items()):
            if not np.isfinite(value) or not 0. <= value <= 1.:
                raise ValueError('Continuation success estimates must be finite and bounded.')
            digest.update(repr((key, value)).encode())
    if digest.hexdigest() != bootstrap.get('training_targets_sha256'):
        raise ValueError('Continuation target predictions changed after cross-fitting.')
    for root in data['roots']:
        values = targets[tuple(root['scene'])]
        if set(values) != set(root['outcomes']):
            raise ValueError('Continuation targets must cover exactly the recorded branches and futures.')
        for key, outcome in root['outcomes'].items():
            if 'continuation_context' not in outcome and values[key] != float(outcome['success']):
                raise ValueError('An already terminal branch must retain its measured outcome.')
    return targets, dict(target_sha256=digest.hexdigest(), folds=len(fits),
        shared_optimizer_updates=sum(fit['updates'] for fit in fits.values()))


def single_vessel_contrast_groups(rows, common_future_ids, defenders):
    """Group measured plans that differ in one vessel at the same clock."""
    grouped = {}
    powers = [10**boat for boat in range(defenders)]
    for index, (row, identity) in enumerate(zip(rows, common_future_ids, strict=True)):
        root, program, delay, span, _ = row
        if root != identity[0]:
            raise ValueError('Common-future labels must refer to their recorded root.')
        if span <= 0:
            continue
        for boat, power in enumerate(powers):
            peers = program-((program//power) % 10)*power
            key = (*identity, delay, span, boat, peers)
            grouped.setdefault(key, []).append(index)
    return [np.asarray(indices, np.int64) for indices in grouped.values() if len(indices) > 1]


def common_future_contrast_loss(estimates, targets, mask):
    """Mean squared contrasts within each valid local intervention group.

    Centering the error is exactly half the mean over ordered action pairs,
    so a common factual-outcome offset cannot dominate the action contrasts.
    Padding has no statistical mass.
    """
    count = mask.sum(-1).clamp_min(1)
    error = (estimates-targets)*mask
    centered = (error-error.sum(-1, keepdim=True)/count[:, None])*mask
    return 2.*(centered.square().sum(-1)/count).mean()


def fit_plain_return(data, output, *, factual_bank=None, onpolicy_bank=None, updates=24000,
                     parameter_budget=145731, seed=42, device='cuda:0',
                     expected_observation_sha256=None, factual_bank_weight=1.,
                     uniform_sampling=True, initial_model=None, learning_rate=8e-4,
                     additional_candidate_options=(), identity_conditioned=False, snapshot_updates=(),
                     root_sampling_fraction=0., value_target='success',
                     expected_training_target_sha256=None, intervention_budget=0, gain_margin=0.,
                     paired_difference_weight=0., action_encoding='categorical',
                     conditional_objective='marginal_events', velocity_reference='ground',
                     transfer_initialization=False, conservative_catalog_weight=0.,
                     conservative_catalog_temperature=.1, conservative_catalog_samples=64,
                     conservative_catalog_roots=4, unique_factual_supervision=False,
                     root_sampling_weights=None, factual_sampling_weights=None, onpolicy_bank_weight=1.,
                     continuation_bootstrap=None, continuation_bootstrap_weight=.5, state_baseline=False,
                     representation='dense', peer_decision_conditioned=True, future_weighted_sampling=False,
                     factual_pretraining_updates=0):
    """Fit only absolute outcomes, independently of conditional-learning code.

    The default pools all observed state/action returns and samples them
    uniformly. Factual continuation is simply the zero-action observation;
    it receives no separate objective or development-future weighting.
    The explicit weighted mode reproduces the earlier matched-data comparator.
    Neither pairing identities nor contrasts enter the ordinary objective.
    An explicit positive paired_difference_weight selects a separate dense
    conditional estimator; it never changes the default return comparator.
    """
    require_plain_predecessor(data['model'])
    if (velocity_reference not in ('ground', 'water') or
            data.get('state_velocity_reference', 'ground') != velocity_reference or
            factual_bank is not None and factual_bank.get('state_velocity_reference', 'ground') != velocity_reference):
        raise ValueError('Dense training must match the observed velocity reference of every input population.')
    if action_encoding not in ('categorical', 'physical_delta'):
        raise ValueError('Unknown dense intervention action representation.')
    if not np.isfinite(paired_difference_weight) or paired_difference_weight < 0.:
        raise ValueError('The paired difference weight must be finite and nonnegative.')
    if (not np.isfinite(conservative_catalog_weight) or conservative_catalog_weight < 0.
            or not np.isfinite(conservative_catalog_temperature) or conservative_catalog_temperature <= 0.
            or not isinstance(conservative_catalog_samples, int) or conservative_catalog_samples < 1
            or not isinstance(conservative_catalog_roots, int) or conservative_catalog_roots < 1):
        raise ValueError('Conservative catalog learning requires valid nonnegative weight, temperature and sample counts.')
    conditional = paired_difference_weight > 0.
    if onpolicy_bank is not None:
        reconstructed_actions = (onpolicy_bank.get('observed_decisions_reconstructed', False)
            and onpolicy_bank.get('reconstruction_policy_sha256') == data.get('predecessor_sha256'))
        if (factual_bank is not None or not uniform_sampling or value_target != 'success'
                or not onpolicy_bank.get('completed') or not onpolicy_bank.get('recurrent_m2')
                or not (onpolicy_bank.get('observed_decisions_recorded') or reconstructed_actions)
                or onpolicy_bank.get('period') != 5
                or onpolicy_bank.get('predecessor_sha256') != data.get('predecessor_sha256')
                or data.get('continuation_context', {}).get('continuation_policy_sha256') != data.get('predecessor_sha256')
                or not np.isfinite(onpolicy_bank_weight) or onpolicy_bank_weight <= 0.):
            raise ValueError('Observed action returns require their completed recurrent bank and common continuation.')
    if not isinstance(unique_factual_supervision, bool):
        raise ValueError('Unique factual supervision must be explicitly enabled or disabled.')
    if unique_factual_supervision and conditional and conditional_objective != 'joint_likelihood':
        raise ValueError('Unique factual supervision requires the joint outcome likelihood.')
    if (conditional_objective not in ('marginal_events', 'event_hierarchy', 'factual_strata', 'shared_potential', 'joint_likelihood', 'paired_ordering', 'paired_catalog_contrast')
            or conditional_objective in ('factual_strata', 'joint_likelihood', 'paired_ordering', 'paired_catalog_contrast') and (not conditional or value_target != 'success')):
        raise ValueError('Factual-stratum dense learning requires paired binary success outcomes.')
    if conditional_objective == 'event_hierarchy' and (not conditional or value_target != 'success'):
        raise ValueError('Change-direction learning requires paired binary success outcomes.')
    if conditional_objective == 'joint_likelihood' and paired_difference_weight != 1.:
        raise ValueError('A joint outcome likelihood requires unit weight for its conditional factor.')
    shared_potential = conditional and conditional_objective in ('shared_potential', 'paired_ordering', 'paired_catalog_contrast')
    bootstrap_targets = bootstrap_metadata = None
    if continuation_bootstrap is not None:
        if (value_target != 'success' or not uniform_sampling or factual_bank is not None
                or initial_model is not None or conditional and conditional_objective != 'shared_potential'
                or not np.isfinite(continuation_bootstrap_weight) or not 0. < continuation_bootstrap_weight < 1.):
            raise ValueError('Cross-fitted continuation mixtures require fresh scalar success learning and a weight in (0,1).')
        bootstrap_targets, bootstrap_metadata = crossfit_continuation_targets(data, continuation_bootstrap)
    policy_neighborhood = bool(data.get('policy_neighborhood', False))
    if policy_neighborhood and (conditional and not shared_potential or factual_bank is not None
            or conservative_catalog_weight or data.get('policy_reference') is None):
        raise ValueError('Policy-local paired learning requires complete-plan values and its actual reference policy.')
    if conditional and not uniform_sampling:
        raise ValueError('Dense conditional fitting requires pooled-observation sampling.')
    if not isinstance(future_weighted_sampling, bool) or future_weighted_sampling and not uniform_sampling:
        raise ValueError('Future population correction requires explicit pooled-observation sampling.')
    if not isinstance(transfer_initialization, bool):
        raise ValueError('Cross-policy initialization must be explicitly enabled or disabled.')
    if transfer_initialization and initial_model is None:
        raise ValueError('Cross-policy initialization requires previously trained parameters.')
    if value_target not in ('success', 'discounted_return'):
        raise ValueError('Choose success or the measured discounted task return.')
    if value_target == 'discounted_return' and data['model'].get('success_objective', False):
        raise ValueError('Discounted returns require the predecessor using the discounted task objective.')
    isolate_factual_start = bool(data.get('isolate_factual_start', False))
    feedback_relative_hold = bool(data.get('feedback_relative_hold', False))
    feedback_relative_scale = data.get('feedback_relative_scale')
    if (feedback_relative_hold != (feedback_relative_scale is not None) or feedback_relative_hold and
            (not isolate_factual_start or action_encoding != 'physical_delta' or
             not np.isfinite(feedback_relative_scale) or not 0. < feedback_relative_scale <= 1. or
             any(not root.get('feedback_relative_hold') or
                 root.get('feedback_relative_scale') != feedback_relative_scale for root in data['roots']))):
        raise ValueError('Dense relative learning requires matching measured relative branches and physical features.')
    if conditional and not isolate_factual_start:
        raise ValueError('Dense paired differences require isolated single-vessel intervention observations.')
    if not isinstance(intervention_budget, int) or intervention_budget < 0:
        raise ValueError('The shared execution budget must be a nonnegative integer.')
    if policy_neighborhood and (not feedback_relative_hold or intervention_budget != 0):
        raise ValueError('Policy-local learning requires recurrent relative execution.')
    if not np.isfinite(gain_margin) or not 0. <= gain_margin <= 1.:
        raise ValueError('The fixed return-gain margin must lie between zero and one.')
    if updates <= 0:
        raise ValueError('Plain return fitting requires a positive optimizer budget.')
    if (type(factual_pretraining_updates) is not int or
            not 0 <= factual_pretraining_updates < updates):
        raise ValueError('Factual pretraining must fit inside the actual optimizer budget.')
    if factual_pretraining_updates and (
            representation not in ('fleet_quadratic', 'fleet_graph_quadratic')
            or not state_baseline or not peer_decision_conditioned or not uniform_sampling
            or value_target != 'success' or initial_model is not None
            or conditional and not shared_potential or policy_neighborhood
            or continuation_bootstrap is not None or conservative_catalog_weight or onpolicy_bank is not None):
        raise ValueError('Frozen factual representations require fresh standalone scalar quadratic success learning.')
    snapshot_updates = tuple(sorted(set(map(int, snapshot_updates))))
    if any(step <= 0 or step > updates for step in snapshot_updates):
        raise ValueError('Learning snapshots must lie within the actual optimizer budget.')
    if initial_model is not None:
        require_plain_predecessor(initial_model, allow_plain_return=not conditional,
                                  allow_paired_return=conditional)
        expected_kind = 'paired_mlp_return' if conditional else 'plain_mlp_return'
        if initial_model.get('successor_task_value', {}).get('kind') != expected_kind:
            raise ValueError('Dense continuation must retain its absolute or conditional learning objective.')
        if initial_model['successor_task_value'].get('ensemble_states'):
            raise ValueError('An ensemble continuation must explicitly retain and update every member.')
        previous_fit = initial_model['successor_task_fit']
        if previous_fit['predecessor_sha256'] != data['predecessor_sha256'] and not transfer_initialization:
            raise ValueError('Initialization from another factual policy requires explicit cross-policy transfer.')
        if conditional and initial_model['successor_task_value'].get(
                'conditional_objective', 'marginal_events') != conditional_objective:
            raise ValueError('Conditional continuation must retain its paired objective.')
    if uniform_sampling and factual_bank_weight != 1.:
        raise ValueError('Uniform return regression has no separate factual loss weight.')
    if factual_sampling_weights is not None and (factual_bank is None or not uniform_sampling):
        raise ValueError('Factual state sampling priors require a factual bank and pooled training.')
    roots = []
    for original in data['roots']:
        used = sorted({q for q, _ in original['outcomes']})
        index = {old: new for new, old in enumerate(used)}
        roots.append(dict(original, options=[original['options'][q] for q in used],
            outcomes={(index[q], f): item for (q, f), item in original['outcomes'].items()}))
    defenders = len(roots[0]['effect_controls'])
    if any(len(root['context']) != (8+identity_conditioned)*defenders+5 for root in roots):
        raise ValueError('Plain return roots must use the requested public identity representation.')
    reference_options = []
    for root in roots:
        reference = tuple(root.get('reference_option', (0, 0, 0)))
        if (conditional or policy_neighborhood) and (tuple(root['options'][0]) != reference
                or not policy_neighborhood and reference != (0, 0, 0)):
            raise ValueError('The first observed option must be the recorded comparison reference.')
        if policy_neighborhood:
            programs_at_root = (np.asarray(root['options'])[:, :1]//(10**np.arange(defenders))) % 10
            if (((programs_at_root != programs_at_root[:1]).sum(-1) > 1).any()
                    or not set(map(tuple, root['options'])) <= set(single_coordinate_catalog(defenders))):
                raise ValueError('Policy-local contrasts may change only one vessel from the recorded reference.')
        reference_options.append(reference)
    digest = hashlib.sha256()
    if bootstrap_metadata is not None:
        digest.update(repr(('crossfit_continuation', bootstrap_metadata['target_sha256'],
                            float(continuation_bootstrap_weight))).encode())
    if feedback_relative_hold:
        digest.update(repr(('feedback_relative_hold', float(feedback_relative_scale))).encode())
    rows, weights, factual_targets, measured_targets = [], [], [], []
    common_future_ids = []
    for root_id, root in enumerate(roots):
        if value_target == 'discounted_return':
            from local_intervention_value import measured_root_values
            measured = measured_root_values(root, value_target)
        future_weights = np.asarray(root.get('future_weights', np.ones(len(root['futures']))), np.float32)
        for array in (np.asarray(root['context'], np.float32),
                      np.asarray(root['effect_controls'], np.float32),
                      np.asarray(root['options'], np.int64), future_weights,
                      np.asarray([(q, f, item['success']) for (q, f), item in
                                  sorted(root['outcomes'].items())], np.float64)):
            digest.update(array.tobytes())
        for (q, future), outcome in sorted(root['outcomes'].items()):
            label = measured[q, future] if value_target == 'discounted_return' else float(outcome['success'])
            measured_targets.append(label)
            if bootstrap_targets is not None:
                estimate = bootstrap_targets[tuple(root['scene'])][q, future]
                label = (1.-continuation_bootstrap_weight)*label+continuation_bootstrap_weight*estimate
            rows.append((root_id, *root['options'][q], label))
            if conditional and conditional_objective == 'paired_catalog_contrast':
                common_future_ids.append((root_id, future))
            weights.append(future_weights[future])
            if conditional:
                if (0, future) not in root['outcomes']:
                    raise ValueError('Every dense conditional label requires its same-future factual outcome.')
                factual_label = (measured[0, future] if value_target == 'discounted_return' else
                                 float(root['outcomes'][0, future]['success']))
                if bootstrap_targets is not None:
                    estimate = bootstrap_targets[tuple(root['scene'])][0, future]
                    factual_label = ((1.-continuation_bootstrap_weight)*factual_label
                                     +continuation_bootstrap_weight*estimate)
                factual_targets.append(factual_label)
    continuation = None
    if factual_bank is not None:
        if factual_bank.get('recurrent_m2', False):
            raise ValueError('Recurrent trajectory values are not zero-intervention labels; measure paired branches instead.')
        if factual_bank['predecessor_sha256'] != data['predecessor_sha256']:
            raise ValueError('The ordinary labels must come from its own frozen predecessor.')
        arrays = [np.concatenate([record[key] for record in factual_bank['records']]).astype(np.float32)
                  for key in ('contexts', 'controls', 'values', 'remaining')]
        if arrays[0].shape[-1] != (8+identity_conditioned)*defenders+5:
            raise ValueError('Plain return factual states require the same measured identity representation.')
        for array in arrays:
            digest.update(array.tobytes())
        if value_target == 'discounted_return':
            from local_intervention_value import discounted_task_values
            arrays[2] = discounted_task_values(arrays[2], arrays[3]).astype(np.float32)
        continuation = [torch.as_tensor(array, device=device) for array in arrays[:3]]
    onpolicy = None; onpolicy_probability = None; onpolicy_record_weights = None
    reconstructed_action_labels = 0
    if onpolicy_bank is not None:
        records = onpolicy_bank['records']
        if not records or any(not len(record['values']) for record in records):
            raise ValueError('Observed recurrent action trajectories must be nonempty.')
        for record in records:
            if record.get('decision_reconstructed', False):
                ticks = np.asarray(record['contexts'])[:, -1]*400.
                steps = np.rint(ticks).astype(np.int64)
                if (not reconstructed_actions or not np.allclose(ticks, steps, atol=1e-3, rtol=0.)
                        or (steps % onpolicy_bank['period']).any()):
                    raise ValueError('Reconstructed actions must use their frozen continuation at an eligible measured clock.')
                reconstructed_action_labels += len(record['values'])
        if onpolicy_bank.get('observed_decisions_reconstructed', False) and not reconstructed_action_labels:
            raise ValueError('Reconstructed action provenance requires measured successor records.')
        arrays = [np.concatenate([record[key] for record in records])
                  for key in ('contexts', 'controls', 'observed_decision_options', 'values')]
        if (arrays[2].shape != (len(arrays[3]), 3) or not np.isfinite(arrays[2]).all()
                or not np.equal(arrays[2], np.rint(arrays[2])).all()):
            raise ValueError('Observed recurrent action labels must be integer catalog options.')
        arrays = [np.asarray(array, np.int64 if index == 2 else np.float32)
                  for index, array in enumerate(arrays)]
        size = len(arrays[3]); codes_observed = arrays[2][:, 0]
        digits = (codes_observed[:, None]//(10**np.arange(defenders))) % 10
        if (not size or arrays[0].shape != (size, (8+identity_conditioned)*defenders+5)
                or arrays[1].shape != (size, defenders, 6) or arrays[2].shape != (size, 3)
                or any(not np.isfinite(array).all() for array in arrays)
                or not np.isin(arrays[3], (0., 1.)).all()
                or not np.isin(digits, (0, 1, 2, 3, 4, 6, 7, 8, 9)).all()
                or (codes_observed < 0).any() or (codes_observed >= 10**defenders).any()
                or ((digits > 0).sum(-1) > 1).any()
                or not np.array_equal(arrays[2][:, 1:], np.c_[np.zeros(size, np.int64), 5*(codes_observed > 0)])
                or any(len(record['values']) != len(record['observed_decision_options'])
                    or not np.all(record['values'] == record['values'][0]) for record in records)):
            raise ValueError('Observed recurrent returns must retain actual immediate single-vessel action labels.')
        digest.update(b'observed recurrent action returns')
        for array in arrays:
            digest.update(array.tobytes())
        onpolicy = [torch.as_tensor(array, device=device) for array in arrays]
        if 'record_sampling_weights' in onpolicy_bank:
            onpolicy_record_weights = np.asarray(onpolicy_bank['record_sampling_weights'], np.float64)
            if (onpolicy_record_weights.shape != (len(records),) or not np.isfinite(onpolicy_record_weights).all()
                    or (onpolicy_record_weights <= 0.).any() or not np.isfinite(onpolicy_record_weights.sum())):
                raise ValueError('Observed action record masses must be finite, positive and complete.')
            onpolicy_record_weights = onpolicy_record_weights/onpolicy_record_weights.sum()
            onpolicy_probability = np.concatenate([np.full(len(record['values']), weight/len(record['values']))
                for record, weight in zip(records, onpolicy_record_weights)])
            digest.update(b'observed action record masses'); digest.update(onpolicy_record_weights.tobytes())
        else:
            onpolicy_probability = np.concatenate([np.full(len(record['values']),
                1./(len(records)*len(record['values']))) for record in records])
        if reconstructed_action_labels:
            digest.update(b'reconstructed frozen continuation actions')
            digest.update(onpolicy_bank['reconstruction_policy_sha256'].encode())
    observation_sha = digest.hexdigest()
    if expected_observation_sha256 is not None and observation_sha != expected_observation_sha256:
        raise ValueError('The plain MLP must see exactly the matched observations.')
    rows = np.asarray(rows)
    row_probability = root_sampling_probabilities(rows[:, 0], root_sampling_fraction, root_sampling_weights,
        np.asarray(weights, np.float64) if future_weighted_sampling else None)
    root_prior = (None if root_sampling_weights is None else
        root_sampling_probabilities(np.arange(len(roots)), 1., root_sampling_weights))
    factual_prior = (None if factual_sampling_weights is None else
        root_sampling_probabilities(np.arange(len(continuation[2])), 1., factual_sampling_weights))
    context = torch.as_tensor(np.asarray([r['context'] for r in roots]), dtype=torch.float32, device=device)
    controls = torch.as_tensor(np.asarray([r['effect_controls'] for r in roots]), dtype=torch.float32, device=device)
    root_ids = torch.as_tensor(rows[:, 0], dtype=torch.long, device=device)
    codes = torch.as_tensor(rows[:, 1], dtype=torch.long, device=device)
    programs = (codes[:, None] // (10 ** torch.arange(defenders, device=device))) % 10
    extra_options = {tuple(map(int, option)) for option in additional_candidate_options}
    if any(len(option) != 3 or not (0 <= option[0] < 10**defenders) or min(option[1:]) < 0
           or (option[0] == 0 and option != (0, 0, 0)) for option in extra_options):
        raise ValueError('The shared catalog must contain fleet-compatible finite interventions.')
    timing = torch.as_tensor(rows[:, 2:4], dtype=torch.float32, device=device)
    target = torch.as_tensor(rows[:, 4], dtype=torch.float32, device=device)
    target_digest = hashlib.sha256(np.asarray(rows[:, 4], np.float32).tobytes())
    if continuation is not None:
        target_digest.update(continuation[2].detach().cpu().numpy().tobytes())
    if onpolicy is not None:
        target_digest.update(onpolicy[3].detach().cpu().numpy().tobytes())
    training_target_sha256 = target_digest.hexdigest()
    if expected_training_target_sha256 is not None and training_target_sha256 != expected_training_target_sha256:
        raise ValueError('The plain MLP must use exactly the matched measured return targets.')
    sampled_options = {tuple(option) for root in roots for option in root['options'] if option[0]}
    options = [(0, 0, 0)]+sorted((sampled_options | extra_options)-{(0, 0, 0)})
    if policy_neighborhood:
        options = single_coordinate_catalog(defenders)
    if isolate_factual_start and not policy_neighborhood and any(sum(code//10**boat % 10 != 0 for boat in range(defenders)) > 1
                                    for code, _, _ in options):
        raise ValueError('Isolated plain-return learning requires a single-vessel candidate catalog.')
    importance = torch.as_tensor(np.asarray(weights), device=device)
    action_size = 10*defenders+2 if action_encoding == 'categorical' else 5*defenders+4
    input_size = (14+identity_conditioned)*defenders+5+action_size
    output_width = 3 if conditional and not shared_potential else 1
    if state_baseline and (action_encoding != 'physical_delta' or output_width != 1):
        raise ValueError('A separate state baseline requires scalar physical-action values.')
    if representation not in ('dense', 'fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic'):
        raise ValueError('Unknown intervention state/action representation.')
    if representation in ('fleet_attention', 'fleet_quadratic', 'fleet_graph_quadratic') and (not state_baseline or not feedback_relative_hold
            or policy_neighborhood):
        raise ValueError('Shared fleet fitting requires standalone relative interventions with a state baseline.')
    count = lambda width: (2*width*width + (input_size+3+output_width)*width + output_width
        + (2*width*width+((14+identity_conditioned)*defenders+9)*width+1 if state_baseline else 0))
    if representation == 'fleet_attention':
        count = lambda width: 21*width*width+(80+identity_conditioned)*width+98
    elif representation == 'fleet_quadratic':
        count = lambda width: 14*width*width+(60+identity_conditioned)*width+15
    elif representation == 'fleet_graph_quadratic':
        count = lambda width: 12*width*width+(63+identity_conditioned)*width+15
    width = min(range(8, 2049, 4 if representation in ('fleet_attention', 'fleet_quadratic') else 1),
                key=lambda value: abs(count(value)-parameter_budget))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network_type = PairedReturnNetwork if conditional and not shared_potential else PlainReturnNetwork
    net = network_type(defenders, width, identity_conditioned, action_encoding=action_encoding,
        feedback_relative_scale=feedback_relative_scale, state_baseline=state_baseline,
        representation=representation, peer_decision_conditioned=peer_decision_conditioned).to(device)
    assert sum(parameter.numel() for parameter in net.parameters()) == count(width)
    initialization_sha256 = None
    initialization_updates = 0
    if initial_model is not None:
        recipe = initial_model['successor_task_value']
        if recipe.get('velocity_reference', 'ground') != velocity_reference:
            raise ValueError('Continued dense fitting must retain its observed velocity reference.')
        if (initial_model['successor_task_fit'].get('value_target', 'success') != value_target
                or bool(recipe.get('isolate_factual_start', False)) != isolate_factual_start
                or bool(recipe.get('feedback_relative_hold', False)) != feedback_relative_hold
                or recipe.get('feedback_relative_scale') != feedback_relative_scale
                or recipe.get('state_baseline', False) != state_baseline
                or recipe.get('representation', 'dense') != representation
                or recipe.get('peer_decision_conditioned', True) != peer_decision_conditioned
                or recipe.get('action_encoding', 'categorical') != action_encoding):
            raise ValueError('Plain-return continuation must retain its measured target and execution semantics.')
        if (recipe['defenders'], recipe['width'], recipe.get('identity_conditioned',False)) != (defenders, width, identity_conditioned):
            raise ValueError('Plain return continuation must retain its input and parameter dimensions.')
        net.load_state_dict(recipe['state'])
        initialization = hashlib.sha256()
        for key, value in sorted(recipe['state'].items()):
            initialization.update(key.encode())
            initialization.update(value.detach().cpu().numpy().tobytes())
        initialization_sha256 = initialization.hexdigest()
        previous_fit = initial_model['successor_task_fit']
        initialization_updates = previous_fit.get('total_parameter_updates', previous_fit['updates'])
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=.005)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, factual_pretraining_updates or updates, eta_min=learning_rate/10.)
    snapshots = {}
    pooled_features = pooled_target = None
    pooled_probability = None
    if uniform_sampling:
        pooled_features = torch.cat((context[root_ids], controls[root_ids].flatten(1),
                                     net.action_features(programs, timing, controls[root_ids])), -1)
        pooled_target = target
        if continuation is not None:
            count_factual = len(continuation[2])
            factual_features = torch.cat((continuation[0], continuation[1].flatten(1),
                net.action_features(torch.zeros((count_factual, defenders), dtype=torch.long, device=device),
                                    torch.zeros((count_factual, 2), device=device), continuation[1])), -1)
            pooled_features = torch.cat((pooled_features, factual_features))
            pooled_target = torch.cat((pooled_target, continuation[2]))
        factual_mass = (np.ones(len(pooled_target)-len(rows)) if factual_prior is None else
                        factual_prior*len(factual_prior))
        if row_probability is not None or factual_prior is not None:
            # Keep both populations' total observed mass. Positive priors
            # change only the distribution within each measured population.
            paired_mass = np.ones(len(rows)) if row_probability is None else row_probability*len(rows)
            pooled_probability = np.concatenate((paired_mass, factual_mass))/len(pooled_target)
    pretraining_ids = pretraining_probability = None
    pretraining_draws = hashlib.sha256()
    pretrained_state_sha256 = None
    def factual_state_digest():
        digest = hashlib.sha256()
        for name, value in sorted(net.state_dict().items()):
            if not name.startswith('layers.'):
                digest.update(name.encode())
                digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    if factual_pretraining_updates:
        measured_factual = np.flatnonzero((rows[:, 1:4] == 0).all(-1))
        pretraining_ids = np.r_[measured_factual, np.arange(len(rows), len(pooled_target))]
        if not len(pretraining_ids):
            raise ValueError('Factual pretraining requires actual zero-action outcomes.')
        mass = (np.ones(len(pretraining_ids)) if pooled_probability is None
                else pooled_probability[pretraining_ids])
        pretraining_probability = mass/mass.sum()
        pretraining_rng = np.random.default_rng(np.random.SeedSequence([seed, 840016]))
    pooled_reference = paired_factual_target = paired_active = None
    paired_target_sha256 = None
    if conditional:
        # Common-current-state inputs are identical to the ordinary learner.
        # Pairing enters only the targets, never a realized future input.
        action_width = net.action_size
        count_pooled = len(pooled_target)
        reference_options_tensor = torch.as_tensor(reference_options, dtype=torch.long, device=device)[root_ids]
        reference_programs = (reference_options_tensor[:, :1]//(10**torch.arange(defenders, device=device))) % 10
        if count_pooled > len(rows):
            reference_programs = torch.cat((reference_programs,
                torch.zeros((count_pooled-len(rows), defenders), dtype=torch.long, device=device)))
            reference_options_tensor = torch.cat((reference_options_tensor,
                torch.zeros((count_pooled-len(rows), 3), dtype=torch.long, device=device)))
        reference_actions = net.action_features(
            reference_programs, reference_options_tensor[:, 1:],
            pooled_features[:, net.state_size-6*defenders:net.state_size].reshape(-1, defenders, 6))
        pooled_reference = torch.cat((pooled_features[:, :-action_width], reference_actions), -1)
        factual_values = np.asarray(factual_targets, np.float32)
        if continuation is not None:
            factual_values = np.concatenate((factual_values, continuation[2].detach().cpu().numpy()))
        paired_factual_target = torch.as_tensor(factual_values, device=device)
        changed = ((programs != reference_programs[:len(rows)]).any(-1) if policy_neighborhood
                   else net.active_vessels(programs).any(-1))
        paired_active = torch.cat((changed,
            torch.zeros(count_pooled-len(codes), dtype=torch.bool, device=device)))
        paired_target_sha256 = hashlib.sha256(factual_values.tobytes()+
            (pooled_target-paired_factual_target).detach().cpu().numpy().tobytes()).hexdigest()
    def draw(count, size, probability):
        return rng.integers(count, size=size) if probability is None else rng.choice(count, size=size, p=probability)
    factual_reference_ids = factual_reference_probability = None
    factual_reference_draws = hashlib.sha256()
    if conditional and unique_factual_supervision:
        # One realized factual future can be shared by many candidate labels.
        # Its factual likelihood is counted once, independent of how many
        # candidate branches happened to be measured on that same future.
        measured_ids = np.flatnonzero(rows[:, 1] == 0)
        if len(set(rows[measured_ids, 0])) != len(roots):
            raise ValueError('Unique factual supervision requires an actual factual observation at every root.')
        factual_reference_ids = np.r_[measured_ids, np.arange(len(rows), len(pooled_target))]
        probability = root_sampling_probabilities(rows[measured_ids, 0], root_sampling_fraction, root_sampling_weights,
            np.asarray(weights, np.float64)[measured_ids] if future_weighted_sampling else None)
        if probability is None:
            probability = np.full(len(measured_ids), 1./len(measured_ids))
        # Keep the previous paired-root versus trajectory-bank mass; only
        # remove repetitions caused by unequal candidate coverage per future.
        factual_reference_probability = np.r_[probability*len(rows), factual_mass]/len(pooled_target)
        factual_reference_rng = np.random.default_rng(np.random.SeedSequence([seed, 840014]))
    contrast_groups = contrast_mask = contrast_probability = None
    contrast_sampling_mass = 0.
    contrast_draws = hashlib.sha256()
    contrast_target_sha256 = contrast_sampling_sha256 = None
    if conditional and conditional_objective == 'paired_catalog_contrast':
        # Keep future, clock and every other vessel's plan fixed. Outcome
        # pairing never turns cross-vessel or cross-duration changes into a
        # local intervention edge; factual references retain absolute labels.
        groups = single_vessel_contrast_groups(rows, common_future_ids, defenders)
        if not groups:
            raise ValueError('Catalog contrasts need a measured single-vessel edge at the same clock.')
        sizes = np.asarray([len(indices) for indices in groups], np.int64)
        padded = np.zeros((len(groups), int(sizes.max())), np.int64)
        valid = np.arange(padded.shape[1])[None] < sizes[:, None]
        for index, indices in enumerate(groups):
            padded[index, :len(indices)] = indices
        masses = (np.full(len(pooled_target), 1./len(pooled_target))
                  if pooled_probability is None else pooled_probability)
        membership = np.zeros(len(rows), np.int64)
        for indices in groups:
            membership[indices] += 1
        group_mass = np.asarray([(masses[indices]/membership[indices]).sum() for indices in groups])
        contrast_sampling_mass = float(group_mass.sum())
        contrast_probability = group_mass/contrast_sampling_mass
        contrast_groups = torch.as_tensor(padded, device=device)
        contrast_mask = torch.as_tensor(valid, device=device)
        contrast_rng = np.random.default_rng(np.random.SeedSequence([seed, 840015]))
        contrast_target_sha256 = hashlib.sha256(
            np.asarray(common_future_ids, np.int64).tobytes()+padded.tobytes()+valid.tobytes()+
            pooled_target[:len(rows)].detach().cpu().numpy().tobytes()).hexdigest()
        contrast_sampling_sha256 = hashlib.sha256(contrast_probability.tobytes()).hexdigest()
    catalog_draws = hashlib.sha256()
    onpolicy_draws = hashlib.sha256()
    if onpolicy is not None:
        onpolicy_rng = np.random.default_rng(np.random.SeedSequence([seed, 84272]))
        onpolicy_programs = (onpolicy[2][:, :1]//(10**torch.arange(defenders, device=device))) % 10
    if conservative_catalog_weight:
        from local_intervention_value import conservative_catalog_loss
        catalog_rng = np.random.default_rng(np.random.SeedSequence([seed, 840013]))
        catalog = torch.as_tensor(options, dtype=torch.long, device=device)
        lookup = {option: index for index, option in enumerate(options)}
        observed = [sorted({lookup[tuple(option)] for option in root['options']}) for root in roots]
        maximum = max(map(len, observed))
        observed_ids = torch.tensor([ids+[0]*(maximum-len(ids)) for ids in observed], device=device)
        observed_mask = torch.arange(maximum, device=device)[None] < torch.tensor(
            list(map(len, observed)), device=device)[:, None]
    with torch.enable_grad():
        for step in range(updates):
            if factual_pretraining_updates and step == factual_pretraining_updates:
                pretrained_state_sha256 = factual_state_digest()
                for name, parameter in net.named_parameters():
                    parameter.requires_grad_(name.startswith('layers.'))
                optimizer = torch.optim.AdamW(net.layers.parameters(), lr=learning_rate, weight_decay=.005)
                schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, updates-factual_pretraining_updates, eta_min=learning_rate/10.)
            if step < factual_pretraining_updates:
                selected = pretraining_rng.choice(pretraining_ids, size=192, p=pretraining_probability)
                pretraining_draws.update(selected.astype(np.int64).tobytes())
                selected = torch.as_tensor(selected, device=device)
                # Only measured factual outcomes train the state encoder.
                # The action response head is absent from this computation.
                logits = net.encode_fleet_state(pooled_features[selected, :net.state_size])[2][:, 0]
                loss = F.binary_cross_entropy_with_logits(logits, pooled_target[selected])
            elif uniform_sampling:
                index = torch.as_tensor(draw(len(pooled_target), 192, pooled_probability), device=device)
                raw = net.forward_features(pooled_features[index])
                if shared_potential:
                    # Both values come from exactly the ordinary scalar network.
                    # Only the common-future difference adds supervision; the
                    # absolute labels keep precisely their ordinary pooled mass.
                    loss = F.binary_cross_entropy_with_logits(raw[:, 0], pooled_target[index])
                    if conditional_objective == 'paired_catalog_contrast':
                        selected = contrast_rng.choice(len(contrast_groups), size=8, p=contrast_probability)
                        contrast_draws.update(selected.astype(np.int64).tobytes())
                        chosen = torch.as_tensor(selected, device=device)
                        group_indices = contrast_groups[chosen]
                        predictions = net.forward_features(pooled_features[group_indices.flatten()])[:, 0]
                        predictions = predictions.sigmoid().reshape_as(group_indices)
                        loss += paired_difference_weight*contrast_sampling_mass*common_future_contrast_loss(
                            predictions, pooled_target[group_indices], contrast_mask[chosen])
                    else:
                        factual_logits = net.forward_features(pooled_reference[index])[:, 0]
                        gain = raw[:, 0].sigmoid()-factual_logits.sigmoid()
                        difference = pooled_target[index]-paired_factual_target[index]
                        if conditional_objective == 'paired_ordering':
                            # Concordant pairs still supply their absolute labels.
                            # Only discordant common futures identify a direction;
                            # their auxiliary loss is not an effect probability.
                            contrast = raw[:, 0]-factual_logits
                            pair_error = F.softplus(-difference*contrast)*difference.abs()
                        else:
                            pair_error = (gain-difference).square()
                        loss += paired_difference_weight*(pair_error*paired_active[index]).mean()
                elif conditional:
                    from local_intervention_value import paired_difference_loss
                    factual_logits = net.forward_features(pooled_reference[index])[:, 0]
                    active = paired_active[index]
                    if conditional_objective == 'joint_likelihood':
                        if unique_factual_supervision:
                            rescue_error = F.binary_cross_entropy_with_logits(raw[:, 1], pooled_target[index], reduction='none')
                            harm_error = F.binary_cross_entropy_with_logits(raw[:, 2], 1.-pooled_target[index], reduction='none')
                            loss = (active*torch.where(paired_factual_target[index] > .5,
                                                      harm_error, rescue_error)).mean()
                            selected = factual_reference_rng.choice(factual_reference_ids, size=192,
                                                                    p=factual_reference_probability)
                            factual_reference_draws.update(selected.astype(np.int64).tobytes())
                            selected = torch.as_tensor(selected, device=device)
                            reference_logits = net.forward_features(pooled_reference[selected])[:, 0]
                            loss += F.binary_cross_entropy_with_logits(reference_logits, paired_factual_target[selected])
                        else:
                            loss = net.joint_outcome_loss(factual_logits, raw, paired_factual_target[index],
                                                         pooled_target[index], active)
                    else:
                        factual_probability = factual_logits.sigmoid()
                        gain = net.signed_gain(raw, active,
                            factual_probability if conditional_objective == 'factual_strata' else None,
                            event_hierarchy=conditional_objective == 'event_hierarchy')
                        estimate = (factual_logits.sigmoid()+gain).clamp(1e-6, 1.-1e-6)
                        loss = F.binary_cross_entropy(estimate, pooled_target[index])
                        loss += F.binary_cross_entropy_with_logits(factual_logits, paired_factual_target[index])
                        difference = pooled_target[index]-paired_factual_target[index]
                        if conditional_objective == 'factual_strata':
                            # The realized factual outcome selects only its training
                            # stratum. Deployment uses the state-only factual prediction.
                            rescue_error = F.binary_cross_entropy_with_logits(raw[:, 1], pooled_target[index], reduction='none')
                            harm_error = F.binary_cross_entropy_with_logits(raw[:, 2], 1.-pooled_target[index], reduction='none')
                            event_error = torch.where(paired_factual_target[index] > .5, harm_error, rescue_error)
                        else:
                            event_error = paired_difference_loss(net.event_probabilities(raw,
                                conditional_objective == 'event_hierarchy'), difference, value_target)
                        # All neutral and harmful futures keep their observed mass.
                        # Factual bank rows have exactly zero intervention effect.
                        loss += paired_difference_weight*(event_error*active).mean()
                else:
                    logits = raw.squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, pooled_target[index])
            else:
                index = torch.as_tensor(draw(len(rows), 128, row_probability), device=device)
                logits = net(context[root_ids[index]], controls[root_ids[index]], programs[index], timing[index])
                loss = (F.binary_cross_entropy_with_logits(logits, target[index], reduction='none') *
                        importance[index]).sum()/importance[index].sum()
                if continuation is not None:
                    index = torch.as_tensor(rng.integers(len(continuation[2]), size=64), device=device)
                    factual_logits = net(continuation[0][index], continuation[1][index],
                        torch.zeros((len(index), defenders), dtype=torch.long, device=device),
                        torch.zeros((len(index), 2), device=device))
                    loss += factual_bank_weight*F.binary_cross_entropy_with_logits(
                        factual_logits, continuation[2][index])
            if conservative_catalog_weight:
                selected = (catalog_rng.integers(len(roots), size=conservative_catalog_roots)
                    if root_prior is None else
                    catalog_rng.choice(len(roots), size=conservative_catalog_roots, p=root_prior))
                proposed = catalog_rng.integers(len(options),
                    size=(conservative_catalog_roots, conservative_catalog_samples))
                catalog_draws.update(selected.astype(np.int64).tobytes())
                catalog_draws.update(proposed.astype(np.int64).tobytes())
                selected = torch.as_tensor(selected, device=device)
                query = torch.cat((torch.as_tensor(proposed, device=device), observed_ids[selected]), -1)
                root_index = selected[:, None].expand_as(query).reshape(-1)
                actions = catalog[query].reshape(-1, 3)
                digits = (actions[:, :1]//(10**torch.arange(defenders, device=device))) % 10
                raw = net(context[root_index], controls[root_index], digits, actions[:, 1:])
                if conditional and not shared_potential:
                    factual = net(context[selected], controls[selected],
                        torch.zeros((len(selected), defenders), dtype=torch.long, device=device),
                        torch.zeros((len(selected), 2), dtype=torch.long, device=device))[:, 0].sigmoid()
                    reference = factual[:, None].expand_as(query).reshape(-1)
                    gain = net.signed_gain(raw, net.active_vessels(digits).any(-1),
                        reference if conditional_objective in ('factual_strata', 'joint_likelihood') else None,
                        event_hierarchy=conditional_objective == 'event_hierarchy')
                    values = reference+gain
                else:
                    values = raw.sigmoid()
                values = values.reshape_as(query)
                mask = observed_mask[selected]
                observed_mean = (values[:, conservative_catalog_samples:]*mask).sum(-1)/mask.sum(-1)
                loss += conservative_catalog_weight*conservative_catalog_loss(
                    values[:, :conservative_catalog_samples], observed_mean,
                    conservative_catalog_temperature).mean()
            if onpolicy is not None:
                selected = onpolicy_rng.choice(len(onpolicy[3]), size=192, p=onpolicy_probability)
                onpolicy_draws.update(selected.astype(np.int64).tobytes())
                selected = torch.as_tensor(selected, device=device)
                observed_context, observed_controls = onpolicy[0][selected], onpolicy[1][selected]
                observed_programs = onpolicy_programs[selected]
                observed_raw = net(observed_context, observed_controls, observed_programs, onpolicy[2][selected, 1:])
                if conditional and not shared_potential:
                    factual_raw = net(observed_context, observed_controls, torch.zeros_like(observed_programs),
                        torch.zeros((len(selected), 2), device=device))[:, 0]
                    factual_probability = factual_raw.sigmoid()
                    observed_gain = net.signed_gain(observed_raw, net.active_vessels(observed_programs).any(-1),
                        factual_probability if conditional_objective in ('factual_strata', 'joint_likelihood') else None,
                        event_hierarchy=conditional_objective == 'event_hierarchy')
                    observed_loss = F.binary_cross_entropy((factual_probability+observed_gain).clamp(1e-6, 1.-1e-6),
                        onpolicy[3][selected])
                else:
                    observed_loss = F.binary_cross_entropy_with_logits(observed_raw, onpolicy[3][selected])
                # These are actual action returns. No factual or paired-effect
                # label is manufactured for an unobserved alternative.
                loss += onpolicy_bank_weight*observed_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            schedule.step()
            if step+1 in snapshot_updates:
                snapshots[step+1] = {key:value.detach().cpu().clone() for key,value in net.state_dict().items()}
            if (step+1) % 1000 == 0 or step+1 == updates:
                print(dict(phase='paired_dense_return_fit' if conditional else 'plain_return_fit', update=step+1, updates=updates,
                           parameters=count(width), loss=float(loss.detach())), flush=True)
    if factual_pretraining_updates and factual_state_digest() != pretrained_state_sha256:
        raise RuntimeError('Intervention fitting changed the frozen factual representation.')
    model = copy.deepcopy(data['model'])
    model['successor_task_value'] = dict(kind='paired_mlp_return' if conditional else 'plain_mlp_return', defenders=defenders, width=width,
        identity_conditioned=identity_conditioned, action_encoding=action_encoding, velocity_reference=velocity_reference,
        state={key:value.detach().cpu() for key, value in net.state_dict().items()},
        options=options, decision_period=5, inference_device=device,
        isolate_factual_start=isolate_factual_start, intervention_budget=intervention_budget,
        feedback_relative_hold=feedback_relative_hold, feedback_relative_scale=feedback_relative_scale,
        gain_margin=float(gain_margin))
    if state_baseline:
        model['successor_task_value']['state_baseline'] = True
    if representation != 'dense':
        model['successor_task_value']['representation'] = representation
    if not peer_decision_conditioned:
        model['successor_task_value']['peer_decision_conditioned'] = False
    if policy_neighborhood:
        model['successor_task_value'].update(policy_neighborhood=True,
            policy_reference=copy.deepcopy(data['policy_reference']))
        recurrent_decision_specification(model['successor_task_value'])
    if conditional:
        model['successor_task_value']['learning_objective'] = 'conditional'
        model['successor_task_value']['conditional_objective'] = conditional_objective
    removal = tuple(model.get('mechanism_ablation', (False, False, False)))+(False,)*3
    model['mechanism_ablation'] = (removal[0], False, removal[2])
    model.update(gate_period=1, physical_interventions=True, fixed_intervention_clock=True,
                 success_objective=value_target == 'success')
    factual_labels = 0 if continuation is None else len(continuation[2])
    onpolicy_labels = 0 if onpolicy is None else len(onpolicy[3])
    model['successor_task_fit'] = dict(ordinary=not conditional,
        objective=('joint likelihood of factual success and same-future intervention outcome; no duplicate marginal loss'
                   if conditional_objective == 'joint_likelihood' else
                   'absolute success and same-future change likelihood with rescue direction conditional on change'
                   if conditional and conditional_objective == 'event_hierarchy' else
                   'absolute-outcome cross entropy plus discordant common-future logit ordering of one shared task potential'
                   if conditional and conditional_objective == 'paired_ordering' else
                   'absolute-outcome cross entropy plus same-clock single-vessel common-future contrasts of one shared task potential'
                   if conditional and conditional_objective == 'paired_catalog_contrast' else
                   'absolute-outcome cross entropy plus squared same-future differences of one shared task potential'
                   if shared_potential else
                   'absolute success and same-future rescue/harm likelihood conditional on factual outcomes'
                   if conditional and conditional_objective == 'factual_strata' else
                   'absolute returns and same-future signed intervention-return mass' if conditional else
                  'absolute normalized discounted-return binary cross entropy'
                   if value_target == 'discounted_return' else 'absolute-outcome binary cross entropy'),
        value_target=value_target, training_target_sha256=training_target_sha256,
        measured_terminal_target_sha256=hashlib.sha256(np.asarray(measured_targets, np.float32).tobytes()).hexdigest(),
        continuation_bootstrap_weight=0. if bootstrap_metadata is None else float(continuation_bootstrap_weight),
        continuation_bootstrap=bootstrap_metadata,
        return_gamma=.995 if value_target == 'discounted_return' else None,
        return_tick_cost=.001 if value_target == 'discounted_return' else None,
        return_normalization='(discounted task return + 1.001) / 2; common root shaping offset omitted'
            if value_target == 'discounted_return' else None,
        isolate_factual_start=isolate_factual_start,
        comparison_reference=('actual frozen-policy option at each root' if policy_neighborhood else 'zero intervention'),
        feedback_relative_hold=feedback_relative_hold, feedback_relative_scale=feedback_relative_scale,
        architecture=('public relative-motion message passing with fleet-conditioned quadratic action response and shared state baseline'
                      if representation == 'fleet_graph_quadratic' else
                      'fleet-conditioned quadratic local response in feedback offsets and clipped physical command changes; shared state baseline'
                      if representation == 'fleet_quadratic' else
                      'shared per-vessel state and physical-action attention; whole-plan pooled state baseline and gated residual'
                      if representation == 'fleet_attention' else
                      'separate state baseline and action residual, each three ReLU hidden layers, zero residual for inactive plans'
                      if state_baseline else 'single dense MLP, three ReLU hidden layers, factual and paired-event outputs'
                      if conditional and not shared_potential else 'single dense MLP, three ReLU hidden layers'), width=width,
        state_baseline=bool(state_baseline), representation=representation,
        peer_decision_conditioned=bool(peer_decision_conditioned),
        action_encoding=action_encoding, dense_input_width=input_size, velocity_reference=velocity_reference,
        physical_action_features=('active vessel, signed feedback offset, quarter-scaled clipped current command difference; linear and logarithmic timing'
                                 if feedback_relative_hold else
                                 'active vessel, coefficients, quarter-scaled current command difference; linear and logarithmic timing'
                                 if action_encoding == 'physical_delta' else None),
        trainable_parameters=count(width), reference_parameter_budget=parameter_budget,
        updates=updates, member_updates=[updates], seed=seed, roots=len(roots),
        factual_pretraining_updates=factual_pretraining_updates,
        member_factual_pretraining_updates=[factual_pretraining_updates],
        total_factual_pretraining_updates=factual_pretraining_updates,
        effect_updates=updates-factual_pretraining_updates,
        factual_pretraining_observations=0 if pretraining_ids is None else len(pretraining_ids),
        factual_pretraining_draw_sha256=None if pretraining_ids is None else pretraining_draws.hexdigest(),
        factual_pretrained_state_sha256=pretrained_state_sha256,
        factual_frozen_state_sha256=None if pretraining_ids is None else factual_state_digest(),
        factual_encoder_frozen_after_pretraining=bool(factual_pretraining_updates),
        effect_trainable_parameters=(sum(parameter.numel() for parameter in net.layers.parameters())
                                     if factual_pretraining_updates else count(width)),
        initialization_parameter_sha256=initialization_sha256,
        initialization_updates=initialization_updates,
        total_parameter_updates=initialization_updates+updates,
        initialization_observation_sha256=(None if initial_model is None else
            initial_model['successor_task_fit']['training_observation_sha256']),
        learning_rate_schedule=(f'cosine {learning_rate:g} to {learning_rate/10.:g}'+
            (f'; reset after {factual_pretraining_updates} factual updates and freeze encoder' if factual_pretraining_updates else '')),
        unique_terminal_observations=data.get('measured_terminal_observations', len(rows)),
        supervised_node_labels=len(rows)+factual_labels+onpolicy_labels, additional_factual_state_labels=factual_labels,
        additional_onpolicy_action_labels=onpolicy_labels,
        onpolicy_action_trajectories=0 if onpolicy_bank is None else len(onpolicy_bank['records']),
        onpolicy_action_draw_sha256=None if onpolicy is None else onpolicy_draws.hexdigest(),
        onpolicy_bank_weight=0. if onpolicy is None else float(onpolicy_bank_weight),
        onpolicy_sampling=(None if onpolicy is None else
            'specified record masses, then uniform five-tick decisions' if onpolicy_record_weights is not None else
            'uniform trajectories, then uniform recorded five-tick decisions'),
        onpolicy_record_sampling_sha256=(None if onpolicy_record_weights is None else
            hashlib.sha256(onpolicy_record_weights.tobytes()).hexdigest()),
        onpolicy_reconstructed_action_labels=reconstructed_action_labels,
        onpolicy_action_reconstruction_policy_sha256=(onpolicy_bank['reconstruction_policy_sha256']
            if reconstructed_action_labels else None),
        onpolicy_minibatch=0 if onpolicy is None else 192,
        onpolicy_action_supervision=None if onpolicy is None else 'absolute observed-action return only; no unobserved factual or difference labels',
        training_observation_sha256=observation_sha, training_data=data.get('training_data'),
        factual_bank_training_data=None if factual_bank is None else factual_bank.get('training_data'),
        predecessor_sha256=data['predecessor_sha256'], inherited_stages=initial_model is not None,
        initialization_predecessor_sha256=(None if initial_model is None else
            initial_model['successor_task_fit']['predecessor_sha256']),
        predecessor_transfer=(initial_model is not None and
            initial_model['successor_task_fit']['predecessor_sha256'] != data['predecessor_sha256']),
        initialization_optimizer_state_retained=False,
        paired_local_observations=int((codes > 0).sum()) if conditional else 0,
        factual_comparisons=int((codes > 0).sum()) if conditional else 0,
        paired_difference_weight=float(paired_difference_weight), paired_training_target_sha256=paired_target_sha256,
        paired_catalog_groups=0 if contrast_groups is None else len(contrast_groups),
        paired_catalog_minibatch_groups=0 if contrast_groups is None else 8,
        paired_catalog_sampling_mass=contrast_sampling_mass,
        paired_catalog_grouping=None if contrast_groups is None else 'root, future, clock, subject, unchanged peer plans',
        paired_catalog_target_sha256=contrast_target_sha256,
        paired_catalog_sampling_sha256=contrast_sampling_sha256,
        paired_catalog_draw_sha256=None if contrast_groups is None else contrast_draws.hexdigest(),
        paired_event_difference=conditional and conditional_objective in ('marginal_events', 'event_hierarchy'),
        conditional_objective=conditional_objective if conditional else None, future_public_states_used=False,
        common_motion_response_features=False,
        joint_factual_calibration=False, local_reconciliation=False, local_rank_weight=0.,
        public_vessel_identity=identity_conditioned,
        sampling=(('root-prior pooled absolute returns' if root_prior is not None else
                   'root-balanced pooled absolute returns' if root_sampling_fraction else 'uniform pooled absolute returns')
                  if uniform_sampling else 'matched weighted observations'),
        root_sampling_fraction=root_sampling_fraction,
        future_weighted_sampling=future_weighted_sampling,
        future_sampling_weight_sha256=(hashlib.sha256(np.asarray(weights, np.float64).tobytes()).hexdigest()
            if future_weighted_sampling else None),
        paired_row_sampling_probability_sha256=(None if row_probability is None else
            hashlib.sha256(row_probability.tobytes()).hexdigest()),
        root_sampling_prior=None if root_prior is None else root_prior.tolist(),
        root_sampling_prior_sha256=(None if root_prior is None else hashlib.sha256(root_prior.tobytes()).hexdigest()),
        factual_sampling_prior=None if factual_prior is None else factual_prior.tolist(),
        factual_sampling_prior_sha256=(None if factual_prior is None else
            hashlib.sha256(factual_prior.tobytes()).hexdigest()),
        batch_size=192 if uniform_sampling else 128,
        development_future_weight=1. if uniform_sampling else data.get('development_future_weight', 1.),
        separate_factual_objective=(conditional and not shared_potential) or (not uniform_sampling and factual_bank is not None),
        factual_bank_weight=(factual_bank_weight if factual_bank is not None and not uniform_sampling else 0.),
        candidate_catalog=('shared complete programs, including unobserved prior candidates'
                           if extra_options else 'all sampled complete programs, including negative outcomes'),
        additional_unobserved_candidates=len(extra_options-sampled_options-{(0, 0, 0)}),
        unique_factual_supervision=conditional and unique_factual_supervision,
        unique_factual_observations=None if factual_reference_ids is None else len(factual_reference_ids),
        unique_factual_target_sha256=(None if factual_reference_ids is None else hashlib.sha256(
            paired_factual_target[factual_reference_ids].detach().cpu().numpy().tobytes()).hexdigest()),
        unique_factual_draw_sha256=factual_reference_draws.hexdigest() if factual_reference_ids is not None else None,
        conservative_catalog_weight=conservative_catalog_weight,
        conservative_catalog_temperature=conservative_catalog_temperature,
        conservative_catalog_samples=conservative_catalog_samples if conservative_catalog_weight else 0,
        conservative_catalog_roots=conservative_catalog_roots if conservative_catalog_weight else 0,
        conservative_training_draw_sha256=catalog_draws.hexdigest() if conservative_catalog_weight else None,
        conservative_catalog_reference=('uniform unique measured choices; no synthetic outcome labels'
            if conservative_catalog_weight else None),
        simulation_steps=data.get('simulation_steps', 0)+(0 if factual_bank is None else factual_bank['simulation_steps'])
            +(0 if onpolicy_bank is None else onpolicy_bank['simulation_steps']),
        last_loss=float(loss.detach()))
    fit = model['successor_task_fit']
    if bootstrap_metadata is not None:
        fit['objective'] = ('measured terminal outcomes mixed with scene-cross-fitted continuation success estimates; '
                            +fit['objective'])
        fit['successor_states_used_for_training_targets'] = True
        fit['successor_states_added_to_query_inputs'] = False
    inherited = []
    if initial_model is not None:
        previous_fit = initial_model['successor_task_fit']
        inherited = copy.deepcopy(previous_fit.get('training_stages', [dict(
            predecessor_sha256=previous_fit['predecessor_sha256'],
            updates=initialization_updates,
            training_observation_sha256=previous_fit['training_observation_sha256'],
            training_target_sha256=previous_fit.get('training_target_sha256'),
            paired_training_target_sha256=previous_fit.get('paired_training_target_sha256'),
            training_data=previous_fit.get('training_data'),
            historical_stage_detail_available=False)]))
    fit['training_stages'] = inherited+[dict(
        predecessor_sha256=data['predecessor_sha256'], updates=updates, seed=seed,
        training_observation_sha256=observation_sha, training_target_sha256=training_target_sha256,
        paired_training_target_sha256=paired_target_sha256,
        training_data=data.get('training_data'), factual_bank_training_data=fit['factual_bank_training_data'],
        roots=len(roots), factual_state_labels=factual_labels, onpolicy_action_labels=onpolicy_labels,
        continuation_bootstrap=bootstrap_metadata,
        learning_rate_schedule=fit['learning_rate_schedule'], optimizer_state_retained=False)]
    if sum(stage['updates'] for stage in fit['training_stages']) != fit['total_parameter_updates']:
        raise ValueError('Dense continuation history must account for every previous optimizer update.')
    if snapshots:
        model['learning_snapshots'] = dict(kind='actual optimizer states', final_updates=updates,
            states=snapshots, calibration=('none; direct paired-effect argmax' if conditional else
                                          'none; ordinary absolute-return argmax'))
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, output)
    return model


def fit_paired_mlp_return(data, output, *, difference_weight=4., **kwargs):
    """Fit the matched dense representation directly on common-future effects."""
    if not np.isfinite(difference_weight) or difference_weight <= 0.:
        raise ValueError('Dense conditional fitting requires a positive paired difference weight.')
    return fit_plain_return(data, output, paired_difference_weight=difference_weight, **kwargs)
