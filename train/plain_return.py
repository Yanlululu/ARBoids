"""Independent absolute-return regression over the common action catalog."""
import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def root_sampling_probabilities(root_ids, fraction=0., root_weights=None):
    """Mix observation-uniform sampling with positive mass per measured state.

    Optional weights follow sorted unique root ids. No state can be removed.
    """
    if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError('Root sampling fraction must lie between zero and one.')
    root_ids = np.asarray(root_ids)
    if root_ids.ndim != 1 or not len(root_ids):
        raise ValueError('Root sampling requires a nonempty one-dimensional population.')
    if fraction == 0. and root_weights is None:
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


class PlainReturnNetwork(nn.Module):
    def __init__(self, defenders, width, identity_conditioned=False, output_width=1,
                 action_encoding='categorical', feedback_relative_scale=None):
        super().__init__()
        if action_encoding not in ('categorical', 'physical_delta'):
            raise ValueError('Unknown dense intervention action representation.')
        self.defenders = defenders
        self.width = width
        self.identity_conditioned = identity_conditioned
        self.action_encoding = action_encoding
        if feedback_relative_scale is not None and (action_encoding != 'physical_delta' or
                not np.isfinite(feedback_relative_scale) or not 0. < feedback_relative_scale <= 1.):
            raise ValueError('Relative dense features need a finite scale in (0,1] and physical commands.')
        self.feedback_relative_scale = feedback_relative_scale
        self.state_size = (14 + identity_conditioned) * defenders + 5
        self.action_size = 10*defenders+2 if action_encoding == 'categorical' else 5*defenders+4
        self.layers = nn.Sequential(
            nn.Linear(self.state_size+self.action_size, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, output_width))

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
        return self.layers(features).squeeze(-1)

    def forward_catalog(self, context, controls, programs, timing, chunk_size=1024):
        # Factor the first affine layer only; this equals ordinary dense MLP
        # evaluation of every state/action pair without duplicating its input.
        first = self.layers[0]
        state = torch.cat((context, controls.flatten(1)), -1)
        state_hidden = F.linear(state, first.weight[:, :self.state_size], first.bias)
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
            output.append(value.squeeze(-1))
        return torch.cat(output, 1)


class PairedReturnNetwork(PlainReturnNetwork):
    """The common dense state/action trunk with factual and signed-event outputs."""
    def __init__(self, defenders, width, identity_conditioned=False, action_encoding='categorical',
                 feedback_relative_scale=None):
        super().__init__(defenders, width, identity_conditioned, output_width=3,
                         action_encoding=action_encoding, feedback_relative_scale=feedback_relative_scale)

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
                 ensemble_gain_penalty=0., feedback_relative_hold=False, feedback_relative_scale=None):
        super().__init__()
        if velocity_reference not in ('ground', 'water'):
            raise ValueError('Dense values require an observed ground or water velocity reference.')
        self.velocity_reference = velocity_reference
        if (feedback_relative_hold != (feedback_relative_scale is not None)
                or feedback_relative_hold and not isolate_factual_start):
            raise ValueError('Dense relative execution requires its measured scale and isolated factual start.')
        self.feedback_relative_hold = bool(feedback_relative_hold)
        self.feedback_relative_scale = feedback_relative_scale
        self.network = self.network_type(defenders, width, identity_conditioned,
            action_encoding=action_encoding, feedback_relative_scale=feedback_relative_scale)
        self.network.load_state_dict(state)
        self.ensemble = nn.ModuleList()
        for member_state in ensemble_states:
            member = self.network_type(defenders, width, identity_conditioned,
                action_encoding=action_encoding, feedback_relative_scale=feedback_relative_scale)
            member.load_state_dict(member_state)
            self.ensemble.append(member)
        self.register_buffer('options', torch.as_tensor(options, dtype=torch.long))
        self.register_buffer('programs',
            (self.options[:, :1] // (10 ** torch.arange(defenders))) % 10)
        self.isolate_factual_start = bool(isolate_factual_start)
        if self.isolate_factual_start and bool(((self.programs > 0).sum(-1) > 1).any()):
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

    def catalog_estimates(self, context, controls):
        logits = self.network.forward_catalog(context, controls, self.programs, self.options[:, 1:])
        if not self.ensemble:
            return logits, logits.sigmoid()
        return self.aggregate_estimates([logits.sigmoid()]+[
            member.forward_catalog(context, controls, self.programs, self.options[:, 1:]).sigmoid()
            for member in self.ensemble])

    def aggregate_estimates(self, estimates):
        estimates = torch.stack(estimates)
        values = estimates.mean(0)
        # Penalize disagreement of paired gains, not absolute value offsets.
        # This is a decision criterion, not a calibrated confidence interval
        # or a correction to the raw neural success probabilities.
        gains = estimates-estimates[:, :, :1]
        ranking = values-self.ensemble_gain_penalty*gains.std(0, unbiased=False)
        return ranking, values

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
        if order.shape[1] != self.network.defenders:
            raise ValueError('The plain MLP and deployed fleet must have the same size.')
        controls = packet['_plain_return_controls'].gather(
            1, order[..., None].expand(-1, -1, 6))
        device = self.options.device
        context, controls, order, action, eligible = [
            item.to(device) for item in (context, controls, order, action, eligible)]
        ranking, values = self.catalog_estimates(context, controls)
        choice = ranking.argmax(-1)
        accept = eligible & (choice != 0)
        gain = values.gather(1, choice[:, None]).flatten()-values[:, 0]
        decision_gain = (ranking.gather(1, choice[:, None]).flatten()-ranking[:, 0]
                         if self.ensemble_gain_penalty else gain)
        if self.gain_margin:
            accept &= decision_gain > self.gain_margin
        wait = None
        if self.wait_value is not None:
            wait_margin = self.gain_margin if getattr(self.wait_value, 'margin_adjusted', True) else 0.
            wait = self.wait_value(context, controls)
            accept &= decision_gain-wait_margin > wait
        if self.record_catalog:
            self.last_catalog = dict(options=self.options.detach().cpu(),
                factual_value=values[:, 0].detach().cpu(),
                predicted_gain=(values-values[:, :1]).detach().cpu(),
                score=(ranking if self.ensemble_gain_penalty else values).detach().cpu())
            if wait is not None:
                self.last_catalog['wait_value'] = wait.detach().cpu()
        if self.record_decisions:
            gain = values.gather(1, choice[:, None]).flatten()-values[:, 0]
            self.last_decision = dict(reason='catalog', factual_value=values[:, 0].detach().cpu(),
                option=self.options[choice].detach().cpu(), accepted=accept.detach().cpu(),
                predicted_gain=torch.where(accept, gain, torch.zeros_like(gain)).detach().cpu(),
                candidate_gain=gain.detach().cpu(),
                decision_gain=decision_gain.detach().cpu(),
                score=(ranking if self.ensemble_gain_penalty else values).gather(
                    1, choice[:, None]).flatten().detach().cpu())
            if wait is not None:
                self.last_decision['wait_value'] = wait.detach().cpu()
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
        if conditional_objective not in ('marginal_events', 'event_hierarchy', 'factual_strata', 'shared_potential', 'joint_likelihood'):
            raise ValueError('Unknown paired dense conditional objective.')
        self.conditional_objective = conditional_objective
        if conditional_objective == 'shared_potential':
            self.network_type = PlainReturnNetwork
        super().__init__(*args, **kwargs)
        if not torch.equal(self.options[0], torch.zeros_like(self.options[0])):
            raise ValueError('The paired dense catalog must start with its factual reference.')
        self.ordinary = False

    def catalog_estimates(self, context, controls):
        estimates = []
        for member in [self.network, *self.ensemble]:
            raw = member.forward_catalog(context, controls, self.programs, self.options[:, 1:])
            if self.conditional_objective == 'shared_potential':
                estimates.append(raw.sigmoid())
                continue
            factual = raw[:, :1, 0].sigmoid()
            gain = member.signed_gain(raw, member.active_vessels(self.programs).any(-1)[None],
                factual if self.conditional_objective in ('factual_strata', 'joint_likelihood') else None,
                event_hierarchy=self.conditional_objective == 'event_hierarchy')
            estimates.append(factual+gain)
        # The common factual value cancels before ranking and margin checks.
        # Avoid clipping a beneficial contrast at the factual value boundary.
        return self.aggregate_estimates(estimates)


def fit_plain_return(data, output, *, factual_bank=None, updates=24000,
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
                     root_sampling_weights=None, factual_sampling_weights=None):
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
    if not isinstance(unique_factual_supervision, bool):
        raise ValueError('Unique factual supervision must be explicitly enabled or disabled.')
    if unique_factual_supervision and conditional and conditional_objective != 'joint_likelihood':
        raise ValueError('Unique factual supervision requires the joint outcome likelihood.')
    if (conditional_objective not in ('marginal_events', 'event_hierarchy', 'factual_strata', 'shared_potential', 'joint_likelihood')
            or conditional_objective in ('factual_strata', 'joint_likelihood') and (not conditional or value_target != 'success')):
        raise ValueError('Factual-stratum dense learning requires paired binary success outcomes.')
    if conditional_objective == 'event_hierarchy' and (not conditional or value_target != 'success'):
        raise ValueError('Change-direction learning requires paired binary success outcomes.')
    if conditional_objective == 'joint_likelihood' and paired_difference_weight != 1.:
        raise ValueError('A joint outcome likelihood requires unit weight for its conditional factor.')
    shared_potential = conditional and conditional_objective == 'shared_potential'
    if conditional and not uniform_sampling:
        raise ValueError('Dense conditional fitting requires pooled-observation sampling.')
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
    if not np.isfinite(gain_margin) or not 0. <= gain_margin <= 1.:
        raise ValueError('The fixed return-gain margin must lie between zero and one.')
    if updates <= 0:
        raise ValueError('Plain return fitting requires a positive optimizer budget.')
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
    digest = hashlib.sha256()
    if feedback_relative_hold:
        digest.update(repr(('feedback_relative_hold', float(feedback_relative_scale))).encode())
    rows, weights, factual_targets = [], [], []
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
            rows.append((root_id, *root['options'][q], label))
            weights.append(future_weights[future])
            if conditional:
                if tuple(root['options'][0]) != (0, 0, 0) or (0, future) not in root['outcomes']:
                    raise ValueError('Every dense conditional label requires its same-future factual outcome.')
                factual_targets.append(measured[0, future] if value_target == 'discounted_return' else
                                       float(root['outcomes'][0, future]['success']))
    continuation = None
    if factual_bank is not None:
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
    observation_sha = digest.hexdigest()
    if expected_observation_sha256 is not None and observation_sha != expected_observation_sha256:
        raise ValueError('The plain MLP must see exactly the matched observations.')
    rows = np.asarray(rows)
    row_probability = root_sampling_probabilities(rows[:, 0], root_sampling_fraction, root_sampling_weights)
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
    training_target_sha256 = target_digest.hexdigest()
    if expected_training_target_sha256 is not None and training_target_sha256 != expected_training_target_sha256:
        raise ValueError('The plain MLP must use exactly the matched measured return targets.')
    sampled_options = {tuple(option) for root in roots for option in root['options'] if option[0]}
    options = [(0, 0, 0)]+sorted((sampled_options | extra_options)-{(0, 0, 0)})
    if isolate_factual_start and any(sum(code//10**boat % 10 != 0 for boat in range(defenders)) > 1
                                    for code, _, _ in options):
        raise ValueError('Isolated plain-return learning requires a single-vessel candidate catalog.')
    importance = torch.as_tensor(np.asarray(weights), device=device)
    action_size = 10*defenders+2 if action_encoding == 'categorical' else 5*defenders+4
    input_size = (14+identity_conditioned)*defenders+5+action_size
    output_width = 3 if conditional and not shared_potential else 1
    count = lambda width: 2*width*width + (input_size+3+output_width)*width + output_width
    width = min(range(8, 2049), key=lambda value: abs(count(value)-parameter_budget))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network_type = PairedReturnNetwork if conditional and not shared_potential else PlainReturnNetwork
    net = network_type(defenders, width, identity_conditioned, action_encoding=action_encoding,
        feedback_relative_scale=feedback_relative_scale).to(device)
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
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=learning_rate/10.)
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
    pooled_reference = paired_factual_target = paired_active = None
    paired_target_sha256 = None
    if conditional:
        # Common-current-state inputs are identical to the ordinary learner.
        # Pairing enters only the targets, never a realized future input.
        action_width = net.action_size
        count_pooled = len(pooled_target)
        reference_actions = net.action_features(
            torch.zeros((count_pooled, defenders), dtype=torch.long, device=device),
            torch.zeros((count_pooled, 2), device=device),
            pooled_features[:, net.state_size-6*defenders:net.state_size].reshape(-1, defenders, 6))
        pooled_reference = torch.cat((pooled_features[:, :-action_width], reference_actions), -1)
        factual_values = np.asarray(factual_targets, np.float32)
        if continuation is not None:
            factual_values = np.concatenate((factual_values, continuation[2].detach().cpu().numpy()))
        paired_factual_target = torch.as_tensor(factual_values, device=device)
        paired_active = torch.cat((net.active_vessels(programs).any(-1),
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
        probability = root_sampling_probabilities(rows[measured_ids, 0], root_sampling_fraction, root_sampling_weights)
        if probability is None:
            probability = np.full(len(measured_ids), 1./len(measured_ids))
        # Keep the previous paired-root versus trajectory-bank mass; only
        # remove repetitions caused by unequal candidate coverage per future.
        factual_reference_probability = np.r_[probability*len(rows), factual_mass]/len(pooled_target)
        factual_reference_rng = np.random.default_rng(np.random.SeedSequence([seed, 840014]))
    catalog_draws = hashlib.sha256()
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
            if uniform_sampling:
                index = torch.as_tensor(draw(len(pooled_target), 192, pooled_probability), device=device)
                raw = net.layers(pooled_features[index])
                if shared_potential:
                    # Both values come from exactly the ordinary scalar network.
                    # Only the common-future difference adds supervision; the
                    # absolute labels keep precisely their ordinary pooled mass.
                    factual_logits = net.layers(pooled_reference[index])[:, 0]
                    gain = raw[:, 0].sigmoid()-factual_logits.sigmoid()
                    difference = pooled_target[index]-paired_factual_target[index]
                    loss = F.binary_cross_entropy_with_logits(raw[:, 0], pooled_target[index])
                    loss += paired_difference_weight*((gain-difference).square()*paired_active[index]).mean()
                elif conditional:
                    from local_intervention_value import paired_difference_loss
                    factual_logits = net.layers(pooled_reference[index])[:, 0]
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
                            reference_logits = net.layers(pooled_reference[selected])[:, 0]
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
    model = copy.deepcopy(data['model'])
    model['successor_task_value'] = dict(kind='paired_mlp_return' if conditional else 'plain_mlp_return', defenders=defenders, width=width,
        identity_conditioned=identity_conditioned, action_encoding=action_encoding, velocity_reference=velocity_reference,
        state={key:value.detach().cpu() for key, value in net.state_dict().items()},
        options=options, decision_period=5, inference_device=device,
        isolate_factual_start=isolate_factual_start, intervention_budget=intervention_budget,
        feedback_relative_hold=feedback_relative_hold, feedback_relative_scale=feedback_relative_scale,
        gain_margin=float(gain_margin))
    if conditional:
        model['successor_task_value']['learning_objective'] = 'conditional'
        model['successor_task_value']['conditional_objective'] = conditional_objective
    removal = tuple(model.get('mechanism_ablation', (False, False, False)))+(False,)*3
    model['mechanism_ablation'] = (removal[0], False, removal[2])
    model.update(gate_period=1, physical_interventions=True, fixed_intervention_clock=True,
                 success_objective=value_target == 'success')
    factual_labels = 0 if continuation is None else len(continuation[2])
    model['successor_task_fit'] = dict(ordinary=not conditional,
        objective=('joint likelihood of factual success and same-future intervention outcome; no duplicate marginal loss'
                   if conditional_objective == 'joint_likelihood' else
                   'absolute success and same-future change likelihood with rescue direction conditional on change'
                   if conditional and conditional_objective == 'event_hierarchy' else
                   'absolute-outcome cross entropy plus squared same-future differences of one shared task potential'
                   if shared_potential else
                   'absolute success and same-future rescue/harm likelihood conditional on factual outcomes'
                   if conditional and conditional_objective == 'factual_strata' else
                   'absolute returns and same-future signed intervention-return mass' if conditional else
                  'absolute normalized discounted-return binary cross entropy'
                   if value_target == 'discounted_return' else 'absolute-outcome binary cross entropy'),
        value_target=value_target, training_target_sha256=training_target_sha256,
        return_gamma=.995 if value_target == 'discounted_return' else None,
        return_tick_cost=.001 if value_target == 'discounted_return' else None,
        return_normalization='(discounted task return + 1.001) / 2; common root shaping offset omitted'
            if value_target == 'discounted_return' else None,
        isolate_factual_start=isolate_factual_start,
        feedback_relative_hold=feedback_relative_hold, feedback_relative_scale=feedback_relative_scale,
        architecture=('single dense MLP, three ReLU hidden layers, factual and paired-event outputs'
                      if conditional and not shared_potential else 'single dense MLP, three ReLU hidden layers'), width=width,
        action_encoding=action_encoding, dense_input_width=input_size, velocity_reference=velocity_reference,
        physical_action_features=('active vessel, signed feedback offset, quarter-scaled clipped current command difference; linear and logarithmic timing'
                                 if feedback_relative_hold else
                                 'active vessel, coefficients, quarter-scaled current command difference; linear and logarithmic timing'
                                 if action_encoding == 'physical_delta' else None),
        trainable_parameters=count(width), reference_parameter_budget=parameter_budget,
        updates=updates, member_updates=[updates], seed=seed, roots=len(roots),
        initialization_parameter_sha256=initialization_sha256,
        initialization_updates=initialization_updates,
        total_parameter_updates=initialization_updates+updates,
        initialization_observation_sha256=(None if initial_model is None else
            initial_model['successor_task_fit']['training_observation_sha256']),
        learning_rate_schedule=f'cosine {learning_rate:g} to {learning_rate/10.:g}',
        unique_terminal_observations=data.get('measured_terminal_observations', len(rows)),
        supervised_node_labels=len(rows)+factual_labels, additional_factual_state_labels=factual_labels,
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
        paired_event_difference=conditional and conditional_objective in ('marginal_events', 'event_hierarchy'),
        conditional_objective=conditional_objective if conditional else None, future_public_states_used=False,
        common_motion_response_features=False,
        joint_factual_calibration=False, local_reconciliation=False, local_rank_weight=0.,
        public_vessel_identity=identity_conditioned,
        sampling=(('root-prior pooled absolute returns' if root_prior is not None else
                   'root-balanced pooled absolute returns' if root_sampling_fraction else 'uniform pooled absolute returns')
                  if uniform_sampling else 'matched weighted observations'),
        root_sampling_fraction=root_sampling_fraction,
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
        simulation_steps=data.get('simulation_steps', 0)+(0 if factual_bank is None else factual_bank['simulation_steps']),
        last_loss=float(loss.detach()))
    fit = model['successor_task_fit']
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
        roots=len(roots), factual_state_labels=factual_labels,
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
