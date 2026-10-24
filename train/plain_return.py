"""Independent absolute-return regression over the common action catalog."""
import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def root_sampling_probabilities(root_ids, fraction=0.):
    """Mix observation-uniform sampling with equal mass per measured state."""
    if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError('Root sampling fraction must lie between zero and one.')
    root_ids = np.asarray(root_ids)
    if root_ids.ndim != 1 or not len(root_ids):
        raise ValueError('Root sampling requires a nonempty one-dimensional population.')
    if fraction == 0.:
        return None
    _, inverse, counts = np.unique(root_ids, return_inverse=True, return_counts=True)
    probability = (1-fraction)/len(root_ids)+fraction/(len(counts)*counts[inverse])
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
                 action_encoding='categorical'):
        super().__init__()
        if action_encoding not in ('categorical', 'physical_delta'):
            raise ValueError('Unknown dense intervention action representation.')
        self.defenders = defenders
        self.width = width
        self.identity_conditioned = identity_conditioned
        self.action_encoding = action_encoding
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
        active = programs > 0
        coefficients = controls.new_tensor([[0., 0.]]+
            [[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])
        preference = coefficients[programs]
        source, rule, factual = controls.split(2, -1)
        delta = torch.where(active[..., None],
            preference*source+(1.-preference)*rule-factual, torch.zeros_like(factual))
        # The coefficient remains visible when two current commands coincide:
        # a finite hold can still separate their later closed-loop commands.
        local = torch.cat((active[..., None].expand_as(delta[..., :1]),
            (preference*active[..., None]).expand_as(delta), .25*delta), -1).flatten(-2)
        temporal = torch.cat((timing.float()/400., timing.float().log1p()/np.log(401.)), -1)
        return torch.cat((local, temporal.expand(*local.shape[:-1], 4)), -1)

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
    def __init__(self, defenders, width, identity_conditioned=False, action_encoding='categorical'):
        super().__init__(defenders, width, identity_conditioned, output_width=3,
                         action_encoding=action_encoding)

    @staticmethod
    def event_probabilities(raw):
        return torch.cat((torch.zeros_like(raw[..., :1]), raw[..., 1:]), -1).softmax(-1)

    @classmethod
    def signed_gain(cls, raw, active):
        probabilities = cls.event_probabilities(raw)
        return (probabilities[..., 1]-probabilities[..., 2])*active.to(raw.dtype)


class PlainReturnDecision(nn.Module):
    """One MLP and argmax, optionally requiring a fixed predicted return gain."""
    network_type = PlainReturnNetwork

    def __init__(self, defenders, width, state, options, decision_period=5,
                 inference_device=None, intervention_budget=0, identity_conditioned=False,
                 gain_margin=0., isolate_factual_start=False, wait_value=None,
                 action_encoding='categorical'):
        super().__init__()
        self.network = self.network_type(defenders, width, identity_conditioned,
                                         action_encoding=action_encoding)
        self.network.load_state_dict(state)
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
        return logits, logits.sigmoid()

    @torch.no_grad()
    def forward(self, packet, action):
        from conditional_fusion_value import intervention_task_context
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
        context, order = intervention_task_context(packet, True, True)
        if self.network.identity_conditioned:
            from conditional_fusion_value import append_vessel_identity
            context = append_vessel_identity(context, order)
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
        if self.gain_margin:
            gain = values.gather(1, choice[:, None]).flatten()-values[:, 0]
            accept &= gain > self.gain_margin
        wait = None
        if self.wait_value is not None:
            gain = values.gather(1, choice[:, None]).flatten()-values[:, 0]
            wait_margin = self.gain_margin if getattr(self.wait_value, 'margin_adjusted', True) else 0.
            wait = self.wait_value(context, controls)
            accept &= gain-wait_margin > wait
        if self.record_catalog:
            self.last_catalog = dict(options=self.options.detach().cpu(),
                factual_value=values[:, 0].detach().cpu(),
                predicted_gain=(values-values[:, :1]).detach().cpu(),
                score=values.detach().cpu())
            if wait is not None:
                self.last_catalog['wait_value'] = wait.detach().cpu()
        if self.record_decisions:
            gain = values.gather(1, choice[:, None]).flatten()-values[:, 0]
            self.last_decision = dict(reason='catalog', factual_value=values[:, 0].detach().cpu(),
                option=self.options[choice].detach().cpu(), accepted=accept.detach().cpu(),
                predicted_gain=torch.where(accept, gain, torch.zeros_like(gain)).detach().cpu(),
                candidate_gain=gain.detach().cpu(),
                score=values.gather(1, choice[:, None]).flatten().detach().cpu())
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

    def __init__(self, *args, learning_objective='conditional', **kwargs):
        if learning_objective != 'conditional':
            raise ValueError('Paired dense returns require the conditional learning objective.')
        super().__init__(*args, **kwargs)
        if not torch.equal(self.options[0], torch.zeros_like(self.options[0])):
            raise ValueError('The paired dense catalog must start with its factual reference.')
        self.ordinary = False

    def catalog_estimates(self, context, controls):
        raw = self.network.forward_catalog(context, controls, self.programs, self.options[:, 1:])
        gain = self.network.signed_gain(raw, (self.programs > 0).any(-1)[None])
        # The common factual value cancels before ranking and margin checks.
        # Avoid clipping a beneficial contrast at the factual value boundary.
        values = raw[:, :1, 0].sigmoid()+gain
        return values, values


def fit_plain_return(data, output, *, factual_bank=None, updates=24000,
                     parameter_budget=145731, seed=42, device='cuda:0',
                     expected_observation_sha256=None, factual_bank_weight=1.,
                     uniform_sampling=True, initial_model=None, learning_rate=8e-4,
                     additional_candidate_options=(), identity_conditioned=False, snapshot_updates=(),
                     root_sampling_fraction=0., value_target='success',
                     expected_training_target_sha256=None, intervention_budget=0, gain_margin=0.,
                     paired_difference_weight=0., action_encoding='categorical'):
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
    if action_encoding not in ('categorical', 'physical_delta'):
        raise ValueError('Unknown dense intervention action representation.')
    if not np.isfinite(paired_difference_weight) or paired_difference_weight < 0.:
        raise ValueError('The paired difference weight must be finite and nonnegative.')
    conditional = paired_difference_weight > 0.
    if conditional and (not uniform_sampling or initial_model is not None):
        raise ValueError('Dense conditional fitting currently requires a fresh pooled-observation fit.')
    if value_target not in ('success', 'discounted_return'):
        raise ValueError('Choose success or the measured discounted task return.')
    if value_target == 'discounted_return' and data['model'].get('success_objective', False):
        raise ValueError('Discounted returns require the predecessor using the discounted task objective.')
    if data.get('feedback_relative_hold', False):
        raise ValueError('The plain-return comparator requires absolute finite interventions.')
    isolate_factual_start = bool(data.get('isolate_factual_start', False))
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
        require_plain_predecessor(initial_model, allow_plain_return=True)
        if initial_model.get('successor_task_value', {}).get('kind') != 'plain_mlp_return':
            raise ValueError('A plain MLP can resume only its own absolute-return parameters.')
    if uniform_sampling and factual_bank_weight != 1.:
        raise ValueError('Uniform return regression has no separate factual loss weight.')
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
    row_probability = root_sampling_probabilities(rows[:, 0], root_sampling_fraction)
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
    output_width = 3 if conditional else 1
    count = lambda width: 2*width*width + (input_size+3+output_width)*width + output_width
    width = min(range(8, 2049), key=lambda value: abs(count(value)-parameter_budget))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network_type = PairedReturnNetwork if conditional else PlainReturnNetwork
    net = network_type(defenders, width, identity_conditioned, action_encoding=action_encoding).to(device)
    initialization_sha256 = None
    initialization_updates = 0
    if initial_model is not None:
        recipe = initial_model['successor_task_value']
        if (initial_model['successor_task_fit'].get('value_target', 'success') != value_target
                or bool(recipe.get('isolate_factual_start', False)) != isolate_factual_start
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
        if row_probability is not None:
            # Retain the observed mass of the factual-state bank. Only the
            # distribution within the paired-root population is rebalanced.
            pooled_probability = np.concatenate((row_probability*len(rows),
                np.ones(len(pooled_target)-len(rows))))/len(pooled_target)
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
        paired_active = torch.cat((codes > 0, torch.zeros(count_pooled-len(codes), dtype=torch.bool, device=device)))
        paired_target_sha256 = hashlib.sha256(factual_values.tobytes()+
            (pooled_target-paired_factual_target).detach().cpu().numpy().tobytes()).hexdigest()
    def draw(count, size, probability):
        return rng.integers(count, size=size) if probability is None else rng.choice(count, size=size, p=probability)
    with torch.enable_grad():
        for step in range(updates):
            if uniform_sampling:
                index = torch.as_tensor(draw(len(pooled_target), 192, pooled_probability), device=device)
                raw = net.layers(pooled_features[index])
                if conditional:
                    from local_intervention_value import paired_difference_loss
                    factual_logits = net.layers(pooled_reference[index])[:, 0]
                    active = paired_active[index]
                    gain = net.signed_gain(raw, active)
                    estimate = (factual_logits.sigmoid()+gain).clamp(1e-6, 1.-1e-6)
                    loss = F.binary_cross_entropy(estimate, pooled_target[index])
                    loss += F.binary_cross_entropy_with_logits(factual_logits, paired_factual_target[index])
                    difference = pooled_target[index]-paired_factual_target[index]
                    event_error = paired_difference_loss(net.event_probabilities(raw), difference, value_target)
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
        identity_conditioned=identity_conditioned, action_encoding=action_encoding,
        state={key:value.detach().cpu() for key, value in net.state_dict().items()},
        options=options, decision_period=5, inference_device=device,
        isolate_factual_start=isolate_factual_start, intervention_budget=intervention_budget,
        gain_margin=float(gain_margin))
    if conditional:
        model['successor_task_value']['learning_objective'] = 'conditional'
    removal = tuple(model.get('mechanism_ablation', (False, False, False)))+(False,)*3
    model['mechanism_ablation'] = (removal[0], False, removal[2])
    model.update(gate_period=1, physical_interventions=True, fixed_intervention_clock=True,
                 success_objective=value_target == 'success')
    factual_labels = 0 if continuation is None else len(continuation[2])
    model['successor_task_fit'] = dict(ordinary=not conditional,
        objective=('absolute returns and same-future signed intervention-return mass' if conditional else
                  'absolute normalized discounted-return binary cross entropy'
                   if value_target == 'discounted_return' else 'absolute-outcome binary cross entropy'),
        value_target=value_target, training_target_sha256=training_target_sha256,
        return_gamma=.995 if value_target == 'discounted_return' else None,
        return_tick_cost=.001 if value_target == 'discounted_return' else None,
        return_normalization='(discounted task return + 1.001) / 2; common root shaping offset omitted'
            if value_target == 'discounted_return' else None,
        isolate_factual_start=isolate_factual_start,
        architecture=('single dense MLP, three ReLU hidden layers, factual and paired-event outputs'
                      if conditional else 'single dense MLP, three ReLU hidden layers'), width=width,
        action_encoding=action_encoding, dense_input_width=input_size,
        physical_action_features=('active vessel, coefficients, quarter-scaled current command difference; linear and logarithmic timing'
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
        predecessor_sha256=data['predecessor_sha256'], inherited_stages=False,
        paired_local_observations=int((codes > 0).sum()) if conditional else 0,
        factual_comparisons=int((codes > 0).sum()) if conditional else 0,
        paired_difference_weight=float(paired_difference_weight), paired_training_target_sha256=paired_target_sha256,
        paired_event_difference=conditional, future_public_states_used=False,
        common_motion_response_features=False,
        joint_factual_calibration=False, local_reconciliation=False, local_rank_weight=0.,
        public_vessel_identity=identity_conditioned,
        sampling=(('root-balanced pooled absolute returns' if root_sampling_fraction else 'uniform pooled absolute returns')
                  if uniform_sampling else 'matched weighted observations'),
        root_sampling_fraction=root_sampling_fraction,
        batch_size=192 if uniform_sampling else 128,
        development_future_weight=1. if uniform_sampling else data.get('development_future_weight', 1.),
        separate_factual_objective=conditional or (not uniform_sampling and factual_bank is not None),
        factual_bank_weight=(factual_bank_weight if factual_bank is not None and not uniform_sampling else 0.),
        candidate_catalog=('shared complete programs, including unobserved prior candidates'
                           if extra_options else 'all sampled complete programs, including negative outcomes'),
        additional_unobserved_candidates=len(extra_options-sampled_options-{(0, 0, 0)}),
        simulation_steps=data.get('simulation_steps', 0)+(0 if factual_bank is None else factual_bank['simulation_steps']),
        last_loss=float(loss.detach()))
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
