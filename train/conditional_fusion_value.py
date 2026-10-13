"""Candidate-conditioned values with anchored, single-vessel credit learning."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def component_relation_features(packet, base, programs, parts, bias=-.05):
    """Bind a pair's current actuator candidates to relative motion geometry."""
    from policy.interaction_sac import CandidateCoordinationGate
    motion, obs = packet['motion'], packet['obs']
    count, options, vessels = programs.shape[:3]
    reference = (base[..., 2] + bias).clamp(0., 1.)
    gates = (reference[:, None, :, None] + programs).clamp(0., 1.)
    rule = 750. * obs[..., 12:14] + 250.
    learned = 750. * base[..., :2] + 250.
    thrust = rule[:, None] + gates * (learned - rule)[:, None]
    _, paths = CandidateCoordinationGate.candidate_responses(
        packet, base, applied_thrust=thrust.transpose(1, 2))
    relative = paths[:, :, -1].transpose(1, 2)
    displacement = (relative[:, :1] - relative) / 5.
    rows = torch.arange(count)[:, None].expand(-1, options)
    first, second = programs[:, parts[:, 0]], programs[:, parts[:, 1]]
    left, right = first.abs().amax(-1).argmax(-1), second.abs().amax(-1).argmax(-1)
    di, dj = displacement[rows, parts[:, 0], left], displacement[rows, parts[:, 1], right]
    pi, pj = motion[rows, left, :2], motion[rows, right, :2]
    vi, vj = motion[rows, left, 3:5], motion[rows, right, 3:5]
    angle = motion[..., 2] + obs[..., 3]
    direction = torch.stack((angle.cos(), angle.sin()), -1)
    ni, nj = direction[rows, left], direction[rows, right]
    distance_i, distance_j = obs[rows, left, 2], obs[rows, right, 2]
    separation = pi - pj
    length = separation.norm(dim=-1)
    unit = separation / length[..., None].clamp_min(1e-6)
    cross = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    norm_i, norm_j = di.norm(dim=-1), dj.norm(dim=-1)
    features = torch.stack(((ni * nj).sum(-1), cross(ni, nj).abs(), torch.exp(-length / 10.),
        ((vi - vj) * unit).sum(-1) / 5., torch.minimum(distance_i, distance_j) / 20.,
        (distance_i - distance_j).abs() / 20., (di * dj).sum(-1), ((di - dj) * unit).sum(-1),
        (di * ni + dj * nj).sum(-1), (cross(di, ni) - cross(dj, nj)).abs(),
        norm_i + norm_j, (norm_i - norm_j).abs()), -1).clamp(-3., 3.)
    changed = (first.abs().flatten(2).amax(-1) > 1e-7) & (second.abs().flatten(2).amax(-1) > 1e-7)
    return features * (changed & (parts > 0).all(-1)[None])[..., None]


def conditional_edges(parts):
    """Each edge changes one vessel while retaining its partner's control."""
    edges = []
    for option, (left, right) in enumerate(parts.tolist()):
        if option == 0:
            continue
        if left and right:
            edges.extend(((option, left), (option, right)))
        else:
            edges.append((option, 0))
    return torch.tensor(edges, dtype=torch.long, device=parts.device)


@torch.no_grad()
def component_response_features(packet, base, programs, parts, response, bias=-.05, response_scale=None):
    """Encode conditional joint motion from the same public, filtered rollout.

    The second difference removes each isolated control's motion and common
    target motion. It measures how the partner changes a control's effect.
    No realized return or environment future is available to this encoder.
    """
    features = component_relation_features(packet, base, programs, parts, bias)
    count, options = programs.shape[:2]
    rows = torch.arange(count, device=programs.device)[:, None].expand(-1, options)
    first, second = programs[:, parts[:, 0]], programs[:, parts[:, 1]]
    left = first.abs().amax(-1).argmax(-1)
    right = second.abs().amax(-1).argmax(-1)
    joint = response - response[:, parts[:, 0]] - response[:, parts[:, 1]] + response[:, :1]
    di, dj = joint[rows, torch.arange(options), left] / 5., joint[rows, torch.arange(options), right] / 5.
    pi, pj = response[rows, 0, left], response[rows, 0, right]
    ni = pi / pi.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    nj = pj / pj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    separation = pi - pj
    unit = separation / separation.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cross = lambda x, y: x[..., 0]*y[..., 1] - x[..., 1]*y[..., 0]
    effects = torch.stack((-(di*ni+dj*nj).sum(-1),
        (di-dj).mul(unit).sum(-1), di.norm(dim=-1)+dj.norm(dim=-1),
        (cross(di, ni)-cross(dj, nj)).abs()), -1).flatten(-2).clamp(-3., 3.)
    changed = (first.abs().flatten(2).amax(-1)>1e-7) & (second.abs().flatten(2).amax(-1)>1e-7)
    effects *= (changed & (parts>0).all(-1)[None])[..., None]
    if response_scale is not None:
        effects = (effects / effects.new_tensor(response_scale)).clamp(-5., 5.)
    return torch.cat((features, effects), -1)


def paired_targets(costs, edges):
    """Cross-fit precision using different futures from each target mean."""
    if costs.shape[1] < 4:
        raise ValueError('At least four paired futures are required.')
    contrast = costs[:, :, edges[:, 0]] - costs[:, :, edges[:, 1]]
    folds = []
    for half in (0, 1):
        target = contrast[:, half::2].mean(1)
        other = contrast[:, 1-half::2]
        precision = 1. / (4. + other.var(1, unbiased=True) / other.shape[1])
        folds.append((target, precision))
    return folds


def paired_covariance_targets(costs, edges):
    """Cross-fit the shared noise directions of conditional return edges."""
    contrast = costs[:, :, edges[:, 0]] - costs[:, :, edges[:, 1]]
    folds = []
    for half in (0, 1):
        target = contrast[:, half::2].mean(1)
        other = contrast[:, 1-half::2]
        centered = other - other.mean(1, keepdim=True)
        size, repeats, edges_count = centered.shape
        covariance = torch.einsum('bre,brf->ef', centered, centered) / (size * (repeats-1) * repeats)
        covariance = .8*covariance + .2*torch.diag(covariance.diag()) + 4.*torch.eye(edges_count, dtype=costs.dtype, device=costs.device)
        precision = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
        folds.append((target, precision))
    return folds


def conditional_auxiliary(energy, edges, folds, mask, *, quadratic=False, covariance=False):
    prediction = energy[:, edges[:, 0]] - energy[:, edges[:, 1]]
    losses = []
    for target, precision in folds:
        if covariance:
            residual = (prediction-target) * mask.sqrt()
            losses.append(.5*torch.einsum('be,ef,bf->b', residual, precision, residual) / mask.sum(-1).clamp_min(1))
            continue
        error = .5 * (prediction-target).square() if quadratic else F.smooth_l1_loss(prediction, target, beta=2., reduction='none')
        losses.append((error * precision * mask).sum(-1) / mask.sum(-1).clamp_min(1))
    return torch.stack(losses).mean(0)


@torch.no_grad()
def decision_credit_mask(subject, initial, data, prior, settings, edges, mask):
    """Focus credit on the incumbent's candidate neighborhood, with coverage."""
    unary = subject.reactive_monotone_estimates(initial, prior.double(), data['parts'], data['packet'], False).mean((0, 2))
    score = unary + settings['prior'] * data['options'].square().mean((-1, -2))
    singles = 1 + 4 * data['base'].shape[1]
    support = torch.zeros_like(score, dtype=torch.bool)
    support[:, :singles] = True
    support.scatter_(1, score[:, singles:].argsort(-1)[:, :settings['component_candidate_budget']]+singles, True)
    probability = (-score.masked_fill(~support, float('inf')) / 2.).softmax(-1)
    weight = .2 + .8 * score.shape[-1] * probability[:, edges[:, 0]]
    return mask * weight


def state_features(data, prior):
    distance = data['packet']['obs'][..., 2].double()
    remaining = data['packet']['time_remaining'].double()
    baseline = prior.double().mean(1)[:, 0]
    return torch.stack((torch.ones_like(baseline), baseline / 20., distance.amin(-1) / 20.,
        distance.mean(-1) / 30., remaining, baseline.square() / 400.), -1)


def decisions(subject, coefficients, data, prior, settings, interaction, risk):
    prediction = subject.reactive_monotone_estimates(
        coefficients.double(), prior.double(), data['parts'], data['packet'], interaction).mean(2)
    independent = subject.reactive_monotone_estimates(
        coefficients.double(), prior.double(), data['parts'], data['packet'], False).mean((0, 2))
    penalty = settings['prior'] * data['options'].square().mean((-1, -2))
    initial = independent + penalty
    reference = initial.argmin(-1)
    if not interaction:
        return reference, prediction
    score = subject.reactive_component_paired_score(prediction, data['options'], reference, risk, settings['prior'])
    singles = 1 + 4 * data['base'].shape[1]
    joints = initial[:, singles:].argsort(-1)[:, :settings['component_candidate_budget']] + singles
    support = torch.zeros_like(score, dtype=torch.bool)
    support[:, :singles] = True
    support.scatter_(1, joints, True)
    return score.masked_fill(~support, float('inf')).argmin(-1), prediction


def fit_conditional_component(args, subject):
    """Keep absolute value fitting and add local conditional credit supervision."""
    torch.set_num_threads(2)
    bank = torch.load(args.reactive_bootstrap_data, map_location='cpu', weights_only=False)
    data, settings = bank['data'], dict(bank['settings'])
    if not bank.get('complete') or not settings.get('component_all_pairs'):
        raise ValueError('Collect complete deployment-supported joint controls first.')
    if len(data['parts']) != 1 + 4 * settings['defenders'] + 16 * settings['defenders'] * (settings['defenders']-1) // 2:
        raise ValueError('Training and deployment must have the same candidate support.')
    settings.update(component_orthogonal=True, component_contextual=True, component_candidate_budget=16,
        conditional_value_version='absolute-plus-single-vessel-credit-v1')
    focused = getattr(args, 'decision_loss', False)
    covariance = getattr(args, 'component_covariance_credit', False)
    if focused:
        settings['conditional_value_version'] = 'absolute-plus-decision-local-mean-credit-v2'
    if covariance:
        settings['conditional_value_version'] = 'absolute-plus-correlated-single-vessel-credit-v3'
    if 'model_response' in data:
        settings['component_response_features'] = True
        features = component_response_features(data['packet'], data['base'], data['options'],
            data['parts'], data['model_response'], settings['bias'])
        training_rows = torch.isin(data['origin'], bank['training_origins'])
        scale = features[training_rows, :, 12:].square().mean((0,1)).sqrt().clamp_min(.02)
        settings['component_response_scale'] = scale.tolist()
        features = torch.cat((features[..., :12], (features[..., 12:]/scale).clamp(-5.,5.)), -1)
    else:
        features = component_relation_features(data['packet'], data['base'], data['options'], data['parts'], settings['bias'])
    packet = dict(data['packet'], candidate_interactions=features)
    data = dict(data, packet=packet)
    train = torch.isin(data['origin'], bank['training_origins']).nonzero().flatten()
    valid = torch.isin(data['origin'], bank['held_origins']).nonzero().flatten()
    if len(train) < 32 or len(valid) < 16:
        raise ValueError('Use at least 32 training and 16 held states.')
    training, validation = subject.reactive_subset(data, train), subject.reactive_subset(data, valid)
    prior, valid_prior = bank['prior'][train], bank['prior'][valid]
    seed = args.seed if args.bootstrap_seed is None else args.bootstrap_seed
    parts = data['parts']; singles = 1 + 4 * settings['defenders']
    unary_data = dict(training, costs=training['costs'][:, :, :singles],
        options=training['options'][:, :singles], parts=parts[:singles],
        packet={k:(v[:, :singles] if k == 'candidate_interactions' else v) for k, v in training['packet'].items()})
    # The shared initialization uses ordinary returns. Conditional supervision
    # is absent from the ordinary-return ablation, including its unary fit.
    initial, _ = subject.fit_prior_monotone(unary_data, prior[:, :, :singles], False, False,
        seed, 3, True, False, False, True)
    initial = F.pad(initial, (0, features.shape[-1])).double()
    costs = training['costs'].double(); target = costs.mean(1)
    variance = 4. + costs.var(1, unbiased=True) / costs.shape[1]
    unique = subject.reactive_unique_control_mask(training['options'].flatten(2)).double()
    state = state_features(training, prior)
    edges = conditional_edges(parts)
    changed = (training['options'][:, edges[:, 0]] - training['options'][:, edges[:, 1]]).abs().flatten(2).amax(-1) > 1e-7
    edge_mask = changed.double() * unique[:, edges[:, 0]]
    if focused:
        edge_mask = decision_credit_mask(subject, initial, training, prior, settings, edges, edge_mask)
    folds = paired_covariance_targets(costs, edges) if covariance else paired_targets(costs, edges)
    for arm in args.evaluation_arms.split(','):
        if arm not in ('difference', 'absolute', 'additive'):
            raise ValueError('Expected difference, absolute, or additive.')
        conditional, interaction = arm != 'absolute', arm != 'additive'
        fitted, histories, baselines = [], [], []
        for head in range(len(initial)):
            generator = torch.Generator().manual_seed(seed + head * 104729)
            rows, _ = subject.reactive_positive_credit_weights(subject.reactive_state_groups(training), costs.shape[1], generator)
            weights = nn.Parameter(initial[head:head+1].clone())
            baseline = nn.Parameter(torch.linalg.lstsq(state, target[:, 0]).solution)
            optimizer = torch.optim.LBFGS([weights, baseline], max_iter=args.updates, line_search_fn='strong_wolfe')
            def losses():
                energy = subject.reactive_monotone_estimates(weights, prior.double(), parts, training['packet'], interaction)[0].mean(1)
                total = energy + (state @ baseline)[:, None]
                absolute = (((total-target).square() / variance) * unique).sum(-1) / unique.sum(-1)
                auxiliary = conditional_auxiliary(energy, edges, folds, edge_mask, quadratic=focused, covariance=covariance)
                regularizer = .02 * (weights[..., :4]-initial[head:head+1, :, :4]).square().mean() + .1 * weights[..., 4:].square().mean()
                loss = (rows * (absolute + (args.aux_weight * auxiliary if conditional else 0.))).mean() + regularizer
                return loss, absolute.mean(), auxiliary.mean()
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = losses(); loss.backward(); return loss
            optimizer.step(closure)
            loss, absolute, auxiliary = losses()
            fitted.append(weights.detach()[0].float()); baselines.append(baseline.detach().float())
            histories.append(dict(loss=float(loss.detach()), absolute=float(absolute.detach()), conditional=float(auxiliary.detach())))
        coefficients = torch.stack(fitted); trials = []
        for risk in (0., .5, 1., 2.):
            choice, _ = decisions(subject, coefficients, validation, valid_prior, settings, interaction, risk)
            actual = validation['costs'].mean(1)[torch.arange(len(valid)), choice]
            trials.append(dict(risk=risk, cost=float(actual.mean())))
        selection = min(trials, key=lambda item: item['cost'])
        actor = subject.ReactiveComponentGate(settings, coefficients, selection['risk'], interaction)
        saved = dict(kind='reactive-component-v1', actor=actor.state_dict(), settings=settings, calibration=coefficients,
            risk=selection['risk'], interaction=interaction, conditional=conditional, seed=args.seed, arm=arm,
            training_seed=seed, selection=selection, calibration_trials=trials, history=histories,
            absolute_baseline=torch.stack(baselines), auxiliary_weight=args.aux_weight if conditional else 0.,
            training_origins=bank['training_origins'], held_origins=bank['held_origins'],
            actual_training_states=len(train), actual_held_states=len(valid), collecting_policy=bank['collecting_policy'],
            data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path(subject.__file__).read_bytes()).hexdigest())
        subject.atomic_torch(args.output / f'reactive-{arm}-{args.seed}.pth', saved)
        print(json.dumps(dict(arm=arm, selection=selection, trials=trials, training=histories)), flush=True)
