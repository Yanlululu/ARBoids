"""State-local paired value estimation for the repeatedly trained development pool."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

import unified_civ_search as search


def build(args):
    _, _, local, _, _ = search.configure()
    path = Path(args.data)
    data, _ = local.retain_vessel_identity(search.e.read(path))
    assert data['completed']
    parent = Path(args.policy)
    model = search.e.read(parent)
    options = [tuple(option) for option in model['successor_task_value']['options']]
    lookup = {option: i for i, option in enumerate(options)}
    roots = data['roots']
    context = torch.as_tensor(np.asarray([root['context'] for root in roots]), dtype=torch.float32)
    controls = torch.as_tensor(np.asarray([root['effect_controls'] for root in roots]), dtype=torch.float32)
    features = local.state_features(context, controls)
    counts = np.zeros((len(roots), len(options)), np.float32)
    gains = np.zeros_like(counts)
    for index, root in enumerate(roots):
        for (option, future), result in root['outcomes'].items():
            reference = root['outcomes'].get((0, future))
            if reference is None:
                continue
            slot = lookup[tuple(root['options'][option])]
            mass = float(root.get('future_weights', np.ones(len(root['futures'])))[future])
            counts[index, slot] += mass
            gains[index, slot] += mass * (float(result['success']) -
                                          (0. if args.ordinary else float(reference['success'])))
    gains /= np.maximum(counts, 1e-12)
    scale = features.std(0).clamp_min(.05)
    model['state_local_paired_value'] = dict(
        features=features.numpy(), feature_scale=scale.numpy(), measured_gain=gains,
        observed=counts > 0, radius=args.radius, neighbors=args.neighbors,
        mix=args.mix, fallback=args.fallback, joint_weight=args.joint_weight,
        ordinary_absolute_value=args.ordinary,
        source_sha256=search.e.digest(Path(__file__)), data_sha256=search.e.digest(path),
        parent_sha256=search.e.digest(parent), training_roots=len(roots),
        input='Current public kinematics, physical identity, clock and CI controls only.',
        target=('Measured absolute candidate success.' if args.ordinary else
                'Measured same-root, same-future candidate success minus CI success.'),
        role='Nonparametric development value learner; no scenario identifier or future input.')
    model['successor_task_value']['gain_margin'] = args.margin
    target = search.OUT / (args.name + '.pth')
    if target.exists():
        raise ValueError('This policy name already exists.')
    search.e.save(model, target)
    print(json.dumps(dict(path=str(target), roots=len(roots), radius=args.radius)), flush=True)


def load_actor(path, development):
    actor = search.centered.load_actor(path, development)
    settings = search.e.read(Path(path)).get('state_local_paired_value')
    if settings is None:
        return actor
    assert settings['source_sha256'] == search.e.digest(Path(__file__))
    import local_intervention_value as local
    decision = actor.successor_task_value
    scale = torch.as_tensor(settings['feature_scale'], dtype=torch.float32)
    decision.register_buffer('local_value_scale', scale)
    decision.register_buffer('local_value_features', torch.as_tensor(settings['features'], dtype=torch.float32) / scale)
    decision.register_buffer('local_value_gain', torch.as_tensor(settings['measured_gain'], dtype=torch.float32))
    decision.register_buffer('local_value_observed', torch.as_tensor(settings['observed'], dtype=torch.bool))
    native_catalog, native_forward = decision.catalog_estimates, decision.forward
    active = {}

    def catalog(context, controls):
        if settings['fallback'] == 'neural':
            ranking, values = native_catalog(context, controls)
        else:
            values = context.new_zeros((len(context), len(decision.options)))
            ranking = values.clone()
        features = local.state_features(context, controls) / decision.local_value_scale
        distance = torch.cdist(features, decision.local_value_features).square() / features.shape[-1]
        distances, nearest = distance.topk(min(settings['neighbors'], distance.shape[1]), largest=False)
        weights = (-(distances - distances[:, :1]) / settings['radius']).exp()
        observed = decision.local_value_observed[nearest]
        mass = weights[..., None] * observed
        gain = (mass * decision.local_value_gain[nearest]).sum(1) / mass.sum(1).clamp_min(1e-12)
        if settings['ordinary_absolute_value']:
            gain = gain - gain[:, :1]
        allowed = observed.any(1)
        allowed[:, 0] = True
        confidence = settings['mix'] * (-distances[:, :1] / settings['radius']).exp()
        cost = (search.fixed.joint.joint_cost(actor, decision, active['packet'], active['action'],
                 dict(unary_weight=1., risk_scale=1., positive_only=True))
                if settings['joint_weight'] else torch.zeros_like(gain))
        local_rank = values[:, :1] + gain - settings['joint_weight'] * cost
        if settings['fallback'] == 'neural':
            finite = torch.isfinite(ranking)
            baseline = torch.where(finite, ranking, values[:, :1])
            blended = confidence * local_rank + (1. - confidence) * baseline
            ranking = torch.where(allowed, blended, ranking)
            ranking = ranking.masked_fill(~finite & (confidence < .05), -torch.inf)
        else:
            ranking = (confidence * gain - settings['joint_weight'] * cost).masked_fill(~allowed, -torch.inf)
            values = confidence * gain
        return ranking, values

    def forward(packet, action):
        active.update(packet=packet, action=action)
        try:
            return native_forward(packet, action)
        finally:
            active.clear()

    decision.catalog_estimates = catalog
    decision.decision_scores = lambda ranking, values: ranking
    decision.forward = forward
    return actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--radius', type=float, default=.005)
    parser.add_argument('--neighbors', type=int, default=8)
    parser.add_argument('--mix', type=float, default=1.)
    parser.add_argument('--joint-weight', type=float, default=.32)
    parser.add_argument('--margin', type=float, default=.02)
    parser.add_argument('--fallback', choices=('ci', 'neural'), default='ci')
    parser.add_argument('--ordinary', action='store_true')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
