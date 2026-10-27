"""Learn a state-local factual value at measured proposal states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import unified_civ_search as search
import unified_civ_local_value as paired


def bank(args):
    _, _, local, _, _ = search.configure()
    from conditional_fusion_value import append_vessel_identity
    sources = [search.OLD / 'combined-factual.pth'] + [
        search.OUT / f'added-factual-part{i}.pth' for i in range(12)]
    states, controls, targets = [], [], []
    for path in sources:
        data = search.e.read(path)
        assert data['completed']
        for record in data['records']:
            context = torch.as_tensor(record['contexts'])
            step = record['root_snapshot']['root_steps']
            indices = torch.nonzero((context[:, -1] * 400.).round().long() == step).flatten()
            assert len(indices) == 1
            i = int(indices[0])
            states.append(append_vessel_identity(context[i], torch.as_tensor(record['orders'][i])))
            controls.append(torch.as_tensor(record['controls'][i]))
            targets.append(float(record['values'][i]))
    features = local.state_features(torch.stack(states), torch.stack(controls))
    target = search.OUT / 'proposal-factual-value-bank.pth'
    assert len(targets) == 9000 and not target.exists()
    search.e.save(dict(completed=True, features=features.numpy(), values=np.asarray(targets, np.float32),
        feature_scale=features.std(0).clamp_min(.05).numpy(),
        source_hashes={str(p):search.e.digest(p) for p in sources},
        role='Factual success supervision at 9000 measured public proposal states.'), target)
    print(json.dumps(dict(path=str(target), states=len(targets))), flush=True)


def candidate(args):
    search.configure()
    parent = Path(args.policy)
    model = search.e.read(parent)
    bank_path = search.OUT / 'proposal-factual-value-bank.pth'
    assert search.e.read(bank_path)['completed']
    model['state_local_factual_value'] = dict(bank_name=bank_path.name,
        bank_sha256=search.e.digest(bank_path), source_sha256=search.e.digest(Path(__file__)),
        parent_sha256=search.e.digest(parent), radius=args.radius, neighbors=args.neighbors,
        joint_weight=args.joint_weight, require_local=args.require_local,
        role='Development factual-value learner from reused state/outcome supervision.',
        deployment_inputs='Current public state and factual controls; no scenario ID or future value input.')
    model['successor_task_value']['gain_margin'] = args.margin
    target = search.OUT / (args.name + '.pth')
    assert not target.exists()
    search.e.save(model, target)
    print(json.dumps(dict(path=str(target))), flush=True)


def load_actor(path, development):
    actor = paired.load_actor(path, development)
    model = search.e.read(Path(path))
    settings = model.get('state_local_factual_value')
    if settings is None:
        return actor
    import local_intervention_value as local
    assert settings['source_sha256'] == search.e.digest(Path(__file__))
    bank_path = Path(path).parent / settings['bank_name']
    assert search.e.digest(bank_path) == settings['bank_sha256']
    data = search.e.read(bank_path)
    decision = actor.successor_task_value
    scale = torch.as_tensor(data['feature_scale'], dtype=torch.float32)
    decision.register_buffer('factual_local_scale', scale)
    decision.register_buffer('factual_local_features', torch.as_tensor(data['features'], dtype=torch.float32) / scale)
    decision.register_buffer('factual_local_labels', torch.as_tensor(data['values'], dtype=torch.float32))
    native_forward = decision.forward
    active = {}

    def catalog(context, controls):
        features = local.state_features(context, controls) / decision.factual_local_scale
        distance = torch.cdist(features, decision.factual_local_features).square() / features.shape[-1]
        distances, nearest = distance.topk(min(settings['neighbors'], distance.shape[1]), largest=False)
        weights = (-(distances - distances[:, :1]) / settings['radius']).softmax(-1)
        factual_local = (weights * decision.factual_local_labels[nearest]).sum(-1, keepdim=True)
        confidence = (-distances[:, :1] / settings['radius']).exp()
        estimates = []
        for member in [decision.network, *decision.ensemble]:
            raw = decision.network_catalog(member, context, controls)
            if decision.ordinary:
                value = raw.sigmoid()
                factual = confidence * factual_local + (1. - confidence) * value[:, :1]
                value = torch.cat((factual, value[:, 1:]), -1)
            else:
                factual = confidence * factual_local + (1. - confidence) * raw[:, :1, 0].sigmoid()
                gain = ((1. - factual) * raw[..., 1].sigmoid() - factual * raw[..., 2].sigmoid())
                gain = gain * (decision.programs != 0).any(-1)[None]
                value = factual + gain
            estimates.append(value)
        ranking, values = decision.aggregate_estimates(estimates)
        support = model.get('candidate_support')
        if support is not None:
            state = local.state_features(context, controls)
            support_distance = (state[:, None] - decision.support_features[None]).square().mean(-1)
            indices = support_distance.topk(min(support['neighbors'], support_distance.shape[1]), largest=False).indices
            allowed = decision.support_options[indices].any(1)
            allowed[:, 0] = True
            ranking = ranking.masked_fill(~allowed, -torch.inf)
        if settings['joint_weight']:
            cost = search.fixed.joint.joint_cost(actor, decision, active['packet'], active['action'],
                dict(unary_weight=1., risk_scale=1., positive_only=True))
            ranking = ranking - settings['joint_weight'] * cost
        if settings['require_local']:
            ranking[:, 1:] = ranking[:, 1:].masked_fill(confidence < .5, -torch.inf)
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
    parser.add_argument('operation', choices=('bank', 'candidate'))
    parser.add_argument('--name')
    parser.add_argument('--policy')
    parser.add_argument('--radius', type=float, default=.0001)
    parser.add_argument('--neighbors', type=int, default=4)
    parser.add_argument('--joint-weight', type=float, default=.32)
    parser.add_argument('--margin', type=float, default=.02)
    parser.add_argument('--require-local', action='store_true')
    args = parser.parse_args()
    globals()[args.operation](args)


if __name__ == '__main__':
    main()
