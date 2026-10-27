"""A shared learned factual predictor and paired conditional candidate values."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import unified_civ_search as search
import unified_civ_factual_value as previous


class ForestValue(torch.nn.Module):
    def __init__(self, record):
        super().__init__()
        self.depth = record['depth']
        for name in ('roots', 'feature', 'left', 'right'):
            self.register_buffer(name, torch.as_tensor(record[name], dtype=torch.long))
        self.register_buffer('threshold', torch.as_tensor(record['threshold'], dtype=torch.float64))
        self.register_buffer('probability', torch.as_tensor(record['probability'], dtype=torch.float32))

    def forward(self, features):
        index = self.roots[None].expand(len(features), -1)
        for _ in range(self.depth):
            feature = self.feature[index].clamp_min(0)
            value = features.gather(1, feature)
            index = torch.where(value <= self.threshold[index], self.left[index], self.right[index])
        return self.probability[index].mean(-1, keepdim=True)


def fit(args):
    search.configure()
    from sklearn.ensemble import ExtraTreesClassifier
    bank_path = search.OUT / 'proposal-factual-value-bank.pth'
    bank = search.e.read(bank_path)
    forest = ExtraTreesClassifier(n_estimators=256, max_depth=24, max_features=.7,
        min_samples_leaf=1, n_jobs=1, random_state=args.seed)
    forest.fit(bank['features'], bank['values'])
    arrays = {key:[] for key in ('feature', 'left', 'right', 'threshold', 'probability')}
    roots = []
    offset = 0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        roots.append(offset)
        nodes = np.arange(tree.node_count) + offset
        arrays['feature'].append(tree.feature)
        arrays['left'].append(np.where(tree.children_left < 0, nodes, tree.children_left + offset))
        arrays['right'].append(np.where(tree.children_right < 0, nodes, tree.children_right + offset))
        arrays['threshold'].append(tree.threshold)
        arrays['probability'].append(tree.value[:, 0, 1] / tree.value[:, 0].sum(-1))
        offset += tree.node_count
    result = {key:np.concatenate(value) for key, value in arrays.items()}
    result.update(roots=np.asarray(roots), depth=max(est.tree_.max_depth for est in forest.estimators_),
        completed=True, seed=args.seed, source_sha256=search.e.digest(Path(__file__)),
        training_bank_sha256=search.e.digest(bank_path), training_states=len(bank['values']),
        inputs='Public state and CI controls at measured proposal states.',
        target='Observed factual CI success; no future input at deployment.')
    deployed = ForestValue(result)
    query = torch.as_tensor(bank['features'][:128], dtype=torch.float32)
    expected = forest.predict_proba(bank['features'][:128])[:, 1]
    actual = deployed(query).detach().numpy().ravel()
    assert np.max(np.abs(expected - actual)) < 1e-6
    target = search.OUT / f'factual-forest-seed{args.seed}.pth'
    assert not target.exists()
    search.e.save(result, target)
    print(json.dumps(dict(path=str(target), nodes=offset,
        training_accuracy=forest.score(bank['features'], bank['values']))), flush=True)


def candidate(args):
    search.configure()
    parent = Path(args.policy)
    model = search.e.read(parent)
    forest = search.OUT / f'factual-forest-seed{args.seed}.pth'
    assert search.e.read(forest)['completed']
    model['factual_forest_value'] = dict(forest_name=forest.name, forest_sha256=search.e.digest(forest),
        source_sha256=search.e.digest(Path(__file__)), parent_sha256=search.e.digest(parent),
        joint_weight=args.joint_weight, failure_threshold=args.failure_threshold,
        role='Development factual predictor shared between conditional and ordinary candidate learners.')
    model['successor_task_value']['gain_margin'] = args.margin
    target = search.OUT / (args.name + '.pth')
    assert not target.exists()
    search.e.save(model, target)
    print(json.dumps(dict(path=str(target))), flush=True)


def load_actor(path, development):
    actor = previous.load_actor(path, development)
    model = search.e.read(Path(path))
    settings = model.get('factual_forest_value')
    if settings is None:
        return actor
    import local_intervention_value as local
    assert settings['source_sha256'] == search.e.digest(Path(__file__))
    forest_path = Path(path).parent / settings['forest_name']
    assert search.e.digest(forest_path) == settings['forest_sha256']
    decision = actor.successor_task_value
    decision.factual_forest = ForestValue(search.e.read(forest_path))
    native_forward = decision.forward
    active = {}

    def catalog(context, controls):
        features = local.state_features(context, controls)
        factual = decision.factual_forest(features)
        estimates = []
        for member in [decision.network, *decision.ensemble]:
            raw = decision.network_catalog(member, context, controls)
            if decision.ordinary:
                value = torch.cat((factual, raw[:, 1:].sigmoid()), -1)
            else:
                assert decision.conditional_objective == 'joint_likelihood'
                gain = ((1. - factual) * raw[..., 1].sigmoid() - factual * raw[..., 2].sigmoid())
                gain = gain * (decision.programs != 0).any(-1)[None]
                value = factual + gain
            estimates.append(value)
        ranking, values = decision.aggregate_estimates(estimates)
        support = model.get('candidate_support')
        if support is not None:
            distance = (features[:, None] - decision.support_features[None]).square().mean(-1)
            indices = distance.topk(min(support['neighbors'], distance.shape[1]), largest=False).indices
            allowed = decision.support_options[indices].any(1)
            allowed[:, 0] = True
            ranking = ranking.masked_fill(~allowed, -torch.inf)
        if settings['joint_weight']:
            cost = search.fixed.joint.joint_cost(actor, decision, active['packet'], active['action'],
                dict(unary_weight=1., risk_scale=1., positive_only=True))
            ranking = ranking - settings['joint_weight'] * cost
        ranking[:, 1:] = ranking[:, 1:].masked_fill(factual > settings['failure_threshold'], -torch.inf)
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


def evaluate(args):
    _, development, _, _, runtime = search.configure()
    design = search.e.read(search.DESIGN)
    tasks = design['screening_tasks' if args.screen else 'validation_tasks']
    policy = Path(args.policy)
    target = search.OUT / (args.name + '-evaluation.pth')
    assert not target.exists()
    start = time.monotonic()
    rows = runtime.batched_credit_rows(policy, tasks, batch_size=args.batch,
        device='cuda:0', actor_override=load_actor(policy, development), fixed_inference_width=80)
    result = dict(completed=True, tasks=tasks, rows=rows, policy=str(policy),
        policy_sha256=search.e.digest(policy), source_sha256=search.e.digest(Path(__file__)),
        success=sum(row['success'] for row in rows), fixed_inference_width=80,
        elapsed_seconds=time.monotonic()-start, role='Repeated development evaluation of learned factual values.')
    search.e.save(result, target)
    print(json.dumps(dict(path=str(target), success=result['success'], count=len(rows))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('fit', 'candidate', 'evaluate'))
    parser.add_argument('--name')
    parser.add_argument('--policy')
    parser.add_argument('--seed', type=int, default=20261210)
    parser.add_argument('--joint-weight', type=float, default=0.)
    parser.add_argument('--failure-threshold', type=float, default=.5)
    parser.add_argument('--margin', type=float, default=.02)
    parser.add_argument('--batch', type=int, default=400)
    parser.add_argument('--screen', action='store_true')
    args = parser.parse_args()
    globals()[args.operation](args)


if __name__ == '__main__':
    main()
