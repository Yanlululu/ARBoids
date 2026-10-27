"""Development recurrent correction after a learned factual failure gate."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import unified_civ_factual_value as factual
import unified_civ_search as search


def load_actor(path, development):
    actor = factual.load_actor(path, development)
    model = search.e.read(Path(path))
    settings = model['recurrent_factual_correction']
    assert settings['source_sha256'] == search.e.digest(Path(__file__))
    import local_intervention_value as local
    decision = actor.successor_task_value
    bank = model['state_local_factual_value']
    native_forward = decision.forward
    active = {}

    def catalog(context, controls):
        features = local.state_features(context, controls)
        normalized = features / decision.factual_local_scale
        distance = torch.cdist(normalized, decision.factual_local_features).square() / normalized.shape[-1]
        distances, nearest = distance.topk(min(bank['neighbors'], distance.shape[1]), largest=False)
        weights = (-(distances-distances[:, :1])/bank['radius']).softmax(-1)
        measured = (weights*decision.factual_local_labels[nearest]).sum(-1, keepdim=True)
        confidence = (-distances[:, :1]/bank['radius']).exp()
        count = active['packet'].get('_conditional_count')
        corrected = (torch.zeros(len(context), dtype=torch.bool, device=context.device)
                     if count is None else count.flatten().to(context.device) > 0)
        estimates = []
        for member in [decision.network, *decision.ensemble]:
            raw = decision.network_catalog(member, context, controls)
            if decision.ordinary:
                value = raw.sigmoid()
                p0 = confidence*measured + (1.-confidence)*value[:, :1]
                value = torch.cat((p0, value[:, 1:]), -1)
            else:
                p0 = confidence*measured + (1.-confidence)*raw[:, :1, 0].sigmoid()
                gain = (1.-p0)*raw[..., 1].sigmoid()-p0*raw[..., 2].sigmoid()
                value = p0 + gain*(decision.programs != 0).any(-1)[None]
            estimates.append(value)
        ranking, values = decision.aggregate_estimates(estimates)
        support = model.get('candidate_support')
        if support is not None:
            distances = (features[:, None]-decision.support_features[None]).square().mean(-1)
            indices = distances.topk(min(support['neighbors'], distances.shape[1]), largest=False).indices
            allowed = decision.support_options[indices].any(1)
            allowed[:, 0] = True
            ranking = ranking.masked_fill(~allowed, -torch.inf)
        if settings['repeat_joint_weight']:
            cost = search.fixed.joint.joint_cost(actor, decision, active['packet'], active['action'],
                dict(unary_weight=1., risk_scale=1., positive_only=True))
            ranking = ranking-settings['repeat_joint_weight']*corrected[:, None]*cost
        ranking[:, 1:] = ranking[:, 1:].masked_fill((confidence < .5) & ~corrected[:, None], -torch.inf)
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
    parser.add_argument('--policy', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--screen', action='store_true')
    parser.add_argument('--batch', type=int, default=400)
    args = parser.parse_args()
    _, development, _, _, runtime = search.configure()
    path = Path(args.policy)
    tasks = search.e.read(search.DESIGN)['screening_tasks' if args.screen else 'validation_tasks']
    target = search.OUT/(args.name+'-evaluation.pth')
    assert not target.exists()
    start = time.monotonic()
    rows = runtime.batched_credit_rows(path, tasks, batch_size=args.batch, device='cuda:0',
        actor_override=load_actor(path, development), fixed_inference_width=80)
    result = dict(completed=True, tasks=tasks, rows=rows, policy=str(path),
        policy_sha256=search.e.digest(path), source_sha256=search.e.digest(Path(__file__)),
        success=sum(row['success'] for row in rows), fixed_inference_width=80,
        elapsed_seconds=time.monotonic()-start,
        role='Reused-development-pool evaluation of recurrent learned correction.')
    search.e.save(result, target)
    print(dict(path=str(target), success=result['success'], count=len(rows)), flush=True)


if __name__ == '__main__':
    main()
