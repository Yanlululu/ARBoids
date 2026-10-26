"""Conditional value selection regularized by the factual joint fusion objective."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import conditional_policy_improvement as previous

e = previous.e
OUT = e.BASE / 'joint-value-policy-improvement-20261026-2059'


def initialize():
    state = previous.initialize()
    OUT.mkdir(exist_ok=True)
    return state


def joint_cost(actor, decision, packet, action, settings):
    """Score complete candidate configurations with the unchanged CI forecast.

    Catalog digit zero means the already selected factual control. All other
    digits retain their original measured intervention semantics. This is a
    public-state planning cost, not a replacement for observed terminal labels.
    """
    controls = packet['_plain_return_controls']
    source, rule, factual = controls.split(2, -1)
    difference = source-rule
    preferred = action[..., 2:3].expand_as(source)
    factual_gate = torch.where(difference.abs() > 1e-5,
        (factual-rule)/difference.masked_fill(difference.abs() <= 1e-5, 1.), preferred).clamp(0., 1.)
    b, n = source.shape[:2]
    coefficients = decision.coefficients.to(source)
    grid = torch.cat((factual_gate[:, :, None],
        coefficients[None, None, 1:].expand(b, n, -1, -1)), 2)
    risk = actor.candidate_risk(dict(packet), source, pairwise=True, candidate_grid=grid)
    _, order = decision.public_context(packet)
    batch = torch.arange(b, device=source.device)
    risk = risk[batch[:, None, None], order[:, :, None], order[:, None, :]]
    width = float(actor.actuator_fusion_refinement['preference_width'])
    unary = ((grid-preferred[:, :, None])/width).square().mean(-1)
    unary = unary.gather(1, order[:, :, None].expand_as(unary))
    programs = decision.programs.to(source.device)
    energy = source.new_zeros((b, len(programs)))
    k = grid.shape[2]
    for vessel in range(n):
        energy += settings['unary_weight']*unary[:, vessel, programs[:, vessel]]
        for other in range(vessel):
            code = programs[:, other]*k+programs[:, vessel]
            energy += (settings['risk_scale']*actor.actuator_fusion_refinement['risk_weight']
                *risk[:, other, vessel].flatten(1)[:, code])
    difference = energy-energy[:, :1]
    return difference.clamp_min(0.) if settings['positive_only'] else difference


def load_actor(path, development):
    actor = previous.load_actor(path, development)
    model = e.read(Path(path))
    settings = model.get('joint_value_policy_improvement')
    if settings is None or not settings['joint_weight']:
        return actor
    decision = actor.successor_task_value
    native_forward, native_catalog = decision.forward, decision.catalog_estimates
    active = {}

    def catalog(context, controls):
        ranking, values = native_catalog(context, controls)
        cost = joint_cost(actor, decision, active['packet'], active['action'], settings)
        ranking = ranking-settings['joint_weight']*cost.to(ranking)
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


def candidate(args):
    initialize()
    path = Path(args.policy)
    model = e.read(path)
    model['successor_task_value']['ensemble_gain_penalty'] = args.beta
    model['successor_task_value']['gain_margin'] = args.margin
    model['joint_value_policy_improvement'] = dict(
        parent_policy=str(path), parent_sha256=e.digest(path), joint_weight=args.joint_weight,
        unary_weight=args.unary_weight, risk_scale=args.risk_scale,
        positive_only=not args.signed_cost, beta=args.beta, margin=args.margin,
        objective='paired conditional gain minus factual-relative joint fusion objective increase',
        candidate_execution_unchanged=True, physical_dynamics_unchanged=True,
        reward_unchanged=True, predecessor_ci_unchanged=True,
        planning_cost_uses_public_state_only=True)
    target = OUT/(args.name+'.pth')
    if target.exists():
        raise ValueError('A policy identity cannot be overwritten.')
    e.save(model, target)
    print(json.dumps(dict(path=str(target), sha256=e.digest(target))), flush=True)


def design(args):
    initialize()
    previous_tasks = list(e.read(previous.PRIOR/'mixed-design.pth')['tasks'])
    previous_tasks += e.read(previous.PRIOR/'joint-final-new200-design.pth')['tasks']
    previous_tasks += e.read(previous.OUT/'fresh400-final-design.pth')['tasks']
    used = {int(scene[0]) for scene, _ in previous_tasks}
    for path in OUT.glob('*design.pth'):
        used.update(int(scene[0]) for scene, _ in e.read(path)['tasks'])
    tasks = []
    index = 0
    while len(tasks) < args.count:
        seed = e.seed32(args.seed, index)
        index += 1
        if seed in used:
            continue
        used.add(seed)
        tasks.append(((seed, 6, 2. if len(tasks) < args.count//2 else 2.5), None))
    policies = {str(Path(path)):e.digest(Path(path)) for path in args.policies}
    for path in args.policies:
        model = e.read(Path(path))
        training = {int(scene[0]) for scene in model.get('successor_task_fit', {}).get('training_layouts', [])}
        assert not training.intersection(int(scene[0]) for scene, _ in tasks)
    result = dict(tasks=tasks, policies=policies, frozen_unix=time.time(),
        driver_sha256=e.digest(Path(__file__)), previous_driver_sha256=e.digest(Path(previous.__file__)),
        role='New geometry after fixing policy checkpoints and joint-value decision settings.')
    target = OUT/(args.name+'.pth')
    if target.exists():
        raise ValueError('The independent design cannot be overwritten.')
    e.save(result, target)
    print(json.dumps(dict(path=str(target), count=len(tasks), policies=len(policies))), flush=True)


def evaluate(args):
    _, d, _, _, r = initialize()
    policy = Path(args.policy)
    if args.design:
        specification = e.read(Path(args.design))
        assert specification['driver_sha256'] == e.digest(Path(__file__))
        assert specification['previous_driver_sha256'] == e.digest(Path(previous.__file__))
        assert specification['policies'][str(policy)] == e.digest(policy)
        tasks, role = specification['tasks'], specification['role']
    else:
        source = (previous.OUT/'fresh400-final-design.pth' if args.suite == 'round1-400'
                  else previous.PRIOR/'joint-final-new200-design.pth')
        tasks = e.read(source)['tasks']
        if args.count != len(tasks):
            tasks = [tasks[int(i)] for i in np.linspace(0, len(tasks)-1, args.count, dtype=int)]
        role = 'Previously evaluated geometry used for development of the joint-value decision.'
    full_task_count = len(tasks)
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError('The task shard must be a valid partition of the frozen design.')
    if args.shards != 1:
        partitions = np.array_split(np.arange(len(tasks)), args.shards)
        tasks = [tasks[int(index)] for index in partitions[args.shard_index]]
    target = OUT/(args.name+f'-n{len(tasks)}-evaluation.pth')
    if target.exists():
        result = e.read(target)
        assert result['tasks'] == tasks and result['policy_sha256'] == e.digest(policy)
        if result['completed']:
            print(json.dumps(dict(cached=True, path=str(target), success=result['success'])), flush=True)
            return
    start = time.monotonic()
    rows = r.batched_credit_rows(policy, tasks, batch_size=args.batch, device=args.device,
        actor_override=load_actor(policy, d), fixed_inference_width=80)
    result = dict(policy=str(policy), policy_sha256=e.digest(policy), tasks=tasks, rows=rows,
        success=sum(row['success'] for row in rows), completed=True,
        elapsed_seconds=time.monotonic()-start, role=role, fixed_inference_width=80,
        driver_sha256=e.digest(Path(__file__)), full_design_task_count=full_task_count,
        shard_index=args.shard_index, shards=args.shards)
    e.save(result, target)
    print(json.dumps(dict(path=str(target), success=result['success'], count=len(tasks),
        collisions=sum(row['collision'] for row in rows), elapsed_seconds=result['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('candidate', 'design', 'evaluate'))
    parser.add_argument('--name', required=True)
    parser.add_argument('--policy')
    parser.add_argument('--policies', nargs='+', default=[])
    parser.add_argument('--design')
    parser.add_argument('--suite', choices=('development200', 'round1-400'), default='development200')
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--batch', type=int, default=200)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=11262159)
    parser.add_argument('--joint-weight', type=float, default=.1)
    parser.add_argument('--unary-weight', type=float, default=1.)
    parser.add_argument('--risk-scale', type=float, default=1.)
    parser.add_argument('--signed-cost', action='store_true')
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--margin', type=float, default=.02)
    args = parser.parse_args()
    {'candidate':candidate, 'design':design, 'evaluate':evaluate}[args.operation](args)


if __name__ == '__main__':
    main()
