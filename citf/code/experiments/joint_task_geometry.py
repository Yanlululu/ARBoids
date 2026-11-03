"""Candidate-interaction prediction of pursuit, radial cover, and clearance.

Every supervised quantity is a distance on the observed stopped short branch.
The learned task geometry directly scores a complete current configuration;
no nominal radial lower bound or terminal outcome target is used.
"""
import argparse
from dataclasses import dataclass
import math
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from citf.bridge import Controller, factorize, inputs
from citf.contracts import canonical_hash, file_hash, tree_hash
from citf.risk import bounded_mean, pair_indices, rectified_gaussian_risk
from citf.search import nominal_beam, initialize_bank, whole_configuration_search
from experiments import pair_geometry as pair
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team


@dataclass(frozen=True)
class TaskGeometryConfig(guard.GuardConfig):
    revision: str = 'CITF-joint-stopped-task-geometry-v1'
    prediction_condition: str = 'held fusion coefficients with state feedback'
    guard_weight: float = 16.
    guard_margin: float = 6.
    guard_nll_weight: float = 1.
    pursuit_weight: float = .25
    pursuit_radius: float = 5.
    delta_weight: float = 2.
    delta_scale: float = .5
    impulse_reference: bool = False


def all_pairs(defenders, device=None):
    return torch.triu_indices(defenders + 1, defenders + 1, 1, device=device)


def reference(batch, cfg):
    if not cfg.impulse_reference:
        return batch
    nominal = batch['nominal']
    ticks = torch.arange(1, nominal.shape[-2] + 1, dtype=nominal.dtype, device=nominal.device)
    velocity = batch['history'][:, None, :, -1, 2:4] * 5
    ballistic = batch['positions'][:, None, :, None] + velocity[..., None, :] * (ticks * cfg.dt)[None, None, None, :, None]
    first = nominal[..., :1, :] - ballistic[..., :1, :]
    return dict(batch, nominal=ballistic + first * (2 * ticks - 1)[None, None, None, :, None])


class TaskGeometryPredictor(guard.GuardPredictor):
    def __init__(self, cfg=TaskGeometryConfig()):
        super().__init__(cfg)
        self.order_embed = nn.Linear(3, cfg.width, bias=False)
        nn.init.zeros_(self.order_embed.weight)

    def encode_history(self, b):
        history = super().encode_history(b)
        entities = history.shape[1]
        index = torch.arange(entities, dtype=history.dtype, device=history.device)
        defender = index < entities - 1
        code = torch.stack((index / max(1, entities - 2), index / 8, defender.to(history.dtype)), -1) * defender[:, None]
        return history + self.order_embed(code)[None] * b['entity_valid'][..., None]

    def forward(self, b, history_cache=None):
        work = reference(b, self.cfg)
        out = super().forward(work, history_cache=history_cache)
        defenders = b['nominal'].shape[2] - 1
        i, j = all_pairs(defenders, b['nominal'].device)
        collision, pursuit = j < defenders, j == defenders
        out['all_mu'], out['all_sigma'] = out['mu'], out['sigma']
        out['all_mean_distance'] = out['mean_distance']
        out['all_pair_valid'] = out['pair_valid']
        out['all_nominal_mu'] = out['nominal_mu']
        for key in ('mu', 'sigma', 'mean_distance', 'nominal_mu', 'risk', 'det_risk'):
            out[key] = out[key][:, :, collision]
        out['pair_valid'] = out['pair_valid'][:, collision]
        out['pursuit_mu'] = out['all_mu'][:, :, pursuit]
        out['pursuit_sigma'] = out['all_sigma'][:, :, pursuit]
        out['pursuit_risk'] = guard.positive_gaussian_square(
            out['pursuit_mu'], out['pursuit_sigma'], -self.cfg.pursuit_radius)
        # The radial queries share one team target. Moment pooling permits
        # each defender query to describe its contribution without selecting
        # a spurious minimum over independently noisy learned predictions.
        valid = b['entity_valid'][:, :-1]
        count = valid.sum(-1)[:, None].clamp_min(1)
        mu = (out['guard_mu'] * valid[:, None]).sum(-1) / count
        moment = ((out['guard_sigma'].square() + out['guard_mu'].square()) * valid[:, None]).sum(-1) / count
        sigma = (moment - mu.square()).clamp_min(.0025).sqrt()
        out['team_mu'], out['team_sigma'] = mu, sigma
        out['team_risk'] = guard.positive_gaussian_square(mu, sigma, self.cfg.guard_margin)
        out['pursuit_weight'], out['pursuit_radius'] = self.cfg.pursuit_weight, self.cfg.pursuit_radius
        out['delta_weight'], out['delta_scale'] = self.cfg.delta_weight, self.cfg.delta_scale
        out['collision_source'] = self.cfg.collision_source
        return out


def targets(batch):
    xy, valid = batch['real_positions'], batch['time_valid']
    defenders = xy.shape[3] - 1
    i, j = all_pairs(defenders, xy.device)
    distances = (xy[:, :, :, i] - xy[:, :, :, j]).norm(dim=-1)
    distances = distances.masked_fill(~valid[:, :, :, None], torch.inf).amin(-1)
    radial = team.actual_team(batch)[:, :, :, 0]
    return distances, radial, j < defenders, j == defenders


def costs(out, batch, clearance=7.):
    if out['collision_source'] == 'nominal':
        collision = guard.physical_pair_cost(batch, clearance)
    else:
        collision = (out['risk'] * out['pair_valid'][:, None]).sum(-1)
    pursuit = out['pursuit_risk'].masked_fill(~batch['entity_valid'][:, None, :-1], torch.inf).amin(-1)
    return (batch['unary'] + out['objective_risk_weight'] * collision
            + out['guard_weight'] * out['team_risk'] + out['pursuit_weight'] * pursuit)


def task_objective(out, b, clearance=7., mode='full', risk_weight=1., rank_weight=.5,
                   temperature=2., **kwargs):
    distance, radial, collision, pursuit = targets(b)
    cv = b['candidate_valid']
    pm = out['all_pair_valid'][:, None, None] & cv[:, :, None, None]
    mean, sigma = out['all_mu'][:, :, None], out['all_sigma'][:, :, None]
    dnll = bounded_mean(.5 * ((distance - mean) / sigma).square() + sigma.log()
                       + .5 * math.log(2 * math.pi), pm)
    gm, gs = out['team_mu'][..., None], out['team_sigma'][..., None]
    gnll = bounded_mean(.5 * ((radial - gm) / gs).square() + gs.log()
                       + .5 * math.log(2 * math.pi), cv[..., None])
    true_coll = (clearance - distance[:, :, :, collision]).clamp_min(0).square().mean(2)
    true_guard = (out['guard_margin'] + radial).clamp_min(0).square().mean(-1)
    true_pursuit = (distance[:, :, :, pursuit] - out['pursuit_radius']).clamp_min(0).square().mean(2)
    risk = bounded_mean(F.smooth_l1_loss(out['team_risk'] / 36, true_guard / 36,
                                       reduction='none'), cv)
    risk = risk + bounded_mean(F.smooth_l1_loss(out['pursuit_risk'] / 100,
        true_pursuit / 100, reduction='none'), cv[..., None] & b['entity_valid'][:, None, :-1])
    risk = risk + bounded_mean(F.smooth_l1_loss(out['risk'] / 49, true_coll / 49,
        reduction='none'), cv[..., None] & out['pair_valid'][:, None])
    colcost = guard.physical_pair_cost(b, clearance) if out['collision_source'] == 'nominal' else (true_coll * out['pair_valid'][:, None]).sum(-1)
    empirical = (b['unary'] + out['objective_risk_weight'] * colcost
        + out['guard_weight'] * true_guard
        + out['pursuit_weight'] * true_pursuit.masked_fill(~b['entity_valid'][:, None, :-1], torch.inf).amin(-1))
    estimated = costs(out, b, clearance)
    target = (-empirical / temperature).masked_fill(~cv, -torch.inf).softmax(-1).detach()
    logp = (-estimated / temperature).masked_fill(~cv, -torch.inf).log_softmax(-1)
    rank = -(target * torch.where(cv, logp, torch.zeros_like(logp))).sum(-1).mean()
    actual = torch.cat((distance.mean(2), radial.mean(-1)[..., None]), -1)
    predicted = torch.cat((out['all_mu'], out['team_mu'][..., None]), -1)
    valid = torch.cat((out['all_pair_valid'], b['entity_valid'][:, -1:]), -1)[:, None] & cv[..., None]
    count = cv.sum(-1)[:, None, None].clamp_min(1)
    actual = actual - (actual * cv[..., None]).sum(1, keepdim=True) / count
    predicted = predicted - (predicted * cv[..., None]).sum(1, keepdim=True) / count
    contrast = bounded_mean(F.smooth_l1_loss(predicted / out['delta_scale'], actual / out['delta_scale'], reduction='none'), valid)
    nll = dnll + gnll
    total = out['guard_nll_weight'] * nll
    if mode == 'full':
        total = total + risk_weight * risk + rank_weight * rank + out['delta_weight'] * contrast
    return dict(total=total, motion=gm.sum() * 0, nll=nll, risk=risk, rank=rank,
                context=contrast, empirical_cost=empirical, estimated_cost=estimated)


class TaskGeometryController(Controller):
    def __init__(self, *args, task_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_config = task_config

    @torch.no_grad()
    def act(self, world):
        started = time.perf_counter()
        f, protocol = factorize(world, self.source), world.protocol
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), protocol.risk_weight, 128)
        initial, centre = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(), f['prior'].numpy(), self.bank, self.budget)
        history = None
        if self.model is not None:
            first, _, _ = inputs(world, f, initial[:1])
            history = self.model.encode_history({k: v.to(self.device) for k, v in first.items()})
        details = []

        def score(plans):
            b, _, _ = inputs(world, f, plans)
            b = {k: v.to(self.device) for k, v in b.items()}
            if self.model is not None:
                out = self.model(b, history_cache=history)
                out.update(objective_risk_weight=protocol.risk_weight,
                    guard_weight=self.task_config.guard_weight,
                    pursuit_weight=self.task_config.pursuit_weight,
                    collision_source=self.task_config.collision_source)
            else:
                cfg = self.task_config
                paths = reference(b, cfg)['nominal']
                i, j = pair_indices(paths.shape[2] - 1, paths.device)
                separation = (paths[:, :, i] - paths[:, :, j]).norm(dim=-1).masked_fill(~b['known_time_valid'][:, None, None], torch.inf).amin(-1)
                pursuit = (paths[:, :, :-1] - paths[:, :, -1:]).norm(dim=-1).masked_fill(~b['known_time_valid'][:, None, None], torch.inf).amin(-1)
                gap = team.nominal_team(dict(b, nominal=paths))[1][:, :, 0]
                out = dict(risk=(cfg.clearance - separation).clamp_min(0).square(),
                    pair_valid=b['entity_valid'][:, i] & b['entity_valid'][:, j],
                    pursuit_risk=(pursuit - cfg.pursuit_radius).clamp_min(0).square(),
                    team_risk=(gap + cfg.guard_margin).clamp_min(0).square(),
                    collision_source=cfg.collision_source, objective_risk_weight=protocol.risk_weight,
                    guard_weight=cfg.guard_weight, pursuit_weight=cfg.pursuit_weight)
            value = costs(out, b, protocol.risk_clearance)[0]
            details.append(dict(configurations=plans.tolist(), total_cost=value.cpu().tolist(),
                radial_cost_m2=out['team_risk'][0].cpu().tolist(),
                pursuit_cost_m2=out['pursuit_risk'][0].amin(-1).cpu().tolist()))
            return value.cpu().numpy()

        indices, trace = whole_configuration_search(initial, score, [row.nonzero()[:, 0].numpy() for row in f['valid']], self.budget, self.rounds)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        trace.update(gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            source_center_indices=centre.tolist(), source_center_scored=centre.tolist() in trace['configurations'],
            initial_configurations=initial.tolist(), geometric_score_batches=details,
            compute_seconds=time.perf_counter() - started)
        return force, trace


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--guard-weight', type=float, default=16.)
    parser.add_argument('--guard-margin', type=float, default=6.)
    parser.add_argument('--pursuit-weight', type=float, default=.25)
    parser.add_argument('--guard-nll-weight', type=float, default=1.)
    parser.add_argument('--collision-source', choices=['nominal', 'learned'], default='nominal')
    parser.add_argument('--impulse-reference', action='store_true')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    values = vars(args)
    if args.impulse_reference:
        values.update(prediction_condition='one action followed by fixed physical recourse', delta_scale=.1)
    pair.pair_indices = all_pairs
    guard.nominal_gaps = team.nominal_team
    if remaining and remaining[0] == 'train':
        sys.argv.pop(1)
        import citf.train as training
        training.Config = lambda **kwargs: TaskGeometryConfig(**values, **kwargs)
        training.Predictor = TaskGeometryPredictor
        training.objective = task_objective
        training.tree_hash = lambda path: canonical_hash({'core': tree_hash(path), 'joint_task_geometry': file_hash(__file__)})
        training.main()
    else:
        import citf.collect as collection
        import citf.evaluate as evaluation
        collection.Config, collection.Predictor = TaskGeometryConfig, TaskGeometryPredictor
        cfg = TaskGeometryConfig(**values)
        evaluation.Controller = lambda *positional, **kwargs: TaskGeometryController(*positional, task_config=cfg, **kwargs)
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['method'] = 'CITF-JointTaskGeometry-development' if result['checkpoint_sha256'] else 'JointTaskGeometry-Phys-development'
            result['controller'].update(values, prediction_targets='stopped all-pair distance and worst synchronized radial gap',
                guard_combination='learned without nominal lower bound')
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'joint_task_geometry': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        evaluation.main()


if __name__ == '__main__':
    main()
