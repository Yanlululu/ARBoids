"""Direct stopped encounter geometry for candidate-interaction control.

The two targets are physical distances: each defender's closest approach to
the attacker and largest signed radial gap over the observed short branch.
Neither target is a terminal return, success classifier or unobserved tail.
"""
import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from citf.bridge import Controller, factorize, inputs
from citf.contracts import canonical_hash, file_hash, tree_hash
from citf.losses import objective
from citf.risk import bounded_mean, rectified_gaussian_nll
from citf.search import nominal_beam, nominal_score, initialize_bank, whole_configuration_search
from experiments.guard_prediction import positive_gaussian_square, physical_pair_cost
from experiments.pair_geometry import PairGeometryConfig, PairGeometryPredictor, PairContext


def positive_gaussian_mean(mu, sigma, offset=0.):
    a, s = mu.double() + offset, sigma.double()
    z = a / s
    value = a * torch.special.ndtr(z) + s * torch.exp(-.5 * z.square()) / math.sqrt(2 * math.pi)
    return value.clamp_min(0).to(mu.dtype)


def geometry_targets(paths, time_valid):
    """Works with [B,C,(F),E,T,2], returning [...,defenders]."""
    distance = (paths[..., :-1, :, :] - paths[..., -1:, :, :]).norm(dim=-1)
    radius = paths.norm(dim=-1)
    gap = radius[..., :-1, :] - radius[..., -1:, :]
    valid = time_valid.unsqueeze(-2)
    closest = distance.masked_fill(~valid, torch.inf).amin(-1)
    largest_gap = gap.masked_fill(~valid, -torch.inf).amax(-1)
    return closest, largest_gap, distance, gap


@dataclass(frozen=True)
class EncounterConfig(PairGeometryConfig):
    revision: str = 'CITF-stopped-encounter-query-v1'
    encounter_weight: float = 4.
    guard_weight: float = 4.
    guard_margin: float = 2.
    capture_radius: float = 5.
    encounter_nll_weight: float = .08
    guard_nll_weight: float = .05
    effect_weight: float = .25
    objective_risk_weight: float = 64.

    def __post_init__(self):
        super().__post_init__()
        if not all(math.isfinite(x) and x >= 0 for x in (self.encounter_weight, self.guard_weight,
                self.guard_margin, self.capture_radius, self.encounter_nll_weight,
                self.guard_nll_weight, self.effect_weight, self.objective_risk_weight)):
            raise ValueError('Invalid stopped encounter contract')


class EncounterPredictor(PairGeometryPredictor):
    def __init__(self, cfg=EncounterConfig()):
        super().__init__(cfg)
        width = cfg.width
        self.encounter_embed = nn.Sequential(nn.Linear(2 * width + 3 * cfg.horizon + 4, width),
                                             nn.GELU(), nn.Linear(width, width))
        self.encounter_context = PairContext(width, cfg.heads)
        self.encounter_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))
        nn.init.zeros_(self.encounter_head[-1].weight)
        nn.init.zeros_(self.encounter_head[-1].bias)
        with torch.no_grad():
            self.encounter_head[-1].bias[[1, 3]] = .5
        self._entity_features = None
        self.motion.register_forward_pre_hook(self._capture_entities)

    def _capture_entities(self, module, arguments):
        self._entity_features = arguments[0]

    def forward(self, b, history_cache=None):
        history = self.encode_history(b) if history_cache is None else history_cache
        out = super().forward(b, history_cache=history)
        z, self._entity_features = self._entity_features, None
        nominal, valid, known = b['nominal'], b['entity_valid'], b['known_time_valid']
        batch, candidates, entities, steps, _ = nominal.shape
        defenders, width = entities - 1, self.cfg.width
        closest, gap, distance_curve, gap_curve = geometry_targets(nominal, known[:, None])
        current_distance = (b['positions'][:, :-1] - b['positions'][:, -1:]).norm(dim=-1)
        current_radius = b['positions'].norm(dim=-1)
        current_gap = current_radius[:, :-1] - current_radius[:, -1:]
        initial_distance = current_distance[:, None].expand(batch, candidates, defenders)
        initial_gap = current_gap[:, None].expand(batch, candidates, defenders)
        closing = (distance_curve - torch.cat((initial_distance[..., None], distance_curve[..., :-1]), -1)) / self.cfg.dt
        curves = torch.cat((distance_curve / 20, gap_curve / 20, closing / 5), -1)
        curve_mask = known[:, None, None].expand(batch, candidates, defenders, steps).repeat(1, 1, 1, 3)
        scalars = torch.stack((initial_distance / 20, initial_gap / 20, closest / 20, gap / 20), -1)
        features = torch.cat((z[:, :-1], z[:, -1:].expand(-1, defenders, -1),
                              (curves * curve_mask).reshape(batch * candidates, defenders, 3 * steps),
                              scalars.reshape(batch * candidates, defenders, 4)), -1)
        query = self.encounter_embed(features)
        if self.cfg.fleet_candidate_context:
            paths = nominal.reshape(batch * candidates, entities, steps, 2)
            distances = (paths[:, None] - paths[:, :, None]).norm(dim=-1)
            distances = distances.masked_fill(~known[:, None].expand(batch, candidates, steps).reshape(batch * candidates, 1, 1, steps), torch.inf).amin(-1)
            keys = z
        else:
            positions = b['positions'][:, None].expand(batch, candidates, entities, 2).reshape(batch * candidates, entities, 2)
            distances = (positions[:, None] - positions[:, :, None]).norm(dim=-1)
            keys = history[:, None].expand(batch, candidates, entities, width).reshape(batch * candidates, entities, width)
        di, da = distances[:, :-1], distances[:, -1:].expand(-1, defenders, -1)
        ids = torch.arange(entities, device=nominal.device)
        own = (ids[None] == torch.arange(defenders, device=nominal.device)[:, None]).to(nominal.dtype)
        enemy = (ids == defenders).to(nominal.dtype)[None].expand(defenders, -1)
        geometry = torch.stack((torch.minimum(di, da) / 10, torch.maximum(di, da) / 20,
            (di - da).abs() / 20, own[None].expand(batch * candidates, -1, -1),
            enemy[None].expand(batch * candidates, -1, -1)), -1)
        mask = valid[:, None].expand(batch, candidates, entities).reshape(batch * candidates, entities)
        query = self.encounter_context(query, keys, mask, geometry)
        q = self.encounter_head(query).reshape(batch, candidates, defenders, 4)
        encounter_mu, guard_mu = closest + q[..., 0], gap + q[..., 2]
        encounter_sigma, guard_sigma = (F.softplus(q[..., 1]) + .05).clamp_max(8), (F.softplus(q[..., 3]) + .05).clamp_max(8)
        out.update(encounter_mu=encounter_mu, encounter_sigma=encounter_sigma, guard_mu=guard_mu,
                   guard_sigma=guard_sigma, encounter_config=self.cfg)
        return out


def geometry_cost(encounter_mu, encounter_sigma, guard_mu, guard_sigma, valid, cfg):
    pursuit = positive_gaussian_mean(encounter_mu, encounter_sigma, -cfg.capture_radius)
    guarding = positive_gaussian_square(guard_mu, guard_sigma, cfg.guard_margin)
    return cfg.encounter_weight * pursuit.masked_fill(~valid[:, None], torch.inf).amin(-1) + cfg.guard_weight * guarding.masked_fill(~valid[:, None], torch.inf).amin(-1)


def encounter_objective(out, b, **kwargs):
    cfg = out['encounter_config']
    kwargs = dict(kwargs, weight=cfg.objective_risk_weight)
    result = objective(out, b, **kwargs)
    distance, gap, _, _ = geometry_targets(b['real_positions'].detach(), b['time_valid'])
    valid, cv = b['entity_valid'][:, :-1], b['candidate_valid']
    mask = valid[:, None, None] & cv[:, :, None, None]
    distance_nll = bounded_mean(rectified_gaussian_nll(distance, out['encounter_mu'][:, :, None], out['encounter_sigma'][:, :, None]), mask)
    gs = out['guard_sigma'][:, :, None]
    gap_nll = bounded_mean(.5 * ((gap - out['guard_mu'][:, :, None]) / gs).square() + gs.log() + .5 * math.log(2 * math.pi), mask)
    true_pursuit = (distance - cfg.capture_radius).clamp_min(0).mean(2).masked_fill(~valid[:, None], torch.inf).amin(-1)
    true_guard = (gap + cfg.guard_margin).clamp_min(0).square().mean(2).masked_fill(~valid[:, None], torch.inf).amin(-1)
    truth_extra = cfg.encounter_weight * true_pursuit + cfg.guard_weight * true_guard
    prediction_extra = geometry_cost(out['encounter_mu'], out['encounter_sigma'], out['guard_mu'], out['guard_sigma'], valid, cfg)
    base = b['unary'] + cfg.objective_risk_weight * physical_pair_cost(b, kwargs.get('clearance', 7.))
    empirical, estimated = base + truth_extra, base + prediction_extra
    temperature = kwargs.get('temperature', 8.)
    probability = (-empirical / temperature).masked_fill(~cv, -torch.inf).softmax(-1).detach()
    logp = (-estimated / temperature).masked_fill(~cv, -torch.inf).log_softmax(-1)
    rank = -(probability * torch.where(cv, logp, torch.zeros_like(logp))).sum(-1).mean()
    residual = prediction_extra - truth_extra
    centre = (residual * cv).sum(-1, keepdim=True) / cv.sum(-1, keepdim=True)
    effect = bounded_mean(F.smooth_l1_loss((residual - centre) / 4., torch.zeros_like(residual), reduction='none'), cv)
    mode = kwargs.get('mode', 'full')
    if mode != 'motion_only':
        result['total'] = result['total'] + cfg.encounter_nll_weight * distance_nll + cfg.guard_nll_weight * gap_nll
    if mode == 'full':
        result['total'] = result['total'] + kwargs.get('rank_weight', .05) * (rank - result['rank']) + cfg.effect_weight * effect
    result.update(rank=rank, encounter_nll=distance_nll, guard_nll=gap_nll, geometry_effect=effect,
                  empirical_cost=empirical, estimated_cost=estimated)
    return result


def implementation_hash(*args, **kwargs):
    return canonical_hash({'core': tree_hash(*args, **kwargs), 'encounter': file_hash(__file__),
        'guard': file_hash(Path(__file__).with_name('guard_prediction.py')),
        'pair': file_hash(Path(__file__).with_name('pair_geometry.py'))})


def train_main():
    import citf.train as training
    ap = argparse.ArgumentParser(add_help=False)
    for key in ('encounter_weight', 'guard_weight', 'guard_margin', 'encounter_nll_weight', 'guard_nll_weight', 'effect_weight', 'objective_risk_weight'):
        ap.add_argument('--' + key.replace('_', '-'), type=float, default=getattr(EncounterConfig(), key))
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    training.Config = lambda **kwargs: EncounterConfig(**vars(args), **kwargs)
    training.Predictor = EncounterPredictor
    training.objective = encounter_objective
    training.tree_hash = implementation_hash
    training.main()


class EncounterController(Controller):
    def __init__(self, *args, encounter_weight, guard_weight, guard_margin, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_config = EncounterConfig(encounter_weight=encounter_weight, guard_weight=guard_weight, guard_margin=guard_margin)

    @torch.no_grad()
    def act(self, world):
        start = time.perf_counter()
        f, p = factorize(world, self.source), world.protocol
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), p.risk_weight, 128)
        initial, centre = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(), f['prior'].numpy(), self.bank, self.budget)
        history = None
        if self.model is not None:
            first, _, _ = inputs(world, f, initial[:1])
            history = self.model.encode_history({k: v.to(self.device) for k, v in first.items()})
        details = []
        def score(plans):
            b, _, _ = inputs(world, f, plans)
            b = {k: v.to(self.device) for k, v in b.items()}
            if self.model is None:
                distance, gap, _, _ = geometry_targets(b['nominal'], b['known_time_valid'][:, None])
                extra = self.cost_config.encounter_weight * (distance - p.capture_radius).clamp_min(0).amin(-1)
                extra += self.cost_config.guard_weight * (gap + self.cost_config.guard_margin).clamp_min(0).square().amin(-1)
            else:
                o = self.model(b, history_cache=history)
                extra = geometry_cost(o['encounter_mu'], o['encounter_sigma'], o['guard_mu'], o['guard_sigma'], b['entity_valid'][:, :-1], self.cost_config)
            base = nominal_score(plans, f['unary'].numpy(), f['pair'].numpy(), p.risk_weight)
            extra = extra[0].cpu().numpy()
            details.append({'configurations': plans.tolist(), 'encounter_guard_cost': extra.tolist(), 'collision_and_prior_cost': base.tolist()})
            return base + extra
        indices, trace = whole_configuration_search(initial, score, [row.nonzero()[:, 0].numpy() for row in f['valid']], self.budget, self.rounds)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        trace.update(gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            source_center_indices=centre.tolist(), source_center_scored=centre.tolist() in trace['configurations'],
            initial_configurations=initial.tolist(), geometric_score_batches=details,
            compute_seconds=time.perf_counter() - start)
        return force, trace


def evaluate_main():
    ap = argparse.ArgumentParser(add_help=False)
    for key in ('encounter_weight', 'guard_weight', 'guard_margin'):
        ap.add_argument('--' + key.replace('_', '-'), type=float, default=getattr(EncounterConfig(), key))
    args, remaining = ap.parse_known_args()
    EncounterConfig(**vars(args))
    sys.argv = [sys.argv[0], *remaining]
    import citf.collect as collection
    import citf.evaluate as evaluation
    collection.Config, collection.Predictor = EncounterConfig, EncounterPredictor
    original_identity = evaluation.runtime_identity
    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['method'] = 'CITF-StoppedEncounter-development' if result['checkpoint_sha256'] else 'StoppedEncounter-Phys-development'
        result['controller'].update(vars(args))
        result['controller']['geometry_target'] = 'per-defender closest attacker distance and largest signed radial gap on stopped interval'
        result['controller']['collision_source'] = 'nominal'
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'encounter': file_hash(__file__),
            'guard': file_hash(Path(__file__).with_name('guard_prediction.py')),
            'pair': file_hash(Path(__file__).with_name('pair_geometry.py'))})
        return result
    evaluation.Controller = lambda *positional, **kwargs: EncounterController(*positional, **vars(args), **kwargs)
    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        train_main()
    else:
        evaluate_main()
