"""Candidate-conditioned, stopped radial guarding geometry.

Each defender predicts its largest target-relative radial gap to the attacker
on the actually observed short branch. The controller chooses a configuration
that retains at least one consistent guard. This is a physical signed-distance
target, not a terminal value/outcome head. Other vessels' candidates can affect
the pair through both source feedback and the attacker's feedback maneuver.
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
from citf.risk import bounded_mean, pair_indices
from citf.search import nominal_beam, nominal_score, initialize_bank, whole_configuration_search
from experiments.pair_geometry import PairGeometryConfig, PairGeometryPredictor, PairContext


def positive_gaussian_square(mu, sigma, margin):
    dtype = mu.dtype
    a, s = mu.double() + margin, sigma.double()
    z = a / s
    value = (a.square() + s.square()) * torch.special.ndtr(z) + a * s * torch.exp(-.5 * z.square()) / math.sqrt(2 * math.pi)
    return value.clamp_min(0).to(dtype)


def nominal_gaps(b):
    radius = b['nominal'].norm(dim=-1)
    curve = radius[:, :, :-1] - radius[:, :, -1:]
    maximum = curve.masked_fill(~b['known_time_valid'][:, None, None], -torch.inf).amax(-1)
    return curve, maximum


def actual_gaps(b):
    radius = b['real_positions'].detach().norm(dim=-1)
    curve = radius[:, :, :, :-1] - radius[:, :, :, -1:]
    return curve.masked_fill(~b['time_valid'][:, :, :, None], -torch.inf).amax(-1)


def physical_pair_cost(b, clearance):
    i, j = pair_indices(b['nominal'].shape[2] - 1, b['nominal'].device)
    distance = (b['nominal'][:, :, i] - b['nominal'][:, :, j]).norm(dim=-1)
    distance = distance.masked_fill(~b['known_time_valid'][:, None, None], torch.inf).amin(-1)
    mask = b['entity_valid'][:, i] & b['entity_valid'][:, j]
    return ((clearance - distance).clamp_min(0).square() * mask[:, None]).sum(-1)


@dataclass(frozen=True)
class GuardConfig(PairGeometryConfig):
    revision: str = 'CITF-stopped-guard-query-v1'
    guard_weight: float = 4.
    guard_margin: float = 2.
    guard_nll_weight: float = .05
    collision_source: str = 'nominal'
    objective_risk_weight: float = 64.

    def __post_init__(self):
        super().__post_init__()
        if not all(math.isfinite(x) and x >= 0 for x in (self.guard_weight, self.guard_margin, self.guard_nll_weight, self.objective_risk_weight)):
            raise ValueError('Invalid guarding contract')
        if self.collision_source not in ('nominal', 'learned'):
            raise ValueError('Invalid collision prediction source')


class GuardPredictor(PairGeometryPredictor):
    def __init__(self, cfg=GuardConfig()):
        super().__init__(cfg)
        width = cfg.width
        self.guard_embed = nn.Sequential(nn.Linear(2 * width + cfg.horizon + 2, width),
                                         nn.GELU(), nn.Linear(width, width))
        self.guard_context = PairContext(width, cfg.heads)
        self.guard_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.guard_head[-1].weight)
        nn.init.zeros_(self.guard_head[-1].bias)
        with torch.no_grad():
            self.guard_head[-1].bias[1] = .5
        self._entity_features = None
        # Reuse the existing candidate encoder without changing its versioned
        # implementation or invoking a second candidate graph forward pass.
        self.motion.register_forward_pre_hook(self._capture_entities)

    def _capture_entities(self, module, arguments):
        self._entity_features = arguments[0]

    def forward(self, b, history_cache=None):
        history = self.encode_history(b) if history_cache is None else history_cache
        out = super().forward(b, history_cache=history)
        z = self._entity_features
        self._entity_features = None
        nominal, valid, known = b['nominal'], b['entity_valid'], b['known_time_valid']
        batch, candidates, entities, steps, _ = nominal.shape
        defenders, width = entities - 1, self.cfg.width
        curve, base = nominal_gaps(b)
        current_radius = b['positions'].norm(dim=-1)
        initial = current_radius[:, :-1] - current_radius[:, -1:]
        feature = torch.cat((z[:, :-1], z[:, -1:].expand(-1, defenders, -1),
            (curve * known[:, None, None] / 10).reshape(batch * candidates, defenders, steps),
            initial[:, None, :, None].expand(batch, candidates, defenders, 1).reshape(batch * candidates, defenders, 1) / 10,
            base.reshape(batch * candidates, defenders, 1) / 10), -1)
        query = self.guard_embed(feature)
        if self.cfg.fleet_candidate_context:
            paths = nominal.reshape(batch * candidates, entities, steps, 2)
            distance = (paths[:, None] - paths[:, :, None]).norm(dim=-1)
            distance = distance.masked_fill(~known[:, None].expand(batch, candidates, steps).reshape(batch * candidates, 1, 1, steps), torch.inf).amin(-1)
            keys = z
        else:
            positions = b['positions'][:, None].expand(batch, candidates, entities, 2).reshape(batch * candidates, entities, 2)
            distance = (positions[:, None] - positions[:, :, None]).norm(dim=-1)
            keys = history[:, None].expand(batch, candidates, entities, width).reshape(batch * candidates, entities, width)
        di, da = distance[:, :-1], distance[:, -1:].expand(-1, defenders, -1)
        ids = torch.arange(entities, device=nominal.device)
        own = (ids[None] == torch.arange(defenders, device=nominal.device)[:, None]).to(nominal.dtype)
        attacker = (ids == defenders).to(nominal.dtype)[None].expand(defenders, -1)
        geometry = torch.stack((torch.minimum(di, da) / 10, torch.maximum(di, da) / 20,
            (di - da).abs() / 20, own[None].expand(batch * candidates, -1, -1),
            attacker[None].expand(batch * candidates, -1, -1)), -1)
        mask = valid[:, None].expand(batch, candidates, entities).reshape(batch * candidates, entities)
        query = self.guard_context(query, keys, mask, geometry)
        q = self.guard_head(query).reshape(batch, candidates, defenders, 2)
        mu = base + q[..., 0]
        sigma = (F.softplus(q[..., 1]) + .05).clamp_max(8)
        out.update(guard_mu=mu, guard_sigma=sigma, nominal_guard_mu=base,
                   guard_risk=positive_gaussian_square(mu, sigma, self.cfg.guard_margin),
                   guard_weight=self.cfg.guard_weight, guard_margin=self.cfg.guard_margin,
                   guard_nll_weight=self.cfg.guard_nll_weight, collision_source=self.cfg.collision_source,
                   objective_risk_weight=self.cfg.objective_risk_weight)
        return out


def guard_objective(out, b, **kwargs):
    kwargs = dict(kwargs, weight=out['objective_risk_weight'])
    result = objective(out, b, **kwargs)
    target = actual_gaps(b)
    sigma = out['guard_sigma'][:, :, None]
    nll = .5 * ((target - out['guard_mu'][:, :, None]) / sigma).square() + sigma.log() + .5 * math.log(2 * math.pi)
    valid = b['entity_valid'][:, :-1]
    mask = valid[:, None, None] & b['candidate_valid'][:, :, None, None]
    guard_nll = bounded_mean(nll, mask)
    true_cost = (out['guard_margin'] + target).clamp_min(0).square().mean(2)
    true_cost = true_cost.masked_fill(~valid[:, None], torch.inf).amin(-1)
    estimated_guard = out['guard_risk'].masked_fill(~valid[:, None], torch.inf).amin(-1)
    if out['collision_source'] == 'nominal':
        base = b['unary'] + kwargs.get('weight', 32.) * physical_pair_cost(b, kwargs.get('clearance', 7.))
        empirical = base + out['guard_weight'] * true_cost
        estimated = base + out['guard_weight'] * estimated_guard
    else:
        empirical = result['empirical_cost'] + out['guard_weight'] * true_cost
        estimated = result['estimated_cost'] + out['guard_weight'] * estimated_guard
    cv = b['candidate_valid']
    temperature = kwargs.get('temperature', 8.)
    probabilities = (-empirical / temperature).masked_fill(~cv, -torch.inf).softmax(-1).detach()
    logp = (-estimated / temperature).masked_fill(~cv, -torch.inf).log_softmax(-1)
    rank = -(probabilities * torch.where(cv, logp, torch.zeros_like(logp))).sum(-1).mean()
    mode = kwargs.get('mode', 'full')
    if mode != 'motion_only':
        result['total'] = result['total'] + out['guard_nll_weight'] * guard_nll
    if mode == 'full':
        result['total'] = result['total'] + kwargs.get('rank_weight', .05) * (rank - result['rank'])
    result.update(rank=rank, guard_nll=guard_nll, empirical_cost=empirical, estimated_cost=estimated)
    return result


def implementation_hash(path):
    return canonical_hash({'core': tree_hash(path), 'guard_prediction': file_hash(__file__),
                           'pair_geometry': file_hash(Path(__file__).with_name('pair_geometry.py'))})


def train_main():
    import citf.train as training
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--guard-weight', type=float, default=4.)
    ap.add_argument('--guard-margin', type=float, default=2.)
    ap.add_argument('--guard-nll-weight', type=float, default=.05)
    ap.add_argument('--collision-source', choices=['nominal', 'learned'], default='nominal')
    ap.add_argument('--objective-risk-weight', type=float, default=64.)
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    original_evaluate = training.evaluate
    @torch.no_grad()
    def evaluate(model, data, batch_size=8, device='cpu', **kwargs):
        result = original_evaluate(model, data, batch_size=batch_size, device=device, **kwargs)
        count, absolute_error = 0, 0.
        for start in range(0, len(data['history']), batch_size):
            b = {k: v[start:start + batch_size].to(device) for k, v in data.items()}
            out = model(b)
            error = (actual_gaps(b) - out['guard_mu'][:, :, None]).abs()
            mask = torch.broadcast_to(b['entity_valid'][:, None, None, :-1] & b['candidate_valid'][:, :, None, None], error.shape)
            absolute_error += float(error[mask].sum())
            count += int(mask.sum())
        result['guard_gap_mae_m'] = absolute_error / count
        result['guard_gap_count'] = count
        return result
    training.Config = lambda **kwargs: GuardConfig(**vars(args), **kwargs)
    training.Predictor = GuardPredictor
    training.objective = guard_objective
    training.evaluate = evaluate
    training.tree_hash = implementation_hash
    training.main()


class PredictedGuardController(Controller):
    def __init__(self, *args, guard_weight, guard_margin, collision_source, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_weight, self.guard_margin = guard_weight, guard_margin
        self.collision_source = collision_source

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
                _, gap = nominal_gaps(b)
                guard = (self.guard_margin + gap).clamp_min(0).square().amin(-1)[0]
            else:
                out = self.model(b, history_cache=history)
                guard = positive_gaussian_square(out['guard_mu'], out['guard_sigma'], self.guard_margin).amin(-1)[0]
            if self.model is None or self.collision_source == 'nominal':
                base = torch.as_tensor(nominal_score(plans, f['unary'].numpy(), f['pair'].numpy(), p.risk_weight), device=guard.device)
            else:
                risk = out['risk'] if self.risk_mode == 'expected' else out['det_risk']
                base = (b['unary'] + p.risk_weight * (risk * out['pair_valid'][:, None]).sum(-1))[0]
            details.append({'configurations': plans.tolist(), 'guard_cost_m2': guard.cpu().tolist(), 'collision_and_prior_cost': base.cpu().tolist()})
            return (base + self.guard_weight * guard).cpu().numpy()
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
    ap.add_argument('--guard-weight', type=float, default=4.)
    ap.add_argument('--guard-margin', type=float, default=2.)
    ap.add_argument('--collision-source', choices=['nominal', 'learned'], default='nominal')
    args, remaining = ap.parse_known_args()
    GuardConfig(**vars(args))
    sys.argv = [sys.argv[0], *remaining]
    import citf.collect as collection
    import citf.evaluate as evaluation
    collection.Config, collection.Predictor = GuardConfig, GuardPredictor
    original_identity = evaluation.runtime_identity
    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['method'] = 'CITF-StoppedGuard-development' if result['checkpoint_sha256'] else 'StoppedGuard-Phys-development'
        result['controller'].update(vars(args))
        result['controller']['guard_target'] = 'per-defender max signed radial gap on observed stopped interval; minimum expected positive-square cost across defenders'
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'guard_prediction': file_hash(__file__),
            'pair_geometry': file_hash(Path(__file__).with_name('pair_geometry.py'))})
        return result
    evaluation.Controller = lambda *positional, **kwargs: PredictedGuardController(*positional, **vars(args), **kwargs)
    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        train_main()
    else:
        evaluate_main()
