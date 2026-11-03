"""Predict the stopped, synchronized radial coverage of a whole configuration.

The physical target is max_t min_i (radius_i(t) - radius_attacker(t)).
Different defenders may supply the inner guard at different times. This is
a finite-interval signed geometric margin, with no outcome or value target.
The scalar distribution is broadcast over valid defenders only to reuse the
existing radial training and controller interfaces without changing them.
"""
import argparse
from dataclasses import dataclass
import sys

import torch
from torch import nn
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from citf.risk import pair_indices
from experiments import guard_prediction as guard
from experiments import guard_focus as focus
from experiments import guard_envelope
from experiments.ordered_guard import OrderedConfig, OrderedPredictor


TARGET = 'maximum over valid stopped times of minimum defender signed radial gap'


def nominal_team(b):
    radius = b['nominal'].norm(dim=-1)
    gaps = radius[:, :, :-1] - radius[:, :, -1:]
    gaps = gaps.masked_fill(~b['entity_valid'][:, None, :-1, None], torch.inf)
    curve = gaps.amin(-2)
    maximum = curve.masked_fill(~b['known_time_valid'][:, None], -torch.inf).amax(-1)
    defenders = gaps.shape[-2]
    return (curve[:, :, None].expand(-1, -1, defenders, -1),
            maximum[:, :, None].expand(-1, -1, defenders))


def actual_team(b):
    radius = b['real_positions'].detach().norm(dim=-1)
    gaps = radius[:, :, :, :-1] - radius[:, :, :, -1:]
    gaps = gaps.masked_fill(~b['entity_valid'][:, None, None, :-1, None], torch.inf)
    curve = gaps.amin(-2)
    maximum = curve.masked_fill(~b['time_valid'], -torch.inf).amax(-1)
    return maximum[..., None].expand(-1, -1, -1, gaps.shape[-2])


@dataclass(frozen=True)
class SynchronizedConfig(OrderedConfig):
    revision: str = 'CITF-synchronized-team-guard-v1'
    geometric_target: str = TARGET


class SynchronizedPredictor(OrderedPredictor):
    def __init__(self, cfg=SynchronizedConfig()):
        super().__init__(cfg)
        for name in ('motion', 'pair_embed', 'pair_context', 'distance_head'):
            delattr(self, name)

    def forward(self, b, history_cache=None):
        nominal, positions = b['nominal'], b['positions']
        valid, known = b['entity_valid'], b['known_time_valid']
        batch, candidates, entities, steps, _ = nominal.shape
        defenders, width = entities - 1, self.cfg.width
        history = self.encode_history(b) if history_cache is None else history_cache
        displacement = ((nominal - positions[:, None, :, None]) / 10
                        * known[:, None, None, :, None])
        z = (history[:, None] + self.candidate_embed(torch.cat(
            (b['candidate'], displacement.flatten(-2)), -1))).reshape(
                batch * candidates, entities, width)
        mask = valid[:, None].expand(batch, candidates, entities).reshape(
            batch * candidates, entities)
        paths = nominal.reshape(batch * candidates, entities, steps, 2)
        current = positions[:, None].expand(batch, candidates, entities, 2).reshape(
            batch * candidates, entities, 2)
        known_flat = known[:, None].expand(batch, candidates, steps).reshape(
            batch * candidates, steps)
        distance = (paths[:, None] - paths[:, :, None]).norm(dim=-1)
        nearest, when = distance.masked_fill(~known_flat[:, None, None], torch.inf).min(-1)
        last = (known_flat.sum(-1) - 1)[:, None, None, None].expand(-1, entities, entities, 1)
        last_distance = distance.gather(-1, last).squeeze(-1)
        relative = current[:, None] - current[:, :, None]
        current_distance = relative.norm(dim=-1)
        edges = torch.cat((relative / 20, nearest[..., None] / 10,
                           ((last_distance - current_distance) / 10)[..., None],
                           (when / steps)[..., None]), -1)
        if self.cfg.fleet_candidate_context:
            for block in self.context:
                z = block(z, mask, edges)
        z = z * mask[..., None]
        radial = paths.norm(dim=-1)
        own_gap = radial[:, :-1] - radial[:, -1:]
        # Smooth geometric pooling preserves contributions from alternative
        # guarding vessels, including nominal handovers within the horizon.
        ownership = (-own_gap / 2).masked_fill(~mask[:, :-1, None], -torch.inf).softmax(1)
        ownership = (ownership * known_flat[:, None]).sum(-1) / known_flat.sum(-1, keepdim=True)
        pooled = (z[:, :-1] * ownership[..., None]).sum(1, keepdim=True)
        curve, base = nominal_team(b)
        curve_flat = curve[:, :, 0].reshape(batch * candidates, steps)
        base_flat = base[:, :, 0].reshape(batch * candidates, 1)
        current_radius = current.norm(dim=-1)
        initial = (current_radius[:, :-1] - current_radius[:, -1:]).masked_fill(
            ~mask[:, :-1], torch.inf).amin(-1, keepdim=True)
        feature = torch.cat((pooled[:, 0], z[:, -1], curve_flat * known_flat / 10,
                             initial / 10, base_flat / 10), -1)
        query = self.guard_embed(feature)[:, None]
        if self.cfg.fleet_candidate_context:
            radius_feature = radial.masked_fill(~known_flat[:, None], 0).sum(-1) / known_flat.sum(-1, keepdim=True)
            geometry = torch.stack((nearest[:, -1] / 10, current_distance[:, -1] / 20,
                (radius_feature - radius_feature[:, -1:]) / 20,
                F.pad(ownership, (0, 1)),
                torch.arange(entities, device=nominal.device).eq(defenders).to(nominal.dtype)[None].expand(batch * candidates, -1)), -1)
            query = self.guard_context(query, z, mask, geometry[:, None])
        parameters = self.guard_head(query).reshape(batch, candidates, 2)
        mean = base[:, :, 0] + parameters[..., 0]
        sigma = (F.softplus(parameters[..., 1]) + .05).clamp_max(8)
        mean, sigma = (x[..., None].expand(-1, -1, defenders) for x in (mean, sigma))
        i, j = pair_indices(defenders, nominal.device)
        separation = (nominal[:, :, i] - nominal[:, :, j]).norm(dim=-1)
        separation = separation.masked_fill(~known[:, None, None], torch.inf).amin(-1)
        risk = (self.cfg.clearance - separation).clamp_min(0).square()
        return dict(xy=nominal, mu=separation, sigma=torch.full_like(separation, .03),
            mean_distance=separation, nominal_mu=separation, risk=risk, det_risk=risk,
            pair_valid=valid[:, i] & valid[:, j], guard_mu=mean, guard_sigma=sigma,
            nominal_guard_mu=base,
            guard_risk=guard.positive_gaussian_square(mean, sigma, self.cfg.guard_margin),
            guard_weight=self.cfg.guard_weight, guard_margin=self.cfg.guard_margin,
            guard_nll_weight=self.cfg.guard_nll_weight, collision_source=self.cfg.collision_source,
            objective_risk_weight=self.cfg.objective_risk_weight)


def main():
    guard.nominal_gaps, guard.actual_gaps = nominal_team, actual_team
    guard.GuardPredictor = SynchronizedPredictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'synchronized_guard': file_hash(__file__),
        'geometric_target': TARGET})
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        focus.FocusConfig = SynchronizedConfig
        focus.main()
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--guard-weight', type=float, default=16.)
        parser.add_argument('--guard-margin', type=float, default=6.)
        args, remaining = parser.parse_known_args()
        sys.argv = [sys.argv[0], *remaining]
        import citf.collect as collection
        import citf.evaluate as evaluation
        collection.Config, collection.Predictor = SynchronizedConfig, SynchronizedPredictor
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['method'] = ('CITF-SynchronizedGuard-development' if result['checkpoint_sha256']
                                else 'SynchronizedGuard-Phys-development')
            result['controller'].update(vars(args), guard_target=TARGET,
                collision_source='nominal', guard_combination='maximum of nominal and learned team risk')
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                'synchronized_guard': file_hash(__file__),
                'guard_envelope': file_hash(guard_envelope.__file__)})
            return result

        evaluation.runtime_identity = identity
        evaluation.Controller = lambda *positional, **kwargs: guard_envelope.EnvelopeController(
            *positional, **vars(args), collision_source='nominal', **kwargs)
        evaluation.main()


if __name__ == '__main__':
    main()
