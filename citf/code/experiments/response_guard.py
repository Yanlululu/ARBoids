"""Expose other vessels' control changes directly in attention values.

The history and focal-candidate query determine which teammates matter. The
value paths multiply each teammate's two current fusion coefficients relative
to 0.5, so their contribution is linear when the focal candidate is fixed.
The own-candidate radial response remains nonlinear. This inductive bias is
suited to the matched single-vessel changes in the local candidate data.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from citf.risk import pair_indices
from experiments import guard_envelope
from experiments import guard_focus as focus
from experiments import guard_prediction as guard
from experiments.ordered_guard import OrderedConfig, OrderedPredictor


@dataclass(frozen=True)
class ResponseConfig(OrderedConfig):
    revision: str = 'CITF-control-response-attention-guard-v1'
    response_reference: float = .5


class ResponseAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(width)
        self.key_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, 2 * width)
        self.geometry = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, heads))
        self.readout = nn.Linear(width, 2, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, query, history, change, valid, geometry):
        batch, candidates, defenders, width = query.shape
        q = self.query(self.query_norm(query)).reshape(
            batch, candidates, defenders, self.heads, -1).permute(0, 1, 3, 2, 4)
        h = self.key_norm(history)
        key = self.key(h).reshape(batch, defenders, self.heads, -1).transpose(1, 2)
        weight = q @ key[:, None].transpose(-1, -2) / math.sqrt(width // self.heads)
        weight = weight + self.geometry(geometry).permute(0, 3, 1, 2)[:, None]
        other = ~torch.eye(defenders, dtype=torch.bool, device=query.device)
        mask = valid[:, None, None, None] & other[None, None, None]
        weight = weight.masked_fill(~mask, -torch.inf).softmax(-1)
        # Values are affine in the queried other-vessel control. No other
        # candidate enters the query, attention weights or history keys.
        value = self.value(h).reshape(batch, defenders, 2, width)
        value = (value[:, None] * change[..., None]).sum(-2)
        value = value.reshape(batch, candidates, defenders, self.heads, -1).transpose(2, 3)
        response = (weight @ value).transpose(2, 3).reshape(batch, candidates, defenders, width)
        return self.readout(response)


class ResponsePredictor(OrderedPredictor):
    def __init__(self, cfg=ResponseConfig()):
        super().__init__(cfg)
        for name in ('context', 'motion', 'pair_embed', 'pair_context',
                     'distance_head', 'guard_context'):
            delattr(self, name)
        self.response = ResponseAttention(cfg.width, cfg.heads)

    def forward(self, b, history_cache=None):
        nominal = b['nominal']
        batch, candidates, entities, steps, _ = nominal.shape
        defenders, width = entities - 1, self.cfg.width
        history = self.encode_history(b) if history_cache is None else history_cache
        known, valid = b['known_time_valid'], b['entity_valid']
        displacement = ((nominal - b['positions'][:, None, :, None]) / 10
                        * known[:, None, None, :, None])
        own = history[:, None] + self.candidate_embed(
            torch.cat((b['candidate'], displacement.flatten(-2)), -1))
        curve, base = guard.nominal_gaps(b)
        radius = b['positions'].norm(dim=-1)
        initial = radius[:, :-1] - radius[:, -1:]
        feature = torch.cat((own[:, :, :-1], history[:, None, -1:, :].expand(
            batch, candidates, defenders, width), curve * known[:, None, None] / 10,
            initial[:, None, :, None].expand(batch, candidates, defenders, 1) / 10,
            base[..., None] / 10), -1)
        query = self.guard_embed(feature)
        parameters = self.guard_head(query)
        if self.cfg.fleet_candidate_context:
            positions = b['positions'][:, :-1]
            relative = positions[:, None] - positions[:, :, None]
            distance = relative.norm(dim=-1)
            radial_difference = radius[:, None, :-1] - radius[:, :-1, None]
            geometry = torch.cat((relative / 20, distance[..., None] / 10,
                                  radial_difference[..., None] / 10), -1)
            change = b['candidate'][:, :, :-1, :2] - self.cfg.response_reference
            parameters = parameters + self.response(query, history[:, :-1], change,
                                                    valid[:, :-1], geometry)
        mean = base + parameters[..., 0]
        sigma = (F.softplus(parameters[..., 1]) + .05).clamp_max(8)
        i, j = pair_indices(defenders, nominal.device)
        separation = (nominal[:, :, i] - nominal[:, :, j]).norm(dim=-1)
        separation = separation.masked_fill(~known[:, None, None], torch.inf).amin(-1)
        pair_risk = (self.cfg.clearance - separation).clamp_min(0).square()
        return dict(xy=nominal, mu=separation, sigma=torch.full_like(separation, .03),
            mean_distance=separation, nominal_mu=separation, risk=pair_risk,
            det_risk=pair_risk, pair_valid=valid[:, i] & valid[:, j],
            guard_mu=mean, guard_sigma=sigma, nominal_guard_mu=base,
            guard_risk=guard.positive_gaussian_square(mean, sigma, self.cfg.guard_margin),
            guard_weight=self.cfg.guard_weight, guard_margin=self.cfg.guard_margin,
            guard_nll_weight=self.cfg.guard_nll_weight,
            collision_source=self.cfg.collision_source,
            objective_risk_weight=self.cfg.objective_risk_weight)


def main():
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'response_guard': file_hash(__file__),
        'ordered_guard': file_hash(Path(__file__).with_name('ordered_guard.py'))})
    guard.GuardPredictor = ResponsePredictor
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        focus.FocusConfig = ResponseConfig
        focus.main()
    else:
        import citf.evaluate as evaluation
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller']['radial_predictor'] = ResponseConfig().revision
            result['runtime_code'] = canonical_hash({
                'core': result['runtime_code'], 'response_guard': file_hash(__file__),
                'ordered_guard': file_hash(Path(__file__).with_name('ordered_guard.py'))})
            return result

        evaluation.runtime_identity = identity
        guard.GuardConfig = ResponseConfig
        guard_envelope.main()


if __name__ == '__main__':
    main()
