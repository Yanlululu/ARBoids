"""Separate a common geometric offset from small joint-control responses.

Observed-history attention supplies a state-dependent offset and uncertainty.
Candidate effects use the physically computed first-interval impulse, with
bounded linear and pairwise quadratic response coefficients. This preserves
the scale of a single executed action and prevents candidate-specific noise
in the predicted uncertainty from determining the configuration ordering.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from citf.risk import pair_indices
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team
from experiments import synchronized_delta as precise
from experiments.ordered_guard import OrderedPredictor


@dataclass(frozen=True)
class StructuredConfig(precise.PreciseTeamConfig):
    revision: str = 'CITF-state-offset-joint-impulse-response-v1'
    candidate_response: str = 'bounded linear and pairwise quadratic physical impulse'
    candidate_independent_uncertainty: bool = True


class StructuredPredictor(OrderedPredictor):
    def __init__(self, config=StructuredConfig()):
        super().__init__(config)
        for name in ('candidate_embed', 'context', 'motion', 'pair_embed',
                     'pair_context', 'distance_head', 'guard_embed',
                     'guard_context', 'guard_head'):
            delattr(self, name)
        width = config.width
        self.offset = nn.Sequential(nn.LayerNorm(2 * width + 2),
            nn.Linear(2 * width + 2, width), nn.GELU(), nn.Linear(width, 2))
        self.linear_response = nn.Sequential(nn.Linear(3 * width + 3, width),
            nn.GELU(), nn.Linear(width, 2))
        self.pair_response = nn.Sequential(nn.Linear(2 * width + 3, width),
            nn.GELU(), nn.Linear(width, 4))
        for module in (self.offset, self.linear_response, self.pair_response):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        with torch.no_grad():
            self.offset[-1].bias[1] = .5

    def forward(self, batch, history_cache=None):
        nominal = batch['nominal']
        valid, known = batch['entity_valid'], batch['known_time_valid']
        count, candidates, entities, steps, _ = nominal.shape
        defenders = entities - 1
        history = self.encode_history(batch) if history_cache is None else history_cache
        positions = batch['positions']
        ticks = torch.arange(1, steps + 1, dtype=nominal.dtype, device=nominal.device)
        velocity = batch['history'][:, None, :, -1, 2:4] * 5
        ballistic = (positions[:, None, :, None]
                     + velocity[..., None, :] * (ticks * self.cfg.dt)[None, None, None, :, None])
        first_effect = nominal[..., :1, :] - ballistic[..., :1, :]
        impulse_path = ballistic + first_effect * (2 * ticks - 1)[None, None, None, :, None]
        base = team.nominal_team(dict(batch, nominal=impulse_path))[1][:, :, 0]
        # Use the current known horizon, never the future branch stop time.
        impulse = first_effect[:, :, :-1, 0] * (2 * known.sum(-1) - 1)[:, None, None, None]
        radius = positions.norm(dim=-1)
        current_gap = radius[:, :-1] - radius[:, -1:]
        owner = (-current_gap / 2).masked_fill(~valid[:, :-1], -torch.inf).softmax(-1)
        pooled = (history[:, :-1] * owner[..., None]).sum(1)
        initial_gap = current_gap.masked_fill(~valid[:, :-1], torch.inf).amin(-1, keepdim=True)
        state = torch.cat((pooled, history[:, -1], initial_gap / 10,
                           known.sum(-1, keepdim=True) * self.cfg.dt / 4), -1)
        parameters = self.offset(state)
        relative = positions[:, :-1] - positions[:, -1:]
        linear_features = torch.cat((history[:, :-1],
            history[:, -1:].expand(-1, defenders, -1),
            pooled[:, None].expand(-1, defenders, -1),
            relative / 20, current_gap[..., None] / 10), -1)
        linear = 2 * torch.tanh(self.linear_response(linear_features))
        linear = linear * valid[:, :-1, None]
        response = (linear[:, None] * impulse).sum((-1, -2)) / math.sqrt(defenders)
        if self.cfg.fleet_candidate_context:
            i, j = torch.triu_indices(defenders, defenders, 1, device=nominal.device)
            delta = positions[:, i] - positions[:, j]
            features = torch.cat((history[:, i] + history[:, j],
                history[:, i] - history[:, j], delta / 20,
                (radius[:, i] - radius[:, j])[..., None] / 10), -1)
            coefficient = torch.tanh(self.pair_response(features)).reshape(count, -1, 2, 2)
            pair_valid = valid[:, i] & valid[:, j]
            quadratic = torch.einsum('bcpi,bpij,bcpj->bcp',
                                      impulse[:, :, i], coefficient, impulse[:, :, j])
            response = response + (quadratic * pair_valid[:, None]).sum(-1) / defenders
        mean = base + parameters[:, :1] + response
        sigma = (F.softplus(parameters[:, 1:2]) + .05).clamp_max(8)
        mean = mean[..., None].expand(-1, -1, defenders)
        sigma = sigma[:, :, None].expand(-1, candidates, defenders)
        i, j = pair_indices(defenders, nominal.device)
        separation = (nominal[:, :, i] - nominal[:, :, j]).norm(dim=-1)
        separation = separation.masked_fill(~known[:, None, None], torch.inf).amin(-1)
        risk = (self.cfg.clearance - separation).clamp_min(0).square()
        return dict(xy=impulse_path, mu=separation, sigma=torch.full_like(separation, .03),
            mean_distance=separation, nominal_mu=separation, risk=risk, det_risk=risk,
            pair_valid=valid[:, i] & valid[:, j], guard_mu=mean, guard_sigma=sigma,
            nominal_guard_mu=team.nominal_team(batch)[1],
            guard_risk=guard.positive_gaussian_square(mean, sigma, self.cfg.guard_margin),
            guard_weight=self.cfg.guard_weight, guard_margin=self.cfg.guard_margin,
            guard_nll_weight=self.cfg.guard_nll_weight, collision_source=self.cfg.collision_source,
            objective_risk_weight=self.cfg.objective_risk_weight,
            response_delta_weight=self.cfg.response_delta_weight,
            response_delta_scale_m=self.cfg.response_delta_scale_m)


def main():
    precise.PreciseTeamConfig, precise.PreciseTeamPredictor = StructuredConfig, StructuredPredictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'structured_response': file_hash(__file__)})
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(candidate_response=StructuredConfig().candidate_response,
                                    candidate_independent_uncertainty=True)
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'structured_response': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    precise.main()


if __name__ == '__main__':
    main()
