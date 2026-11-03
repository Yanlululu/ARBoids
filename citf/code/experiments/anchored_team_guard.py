"""Separate shared stopped geometry from candidate interaction responses.

A fixed half-blend reference is reconstructed from current observable state
and source actions. The candidate graph predicts changes relative to this
same reference, while a history head predicts its common offset and spread.
Only stopped short-interval physical geometry is supervised.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from citf.physics import forecast
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team
from experiments import synchronized_delta as precise


@dataclass(frozen=True)
class AnchoredConfig(precise.PreciseTeamConfig):
    revision: str = 'CITF-anchored-candidate-team-response-v1'
    response_reference: str = 'current half-blend first impulse from observable state'
    maximum_response_m: float = 2.
    shared_reference_uncertainty: bool = True


class AnchoredPredictor(precise.PreciseTeamPredictor):
    def __init__(self, cfg=AnchoredConfig()):
        super().__init__(cfg)
        self.shared_geometry = nn.Sequential(nn.LayerNorm(2 * cfg.width + 3),
            nn.Linear(2 * cfg.width + 3, cfg.width), nn.GELU(), nn.Linear(cfg.width, 2))
        nn.init.zeros_(self.shared_geometry[-1].weight)
        nn.init.zeros_(self.shared_geometry[-1].bias)
        with torch.no_grad():
            self.shared_geometry[-1].bias[1] = .5

    def reference(self, batch):
        history = batch['history'][:, :, -1]
        positions = batch['positions']
        yaw = torch.atan2(history[:, :-1, 4], history[:, :-1, 5])
        velocity = history[..., 2:4] * 5
        state = torch.cat((positions[:, :-1], yaw[..., None], velocity[:, :-1],
                           history[:, :-1, 6:7]), -1)
        source = batch['candidate'][:, :1].clone()
        source[:, :, :-1, :2] = .5
        source[:, :, :-1, 6:8] = .5 * (source[:, :, :-1, 2:4] + source[:, :, :-1, 4:6])
        force = 750 * source[:, 0, :-1, 6:8] + 250
        first = forecast(state.double(), force.double(), 1, self.cfg.dt).to(positions.dtype)
        ticks = torch.arange(1, self.cfg.horizon + 1, device=positions.device, dtype=positions.dtype)
        ballistic = positions[:, :, None] + velocity[:, :, None] * (ticks * self.cfg.dt)[None, None, :, None]
        first_effect = first - ballistic[:, :-1, :1]
        defender = ballistic[:, :-1] + first_effect * (2 * ticks - 1)[None, None, :, None]
        nominal = torch.cat((defender, ballistic[:, -1:]), 1)[:, None]
        view = dict(batch, candidate=source, nominal=nominal,
                    candidate_valid=torch.ones_like(batch['candidate_valid'][:, :1]),
                    unary=torch.zeros_like(batch['unary'][:, :1]))
        return view

    def forward(self, batch, history_cache=None):
        history = self.encode_history(batch) if history_cache is None else history_cache
        result = super().forward(batch, history_cache=history)
        reference = self.reference(batch)
        anchor = super().forward(reference, history_cache=history)
        valid = batch['entity_valid'][:, :-1]
        radius = batch['positions'].norm(dim=-1)
        gap = radius[:, :-1] - radius[:, -1:]
        ownership = (-gap / 2).masked_fill(~valid, -torch.inf).softmax(-1)
        pooled = (history[:, :-1] * ownership[..., None]).sum(1)
        initial = gap.masked_fill(~valid, torch.inf).amin(-1, keepdim=True)
        state = torch.cat((pooled, history[:, -1], initial / 10, batch['time_context']), -1)
        parameters = self.shared_geometry(state)
        reference_base = team.nominal_team(reference)[1]
        response = result['guard_mu'] - anchor['guard_mu']
        response = self.cfg.maximum_response_m * torch.tanh(response / self.cfg.maximum_response_m)
        result['guard_mu'] = reference_base + parameters[:, None, :1] + response
        spread = (F.softplus(parameters[:, None, 1:2]) + .05).clamp_max(8)
        result['guard_sigma'] = spread.expand_as(result['guard_mu'])
        result['guard_risk'] = guard.positive_gaussian_square(
            result['guard_mu'], result['guard_sigma'], result['guard_margin'])
        return result


def main():
    precise.PreciseTeamConfig, precise.PreciseTeamPredictor = AnchoredConfig, AnchoredPredictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'anchored_team_guard': file_hash(__file__)})
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(response_reference=AnchoredConfig().response_reference,
            maximum_response_m=AnchoredConfig().maximum_response_m,
            shared_reference_uncertainty=True)
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'anchored_team_guard': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    precise.main()


if __name__ == '__main__':
    main()
