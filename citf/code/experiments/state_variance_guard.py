"""Separate first-action geometry from shared short-future uncertainty.

The candidate graph predicts each configuration's stopped radial margin.
Its Gaussian spread is conditioned on observable history and known time only,
so candidate ranking cannot exploit action-dependent variance estimates.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import synchronized_delta as precise


@dataclass(frozen=True)
class StateVarianceConfig(precise.PreciseTeamConfig):
    revision: str = 'CITF-first-action-team-state-variance-v1'
    variance_condition: str = 'current history, current radial geometry and known time only'


class StateVariancePredictor(precise.PreciseTeamPredictor):
    def __init__(self, cfg=StateVarianceConfig()):
        super().__init__(cfg)
        width = cfg.width
        self.state_spread = nn.Sequential(nn.Linear(2 * width + 3, width),
            nn.GELU(), nn.LayerNorm(width), nn.Linear(width, 1))
        nn.init.zeros_(self.state_spread[-1].weight)
        nn.init.constant_(self.state_spread[-1].bias, .5)

    def forward(self, batch, history_cache=None):
        history = self.encode_history(batch) if history_cache is None else history_cache
        result = super().forward(batch, history_cache=history)
        valid = batch['entity_valid'][:, :-1]
        pooled = (history[:, :-1] * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        radius = batch['positions'].norm(dim=-1)
        nearest_gap = (radius[:, :-1] - radius[:, -1:]).masked_fill(~valid, torch.inf).amin(-1, keepdim=True)
        state = torch.cat((pooled, history[:, -1], nearest_gap / 10, batch['time_context']), -1)
        sigma = (F.softplus(self.state_spread(state)) + .05).clamp_max(8)
        result['guard_sigma'] = sigma[:, None].expand_as(result['guard_mu'])
        result['guard_risk'] = guard.positive_gaussian_square(
            result['guard_mu'], result['guard_sigma'], result['guard_margin'])
        return result


def main():
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'state_variance_guard': file_hash(__file__)})
    precise.PreciseTeamConfig, precise.PreciseTeamPredictor = StateVarianceConfig, StateVariancePredictor
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller']['variance_condition'] = StateVarianceConfig().variance_condition
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                     'state_variance_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        original_main()

    evaluation.main = evaluate
    precise.main()


if __name__ == '__main__':
    main()
