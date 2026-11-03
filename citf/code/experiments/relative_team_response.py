"""Learn relative short-interval geometry without fitting the common offset.

The response readout starts at zero. Its target is the candidate-centered
stopped team radial gap, averaged over matched futures. Absolute state error
and predicted spread do not compete with this small control-response target.
Online use is limited to the declared physical candidate allowance.
"""
from dataclasses import dataclass
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import guard_focus as focus
from experiments import response_delta_guard as delta
from experiments import synchronized_guard as team
from experiments import synchronized_delta as precise


@dataclass(frozen=True)
class RelativeConfig(precise.PreciseTeamConfig):
    revision: str = 'CITF-relative-stopped-team-response-v1'
    geometric_target: str = 'candidate-centered short stopped team radial gap, not absolute geometry'
    response_shrinkage: float = .02
    candidate_uncertainty_used: bool = False


def relative_truth(batch):
    values = team.actual_team(batch)
    valid = batch['candidate_valid'][:, :, None, None]
    return values - (values * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)


class RelativePredictor(precise.PreciseTeamPredictor):
    def forward(self, batch, history_cache=None):
        out = super().forward(batch, history_cache=history_cache)
        nominal = batch['nominal']
        ticks = torch.arange(1, nominal.shape[-2] + 1, dtype=nominal.dtype, device=nominal.device)
        velocity = batch['history'][:, None, :, -1, 2:4] * 5
        ballistic = batch['positions'][:, None, :, None] + velocity[..., None, :] * (ticks * self.cfg.dt)[None, None, None, :, None]
        impulse = ballistic + (nominal[..., :1, :] - ballistic[..., :1, :]) * (2 * ticks - 1)[None, None, None, :, None]
        response = out['guard_mu'] - team.nominal_team(dict(batch, nominal=impulse))[1]
        valid = batch['candidate_valid'][..., None]
        response = response - (response * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
        out.update(guard_mu=response, guard_sigma=torch.ones_like(response),
                   guard_risk=torch.zeros_like(response), response_shrinkage=self.cfg.response_shrinkage)
        return out


def response_objective(out, batch, clearance=7., mode='full', **kwargs):
    if mode != 'full':
        raise ValueError('This response-only experiment declares full geometric response training')
    samples = relative_truth(batch)
    target = samples.mean(2)
    variance = samples.var(2, unbiased=True) / samples.shape[2] if samples.shape[2] > 1 else torch.zeros_like(target)
    scale_squared = variance + out['response_delta_scale_m'] ** 2
    valid = batch['candidate_valid'][..., None] & batch['entity_valid'][:, None, :-1]
    mean = out['guard_mu']
    error = (mean - target).square() / scale_squared
    contrast = (error * valid).sum() / valid.sum().clamp_min(1)
    shrink = (mean.square() * valid).sum() / valid.sum().clamp_min(1) / out['response_delta_scale_m'] ** 2
    total = out['response_delta_weight'] * (kwargs.get('risk_weight', 1.) * contrast + out['response_shrinkage'] * shrink)
    physical = batch['unary'] + out['objective_risk_weight'] * guard.physical_pair_cost(batch, clearance)
    physical = physical + out['guard_weight'] * (out['nominal_guard_mu'][:, :, 0] + out['guard_margin']).clamp_min(0).square()
    physical = physical.masked_fill(~batch['candidate_valid'], torch.inf)
    feasible = batch['candidate_valid'] & (physical <= physical.amin(1, keepdim=True) + 1.)
    zero = mean.sum() * 0
    return dict(total=total, motion=zero, nll=zero, risk=contrast, rank=zero,
        context=contrast, guard_nll=zero,
        empirical_cost=target[:, :, 0].masked_fill(~feasible, torch.inf),
        estimated_cost=mean[:, :, 0].masked_fill(~feasible, torch.inf))


def main():
    training = len(sys.argv) > 1 and sys.argv[1] == 'train'
    precise.PreciseTeamConfig, precise.PreciseTeamPredictor = RelativeConfig, RelativePredictor
    delta.difference_objective = response_objective
    original_focus = focus.main

    def train_entry():
        guard.actual_gaps = relative_truth
        original_focus()

    focus.main = train_entry
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'relative_team_response': file_hash(__file__)})
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(geometric_target=RelativeConfig().geometric_target,
            mean_readout='relative geometry used only for within-root ordering',
            response_shrinkage=RelativeConfig().response_shrinkage)
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'relative_team_response': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    if training:
        precise.main()
    else:
        from experiments import trusted_team_response
        trusted_team_response.main()


if __name__ == '__main__':
    main()
