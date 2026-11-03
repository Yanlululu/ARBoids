"""Supervise candidate-induced changes in short radial physical geometry.

Centering within a root removes its common state offset. The target remains
the stopped physical radial gap, with matched random futures; it is not a
terminal return. Absolute likelihood and online risk supervision are retained.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from experiments import guard_focus as focus
from experiments import guard_prediction as guard
from experiments import response_guard as response


@dataclass(frozen=True)
class DeltaConfig(response.ResponseConfig):
    revision: str = 'CITF-control-response-radial-difference-v1'
    response_delta_weight: float = 2.
    response_delta_scale_m: float = .5


class DeltaPredictor(response.ResponsePredictor):
    def forward(self, batch, history_cache=None):
        output = super().forward(batch, history_cache=history_cache)
        output.update(response_delta_weight=self.cfg.response_delta_weight,
                      response_delta_scale_m=self.cfg.response_delta_scale_m)
        return output


base_objective = focus.focused_objective


def difference_objective(output, batch, **kwargs):
    result = base_objective(output, batch, **kwargs)
    truth = guard.actual_gaps(batch).mean(2)
    mean = output['guard_mu']
    candidate = batch['candidate_valid'][..., None]
    count = candidate.sum(1, keepdim=True).clamp_min(1)
    truth_change = truth - (truth * candidate).sum(1, keepdim=True) / count
    predicted_change = mean - (mean * candidate).sum(1, keepdim=True) / count
    error = (predicted_change - truth_change) / output['response_delta_scale_m']
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
    valid = batch['entity_valid'][:, None, :-1] & candidate
    nominal = output['nominal_guard_mu']
    nearest = nominal.masked_fill(~valid, torch.inf).amin(-1, keepdim=True)
    # This emphasis uses only observable nominal geometry, never outcomes.
    weight = torch.exp(-(nominal - nearest).clamp_min(0) / 10) * valid
    contrast = (loss * weight).sum() / weight.sum().clamp_min(1)
    if kwargs.get('mode', 'full') == 'full':
        result['total'] = result['total'] + output['response_delta_weight'] * contrast
    result['context'] = contrast
    return result


def main():
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'response_delta_guard': file_hash(__file__),
        'target': 'candidate-centered expected stopped radial gaps in metres'})
    response.ResponseConfig, response.ResponsePredictor = DeltaConfig, DeltaPredictor
    focus.focused_objective = difference_objective
    response.main()


if __name__ == '__main__':
    main()
