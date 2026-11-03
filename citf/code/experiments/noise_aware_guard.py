"""Learn first-action geometry with the precision of paired future samples.

Candidate contrasts are centered separately inside each common random future.
Their standard error controls the contrast loss scale, preventing uncertain
finite-sample differences from receiving the same precision as resolved ones.
The target is still the stopped physical radial gap in metres.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import response_delta_guard as delta
from experiments import synchronized_delta as precise


@dataclass(frozen=True)
class NoiseAwareConfig(precise.PreciseTeamConfig):
    revision: str = 'CITF-first-action-team-noise-aware-difference-v1'
    contrast_noise: str = 'paired-future standard error plus fixed physical scale in quadrature'


class NoiseAwarePredictor(precise.PreciseTeamPredictor):
    pass


def noise_aware_objective(output, batch, **kwargs):
    result = delta.base_objective(output, batch, **kwargs)
    samples = guard.actual_gaps(batch)
    candidate = batch['candidate_valid'][:, :, None, None]
    count = candidate.sum(1, keepdim=True).clamp_min(1)
    paired = samples - (samples * candidate).sum(1, keepdim=True) / count
    target = paired.mean(2)
    futures = paired.shape[2]
    variance = (paired.var(2, unbiased=True) / futures if futures > 1
                else torch.zeros_like(target))
    scale = (variance + output['response_delta_scale_m'] ** 2).sqrt()
    valid_candidate = batch['candidate_valid'][..., None]
    mean = output['guard_mu']
    prediction = mean - (mean * valid_candidate).sum(1, keepdim=True) / count[:, :, 0]
    error = (prediction - target) / scale
    loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
    valid = batch['entity_valid'][:, None, :-1] & valid_candidate
    contrast = (loss * valid).sum() / valid.sum().clamp_min(1)
    if kwargs.get('mode', 'full') == 'full':
        result['total'] = result['total'] + output['response_delta_weight'] * contrast
    result['context'] = contrast
    return result


def main():
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'noise_aware_guard': file_hash(__file__)})
    precise.PreciseTeamConfig, precise.PreciseTeamPredictor = NoiseAwareConfig, NoiseAwarePredictor
    delta.difference_objective = noise_aware_objective
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller']['contrast_noise'] = NoiseAwareConfig().contrast_noise
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                     'noise_aware_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        original_main()

    evaluation.main = evaluate
    precise.main()


if __name__ == '__main__':
    main()
