"""Supervise short prefixes of the same stopped radial recourse branch.

The online head still predicts the 6.4-second geometric margin. Two additional
heads learn the observed 1.6- and 3.2-second prefixes, giving the shared candidate
representation dense physical targets without any terminal-value labels.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team
from experiments import noise_aware_guard as noise
from experiments import dual_interval_guard as dual


@dataclass(frozen=True)
class MultiscaleConfig(dual.DualIntervalConfig):
    revision: str = 'CITF-stopped-multiscale-recourse-geometry-v1'
    supervised_prefix_steps: tuple = (8, 16)
    prefix_likelihood_weight: float = .5
    prefix_difference_weight: float = 2.


class MultiscalePredictor(dual.DualIntervalPredictor):
    def __init__(self, cfg=MultiscaleConfig()):
        super().__init__(cfg)
        self.prefix_head = nn.Sequential(nn.LayerNorm(cfg.width),
            nn.Linear(cfg.width, cfg.width), nn.GELU(), nn.Linear(cfg.width, 4))
        nn.init.zeros_(self.prefix_head[-1].weight)
        nn.init.zeros_(self.prefix_head[-1].bias)
        with torch.no_grad():
            self.prefix_head[-1].bias[1::2] = .5
        self._prefix_features = None
        self.guard_head.register_forward_pre_hook(self._capture_prefix)

    def _capture_prefix(self, module, arguments):
        self._prefix_features = arguments[0]

    def forward(self, batch, history_cache=None):
        result = super().forward(batch, history_cache=history_cache)
        shape = result['guard_mu'].shape
        parameters = self.prefix_head(self._prefix_features).reshape(shape[0], shape[1], 2, 2)
        self._prefix_features = None
        nominal = batch['nominal']
        ticks = torch.arange(1, nominal.shape[-2] + 1, device=nominal.device, dtype=nominal.dtype)
        velocity = batch['history'][:, None, :, -1, 2:4] * 5
        ballistic = (batch['positions'][:, None, :, None]
                     + velocity[..., None, :] * (ticks * self.cfg.dt)[None, None, None, :, None])
        first_effect = nominal[..., :1, :] - ballistic[..., :1, :]
        impulse = ballistic + first_effect * (2 * ticks - 1)[None, None, None, :, None]
        predictions = []
        for index, steps in enumerate(self.cfg.supervised_prefix_steps):
            view = dict(batch, nominal=impulse[..., :steps, :],
                        known_time_valid=batch['known_time_valid'][..., :steps])
            base = team.nominal_team(view)[1]
            mean = base + parameters[:, :, index, :1]
            sigma = (F.softplus(parameters[:, :, index, 1:2]) + .05).clamp_max(8).expand_as(mean)
            predictions.append(dict(steps=steps, mean=mean, sigma=sigma))
        result.update(prefix_predictions=predictions,
                      prefix_likelihood_weight=self.cfg.prefix_likelihood_weight,
                      prefix_difference_weight=self.cfg.prefix_difference_weight)
        return result


base_objective = noise.noise_aware_objective


def multiscale_objective(output, batch, **kwargs):
    result = base_objective(output, batch, **kwargs)
    valid = batch['entity_valid'][:, None, :-1] & batch['candidate_valid'][..., None]
    candidate = batch['candidate_valid'][..., None]
    count = candidate.sum(1, keepdim=True).clamp_min(1)
    prefix_nll, prefix_contrast = [], []
    for prediction in output['prefix_predictions']:
        steps = prediction['steps']
        view = dict(batch, real_positions=batch['real_positions'][..., :steps, :],
                    time_valid=batch['time_valid'][..., :steps])
        truth = team.actual_team(view)
        mean, sigma = prediction['mean'], prediction['sigma']
        nll = .5 * ((truth - mean[:, :, None]) / sigma[:, :, None]).square() + sigma[:, :, None].log() + .5 * math.log(2 * math.pi)
        mask = valid[:, :, None].expand_as(nll)
        prefix_nll.append((nll * mask).sum() / mask.sum().clamp_min(1))
        paired = truth - (truth * candidate[:, :, None]).sum(1, keepdim=True) / count[:, :, None]
        target = paired.mean(2)
        variance = paired.var(2, unbiased=True) / paired.shape[2] if paired.shape[2] > 1 else torch.zeros_like(target)
        scale = (variance + output['response_delta_scale_m'] ** 2).sqrt()
        centered = mean - (mean * candidate).sum(1, keepdim=True) / count
        error = (centered - target) / scale
        loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
        prefix_contrast.append((loss * valid).sum() / valid.sum().clamp_min(1))
    result['total'] = result['total'] + output['prefix_likelihood_weight'] * torch.stack(prefix_nll).mean()
    if kwargs.get('mode', 'full') == 'full':
        result['total'] = result['total'] + output['prefix_difference_weight'] * torch.stack(prefix_contrast).mean()
    return result


def main():
    dual.DualIntervalConfig, dual.DualIntervalPredictor = MultiscaleConfig, MultiscalePredictor
    noise.noise_aware_objective = multiscale_objective
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'multiscale_recourse_guard': file_hash(__file__)})
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller']['supervised_prefix_steps'] = [8, 16]
        result['runtime_code'] = canonical_hash({
            'core': result['runtime_code'], 'multiscale_recourse_guard': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    dual.main()


if __name__ == '__main__':
    main()
