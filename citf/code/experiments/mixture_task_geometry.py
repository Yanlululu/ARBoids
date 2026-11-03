"""Candidate-conditioned multimodal distributions of stopped task geometry.

Mixture likelihoods retain distinct short-branch responses. Expected collision,
cover, and pursuit costs are integrated component by component instead of
replacing the distribution with a single moment-matched Gaussian.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from citf.risk import bounded_mean, rectified_gaussian_risk
from experiments import joint_task_geometry as geometry


@dataclass(frozen=True)
class MixtureConfig(geometry.TaskGeometryConfig):
    revision: str = 'CITF-mixture-stopped-task-geometry-v1'
    mixture_components: int = 3
    distance_distribution: str = 'conditional Gaussian mixture with component-wise expected risk'


def mixture_head(width, components, initial_scales):
    head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, 3 * components))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    with torch.no_grad():
        bias = head[-1].bias.reshape(components, 3)
        bias[:, 2] = torch.tensor([math.log(math.expm1(s - .03)) for s in initial_scales])
    return head


def distribution(raw, reference):
    logits, offset, scale = raw.unbind(-1)
    return logits.log_softmax(-1), reference[..., None] + offset, F.softplus(scale) + .03


def moments(log_weights, mean, sigma):
    weights = log_weights.exp()
    mu = (weights * mean).sum(-1)
    variance = (weights * (sigma.square() + mean.square())).sum(-1) - mu.square()
    return mu, variance.clamp_min(.0009).sqrt()


class MixturePredictor(geometry.TaskGeometryPredictor):
    def __init__(self, cfg=MixtureConfig()):
        super().__init__(cfg)
        if cfg.mixture_components != 3:
            raise ValueError('This development distribution declares three components')
        self.distance_mixture = mixture_head(cfg.width, cfg.mixture_components, [.25, 1., 4.])
        self.cover_mixture = mixture_head(cfg.width, cfg.mixture_components, [.5, 2., 6.])
        self._distance_queries = None
        self._cover_queries = None
        self.distance_head.register_forward_pre_hook(self._capture_distance)
        self.guard_head.register_forward_pre_hook(self._capture_cover)

    def _capture_distance(self, module, arguments):
        self._distance_queries = arguments[0]

    def _capture_cover(self, module, arguments):
        self._cover_queries = arguments[0]

    def forward(self, batch, history_cache=None):
        out = super().forward(batch, history_cache=history_cache)
        n, c, p = out['all_mu'].shape
        defenders = batch['positions'].shape[1] - 1
        components = self.cfg.mixture_components
        raw = self.distance_mixture(self._distance_queries).reshape(n, c, p, components, 3)
        cover = self.cover_mixture(self._cover_queries).reshape(n, c, defenders, components, 3)
        self._distance_queries = self._cover_queries = None
        logw, mu, sigma = distribution(raw, out['all_nominal_mu'])
        mean, deviation = moments(logw, mu, sigma)
        _, j = geometry.all_pairs(defenders, mean.device)
        collision, pursuit = j < defenders, j == defenders
        out.update(distance_logw=logw, distance_component_mu=mu, distance_component_sigma=sigma,
                   all_mu=mean, all_sigma=deviation, mu=mean[:, :, collision],
                   sigma=deviation[:, :, collision])
        positive_mean = mu * torch.special.ndtr(mu / sigma) + sigma * torch.exp(-.5 * (mu / sigma).square()) / math.sqrt(2 * math.pi)
        out['all_mean_distance'] = (logw.exp() * positive_mean).sum(-1)
        out['mean_distance'] = out['all_mean_distance'][:, :, collision]
        risk = (logw.exp() * rectified_gaussian_risk(mu, sigma, self.cfg.clearance)).sum(-1)
        out['risk'] = risk[:, :, collision]
        out['det_risk'] = (self.cfg.clearance - out['mean_distance']).clamp_min(0).square()
        out['pursuit_mu'], out['pursuit_sigma'] = mean[:, :, pursuit], deviation[:, :, pursuit]
        out['pursuit_risk'] = (logw[:, :, pursuit].exp() * geometry.guard.positive_gaussian_square(
            mu[:, :, pursuit], sigma[:, :, pursuit], -self.cfg.pursuit_radius)).sum(-1)

        glogw, gmu, gsigma = distribution(cover, out['nominal_guard_mu'])
        valid = batch['entity_valid'][:, :-1]
        glogw = glogw - valid.sum(-1)[:, None, None, None].clamp_min(1).log()
        glogw = glogw.masked_fill(~valid[:, None, :, None], -torch.inf)
        glogw, gmu, gsigma = (x.flatten(-2) for x in (glogw, gmu, gsigma))
        out['team_mu'], out['team_sigma'] = moments(glogw, gmu, gsigma)
        out['team_risk'] = (glogw.exp() * geometry.guard.positive_gaussian_square(
            gmu, gsigma, self.cfg.guard_margin)).sum(-1)
        out.update(cover_logw=glogw, cover_component_mu=gmu, cover_component_sigma=gsigma)
        return out


ordinary_objective = geometry.task_objective


def mixture_objective(out, batch, **kwargs):
    result = ordinary_objective(out, batch, **kwargs)
    distance, radial, _, _ = geometry.targets(batch)
    cv = batch['candidate_valid']
    pm = out['all_pair_valid'][:, None, None] & cv[:, :, None, None]
    dmu, dsigma = out['distance_component_mu'][:, :, None], out['distance_component_sigma'][:, :, None]
    logp = out['distance_logw'][:, :, None] - .5 * ((distance[..., None] - dmu) / dsigma).square() - dsigma.log() - .5 * math.log(2 * math.pi)
    dnll = bounded_mean(-logp.logsumexp(-1), pm)
    gmu, gsigma = out['cover_component_mu'][:, :, None], out['cover_component_sigma'][:, :, None]
    logp = out['cover_logw'][:, :, None] - .5 * ((radial[..., None] - gmu) / gsigma).square() - gsigma.log() - .5 * math.log(2 * math.pi)
    gnll = bounded_mean(-logp.logsumexp(-1), cv[..., None])
    nll = dnll + gnll
    result['total'] = result['total'] + out['guard_nll_weight'] * (nll - result['nll'])
    result['nll'] = nll
    return result


def main():
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = previous(*args, **kwargs)
        value['controller']['distance_distribution'] = MixtureConfig().distance_distribution
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'], mixture_geometry=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    previous_hash = geometry.tree_hash
    geometry.tree_hash = lambda path: canonical_hash(dict(core=previous_hash(path),
                                                         mixture_geometry=file_hash(__file__)))
    geometry.TaskGeometryConfig, geometry.TaskGeometryPredictor = MixtureConfig, MixturePredictor
    geometry.task_objective = mixture_objective
    geometry.main()


if __name__ == '__main__':
    main()
