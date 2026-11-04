"""Learn the stopped short capture boundary jointly across current candidates.

The added outputs are distances within the existing short physical horizon.
They are never trained on complete-task returns. Current-action policy training
continues to use the ordinary complete-task policy gradient.
"""
from dataclasses import dataclass
import sys

import torch
from torch import nn

from experiments import contact_event_policy as contact
from experiments import contact_distribution_policy as distribution
from experiments.candidate_response import stopped_positions


@dataclass(frozen=True)
class CaptureBoundaryConfig(distribution.ContactDistributionConfig):
    revision: str = 'CITF-contact-stopped-capture-boundary-v1'
    physical_objective: str = 'joint-response-energy-plus-stopped-capture-margin-energy'
    boundary_reference_scale: float = .5
    boundary_response_scale: float = .02
    boundary_loss_weight: float = .5
    boundary_readout: str = 'short-capture-margin-mean-and-spread'


def capture_margin(paths, known):
    distance = (paths[..., :-1, :, :] - paths[..., -1:, :, :]).norm(dim=-1).amin(-2)
    return distance.masked_fill(~known[:, None, None], torch.inf).amin(-1) - 5.


class CaptureBoundaryModel(contact.ContactEventModel):
    def __init__(self, cfg=CaptureBoundaryConfig()):
        super().__init__(cfg)
        size = 4 * cfg.width + 16
        self.boundary_absolute = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, cfg.width),
            nn.GELU(), nn.Linear(cfg.width, cfg.modes))
        self.boundary_response = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, cfg.width),
            nn.GELU(), nn.Linear(cfg.width, cfg.modes))
        for head in (self.boundary_absolute[-1], self.boundary_response[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, batch, history_cache=None):
        output = super().forward(batch, history_cache)
        feature = output['features']
        half = feature.shape[-1] // 2
        reference = feature[..., :half] - self.cfg.policy_reference_feature_scale * feature[..., half:]
        reference_feature = torch.cat((reference, torch.zeros_like(reference)), -1)
        margin = capture_margin(output['xy'], batch['known_time_valid'])
        margin = margin + self.boundary_absolute(reference_feature)
        margin = margin + self.cfg.boundary_response_scale * (
            self.boundary_response(feature) - self.boundary_response(reference_feature))
        output['capture_margin'] = margin
        # The fixed physical reference is always the first candidate in this
        # controller and its replay. No candidate-set normalization is used.
        value = feature[..., :half].clone()
        reference = reference.clone()
        slot = 2 * self.cfg.width + 5
        value[..., slot] = torch.asinh(margin.mean(-1) / self.cfg.boundary_reference_scale)
        value[..., slot + 1] = margin.std(-1, unbiased=False) / self.cfg.boundary_reference_scale
        reference[..., slot] = value[:, :1, slot]
        reference[..., slot + 1] = value[:, :1, slot + 1]
        feature = torch.cat((value, (value - reference) / self.cfg.policy_reference_feature_scale), -1)
        output['features'] = feature
        if 'teacher_logits' in batch:
            feature = feature + self.phase_embed(batch['event_features'])[:, None]
            logits = batch['teacher_logits'] + self.cfg.policy_logit_scale * self.actor(feature).squeeze(-1)
            output['logits'] = logits.masked_fill(~batch['candidate_valid'], -torch.inf)
        return output


_response_objective = distribution.distribution_objective


def boundary_objective(output, batch, cfg):
    result = _response_objective(output, batch, cfg)
    predicted = output['capture_margin']
    actual = capture_margin(stopped_positions(batch)[:, :, :cfg.modes], batch['known_time_valid'])
    valid = batch['candidate_valid'].clone()
    valid[:, 0] = False
    count = valid.sum(-1).clamp_min(1).sqrt()[:, None, None]

    def features(margin):
        reference = margin[:, :1] / cfg.boundary_reference_scale
        response = (margin - margin[:, :1]) * valid[..., None] / (cfg.boundary_response_scale * count)
        return torch.cat((reference, response), 1).transpose(1, 2)

    p, t = features(predicted), features(actual)
    energy = (torch.cdist(p, t).mean((1, 2)) - .5 * torch.cdist(p, p).mean((1, 2))).mean()
    if cfg.supervision == 'response':
        result['total'] = result['total'] + cfg.boundary_loss_weight * energy
    result['capture_boundary_energy'] = energy
    return result


_initialize_contact_worker = contact.initialize


def initialize_boundary_worker(project, source, config):
    contact.ContactEventConfig = CaptureBoundaryConfig
    contact.ContactEventModel = CaptureBoundaryModel
    _initialize_contact_worker(project, source, config)


def main():
    contact.ContactEventModel = CaptureBoundaryModel
    distribution.ContactDistributionConfig = CaptureBoundaryConfig
    distribution.distribution_objective = boundary_objective
    distribution.initialize_distribution_worker = initialize_boundary_worker
    if len(sys.argv) <= 1 or sys.argv[1] not in ('prepare', 'train'):
        import citf.evaluate as evaluation
        from citf.contracts import canonical_hash, file_hash
        original = evaluation.runtime_identity

        def identity(*args, **kwargs):
            value = original(*args, **kwargs)
            value['runtime_code'] = canonical_hash(dict(
                core=value['runtime_code'], capture_boundary=file_hash(__file__)))
            return value

        evaluation.runtime_identity = identity
    distribution.main()


if __name__ == '__main__':
    main()
