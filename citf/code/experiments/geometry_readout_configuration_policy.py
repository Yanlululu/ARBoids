"""Learn current actions from explicit predicted geometry and uncertainty.

The candidate-interaction network keeps its stopped-branch distance targets.
A compact action head reads these predictions, physical references, and
current geometry instead of a large pooled latent vector. Ordinary task
returns jointly refine the action head and geometric encoder.
"""
from dataclasses import dataclass
import sys

import torch
from torch import nn

from citf.contracts import canonical_hash, file_hash
from citf.risk import signed_log1p
from experiments import balanced_configuration_policy as balanced


@dataclass(frozen=True)
class GeometryReadoutConfig(balanced.BalancedPolicyConfig):
    revision: str = 'CITF-geometric-readout-current-configuration-policy-v1'
    action_features: str = '16 explicit predicted, physical and current geometric quantities with within-bank differences'


class GeometryReadoutPolicy(balanced.BalancedConfigurationPolicy):
    def __init__(self, cfg=GeometryReadoutConfig()):
        super().__init__(cfg)
        self.actor = nn.Sequential(nn.Linear(32, cfg.width), nn.SiLU(), nn.Linear(cfg.width, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def features(self, batch, history_cache=None):
        out = self.encoder(batch, history_cache=history_cache)
        self._cover, self._pairs = None, None
        physical, collision, gap = balanced.policy.physical_components(batch, self.cfg)
        valid_defenders = batch['entity_valid'][:, :-1]
        collision_mu, collision_index = out['mu'].masked_fill(
            ~out['pair_valid'][:, None], torch.inf).min(-1)
        collision_sigma = out['sigma'].gather(-1, collision_index[..., None]).squeeze(-1)
        pursuit_mu, pursuit_index = out['pursuit_mu'].masked_fill(
            ~valid_defenders[:, None], torch.inf).min(-1)
        pursuit_sigma = out['pursuit_sigma'].gather(-1, pursuit_index[..., None]).squeeze(-1)
        positions = batch['positions']
        radius = positions.norm(dim=-1)
        current_gap = (radius[:, :-1] - radius[:, -1:]).masked_fill(
            ~valid_defenders, torch.inf).amin(-1)
        pursuit = (positions[:, :-1] - positions[:, -1:]).norm(dim=-1).masked_fill(
            ~valid_defenders, torch.inf).amin(-1)
        separation = torch.cdist(positions[:, :-1], positions[:, :-1])
        diagonal = torch.eye(separation.shape[-1], device=separation.device, dtype=torch.bool)
        pair_valid = valid_defenders[:, :, None] & valid_defenders[:, None, :] & ~diagonal
        separation = separation.masked_fill(~pair_valid, torch.inf).flatten(1).amin(-1)
        candidates = batch['candidate_valid'].shape[1]
        expand = lambda value: value[:, None].expand(-1, candidates)
        signed_log = signed_log1p
        absolute = torch.stack((
            batch['unary'] / 10,
            torch.log1p(collision), signed_log(gap / 10),
            torch.log1p((out['risk'] * out['pair_valid'][:, None]).sum(-1)),
            signed_log(out['team_mu'] / 10), torch.log1p(out['team_sigma'] / 5),
            torch.log1p(out['team_risk'] / 36),
            signed_log(pursuit_mu / 10), torch.log1p(pursuit_sigma / 5),
            signed_log(collision_mu / 7), torch.log1p(collision_sigma / 5),
            batch['time_context'][:, :1].expand(-1, candidates),
            expand(radius[:, -1] / 60), expand(signed_log(current_gap / 10)),
            expand(torch.log1p(pursuit / 10)), expand(torch.log1p(separation / 7))), -1)
        valid = batch['candidate_valid'][..., None]
        count = valid.sum(1, keepdim=True).clamp_min(1)
        center = (absolute * valid).sum(1, keepdim=True) / count
        difference = absolute - center
        scale = ((difference.square() * valid).sum(1, keepdim=True) / count).sqrt()
        relative = difference / scale.detach().clamp_min(self.cfg.relative_feature_floor)
        best = physical.masked_fill(~batch['candidate_valid'], torch.inf).amin(-1, keepdim=True)
        baseline = -torch.log1p((physical - best).clamp_min(0) / self.cfg.policy_prior_scale)
        return torch.cat((absolute, relative), -1), baseline


_original_worker = balanced.initialize_worker


def bind():
    balanced.BalancedPolicyConfig = GeometryReadoutConfig
    balanced.BalancedConfigurationPolicy = GeometryReadoutPolicy
    balanced.initialize_worker = initialize_worker


def initialize_worker(*args):
    bind()
    _original_worker(*args)


def main():
    bind()
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        balanced.train_main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['geometric_action_readout'] = GeometryReadoutConfig().action_features
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            geometry_readout_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    balanced.main()


if __name__ == '__main__':
    main()
