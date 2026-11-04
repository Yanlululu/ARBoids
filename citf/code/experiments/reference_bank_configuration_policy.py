"""Expand current configurations using a fixed per-state score reference.

The original physical bank supplies feature mean, standard deviation and
physical minimum for the whole decision. Added configurations are scored
against those same statistics. Existing policy weights and original-bank
features are retained; expansion cannot change their normalization.
"""
from dataclasses import dataclass

import torch

from citf.contracts import canonical_hash, file_hash
from citf.risk import signed_log1p
from experiments import cooperative_search_configuration_policy as composed

joint = composed.joint


@dataclass(frozen=True)
class ReferenceBankConfig(composed.CooperativeSearchConfig):
    revision: str = 'CITF-reference-bank-composed-current-configuration-v2'
    score_reference: str = 'fixed original physical bank statistics for all current search calls'


class ReferenceBankPolicy(joint.JointConfigurationPolicy):
    def __init__(self, cfg=ReferenceBankConfig()):
        super().__init__(cfg)
        self.reference_center = None
        self.reference_scale = None
        self.reference_physical = None
        self.reference_count = None

    def features(self, batch, history_cache=None):
        combined, _ = super().features(batch, history_cache)
        feature = combined[..., :4 * self.cfg.width + 8]
        physical = joint.policy.physical_components(batch, self.cfg)[0]
        if self.reference_center is None:
            valid = batch['candidate_valid'][..., None]
            count = valid.sum(1, keepdim=True).clamp_min(1)
            center = (feature * valid).sum(1, keepdim=True) / count
            scale = (((feature - center).square() * valid).sum(1, keepdim=True) / count).sqrt()
            self.reference_center = center.detach()
            self.reference_scale = scale.detach().clamp_min(self.cfg.relative_feature_floor)
            self.reference_physical = physical.masked_fill(
                ~batch['candidate_valid'], torch.inf).amin(-1, keepdim=True).detach()
            self.reference_count = batch['candidate_valid'].sum(-1).tolist()
        relative = (feature - self.reference_center) / self.reference_scale
        difference = (physical - self.reference_physical) / self.cfg.policy_prior_scale
        # The original bank has nonnegative differences and is unchanged.
        # A newly discovered lower-cost configuration must retain its advantage.
        baseline = -signed_log1p(difference)
        return torch.cat((feature, relative), -1), baseline


class ReferenceBankController(composed.CooperativeSearchController):
    @torch.no_grad()
    def act(self, world):
        if self.stochastic:
            raise ValueError('Fixed-reference search is a frozen-weight development evaluation')
        self.model.reference_center = None
        self.model.reference_scale = None
        self.model.reference_physical = None
        self.model.reference_count = None
        force, trace = super().act(world)
        trace.update(score_reference=self.model.cfg.score_reference,
                     score_reference_bank_size=self.model.reference_count)
        return force, trace


def main():
    composed.CooperativeSearchConfig = ReferenceBankConfig
    composed.CooperativeSearchController = ReferenceBankController
    joint.JointConfigurationPolicy = ReferenceBankPolicy
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['score_reference'] = ReferenceBankConfig().score_reference
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            reference_bank_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    composed.main()


if __name__ == '__main__':
    main()
