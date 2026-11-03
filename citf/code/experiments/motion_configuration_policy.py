"""Use coherent predicted motion in the current configuration action head.

The proposal network is frozen and retains its observed short-branch target.
The action head sees synchronized geometry at three temporal prefixes and
the dispersion across the same coherent futures, in addition to the trained
candidate interaction representation. Ordinary task policy gradients train
the current categorical action; no terminal-outcome estimator is introduced.
"""
from dataclasses import dataclass
import sys

import torch
from torch import nn

from citf.contracts import canonical_hash, file_hash
from experiments import joint_configuration_policy as joint
from experiments import joint_stopped_motion as motion


@dataclass(frozen=True)
class MotionPolicyConfig(joint.JointPolicyConfig):
    revision: str = 'CITF-motion-informed-current-configuration-policy-v1'
    action_features: str = 'candidate interaction plus coherent motion geometry and dispersion at three prefixes'
    motion_prefixes: tuple = (4, 8, 16)


class MotionConfigurationPolicy(joint.JointConfigurationPolicy):
    def __init__(self, cfg=MotionPolicyConfig()):
        super().__init__(cfg)
        size = 2 * (4 * cfg.width + 8) + 12 * len(cfg.motion_prefixes)
        self.actor = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, cfg.width),
                                   nn.GELU(), nn.Linear(cfg.width, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def features(self, batch, history_cache=None):
        interaction, baseline = super().features(batch, history_cache)
        if self.proposal is None:
            raise ValueError('Motion-informed policy requires a saved motion proposal model')
        with torch.no_grad():
            prediction = self.proposal(batch)
            paths = prediction['stopped_motion']
            summaries = []
            for prefix in self.cfg.motion_prefixes:
                known = batch['known_time_valid'].clone()
                known[:, min(prefix, known.shape[-1]):] = False
                measurement = motion.motion_geometry(paths, dict(batch, known_time_valid=known),
                    self.cfg.clearance, self.cfg.guard_margin, self.cfg.pursuit_radius)
                distance = measurement['distance'].masked_fill(
                    ~measurement['pair_valid'][:, None, None], torch.inf).amin(-1)
                radial = measurement['radial']
                pursuit = measurement['pursuit'].sqrt()
                for value in (distance / 7, radial / 6, pursuit / 10):
                    mean = value.mean(-1)
                    summaries.extend((mean.sign() * torch.log1p(mean.abs()),
                                      torch.log1p(value.std(-1, unbiased=False))))
            absolute = torch.stack(summaries, -1)
            valid = batch['candidate_valid'][..., None]
            count = valid.sum(1, keepdim=True).clamp_min(1)
            center = (absolute * valid).sum(1, keepdim=True) / count
            difference = absolute - center
            scale = ((difference.square() * valid).sum(1, keepdim=True) / count).sqrt()
            relative = difference / scale.clamp_min(self.cfg.relative_feature_floor)
        return torch.cat((interaction, absolute, relative), -1), baseline


_original_worker = joint.initialize_worker


def initialize_worker(*args):
    joint.JointPolicyConfig = MotionPolicyConfig
    joint.JointConfigurationPolicy = MotionConfigurationPolicy
    _original_worker(*args)


def main():
    joint.JointPolicyConfig = MotionPolicyConfig
    joint.JointConfigurationPolicy = MotionConfigurationPolicy
    joint.initialize_worker = initialize_worker
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        joint.train_main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['motion_action_features'] = MotionPolicyConfig().action_features
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            motion_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    joint.main()


if __name__ == '__main__':
    main()
