"""Current-configuration learning with synchronized physical guard semantics.

The physical prior, candidate search and scalar action feature use the same
max-time/min-defender radial coverage as the observed short-branch labels.
This isolated revision preserves every already-running policy experiment.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import joint_configuration_policy as joint


@dataclass(frozen=True)
class SynchronizedPolicyConfig(joint.JointPolicyConfig):
    revision: str = 'CITF-synchronized-current-configuration-policy-v1'
    physical_guard_target: str = 'max over known times of min over valid defender radial gaps'


def physical_components(batch, cfg):
    radius = batch['nominal'].norm(dim=-1)
    gaps = radius[:, :, :-1] - radius[:, :, -1:]
    gaps = gaps.masked_fill(~batch['entity_valid'][:, None, :-1, None], torch.inf)
    curve = gaps.amin(-2)
    gap = curve.masked_fill(~batch['known_time_valid'][:, None], -torch.inf).amax(-1)
    collision = joint.policy.geometry.guard.physical_pair_cost(batch, cfg.clearance)
    cost = (batch['unary'] + cfg.objective_risk_weight * collision
            + cfg.guard_weight * (gap + cfg.guard_margin).clamp_min(0).square())
    return cost, collision, gap


original_worker = joint.initialize_worker
original_save = joint.atomic_save


def install_semantics():
    joint.JointPolicyConfig = SynchronizedPolicyConfig
    joint.policy.physical_components = physical_components


def initialize_worker(*args):
    install_semantics()
    original_worker(*args)


def save_checkpoint(value, path):
    original_save(value, path)
    path = Path(path)
    update = value.get('update')
    if path.name != 'last.pt' or update is None or update % 40:
        return
    stage = path.parent / f'stage_{update:06d}.pt'
    if not stage.exists():
        original_save(value, stage)
    deployed = path.parent / f'stage_{update:06d}_scale2.pt'
    if not deployed.exists():
        record = dict(source_checkpoint=str(stage), selected_update=update,
            original_policy_logit_scale=value['config']['policy_logit_scale'],
            policy_logit_scale=2., selected_weights_unchanged=True,
            role='predeclared deployment scale for immutable validation stage')
        original_save(dict(value, config=dict(value['config'], policy_logit_scale=2.),
                           derived_deployment=record), deployed)


def combined_identity(value):
    return canonical_hash(dict(core=value, synchronized_configuration_policy=file_hash(__file__)))


def main():
    install_semantics()
    joint.initialize_worker = initialize_worker
    joint.atomic_save = save_checkpoint
    joint.canonical_hash = combined_identity
    joint.main()


if __name__ == '__main__':
    main()
