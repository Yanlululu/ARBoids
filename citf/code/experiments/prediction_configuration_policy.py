"""Matched current-action policy with ordinary likelihood-only geometry replay.

The candidate encoder, action architecture, task returns, and policy updates
match joint_configuration_policy. Both its declared pretrained encoder and
the subsequent supervised replay use predictive likelihood alone; risk,
ranking, and candidate-centered contrast losses are omitted in this control.
"""
from dataclasses import dataclass

from citf.contracts import canonical_hash, file_hash
from experiments import joint_configuration_policy as joint


@dataclass(frozen=True)
class PredictionPolicyConfig(joint.JointPolicyConfig):
    revision: str = 'CITF-prediction-replay-current-configuration-policy-v1'
    training_signal: str = 'ordinary task policy gradient plus likelihood-only stopped-geometry replay'


_original_objective = joint.policy.geometry.task_objective


def predictive_replay(*args, **kwargs):
    kwargs['mode'] = 'predictive_only'
    return _original_objective(*args, **kwargs)


def combined_identity(value):
    return canonical_hash(dict(core=value, prediction_configuration_policy=file_hash(__file__)))


def main():
    joint.JointPolicyConfig = PredictionPolicyConfig
    joint.policy.geometry.task_objective = predictive_replay
    joint.canonical_hash = combined_identity
    joint.main()


if __name__ == '__main__':
    main()
