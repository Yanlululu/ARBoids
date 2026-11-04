"""Refine a retained learned controller under its actual greedy execution.

The existing interaction representation, trained actor projection, physical
bank and prior stay intact. Only the final action readout is optimized by
paired complete-task returns; no value or outcome predictor is introduced.
"""
from dataclasses import dataclass, replace
import sys

import torch

from citf.bridge import Source
from citf.contracts import canonical_hash, file_hash
from experiments import deterministic_response_search as search
from experiments import synchronized_configuration_policy as synchronized


@dataclass(frozen=True)
class RetainedPolicyConfig(synchronized.SynchronizedPolicyConfig):
    revision: str = 'CITF-retained-greedy-readout-refinement-v1'


def model_factory(value):
    synchronized.install_semantics()
    cfg = replace(synchronized.SynchronizedPolicyConfig(**value['config']),
                  revision='CITF-retained-greedy-readout-refinement-v1')
    if cfg.action_bank != 'physical_search':
        raise ValueError('Retained refinement requires the original physical search bank')
    model = synchronized.joint.JointConfigurationPolicy(cfg).eval()
    model.load_state_dict(value['state'])
    metadata = dict(representation_kind='retained trained pair-interaction Transformer and actor projection',
        initialization='all retained learned weights; optimize only the final action readout',
        candidate_bank_unchanged=True, candidate_set_normalization_unchanged=True,
        physical_prior_unchanged=True,
        parameter_estimator='antithetic Gaussian directions on a retained trained representation')
    return cfg, model, metadata


def initialize(project, source, config):
    torch.set_num_threads(1)
    synchronized.install_semantics()
    joint = synchronized.joint
    joint.policy.PolicyController = joint.JointPolicyController
    model = joint.JointConfigurationPolicy(synchronized.SynchronizedPolicyConfig(**config)).eval()
    joint.policy._worker.update(project=project, source=Source(project, source, 'cpu'), model=model)


@torch.no_grad()
def rollout(job):
    policy = synchronized.joint.policy
    model = policy._worker['model']
    model.load_state_dict({k: torch.from_numpy(v) for k, v in job['state'].items()})
    result = policy.rollout(dict(job, actor=model.actor.state_dict(), training=False))
    if 'record' not in result:
        return result
    row = result['record']
    duration = row['duration'] if row['capture'] else 80.
    row.update(execution='greedy', reward_duration='capped_capture',
        reward=(1. if row['success'] else -1.) - .25 * row['collision']
               + .02 * row['capture'] - .0005 * duration)
    return dict(result, completed=True)


def evaluate_main():
    synchronized.install_semantics()
    joint = synchronized.joint
    joint.JointPolicyConfig = RetainedPolicyConfig
    joint.canonical_hash = lambda value: canonical_hash(dict(
        core=synchronized.combined_identity(value), retained_policy_search=file_hash(__file__)))
    joint.main()


if __name__ == '__main__':
    if '--checkpoint' in sys.argv:
        evaluate_main()
    else:
        search.main(model_factory, initialize, rollout)
