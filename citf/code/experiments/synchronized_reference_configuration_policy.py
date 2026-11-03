"""Refine an existing task policy under synchronized guard semantics.

The learned encoder and current-action head are copied from the declared
starting policy. The physical prior, candidate search and radial feature all
use synchronized team coverage. Matched runs declare their geometry replay
weight explicitly, allowing task-only refinement to retain its pretrained
representation without assuming its distance heads remain calibrated.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys

from citf.contracts import canonical_hash, file_hash
from experiments import reference_configuration_policy as reference
from experiments import synchronized_configuration_policy as synchronized


@dataclass(frozen=True)
class SynchronizedReferenceConfig(reference.ReferencePolicyConfig):
    revision: str = 'CITF-synchronized-reference-current-policy-v1'
    physical_guard_target: str = 'max over known times of min over valid defender radial gaps'
    training_signal: str = 'ordinary task policy gradient, frozen learned-reference KL and explicitly declared geometry replay weight'


base_worker = reference.initialize_worker


def bind():
    reference.ReferencePolicyConfig = SynchronizedReferenceConfig
    reference.joint.policy.physical_components = synchronized.physical_components


def initialize_worker(*args):
    bind()
    base_worker(*args)


def implementation_identity(core):
    return canonical_hash(dict(core=core,
        synchronized_reference_policy=file_hash(__file__),
        synchronized_guard_semantics=file_hash(synchronized.__file__)))


def main():
    bind()
    reference.initialize_worker = initialize_worker
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        save = reference.atomic_save
        recorded = False

        def save_checkpoint(value, path):
            nonlocal recorded
            protocol = dict(value['protocol'])
            weight = protocol['arguments']['replay_weight']
            protocol['synchronized_refinement'] = dict(
                physical_guard_target=SynchronizedReferenceConfig().physical_guard_target,
                geometry_replay_weight=weight,
                representation='pretrained and task-refined candidate interaction encoder',
                calibrated_prediction_after_task_only_refinement_claimed=False,
                no_value_head_or_future_plan=True)
            protocol['runtime_code'] = implementation_identity(protocol['runtime_code'])
            if not recorded:
                (Path(path).parent / 'protocol.json').write_text(json.dumps(protocol, indent=2))
                recorded = True
            save(dict(value, protocol=protocol), path)

        reference.atomic_save = save_checkpoint
        reference.main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['physical_guard_target'] = SynchronizedReferenceConfig().physical_guard_target
        result['runtime_code'] = implementation_identity(result['runtime_code'])
        return result

    evaluation.runtime_identity = identity
    reference.main()


if __name__ == '__main__':
    main()
