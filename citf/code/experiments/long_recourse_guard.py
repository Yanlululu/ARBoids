"""Longer first-action radial prediction with unchanged physical recourse.

The learned radial interval is declared by the task bank. Collision scoring,
behavior and every branch continuation retain the established 16-step physical
controller. This isolates additional prediction time from a changed teacher.
"""
from dataclasses import dataclass, replace
import sys

from citf.contracts import canonical_hash, file_hash
from experiments import collect_guard_local as local
from experiments import collect_recourse_boundary as boundary
from experiments import guard_prediction as guard
from experiments import guard_envelope
from experiments import long_geometry
from experiments import noise_aware_guard as noise
from experiments import recourse_guard as recourse


@dataclass(frozen=True)
class LongRecourseConfig(noise.NoiseAwareConfig):
    revision: str = 'CITF-long-first-action-team-geometry-v1'
    horizon: int = 32
    collision_prediction_steps: int = 16
    recourse_prediction_steps: int = 16


class LongRecoursePredictor(noise.NoiseAwarePredictor):
    pass


class ShortPhysicalBehavior(local.PhysicalGuardBehavior):
    def act(self, world):
        protocol = world.protocol
        try:
            world.protocol = replace(protocol, prediction_steps=16)
            return super().act(world)
        finally:
            world.protocol = protocol


def collect_main():
    local.PhysicalGuardBehavior = ShortPhysicalBehavior
    boundary.factorize = local.factorize = recourse.factorize = long_geometry.factorize_long
    recourse.RECOURSE = dict(recourse.RECOURSE, prediction_steps=16)
    original_sample = recourse.sample_root

    def sample(world, source, plans, future_seeds, metadata, **kwargs):
        behavior = dict(metadata['behavior_identity'], prediction_steps=16)
        metadata = dict(metadata, behavior_identity=behavior,
            root_id=canonical_hash(dict(original_root=metadata['root_id'], behavior=behavior)))
        return original_sample(world, source, plans, future_seeds, metadata, **kwargs)

    recourse.sample_root = sample
    original_validation = boundary.validate_dataset

    def commit_identity(blob):
        if blob['protocol']['prediction_steps'] != 32:
            raise ValueError('This acquisition declares a 32-step radial interval')
        provenance = blob['provenance']
        provenance.update(collection_revision='CITF-long-first-action-boundary-data-v1',
                          behavior_prediction_steps=16, recourse_prediction_steps=16,
                          collision_prediction_steps=16)
        provenance['collection_code_sha256'] = canonical_hash({
            'boundary_collection': provenance['collection_code_sha256'],
            'long_recourse_guard': file_hash(__file__),
            'long_geometry': file_hash(long_geometry.__file__)})
        return original_validation(blob)

    boundary.validate_dataset = commit_identity
    boundary.main()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        sys.argv.pop(1)
        collect_main()
        return
    original_pair_cost = guard.physical_pair_cost
    guard.physical_pair_cost = lambda batch, clearance: original_pair_cost(
        long_geometry.short_collision_batch(batch, 16), clearance)
    guard_envelope.factorize = long_geometry.factorize_long
    noise.NoiseAwareConfig, noise.NoiseAwarePredictor = LongRecourseConfig, LongRecoursePredictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'long_recourse_guard': file_hash(__file__),
        'long_geometry': file_hash(long_geometry.__file__)})
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller'].update(collision_prediction_steps=16,
                                         recourse_prediction_steps=16,
                                         radial_prediction_steps=32)
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                     'long_recourse_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        original_main()

    evaluation.main = evaluate
    noise.main()


if __name__ == '__main__':
    main()
