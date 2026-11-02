"""Collect matched local candidate interventions on the strong guard policy.

Roots are chosen at prespecified times. At every root, candidate proposals
include the behavior choice, the source centre, and up to three alternative
choices for each defender with every other candidate held fixed. No future
label or terminal success selects roots or candidate configurations.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

from citf.bridge import factorize
from citf.contracts import canonical_hash, file_hash, tree_hash
from experiments.collect_variant import PreallocatedRegistry
from experiments.guard_prediction import PredictedGuardController


class PhysicalGuardBehavior(PredictedGuardController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, guard_weight=16., guard_margin=6., collision_source='nominal', **kwargs)
        if self.model is not None:
            raise ValueError('This data version declares the physical guard behavior')

    def act(self, world):
        original = world.protocol
        world.protocol = replace(original, risk_weight=64.)
        try:
            return super().act(world)
        finally:
            world.protocol = original


def main():
    import citf.collect as collection
    original_sample = collection.sample_root

    def sample_root(world, source, plans, future_seeds, root_metadata, **kwargs):
        policy = PhysicalGuardBehavior(source, None, source.device, 12, 96, 1)
        _, trace = policy.act(world)
        f = factorize(world, source)
        chosen = tuple(trace['selected_indices'])
        centre = tuple(trace['source_center_indices'])
        count = len(plans)
        proposals = []
        seen = set()

        def add(candidate):
            key = tuple(map(int, candidate))
            if key not in seen and len(proposals) < count:
                seen.add(key)
                proposals.append(key)

        add(chosen)
        add(centre)
        rng = np.random.default_rng(int(root_metadata['scene_seed']) +
                                    170003 * int(root_metadata['root_step']) + 9173)
        alternatives = []
        for i, valid in enumerate(f['valid']):
            choices = [int(x) for x in valid.nonzero()[:, 0] if int(x) != chosen[i]]
            alternatives.append(rng.permutation(choices).tolist())
        local_count = len(proposals)
        for alternative in range(3):
            for i in rng.permutation(len(chosen)):
                if alternative < len(alternatives[i]):
                    candidate = list(chosen)
                    candidate[i] = alternatives[i][alternative]
                    add(candidate)
        local_count = len(proposals) - local_count
        for candidate in trace['initial_configurations']:
            add(candidate)
        for candidate in plans:
            add(candidate)
        if len(proposals) != count:
            raise ValueError('Local proposal union did not provide the declared candidate count')
        behavior = dict(root_metadata['behavior_identity'], method='StoppedGuard-Phys-development',
                        risk_weight=64., guard_weight=16., guard_margin=6.)
        proposal = dict(kind='three balanced single-vessel interventions around observable behavior choice',
                        local_interventions=local_count, behavior_choice=list(chosen),
                        source_centre=list(centre), requested_candidates=count)
        metadata = dict(root_metadata, behavior_identity=behavior, candidate_proposal=proposal,
                        candidate_indices=[list(x) for x in proposals])
        metadata['root_id'] = canonical_hash({'original_root': root_metadata['root_id'],
                                             'behavior': behavior, 'proposal': proposal})
        return original_sample(world, source, np.asarray(proposals), future_seeds, metadata, **kwargs)

    collection.Controller = PhysicalGuardBehavior
    collection.LayoutRegistry = PreallocatedRegistry
    collection.sample_root = sample_root
    collection.tree_hash = lambda path: canonical_hash({'core': tree_hash(path),
        'collect_guard_local': file_hash(__file__),
        'guard_prediction': file_hash(Path(__file__).with_name('guard_prediction.py'))})
    collection.RUNTIME_REVISION = 'CITF-physical-guard-local-interventions-v1'
    collection.main()


if __name__ == '__main__':
    main()
