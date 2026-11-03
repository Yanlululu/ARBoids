"""Give every physical training layout equal probability mass.

Within a layout the existing current-geometry priority is retained. This
prevents a handful of long behavior trajectories from dominating exposure,
without removing roots, changing labels, or using future outcomes to sample.
"""
from dataclasses import dataclass

import numpy as np

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import priority_guard as priority
from experiments import synchronized_delta as precise


@dataclass(frozen=True)
class BalancedConfig(precise.PreciseTeamConfig):
    revision: str = 'CITF-layout-balanced-first-action-team-guard-v1'
    training_measure: str = 'equal physical layout mass; original observable priority renormalized within each layout'


class BalancedPredictor(precise.PreciseTeamPredictor):
    pass


def main():
    import citf.train as training
    original_load = training.load_shards
    original_sampler = priority.RootSampler
    layout_groups = {}

    def load(paths, role):
        result = original_load(paths, role)
        if role == 'train':
            names = [root['geometry_family_id'] for root in result[1]]
            _, groups = np.unique(names, return_inverse=True)
            layout_groups['indices'] = groups
        return result

    class BalancedSampler(original_sampler):
        def __init__(self, original, probability):
            groups = layout_groups['indices']
            mass = np.bincount(groups, weights=probability)
            balanced = probability / mass[groups] / len(mass)
            super().__init__(original, balanced / balanced.sum())
            print(dict(sampling=BalancedConfig().training_measure, roots=len(groups),
                layouts=len(mass), original_effective_layouts=float(1 / np.square(mass).sum()),
                balanced_effective_layouts=len(mass)), flush=True)

    training.load_shards, priority.RootSampler = load, BalancedSampler
    precise.PreciseTeamConfig, precise.PreciseTeamPredictor = BalancedConfig, BalancedPredictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'family_balanced_guard': file_hash(__file__)})
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller']['training_measure'] = BalancedConfig().training_measure
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'family_balanced_guard': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    precise.main()


if __name__ == '__main__':
    main()
