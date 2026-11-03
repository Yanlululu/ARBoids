"""Collect physical recourse labels at states visited by a fixed learned guard.

The frozen behavior predictor changes which states are visited. Branches keep
the original physical recourse controller, so their prediction condition is
unchanged and they can augment the existing first-action training pool.
"""
import argparse
from pathlib import Path

import numpy as np

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import guard_envelope
from experiments import recourse_guard as recourse
from experiments import synchronized_guard as team
from experiments.synchronized_delta import PreciseTeamConfig, PreciseTeamPredictor
from experiments.collect_variant import PreallocatedRegistry


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--behavior-checkpoint', required=True)
    args, _ = parser.parse_known_args()
    import citf.collect as collection
    physical_gaps = guard.nominal_gaps
    guard.nominal_gaps = team.nominal_team
    collection.Config, collection.Predictor = PreciseTeamConfig, PreciseTeamPredictor
    collection.LayoutRegistry = PreallocatedRegistry
    collection.RUNTIME_REVISION = 'CITF-first-action-team-onpolicy-data-v1'
    current_policy = None

    def controller(source, model, device='cpu', bank=12, budget=96, rounds=1):
        nonlocal current_policy
        if model is None or (model.cfg.guard_weight, model.cfg.guard_margin) != (16., 6.):
            raise ValueError('This collection requires the declared trained team guard')
        current_policy = guard_envelope.EnvelopeController(source, model, device, bank, budget, rounds,
            guard_weight=16., guard_margin=6., collision_source='nominal')
        return current_policy

    collection.Controller = controller

    def sample_root(world, source, plans, future_seeds, metadata, **kwargs):
        _, trace = current_policy.act(world)
        choices = []
        seen = set()
        for candidate in [trace['selected_indices'], trace['source_center_indices'], *plans.tolist()]:
            key = tuple(map(int, candidate))
            if key not in seen and len(choices) < len(plans):
                seen.add(key)
                choices.append(key)
        behavior = dict(metadata['behavior_identity'],
            method='CITF-first-action-team-precise-development',
            risk_weight=64., guard_weight=16., guard_margin=6.,
            geometric_target=team.TARGET,
            checkpoint=args.behavior_checkpoint,
            predictor_sha256=file_hash(args.behavior_checkpoint))
        metadata = dict(metadata, behavior_identity=behavior,
            candidate_indices=[list(row) for row in choices],
            candidate_proposal=dict(kind='learned behavior choice, source centre, nominal bank and random configurations',
                                    behavior_choice=trace['selected_indices']))
        metadata['root_id'] = canonical_hash(dict(original_root=metadata['root_id'],
                                                 behavior=behavior, candidate_indices=choices))
        # The behavior uses synchronized team geometry. All branch continuations
        # retain the previously declared per-vessel physical guard geometry.
        guard.nominal_gaps = physical_gaps
        try:
            return recourse.sample_root(world, source, np.asarray(choices), future_seeds, metadata, **kwargs)
        finally:
            guard.nominal_gaps = team.nominal_team

    collection.sample_root = sample_root
    original_hash = collection.tree_hash
    collection.tree_hash = lambda path: canonical_hash({
        'core': original_hash(path), 'collect_team_onpolicy': file_hash(__file__),
        'recourse_guard': file_hash(recourse.__file__),
        'synchronized_guard': file_hash(team.__file__),
        'synchronized_delta': file_hash(Path(__file__).with_name('synchronized_delta.py'))})
    collection.main()


if __name__ == '__main__':
    main()
