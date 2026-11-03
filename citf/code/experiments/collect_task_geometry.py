"""Stopped held-coefficient branches at learned task-geometry behavior states."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from citf.contracts import canonical_hash, file_hash
from experiments import joint_task_geometry as geometry
from experiments.collect_variant import PreallocatedRegistry


class GeometryBehavior(geometry.TaskGeometryController):
    def act(self, world):
        original = world.protocol
        world.protocol = replace(original, risk_weight=self.model.cfg.objective_risk_weight)
        try:
            return super().act(world)
        finally:
            world.protocol = original


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--behavior-checkpoint', required=True)
    args, _ = parser.parse_known_args()
    import citf.collect as collection
    geometry.pair.pair_indices = geometry.all_pairs
    geometry.guard.nominal_gaps = geometry.team.nominal_team
    collection.Config = geometry.TaskGeometryConfig
    collection.Predictor = geometry.TaskGeometryPredictor
    collection.LayoutRegistry = PreallocatedRegistry
    collection.RUNTIME_REVISION = 'CITF-joint-task-geometry-onpolicy-held-v1'
    sample = collection.sample_root
    current = None

    def controller(source, model, device='cpu', bank=12, budget=96, rounds=1):
        nonlocal current
        if model is None or model.cfg.impulse_reference:
            raise ValueError('Held geometry acquisition requires its trained behavior model')
        current = GeometryBehavior(source, model, device, bank, budget,
                                   rounds, task_config=model.cfg)
        return current

    collection.Controller = controller

    def sample_root(world, source, plans, future_seeds, metadata, **kwargs):
        _, trace = current.act(world)
        count = len(plans)
        ranked = sorted(zip(trace['scores'], trace['configurations']),
                        key=lambda x: (x[0], x[1]))
        candidates = []
        seen = set()

        def add(candidate):
            key = tuple(map(int, candidate))
            if key not in seen and len(candidates) < count:
                candidates.append(key)
                seen.add(key)

        add(trace['selected_indices'])
        add(trace['source_center_indices'])
        for _, candidate in ranked:
            if len(candidates) >= max(2, count // 2):
                break
            add(candidate)
        # Original random proposals precede the nominal bank when filling
        # the remaining slots. Selection never reads future branch outcomes.
        for candidate in [*plans[count // 2:].tolist(), *plans.tolist()]:
            add(candidate)
        if len(candidates) != count:
            raise ValueError('Could not retain the requested unique candidate count')
        behavior = dict(metadata['behavior_identity'],
                        method='CITF-JointTaskGeometry-development',
                        risk_weight=current.model.cfg.objective_risk_weight,
                        task_geometry_config=current.model.cfg.__dict__,
                        checkpoint=args.behavior_checkpoint)
        proposal = dict(kind='learned choice, source centre, scored neighborhood and random configurations',
                        selected_indices=trace['selected_indices'],
                        learned_slots=max(2, count // 2))
        metadata = dict(metadata, behavior_identity=behavior,
                        candidate_indices=[list(p) for p in candidates],
                        candidate_proposal=proposal,
                        prediction_condition='held fusion coefficients with state feedback')
        metadata['root_id'] = canonical_hash(dict(original_root=metadata['root_id'],
                                                behavior=behavior, candidates=candidates))
        return sample(world, source, np.asarray(candidates), future_seeds, metadata, **kwargs)

    collection.sample_root = sample_root
    previous = collection.tree_hash
    collection.tree_hash = lambda path: canonical_hash(dict(core=previous(path),
        collect_task_geometry=file_hash(__file__),
        task_geometry=file_hash(Path(geometry.__file__))))
    collection.main()


if __name__ == '__main__':
    main()
