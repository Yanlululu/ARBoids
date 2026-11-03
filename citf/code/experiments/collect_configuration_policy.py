"""Observe stopped held-coefficient motion at a frozen task policy's states."""
import argparse
from dataclasses import replace

import numpy as np

from citf.contracts import canonical_hash, file_hash
from experiments import joint_configuration_policy as joint
from experiments.collect_variant import PreallocatedRegistry


class PolicyBehavior(joint.JointPolicyController):
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
    collection.Config = joint.JointPolicyConfig
    collection.Predictor = joint.JointConfigurationPolicy
    collection.LayoutRegistry = PreallocatedRegistry
    collection.RUNTIME_REVISION = 'CITF-current-task-policy-observed-held-geometry-v1'
    sample = collection.sample_root
    current = None

    def controller(source, model, device='cpu', bank=12, budget=96, rounds=1):
        nonlocal current
        if model is None or model.cfg.action_bank != 'physical_search':
            raise ValueError('Acquisition requires the declared frozen task policy on its common bank')
        current = PolicyBehavior(source, model, device, bank, budget, rounds)
        return current

    collection.Controller = controller

    def sample_root(world, source, plans, future_seeds, metadata, **kwargs):
        _, trace = current.act(world)
        count = len(plans)
        ranked = sorted(zip(trace['scores'], trace['configurations']), key=lambda x: (x[0], x[1]))
        candidates, seen = [], set()

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
        for candidate in [*plans[count // 2:].tolist(), *plans.tolist()]:
            add(candidate)
        if len(candidates) != count:
            raise ValueError('Could not retain the declared unique candidate count')
        behavior = dict(metadata['behavior_identity'], method='CITF-JointCurrentConfigurationPolicy-development',
            risk_weight=current.model.cfg.objective_risk_weight,
            task_policy_config=current.model.cfg.__dict__, checkpoint=args.behavior_checkpoint)
        metadata = dict(metadata, behavior_identity=behavior,
            candidate_indices=[list(row) for row in candidates],
            candidate_proposal=dict(kind='frozen task choice, source centre, ranked neighborhood and random configurations',
                selected_indices=trace['selected_indices'], learned_slots=max(2, count // 2)),
            prediction_condition='held fusion coefficients with state feedback')
        metadata['root_id'] = canonical_hash(dict(original_root=metadata['root_id'],
            behavior=behavior, candidates=candidates))
        return sample(world, source, np.asarray(candidates), future_seeds, metadata, **kwargs)

    collection.sample_root = sample_root
    previous = collection.tree_hash
    collection.tree_hash = lambda path: canonical_hash(dict(core=previous(path),
        collect_configuration_policy=file_hash(__file__),
        joint_configuration_policy=file_hash(joint.__file__),
        configuration_policy=file_hash(joint.policy.__file__)))
    collection.main()


if __name__ == '__main__':
    main()
