"""Short physical geometry after one candidate action and fixed-policy recourse.

Only the first control interval uses the queried fusion coefficients. Every
subsequent interval replans with the declared physical guard controller.
Labels remain observed stopped radial distances; no terminal return or value
target is introduced. This data version must not be mixed with held-gate data.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

from citf.bridge import factorize, inputs
from citf.contracts import BRANCH_EVENT_FIELDS, COLLECTION_SCHEMA, canonical_hash, file_hash
from citf.opponents import APFOpponent
from experiments import collect_guard_local as local
from experiments import guard_envelope
from experiments import guard_focus as focus
from experiments import guard_prediction as guard
from experiments.ordered_guard import OrderedConfig, OrderedPredictor


CONDITION = 'selected current blend for one interval, then physical guard replanning at every interval'
RECOURSE = dict(method='StoppedGuard-Phys-development', risk_weight=64.,
                guard_weight=16., guard_margin=6., bank=12, budget=96, rounds=1)


@dataclass(frozen=True)
class RecourseConfig(OrderedConfig):
    revision: str = 'CITF-first-action-guard-recourse-v1'
    prediction_condition: str = CONDITION


def sample_root(world, source, plans, future_seeds, root_metadata,
                opponent=None, stop_file=None):
    opponent = opponent or APFOpponent()
    f = factorize(world, source)
    batch, gates, rotation = inputs(world, f, plans)
    candidates, defenders = gates.shape[:2]
    steps, entities = world.protocol.prediction_steps, defenders + 1
    xy = torch.zeros(candidates, len(future_seeds), entities, steps, 2)
    time_valid = torch.zeros(candidates, len(future_seeds), steps, dtype=torch.bool)
    attacker_actions = torch.zeros(candidates, len(future_seeds), steps, 2)
    shape = (candidates, len(future_seeds))
    records = dict(branch_stop_step=torch.zeros(shape, dtype=torch.int64),
        branch_outcome=torch.zeros(shape, dtype=torch.int64),
        branch_terminated=torch.zeros(shape, dtype=torch.bool),
        branch_prediction_truncated=torch.zeros(shape, dtype=torch.bool),
        branch_stop_time=torch.zeros(shape, dtype=torch.float64),
        branch_events=torch.zeros((*shape, len(BRANCH_EVENT_FIELDS)), dtype=torch.bool))
    recourse = local.PhysicalGuardBehavior(source, None, source.device, 12, 96, 1)
    physics_steps = 0
    for candidate in range(candidates):
        for future, seed in enumerate(future_seeds):
            if stop_file and Path(stop_file).exists():
                raise InterruptedError('Collection stop between real candidate branches')
            branch = world.clone()
            branch.rng = np.random.RandomState(int(seed))
            for tick in range(steps):
                if tick == 0:
                    learned, rules, _ = source.commands(branch.env)
                    force = (750 * (gates[candidate] * learned +
                                    (1 - gates[candidate]) * rules) + 250).numpy()
                else:
                    force, _ = recourse.act(branch)
                result = branch.step(force, attacker_action=opponent.act(branch))
                physics_steps += 1
                attacker_actions[candidate, future, tick] = torch.as_tensor(
                    branch.env.att_action, dtype=torch.float32)
                xy[candidate, future, :, tick] = torch.as_tensor(
                    branch.state()[:, :2] @ rotation, dtype=torch.float32)
                time_valid[candidate, future, tick] = True
                if branch.done:
                    break
            records['branch_stop_step'][candidate, future] = tick + 1
            records['branch_outcome'][candidate, future] = result['outcome']
            records['branch_terminated'][candidate, future] = branch.done
            records['branch_prediction_truncated'][candidate, future] = not branch.done
            records['branch_stop_time'][candidate, future] = result['duration']
            records['branch_events'][candidate, future] = torch.tensor(
                [bool(result[key]) for key in BRANCH_EVENT_FIELDS])
    tensors = {key: value[0].clone() for key, value in batch.items()}
    tensors.update(real_positions=xy, time_valid=time_valid,
                   executed_attacker_actions=attacker_actions, **records)
    identity = world.initial_identity
    metadata = dict(root_metadata, collection_schema=COLLECTION_SCHEMA,
        branch_event_fields=BRANCH_EVENT_FIELDS, future_seeds=list(map(int, future_seeds)),
        source_sha256=source.sha256, source_implementation_sha256=source.implementation_sha256,
        environment_sha256=world.environment_sha256, capture_stage=f't={world.env.Current_T:.1f}s',
        elapsed_seconds=float(world.env.Current_T),
        remaining_seconds=max(0., world.protocol.time_limit - world.env.Current_T),
        geometry_identity=identity, geometry_family_id=identity['fingerprint'],
        sampling_probability=root_metadata.get('sampling_probability', 1.),
        opponent_identity=opponent.identity, prediction_condition=CONDITION,
        recourse_controller=RECOURSE)
    metadata['root_id'] = canonical_hash(dict(held_root=root_metadata['root_id'],
        condition=CONDITION, recourse=RECOURSE))
    return tensors, metadata, physics_steps


def main():
    command = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1] in ('collect', 'train') else 'evaluate'
    if command == 'collect':
        import citf.collect as collection
        collection.sample_root = sample_root
        original_tree_hash = local.tree_hash
        local.tree_hash = lambda path: canonical_hash({
            'collection': original_tree_hash(path), 'recourse_guard': file_hash(__file__)})
        original_main = collection.main

        def collect_main():
            collection.RUNTIME_REVISION = 'CITF-first-action-physical-guard-recourse-data-v1'
            original_main()

        collection.main = collect_main
        local.main()
        return

    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'recourse_guard': file_hash(__file__),
        'ordered_guard': file_hash(Path(__file__).with_name('ordered_guard.py'))})
    guard.GuardPredictor = OrderedPredictor
    if command == 'train':
        focus.FocusConfig = RecourseConfig
        focus.main()
    else:
        import citf.evaluate as evaluation
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller'].update(prediction_condition=CONDITION,
                                        recourse_controller=RECOURSE)
            result['runtime_code'] = canonical_hash({
                'core': result['runtime_code'], 'recourse_guard': file_hash(__file__),
                'ordered_guard': file_hash(Path(__file__).with_name('ordered_guard.py'))})
            return result

        evaluation.runtime_identity = identity
        guard.GuardConfig = RecourseConfig
        guard_envelope.main()


if __name__ == '__main__':
    main()
