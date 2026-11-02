"""Longer stopped encounter prediction with the established collision horizon.

The branch duration is declared in each data/task bank. Collision avoidance
continues to use at most 16 nominal steps; the direct encounter and guarding
heads use the complete observed stopped branch. This is a separate method
identity from changing every prediction term to a longer horizon.
"""
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import torch

from citf.bridge import Controller, factorize
from citf.contracts import canonical_hash, file_hash, tree_hash
from experiments import encounter_prediction as encounter
from experiments.collect_variant import PreallocatedRegistry


@dataclass(frozen=True)
class LongGeometryConfig(encounter.EncounterConfig):
    revision: str = 'CITF-long-stopped-geometry-v1'
    collision_prediction_steps: int = 16

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.collision_prediction_steps <= self.horizon:
            raise ValueError('Collision horizon must be within the prediction horizon')


def short_collision_batch(b, steps):
    return dict(b, nominal=b['nominal'][..., :steps, :], known_time_valid=b['known_time_valid'][..., :steps])


def long_objective(out, b, **kwargs):
    result = encounter.encounter_objective(out, b, **kwargs)
    cfg = out['encounter_config']
    clearance = kwargs.get('clearance', 7.)
    original = encounter.physical_pair_cost(b, clearance)
    shorter = encounter.physical_pair_cost(short_collision_batch(b, cfg.collision_prediction_steps), clearance)
    difference = cfg.objective_risk_weight * (shorter - original)
    empirical, estimated = result['empirical_cost'] + difference, result['estimated_cost'] + difference
    valid = b['candidate_valid']
    temperature = kwargs.get('temperature', 8.)
    probability = (-empirical / temperature).masked_fill(~valid, -torch.inf).softmax(-1).detach()
    logp = (-estimated / temperature).masked_fill(~valid, -torch.inf).log_softmax(-1)
    rank = -(probability * torch.where(valid, logp, torch.zeros_like(logp))).sum(-1).mean()
    if kwargs.get('mode', 'full') == 'full':
        result['total'] = result['total'] + kwargs.get('rank_weight', .05) * (rank - result['rank'])
    result.update(rank=rank, empirical_cost=empirical, estimated_cost=estimated)
    return result


def implementation_hash(*args, **kwargs):
    return canonical_hash({'encounter': encounter.implementation_hash(*args, **kwargs), 'long_geometry': file_hash(__file__)})


def factorize_long(world, source):
    result = factorize(world, source)
    steps = min(16, world.protocol.known_steps(world.env.Current_T))
    paths = result['paths'][..., :steps, :]
    distance = (paths[:, None, :, None] - paths[None, :, None]).norm(dim=-1).amin(-1)
    result['pair'] = (world.protocol.risk_clearance - distance).clamp_min(0).square()
    return result


def train_main():
    import citf.train as training
    ap = argparse.ArgumentParser(add_help=False)
    for key in ('encounter_weight', 'guard_weight', 'guard_margin', 'encounter_nll_weight', 'guard_nll_weight', 'effect_weight', 'objective_risk_weight'):
        ap.add_argument('--' + key.replace('_', '-'), type=float, default=getattr(LongGeometryConfig(), key))
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    training.Config = lambda **kwargs: LongGeometryConfig(**vars(args), **kwargs)
    training.Predictor = encounter.EncounterPredictor
    training.objective = long_objective
    training.tree_hash = implementation_hash
    training.main()


def evaluate_main():
    import citf.collect as collection
    import citf.evaluate as evaluation
    ap = argparse.ArgumentParser(add_help=False)
    for key in ('encounter_weight', 'guard_weight', 'guard_margin'):
        ap.add_argument('--' + key.replace('_', '-'), type=float, default=getattr(LongGeometryConfig(), key))
    args, remaining = ap.parse_known_args()
    LongGeometryConfig(**vars(args))
    sys.argv = [sys.argv[0], *remaining]
    collection.Config, collection.Predictor = LongGeometryConfig, encounter.EncounterPredictor
    encounter.factorize = factorize_long
    original_identity = evaluation.runtime_identity
    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['method'] = 'CITF-LongStoppedGeometry-development' if result['checkpoint_sha256'] else 'LongStoppedGeometry-Phys-development'
        result['controller'].update(vars(args))
        result['controller'].update(collision_source='nominal', collision_prediction_steps=16,
            geometry_target='stopped closest attacker distance and largest signed radial gap; full declared branch horizon')
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'long_geometry': file_hash(__file__),
            'encounter': file_hash(Path(encounter.__file__)),
            'guard': file_hash(Path(__file__).with_name('guard_prediction.py')),
            'pair': file_hash(Path(__file__).with_name('pair_geometry.py'))})
        return result
    evaluation.Controller = lambda *positional, **kwargs: encounter.EncounterController(*positional, **vars(args), **kwargs)
    evaluation.runtime_identity = identity
    evaluation.main()


class CollectionBehavior(Controller):
    def act(self, world):
        original = world.protocol
        try:
            world.protocol = replace(original, prediction_steps=16, risk_weight=64.)
            return super().act(world)
        finally:
            world.protocol = original


def collect_main():
    import citf.collect as collection
    if '--behavior-checkpoint' in sys.argv:
        raise ValueError('This collection identity uses the established physical controller')
    collection.Controller = CollectionBehavior
    collection.factorize = factorize_long
    collection.LayoutRegistry = PreallocatedRegistry
    original_sample = collection.sample_root
    def sample(world, source, plans, seeds, metadata, **kwargs):
        metadata = dict(metadata, behavior_identity=dict(metadata['behavior_identity'],
            method='CITF-Phys64-16steps', actual_behavior_prediction_steps=16, actual_behavior_risk_weight=64.))
        return original_sample(world, source, plans, seeds, metadata, **kwargs)
    collection.sample_root = sample
    collection.tree_hash = implementation_hash
    collection.RUNTIME_REVISION = 'CITF-events-long-branch-physical-behavior-v1'
    collection.main()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('train', 'collect'):
        command = sys.argv.pop(1)
        train_main() if command == 'train' else collect_main()
    else:
        evaluate_main()
