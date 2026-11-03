"""Synchronized team geometry after one action and declared physical recourse.

The candidate reference represents an impulse over the executed 0.2 seconds.
Only parameters may transfer from a held-coefficient team model; all training
examples, optimizer state, update counts and selection belong to this run.
"""
import argparse
from dataclasses import dataclass
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team
from experiments.recourse_guard import CONDITION, RECOURSE


@dataclass(frozen=True)
class TeamRecourseConfig(team.SynchronizedConfig):
    revision: str = 'CITF-synchronized-first-action-team-guard-v1'
    prediction_condition: str = CONDITION
    learned_nominal_reference: str = 'current velocity plus first-interval control impulse'


class TeamRecoursePredictor(team.SynchronizedPredictor):
    def forward(self, batch, history_cache=None):
        nominal = batch['nominal']
        positions = batch['positions'][:, None, :, None]
        velocity = batch['history'][:, None, :, -1, 2:4] * 5
        ticks = torch.arange(1, nominal.shape[-2] + 1, dtype=nominal.dtype, device=nominal.device)
        time = ticks * self.cfg.dt
        ballistic = positions + velocity[..., None, :] * time[None, None, None, :, None]
        first_effect = nominal[..., :1, :] - ballistic[..., :1, :]
        impulse = ballistic + first_effect * (2 * ticks - 1)[None, None, None, :, None]
        result = super().forward(dict(batch, nominal=impulse), history_cache=history_cache)
        result['nominal_guard_mu'] = team.nominal_team(batch)[1]
        return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--initial-checkpoint')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    predictor = TeamRecoursePredictor
    initializer = None
    if args.initial_checkpoint:
        if not remaining or remaining[0] != 'train':
            raise ValueError('Initial parameters are only an option for a new training run')
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False)
        if checkpoint['config']['revision'] not in (
                team.SynchronizedConfig().revision, TeamRecourseConfig().revision):
            raise ValueError('Only synchronized-team parameters can initialize this run')

        class InitializedPredictor(TeamRecoursePredictor):
            def __init__(self, config):
                super().__init__(config)
                for key in ('width', 'heads', 'history', 'horizon', 'dt', 'clearance', 'fleet_candidate_context'):
                    if getattr(config, key) != checkpoint['config'][key]:
                        raise ValueError('Initializer architecture differs: ' + key)
                self.load_state_dict(checkpoint['state'], strict=True)

        predictor = InitializedPredictor
        initializer = dict(path=args.initial_checkpoint, checkpoint_sha256=file_hash(args.initial_checkpoint),
                           revision=checkpoint['config']['revision'], parameters_only=True)
        print({'initializer': initializer}, flush=True)
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'synchronized_recourse': file_hash(__file__),
        'prediction_condition': CONDITION, 'recourse_controller': RECOURSE,
        'initializer': initializer})
    team.SynchronizedConfig, team.SynchronizedPredictor = TeamRecourseConfig, predictor
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(prediction_condition=CONDITION, recourse_controller=RECOURSE,
            learned_nominal_reference=TeamRecourseConfig().learned_nominal_reference)
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'synchronized_recourse': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    team.main()


if __name__ == '__main__':
    main()
