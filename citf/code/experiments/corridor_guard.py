"""Short geometric coverage of the attacker's current target corridor.

Radial advantage is discounted by the lateral travel needed to reach the
attacker-target line. This is a signed distance on observed short trajectories,
not a success label or a terminal value. The physical and learned controllers
share the same target, candidate set and search budget.
"""
import argparse
from dataclasses import dataclass
import math
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team
from experiments import synchronized_delta as precise


TARGET = 'worst stopped team radial gap plus scaled lateral distance beyond capture radius from attacker-target line'


def corridor_gaps(paths, ratio, capture_radius=5.):
    defenders, attacker = paths[..., :-1, :, :], paths[..., -1:, :, :]
    attacker_radius = attacker.norm(dim=-1)
    radial = defenders.norm(dim=-1) - attacker_radius
    lateral = (defenders[..., 0] * attacker[..., 1] -
               defenders[..., 1] * attacker[..., 0]).abs() / attacker_radius.clamp_min(1e-6)
    return radial + ratio * (lateral - capture_radius).clamp_min(0)


def nominal_corridor(batch, ratio):
    gap = corridor_gaps(batch['nominal'], ratio)
    gap = gap.masked_fill(~batch['entity_valid'][:, None, :-1, None], torch.inf)
    curve = gap.amin(-2)
    maximum = curve.masked_fill(~batch['known_time_valid'][:, None], -torch.inf).amax(-1)
    defenders = gap.shape[-2]
    return (curve[:, :, None].expand(-1, -1, defenders, -1),
            maximum[:, :, None].expand(-1, -1, defenders))


def actual_corridor(batch, ratio):
    gap = corridor_gaps(batch['real_positions'].detach(), ratio)
    gap = gap.masked_fill(~batch['entity_valid'][:, None, None, :-1, None], torch.inf)
    curve = gap.amin(-2)
    maximum = curve.masked_fill(~batch['time_valid'], -torch.inf).amax(-1)
    return maximum[..., None].expand(-1, -1, -1, gap.shape[-2])


@dataclass(frozen=True)
class CorridorConfig(precise.PreciseTeamConfig):
    revision: str = 'CITF-first-action-corridor-geometry-v1'
    geometric_target: str = TARGET
    corridor_ratio: float = .25
    corridor_capture_radius: float = 5.


class CorridorPredictor(precise.PreciseTeamPredictor):
    pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--corridor-ratio', type=float, default=.25)
    args, remaining = parser.parse_known_args()
    if not math.isfinite(args.corridor_ratio) or args.corridor_ratio < 0:
        raise ValueError('The lateral-to-radial ratio must be finite and nonnegative')
    sys.argv = [sys.argv[0], *remaining]

    @dataclass(frozen=True)
    class RunCorridorConfig(CorridorConfig):
        corridor_ratio: float = args.corridor_ratio

    team.nominal_team = lambda batch: nominal_corridor(batch, args.corridor_ratio)
    team.actual_team = lambda batch: actual_corridor(batch, args.corridor_ratio)
    team.TARGET = TARGET
    precise.PreciseTeamConfig, precise.PreciseTeamPredictor = RunCorridorConfig, CorridorPredictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'corridor_guard': file_hash(__file__),
        'corridor_ratio': args.corridor_ratio})
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['method'] = ('CITF-CorridorGuard-development' if result['checkpoint_sha256']
                                else 'CorridorGuard-Phys-development')
            result['controller'].update(guard_target=TARGET, corridor_ratio=args.corridor_ratio,
                                         corridor_capture_radius=5.)
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                     'corridor_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        original_main()

    evaluation.main = evaluate
    precise.main()


if __name__ == '__main__':
    main()
