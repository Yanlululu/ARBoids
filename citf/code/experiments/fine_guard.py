"""Interpolate the current actuator fusion grid before joint selection.

Each channel retains 0, the source centre, and 1, and adds the midpoint
on either side of the source centre. The physical and learned controllers
share this 25-candidate grid; forecast, preference and scoring are unchanged.
"""
import argparse

import torch

from citf.contracts import canonical_hash, file_hash
from citf.physics import forecast
from experiments import guard_prediction as guard
from experiments import guard_envelope


@torch.no_grad()
def factorize_fine(world, source):
    protocol = world.protocol
    learned, rules, prior = source.commands(world.env)
    levels = torch.stack((torch.zeros_like(prior), prior / 2, prior,
                          (1 + prior) / 2, torch.ones_like(prior)), 1)
    index = torch.arange(25, device=prior.device)
    grid = torch.stack((levels[:, index // 5, 0], levels[:, index % 5, 1]), -1)
    equal = (grid[:, :, None] - grid[:, None, :]).abs().amax(-1) < 1e-7
    duplicate = (equal & torch.ones(25, 25, dtype=torch.bool,
                                    device=prior.device).tril(-1)).any(-1)
    valid = ~duplicate
    force = 750 * (grid * learned[:, None] + (1 - grid) * rules[:, None]) + 250
    actual = torch.as_tensor(world.state()[:-1], dtype=torch.float64)
    paths = forecast(actual[:, None].expand(-1, 25, -1), force.double(),
                     protocol.prediction_steps, protocol.control_dt).float()
    steps = protocol.known_steps(world.env.Current_T)
    distance = (paths[:, None, :, None, :steps] -
                paths[None, :, None, :, :steps]).norm(dim=-1).amin(-1)
    pair = (protocol.risk_clearance - distance).clamp_min(0).square()
    unary = ((grid - prior[:, None]) / protocol.prior_scale).square().mean(-1)
    return dict(grid=grid, valid=valid, unary=unary.masked_fill(~valid, torch.inf),
                pair=pair, paths=paths, learned=learned, rules=rules, prior=prior)


def main():
    import citf.evaluate as evaluation
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint')
    args, _ = parser.parse_known_args()
    guard.factorize = guard_envelope.factorize = factorize_fine
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller']['candidate_grid'] = {
            'channel_levels': ['0', 'source/2', 'source', '(1+source)/2', '1'],
            'max_per_vessel': 25,
            'duplicate_tolerance': 1e-7,
        }
        result['runtime_code'] = canonical_hash({
            'core': result['runtime_code'], 'fine_guard': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    revision = None
    if args.checkpoint:
        revision = torch.load(args.checkpoint, map_location='cpu',
                              weights_only=False)['config']['revision']
    if revision == 'CITF-stopped-guard-envelope-focus-v1':
        from experiments import guard_focus_evaluate
        guard_focus_evaluate.main()
    elif revision == 'CITF-ordered-stopped-guard-envelope-v1':
        from experiments import ordered_guard
        ordered_guard.main()
    elif revision in (None, 'CITF-stopped-guard-query-v1'):
        guard.evaluate_main()
    else:
        raise ValueError('Unregistered predictor revision for the fine candidate grid')


if __name__ == '__main__':
    main()
