"""Declared uncertainty scaling for a frozen candidate-team predictor.

This combines the existing spread-only calibration with the synchronized
team controller. Mean predictions, nominal protection and search are retained.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import team_guard_selection as selection


@torch.no_grad()
def calibrate_main():
    from citf.train import load_shards
    from experiments.synchronized_delta import PreciseTeamConfig, PreciseTeamPredictor
    from experiments.synchronized_guard import actual_team
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--validation', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise ValueError('Refuse to overwrite a completed calibration')
    torch.set_num_threads(1)
    data, _, _ = load_shards(args.validation, 'validation')
    floors = [0., .25, .5, 1., 1.5, 2., 3., 4.]
    report = dict(role='development label-validation likelihood calibration',
        criterion='minimum mean Gaussian negative log likelihood; smallest floor on ties',
        floor_grid_m=floors, arguments=vars(args), models=[])
    for model_path in args.models:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if checkpoint['config']['revision'] != 'CITF-synchronized-first-action-precise-difference-v1':
            raise ValueError('This calibration declares the precise-team predictor')
        model = PreciseTeamPredictor(PreciseTeamConfig(**checkpoint['config'])).to(args.device).eval()
        model.load_state_dict(checkpoint['state'], strict=True)
        sums, count = [0.] * len(floors), 0
        for start in range(0, len(data['history']), 8):
            batch = {key: value[start:start + 8].to(args.device) for key, value in data.items()}
            prediction = model(batch)
            truth = actual_team(batch)
            error = truth - prediction['guard_mu'][:, :, None]
            spread = prediction['guard_sigma'][:, :, None]
            valid = (batch['candidate_valid'][:, :, None, None]
                     & batch['entity_valid'][:, None, None, :-1]).expand_as(truth)
            count += int(valid.sum())
            for index, floor in enumerate(floors):
                sigma = (spread.square() + floor ** 2).sqrt()
                nll = .5 * (error / sigma).square() + sigma.log() + .5 * math.log(2 * math.pi)
                sums[index] += float(nll[valid].double().sum())
        scores = [value / count for value in sums]
        best = min(range(len(floors)), key=lambda index: (scores[index], floors[index]))
        row = dict(checkpoint=model_path, samples=count, nll=scores,
                   selected_floor_m=floors[best], selected_nll=scores[best])
        report['models'].append(row)
        print(json.dumps(row), flush=True)
        del model
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'calibrate':
        sys.argv.pop(1)
        calibrate_main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--guard-sigma-scale', type=float, default=1.)
    parser.add_argument('--guard-sigma-floor', type=float, default=0.)
    args, remaining = parser.parse_known_args()
    if any(not math.isfinite(x) or x < 0 for x in vars(args).values()):
        raise ValueError('Uncertainty scale and floor must be finite and nonnegative')
    sys.argv = [sys.argv[0], *remaining]
    original_cost = guard.positive_gaussian_square

    def cost(mean, sigma, margin):
        if args.guard_sigma_scale == args.guard_sigma_floor == 0:
            return (mean + margin).clamp_min(0).square()
        spread = ((sigma * args.guard_sigma_scale).square() + args.guard_sigma_floor ** 2).sqrt()
        return original_cost(mean, spread, margin)

    guard.positive_gaussian_square = cost
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(vars(args),
            radial_spread='sqrt((scale * predicted spread)^2 + fixed validation-calibrated floor^2)')
        result['runtime_code'] = canonical_hash({
            'core': result['runtime_code'], 'team_guard_uncertainty': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    selection.main()


if __name__ == '__main__':
    main()
