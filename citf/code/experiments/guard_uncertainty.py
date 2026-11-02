"""Calibrate the contribution of short-branch radial prediction uncertainty.

The learned signed radial-gap mean is unchanged. A declared online scale
controls how much of its fitted conditional spread contributes to the
positive-square guarding cost. Zero uses the point prediction. Every scale
has a distinct controller identity and retains the same candidate search.
"""
import argparse
import math
import sys

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--guard-sigma-scale', type=float, required=True)
    args, remaining = parser.parse_known_args()
    if not math.isfinite(args.guard_sigma_scale) or args.guard_sigma_scale < 0:
        raise ValueError('A finite nonnegative radial uncertainty scale is required')
    sys.argv = [sys.argv[0], *remaining]
    original_cost = guard.positive_gaussian_square

    def cost(mean, sigma, margin):
        if args.guard_sigma_scale == 0:
            return (mean + margin).clamp_min(0).square()
        return original_cost(mean, sigma * args.guard_sigma_scale, margin)

    guard.positive_gaussian_square = cost
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(
            guard_sigma_scale=args.guard_sigma_scale,
            radial_risk='positive square of signed gap with declared conditional-spread scaling')
        result['runtime_code'] = canonical_hash({
            'core': result['runtime_code'], 'guard_uncertainty': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    guard.evaluate_main()


if __name__ == '__main__':
    main()
