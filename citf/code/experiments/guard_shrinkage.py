"""Shrink a learned radial residual toward its observable nominal forecast.

The declared scale applies to both the conditional mean correction and its
spread. The controller retains the per-defender nominal-risk envelope and
the same collision forecast, candidate bank and joint selection rule.
"""
import argparse
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import guard_envelope


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mixture-alpha', type=float, required=True)
    args, remaining = parser.parse_known_args()
    if not 0 <= args.mixture_alpha <= 1:
        raise ValueError('Radial residual scale must lie between zero and one')
    sys.argv = [sys.argv[0], *remaining]
    checkpoint_parser = argparse.ArgumentParser(add_help=False)
    checkpoint_parser.add_argument('--checkpoint', required=True)
    checkpoint_args, _ = checkpoint_parser.parse_known_args()
    revision = torch.load(checkpoint_args.checkpoint, map_location='cpu',
                          weights_only=False)['config']['revision']
    original_forward = guard.GuardPredictor.forward

    def forward(self, batch, history_cache=None):
        result = original_forward(self, batch, history_cache=history_cache)
        nominal = result['nominal_guard_mu']
        result['guard_mu'] = nominal + args.mixture_alpha * (result['guard_mu'] - nominal)
        result['guard_sigma'] = (args.mixture_alpha * result['guard_sigma']).clamp_min(1e-6)
        result['guard_risk'] = guard.positive_gaussian_square(
            result['guard_mu'], result['guard_sigma'], result['guard_margin'])
        return result

    guard.GuardPredictor.forward = forward
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(
            radial_residual_scale=args.mixture_alpha,
            radial_distribution='nominal + scale * (predicted stochastic gap - nominal)')
        result['runtime_code'] = canonical_hash({
            'core': result['runtime_code'], 'guard_shrinkage': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    if revision == 'CITF-stopped-guard-envelope-focus-v1':
        from experiments import guard_focus_evaluate
        guard_focus_evaluate.main()
    elif revision == 'CITF-ordered-stopped-guard-envelope-v1':
        from experiments import ordered_guard
        ordered_guard.main()
    elif revision == 'CITF-stopped-guard-query-v1':
        guard_envelope.main()
    else:
        raise ValueError('Unregistered radial predictor revision')


if __name__ == '__main__':
    main()
