"""Use a bounded learned radial correction without the one-sided envelope.

The predictive random gap becomes nominal + alpha * (learned - nominal).
Both positive and negative corrections can affect the current configuration;
the nominal collision cost, source preference and candidate search are kept.
"""
import argparse
import sys

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import recourse_impulse_guard as impulse


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mixture-alpha', type=float, required=True)
    parser.add_argument('--guard-combination', choices=['learned'], default='learned')
    args, remaining = parser.parse_known_args()
    if not 0 <= args.mixture_alpha <= 1:
        raise ValueError('The radial correction scale must lie between zero and one')
    sys.argv = [sys.argv[0], *remaining, '--guard-combination', 'learned']
    original_forward = impulse.RecourseImpulsePredictor.forward

    def forward(self, batch, history_cache=None):
        result = original_forward(self, batch, history_cache=history_cache)
        nominal = result['nominal_guard_mu']
        result['guard_mu'] = nominal + args.mixture_alpha * (result['guard_mu'] - nominal)
        result['guard_sigma'] = (args.mixture_alpha * result['guard_sigma']).clamp_min(1e-6)
        result['guard_risk'] = guard.positive_gaussian_square(
            result['guard_mu'], result['guard_sigma'], result['guard_margin'])
        return result

    impulse.RecourseImpulsePredictor.forward = forward
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(radial_residual_scale=args.mixture_alpha,
            radial_distribution='nominal + alpha * (predicted stochastic gap - nominal)',
            allows_negative_nominal_correction=True)
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'recourse_blend': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    impulse.main()


if __name__ == '__main__':
    main()
