"""Compare declared online use of already trained team-geometry predictors."""
import argparse
import importlib
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard


PREDICTORS = {
    'CITF-synchronized-team-guard-v1': ('synchronized_guard', 'SynchronizedPredictor'),
    'CITF-synchronized-first-action-team-guard-v1': ('synchronized_recourse', 'TeamRecoursePredictor'),
    'CITF-synchronized-first-action-precise-difference-v1': ('synchronized_delta', 'PreciseTeamPredictor'),
    'CITF-synchronized-first-action-endpoint-geometry-v1': ('endpoint_guard', 'EndpointPredictor'),
    'CITF-state-offset-joint-impulse-response-v1': ('structured_response', 'StructuredPredictor'),
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--guard-combination', choices=['envelope', 'learned'], required=True)
    parser.add_argument('--mixture-alpha', type=float, default=1.)
    args, remaining = parser.parse_known_args()
    if not 0 <= args.mixture_alpha <= 1:
        raise ValueError('The declared mean and spread scale must lie in [0, 1]')
    view = argparse.ArgumentParser(add_help=False)
    view.add_argument('--checkpoint', required=True)
    view.add_argument('--guard-weight', type=float, required=True)
    view.add_argument('--guard-margin', type=float, required=True)
    online, _ = view.parse_known_args(remaining)
    revision = torch.load(online.checkpoint, map_location='cpu', weights_only=False)['config']['revision']
    if revision not in PREDICTORS:
        raise ValueError('Unregistered team-geometry checkpoint')
    module_name, class_name = PREDICTORS[revision]
    module = importlib.import_module('experiments.' + module_name)
    predictor = getattr(module, class_name)
    original_forward = predictor.forward

    def forward(self, batch, history_cache=None):
        result = original_forward(self, batch, history_cache=history_cache)
        nominal = result['nominal_guard_mu']
        result['guard_mu'] = nominal + args.mixture_alpha * (result['guard_mu'] - nominal)
        result['guard_sigma'] = (args.mixture_alpha * result['guard_sigma']).clamp_min(1e-6)
        result['guard_risk'] = guard.positive_gaussian_square(
            result['guard_mu'], result['guard_sigma'], result['guard_margin'])
        return result

    predictor.forward = forward
    sys.argv = [sys.argv[0], *remaining]
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller'].update(guard_combination=args.guard_combination,
                radial_residual_scale=args.mixture_alpha,
                radial_distribution='nominal + alpha * (predicted stochastic gap - nominal)')
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                     'team_guard_selection': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        if args.guard_combination == 'learned':
            evaluation.Controller = lambda *positional, **kwargs: guard.PredictedGuardController(
                *positional, guard_weight=online.guard_weight, guard_margin=online.guard_margin,
                collision_source='nominal', **kwargs)
        original_main()

    evaluation.main = evaluate
    module.main()


if __name__ == '__main__':
    main()
