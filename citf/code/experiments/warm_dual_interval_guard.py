"""Initialize the longer geometric model from the trained short predictor.

History and shared layers transfer directly. The first sixteen trajectory
columns transfer into the longer input layers; added future columns start at
zero. All optimization, long labels and checkpoint selection belong to the
new run. There is no claim of equivalent initial long-horizon predictions.
"""
import argparse
from dataclasses import dataclass
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import dual_interval_guard as dual


@dataclass(frozen=True)
class WarmDualConfig(dual.DualIntervalConfig):
    revision: str = 'CITF-short-to-long-initialized-recourse-guard-v1'
    parameter_initialization: str = 'short-model shared layers and first sixteen trajectory columns; new future columns zero; new optimizer'


class WarmDualPredictor(dual.DualIntervalPredictor):
    pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--initial-checkpoint')
    args, remaining = parser.parse_known_args()
    training = bool(remaining and remaining[0] == 'train')
    if training != bool(args.initial_checkpoint):
        raise ValueError('An initializer is required only for the new training run')
    sys.argv = [sys.argv[0], *remaining]
    predictor = WarmDualPredictor
    initial = None
    if training:
        checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False)
        if checkpoint['config']['revision'] != 'CITF-synchronized-first-action-precise-difference-v1':
            raise ValueError('This transfer declares the trained precise short-team model')

        class InitializedPredictor(WarmDualPredictor):
            def __init__(self, config):
                super().__init__(config)
                for key in ('width', 'heads', 'history', 'dt', 'clearance', 'fleet_candidate_context'):
                    if getattr(config, key) != checkpoint['config'][key]:
                        raise ValueError('Initializer configuration differs: ' + key)
                if config.horizon != 32 or checkpoint['config']['horizon'] != 16:
                    raise ValueError('Only the declared sixteen-to-thirty-two-step transfer is supported')
                current = self.state_dict()
                for key, value in current.items():
                    old = checkpoint['state'][key]
                    if old.shape == value.shape:
                        current[key] = old.clone()
                    elif key == 'candidate_embed.0.weight':
                        value.zero_()
                        value[:, :8 + 2 * 16] = old
                    elif key == 'guard_embed.0.weight':
                        value.zero_()
                        value[:, :2 * config.width + 16] = old[:, :-2]
                        value[:, -2:] = old[:, -2:]
                    else:
                        raise ValueError('Undeclared horizon-dependent parameter: ' + key)
                self.load_state_dict(current, strict=True)

        predictor = InitializedPredictor
        initial = dict(path=args.initial_checkpoint, revision=checkpoint['config']['revision'],
                       checkpoint_sha256=file_hash(args.initial_checkpoint), parameters_only=True)
        print(dict(initializer=initial, initialization=WarmDualConfig().parameter_initialization), flush=True)
    dual.DualIntervalConfig, dual.DualIntervalPredictor = WarmDualConfig, predictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'warm_dual_interval_guard': file_hash(__file__), 'initializer': initial})
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller']['parameter_initialization'] = WarmDualConfig().parameter_initialization
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'warm_dual_interval_guard': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    dual.main()


if __name__ == '__main__':
    main()
