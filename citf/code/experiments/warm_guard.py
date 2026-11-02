"""Start a separately identified radial training run from existing weights.

The optimizer and update counter start afresh. The initializer is read only;
the new data, validation, zero-update snapshot and selected checkpoint belong
to the new run. This is not a resume with changed data.
"""
import argparse
from dataclasses import asdict
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_focus as focus
from experiments import guard_prediction as guard
from experiments.ordered_guard import OrderedConfig, OrderedPredictor


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--initial-checkpoint', required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False)
    revision = checkpoint['config']['revision']
    if revision == focus.FocusConfig().revision:
        config_class, predictor_class = focus.FocusConfig, guard.GuardPredictor
    elif revision == OrderedConfig().revision:
        config_class, predictor_class = OrderedConfig, OrderedPredictor
    else:
        raise ValueError('Unregistered initializer architecture')

    class InitializedPredictor(predictor_class):
        def __init__(self, config):
            super().__init__(config)
            if asdict(config) != checkpoint['config']:
                raise ValueError('Warm start requires unchanged architecture and physical objective')
            self.load_state_dict(checkpoint['state'], strict=True)

    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'warm_guard': file_hash(__file__),
        'initializer_checkpoint': file_hash(args.initial_checkpoint),
        'initializer_revision': revision})
    guard.GuardPredictor = InitializedPredictor
    focus.FocusConfig = config_class
    print({'initial_checkpoint': args.initial_checkpoint,
           'initial_selected': checkpoint.get('selected'),
           'optimization': 'new optimizer and zero-based updates'}, flush=True)
    focus.main()


if __name__ == '__main__':
    main()
