"""Initialize a new one-action recourse run from compatible response weights.

Only parameters transfer. Held-coefficient examples are not added to the
recourse objective. Data, optimizer, update count and model selection start
as a separately identified run with a real zero-update snapshot.
"""
import argparse
from dataclasses import asdict
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import recourse_impulse_guard as impulse


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--initial-checkpoint', required=True)
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == 'train':
        remaining.pop(0)
    sys.argv = [sys.argv[0], 'train', *remaining]
    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=False)
    revision = checkpoint['config']['revision']
    if revision not in ('CITF-control-response-attention-guard-v1',
                        'CITF-control-response-radial-difference-v1',
                        impulse.RecourseImpulseConfig().revision):
        raise ValueError('Initializer must have the declared control-response parameterization')
    original_predictor = impulse.RecourseImpulsePredictor

    class InitializedPredictor(original_predictor):
        def __init__(self, config):
            super().__init__(config)
            current = asdict(config)
            for key in ('width', 'heads', 'history', 'horizon', 'dt', 'clearance', 'fleet_candidate_context'):
                if current[key] != checkpoint['config'][key]:
                    raise ValueError('Initializer geometry or architecture differs: ' + key)
            self.load_state_dict(checkpoint['state'], strict=True)

    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'warm_recourse': file_hash(__file__),
        'initializer_checkpoint': file_hash(args.initial_checkpoint),
        'initializer_revision': revision})
    impulse.RecourseImpulsePredictor = InitializedPredictor
    print(dict(initial_checkpoint=args.initial_checkpoint, initial_revision=revision,
               initialization='parameters only; new objective, optimizer, data and zero-based updates'), flush=True)
    impulse.main()


if __name__ == '__main__':
    main()
