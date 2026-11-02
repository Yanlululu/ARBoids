"""Guard training with observable-geometry root sampling.

Mix uniform roots with a smooth preference for states whose nominal radial
margin is near the defense boundary. Future outcomes and future labels never
determine sampling probabilities. Validation retains its original measure.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard


class AttributeProxy:
    def __init__(self, original, **replacements):
        self.original = original
        self.replacements = replacements

    def __getattr__(self, name):
        return self.replacements[name] if name in self.replacements else getattr(self.original, name)


class RootSampler:
    def __init__(self, original, probability):
        self.original = original
        self.probability = probability

    @property
    def bit_generator(self):
        return self.original.bit_generator

    def integers(self, low, high=None, size=None, **kwargs):
        if low == 0 and high == len(self.probability) and not kwargs:
            return self.original.choice(high, size=size, replace=True, p=self.probability)
        return self.original.integers(low, high, size, **kwargs)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--priority-fraction', type=float, default=.5)
    ap.add_argument('--priority-window', type=float, default=12.)
    args, remaining = ap.parse_known_args()
    if not 0 <= args.priority_fraction <= 1 or args.priority_window <= 0:
        raise ValueError('Invalid observable-geometry sampling mixture')
    sys.argv = [sys.argv[0], *remaining, '--priority-fraction', str(args.priority_fraction),
                '--priority-window', str(args.priority_window)]
    import citf.train as training
    original_load = training.load_shards
    probabilities = {}
    def load_shards(paths, role):
        result = original_load(paths, role)
        if role == 'train':
            data = result[0]
            gaps = guard.nominal_gaps(data)[1]
            closest_guard = gaps.masked_fill(~data['entity_valid'][:, None, :-1], torch.inf).amin(-1)
            valid = data['candidate_valid']
            critical = (torch.sigmoid((closest_guard + args.priority_window) / 4.) * valid).sum(-1) / valid.sum(-1)
            weight = critical.double().numpy() + 1e-8
            probability = (1 - args.priority_fraction) / len(weight) + args.priority_fraction * weight / weight.sum()
            probabilities['train'] = probability / probability.sum()
            print({'sampling': 'observable nominal radial margin', 'roots': len(weight),
                   'fraction': args.priority_fraction, 'window_m': args.priority_window,
                   'effective_roots': float(1 / np.square(probability).sum())}, flush=True)
        return result
    training.load_shards = load_shards
    original_rng = np.random.default_rng
    def rng(seed=None):
        original = original_rng(seed)
        return RootSampler(original, probabilities['train']) if args.priority_fraction else original
    training.np = AttributeProxy(np, random=AttributeProxy(np.random, default_rng=rng))
    class TrainingParser(argparse.ArgumentParser):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            self.add_argument('--priority-fraction', type=float, default=args.priority_fraction)
            self.add_argument('--priority-window', type=float, default=args.priority_window)
    training.argparse = AttributeProxy(argparse, ArgumentParser=TrainingParser)
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda *positional, **kwargs: canonical_hash({
        'guard': original_hash(*positional, **kwargs), 'priority_guard': file_hash(Path(__file__)),
        'sampler': vars(args), 'criterion': 'mean over candidates of sigmoid((nominal minimum guard gap + window)/4)'})
    guard.train_main()


if __name__ == '__main__':
    main()
