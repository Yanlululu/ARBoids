"""Train candidate effects on the physical stopped radial-guard quantity.

For a fixed focal defender candidate, compare all available configurations
that change other defenders. Shared-future physical labels provide the target
change in radial margin. This is a short geometric difference, not a value or
terminal-outcome target. Sampling uses observable nominal geometry only.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments.priority_guard import AttributeProxy, RootSampler


def interaction_loss(out, b, scale):
    candidate = b['candidate'][:, :, :-1, :2]
    same = (candidate[:, :, None] == candidate[:, None, :]).all(-1)
    count = candidate.shape[1]
    same &= ~torch.eye(count, device=candidate.device, dtype=torch.bool)[None, :, :, None]
    valid = b['candidate_valid']
    same &= valid[:, :, None, None] & valid[:, None, :, None] & b['entity_valid'][:, None, None, :-1]
    target = guard.actual_gaps(b).mean(2)
    residual = out['guard_mu'] - target
    difference = (residual[:, :, None] - residual[:, None, :]) / scale
    nominal = guard.nominal_gaps(b)[1]
    relative = nominal - nominal.masked_fill(~b['entity_valid'][:, None, :-1], torch.inf).amin(-1, keepdim=True)
    relevance = torch.exp(-relative / 8.)
    weight = same * torch.sqrt(relevance[:, :, None] * relevance[:, None, :])
    loss = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction='none')
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--interaction-weight', type=float, default=.25)
    ap.add_argument('--interaction-scale', type=float, default=.2)
    ap.add_argument('--priority-fraction', type=float, default=.5)
    ap.add_argument('--priority-window', type=float, default=12.)
    args, remaining = ap.parse_known_args()
    if args.interaction_weight < 0 or args.interaction_scale <= 0 or not 0 <= args.priority_fraction <= 1 or args.priority_window <= 0:
        raise ValueError('Invalid physical interaction training parameters')
    sys.argv = [sys.argv[0], *remaining]
    for key, value in vars(args).items():
        sys.argv.extend(['--' + key.replace('_', '-'), str(value)])
    import citf.train as training
    original_load = training.load_shards
    probabilities = {}
    def load_shards(paths, role):
        result = original_load(paths, role)
        if role == 'train':
            data = result[0]
            gaps = guard.nominal_gaps(data)[1]
            closest = gaps.masked_fill(~data['entity_valid'][:, None, :-1], torch.inf).amin(-1)
            valid = data['candidate_valid']
            critical = (torch.sigmoid((closest + args.priority_window) / 4.) * valid).sum(-1) / valid.sum(-1)
            weight = critical.double().numpy() + 1e-8
            probability = (1 - args.priority_fraction) / len(weight) + args.priority_fraction * weight / weight.sum()
            probabilities['train'] = probability / probability.sum()
            print({'sampling': 'observable radial margin', 'roots': len(weight), 'parameters': vars(args)}, flush=True)
        return result
    training.load_shards = load_shards
    original_rng = np.random.default_rng
    training.np = AttributeProxy(np, random=AttributeProxy(np.random,
        default_rng=lambda seed=None: RootSampler(original_rng(seed), probabilities['train']) if args.priority_fraction else original_rng(seed)))
    class TrainingParser(argparse.ArgumentParser):
        def __init__(self, *positional, **kwargs):
            super().__init__(*positional, **kwargs)
            for key, value in vars(args).items():
                self.add_argument('--' + key.replace('_', '-'), type=float, default=value)
    training.argparse = AttributeProxy(argparse, ArgumentParser=TrainingParser)
    original_objective = guard.guard_objective
    def objective(out, batch, **kwargs):
        result = original_objective(out, batch, **kwargs)
        effect = interaction_loss(out, batch, args.interaction_scale)
        if kwargs.get('mode', 'full') == 'full':
            result['total'] = result['total'] + args.interaction_weight * effect
        result['context'] = effect
        return result
    guard.guard_objective = objective
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda *positional, **kwargs: canonical_hash({
        'guard': original_hash(*positional, **kwargs), 'interaction_guard': file_hash(Path(__file__)),
        'priority_guard': file_hash(Path(__file__).with_name('priority_guard.py')), 'parameters': vars(args)})
    guard.train_main()


if __name__ == '__main__':
    main()
