"""Learn unrestricted candidate-dependent stopped team geometry.

The prediction determines the radial score in both training and control.
There is no maximum with a nominal radial penalty. Known physical collision
geometry and the common candidate/search interface remain unchanged.
"""
import argparse
from dataclasses import dataclass
import math
import sys

import torch
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash, tree_hash
from citf.risk import bounded_mean
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team
from experiments.synchronized_delta import PreciseTeamConfig, PreciseTeamPredictor


@dataclass(frozen=True)
class FreeConfig(PreciseTeamConfig):
    revision: str = 'CITF-unrestricted-team-response-v1'
    radial_decision: str = 'learned stopped geometry without nominal lower bound'
    guard_weight: float = 16.
    guard_margin: float = 6.
    guard_nll_weight: float = 1.


class FreePredictor(PreciseTeamPredictor):
    pass


def free_objective(out, b, clearance=7., mode='full', risk_weight=1.,
                   rank_weight=.5, temperature=2., **kwargs):
    truth = team.actual_team(b)[:, :, :, 0]
    mean = out['guard_mu'][:, :, 0]
    sigma = out['guard_sigma'][:, :, 0]
    cv = b['candidate_valid']
    nll = bounded_mean(.5 * ((truth - mean[..., None]) / sigma[..., None]).square()
        + sigma[..., None].log() + .5 * math.log(2 * math.pi), cv[..., None])
    true_risk = (out['guard_margin'] + truth).clamp_min(0).square().mean(-1)
    predicted_risk = out['guard_risk'][:, :, 0]
    scale = max(out['guard_margin'], 2.) ** 2
    risk = bounded_mean(F.smooth_l1_loss(predicted_risk / scale,
        true_risk / scale, reduction='none'), cv)
    base = b['unary'] + out['objective_risk_weight'] * guard.physical_pair_cost(b, clearance)
    empirical = base + out['guard_weight'] * true_risk
    estimated = base + out['guard_weight'] * predicted_risk
    target = (-empirical / temperature).masked_fill(~cv, -torch.inf).softmax(-1).detach()
    logp = (-estimated / temperature).masked_fill(~cv, -torch.inf).log_softmax(-1)
    rank = -(target * torch.where(cv, logp, torch.zeros_like(logp))).sum(-1).mean()
    count = cv.sum(-1, keepdim=True).clamp_min(1)
    actual_mean = truth.mean(-1)
    actual_delta = actual_mean - (actual_mean * cv).sum(-1, keepdim=True) / count
    predicted_delta = mean - (mean * cv).sum(-1, keepdim=True) / count
    contrast = bounded_mean(F.smooth_l1_loss(predicted_delta / out['response_delta_scale_m'],
        actual_delta / out['response_delta_scale_m'], reduction='none'), cv)
    total = out['guard_nll_weight'] * nll
    if mode == 'full':
        total = total + risk_weight * risk + rank_weight * rank + out['response_delta_weight'] * contrast
    zero = mean.sum() * 0
    return dict(total=total, motion=zero, nll=nll, risk=risk, rank=rank, context=contrast,
                empirical_cost=empirical, estimated_cost=estimated)


def implementation(path):
    return canonical_hash({'core': tree_hash(path), 'free_team_guard': file_hash(__file__)})


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--guard-weight', type=float, default=16.)
    parser.add_argument('--guard-margin', type=float, default=6.)
    parser.add_argument('--guard-nll-weight', type=float, default=1.)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if remaining and remaining[0] == 'train':
        sys.argv.pop(1)
        import citf.train as training
        training.Config = lambda **kwargs: FreeConfig(**vars(args), **kwargs)
        training.Predictor = FreePredictor
        training.objective = free_objective
        training.tree_hash = implementation
        training.main()
    else:
        import citf.collect as collection
        import citf.evaluate as evaluation
        collection.Config, collection.Predictor = FreeConfig, FreePredictor
        guard.nominal_gaps = team.nominal_team
        evaluation.Controller = lambda *positional, **kwargs: guard.PredictedGuardController(
            *positional, guard_weight=args.guard_weight, guard_margin=args.guard_margin,
            collision_source='nominal', **kwargs)
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['method'] = 'CITF-FreeResponse-development' if result['checkpoint_sha256'] else 'SynchronizedGuard-Phys-development'
            result['controller'].update(guard_weight=args.guard_weight, guard_margin=args.guard_margin,
                guard_combination='learned without nominal lower bound', collision_source='nominal',
                prediction_condition=FreeConfig().prediction_condition)
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'free_team_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        evaluation.main()


if __name__ == '__main__':
    main()
