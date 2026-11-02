"""Train candidate interaction directly for conservative radial geometry.

This development objective supervises the stopped radial-gap distribution and
the exact nominal/learned envelope used by guard_envelope. Auxiliary motion
and defender-pair distance errors do not compete for the shared representation.
The prediction-only control receives the same physical likelihood training,
without risk or configuration-ranking supervision.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from citf.risk import bounded_mean
from experiments import guard_prediction as guard
from experiments import priority_guard


@dataclass(frozen=True)
class FocusConfig(guard.GuardConfig):
    revision: str = 'CITF-stopped-guard-envelope-focus-v1'


def focused_objective(out, b, clearance=7., mode='full', risk_weight=1., rank_weight=.5,
                      temperature=2., **kwargs):
    if mode not in ('full', 'predictive_only'):
        raise ValueError('Focused radial learning supports full or prediction-only training')
    truth = guard.actual_gaps(b)
    mean, sigma = out['guard_mu'][:, :, None], out['guard_sigma'][:, :, None]
    valid = b['entity_valid'][:, :-1]
    cv = b['candidate_valid']
    mask = valid[:, None, None] & cv[:, :, None, None]
    nll = bounded_mean(.5 * ((truth - mean) / sigma).square() + sigma.log() + .5 * math.log(2 * math.pi), mask)
    margin = out['guard_margin']
    nominal = (margin + out['nominal_guard_mu']).clamp_min(0).square()
    true_risk = (margin + truth).clamp_min(0).square().mean(2)
    predicted_risk = out['guard_risk']
    # Per-vessel risk fidelity supplies a dense physical signal; ranking uses
    # the same min-over-guards envelope as the online controller.
    scale = max(margin, 2.) ** 2
    normalized_error = (predicted_risk - true_risk) / scale
    risk = bounded_mean(F.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error),
                                       reduction='none'), valid[:, None] & cv[:, :, None])
    empirical_guard = torch.maximum(nominal, true_risk).masked_fill(~valid[:, None], torch.inf).amin(-1)
    estimated_guard = torch.maximum(nominal, predicted_risk).masked_fill(~valid[:, None], torch.inf).amin(-1)
    base = b['unary'] + out['objective_risk_weight'] * guard.physical_pair_cost(b, clearance)
    empirical = base + out['guard_weight'] * empirical_guard
    estimated = base + out['guard_weight'] * estimated_guard
    probabilities = (-empirical / temperature).masked_fill(~cv, -torch.inf).softmax(-1).detach()
    logp = (-estimated / temperature).masked_fill(~cv, -torch.inf).log_softmax(-1)
    rank = -(probabilities * torch.where(cv, logp, torch.zeros_like(logp))).sum(-1).mean()
    zero = mean.sum() * 0
    total = out['guard_nll_weight'] * nll
    if mode == 'full':
        total = total + risk_weight * risk + rank_weight * rank
    return dict(total=total, motion=zero, nll=nll, risk=risk, rank=rank, context=zero,
                guard_nll=nll, empirical_cost=empirical, estimated_cost=estimated)


def main():
    original_identity = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_identity(path), 'guard_focus': file_hash(__file__),
        'objective': 'radial likelihood plus radial-risk regression and exact envelope choice ranking; no auxiliary motion loss'})
    guard.GuardConfig = FocusConfig
    guard.guard_objective = focused_objective
    priority_guard.main()


if __name__ == '__main__':
    main()
