"""Train the small impulse-response model for its direct online risk score."""
import argparse
from dataclasses import dataclass
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import guard_focus as focus
from experiments import response_delta_guard as difference
from experiments import structured_response as structured


envelope_objective = focus.focused_objective


def direct_objective(output, batch, **kwargs):
    result = envelope_objective(output, batch, **kwargs)
    valid = batch['entity_valid'][:, None, :-1]
    cv = batch['candidate_valid']
    truth = guard.actual_gaps(batch)
    empirical_guard = (output['guard_margin'] + truth).clamp_min(0).square().mean(2)
    empirical_guard = empirical_guard.masked_fill(~valid, torch.inf).amin(-1)
    estimated_guard = output['guard_risk'].masked_fill(~valid, torch.inf).amin(-1)
    base = batch['unary'] + output['objective_risk_weight'] * guard.physical_pair_cost(
        batch, kwargs.get('clearance', 7.))
    empirical = base + output['guard_weight'] * empirical_guard
    estimated = base + output['guard_weight'] * estimated_guard
    temperature = kwargs.get('temperature', 2.)
    probabilities = (-empirical / temperature).masked_fill(~cv, -torch.inf).softmax(-1).detach()
    logp = (-estimated / temperature).masked_fill(~cv, -torch.inf).log_softmax(-1)
    rank = -(probabilities * torch.where(cv, logp, torch.zeros_like(logp))).sum(-1).mean()
    if kwargs.get('mode', 'full') == 'full':
        result['total'] = result['total'] + kwargs.get('rank_weight', .5) * (rank - result['rank'])
    result.update(rank=rank, empirical_cost=empirical, estimated_cost=estimated)
    return result


@dataclass(frozen=True)
class DirectConfig(structured.StructuredConfig):
    revision: str = 'CITF-state-offset-joint-impulse-direct-score-v1'
    selection_objective: str = 'nominal collision plus learned short radial risk and current source preference'


class DirectPredictor(structured.StructuredPredictor):
    pass


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--guard-weight', type=float, default=16.)
    parser.add_argument('--guard-margin', type=float, default=6.)
    online, _ = parser.parse_known_args()
    structured.StructuredConfig, structured.StructuredPredictor = DirectConfig, DirectPredictor
    difference.base_objective = direct_objective
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'structured_direct': file_hash(__file__)})
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller'].update(guard_combination='learned',
                                        selection_objective=DirectConfig().selection_objective)
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                     'structured_direct': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        evaluation.Controller = lambda *positional, **kwargs: guard.PredictedGuardController(
            *positional, guard_weight=online.guard_weight, guard_margin=online.guard_margin,
            collision_source='nominal', **kwargs)
        original_main()

    evaluation.main = evaluate
    structured.main()


if __name__ == '__main__':
    main()
