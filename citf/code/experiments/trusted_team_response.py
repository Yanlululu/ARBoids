"""Use learned first-action geometry to resolve physically admissible choices.

The complete physical search is unchanged. Near the observable radial
boundary, only configurations within a declared physical cost allowance may
be reranked by the predicted mean stopped team gap. Candidate uncertainty and
common state offsets cannot create an additional candidate preference here.
"""
import argparse
import math
import sys
import time

import numpy as np
import torch

from citf.bridge import factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.search import nominal_beam, nominal_score, initialize_bank, whole_configuration_search
from experiments import guard_prediction as guard
from experiments import synchronized_delta as precise


class TeamResponseController(guard.PredictedGuardController):
    def __init__(self, *args, nominal_slack, **kwargs):
        super().__init__(*args, **kwargs)
        self.nominal_slack = nominal_slack

    @torch.no_grad()
    def act(self, world):
        started = time.perf_counter()
        factors, protocol = factorize(world, self.source), world.protocol
        unary, pair = factors['unary'].numpy(), factors['pair'].numpy()
        ranked, _ = nominal_beam(unary, pair, protocol.risk_weight, 128)
        initial, centre = initialize_bank(ranked, factors['grid'].numpy(),
            factors['valid'].numpy(), factors['prior'].numpy(), self.bank, self.budget)

        def physical_score(plans):
            batch, _, _ = inputs(world, factors, plans)
            gap = guard.nominal_gaps(batch)[1]
            radial = (self.guard_margin + gap).clamp_min(0).square().amin(-1)[0]
            return nominal_score(plans, unary, pair, protocol.risk_weight) + self.guard_weight * radial.numpy()

        physical_best, trace = whole_configuration_search(initial, physical_score,
            [row.nonzero()[:, 0].numpy() for row in factors['valid']], self.budget, self.rounds)
        plans, physical_cost = np.asarray(trace['configurations']), np.asarray(trace['scores'])
        feasible = physical_cost <= trace['cost'] + self.nominal_slack + 1e-10
        eligible = np.flatnonzero(feasible)
        selected = min(eligible, key=lambda index: (physical_cost[index], tuple(plans[index])))
        radius = np.linalg.norm(world.state()[:, :2], axis=-1)
        current_gap = float(radius[:-1].min() - radius[-1])
        means = None
        if self.model is not None and len(eligible) > 1 and current_gap >= -16.:
            batch, _, _ = inputs(world, factors, plans[eligible])
            batch = {key: value.to(self.device) for key, value in batch.items()}
            prediction = self.model(batch)
            means = prediction['guard_mu'][0, :, 0].cpu().numpy()
            # This is a mean of stopped physical distances, not a value or
            # outcome target. Tie breaking remains deterministic and physical.
            local = min(range(len(eligible)), key=lambda index:
                (means[index], physical_cost[eligible[index]], tuple(plans[eligible[index]])))
            selected = eligible[local]
        indices = plans[selected]
        gates = factors['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * factors['learned'] + (1 - gates) * factors['rules']) + 250).numpy()
        trace.update(cost=float(physical_cost[selected]), selected_indices=indices.tolist(),
            gates=gates.tolist(), physical_thrust=force.tolist(),
            source_center_indices=centre.tolist(), source_center_scored=centre.tolist() in trace['configurations'],
            initial_configurations=initial.tolist(), physical_best_indices=physical_best.tolist(),
            minimum_nominal_cost=float(physical_cost.min()), nominal_cost_allowance=self.nominal_slack,
            selected_nominal_cost=float(physical_cost[selected]), eligible_configurations=feasible.tolist(),
            current_radial_gap_m=current_gap, learned_mean_gap_m=None if means is None else means.tolist(),
            neural_evaluated_unique=0 if means is None else len(eligible),
            neural_forward_batches=int(means is not None), compute_seconds=time.perf_counter() - started)
        return force, trace


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--nominal-slack', type=float, required=True)
    args, remaining = parser.parse_known_args()
    if not math.isfinite(args.nominal_slack) or args.nominal_slack < 0:
        raise ValueError('A finite nonnegative physical allowance is required')
    view = argparse.ArgumentParser(add_help=False)
    view.add_argument('--guard-weight', type=float, required=True)
    view.add_argument('--guard-margin', type=float, required=True)
    online, _ = view.parse_known_args(remaining)
    sys.argv = [sys.argv[0], *remaining]
    import citf.evaluate as evaluation
    original_main, original_identity = evaluation.main, evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(selection_pool='unchanged complete physical search pool',
            nominal_cost_allowance=args.nominal_slack,
            learned_selection='minimum predicted mean stopped team gap among physically admissible configurations',
            current_gap_activation_m=-16., candidate_uncertainty_used_for_selection=False)
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'trusted_team_response': file_hash(__file__)})
        return result

    def evaluate():
        evaluation.Controller = lambda *positional, **kwargs: TeamResponseController(
            *positional, nominal_slack=args.nominal_slack, **vars(online),
            collision_source='nominal', **kwargs)
        original_main()

    evaluation.runtime_identity, evaluation.main = identity, evaluate
    precise.main()


if __name__ == '__main__':
    main()
