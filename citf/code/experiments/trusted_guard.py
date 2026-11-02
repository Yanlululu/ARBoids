"""Rerank a common physical candidate pool within a nominal-cost allowance.

The established physical controller first proposes and refines its own pool.
Learning may choose only a configuration whose complete nominal cost is at
most the physical optimum plus the declared allowance. This is a current
configuration constraint, with no future outcome or terminal value input.
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


class TrustedController(guard.PredictedGuardController):
    def __init__(self, *args, nominal_slack, **kwargs):
        super().__init__(*args, **kwargs)
        self.nominal_slack = nominal_slack

    @torch.no_grad()
    def act(self, world):
        start = time.perf_counter()
        f, protocol = factorize(world, self.source), world.protocol
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), protocol.risk_weight, 128)
        initial, centre = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                          f['prior'].numpy(), self.bank, self.budget)

        def physical_score(plans):
            batch, _, _ = inputs(world, f, plans)
            gap = guard.nominal_gaps(batch)[1]
            radial = (self.guard_margin + gap).clamp_min(0).square().amin(-1)[0]
            base = nominal_score(plans, f['unary'].numpy(), f['pair'].numpy(), protocol.risk_weight)
            return base + self.guard_weight * radial.numpy()

        physical_best, trace = whole_configuration_search(initial, physical_score,
            [row.nonzero()[:, 0].numpy() for row in f['valid']], self.budget, self.rounds)
        plans = np.asarray(trace['configurations'])
        nominal_cost = np.asarray(trace['scores'])
        feasible = nominal_cost <= trace['cost'] + self.nominal_slack + 1e-10
        if self.model is None:
            cost = nominal_cost
            learned_radial = None
        else:
            batch, _, _ = inputs(world, f, plans)
            batch = {key: value.to(self.device) for key, value in batch.items()}
            output = self.model(batch)
            nominal_radial = (self.guard_margin + output['nominal_guard_mu']).clamp_min(0).square()
            learned_risk = guard.positive_gaussian_square(
                output['guard_mu'], output['guard_sigma'], self.guard_margin)
            radial = torch.maximum(nominal_radial, learned_risk).amin(-1)[0].cpu().numpy()
            base = nominal_score(plans, f['unary'].numpy(), f['pair'].numpy(), protocol.risk_weight)
            cost = base + self.guard_weight * radial
            learned_radial = learned_risk.amin(-1)[0].cpu().tolist()
        selected = min(np.flatnonzero(feasible), key=lambda index: (cost[index], tuple(plans[index])))
        indices = plans[selected]
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        trace.update(cost=float(cost[selected]), scores=cost.tolist(),
            selected_indices=indices.tolist(), gates=gates.tolist(), physical_thrust=force.tolist(),
            source_center_indices=centre.tolist(), source_center_scored=centre.tolist() in trace['configurations'],
            initial_configurations=initial.tolist(), physical_best_indices=physical_best.tolist(),
            nominal_costs=nominal_cost.tolist(), nominal_cost_allowance=self.nominal_slack,
            eligible_configurations=feasible.tolist(), selected_nominal_cost=float(nominal_cost[selected]),
            minimum_nominal_cost=float(nominal_cost.min()), learned_radial_costs=learned_radial,
            neural_forward_batches=int(self.model is not None),
            compute_seconds=time.perf_counter() - start)
        return force, trace


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--nominal-slack', type=float, required=True)
    args, remaining = parser.parse_known_args()
    if not math.isfinite(args.nominal_slack) or args.nominal_slack < 0:
        raise ValueError('Nominal allowance must be finite and nonnegative')
    sys.argv = [sys.argv[0], *remaining]
    checkpoint_parser = argparse.ArgumentParser(add_help=False)
    checkpoint_parser.add_argument('--checkpoint')
    checkpoint_args, _ = checkpoint_parser.parse_known_args()
    revision = None
    if checkpoint_args.checkpoint:
        revision = torch.load(checkpoint_args.checkpoint, map_location='cpu',
                              weights_only=False)['config']['revision']
    if revision == 'CITF-stopped-guard-envelope-focus-v1':
        from experiments.guard_focus import FocusConfig
        guard.GuardConfig = FocusConfig
    elif revision == 'CITF-ordered-stopped-guard-envelope-v1':
        from experiments.ordered_guard import OrderedConfig, OrderedPredictor
        guard.GuardConfig, guard.GuardPredictor = OrderedConfig, OrderedPredictor
    elif revision == 'CITF-control-response-attention-guard-v1':
        from experiments.response_guard import ResponseConfig, ResponsePredictor
        guard.GuardConfig, guard.GuardPredictor = ResponseConfig, ResponsePredictor
    elif revision not in (None, 'CITF-stopped-guard-query-v1'):
        raise ValueError('Unregistered constrained-reranking predictor')
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(
            selection_pool='complete physical guard search pool, learned score does not alter proposals',
            nominal_cost_allowance=args.nominal_slack,
            eligibility='complete nominal cost <= physical pool optimum + allowance',
            learned_score='per-vessel nominal/learned radial risk envelope')
        result['runtime_code'] = canonical_hash({
            'core': result['runtime_code'], 'trusted_guard': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    guard.PredictedGuardController = lambda *positional, **kwargs: TrustedController(
        *positional, nominal_slack=args.nominal_slack, **kwargs)
    guard.evaluate_main()


if __name__ == '__main__':
    main()
