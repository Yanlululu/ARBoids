"""Average the physical radial-risk distributions of every declared seed.

The members share data and architecture. Their expected short-interval costs
are averaged before taking the nominal envelope and choosing a guard. All
declared members are used, without selecting seeds from closed-loop results.
"""
import argparse
from dataclasses import asdict
import sys
import time

import numpy as np
import torch
from torch import nn

from citf.bridge import factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.search import nominal_beam, nominal_score, initialize_bank, whole_configuration_search
from experiments import guard_prediction as guard


class EnsemblePredictor(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.cfg = members[0].cfg
        if any(asdict(member.cfg) != asdict(self.cfg) for member in members[1:]):
            raise ValueError('Ensemble members must use the same predictor configuration')

    def encode_history(self, batch):
        return [member.encode_history(batch) for member in self.members]

    def radial_risk(self, batch, history, margin):
        risks = []
        for member, cached in zip(self.members, history):
            output = member(batch, history_cache=cached)
            risks.append(guard.positive_gaussian_square(
                output['guard_mu'], output['guard_sigma'], margin))
        return torch.stack(risks).mean(0)


class EnsembleController(guard.PredictedGuardController):
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
        first, _, _ = inputs(world, f, initial[:1])
        history = self.model.encode_history({key: value.to(self.device) for key, value in first.items()})
        details = []

        def score(plans, learned=True):
            batch, _, _ = inputs(world, f, plans)
            batch = {key: value.to(self.device) for key, value in batch.items()}
            gap = guard.nominal_gaps(batch)[1]
            nominal = (self.guard_margin + gap).clamp_min(0).square()
            predicted = self.model.radial_risk(batch, history, self.guard_margin) if learned else nominal
            radial = torch.maximum(nominal, predicted).amin(-1)[0].cpu().numpy()
            base = nominal_score(plans, f['unary'].numpy(), f['pair'].numpy(), protocol.risk_weight)
            if learned:
                details.append(dict(configurations=plans.tolist(),
                    nominal_guard_cost_m2=nominal.amin(-1)[0].cpu().tolist(),
                    learned_guard_cost_m2=predicted.amin(-1)[0].cpu().tolist(),
                    envelope_guard_cost_m2=radial.tolist(), collision_and_prior_cost=base.tolist()))
            return base + self.guard_weight * radial

        valid = [row.nonzero()[:, 0].numpy() for row in f['valid']]
        if self.nominal_slack is None:
            indices, trace = whole_configuration_search(initial, score, valid, self.budget, self.rounds)
        else:
            physical_best, trace = whole_configuration_search(initial, lambda plans: score(plans, False),
                                                               valid, self.budget, self.rounds)
            plans, nominal_cost = np.asarray(trace['configurations']), np.asarray(trace['scores'])
            feasible = nominal_cost <= trace['cost'] + self.nominal_slack + 1e-10
            cost = score(plans)
            selected = min(np.flatnonzero(feasible), key=lambda index: (cost[index], tuple(plans[index])))
            indices = plans[selected]
            trace.update(cost=float(cost[selected]), scores=cost.tolist(),
                physical_best_indices=physical_best.tolist(), nominal_costs=nominal_cost.tolist(),
                eligible_configurations=feasible.tolist(), selected_nominal_cost=float(nominal_cost[selected]),
                minimum_nominal_cost=float(nominal_cost.min()), nominal_cost_allowance=self.nominal_slack)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        trace.update(gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            source_center_indices=centre.tolist(), source_center_scored=centre.tolist() in trace['configurations'],
            initial_configurations=initial.tolist(), geometric_score_batches=details,
            ensemble_members=len(self.model.members), compute_seconds=time.perf_counter() - start)
        return force, trace


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ensemble-checkpoints', nargs='+', required=True)
    parser.add_argument('--nominal-slack', type=float)
    args, remaining = parser.parse_known_args()
    if len(args.ensemble_checkpoints) < 2 or len(set(args.ensemble_checkpoints)) != len(args.ensemble_checkpoints):
        raise ValueError('Declare at least two distinct ensemble members')
    if args.nominal_slack is not None and (not np.isfinite(args.nominal_slack) or args.nominal_slack < 0):
        raise ValueError('Nominal cost allowance must be nonnegative')
    sys.argv = [sys.argv[0], *remaining]
    primary = argparse.ArgumentParser(add_help=False)
    primary.add_argument('--checkpoint', required=True)
    checkpoint, _ = primary.parse_known_args()
    if checkpoint.checkpoint != args.ensemble_checkpoints[0]:
        raise ValueError('Primary checkpoint must be the first declared ensemble member')
    revision = torch.load(checkpoint.checkpoint, map_location='cpu', weights_only=False)['config']['revision']
    if revision == 'CITF-stopped-guard-envelope-focus-v1':
        from experiments.guard_focus import FocusConfig
        guard.GuardConfig = FocusConfig
    elif revision == 'CITF-ordered-stopped-guard-envelope-v1':
        from experiments.ordered_guard import OrderedConfig, OrderedPredictor
        guard.GuardConfig, guard.GuardPredictor = OrderedConfig, OrderedPredictor
    elif revision == 'CITF-synchronized-first-action-precise-difference-v1':
        from experiments.synchronized_delta import PreciseTeamConfig, PreciseTeamPredictor
        from experiments.synchronized_guard import nominal_team, actual_team
        guard.GuardConfig, guard.GuardPredictor = PreciseTeamConfig, PreciseTeamPredictor
        guard.nominal_gaps, guard.actual_gaps = nominal_team, actual_team
    else:
        raise ValueError('Unregistered ensemble architecture')
    import citf.evaluate as evaluation
    original_load = evaluation.load_behavior
    evaluation.load_behavior = lambda path, source, world, device: EnsemblePredictor([
        original_load(member, source, world, device) for member in args.ensemble_checkpoints]).eval()
    original_main = evaluation.main

    def evaluate():
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['method'] = 'CITF-GuardEnsemble-development'
            result['controller'].update(ensemble_checkpoints=[
                dict(path=path, sha256=file_hash(path)) for path in args.ensemble_checkpoints],
                ensemble_rule='equal-weight mixture of expected stopped radial-gap costs; all declared seeds',
                guard_combination='min_i max(nominal_i, mean_member learned_expected_i)',
                nominal_cost_allowance=args.nominal_slack)
            if revision == 'CITF-synchronized-first-action-precise-difference-v1':
                from experiments.synchronized_guard import TARGET
                from experiments.recourse_guard import CONDITION, RECOURSE
                result['controller'].update(guard_target=TARGET,
                    prediction_condition=CONDITION, recourse_controller=RECOURSE)
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                     'ensemble_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        original_main()

    evaluation.main = evaluate
    guard.PredictedGuardController = lambda *positional, **kwargs: EnsembleController(
        *positional, nominal_slack=args.nominal_slack, **kwargs)
    guard.evaluate_main()


if __name__ == '__main__':
    main()
