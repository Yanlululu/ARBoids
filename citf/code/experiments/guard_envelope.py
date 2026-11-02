"""Retain nominal radial protection while adding learned interaction risk.

For each possible guarding vessel, take the larger of its held-thrust and
learned stopped-interval positive-square costs. Then choose the least exposed
guard. This changes the current configuration objective only; predictions
remain short physical distances and never include terminal outcome values.
"""
import time

import torch

from citf.bridge import factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.search import nominal_beam, nominal_score, initialize_bank, whole_configuration_search
from experiments import guard_prediction as guard


class EnvelopeController(guard.PredictedGuardController):
    @torch.no_grad()
    def act(self, world):
        start = time.perf_counter()
        f, p = factorize(world, self.source), world.protocol
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), p.risk_weight, 128)
        initial, centre = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                          f['prior'].numpy(), self.bank, self.budget)
        history = None
        if self.model is not None:
            first, _, _ = inputs(world, f, initial[:1])
            history = self.model.encode_history({k: v.to(self.device) for k, v in first.items()})
        details = []

        def score(plans):
            b, _, _ = inputs(world, f, plans)
            b = {k: v.to(self.device) for k, v in b.items()}
            _, gap = guard.nominal_gaps(b)
            nominal = (self.guard_margin + gap).clamp_min(0).square()
            if self.model is None:
                learned = nominal
            else:
                out = self.model(b, history_cache=history)
                learned = guard.positive_gaussian_square(out['guard_mu'], out['guard_sigma'], self.guard_margin)
            envelope = torch.maximum(nominal, learned).amin(-1)[0]
            base = torch.as_tensor(nominal_score(plans, f['unary'].numpy(), f['pair'].numpy(),
                                                 p.risk_weight), device=envelope.device)
            details.append(dict(configurations=plans.tolist(),
                nominal_guard_cost_m2=nominal.amin(-1)[0].cpu().tolist(),
                learned_guard_cost_m2=learned.amin(-1)[0].cpu().tolist(),
                envelope_guard_cost_m2=envelope.cpu().tolist(),
                collision_and_prior_cost=base.cpu().tolist()))
            return (base + self.guard_weight * envelope).cpu().numpy()

        indices, trace = whole_configuration_search(initial, score,
            [row.nonzero()[:, 0].numpy() for row in f['valid']], self.budget, self.rounds)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        trace.update(gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            source_center_indices=centre.tolist(), source_center_scored=centre.tolist() in trace['configurations'],
            initial_configurations=initial.tolist(), geometric_score_batches=details,
            compute_seconds=time.perf_counter() - start)
        return force, trace


def main():
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        if result['controller'].get('collision_source', 'nominal') != 'nominal':
            raise ValueError('The radial envelope retains nominal collision protection')
        result['controller']['guard_combination'] = 'min_i max(nominal_i, learned_expected_i)'
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'guard_envelope': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    guard.PredictedGuardController = EnvelopeController
    guard.evaluate_main()


if __name__ == '__main__':
    main()
