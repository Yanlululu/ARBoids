"""Development objective for predicted short-horizon radial guarding geometry.

The added term measures whether at least one defender stays target-side of the
attacker in the known nominal interval. It uses observable-state forecasts,
not a terminal value or future event input. Original CITF-Phys is unchanged.
"""
import argparse
import math
import sys
import time

import numpy as np
import torch

from citf.bridge import Controller, factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.search import nominal_beam, nominal_score, initialize_bank, whole_configuration_search


def guarding_cost(nominal, known, entity_valid, margin):
    radius = nominal.norm(dim=-1)
    defender_radius = radius[:, :, :-1].masked_fill(~entity_valid[:, None, :-1, None], torch.inf).amin(2)
    slack = (margin + defender_radius - radius[:, :, -1]).clamp_min(0).square()
    return (slack * known[:, None]).sum(-1) / known.sum(-1)[:, None]


class GuardController(Controller):
    def __init__(self, *args, guard_weight, guard_margin, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_weight = guard_weight
        self.guard_margin = guard_margin

    @torch.no_grad()
    def act(self, world):
        start = time.perf_counter()
        f = factorize(world, self.source)
        p = world.protocol
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), p.risk_weight, 128)
        initial, centre = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                          f['prior'].numpy(), self.bank, self.budget)
        history_cache = None
        if self.model is not None:
            first, _, _ = inputs(world, f, initial[:1])
            history_cache = self.model.encode_history({k: v.to(self.device) for k, v in first.items()})
        details = []
        def score(plans):
            b, _, _ = inputs(world, f, plans)
            b = {k: v.to(self.device) for k, v in b.items()}
            guard = guarding_cost(b['nominal'], b['known_time_valid'], b['entity_valid'], self.guard_margin)[0]
            if self.model is None:
                base = torch.as_tensor(nominal_score(plans, f['unary'].numpy(), f['pair'].numpy(), p.risk_weight), device=guard.device)
            else:
                out = self.model(b, history_cache=history_cache)
                risk = out['risk'] if self.risk_mode == 'expected' else out['det_risk']
                base = (b['unary'] + p.risk_weight * (risk * out['pair_valid'][:, None]).sum(-1))[0]
            details.append({'configurations': plans.tolist(), 'guarding_cost_m2': guard.cpu().tolist(),
                            'prior_and_pair_risk_cost': base.cpu().tolist()})
            return (base + self.guard_weight * guard).cpu().numpy()
        indices, trace = whole_configuration_search(initial, score,
            [row.nonzero()[:, 0].numpy() for row in f['valid']], self.budget, self.rounds)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        trace.update(gates=gates.tolist(), selected_indices=indices.tolist(),
                     physical_thrust=force.tolist(), initial_configurations=initial.tolist(),
                     source_center_indices=centre.tolist(),
                     source_center_scored=centre.tolist() in trace['configurations'],
                     geometric_score_batches=details,
                     guard_weight=self.guard_weight, guard_margin_m=self.guard_margin,
                     compute_seconds=time.perf_counter() - start)
        return force, trace


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--guard-weight', type=float, required=True)
    ap.add_argument('--guard-margin', type=float, default=2.)
    args, remaining = ap.parse_known_args()
    if not all(math.isfinite(x) and x >= 0 for x in (args.guard_weight, args.guard_margin)):
        raise ValueError('Finite nonnegative guarding parameters required')
    sys.argv = [sys.argv[0], *remaining]
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity
    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['method'] = 'CITF-GuardGeometry-development' if result['checkpoint_sha256'] else 'GuardGeometry-Phys-development'
        result['controller']['guard_weight'] = args.guard_weight
        result['controller']['guard_margin_m'] = args.guard_margin
        result['controller']['guard_prediction'] = 'held defender thrust plus observed constant attacker velocity, known timeout interval'
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'guard_geometry': file_hash(__file__)})
        return result
    evaluation.Controller = lambda *positional, **kwargs: GuardController(*positional,
        guard_weight=args.guard_weight, guard_margin=args.guard_margin, **kwargs)
    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    main()
