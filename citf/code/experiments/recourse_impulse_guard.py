"""One-action recourse prediction with a one-interval control impulse.

The learned trajectory reference propagates the candidate's first-interval
position effect as a velocity impulse. It does not amplify that effect as
though the queried thrust were held for the entire horizon. The established
held-thrust radial envelope and collision control are retained online.
"""
import argparse
from dataclasses import dataclass
import sys

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import response_delta_guard as delta
from experiments.recourse_guard import CONDITION, RECOURSE


@dataclass(frozen=True)
class RecourseImpulseConfig(delta.DeltaConfig):
    revision: str = 'CITF-first-action-impulse-response-guard-v1'
    prediction_condition: str = CONDITION
    learned_nominal_reference: str = 'current velocity plus candidate first-interval acceleration impulse'


class RecourseImpulsePredictor(delta.DeltaPredictor):
    def forward(self, batch, history_cache=None):
        nominal = batch['nominal']
        positions = batch['positions'][:, None, :, None]
        velocity = batch['history'][:, None, :, -1, 2:4] * 5
        ticks = torch.arange(1, nominal.shape[-2] + 1, dtype=nominal.dtype, device=nominal.device)
        time = ticks * self.cfg.dt
        ballistic = positions + velocity[..., None, :] * time[None, None, None, :, None]
        first_effect = nominal[..., :1, :] - ballistic[..., :1, :]
        impulse = ballistic + first_effect * (2 * ticks - 1)[None, None, None, :, None]
        output = super().forward(dict(batch, nominal=impulse), history_cache=history_cache)
        # The actual online nominal envelope remains the strong physical
        # controller's original held-thrust forecast, as does the loss.
        output['nominal_guard_mu'] = guard.nominal_gaps(batch)[1]
        return output


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--guard-combination', choices=['envelope', 'learned'], default='envelope')
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.guard_combination == 'learned':
        from experiments import guard_envelope
        guard_envelope.main = guard.evaluate_main
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'recourse_impulse_guard': file_hash(__file__),
        'prediction_condition': CONDITION, 'recourse_controller': RECOURSE})
    delta.DeltaConfig, delta.DeltaPredictor = RecourseImpulseConfig, RecourseImpulsePredictor
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(prediction_condition=CONDITION, recourse_controller=RECOURSE,
            learned_nominal_reference=RecourseImpulseConfig().learned_nominal_reference,
            guard_combination=args.guard_combination)
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'recourse_impulse_guard': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    delta.main()


if __name__ == '__main__':
    main()
