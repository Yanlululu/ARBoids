"""Short radial geometry with an observed, decaying attacker acceleration.

The physical reference uses only measured velocity changes in the current
history. Defender collision forecasts and the one-action candidate impulse
remain unchanged. The learning target is the same stopped team radial gap.
"""
from dataclasses import dataclass

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team
from experiments import synchronized_delta as precise


REFERENCE = 'last three observed velocity increments; acceleration clipped at 2m/s2 and exponentially decayed with 1s time constant'
original_nominal_team = team.nominal_team
OriginalTeamPredictor = team.SynchronizedPredictor


def reference_batch(batch, dt=.2):
    history = batch['history'][:, -1]
    valid = batch['history_valid'][:, -1]
    increments = (history[:, 1:, 2:4] - history[:, :-1, 2:4]) * (5 / dt)
    observed = valid[:, 1:] & valid[:, :-1]
    increments, observed = increments[:, -3:], observed[:, -3:]
    acceleration = (increments * observed[..., None]).sum(1) / observed.sum(1, keepdim=True).clamp_min(1)
    acceleration = acceleration * (2 / acceleration.norm(dim=-1, keepdim=True).clamp_min(2))
    nominal = batch['nominal'].clone()
    time = torch.arange(1, nominal.shape[-2] + 1, device=nominal.device,
                        dtype=nominal.dtype) * dt
    integrated = time - (1 - (-time).exp())
    enemy = (batch['positions'][:, -1, None]
             + history[:, -1, 2:4, None].transpose(-1, -2) * 5 * time[None, :, None]
             + acceleration[:, None] * integrated[None, :, None])
    nominal[:, :, -1] = enemy[:, None]
    return dict(batch, nominal=nominal)


def nominal_team(batch):
    return original_nominal_team(reference_batch(batch))


@dataclass(frozen=True)
class HistoryReferenceConfig(precise.PreciseTeamConfig):
    revision: str = 'CITF-first-action-observed-acceleration-reference-v1'
    attacker_reference: str = REFERENCE


class HistoryReferencePredictor(precise.PreciseTeamPredictor):
    def forward(self, batch, history_cache=None):
        nominal = batch['nominal']
        velocity = batch['history'][:, None, :, -1, 2:4] * 5
        ticks = torch.arange(1, nominal.shape[-2] + 1, device=nominal.device,
                             dtype=nominal.dtype)
        ballistic = (batch['positions'][:, None, :, None]
                     + velocity[..., None, :] * (ticks * self.cfg.dt)[None, None, None, :, None])
        first_effect = nominal[..., :1, :] - ballistic[..., :1, :]
        impulse = ballistic + first_effect * (2 * ticks - 1)[None, None, None, :, None]
        adjusted = reference_batch(dict(batch, nominal=impulse), self.cfg.dt)
        result = OriginalTeamPredictor.forward(self, adjusted, history_cache=history_cache)
        result.update(nominal_guard_mu=nominal_team(batch)[1],
                      response_delta_weight=self.cfg.response_delta_weight,
                      response_delta_scale_m=self.cfg.response_delta_scale_m)
        return result


def main():
    team.nominal_team = nominal_team
    precise.PreciseTeamConfig = HistoryReferenceConfig
    precise.PreciseTeamPredictor = HistoryReferencePredictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'history_reference_guard': file_hash(__file__)})
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller']['attacker_reference'] = REFERENCE
            result['runtime_code'] = canonical_hash({
                'core': result['runtime_code'], 'history_reference_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        original_main()

    evaluation.main = evaluate
    precise.main()


if __name__ == '__main__':
    main()
