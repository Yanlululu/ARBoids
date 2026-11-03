"""Predict the expected positive-square short radial geometry directly.

A separate geometric moment head avoids deriving the control cost from a
Gaussian approximation to a possibly multimodal stopped-distance law. The
distance head remains supervised, and every label uses only the short branch.
"""
from dataclasses import dataclass
import time

import torch
from torch import nn
import torch.nn.functional as F

from citf.bridge import factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.search import nominal_beam, nominal_score, initialize_bank, whole_configuration_search
from experiments import guard_prediction as guard
from experiments import guard_envelope
from experiments import noise_aware_guard as noise


@dataclass(frozen=True)
class MomentConfig(noise.NoiseAwareConfig):
    revision: str = 'CITF-first-action-team-geometric-moment-v1'
    geometric_cost: str = 'direct conditional expectation of stopped radial positive square'


class MomentPredictor(noise.NoiseAwarePredictor):
    def __init__(self, cfg=MomentConfig()):
        super().__init__(cfg)
        self.moment_head = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.width),
                                         nn.GELU(), nn.Linear(cfg.width, 1))
        nn.init.zeros_(self.moment_head[-1].weight)
        nn.init.zeros_(self.moment_head[-1].bias)
        self._moment_features = None
        self.guard_head.register_forward_pre_hook(self._capture_moment)

    def _capture_moment(self, module, arguments):
        self._moment_features = arguments[0]

    def forward(self, batch, history_cache=None):
        result = super().forward(batch, history_cache=history_cache)
        shape = result['guard_mu'].shape
        shift = self.moment_head(self._moment_features).reshape(shape[0], shape[1], 1)
        self._moment_features = None
        root_cost = F.softplus(result['nominal_guard_mu'] + self.cfg.guard_margin + shift)
        result['guard_risk'] = root_cost.square()
        return result


distance_objective = noise.noise_aware_objective


def moment_objective(output, batch, **kwargs):
    result = distance_objective(output, batch, **kwargs)
    truth = guard.actual_gaps(batch)
    target = (truth + output['guard_margin']).clamp_min(0).square().mean(2)
    scale = max(output['guard_margin'], 2.) ** 2
    error = (output['guard_risk'] - target) / scale
    valid = batch['candidate_valid'][..., None] & batch['entity_valid'][:, None, :-1]
    mse = (error.square() * valid).sum() / valid.sum().clamp_min(1)
    result['risk'] = mse
    result['total'] = output['guard_nll_weight'] * result['nll'] + kwargs.get('risk_weight', 1.) * mse
    if kwargs.get('mode', 'full') == 'full':
        result['total'] = (result['total'] + kwargs.get('rank_weight', .5) * result['rank']
                           + output['response_delta_weight'] * result['context'])
    return result


class MomentController(guard_envelope.EnvelopeController):
    @torch.no_grad()
    def act(self, world):
        start = time.perf_counter()
        f, protocol = factorize(world, self.source), world.protocol
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), protocol.risk_weight, 128)
        initial, centre = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                          f['prior'].numpy(), self.bank, self.budget)
        history = None
        if self.model is not None:
            first, _, _ = inputs(world, f, initial[:1])
            history = self.model.encode_history({k: v.to(self.device) for k, v in first.items()})
        details = []

        def score(plans):
            batch, _, _ = inputs(world, f, plans)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            _, gap = guard.nominal_gaps(batch)
            nominal = (self.guard_margin + gap).clamp_min(0).square()
            learned = nominal if self.model is None else self.model(batch, history_cache=history)['guard_risk']
            envelope = torch.maximum(nominal, learned).amin(-1)[0]
            base = torch.as_tensor(nominal_score(plans, f['unary'].numpy(), f['pair'].numpy(),
                                                 protocol.risk_weight), device=envelope.device)
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
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'moment_guard': file_hash(__file__)})
    noise.NoiseAwareConfig, noise.NoiseAwarePredictor = MomentConfig, MomentPredictor
    noise.noise_aware_objective = moment_objective
    guard_envelope.EnvelopeController = MomentController
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        original_identity = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = original_identity(*positional, **kwargs)
            result['controller']['geometric_cost'] = MomentConfig().geometric_cost
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                     'moment_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        original_main()

    evaluation.main = evaluate
    noise.main()


if __name__ == '__main__':
    main()
