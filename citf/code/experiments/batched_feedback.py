"""Batched observable-state feedback prediction for current fusion choices.

Each candidate holds its fusion coefficients while source control, Boids and
the nominal APF attacker react to their evolving branch state. Only measured
initial motion is used. Future current and APF perturbations are set to zero;
the declared nominal attacker agility is shared across evaluation tasks.
This is a new development controller, not the held-thrust CITF-Phys identity.
"""
import argparse
import math
import sys
import time
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from citf.bridge import Controller, factorize
from citf.contracts import canonical_hash, file_hash
from citf.search import initialize_bank, nominal_beam, whole_configuration_search


def integrate(state: Tensor, thrust: Tensor) -> Tuple[Tensor, Tensor]:
    position, yaw, velocity = state[..., :2], state[..., 2], state[..., 3:6]
    legacy_velocity = velocity[..., :2]
    for _ in range(4):
        # The source feedback observations use velocity from just before the
        # last acceleration update. Retain it alongside updated measured state.
        legacy_velocity = velocity[..., :2]
        position = position + .05 * velocity[..., :2]
        yaw = torch.remainder(yaw + .05 * velocity[..., 2], 2 * math.pi)
        c, s = yaw.cos(), yaw.sin()
        u = c * velocity[..., 0] + s * velocity[..., 1]
        v = -s * velocity[..., 0] + c * velocity[..., 1]
        r = velocity[..., 2]
        du = (400 * r * v - (100 + 150 * u.abs()) * u + thrust.sum(-1)) / 380
        dv = (-380 * r * u - (100 + 150 * v.abs()) * v) / 400
        dr = (-20 * u * v - (980 + 950 * r.abs()) * r + 1.25 * (thrust[..., 0] - thrust[..., 1])) / 1430
        u, v, r = u + .05 * du, v + .05 * dv, r + .05 * dr
        velocity = torch.stack((c * u - s * v, s * u + c * v, r), -1)
    return torch.cat((position, yaw[..., None], velocity), -1), legacy_velocity


def polar(vector: Tensor, yaw: Tensor) -> Tensor:
    distance = vector.norm(p=2, dim=-1)
    angle = torch.atan2(vector[..., 1], vector[..., 0]) - yaw
    angle = torch.where(angle > math.pi, angle - 2 * math.pi, angle)
    angle = torch.where(angle < -math.pi, angle + 2 * math.pi, angle)
    return torch.stack((distance, angle), -1)


def force_to_thrust(force: Tensor, yaw: Tensor, agility: float = 1.) -> Tensor:
    direction = force / force.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    c, s = yaw.cos(), yaw.sin()
    forward = c * direction[..., 0] + s * direction[..., 1]
    lateral = -s * direction[..., 0] + c * direction[..., 1]
    angle = torch.atan2(lateral, forward)
    return (1000 * agility * torch.stack((forward + .75 * angle, forward - .75 * angle), -1)).clamp(-500 * agility, 1000 * agility)


def source_observations(state: Tensor, legacy_velocity: Tensor) -> Tuple[Tensor, Tensor]:
    defenders = state.shape[1] - 1
    position, yaw = state[:, :-1, :2], state[:, :-1, 2]
    difference = position[:, :, None] - position[:, None]
    distance = difference.norm(p=2, dim=-1)
    off_diagonal = ~torch.eye(defenders, dtype=torch.bool, device=state.device)
    neighbor = (distance < 15) & off_diagonal[None]
    separation = (difference / (distance - 3.5).clamp_min(.01).square()[..., None] * neighbor[..., None]).sum(2)
    alignment = legacy_velocity[:, :-1].mean(1, keepdim=True).expand_as(position)
    cohesion = position.mean(1, keepdim=True) - position
    attraction = state[:, -1:, :2] - position
    boids = force_to_thrust(.5 * attraction + 10 * separation + .1 * alignment + .1 * cohesion, yaw)
    normalized_boids = (boids - 250) / 750
    teammate_polar = polar(-difference, yaw[:, :, None])
    teammates = teammate_polar[:, off_diagonal].reshape(state.shape[0], defenders, 2 * (defenders - 1))
    observations = torch.cat((
        polar(-position, yaw), polar(attraction, yaw),
        polar(legacy_velocity[:, -1:].expand_as(position), yaw),
        polar(separation, yaw), polar(alignment, yaw), polar(cohesion, yaw),
        normalized_boids, teammates), -1)
    return observations, normalized_boids


def attacker_thrust(state: Tensor, agility: float) -> Tensor:
    position, yaw = state[:, -1, :2], state[:, -1, 2]
    force = -.1 * position
    influence = torch.full_like(yaw, 50.)
    # Preserve the archived APF's ordered nearest-obstacle update. Its
    # influence radius changes after each closer obstacle contributes.
    for index in range(state.shape[1] - 1):
        delta = position - state[:, index, :2]
        norm = delta.norm(p=2, dim=-1).clamp_min(1e-12)
        distance = (norm - 8.).clamp_min(.1)
        active = distance < influence
        magnitude = 3000 * (distance.reciprocal() - influence.reciprocal()) / distance.square()
        force = force + torch.where(active[:, None], magnitude[:, None] * delta / norm[:, None], torch.zeros_like(delta))
        influence = torch.where(active, distance, influence)
    return force_to_thrust(force, yaw, agility)


def source_actions(network, observations):
    """Only the deterministic SAC action is needed after holding the gates."""
    shape = observations.shape
    x = observations.reshape(-1, shape[-1])
    activation = network.activation
    first = activation(network.f1(x[:, :6]))
    second = activation(network.f2(x[:, 6:14]))
    teammates = activation(network.t1(x[:, 14:].reshape(len(x), -1, 2))).mean(1)
    hidden = activation(network.l1(torch.cat((first, second, teammates), -1)))
    hidden = activation(network.l2(hidden))
    return torch.tanh(network.mean_layer(hidden)).reshape(shape[0], shape[1], 2)


@torch.no_grad()
def predict(world, source, factor, plans, nominal_agility):
    device = source.device
    plans = torch.as_tensor(plans, dtype=torch.long)
    count, defenders = plans.shape
    gates = factor['grid'][torch.arange(defenders)[None], plans].to(device)
    state = torch.as_tensor(world.state(), dtype=torch.float32, device=device)[None].expand(count, -1, -1).clone()
    initial_learned = factor['learned'].to(device)
    initial_boids = factor['rules'].to(device)
    indices = torch.triu_indices(defenders, defenders, 1, device=device)
    minimum = torch.full((count, indices.shape[1]), torch.inf, device=device)
    gap_maximum = torch.full((count, defenders), -torch.inf, device=device)
    alive = torch.ones(count, dtype=torch.bool, device=device)
    transitions = torch.zeros((), dtype=torch.long, device=device)
    protocol = world.protocol
    for tick in range(protocol.known_steps(world.env.Current_T)):
        if tick == 0:
            learned, boids = initial_learned[None], initial_boids[None]
        else:
            observations, boids = source_observations(state, legacy)
            learned = source_actions(source.net, observations)
        force = 750 * (gates * learned + (1 - gates) * boids) + 250
        thrust = torch.cat((force, attacker_thrust(state, nominal_agility)[:, None]), 1)
        next_state, legacy = integrate(state, thrust)
        state = torch.where(alive[:, None, None], next_state, state)
        positions = state[..., :2]
        distance = (positions[:, indices[0]] - positions[:, indices[1]]).norm(dim=-1)
        minimum = torch.minimum(minimum, distance.masked_fill(~alive[:, None], torch.inf))
        radius = positions.norm(dim=-1)
        gaps = radius[:, :-1] - radius[:, -1:]
        gap_maximum = torch.maximum(gap_maximum, gaps.masked_fill(~alive[:, None], -torch.inf))
        separation = (positions[:, :-1] - positions[:, -1:]).norm(dim=-1)
        breached = radius[:, -1] < protocol.target_radius
        if protocol.radial_early_failure:
            breached = breached | (gaps > 0).all(-1)
        stopped = breached | (distance < protocol.collision_radius).any(-1) | (separation < protocol.capture_radius).any(-1)
        transitions += alive.sum()
        alive = alive & ~stopped
    return minimum, gap_maximum, int(transitions)


class BatchedFeedbackController(Controller):
    def __init__(self, *args, nominal_agility, guard_weight, guard_margin, **kwargs):
        super().__init__(*args, **kwargs)
        self.nominal_agility, self.guard_weight, self.guard_margin = nominal_agility, guard_weight, guard_margin
        if self.model is not None:
            raise ValueError('This physical feedback reference has no learned residual predictor')

    @torch.no_grad()
    def act(self, world):
        start = time.perf_counter()
        factor, protocol = factorize(world, self.source), world.protocol
        unary, pair = factor['unary'].numpy(), factor['pair'].numpy()
        ranked, _ = nominal_beam(unary, pair, protocol.risk_weight, 128)
        initial, center = initialize_bank(ranked, factor['grid'].numpy(), factor['valid'].numpy(), factor['prior'].numpy(), self.bank, self.budget)
        transitions = 0

        def score(plans):
            nonlocal transitions
            distance, gap, steps = predict(world, self.source, factor, plans, self.nominal_agility)
            transitions += steps
            collision = (protocol.risk_clearance - distance).clamp_min(0).square().sum(-1)
            guard = (gap + self.guard_margin).clamp_min(0).square().amin(-1)
            prior = unary[np.arange(len(center))[None], plans].sum(-1)
            return prior + (protocol.risk_weight * collision + self.guard_weight * guard).cpu().numpy()

        indices, trace = whole_configuration_search(initial, score,
            [row.nonzero()[:, 0].numpy() for row in factor['valid']], self.budget, self.rounds)
        gates = factor['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * factor['learned'] + (1 - gates) * factor['rules']) + 250).numpy()
        trace.update(gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            initial_configurations=initial.tolist(), source_center_indices=center.tolist(),
            source_center_scored=center.tolist() in trace['configurations'],
            nominal_branch_transitions=transitions, compute_seconds=time.perf_counter() - start)
        return force, trace


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--nominal-agility', type=float, default=2.25)
    parser.add_argument('--guard-weight', type=float, default=0.)
    parser.add_argument('--guard-margin', type=float, default=2.)
    args, remaining = parser.parse_known_args()
    if not all(math.isfinite(x) and x >= 0 for x in vars(args).values()) or args.nominal_agility <= 0:
        raise ValueError('Invalid feedback geometry parameters')
    sys.argv = [sys.argv[0], *remaining]
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        if result['opponent'].get('kind') != 'apf':
            raise ValueError('This nominal response model is declared for APF opponents')
        result['method'] = 'BatchedObservableFeedback-development'
        result['controller'].update(vars(args))
        result['controller'].update(
            nominal_model='source feedback and ordered APF; fixed gates; stopped branch',
            future_current='zero', future_apf_perturbation='zero',
            initial_velocity='observable updated velocity; private current discarded',
            nominal_precision='float32', nominal_integration_dt=.05,
            opponent_agility_input='declared constant, not task-specific hidden state')
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'batched_feedback': file_hash(__file__)})
        return result

    evaluation.Controller = lambda *positional, **kwargs: BatchedFeedbackController(*positional, **vars(args), **kwargs)
    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    main()
