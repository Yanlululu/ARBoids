"""Longer feedback guarding with the established short collision interval."""
import torch

from citf.contracts import canonical_hash, file_hash
from experiments import batched_feedback as feedback
from experiments import graphed_feedback as graphed
from experiments.long_geometry import factorize_long


def rollout(initial, gates, initial_learned, initial_boids, network, steps, protocol, agility):
    state = initial
    count, defenders = gates.shape[:2]
    indices = torch.triu_indices(defenders, defenders, 1, device=state.device)
    minimum = torch.full((count, indices.shape[1]), torch.inf, device=state.device)
    gap_maximum = torch.full((count, defenders), -torch.inf, device=state.device)
    alive = torch.ones(count, dtype=torch.bool, device=state.device)
    transitions = torch.zeros(count, dtype=torch.long, device=state.device)
    for tick in range(steps):
        if tick == 0:
            learned, boids = initial_learned[None], initial_boids[None]
        else:
            obs, boids = graphed.observations(state, legacy)
            learned = feedback.source_actions(network, obs)
        force = 750 * (gates * learned + (1 - gates) * boids) + 250
        thrust = torch.cat((force, feedback.attacker_thrust(state, agility)[:, None]), 1)
        next_state, legacy = feedback.integrate(state, thrust)
        state = torch.where(alive[:, None, None], next_state, state)
        positions = state[..., :2]
        distance = (positions[:, indices[0]] - positions[:, indices[1]]).norm(dim=-1)
        if tick < 16:
            minimum = torch.minimum(minimum, distance.masked_fill(~alive[:, None], torch.inf))
        radius = positions.norm(dim=-1)
        gaps = radius[:, :-1] - radius[:, -1:]
        gap_maximum = torch.maximum(gap_maximum, gaps.masked_fill(~alive[:, None], -torch.inf))
        separation = (positions[:, :-1] - positions[:, -1:]).norm(dim=-1)
        breached = radius[:, -1] < protocol.target_radius
        if protocol.radial_early_failure:
            breached = breached | (gaps > 0).all(-1)
        stopped = breached | (distance <= protocol.collision_radius).any(-1) | (separation < protocol.capture_radius).any(-1)
        transitions += alive.to(torch.long)
        alive = alive & ~stopped
    return minimum, gap_maximum, transitions


def main():
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(collision_prediction_steps=16, proposal_collision_prediction_steps=16,
                                    radial_prediction_steps=result['protocol']['prediction_steps'])
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'graphed_feedback_long': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    feedback.factorize = factorize_long
    graphed.rollout = rollout
    graphed.main()


if __name__ == '__main__':
    main()
