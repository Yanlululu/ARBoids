"""Reuse CUDA graphs for the observable-state feedback branch calculation.

The equations, stopping masks and current-choice objective are those of
batched_feedback. Candidate batches are padded with duplicate configurations;
only actual unique configurations contribute to selection and branch counts.
"""
import torch

from citf.contracts import canonical_hash, file_hash
from experiments import batched_feedback as feedback


def observations(state, legacy_velocity):
    defenders = state.shape[1] - 1
    position, yaw = state[:, :-1, :2], state[:, :-1, 2]
    difference = position[:, :, None] - position[:, None]
    distance = difference.norm(dim=-1)
    off_diagonal = ~torch.eye(defenders, dtype=torch.bool, device=state.device)
    neighbor = (distance < 15) & off_diagonal[None]
    separation = (difference / (distance - 3.5).clamp_min(.01).square()[..., None] * neighbor[..., None]).sum(2)
    alignment = legacy_velocity[:, :-1].mean(1, keepdim=True).expand_as(position)
    cohesion = position.mean(1, keepdim=True) - position
    attraction = state[:, -1:, :2] - position
    boids = feedback.force_to_thrust(.5 * attraction + 10 * separation + .1 * alignment + .1 * cohesion, yaw)
    normalized_boids = (boids - 250) / 750
    pair = feedback.polar(-difference, yaw[:, :, None])
    # Fixed index slices avoid the device synchronization of boolean indexing.
    teammates = torch.stack([torch.cat((pair[:, i, :i], pair[:, i, i + 1:]), 1)
                             for i in range(defenders)], 1).flatten(2)
    result = torch.cat((feedback.polar(-position, yaw), feedback.polar(attraction, yaw),
        feedback.polar(legacy_velocity[:, -1:].expand_as(position), yaw),
        feedback.polar(separation, yaw), feedback.polar(alignment, yaw),
        feedback.polar(cohesion, yaw), normalized_boids, teammates), -1)
    return result, normalized_boids


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
            obs, boids = observations(state, legacy)
            learned = feedback.source_actions(network, obs)
        force = 750 * (gates * learned + (1 - gates) * boids) + 250
        thrust = torch.cat((force, feedback.attacker_thrust(state, agility)[:, None]), 1)
        next_state, legacy = feedback.integrate(state, thrust)
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
        stopped = breached | (distance <= protocol.collision_radius).any(-1) | (separation < protocol.capture_radius).any(-1)
        transitions += alive.to(torch.long)
        alive = alive & ~stopped
    return minimum, gap_maximum, transitions


@torch.no_grad()
def predict(world, source, factor, plans, nominal_agility):
    device = source.device
    plans = torch.as_tensor(plans, dtype=torch.long)
    count, defenders = plans.shape
    gates = factor['grid'][torch.arange(defenders)[None], plans].to(device)
    state = torch.as_tensor(world.state(), dtype=torch.float32, device=device)
    learned, boids = factor['learned'].to(device), factor['rules'].to(device)
    steps = world.protocol.known_steps(world.env.Current_T)
    if not str(device).startswith('cuda') or steps != world.protocol.prediction_steps:
        output = rollout(state[None].expand(count, -1, -1), gates, learned, boids,
                         source.net, steps, world.protocol, nominal_agility)
        return output[0], output[1], int(output[2].sum())
    bucket = 16 if count <= 16 else ((count + 63) // 64) * 64
    key = (defenders, bucket, steps, nominal_agility)
    if not hasattr(source, '_feedback_graphs'):
        source._feedback_graphs = {}
        source._feedback_graph_pool = torch.cuda.graph_pool_handle()
    cache = source._feedback_graphs
    if key not in cache:
        static_state = state[None].expand(bucket, -1, -1).clone()
        static_gates = gates[-1:].expand(bucket, -1, -1).clone()
        static_gates[:count].copy_(gates)
        static_learned, static_boids = learned.clone(), boids.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                rollout(static_state, static_gates, static_learned, static_boids,
                        source.net, steps, world.protocol, nominal_agility)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=source._feedback_graph_pool):
            outputs = rollout(static_state, static_gates, static_learned, static_boids,
                              source.net, steps, world.protocol, nominal_agility)
        cache[key] = (graph, static_state, static_gates, static_learned, static_boids, outputs)
    graph, static_state, static_gates, static_learned, static_boids, outputs = cache[key]
    static_state.copy_(state[None].expand_as(static_state))
    static_gates.copy_(gates[-1:].expand_as(static_gates))
    static_gates[:count].copy_(gates)
    static_learned.copy_(learned)
    static_boids.copy_(boids)
    graph.replay()
    return outputs[0][:count], outputs[1][:count], int(outputs[2][:count].sum())


def main():
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['controller'].update(feedback_execution='CUDA graphs, eager for shortened remaining horizon',
                                    prediction_padding='duplicate candidates to 16 or next multiple of 64')
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'graphed_feedback': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    feedback.predict = predict
    feedback.main()


if __name__ == '__main__':
    main()
