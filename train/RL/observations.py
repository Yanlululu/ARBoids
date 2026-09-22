"""Versioned physical features; no simulator or ROS dependency."""
import numpy as np
import torch

FEATURE_VERSION = 1
LOCAL_DIM = 18
EDGE_DIM = 8
MESSAGE_DIM = 5
NODE_DIM = 10
GLOBAL_DIM = 9


def body_frame(vectors, headings):
    c, s = np.cos(headings), np.sin(headings)
    return np.stack((vectors[..., 0] * c + vectors[..., 1] * s,
                     -vectors[..., 0] * s + vectors[..., 1] * c), axis=-1)


def build_frame(positions, velocities, headings, yaw_rates, attacker_position,
                attacker_velocity, boids_states, boids_thrust, remaining_time,
                *, relative_velocities=None, attacker_relative_velocity=None,
                attacker_heading=0.0, agility=2.0, motion_features=True):
    """Build one synchronous team frame. Positions use target-centered axes.

    Actors read only local/edges/boids/masks. Nodes/global contain centralized
    training information and never enter the deployed actor.
    """
    p, v, h = (np.asarray(x, dtype=np.float64) for x in (positions, velocities, headings))
    n = len(p)
    if n < 1:
        raise ValueError("A team must contain at least one defender")
    att = np.asarray(attacker_position)
    av = np.asarray(attacker_velocity)
    yaw = np.asarray(yaw_rates)
    own_v = body_frame(v, h) / 5.0
    own_yaw = yaw[:, None]
    delta_p = p[None, :, :] - p[:, None, :]
    delta_v = v[None, :, :] - v[:, None, :]
    delta_h = h[None, :] - h[:, None]
    heading_relation = np.stack((np.sin(delta_h), np.cos(delta_h)), -1)
    velocity_relation = body_frame(delta_v, h[:, None]) / 5.0
    if not motion_features:
        own_v, own_yaw = np.zeros_like(own_v), np.zeros_like(own_yaw)
        velocity_relation = np.zeros_like(velocity_relation)
        heading_relation = np.zeros_like(heading_relation)
    directions = att - p
    directions /= np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-6)
    dot = directions @ directions.T
    cross = directions[:, 0, None] * directions[None, :, 1] - directions[:, 1, None] * directions[None, :, 0]
    edges = np.concatenate((body_frame(delta_p, h[:, None]) / 60.0,
                            velocity_relation, heading_relation,
                            dot[..., None], cross[..., None]), axis=-1)
    # Original separation/alignment/cohesion vectors, retained without learning their weights.
    boid_vectors = np.asarray(boids_states).reshape(n, 3, 2)
    boid_features = np.tanh(body_frame(boid_vectors, h[:, None]) / np.array([10., 5., 60.])[None, :, None])
    local = np.concatenate((body_frame(-p, h) / 60., body_frame(att - p, h) / 60.,
                            body_frame(np.broadcast_to(av[:2], p.shape), h) / 5.,
                            own_v, own_yaw, np.full((n, 1), remaining_time),
                            boid_features.reshape(n, 6),
                            (np.asarray(boids_thrust) - 250.) / 750.), axis=-1)
    vr = np.column_stack((v, yaw)) if relative_velocities is None else np.asarray(relative_velocities)
    avr = np.r_[av[:2], 0.] if attacker_relative_velocity is None else np.asarray(attacker_relative_velocity)
    nodes = np.concatenate((p / 60., vr / np.array([5., 5., 1.]),
                            np.sin(h)[:, None], np.cos(h)[:, None],
                            (att - p) / 60., np.ones((n, 1))), axis=-1)
    global_state = np.r_[att / 60., avr / np.array([5., 5., 1.]),
                         np.sin(attacker_heading), np.cos(attacker_heading), agility / 3., remaining_time]
    frame = dict(local=local, edges=edges, boids=np.asarray(boids_thrust),
                 nodes=nodes, global_state=global_state,
                 mask=np.ones(n, dtype=bool), neighbors=~np.eye(n, dtype=bool))
    for key, value in frame.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite observation: {key}")
        if value.dtype != bool:
            frame[key] = value.astype(np.float32)
    return frame


def tensor_frame(frame, device="cpu"):
    return {key: torch.as_tensor(value, device=device).unsqueeze(0)
            for key, value in frame.items()}


def collate_frames(frames, device="cpu"):
    """Pad variable teams; node identity is preserved along both edge axes."""
    max_n = max(len(f["mask"]) for f in frames)
    output = {}
    for key in frames[0]:
        exemplar = np.asarray(frames[0][key])
        shape = list(exemplar.shape)
        if key != "global_state":
            shape[0] = max_n
        if key in ("edges", "neighbors"):
            shape[1] = max_n
        array = np.zeros((len(frames), *shape), dtype=exemplar.dtype)
        for i, frame in enumerate(frames):
            value = np.asarray(frame[key])
            array[(i, *(slice(0, d) for d in value.shape))] = value
        output[key] = torch.as_tensor(array, device=device)
    return output


def select_frame(frame, index):
    return {key: value[index] for key, value in frame.items()}
