"""Synchronous candidate messages and fixed, zero-current 3-DOF prediction."""
from dataclasses import dataclass, asdict
import numpy as np

from envs.modules import WAMV


@dataclass
class PredictionConfig:
    horizon: float = 2.0
    dt: float = 0.05
    distance_scale: float = 5.0
    speed_scale: float = 5.0

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('Prediction scales and times must be finite and positive.')
        if not np.isclose(self.horizon / self.dt, round(self.horizon / self.dt)):
            raise ValueError('Prediction horizon must be an integer number of substeps.')


@dataclass(frozen=True)
class CandidateMessages:
    ids: np.ndarray
    timestamps: np.ndarray
    states: np.ndarray  # [N, 6]: world x, y, yaw, vx, vy, yaw rate
    boids: np.ndarray   # [N, 2], normalized actions, not Newtons
    proposals: np.ndarray


def exchange_candidates(states, boids, proposals, timestamp, ids=None):
    """One reliable, all-to-all exchange, with no environment advancement."""
    states = np.asarray(states, dtype=np.float64).copy()
    boids = np.asarray(boids, dtype=np.float64).copy()
    proposals = np.asarray(proposals, dtype=np.float64).copy()
    n = len(states)
    stamps = np.broadcast_to(np.asarray(timestamp, dtype=float), (n,)).copy()
    ids = np.arange(n) if ids is None else np.asarray(ids).copy()
    if states.shape != (n, 6) or boids.shape != (n, 2) or proposals.shape != (n, 2):
        raise ValueError('Malformed candidate message shapes.')
    if ids.shape != (n,) or len(np.unique(ids)) != n or n == 0:
        raise ValueError('Every candidate must have a unique vessel ID.')
    if not all(np.isfinite(a).all() for a in (states, boids, proposals, stamps)):
        raise ValueError('Candidate messages must be finite.')
    if np.any(np.abs(boids) > 1.000001) or np.any(np.abs(proposals) > 1.000001):
        raise ValueError('Candidates must use normalized action units.')
    if not np.allclose(stamps, stamps[0], rtol=0., atol=1e-8):
        raise ValueError('This implementation requires synchronous candidate messages.')
    for a in (states, boids, proposals, stamps, ids):
        a.setflags(write=False)
    return CandidateMessages(ids, stamps, states, boids, proposals)


class InteractionPredictor:
    edge_dim = 18  # 4 * (minimum distance, time, approach trend) + 6 geometry terms

    def __init__(self, config=None):
        self.config = config or PredictionConfig()
        self.model = WAMV()
        if not np.isclose(self.config.dt, self.model.dt):
            raise ValueError('Use the training model integration step (0.05 seconds).')
        self.mass_inverse = np.linalg.inv(np.asarray(self.model.M_RB + self.model.M_A))

    def trajectories(self, states, thrusts):
        """Vectorized WAMV integration, holding each candidate in Newtons.

        Matches WAMV.step with zero current, including pose-before-acceleration
        ordering. Message ground velocity initializes a zero-current nominal
        state; this does not reveal future disturbances or execute candidates.
        """
        state = np.asarray(states, dtype=float).copy()
        thrusts = np.asarray(thrusts, dtype=float)
        if state.ndim != 2 or state.shape[1] != 6 or thrusts.shape != (len(state), 2):
            raise ValueError('Expected states [batch,6], thrusts [batch,2].')
        m, dt = self.model, self.config.dt
        thrusts = np.clip(thrusts, m.min_thrust, m.max_thrust)
        count = int(round(self.config.horizon / dt))
        path = np.empty((len(state), count + 1, 2))
        path[:, 0] = state[:, :2]
        tau = np.column_stack((thrusts.sum(-1), np.zeros(len(state)),
                               (thrusts[:, 0] - thrusts[:, 1]) * m.width / 2.))
        for k in range(count):
            state[:, :2] += state[:, 3:5] * dt
            state[:, 2] = (state[:, 2] + state[:, 5] * dt) % (2 * np.pi)
            c, s = np.cos(state[:, 2]), np.sin(state[:, 2])
            u = c * state[:, 3] + s * state[:, 4]
            v = -s * state[:, 3] + c * state[:, 4]
            r = state[:, 5].copy()
            velocity = np.column_stack((u, v, r))
            matrix = np.broadcast_to(np.asarray(m.D), (len(state), 3, 3)).copy()
            matrix[:, 0, 1] += -m.m * r
            matrix[:, 1, 0] += m.m * r
            matrix[:, 0, 2] += m.yDotV * v + m.yDotR * r
            matrix[:, 1, 2] += -m.xDotU * u
            matrix[:, 2, 0] += -m.yDotV * v - m.yDotR * r
            matrix[:, 2, 1] += m.xDotU * u
            matrix[:, 0, 0] -= m.xUU * np.abs(u)
            matrix[:, 1, 1] -= m.yVV * np.abs(v) + m.yRV * np.abs(r)
            matrix[:, 1, 2] -= m.yVR * np.abs(v) + m.yRR * np.abs(r)
            matrix[:, 2, 1] -= m.nVV * np.abs(v) + m.nRV * np.abs(r)
            matrix[:, 2, 2] -= m.nVR * np.abs(v) + m.nRR * np.abs(r)
            force = tau - np.einsum('bij,bj->bi', matrix, velocity)
            velocity += (force @ self.mass_inverse.T) * dt
            state[:, 3] = c * velocity[:, 0] - s * velocity[:, 1]
            state[:, 4] = s * velocity[:, 0] + c * velocity[:, 1]
            state[:, 5] = velocity[:, 2]
            path[:, k + 1] = state[:, :2]
        if not np.isfinite(path).all():
            raise FloatingPointError('Nominal trajectories became non-finite.')
        return path

    def features(self, messages):
        cfg, states = self.config, messages.states
        n = len(states)
        candidates = np.stack((messages.boids, messages.proposals), axis=1)
        # Same affine conversion as TADEnv.action_to_thrust for defenders.
        thrusts = 750. * candidates + 250.
        paths = self.trajectories(np.repeat(states, 2, axis=0),
                                  thrusts.reshape(-1, 2)).reshape(n, 2, -1, 2)
        parts = []
        for own in range(2):
            for other in range(2):  # BB, BL, LB, LL, in this order
                d = np.linalg.norm(paths[:, own, None] - paths[None, :, other], axis=-1)
                closest = d.argmin(axis=-1)
                parts.extend((d.min(axis=-1) / cfg.distance_scale,
                              closest * cfg.dt / cfg.horizon,
                              (d[..., 0] - d[..., -1]) / (cfg.horizon * cfg.speed_scale)))
        position = states[None, :, :2] - states[:, None, :2]
        velocity = states[None, :, 3:5] - states[:, None, 3:5]
        c, s = np.cos(states[:, 2])[:, None], np.sin(states[:, 2])[:, None]
        for vector, scale in ((position, cfg.distance_scale), (velocity, cfg.speed_scale)):
            parts.extend(((c * vector[..., 0] + s * vector[..., 1]) / scale,
                          (-s * vector[..., 0] + c * vector[..., 1]) / scale))
        angle = states[None, :, 2] - states[:, None, 2]
        parts.extend((np.sin(angle), np.cos(angle)))
        edges = np.stack(parts, axis=-1).astype(np.float32)
        mask = ~np.eye(n, dtype=bool)
        edges[~mask] = 0.
        return edges, mask
