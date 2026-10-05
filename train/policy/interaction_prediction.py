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
    mixture_points: int = 0
    task_prediction: bool = False
    terminal_prediction: bool = False
    terminal_capture_radius: float = 4.0
    terminal_buffer: float = 0.4

    def __post_init__(self):
        if not all(np.isfinite(getattr(self, key)) and getattr(self, key) > 0
                   for key in ('horizon', 'dt', 'distance_scale', 'speed_scale',
                               'terminal_capture_radius', 'terminal_buffer')):
            raise ValueError('Prediction scales and times must be finite and positive.')
        if (not isinstance(self.mixture_points, int) or
                self.mixture_points not in (0, 3, 5, 9)):
            raise ValueError('Use zero (legacy endpoints), three, five or nine mixture points.')
        if (self.task_prediction or self.terminal_prediction) and not self.mixture_points:
            raise ValueError('Task prediction requires a mixed-action grid.')
        if not np.isclose(self.horizon / self.dt, round(self.horizon / self.dt)):
            raise ValueError('Prediction horizon must be an integer number of substeps.')


@dataclass(frozen=True)
class CandidateMessages:
    ids: np.ndarray
    timestamps: np.ndarray
    states: np.ndarray  # [N, 6]: world x, y, yaw, vx, vy, yaw rate
    boids: np.ndarray   # [N, 2], normalized actions, not Newtons
    proposals: np.ndarray
    actor_observations: np.ndarray | None = None


def exchange_candidates(states, boids, proposals, timestamp, ids=None, actor_observations=None):
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
    if actor_observations is not None:
        actor_observations = np.asarray(actor_observations, dtype=np.float32).copy()
        if actor_observations.shape != (n, 14 + 2*(n-1)) or not np.isfinite(actor_observations).all():
            raise ValueError('Intent context must contain one finite raw Actor observation per vessel.')
        actor_observations.setflags(write=False)
    for a in (states, boids, proposals, stamps, ids):
        a.setflags(write=False)
    return CandidateMessages(ids, stamps, states, boids, proposals, actor_observations)


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
        mixed_paths = self.mixture_paths([messages]) if cfg.mixture_points else None
        paths = (mixed_paths[0][:, [0, -1]] if mixed_paths is not None else
                 self.trajectories(np.repeat(states, 2, axis=0), thrusts.reshape(-1, 2)).reshape(n, 2, -1, 2))
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
        if cfg.mixture_points:
            edges = np.concatenate((edges, self.mixture_distances([messages], mixed_paths)[0]), axis=-1)
        mask = ~np.eye(n, dtype=bool)
        edges[~mask] = 0.
        return edges, mask

    def features_batch(self, messages):
        """Vectorize independent worlds; vessels from different worlds never interact."""
        if not messages or len({message.states.shape for message in messages}) != 1:
            raise ValueError('A nonempty batch with equal defender counts is required.')
        cfg = self.config
        states = np.stack([message.states for message in messages])
        candidates = np.stack([(message.boids, message.proposals) for message in messages], axis=0)
        candidates = candidates.transpose(0, 2, 1, 3)
        batch, n, _ = states.shape
        mixed_paths = self.mixture_paths(messages) if cfg.mixture_points else None
        paths = (mixed_paths[:, :, [0, -1]] if mixed_paths is not None else
                 self.trajectories(np.repeat(states, 2, axis=1).reshape(-1, 6),
                    (750. * candidates + 250.).reshape(-1, 2)).reshape(batch, n, 2, -1, 2))
        parts = []
        for own in range(2):
            for other in range(2):
                d = np.linalg.norm(paths[:, :, own, None] - paths[:, None, :, other], axis=-1)
                closest = d.argmin(axis=-1)
                parts.extend((d.min(axis=-1) / cfg.distance_scale,
                              closest * cfg.dt / cfg.horizon,
                              (d[..., 0] - d[..., -1]) / (cfg.horizon * cfg.speed_scale)))
        position = states[:, None, :, :2] - states[:, :, None, :2]
        velocity = states[:, None, :, 3:5] - states[:, :, None, 3:5]
        c, s = np.cos(states[:, :, 2])[:, :, None], np.sin(states[:, :, 2])[:, :, None]
        for vector, scale in ((position, cfg.distance_scale), (velocity, cfg.speed_scale)):
            parts.extend(((c * vector[..., 0] + s * vector[..., 1]) / scale,
                          (-s * vector[..., 0] + c * vector[..., 1]) / scale))
        angle = states[:, None, :, 2] - states[:, :, None, 2]
        parts.extend((np.sin(angle), np.cos(angle)))
        edges = np.stack(parts, axis=-1).astype(np.float32)
        if cfg.mixture_points:
            edges = np.concatenate((edges, self.mixture_distances(messages, mixed_paths)), axis=-1)
        mask = np.broadcast_to(~np.eye(n, dtype=bool), (batch, n, n)).copy()
        edges[~mask] = 0.
        return edges, mask

    def mixture_paths(self, messages):
        """One nominal rollout per grid action; reuse its endpoints and task geometry."""
        k = self.config.mixture_points
        if not k:
            raise ValueError('Enable a mixture grid before predicting mixed actions.')
        states = np.stack([message.states for message in messages])
        boids = np.stack([message.boids for message in messages])
        proposals = np.stack([message.proposals for message in messages])
        grid = np.linspace(0., 1., k)[None, None, :, None]
        actions = (1.-grid)*boids[:, :, None] + grid*proposals[:, :, None]
        batch, n, _ = states.shape
        return self.trajectories(np.repeat(states, k, axis=1).reshape(-1, 6),
            (750.*actions+250.).reshape(-1, 2)).reshape(batch, n, k, -1, 2)

    def mixture_distances(self, messages, paths=None):
        """Fixed features of current messages, with no learned gate or future state."""
        paths = self.mixture_paths(messages) if paths is None else paths
        batch, n, k, steps, _ = paths.shape
        # [world, own vessel, peer vessel, own gate, peer gate, time]
        distance = np.linalg.norm(paths[:, :, None, :, None] - paths[:, None, :, None, :], axis=-1)
        result = distance.min(-1).reshape(batch, n, n, k*k)
        if self.config.task_prediction or self.config.terminal_prediction:
            if any(message.actor_observations is None for message in messages):
                raise ValueError('Task prediction requires exchanged current Actor observations.')
            states = np.stack([message.states for message in messages])
            obs = np.stack([message.actor_observations for message in messages])
            bearing = obs[..., 3] + states[..., 2]
            attacker = states[..., :2] + obs[..., 2, None]*np.stack((np.cos(bearing), np.sin(bearing)), -1)
            course = obs[..., 5] + states[..., 2]
            velocity = obs[..., 4, None]*np.stack((np.cos(course), np.sin(course)), -1)
            times = np.arange(steps)*self.config.dt
            target_path = attacker[:, :, None, :] + velocity[:, :, None, :]*times[None, None, :, None]
            separation = np.linalg.norm(paths-target_path[:, :, None], axis=-1)
            if self.config.task_prediction:
                metrics = np.stack((separation.min(-1), separation[..., -1]), axis=-2).reshape(batch, n, 2*k)
                result = np.concatenate((result, np.broadcast_to(metrics[:, :, None], (batch, n, n, 2*k))), axis=-1)
        if self.config.terminal_prediction:
            # Prefix clearances keep all collisions up to the end of a time
            # bin. The actor may shorten its window only after a conservative
            # nominal interception and an additional time buffer.
            prefixes = [distance[..., :int(np.ceil(fraction*(steps-1)))+1].min(-1)
                        for fraction in (.25, .5, .75)]
            result = np.concatenate((result, np.stack(prefixes, axis=-3).reshape(batch, n, n, 3*k*k)), -1)
            captured = separation < self.config.terminal_capture_radius
            capture_time = np.where(captured.any(-1), captured.argmax(-1)*self.config.dt,
                                    2.*self.config.horizon)/self.config.horizon
            result = np.concatenate((result/self.config.distance_scale,
                np.broadcast_to(capture_time[:, :, None], (batch, n, n, k))), -1)
            return result.astype(np.float32)
        return (result/self.config.distance_scale).astype(np.float32)
