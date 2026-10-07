"""Isolated additions to the untouched ARBoids release for the contribution study.

No imports of gate_followup, the improved TADEnv, MAPPO or its predictor occur.
Both gate ablations use identical candidates, costs, observations and timing.
Only the risk calculation and simultaneous versus joint selection differ.
"""
from functools import lru_cache
import numpy as np

from source_arboids import numerical_source, actor_thrust, source_mixture


METHODS = ('original', 'reactive_independent', 'reactive_joint',
           'predictive_independent', 'predictive_joint', 'cbf')


class NominalModel:
    """Vectorized zero-current realization of the author's WAMV Euler integrator."""
    def __init__(self):
        self.model = numerical_source().WAMV()
        self.mass_inverse = np.linalg.inv(self.model.M_RB + self.model.M_A).A

    def acceleration(self, state, thrust):
        m = self.model
        c, s = np.cos(state[:, 2]), np.sin(state[:, 2])
        u = c * state[:, 3] + s * state[:, 4]
        v = -s * state[:, 3] + c * state[:, 4]
        r = state[:, 5]
        velocity = np.column_stack((u, v, r))
        matrix = np.broadcast_to(np.asarray(m.D), (len(state), 3, 3)).copy()
        matrix[:, 0, 1] -= m.m * r
        matrix[:, 1, 0] += m.m * r
        matrix[:, 0, 2] += m.yDotV*v + m.yDotR*r
        matrix[:, 1, 2] -= m.xDotU*u
        matrix[:, 2, 0] -= m.yDotV*v + m.yDotR*r
        matrix[:, 2, 1] += m.xDotU*u
        matrix[:, 0, 0] -= m.xUU*np.abs(u)
        matrix[:, 1, 1] -= m.yVV*np.abs(v) + m.yRV*np.abs(r)
        matrix[:, 1, 2] -= m.yVR*np.abs(v) + m.yRR*np.abs(r)
        matrix[:, 2, 1] -= m.nVV*np.abs(v) + m.nRV*np.abs(r)
        matrix[:, 2, 2] -= m.nVR*np.abs(v) + m.nRR*np.abs(r)
        tau = np.column_stack((thrust.sum(-1), np.zeros(len(state)),
                               (thrust[:, 0]-thrust[:, 1])*m.width/2.))
        body = (tau - np.einsum('bij,bj->bi', matrix, velocity)) @ self.mass_inverse.T
        return np.column_stack((c*body[:, 0]-s*body[:, 1],
                                s*body[:, 0]+c*body[:, 1], body[:, 2]))

    def trajectories(self, states, thrusts, horizon=2.):
        dt = self.model.dt
        if horizon <= 0 or not np.isclose(horizon/dt, round(horizon/dt)):
            raise ValueError('Horizon must be a positive multiple of the source integration step.')
        state = np.asarray(states, dtype=float).copy()
        thrusts = np.clip(thrusts, self.model.min_thrust, self.model.max_thrust)
        count = round(horizon/dt)
        paths = np.empty((len(state), count+1, 2))
        paths[:, 0] = state[:, :2]
        for step in range(count):
            state[:, :2] += state[:, 3:5]*dt
            state[:, 2] = (state[:, 2]+state[:, 5]*dt) % (2*np.pi)
            state[:, 3:] += self.acceleration(state, thrusts)*dt
            paths[:, step+1] = state[:, :2]
        if not np.isfinite(paths).all():
            raise FloatingPointError('Non-finite source-model trajectory.')
        return paths


@lru_cache(maxsize=16)
def joint_indices(n, k):
    if n < 2 or n > 7:
        raise ValueError('The exhaustive study supports two through seven defenders.')
    return np.indices((k,)*n, dtype=np.uint8).reshape(n, -1).T


class GateController:
    def __init__(self, method='predictive_joint', horizon=2., safe_distance=7.,
                 change_penalty=.02, reactive_gain=1.):
        if method not in METHODS[:-1]:
            raise ValueError(method)
        self.method, self.horizon = method, horizon
        self.safe_distance, self.change_penalty = safe_distance, change_penalty
        self.reactive_gain = reactive_gain
        self.model = None if method == 'original' else NominalModel()

    def candidates(self, action, boids):
        n = len(action)
        theta = np.column_stack((action[:, 2], np.tile([0., .25, .5, .75, 1.], (n, 1))))
        theta = theta.astype(action.dtype)
        learned = actor_thrust(action)
        thrusts = np.empty((n, 6, 2), dtype=learned.dtype)
        for i in range(n):
            for k in range(6):
                thrusts[i, k] = theta[i, k]*learned[i] + (1-theta[i, k])*boids[i]
        return theta, thrusts

    def pair_scores(self, states, thrusts):
        n, k, _ = thrusts.shape
        pairs = np.array(np.triu_indices(n, 1)).T
        if self.method.startswith('predictive'):
            path = self.model.trajectories(np.repeat(states, k, axis=0),
                                           thrusts.reshape(-1, 2), self.horizon)
            path = path.reshape(n, k, -1, 2)
            scores = []
            for i, j in pairs:
                distance = np.linalg.norm(path[i, :, None, 1:]-path[j, None, :, 1:], axis=-1).min(-1)
                scores.append(np.maximum(0., 1.-distance/self.safe_distance)**2)
            return pairs, np.asarray(scores)
        acceleration = self.model.acceleration(np.repeat(states, k, axis=0),
                                               thrusts.reshape(-1, 2)).reshape(n, k, 3)
        scores = []
        radius2 = self.safe_distance**2
        gain = self.reactive_gain
        for i, j in pairs:
            p, v = states[i, :2]-states[j, :2], states[i, 3:5]-states[j, 3:5]
            h = (p@p-radius2)/(2*radius2)
            hdot = p@v/radius2
            da = acceleration[i, :, None, :2]-acceleration[j, None, :, :2]
            hddot = (v@v+np.einsum('...j,j->...', da, p))/radius2
            # Instantaneous relative-degree-two barrier, with no trajectory rollout.
            margin = hddot+2*gain*hdot+gain*gain*h
            scores.append(np.maximum(0., -margin/(gain*gain))**2)
        return pairs, np.asarray(scores)

    def select(self, states, action, boids):
        if self.method == 'original':
            return action[:, 2].copy(), {'warning': False, 'candidate_count': 1}
        theta, thrusts = self.candidates(action, boids)
        pairs, risks = self.pair_scores(states, thrusts)
        n, k, _ = thrusts.shape
        deviations = np.mean(((thrusts-thrusts[:, :1])/1500.)**2, axis=-1)
        warning = bool(np.max(risks[:, 0, 0]) > 0.)
        if not warning:
            return theta[:, 0], {'warning': False, 'candidate_count': n*k,
                                 'reference_risk': 0., 'selected_risk': 0.}

        def evaluate(indices):
            worst = np.zeros(len(indices))
            for p, (i, j) in enumerate(pairs):
                worst = np.maximum(worst, risks[p, indices[:, i], indices[:, j]])
            cost = deviations[np.arange(n)[None, :], indices].mean(-1)
            return worst + self.change_penalty*cost, worst

        if self.method.endswith('_joint'):
            options = joint_indices(n, k)
            scores, raw = evaluate(options)
            choice = options[int(np.argmin(scores))]
            count = len(options)
        else:
            choice = np.zeros(n, dtype=int)
            for i in range(n):
                unilateral = np.zeros((k, n), dtype=int)
                unilateral[:, i] = np.arange(k)
                scores, _ = evaluate(unilateral)
                choice[i] = int(np.argmin(scores))
            count = n*k
        _, chosen_risk = evaluate(choice[None])
        return theta[np.arange(n), choice], dict(warning=True, candidate_count=count,
            reference_risk=float(np.max(risks[:, 0, 0])), selected_risk=float(chosen_risk[0]))

    def control(self, states, action, boids):
        theta, info = self.select(states, action, boids)
        modified = action.copy()
        modified[:, 2] = theta
        return modified, source_mixture(modified, boids), info
