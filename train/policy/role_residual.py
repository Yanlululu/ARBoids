"""Structured residual candidates around the frozen joint interception prior."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from interaction_rollout import CandidateRolePolicy


def public_target(packet):
    state, obs = np.asarray(packet['motion']), np.asarray(packet['obs'])
    bearing = state[:, 2] + obs[:, 3]
    position = (state[:, :2] + obs[:, 2, None] * np.column_stack(
        (np.cos(bearing), np.sin(bearing)))).mean(0)
    direction = state[:, 2] + obs[:, 5]
    velocity = (obs[:, 4, None] * np.column_stack(
        (np.cos(direction), np.sin(direction)))).mean(0)
    return position, velocity


class RoleResidualPolicy(CandidateRolePolicy):
    def __init__(self, count=1, gain=1., lead=0., distance_limit=np.inf):
        super().__init__()
        self.count, self.gain = count, gain
        self.lead, self.distance_limit = lead, distance_limit

    def candidate_controls(self, packet):
        candidates, cost = self.candidates(packet)
        rows, columns = linear_sum_assignment(cost)
        prior = np.empty((len(cost), 3), dtype=np.float32)
        prior[rows] = candidates[rows, columns]
        pursuit, eta = self.pursuit_controls(packet)
        return prior, pursuit, eta

    def pursuit_controls(self, packet):
        state = np.asarray(packet['motion'])
        target, velocity = public_target(packet)
        delta = target + self.lead * velocity - state[:, :2]
        distance = np.linalg.norm(delta, axis=-1)
        error = (np.arctan2(delta[:, 1], delta[:, 0]) - state[:, 2] + np.pi) % (2*np.pi) - np.pi
        speed = np.cos(state[:, 2])*state[:, 3] + np.sin(state[:, 2])*state[:, 4]
        desired = 3.2 * np.minimum(distance/5., 1.) * np.maximum(np.cos(error), 0.)
        forward = .5 * (100*desired + 150*desired**2 + 400*(desired-speed))
        turn = 900*error - 700*state[:, 5]
        thrust = np.clip(np.column_stack((forward+turn, forward-turn)), -500., 1000.)
        pursuit = np.column_stack(((thrust-250.)/750., np.ones(len(state)))).astype(np.float32)
        return pursuit, distance/3.2 + .6*np.abs(error)

    def compose(self, packet, coefficients):
        prior, pursuit, _ = self.candidate_controls(packet)
        coefficients = np.asarray(coefficients)[:, None]
        return (prior + coefficients*(pursuit-prior)).astype(np.float32)

    def choose_action(self, packet, deterministic=True):
        prior, pursuit, eta = self.candidate_controls(packet)
        coefficients = np.zeros(len(prior))
        indices = np.argsort(eta)[:self.count]
        distance = np.asarray(packet['obs'])[:, 2]
        coefficients[indices] = self.gain * (distance[indices] < self.distance_limit)
        return (prior + coefficients[:, None]*(pursuit-prior)).astype(np.float32), 0.


RESIDUAL_SETTINGS = {
    'residual_one': dict(count=1),
    'residual_two': dict(count=2),
    'residual_half': dict(count=1, gain=.5),
    'residual_near': dict(count=1, distance_limit=20.),
    'residual_two_near': dict(count=2, distance_limit=20.),
    'residual_lead': dict(count=1, lead=2.),
    'residual_lead_near': dict(count=1, lead=2., distance_limit=20.),
}
