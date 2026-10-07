"""Batched numerical execution for task rollouts, separate from frozen studies.

The original physical environment and CBFpy remain the evaluation references.
Batching the unchanged Actor changes floating-point reduction order, so this
backend is validated with numerical tolerances, not claimed bit-identical.
"""
import study_runtime
from dataclasses import replace

import numpy as np
import torch
import jax.numpy as jnp

from compiled_source_policy import CompiledSourcePolicy
from feedback_joint_control import nominal_scene, observe, minimum_separation
from feedback_joint_fast import FastNominalEnvironment, model
from predictive_interception import constant_turn_displacement
from predictive_interception_v2 import reachable_capture_time
from rollout_interception import RolloutInterceptionController
from source_arboids import source_mixture, actor_thrust


class BatchedSourcePolicy(CompiledSourcePolicy):
    def __call__(self, observations):
        with torch.no_grad():
            return self.program(torch.tensor(observations, dtype=torch.float)).cpu().numpy()


def forces_to_thrust(forces, headings, maximum=1000., minimum=-500.):
    forces = forces / np.linalg.norm(forces, axis=-1, keepdims=True)
    c, s = np.cos(headings), np.sin(headings)
    forward = c * forces[:, 0] + s * forces[:, 1]
    side = -s * forces[:, 0] + c * forces[:, 1]
    angular = .75 * np.arctan2(side, forward)
    return np.clip(maximum * np.column_stack((forward + angular, forward - angular)), minimum, maximum)


def boids_arrays(positions, velocities, headings, goal):
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    neighbor = (distances < 15.) & ~np.eye(len(positions), dtype=bool)
    divisor = np.maximum(distances - 3.5, .01) ** 2
    separation = np.sum(np.where(neighbor[..., None], differences / divisor[..., None], 0.), axis=1)
    alignment = np.broadcast_to(velocities.mean(axis=0), positions.shape)
    cohesion = positions.mean(axis=0) - positions
    forces = .5 * (goal - positions) + 10. * separation + .1 * alignment + .1 * cohesion
    states = np.column_stack((separation, alignment, cohesion))
    return forces_to_thrust(forces, headings), forces, states


def distance_angle(vectors, headings):
    angle = np.arctan2(vectors[..., 1], vectors[..., 0]) - headings
    angle = np.where(angle > np.pi, angle - 2 * np.pi,
                     np.where(angle < -np.pi, angle + 2 * np.pi, angle))
    return np.stack((np.linalg.norm(vectors, axis=-1), angle), axis=-1)


class ArrayNominalEnvironment(FastNominalEnvironment):
    def _Boid_navi_step(self, positions, velocities, phis, goal, robot='def'):
        if robot != 'def':
            raise ValueError('This adapter implements defender Boids only.')
        self.boids_actions, self.boids_forces, self.boids_states = boids_arrays(
            positions, velocities, phis, goal)

    def _get_obs(self):
        if self.LearningSide != 'Def' or not self.boid_state:
            raise ValueError('This adapter implements the frozen defender observation contract.')
        positions = np.asarray([b.pos for b in self.defender_list])
        headings = np.asarray([b.theta for b in self.defender_list])
        pair_vectors = positions[None, :, :] - positions[:, None, :]
        off_diagonal = ~np.eye(self.defender_num, dtype=bool)
        teammates = distance_angle(pair_vectors, headings[:, None])[off_diagonal].reshape(self.defender_num, -1)
        attacker = distance_angle(self.attacker.pos - positions, headings)
        self.def_att_dists = attacker[:, 0].copy()
        self.def_def_dists = np.linalg.norm(pair_vectors, axis=-1)[off_diagonal].reshape(self.defender_num, -1)
        observations = np.column_stack((
            distance_angle(-positions, headings), attacker,
            distance_angle(np.broadcast_to(self.attacker.vel, positions.shape), headings),
            distance_angle(self.boids_states[:, 0:2], headings),
            distance_angle(self.boids_states[:, 2:4], headings),
            distance_angle(self.boids_states[:, 4:6], headings),
            self.thrust_to_action(self.boids_actions), teammates))
        return observations, np.zeros(8)

    def _get_rewards(self, done):
        # Prediction does not use source rewards; termination is still original.
        return np.zeros(self.defender_num)


class ArrayRolloutController(RolloutInterceptionController):
    def __init__(self, defenders, policy, *, prediction_policy=None, tail_policy='baseline', tail_steps=100,
                 failure_cost='legacy', **settings):
        if tail_policy not in ('baseline', 'candidate'):
            raise ValueError('Declare the continuation policy.')
        self.prediction_policy = policy if prediction_policy is None else prediction_policy
        self.tail_policy = tail_policy
        if failure_cost not in ('legacy', 'delay'):
            raise ValueError('Unknown failed-rollout ranking rule.')
        self.failure_cost = failure_cost
        if tail_steps not in (0, 40, 100, 200):
            raise ValueError('Declare zero, 8, 20 or 40 seconds of continuation.')
        super().__init__(defenders, policy, tail_steps=tail_steps if tail_steps else 40, **settings)
        self.tail_steps = int(tail_steps)

    def _forecast_force(self, measurement, template):
        raw = self.prediction_policy(measurement.observations)
        reference = source_mixture(raw, measurement.boids).astype(float)
        if template == 'baseline':
            action = raw
        else:
            proposal = actor_thrust(raw) if template == 'learned' else self.guidance(measurement, template)
            desired = (1. - self.blend) * reference + self.blend * proposal
            action = np.column_stack(((2. * desired - 500.) / 1500., np.ones(self.defenders)))
        nominal = source_mixture(action, measurement.boids).astype(float)
        filtered = self.safety.filter.safety_filter(jnp.asarray(measurement.defenders.reshape(-1)),
                                                    jnp.asarray(nominal.reshape(-1) / 1000.))
        # Diagnostics are recorded for actual controls. Counterfactual scoring
        # needs the same QP solution but not a second evaluation of its residual.
        return (1000. * np.clip(np.asarray(filtered), -.5, 1.)).reshape(self.defenders, 2)

    def guidance(self, measurement, template):
        if template == 'direct':
            return measurement.boids.copy()
        target = measurement.attacker[:2]
        states = measurement.defenders
        nearest = np.linalg.norm(states[:, :2] - target, axis=-1).min()
        seconds = min(2., max(0., (nearest - measurement.capture_radius) / 3.2))
        goal = target + constant_turn_displacement(measurement.attacker[3:5], measurement.attacker[5], seconds)
        actions, forces, _ = boids_arrays(states[:, :2], states[:, 3:5], states[:, 2], goal)
        if template == 'lead_guard':
            keeper = int(np.argmin(np.linalg.norm(states[:, :2], axis=-1)))
            radius = np.linalg.norm(target)
            guard = min(20., max(7., radius - 8.)) * target / max(radius, 1e-9)
            force = forces[keeper] + .5 * (guard - goal)
            actions[keeper] = forces_to_thrust(force[None], states[keeper, 2:3])[0]
        return actions

    def predict(self, measurement, template):
        env = nominal_scene(measurement, self.previous_thrust, self.previous_attacker_thrust,
                            self.velocity_lag, self.attacker_agility)
        env.__class__ = ArrayNominalEnvironment
        env.Defend_R = measurement.capture_radius - self.capture_margin
        current = measurement
        positions, attackers, commands = [current.defenders.copy()], [current.attacker.copy()], []
        nearest_pair = minimum_separation(current.defenders)
        held = None
        done = prefix_done = used = 0
        execution_policy = self.policy
        self.policy = self.prediction_policy
        try:
            for step in range(self.block_steps + self.tail_steps):
                choice = template if step < self.block_steps or self.tail_policy == 'candidate' else 'baseline'
                force = self._forecast_force(current, choice)
                if self.prediction == 'held' and step < self.block_steps:
                    held = force.copy() if held is None else held
                    force = held
                observations, _, done, _ = env.step(env.thrust_to_action(force), 'RL')
                current = replace(observe(env, observations), capture_radius=measurement.capture_radius)
                nearest_pair = min(nearest_pair, minimum_separation(current.defenders))
                if step < self.block_steps:
                    commands.append(force.copy())
                    positions.append(current.defenders.copy())
                    attackers.append(current.attacker.copy())
                    prefix_done = done
                used = step + 1
                if done:
                    break
        finally:
            self.policy = execution_policy
        elapsed = used * .2
        guard_margin = float(np.linalg.norm(current.attacker[:2]) - np.linalg.norm(current.defenders[:, :2], axis=-1).min())
        eta = float(reachable_capture_time(current.defenders, current.attacker, env.Defend_R).min())
        estimated = elapsed if done == 3 else elapsed + eta + .6 * max(0., 5. - guard_margin)
        if done == 4:
            estimated = measurement.total_time - measurement.time
        if self.failure_cost == 'delay' and done in (1, 2):
            # All doomed candidates still need a meaningful recovery ordering.
            # Earlier loss must not look like earlier capture.
            estimated = 2. * (measurement.total_time - measurement.time) - elapsed
        penalty = self.departure_seconds if template != 'baseline' else 0.
        score = (int(done == 2), int(done == 1), estimated + penalty)
        return dict(template=template, score=score, outcome=int(prefix_done), continuation_outcome=int(done),
                    continuation_steps=max(0, used - self.block_steps), minimum_distance=nearest_pair,
                    positions=np.asarray(positions), attackers=np.asarray(attackers), commands=np.asarray(commands),
                    estimated_capture_time=estimated, guard_margin=guard_margin,
                    model_attacker_agility=self.attacker_agility)
