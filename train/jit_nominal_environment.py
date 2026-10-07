"""Compiled source-model integration, Boids and observations for prediction.

This is a numerical backend, not a changed physical or task model. The real
environment and real-time safety filter continue to use their original code.
"""
import study_runtime
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from array_rollout import ArrayNominalEnvironment
from feedback_joint_fast import model


@lru_cache(maxsize=1)
def compiled_step():
    dynamics = model()
    boat = dynamics.model
    mass_inverse = jnp.asarray(dynamics.mass_inverse)
    damping = jnp.asarray(np.asarray(boat.D))

    def acceleration(state, thrust):
        c, s = jnp.cos(state[:, 2]), jnp.sin(state[:, 2])
        u = c * state[:, 3] + s * state[:, 4]
        v = -s * state[:, 3] + c * state[:, 4]
        r = state[:, 5]
        velocity = jnp.stack((u, v, r), axis=-1)
        matrix = jnp.broadcast_to(damping, (len(state), 3, 3))
        matrix = matrix.at[:, 0, 1].add(-boat.m * r)
        matrix = matrix.at[:, 1, 0].add(boat.m * r)
        matrix = matrix.at[:, 0, 2].add(boat.yDotV * v + boat.yDotR * r)
        matrix = matrix.at[:, 1, 2].add(-boat.xDotU * u)
        matrix = matrix.at[:, 2, 0].add(-boat.yDotV * v - boat.yDotR * r)
        matrix = matrix.at[:, 2, 1].add(boat.xDotU * u)
        matrix = matrix.at[:, 0, 0].add(-boat.xUU * jnp.abs(u))
        matrix = matrix.at[:, 1, 1].add(-boat.yVV * jnp.abs(v) - boat.yRV * jnp.abs(r))
        matrix = matrix.at[:, 1, 2].add(-boat.yVR * jnp.abs(v) - boat.yRR * jnp.abs(r))
        matrix = matrix.at[:, 2, 1].add(-boat.nVV * jnp.abs(v) - boat.nRV * jnp.abs(r))
        matrix = matrix.at[:, 2, 2].add(-boat.nVR * jnp.abs(v) - boat.nRR * jnp.abs(r))
        tau = jnp.stack((thrust.sum(-1), jnp.zeros(len(state)),
                         (thrust[:, 0] - thrust[:, 1]) * boat.width / 2.), axis=-1)
        body = (tau - jnp.einsum('bij,bj->bi', matrix, velocity)) @ mass_inverse.T
        return jnp.stack((c * body[:, 0] - s * body[:, 1],
                          s * body[:, 0] + c * body[:, 1], body[:, 2]), axis=-1)

    def forces_to_thrust(force, angle, agility):
        force = force / jnp.linalg.norm(force, axis=-1, keepdims=True)
        c, s = jnp.cos(angle), jnp.sin(angle)
        forward = c * force[..., 0] + s * force[..., 1]
        side = -s * force[..., 0] + c * force[..., 1]
        yaw = .75 * jnp.arctan2(side, forward)
        return jnp.clip(jnp.stack((forward + yaw, forward - yaw), axis=-1) * (1000. * agility),
                        -500. * agility, 1000. * agility)

    def distance_angle(vector, angle):
        phi = jnp.arctan2(vector[..., 1], vector[..., 0]) - angle
        phi = jnp.where(phi > jnp.pi, phi - 2 * jnp.pi,
                        jnp.where(phi < -jnp.pi, phi + 2 * jnp.pi, phi))
        return jnp.stack((jnp.linalg.norm(vector, axis=-1), phi), axis=-1)

    def step(state, defender_thrust, agility):
        n = len(state) - 1
        def repel(i, carry):
            total, radius = carry
            delta = state[0, :2] - state[i + 1, :2]
            length = jnp.linalg.norm(delta)
            distance = jnp.maximum(length - 8., .1)
            contribution = 3000. * (1. / distance - 1. / radius) / distance**2 * delta / length
            inside = distance < radius
            return total + jnp.where(inside, contribution, 0.), jnp.where(inside, distance, radius)
        repulsion, _ = jax.lax.fori_loop(0, n, repel, (jnp.zeros(2, dtype=state.dtype), jnp.asarray(50.)))
        attacker = forces_to_thrust(-.1 * state[0, :2] + repulsion, state[0, 2], agility)
        forces = jnp.concatenate((attacker[None], jnp.clip(defender_thrust, -500., 1000.)), axis=0)

        def integrate(i, carry):
            values, measured = carry
            measured = values[:, 3:]
            values = values.at[:, :2].add(.05 * measured[:, :2])
            values = values.at[:, 2].set((values[:, 2] + .05 * measured[:, 2]) % (2 * jnp.pi))
            values = values.at[:, 3:].add(.05 * acceleration(values, forces))
            return values, measured
        state, measured = jax.lax.fori_loop(0, 4, integrate, (state, state[:, 3:]))
        positions, velocities, headings = state[1:, :2], measured[1:, :2], state[1:, 2]
        delta = positions[:, None, :] - positions[None, :, :]
        distance = jnp.linalg.norm(delta, axis=-1)
        off_diagonal = ~jnp.eye(n, dtype=bool)
        neighbors = (distance < 15.) & off_diagonal
        separation = jnp.sum(jnp.where(neighbors[..., None],
            delta / jnp.maximum(distance - 3.5, .01)[..., None]**2, 0.), axis=1)
        alignment = jnp.broadcast_to(velocities.mean(axis=0), positions.shape)
        cohesion = positions.mean(axis=0) - positions
        boids_forces = .5 * (state[0, :2] - positions) + 10. * separation + .1 * alignment + .1 * cohesion
        boids = forces_to_thrust(boids_forces, headings, 1.)
        boids_states = jnp.concatenate((separation, alignment, cohesion), axis=-1)
        # Static integer indices preserve the source's teammate ordering.
        teammate_ids = np.array([[j for j in range(n) if j != i] for i in range(n)])
        teammates = positions[teammate_ids] - positions[:, None, :]
        teammate_obs = distance_angle(teammates, headings[:, None]).reshape(n, 2 * (n - 1))
        observations = jnp.concatenate((
            distance_angle(-positions, headings), distance_angle(state[0, :2] - positions, headings),
            distance_angle(jnp.broadcast_to(measured[0, :2], positions.shape), headings),
            distance_angle(separation, headings), distance_angle(alignment, headings),
            distance_angle(cohesion, headings), (2. * boids - 500.) / 1500., teammate_obs), axis=-1)
        return state, measured, forces, boids, boids_forces, boids_states, observations

    return jax.jit(step)


class JitNominalEnvironment(ArrayNominalEnvironment):
    def step(self, rl_action, controller='RL', att_action=None):
        if controller != 'RL' or att_action is not None:
            raise ValueError('Only physical defender controls are implemented for prediction.')
        boats = [self.attacker, *self.defender_list]
        state = np.asarray([[b.x, b.y, b.theta, *b.velocity_r] for b in boats])
        result = compiled_step()(state, self.action_to_thrust(rl_action), self.attacker.agility)
        state, measured, forces, boids, boids_forces, boids_states, observations = map(np.asarray, result)
        for i, b in enumerate(boats):
            b.x, b.y, b.theta = state[i, :3]
            b.pos[:] = state[i, :2]
            b.velocity_r, b.velocity = state[i, 3:].copy(), measured[i].copy()
            b.vel = b.velocity[:2]
            b.left_thrust, b.right_thrust = forces[i]
        self.boids_actions, self.boids_forces, self.boids_states = boids, boids_forces, boids_states
        self.def_att_dists = observations[:, 2].copy()
        self.def_def_dists = observations[:, 14::2].copy()
        self.att_action = self.thrust_to_action(forces[0], self.attacker.agility)
        self.Current_T += self.Action_T
        return observations, np.zeros(self.defender_num), self._isTerminate(), np.zeros(8)
