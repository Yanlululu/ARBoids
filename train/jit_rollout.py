"""Task rollouts with the separately validated compiled numerical backend."""
from dataclasses import replace

import numpy as np

from array_rollout import ArrayRolloutController
from feedback_joint_control import nominal_scene, observe, minimum_separation
from jit_nominal_environment import JitNominalEnvironment
from predictive_interception_v2 import reachable_capture_time


class JitRolloutController(ArrayRolloutController):
    prediction_environment_class = JitNominalEnvironment

    def predict(self, measurement, template):
        env = nominal_scene(measurement, self.previous_thrust, self.previous_attacker_thrust,
                            self.velocity_lag, self.attacker_agility)
        env.__class__ = self.prediction_environment_class
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
            estimated = 2. * (measurement.total_time - measurement.time) - elapsed
        penalty = self.departure_seconds if template != 'baseline' else 0.
        score = (int(done == 2), int(done == 1), estimated + penalty)
        return dict(template=template, score=score, outcome=int(prefix_done), continuation_outcome=int(done),
                    continuation_steps=max(0, used - self.block_steps), minimum_distance=nearest_pair,
                    positions=np.asarray(positions), attackers=np.asarray(attackers), commands=np.asarray(commands),
                    estimated_capture_time=estimated, guard_margin=guard_margin,
                    model_attacker_agility=self.attacker_agility)
