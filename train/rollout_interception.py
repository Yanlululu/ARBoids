"""Feedback policy improvement with an explicit CBF continuation value.

Only the first feedback block is an execution forecast. Beyond that block the
model evaluates the stated baseline continuation, not an unknown future replan.
The untouched CBF is both the physical safety filter and the terminal policy.
"""
from dataclasses import replace

import numpy as np

from feedback_joint_control import nominal_scene, observe, minimum_separation
from feedback_joint_fast import FastNominalEnvironment
from predictive_interception_v2 import TaskTimePredictiveController, reachable_capture_time


class RolloutInterceptionController(TaskTimePredictiveController):
    def __init__(self, defenders, policy, block_steps=10, blend=1., tail_steps=100,
                 prediction='feedback', fixed_template=None, adaptive_attacker=True,
                 capture_margin=.5, departure_seconds=.15):
        if tail_steps not in (40, 100, 200):
            raise ValueError('Continuation length must be declared before evaluation.')
        self.tail_steps = int(tail_steps)
        super().__init__(defenders, policy, block_steps, blend, fixed_template,
                         prediction, capture_margin, adaptive_attacker, departure_seconds)

    def predict(self, measurement, template):
        env = nominal_scene(measurement, self.previous_thrust, self.previous_attacker_thrust,
                            self.velocity_lag, self.attacker_agility)
        env.__class__ = FastNominalEnvironment
        env.Defend_R = measurement.capture_radius - self.capture_margin
        current = measurement
        positions, attackers, commands = [current.defenders.copy()], [current.attacker.copy()], []
        nearest_pair = minimum_separation(current.defenders)
        held = None
        done = prefix_done = 0
        used = 0
        for step in range(self.block_steps + self.tail_steps):
            policy = template if step < self.block_steps else 'baseline'
            force, _, _ = self.feedback(current, policy)
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
        elapsed = used * .2
        guard_margin = float(np.linalg.norm(current.attacker[:2]) -
                             np.linalg.norm(current.defenders[:, :2], axis=-1).min())
        eta = float(reachable_capture_time(current.defenders, current.attacker, env.Defend_R).min())
        estimated = elapsed if done == 3 else elapsed + eta + .6 * max(0., 5. - guard_margin)
        if done == 4:
            estimated = measurement.total_time - measurement.time
        penalty = self.departure_seconds if template != 'baseline' else 0.
        score = (int(done == 2), int(done == 1), estimated + penalty)
        return dict(template=template, score=score, outcome=int(prefix_done),
                    continuation_outcome=int(done), continuation_steps=max(0, used-self.block_steps),
                    minimum_distance=nearest_pair, positions=np.asarray(positions),
                    attackers=np.asarray(attackers), commands=np.asarray(commands),
                    estimated_capture_time=estimated, guard_margin=guard_margin,
                    model_attacker_agility=self.attacker_agility)
