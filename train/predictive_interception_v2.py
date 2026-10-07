"""Task-time rollout with original navigation, causal target identification,
and a conservative capture margin. The original CBF remains unchanged.
"""
from dataclasses import replace

import numpy as np

from attacker_motion_observer import AttackerMotionObserver
from feedback_joint_control import (NominalEnvironment, nominal_scene, observe,
    minimum_separation, nominal_attacker_command)
from feedback_joint_fast import FastNominalEnvironment
from predictive_interception import PredictiveInterceptionController, constant_turn_displacement, wrap_angle
from source_arboids import source_mixture, actor_thrust


TEMPLATES = ('baseline', 'learned', 'direct', 'lead', 'lead_guard')


def reachable_capture_time(states, attacker, radius=5., speed=3.2):
    """Optimistic interception time; recognizes targets escaping the reachable cone."""
    relative = attacker[:2] - states[:, :2]
    velocity = attacker[3:5]
    distance = np.linalg.norm(relative, axis=-1)
    a = float(velocity @ velocity - speed*speed)
    b = 2. * (relative @ velocity - speed*radius)
    c = np.sum(relative*relative, axis=-1) - radius*radius
    times = np.full(len(states), 80.)
    if abs(a) < 1e-10:
        roots = np.divide(-c, b, out=np.full(len(states), np.inf), where=np.abs(b)>1e-10)
        times = np.where(roots >= 0., np.minimum(roots, times), times)
    else:
        discriminant = b*b - 4*a*c
        for sign in (-1., 1.):
            roots = (-b + sign*np.sqrt(np.maximum(discriminant, 0.))) / (2*a)
            times = np.where((discriminant>=0.) & (roots>=0.), np.minimum(times, roots), times)
    times = np.where(distance <= radius, 0., times)
    direction = relative + np.minimum(times, 10.)[:, None] * velocity
    turn = np.abs(wrap_angle(np.arctan2(direction[:, 1], direction[:, 0]) - states[:, 2])) / .85
    return np.minimum(80., times + .4*turn)


class TaskTimePredictiveController(PredictiveInterceptionController):
    def __init__(self, defenders, policy, block_steps=10, blend=.5, fixed_template=None,
                 prediction='feedback', capture_margin=.5, adaptive_attacker=True, departure_seconds=.15):
        if fixed_template is not None and fixed_template not in TEMPLATES:
            raise ValueError('Unknown task-time feedback template.')
        self.observer = AttackerMotionObserver()
        self.navigation = NominalEnvironment(defender_num=defenders)
        self.capture_margin = float(capture_margin)
        self.adaptive_attacker = bool(adaptive_attacker)
        super().__init__(defenders, policy, block_steps, blend,
                         None if fixed_template == 'learned' else fixed_template,
                         prediction, departure_seconds)
        self.fixed_template = fixed_template

    def reset(self, previous_thrust=None):
        super().reset(previous_thrust)
        if hasattr(self, 'observer'):
            self.observer.reset()
        self.attacker_agility = 2.25

    def guidance(self, measurement, template):
        if template == 'direct':
            return measurement.boids.copy()
        target = measurement.attacker[:2].copy()
        nearest = np.linalg.norm(measurement.defenders[:, :2] - target, axis=-1).min()
        seconds = min(2., max(0., (nearest-measurement.capture_radius)/3.2))
        goal = target + constant_turn_displacement(measurement.attacker[3:5], measurement.attacker[5], seconds)
        states = measurement.defenders
        self.navigation._Boid_navi_step(states[:, :2], states[:, 3:5], states[:, 2], goal)
        result = self.navigation.boids_actions.copy()
        if template == 'lead_guard':
            keeper = int(np.argmin(np.linalg.norm(states[:, :2], axis=-1)))
            radius = np.linalg.norm(target)
            guard = min(20., max(7., radius-8.)) * target / max(radius, 1e-9)
            force = self.navigation.boids_forces[keeper] + .5 * (guard-goal)
            result[keeper] = self.navigation.force_to_thrust(force, states[keeper, 2])
        return result

    def feedback(self, measurement, template):
        if template not in TEMPLATES:
            raise ValueError(template)
        raw = self.policy(measurement.observations)
        reference = source_mixture(raw, measurement.boids).astype(float)
        if template == 'baseline':
            action = raw
        else:
            proposal = actor_thrust(raw) if template == 'learned' else self.guidance(measurement, template)
            desired = (1.-self.blend)*reference + self.blend*proposal
            action = np.column_stack(((2.*desired-500.)/1500., np.ones(self.defenders)))
        _, force, info = self.safety.control(measurement.defenders, action, measurement.boids)
        return np.clip(force, -500., 1000.), info, reference

    def predict(self, measurement, template):
        env = nominal_scene(measurement, self.previous_thrust, self.previous_attacker_thrust,
                            self.velocity_lag, self.attacker_agility)
        env.__class__ = FastNominalEnvironment
        env.Defend_R = measurement.capture_radius - self.capture_margin
        current = measurement
        positions, attackers, commands = [current.defenders.copy()], [current.attacker.copy()], []
        nearest_pair = minimum_separation(current.defenders)
        done = 0
        held = None
        for _ in range(self.block_steps):
            force, _, _ = self.feedback(current, template)
            if self.prediction == 'held':
                held = force.copy() if held is None else held
                force = held
            commands.append(force.copy())
            observations, _, done, _ = env.step(env.thrust_to_action(force), 'RL')
            current = replace(observe(env, observations), capture_radius=measurement.capture_radius)
            positions.append(current.defenders.copy())
            attackers.append(current.attacker.copy())
            nearest_pair = min(nearest_pair, minimum_separation(current.defenders))
            if done:
                break
        elapsed = len(commands)*.2
        guard_margin = float(np.linalg.norm(current.attacker[:2])-np.linalg.norm(current.defenders[:, :2], axis=-1).min())
        eta = float(reachable_capture_time(current.defenders, current.attacker, env.Defend_R).min())
        estimated = elapsed if done == 3 else elapsed + eta + .6*max(0., 5.-guard_margin)
        if done == 4:
            estimated = 80.
        penalty = self.departure_seconds if template != 'baseline' else 0.
        score = (int(done == 2), int(done == 1), estimated + penalty)
        return dict(template=template, score=score, outcome=int(done), minimum_distance=nearest_pair,
                    positions=np.asarray(positions), attackers=np.asarray(attackers), commands=np.asarray(commands),
                    estimated_capture_time=estimated, guard_margin=guard_margin,
                    model_attacker_agility=self.attacker_agility)

    def control(self, measurement):
        if self.adaptive_attacker:
            self.attacker_agility = self.observer.update(measurement)
        planned = self.remaining == 0
        considered = 0
        if planned:
            if self.fixed_template is None:
                choices = [self.predict(measurement, t) for t in TEMPLATES]
                self.plan = min(choices, key=lambda p: p['score'])
                self.plan['baseline_score'] = choices[0]['score']
                self.template = self.plan['template']
                considered = len(choices)
            else:
                self.template = self.fixed_template
                self.plan = None
            self.remaining = self.block_steps
            self.plan_count += 1
        physical, info, _ = self.feedback(measurement, self.template)
        self.previous_attacker_thrust = nominal_attacker_command(measurement, self.attacker_agility)
        self.previous_thrust = physical.copy()
        offset = self.block_steps-self.remaining
        predicted = (self.plan['positions'][offset+1] if self.plan is not None
                     and offset+1 < len(self.plan['positions']) else None)
        self.remaining -= 1
        self.tick += 1
        info.update(planned=planned, template=self.template, remaining=self.remaining,
                    plan_index=self.plan_count, plan_offset=offset, candidate_policies=considered,
                    predicted_next_state=None if predicted is None else predicted.copy(),
                    prediction_mode=self.prediction, gate_only=False,
                    estimated_agility=self.attacker_agility, observer_updates=self.observer.updates)
        return physical, info
