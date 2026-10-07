"""Mission-level feedback rollout over causal interception and guard policies.

The selected policy parameters are committed for the prediction block. Each
policy recomputes thrust and roles from current measurements at every tick.
The upstream CBF, source dynamics, Actor, and source task rules are unchanged.
"""
import numpy as np

from feedback_joint_control import nominal_attacker_command
from feedback_joint_fast import FastFeedbackJointController
from source_arboids import source_mixture


MISSION_TEMPLATES = ('baseline', 'direct', 'lead', 'direct_guard', 'lead_guard')


def wrap_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def constant_turn_displacement(velocity, yaw_rate, seconds):
    """Public measured velocity and yaw rate only; no true future target input."""
    angle = float(np.clip(yaw_rate, -1.2, 1.2)) * seconds
    along = seconds * np.sinc(angle / np.pi)
    side = seconds * .5 * angle * np.sinc(angle / (2 * np.pi))**2
    return along * velocity + side * np.array([-velocity[1], velocity[0]])


def guidance_thrust(states, goals, maximum_speed=3.2):
    """Damped surge/yaw reference using the same published vessel constants."""
    displacement = goals - states[:, :2]
    distance = np.linalg.norm(displacement, axis=-1)
    bearing = np.arctan2(displacement[:, 1], displacement[:, 0])
    error = wrap_angle(bearing - states[:, 2])
    desired_speed = np.minimum(maximum_speed, .9 * distance) * np.maximum(0., np.cos(error))**2
    desired_yaw_rate = np.clip(1.2 * error, -.85, .85)
    forward = np.cos(states[:, 2]) * states[:, 3] + np.sin(states[:, 2]) * states[:, 4]
    yaw_rate = states[:, 5]
    force = 100. * forward + 150. * np.abs(forward) * forward + 380. * 1.2 * (desired_speed - forward)
    moment = 980. * yaw_rate + 950. * np.abs(yaw_rate) * yaw_rate + 1430. * 1.1 * (desired_yaw_rate - yaw_rate)
    differential = moment / 1.25
    return np.clip(np.column_stack(((force + differential) / 2., (force - differential) / 2.)), -500., 1000.)


class PredictiveInterceptionController(FastFeedbackJointController):
    def __init__(self, defenders, policy, block_steps=20, blend=1., fixed_template=None,
                 prediction='feedback', departure_seconds=.15, velocity_lag=.05):
        if block_steps not in (5, 10, 20, 30) or not 0. < blend <= 1.:
            raise ValueError('Invalid bounded mission-controller configuration.')
        if fixed_template is not None and fixed_template not in MISSION_TEMPLATES:
            raise ValueError('Unknown fixed feedback policy.')
        self.blend = float(blend)
        self.fixed_template = fixed_template
        self.departure_seconds = float(departure_seconds)
        super().__init__(defenders, policy, 10, prediction=prediction,
                         velocity_lag=velocity_lag, force_planning=True)
        self.block_steps = block_steps

    def guidance(self, measurement, template):
        target = measurement.attacker[:2]
        targets = np.broadcast_to(target, (self.defenders, 2)).copy()
        if template.startswith('lead'):
            nearest = np.linalg.norm(measurement.defenders[:, :2]-target, axis=-1).min()
            seconds = min(2., max(0., (nearest - measurement.capture_radius) / 3.2))
            targets += constant_turn_displacement(measurement.attacker[3:5], measurement.attacker[5], seconds)
        if template.endswith('_guard'):
            keeper = int(np.argmin(np.linalg.norm(measurement.defenders[:, :2], axis=-1)))
            radial = np.linalg.norm(target)
            guard_radius = min(20., max(7., radial - 8.))
            targets[keeper] = guard_radius * target / max(radial, 1e-9)
        return guidance_thrust(measurement.defenders, targets)

    def feedback(self, measurement, template):
        if template not in MISSION_TEMPLATES:
            raise ValueError(template)
        raw = self.policy(measurement.observations)
        reference = source_mixture(raw, measurement.boids).astype(float)
        if template == 'baseline':
            action = raw
        else:
            desired = (1. - self.blend) * reference + self.blend * self.guidance(measurement, template)
            action = np.column_stack(((2. * desired - 500.) / 1500., np.ones(self.defenders)))
        _, physical, info = self.safety.control(measurement.defenders, action, measurement.boids)
        physical = np.clip(physical, -500., 1000.)
        if not np.isfinite(physical).all():
            raise FloatingPointError('Mission controller produced non-finite thrust.')
        return physical, info, reference

    def predict(self, measurement, template):
        prediction = super().predict(measurement, template)
        final = prediction['positions'][-1]
        attacker = prediction['attackers'][-1]
        elapsed = .2 * len(prediction['commands'])
        relative = attacker[:2] - final[:, :2]
        distance = np.linalg.norm(relative, axis=-1)
        error = np.abs(wrap_angle(np.arctan2(relative[:, 1], relative[:, 0]) - final[:, 2]))
        remaining = np.maximum(0., distance - measurement.capture_radius) / 3.2 + .6 * error / .85
        guard_cost = .6 * max(0., 5. - prediction['guard_margin'])
        mission_time = elapsed if prediction['outcome'] == 3 else elapsed + float(remaining.min()) + guard_cost
        if prediction['outcome'] == 4:
            mission_time = 80.
        penalty = self.departure_seconds if template != 'baseline' else 0.
        prediction['original_score'] = prediction['score']
        prediction['estimated_capture_time'] = mission_time
        prediction['score'] = (int(prediction['outcome'] == 2), int(prediction['outcome'] == 1), mission_time + penalty)
        return prediction

    def control(self, measurement):
        planned = self.remaining == 0
        considered = 0
        if planned:
            if self.fixed_template is None:
                candidates = [self.predict(measurement, t) for t in MISSION_TEMPLATES]
                self.plan = min(candidates, key=lambda p: p['score'])
                self.plan['baseline_score'] = candidates[0]['score']
                self.template = self.plan['template']
                considered = len(candidates)
            else:
                self.template = self.fixed_template
                self.plan = None
            self.remaining = self.block_steps
            self.plan_count += 1
        physical, info, _ = self.feedback(measurement, self.template)
        self.previous_attacker_thrust = nominal_attacker_command(measurement, self.attacker_agility)
        self.previous_thrust = physical.copy()
        offset = self.block_steps - self.remaining
        predicted = (self.plan['positions'][offset + 1] if self.plan is not None
                     and offset + 1 < len(self.plan['positions']) else None)
        self.remaining -= 1
        self.tick += 1
        info.update(planned=planned, template=self.template, remaining=self.remaining,
                    plan_index=self.plan_count, plan_offset=offset, candidate_policies=considered,
                    predicted_next_state=None if predicted is None else predicted.copy(),
                    prediction_mode=self.prediction, gate_only=False)
        return physical, info
