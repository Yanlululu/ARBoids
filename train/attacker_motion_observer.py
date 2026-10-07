"""Causal identification from past public attacker measurements only."""
import numpy as np

from feedback_joint_control import nominal_attacker_command
from gate_study_control import NominalModel


class AttackerMotionObserver:
    def __init__(self):
        self.model = NominalModel()
        self.grid = np.linspace(1.25, 3.25, 17)
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_unit_command = None
        self.losses = np.zeros(len(self.grid))
        self.updates = 0
        self.estimate = 2.25
        self.uncertainty = .6

    def update(self, measurement):
        if self.previous is not None:
            elapsed = measurement.time - self.previous.time
            if not np.isclose(elapsed, .2, atol=1e-8):
                raise ValueError('Motion observer requires consecutive 0.2-second measurements.')
            states = np.broadcast_to(self.previous.attacker, (len(self.grid), 6)).copy()
            if self.previous_unit_command is not None:
                old_forces = self.grid[:, None] * self.previous_unit_command
                states[:, 3:] += .05 * self.model.acceleration(states, old_forces)
            unit_command = nominal_attacker_command(self.previous, 1.)
            forces = self.grid[:, None] * unit_command
            for _ in range(4):
                cached = states[:, 3:].copy()
                states[:, :2] += .05 * cached[:, :2]
                states[:, 2] = (states[:, 2] + .05 * cached[:, 2]) % (2*np.pi)
                states[:, 3:] += .05 * self.model.acceleration(states, forces)
            position_error = np.sum(((states[:, :2] - measurement.attacker[:2]) / .08)**2, axis=-1)
            velocity_error = np.sum(((cached[:, :2] - measurement.attacker[3:5]) / .35)**2, axis=-1)
            yaw_error = ((cached[:, 2] - measurement.attacker[5]) / .07)**2
            loss = position_error + velocity_error + yaw_error
            self.losses = .9 * self.losses + loss
            weights = np.exp(-.5 * (self.losses - self.losses.min()))
            weights /= weights.sum()
            self.estimate = float(weights @ self.grid)
            self.uncertainty = float(np.sqrt(weights @ ((self.grid - self.estimate)**2)))
            self.previous_unit_command = unit_command.copy()
            self.updates += 1
        self.previous = measurement
        return self.estimate
