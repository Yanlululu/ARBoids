"""Matched execution experiment: hold fusion coefficients, retain source feedback.

The same controller wrapper is used by zero-actor physical and learned models.
The original every-tick physical reference remains a separate comparison.
"""
from dataclasses import dataclass
import time

import torch

from experiments import synchronized_configuration_policy as synchronized


joint = synchronized.joint


@dataclass(frozen=True)
class HeldPolicyConfig(synchronized.SynchronizedPolicyConfig):
    revision: str = 'CITF-held-current-configuration-development-v1'
    hold_steps: int = 4
    execution_condition: str = 'fixed fusion coefficients for declared ticks; SAC and Boids recomputed every tick'


class HeldPolicyController(joint.JointPolicyController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remaining = 0
        self.gates = None
        self.last_trace = None

    @torch.no_grad()
    def act(self, world):
        started = time.perf_counter()
        if self.remaining == 0:
            force, trace = super().act(world)
            self.gates = torch.as_tensor(trace['gates'])
            self.last_trace = dict(trace)
            self.remaining = self.model.cfg.hold_steps - 1
            return force, dict(trace, held_coefficients=False,
                               hold_steps=self.model.cfg.hold_steps)
        a, b, _ = self.source.commands(world.env)
        force = (750 * (self.gates * a + (1 - self.gates) * b) + 250).numpy()
        self.remaining -= 1
        trace = dict(self.last_trace, held_coefficients=True,
            physical_thrust=force.tolist(), evaluated_unique=0,
            hold_steps=self.model.cfg.hold_steps, hold_remaining=self.remaining,
            compute_seconds=time.perf_counter() - started)
        return force, trace


def main():
    synchronized.install_semantics()
    joint.JointPolicyConfig = HeldPolicyConfig
    joint.JointPolicyController = HeldPolicyController
    joint.main()


if __name__ == '__main__':
    main()
