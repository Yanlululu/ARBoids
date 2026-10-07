"""Causal planning cadence: longer commitment for slowly maneuvering targets."""
from jit_rollout import JitRolloutController


class AdaptiveIntervalController(JitRolloutController):
    def __init__(self, defenders, policy, *, agility_threshold=1.75, **settings):
        self.agility_threshold = float(agility_threshold)
        if 'block_steps' in settings and settings['block_steps'] != 5:
            raise ValueError('Adaptive control starts with the one-second interval.')
        settings['block_steps'] = 5
        super().__init__(defenders, policy, **settings)

    def reset(self, previous_thrust=None):
        super().reset(previous_thrust)
        self.block_steps = 5
        self.interval_plan_counts = {5:0, 10:0}

    def control(self, measurement):
        identify = self.adaptive_attacker
        if identify:
            self.attacker_agility = self.observer.update(measurement)
        if self.remaining == 0:
            # The estimate is based only on measurements up to this control tick.
            # Do not change an interval while its feedback plan is being executed.
            self.block_steps = 10 if self.attacker_agility < self.agility_threshold else 5
        self.adaptive_attacker = False
        try:
            force, info = super().control(measurement)
        finally:
            self.adaptive_attacker = identify
        info['execution_interval_steps'] = self.block_steps
        if info['planned']:
            self.interval_plan_counts[self.block_steps] += 1
        return force, info
