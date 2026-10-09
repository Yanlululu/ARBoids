"""Parallel execution of the same causally chosen planning interval."""
from adaptive_interval_rollout import AdaptiveIntervalController
from parallel_jit_rollout import _forecast
from predictive_interception_v2 import TEMPLATES


class ParallelAdaptiveController(AdaptiveIntervalController):
    def __init__(self,defenders,policy,pool,**settings):
        self.pool = pool
        self.worker_settings = {k:v for k,v in settings.items() if k!='agility_threshold'}
        super().__init__(defenders,policy,**settings)

    def reset(self,previous_thrust=None):
        super().reset(previous_thrust)
        self._batch_key = None
        self._batch = {}

    def predict(self,measurement,template):
        previous = None if self.previous_thrust is None else self.previous_thrust.tobytes()
        attacker = None if self.previous_attacker_thrust is None else self.previous_attacker_thrust.tobytes()
        key = id(measurement),self.tick,self.block_steps,self.attacker_agility,previous,attacker
        if key != self._batch_key:
            self._measurement = measurement
            self._batch_key = key
            settings = dict(self.worker_settings,block_steps=self.block_steps)
            self._batch = {t:self.pool.executor.submit(_forecast,(
                self.defenders,settings,measurement,self.previous_thrust,self.previous_attacker_thrust,
                self.attacker_agility,t)) for t in TEMPLATES}
        return self._batch[template].result()
