"""Five independent candidate jobs for the batched rollout implementation."""
import study_runtime
from concurrent.futures import ProcessPoolExecutor

import torch

from array_rollout import ArrayRolloutController, BatchedSourcePolicy
from predictive_interception_v2 import TEMPLATES
from source_arboids import SourcePolicy


_POLICY = _BATCHED = None
_RULES = {}


def _initialize(checkpoint):
    global _POLICY, _BATCHED
    torch.set_num_threads(1)
    _POLICY = SourcePolicy(checkpoint)
    _BATCHED = BatchedSourcePolicy(_POLICY)


def _forecast(request):
    n, settings, measurement, previous, previous_attacker, agility, template = request
    key = n, tuple(sorted(settings.items()))
    if key not in _RULES:
        _RULES[key] = ArrayRolloutController(n, _POLICY, prediction_policy=_BATCHED, **settings)
    rule = _RULES[key]
    rule.previous_thrust = previous
    rule.previous_attacker_thrust = previous_attacker
    rule.attacker_agility = agility
    return rule.predict(measurement, template)


class ArrayPredictionPool:
    def __init__(self, checkpoint, workers=5):
        self.executor = ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
                                            initargs=(str(checkpoint),))
        self.workers = workers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=True, cancel_futures=True)


class ParallelArrayController(ArrayRolloutController):
    def __init__(self, defenders, policy, pool, **settings):
        self.pool = pool
        self.worker_settings = dict(settings)
        super().__init__(defenders, policy, **settings)

    def reset(self, previous_thrust=None):
        super().reset(previous_thrust)
        self._batch_key = None
        self._batch = {}

    def predict(self, measurement, template):
        previous = None if self.previous_thrust is None else self.previous_thrust.tobytes()
        attacker = None if self.previous_attacker_thrust is None else self.previous_attacker_thrust.tobytes()
        key = id(measurement), self.tick, self.attacker_agility, previous, attacker
        if key != self._batch_key:
            self._measurement = measurement
            self._batch_key = key
            self._batch = {candidate: self.pool.executor.submit(_forecast, (
                self.defenders, self.worker_settings, measurement, self.previous_thrust,
                self.previous_attacker_thrust, self.attacker_agility, candidate)) for candidate in TEMPLATES}
        return self._batch[template].result()
