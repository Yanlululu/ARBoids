"""Independent candidate workers for the compiled task-prediction backend."""
import study_runtime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import torch

from array_rollout import BatchedSourcePolicy
from jit_rollout import JitRolloutController
from predictive_interception_v2 import TEMPLATES
from source_arboids import SourcePolicy


_POLICY = _BATCHED = None
_RULES = {}


def _initialize(checkpoint, warmup=None, barrier=None):
    global _POLICY, _BATCHED
    torch.set_num_threads(1)
    _POLICY = SourcePolicy(checkpoint)
    _BATCHED = BatchedSourcePolicy(_POLICY)
    if warmup is not None:
        n, settings, measurement = warmup
        _forecast((n, settings, measurement, None, None, 2.25, 'baseline'))
    if barrier is not None:
        barrier.wait(timeout=60.)


def _forecast(request):
    n, settings, measurement, previous, previous_attacker, agility, template = request
    key = n, tuple(sorted(settings.items()))
    if key not in _RULES:
        _RULES[key] = JitRolloutController(n, _POLICY, prediction_policy=_BATCHED, **settings)
    rule = _RULES[key]
    rule.previous_thrust = previous
    rule.previous_attacker_thrust = previous_attacker
    rule.attacker_agility = agility
    return rule.predict(measurement, template)


class JitPredictionPool:
    def __init__(self, checkpoint, workers=5, warmup=None):
        context = multiprocessing.get_context('spawn')
        barrier = context.Barrier(workers) if warmup is not None else None
        self.executor = ProcessPoolExecutor(max_workers=workers, initializer=_initialize,
            initargs=(str(checkpoint),warmup,barrier), mp_context=context)
        self.workers = workers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=True, cancel_futures=True)


class ParallelJitController(JitRolloutController):
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
            self._batch = {candidate:self.pool.executor.submit(_forecast, (
                self.defenders, self.worker_settings, measurement, self.previous_thrust,
                self.previous_attacker_thrust, self.attacker_agility, candidate)) for candidate in TEMPLATES}
        return self._batch[template].result()
