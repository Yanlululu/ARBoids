"""Optional process parallelism for identical independent candidate forecasts.

Each worker runs the frozen predictor with the same public measurement, known
past commands, checkpoint and model estimate. Optional Actor compilation removes
Python overhead while retaining per-vessel arithmetic. Exact equivalence is
checked against the serial implementation by the runtime audit.
"""
import study_runtime
from concurrent.futures import ProcessPoolExecutor

import torch

from predictive_interception_v2 import TEMPLATES
from rollout_interception import RolloutInterceptionController
from source_arboids import SourcePolicy
from compiled_source_policy import CompiledSourcePolicy


_POLICY = None
_RULES = {}


def _initialize(checkpoint,compile_actor):
    global _POLICY
    torch.set_num_threads(1)
    _POLICY = SourcePolicy(checkpoint)
    if compile_actor:
        _POLICY=CompiledSourcePolicy(_POLICY)


def _forecast(request):
    defenders,settings,measurement,previous,previous_attacker,agility,template=request
    key=defenders,tuple(sorted(settings.items()))
    if key not in _RULES:
        _RULES[key]=RolloutInterceptionController(defenders,_POLICY,**settings)
    rule=_RULES[key]
    rule.previous_thrust=previous
    rule.previous_attacker_thrust=previous_attacker
    rule.attacker_agility=agility
    return rule.predict(measurement,template)


class PredictionPool:
    def __init__(self,checkpoint,workers=5,compile_actor=True):
        self.executor=ProcessPoolExecutor(max_workers=workers,initializer=_initialize,
                                          initargs=(str(checkpoint),bool(compile_actor)))
        self.workers=workers
        self.compile_actor=bool(compile_actor)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.executor.shutdown(wait=True,cancel_futures=True)


class ParallelRolloutInterceptionController(RolloutInterceptionController):
    def __init__(self,defenders,policy,pool,**settings):
        self.pool=pool
        self.worker_settings=dict(settings)
        super().__init__(defenders,policy,**settings)

    def reset(self,previous_thrust=None):
        super().reset(previous_thrust)
        self._batch_key=None
        self._batch={}

    def predict(self,measurement,template):
        previous=None if self.previous_thrust is None else self.previous_thrust.tobytes()
        previous_attacker=None if self.previous_attacker_thrust is None else self.previous_attacker_thrust.tobytes()
        key=id(measurement),self.tick,self.attacker_agility,previous,previous_attacker
        if key!=self._batch_key:
            self._measurement=measurement  # retain the identity until the batch ends
            self._batch_key=key
            self._batch={candidate:self.pool.executor.submit(_forecast,(self.defenders,self.worker_settings,
                measurement,self.previous_thrust,self.previous_attacker_thrust,self.attacker_agility,candidate))
                for candidate in TEMPLATES}
        return self._batch[template].result()
