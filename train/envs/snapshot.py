"""Full simulation snapshots for offline interventions, not policy observations."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
import torch

from envs.TADgame import TADEnv


@dataclass
class RandomState:
    numpy: tuple
    python: object
    torch_cpu: torch.Tensor
    torch_cuda: list | None

    @classmethod
    def capture(cls):
        # Do not initialize CUDA just to take a CPU simulation snapshot.
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        return cls(deepcopy(np.random.get_state()), random.getstate(),
                   torch.get_rng_state().clone(), cuda)

    def restore(self):
        np.random.set_state(deepcopy(self.numpy))
        random.setstate(self.python)
        torch.set_rng_state(self.torch_cpu.clone())
        if self.torch_cuda is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda)


def seed_random(seed):
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError('Use an integer seed in [0, 2**32).')
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.random.default_generator.manual_seed(int(seed))
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(int(seed))


@contextmanager
def preserved_random_state():
    state = RandomState.capture()
    try:
        yield
    finally:
        state.restore()


@dataclass
class SimulationSnapshot:
    """Own an independent copy of every environment field, including histories.

    RNG state includes APF force noise and water currents. Restoring with a new
    seed changes only the future randomness, not the captured physical state.
    Snapshots must be serialized/loaded with the same simulator code version.
    """
    environment: TADEnv
    random_state: RandomState
    version: int = 1

    @classmethod
    def capture(cls, env):
        if not isinstance(env, TADEnv) or not hasattr(env, 'Rewards'):
            raise ValueError('Capture a reset TADEnv.')
        return cls(deepcopy(env), RandomState.capture())

    def restore(self, *, future_seed=None):
        if self.version != 1:
            raise ValueError('Unsupported simulation snapshot version.')
        env = deepcopy(self.environment)
        if future_seed is None:
            self.random_state.restore()
        else:
            seed_random(future_seed)
        return env
