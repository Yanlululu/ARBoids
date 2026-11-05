"""Explicit experiment settings while retaining the released-code defaults."""
from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def evaluation_rng():
    """Keep validation independent of the training random-number streams."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def environment_kwargs(cfg):
    options = getattr(cfg, 'environment', None)
    return {} if options is None else dict(options if isinstance(options, dict) else vars(options))


def controller_name(agent):
    if not agent.residual:
        return 'RL'
    return 'AdaRes' if agent.adaptive else 'Res'


def apply_adapter_exploration(action, training):
    distribution = getattr(training, 'adapter_noise_distribution', 'uniform')
    scale = float(getattr(training, 'adapter_noise_scale', 0.1))
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('Adapter exploration scale must be finite and nonnegative')
    if distribution == 'uniform':
        # Preserve the released code's shared scalar draw and RNG sequence.
        noise = np.random.uniform(-scale, scale)
    elif distribution == 'normal':
        # Eq. (14): Gaussian noise, with 0.1 denoting standard deviation.
        noise = np.random.normal(0.0, scale, action.shape[0])
    else:
        raise ValueError(f'Unsupported adapter exploration distribution: {distribution}')
    action[:, -1] = np.clip(action[:, -1] + noise, 0.0, 1.0)
    return action
