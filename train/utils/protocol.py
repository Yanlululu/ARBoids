"""Explicit experiment settings while retaining the released-code defaults."""
import numpy as np


def environment_kwargs(cfg):
    options = getattr(cfg, 'environment', None)
    return {} if options is None else dict(options if isinstance(options, dict) else vars(options))


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
