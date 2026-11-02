"""Load byte-identical author release code without importing the improved codebase.

The vendored files are git blobs from the first upstream code release. The
original numerical environment, rewards, observations, actor and VRX functions
are executed as published. Adapters and added controllers live elsewhere.
"""
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


SOURCE_COMMIT = '9eb6df464808af6b9729f39644b4d658e1695e2d'
SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'third_party' / 'arboids_release'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source():
    manifest = json.loads((SOURCE_ROOT / 'provenance.json').read_text(encoding='utf-8'))
    if manifest['commit'] != SOURCE_COMMIT:
        raise RuntimeError('The comparison requires the pinned upstream release.')
    for name, expected in manifest['files'].items():
        if sha256(SOURCE_ROOT / name) != expected:
            raise RuntimeError(f'Original source has been modified: {name}')
    return manifest


def verify_source_config(path):
    import yaml
    expected = yaml.safe_load((SOURCE_ROOT/'train/configs/train.yaml').read_text(encoding='utf-8'))
    actual = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if actual != expected:
        raise ValueError('Checkpoint training configuration differs from the untouched upstream source protocol.')
    return sha256(path)


@contextmanager
def _aliases(mapping):
    previous = {name: sys.modules.get(name) for name in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load(name, relative):
    path = SOURCE_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def numerical_source():
    verify_source()
    dynamics = _load('_author_dynamics', 'train/envs/modules.py')
    with _aliases({'envs.modules': dynamics}):
        environment = _load('_author_environment', 'train/envs/TADgame.py')
    networks = _load('_author_networks', 'train/policy/networks.py')
    with _aliases({'policy.networks': networks}):
        sac = _load('_author_sac', 'train/policy/SAC.py')
    return SimpleNamespace(TADEnv=environment.TADEnv, WAMV=dynamics.WAMV,
                           ActorAdap=networks.ActorAdap, SAC=sac.SAC)


@lru_cache(maxsize=1)
def vrx_source():
    verify_source()
    models = _load('_author_vrx_models', 'vrx/models.py')
    utilities = _load('_author_vrx_utils', 'vrx/utils.py')
    with _aliases({'models': models, 'utils': utilities}):
        controller = _load('_author_vrx_controller', 'vrx/tad_vrx_experiment.py')
    return SimpleNamespace(models=models, controller=controller)


class SourcePolicy:
    """Original ActorAdap, frozen source-protocol weights, original SAC inference."""
    def __init__(self, checkpoint):
        verify_source_config(Path(checkpoint).with_name('config.yaml'))
        source = numerical_source()
        self.actor = source.ActorAdap(6, 8, 3, 512)
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.actor.load_state_dict(weights, strict=True)
        self.actor.eval()
        self.device = torch.device('cpu')
        self.choose_action = source.SAC.choose_action.__get__(self)

    def __call__(self, observations):
        # Match train.py's original per-defender invocation, including rounding.
        return np.stack([self.choose_action(row[None], deterministic=True)
                         for row in observations])


def snapshot(env):
    return np.asarray([[boat.x, boat.y, boat.theta, *boat.velocity]
                       for boat in env.defender_list], dtype=float)


def actor_thrust(action):
    # Preserve the original expression and float32 assignment, not just its
    # algebraic simplification, so the nominal command remains byte-identical.
    return (action[:, :2].copy() * 1500. + 1000. - 500.) / 2.


def source_mixture(action, boids):
    thrust = actor_thrust(action)
    for i in range(len(action)):
        thrust[i] = action[i, 2] * thrust[i] + (1 - action[i, 2]) * boids[i]
    return thrust
