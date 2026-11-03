"""Train current actions from both initial and encountered training states.

Some training episodes begin learning after a deterministic physical-policy
prefix. This gives the task policy shorter credit paths near pursuit and
defense events. The source, world, original deadline and terminal outcomes
are unchanged. Evaluation always begins at the original initial state.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import torch

from citf.bridge import World, factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.search import nominal_beam, initialize_bank, whole_configuration_search
from experiments import balanced_configuration_policy as balanced

joint, policy = balanced.joint, balanced.policy


@dataclass(frozen=True)
class RestartPolicyConfig(balanced.BalancedPolicyConfig):
    revision: str = 'CITF-encountered-state-current-configuration-policy-v1'
    training_signal: str = 'ordinary task policy gradient from mixed original and physical-prefix training states'
    prefix_probability: float = .5
    prefix_seconds: tuple = (5., 10., 20., 30.)


class PhysicalPrefixController:
    def __init__(self, source, cfg):
        self.source, self.cfg = source, cfg

    def act(self, world):
        f, cfg = factorize(world, self.source), self.cfg
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), cfg.objective_risk_weight, 128)
        initial, _ = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                     f['prior'].numpy(), 12, 96)

        def score(plans):
            batch, _, _ = inputs(world, f, plans)
            return policy.physical_components(batch, cfg)[0][0].numpy()

        indices, _ = whole_configuration_search(initial, score,
            [row.nonzero()[:, 0].numpy() for row in f['valid']], 96, 1)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        return (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()


class RestartWorld(World):
    def reset(self, scene_seed, future_seed, agility):
        super().reset(scene_seed, future_seed, agility)
        state = policy._worker
        info = dict(prefix_requested_seconds=0., prefix_seconds=0.,
                    prefix_simulation_steps=0, prefix_terminated_fallback=False)
        state['prefix_info'] = info
        if not state.get('prefix_training', False):
            return
        cfg = state['model'].cfg
        rng = np.random.default_rng(int(future_seed) + 730003)
        if rng.random() >= cfg.prefix_probability:
            return
        seconds = float(rng.choice(cfg.prefix_seconds))
        info['prefix_requested_seconds'] = seconds
        controller = PhysicalPrefixController(state['source'], cfg)
        for step in range(round(seconds / self.protocol.control_dt)):
            if step % 25 == 0 and (time.time() >= state['prefix_deadline']
                                  or Path(state['prefix_stop_file']).exists()):
                break
            self.step(controller.act(self))
            info['prefix_simulation_steps'] += 1
            if self.done:
                info['prefix_terminated_fallback'] = True
                # A completed prefix is not relabelled as a learned episode.
                # Replay the original world and train a complete policy run.
                super().reset(scene_seed, future_seed, agility)
                return
        info['prefix_seconds'] = float(self.env.Current_T)


_original_worker = balanced.initialize_worker


def bind():
    balanced.BalancedPolicyConfig = RestartPolicyConfig
    balanced.initialize_worker = initialize_worker
    balanced.rollout = rollout


def initialize_worker(*args):
    bind()
    _original_worker(*args)
    policy.World = RestartWorld


def rollout(job):
    policy._worker.update(prefix_training=job['training'], prefix_deadline=job['deadline'],
                          prefix_stop_file=job['stop_file'])
    result = joint.rollout(job)
    if 'record' in result:
        result['record'].update(policy._worker['prefix_info'])
    return result


def main():
    bind()
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        balanced.train_main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['training_state_curriculum'] = RestartPolicyConfig().training_signal
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            restart_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    balanced.main()


if __name__ == '__main__':
    main()
