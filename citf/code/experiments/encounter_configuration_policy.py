"""Refine current choices from original and encountered interception states.

A declared fraction of training groups first follows the current greedy
policy to its first close approach. The prefix is simulated but excluded
from policy gradients. The remaining trajectory uses the same ordinary
task return, learned reference KL and stopped-geometry replay as its matched
control. All evaluations start from the original task state.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

from citf.bridge import World
from citf.contracts import canonical_hash, file_hash
from experiments import reference_configuration_policy as reference


joint, policy = reference.joint, reference.policy
base_worker = reference.initialize_worker
base_rollout = reference.rollout


@dataclass(frozen=True)
class EncounterPolicyConfig(reference.ReferencePolicyConfig):
    revision: str = 'CITF-encounter-current-configuration-policy-v1'
    training_signal: str = 'ordinary task policy gradient from original and current-greedy first-approach states'
    encounter_probability: float = .5
    encounter_distances: tuple = (12., 8.)
    prefix_limit_seconds: float = 20.


class EncounterWorld(World):
    def reset(self, scene_seed, future_seed, agility):
        super().reset(scene_seed, future_seed, agility)
        worker = policy._worker
        info = dict(prefix_requested=False, prefix_seconds=0.,
                    prefix_simulation_steps=0, prefix_found_encounter=False,
                    prefix_fallback=False)
        worker['prefix_info'] = info
        if not worker.get('prefix_training', False):
            return
        cfg = worker['model'].cfg
        rng = np.random.default_rng(int(future_seed) + 390017)
        if rng.random() >= cfg.encounter_probability:
            return
        distance = float(rng.choice(cfg.encounter_distances))
        info.update(prefix_requested=True, encounter_distance=distance)
        controller = joint.JointPolicyController(worker['source'], worker['model'])
        for step in range(round(cfg.prefix_limit_seconds / self.protocol.control_dt) + 1):
            if step % 25 == 0 and (time.time() >= worker['prefix_deadline']
                                  or Path(worker['prefix_stop_file']).exists()):
                break
            state = self.state()
            nearest = np.linalg.norm(state[:-1, :2] - state[-1, :2], axis=-1).min()
            if nearest <= distance:
                info.update(prefix_found_encounter=True,
                            prefix_seconds=float(self.env.Current_T),
                            encountered_distance=float(nearest))
                return
            if step == round(cfg.prefix_limit_seconds / self.protocol.control_dt):
                break
            force, _ = controller.act(self)
            self.step(force)
            info['prefix_simulation_steps'] += 1
            if self.done:
                break
        # Neither a completed prefix nor a missing encounter becomes a
        # shortened successful episode. Keep the original task available.
        info['prefix_fallback'] = True
        super().reset(scene_seed, future_seed, agility)


def bind():
    reference.ReferencePolicyConfig = EncounterPolicyConfig
    reference.initialize_worker = initialize_worker
    reference.rollout = rollout


def initialize_worker(*args):
    bind()
    base_worker(*args)
    policy.World = EncounterWorld


def rollout(job):
    policy._worker.update(prefix_training=job['training'],
                          prefix_deadline=job['deadline'],
                          prefix_stop_file=job['stop_file'])
    result = base_rollout(job)
    if 'record' in result:
        result['record'].update(policy._worker['prefix_info'])
    return result


def main():
    bind()
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        save = reference.atomic_save
        implementation = file_hash(__file__)
        recorded = False

        def save_checkpoint(value, path):
            nonlocal recorded
            protocol = dict(value['protocol'])
            protocol['encounter_state_training'] = dict(
                prefix='current greedy policy to first 12m or 8m approach, at most 20s',
                probability=.5, group_shared_prefix=True,
                prefix_not_differentiated=True, no_value_head=True,
                fallback='original task if no encounter or prefix terminates',
                evaluation='original initial state, complete 80s task')
            protocol['runtime_code'] = canonical_hash(dict(core=protocol['runtime_code'],
                encounter_configuration_policy=implementation))
            if not recorded:
                (Path(path).parent / 'protocol.json').write_text(
                    reference.json.dumps(protocol, indent=2))
                recorded = True
            save(dict(value, protocol=protocol), path)

        reference.atomic_save = save_checkpoint
        reference.main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['training_state_curriculum'] = EncounterPolicyConfig().training_signal
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            encounter_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    reference.main()


if __name__ == '__main__':
    main()
