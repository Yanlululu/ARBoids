"""Current-configuration learning with one sampled contact decision per episode.

All other observed events use the current greedy policy. Four samples of a
training task share the same randomly assigned event ordinal and environmental
future. Complete returns update only the sampled action's policy probability;
there is no outcome predictor, critic, or stored action-value training target.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import torch

from citf.bridge import World
from citf.contracts import Protocol
from experiments import contact_event_policy as contact


@dataclass(frozen=True)
class SingleEventConfig(contact.ContactEventConfig):
    revision: str = 'CITF-contact-single-sampled-event-policy-v1'
    training_action_sampling: str = 'one uniformly assigned event ordinal; current greedy feedback elsewhere'


_initialize_contact_worker = contact.initialize


def initialize_single_event_worker(project, source, config):
    contact.ContactEventConfig = SingleEventConfig
    _initialize_contact_worker(project, source, config)


@torch.no_grad()
def rollout(job):
    worker = contact.worker
    model = worker['model']
    model.load_state_dict({key: torch.from_numpy(value) for key, value in job['state'].items()})
    torch.manual_seed(job['policy_seed'])
    task = job['task']
    world = World(worker['project'], Protocol(**job['protocol']), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    controller = contact.ContactEventController(worker['source'], model, teacher=worker['teacher'])
    # Independent sampling design, shared across the four policy draws. The seed
    # is not a model input and no future action or event is inspected to assign it.
    probe = int(np.random.default_rng(int(task['future_seed']) ^ 0x5EED9101).integers(4))
    transitions, entropies = [], []
    steps, events, changed, sampled = 0, 0, 0, 0
    while not world.done:
        if steps % 20 == 0 and (time.time() >= job['deadline'] or Path(job['stop_file']).exists()):
            return dict(completed=False, task=task, steps=steps, reason='deadline or cooperative stop')
        sample = bool(job['training'] and events == probe)
        controller.stochastic = sample
        force, _ = controller.act(world)
        transition = controller.transition
        if transition is not None:
            events += 1
            changed += transition['changed']
            entropies.append(transition['entropy'])
            if sample:
                sampled += 1
                transitions.append(transition)
        result = world.step(force)
        steps += 1
    duration = result['duration'] if result['capture'] else 80.
    reward = (1. if result['success'] else -1.) - .25 * result['collision'] + .02 * result['capture'] - .0005 * duration
    record = dict(completed=True, task=task, policy_seed=job['policy_seed'], steps=steps,
        decision_events=events, changed_events=changed, sampled_events=sampled,
        probe_event_ordinal=probe if job['training'] else None,
        execution='one_event_sample' if job['training'] else 'greedy',
        reward=reward, reward_duration='capped_capture',
        entropy=float(np.mean(entropies)) if entropies else 0., **result)
    return dict(completed=True, record=record, transitions=transitions)


def main():
    contact.ContactEventConfig = SingleEventConfig
    contact.initialize = initialize_single_event_worker
    contact.rollout = rollout
    if len(sys.argv) <= 1 or sys.argv[1] not in ('prepare', 'train'):
        import citf.evaluate as evaluation
        from citf.contracts import canonical_hash, file_hash
        original = evaluation.runtime_identity

        def identity(*args, **kwargs):
            value = original(*args, **kwargs)
            value['runtime_code'] = canonical_hash(dict(
                core=value['runtime_code'], single_event_policy=file_hash(__file__)))
            return value

        evaluation.runtime_identity = identity
    contact.main()


if __name__ == '__main__':
    main()
