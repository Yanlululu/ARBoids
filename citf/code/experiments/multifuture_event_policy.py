"""One observed event, four sampled actions and four common future continuations.

The prefix follows the current greedy controller. Future RNG is resampled only
after the selected event's action has been drawn from observed inputs. All
continuations use current greedy feedback. No complete-return predictor is fit.
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
class MultiFutureEventConfig(contact.ContactEventConfig):
    revision: str = 'CITF-contact-common-future-event-policy-v1'
    training_action_sampling: str = 'one observed event; four sampled actions by four common futures'
    policy_draws_per_group: int = 4
    future_draws_per_action: int = 4
    group_reward_scale: float = 1.25


_initialize_contact_worker = contact.initialize


def initialize_multifuture_worker(project, source, config):
    contact.ContactEventConfig = MultiFutureEventConfig
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
    probe = int(np.random.default_rng(int(task['future_seed']) ^ 0x5EED9101).integers(4))
    transitions, entropies = [], []
    steps, events, changed, sampled = 0, 0, 0, 0
    intervention_step = None
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
                intervention_step = steps
                # No future sample is available to the model that chose force.
                world.rng = np.random.RandomState(task['branch_future_seed'])
        result = world.step(force)
        steps += 1
    duration = result['duration'] if result['capture'] else 80.
    raw_reward = (1. if result['success'] else -1.) - .25 * result['collision'] + .02 * result['capture'] - .0005 * duration
    # The existing trainer uses the 16-replica leave-one-out factor 16/15.
    # Replicas share only four independent policy draws. Scaling by 5/4 makes
    # the summed gradient equal the four-action leave-one-out factor 4/3.
    reward = raw_reward * model.cfg.group_reward_scale if job['training'] else raw_reward
    record = dict(completed=True, task=task, policy_seed=job['policy_seed'], steps=steps,
        decision_events=events, changed_events=changed, sampled_events=sampled,
        sampled_action=transitions[0]['action'] if transitions else None,
        probe_event_ordinal=probe if job['training'] else None,
        intervention_step=intervention_step,
        branch_future_seed=task.get('branch_future_seed') if job['training'] else None,
        execution='one_event_common_futures' if job['training'] else 'greedy',
        reward=reward, unscaled_task_reward=raw_reward, reward_duration='capped_capture',
        entropy=float(np.mean(entropies)) if entropies else 0., **result)
    return dict(completed=True, record=record, transitions=transitions)


def main():
    contact.ContactEventConfig = MultiFutureEventConfig
    contact.initialize = initialize_multifuture_worker
    contact.rollout = rollout
    if len(sys.argv) <= 1 or sys.argv[1] not in ('prepare', 'train'):
        import citf.evaluate as evaluation
        from citf.contracts import canonical_hash, file_hash
        original = evaluation.runtime_identity

        def identity(*args, **kwargs):
            value = original(*args, **kwargs)
            value['runtime_code'] = canonical_hash(dict(
                core=value['runtime_code'], multifuture_event=file_hash(__file__)))
            return value

        evaluation.runtime_identity = identity
    contact.main()


if __name__ == '__main__':
    main()
