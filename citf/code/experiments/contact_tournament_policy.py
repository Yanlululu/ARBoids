"""Compare the current leading alternative with retained and exploratory actions.

Every deployment candidate remains available. Training evaluates three current
joint actions under two common futures, then ranks all observed action pairs.
This is a sampled tournament objective, not an unbiased uniform-bank objective
or a fitted prediction of complete-task returns.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from citf.bridge import World
from citf.contracts import Protocol
from citf.search import candidate_grid
from experiments import contact_rank_policy as ranking

common = ranking.common
training = ranking.training


@dataclass(frozen=True)
class TournamentConfig(ranking.RankConfig):
    revision: str = 'CITF-common-future-leading-candidate-tournament-v1'
    training_action_sampling: str = 'retained, leading alternative, exploratory alternative; two common futures each'
    decision_domain: str = 'all original joint proposals; retained-relative greedy ranking'


def initialize(project, source, config):
    training.ModelConfig = TournamentConfig
    common._initialize_training_worker(project, source, config)


@torch.no_grad()
def rollout(job):
    if not job['training']:
        return common.rollout(job)
    worker = training.worker
    model = worker['model']
    model.load_state_dict({k: torch.from_numpy(v) for k, v in job['state'].items()})
    torch.manual_seed(job['policy_seed'])
    task = job['task']
    world = World(worker['project'], Protocol(**job['protocol']), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    controller = common.CommonFutureController(worker['source'], model, teacher=worker['teacher'])
    probe = int(np.random.default_rng(int(task['future_seed']) ^ 0x5EED9101).integers(4))
    transitions, steps, events, changed = [], 0, 0, 0
    intervention_step, leading, sampled, sampling_probability = None, None, None, None
    while not world.done:
        if steps % 20 == 0 and (time.time() >= job['deadline'] or Path(job['stop_file']).exists()):
            return dict(completed=False, task=task, steps=steps, reason='deadline or cooperative stop')
        force, trace = controller.act(world)
        transition = controller.transition
        if transition is not None:
            if events == probe:
                batch = {k: torch.from_numpy(v[None]) for k, v in transition['inputs'].items()}
                logits = model(batch)['logits'][0]
                probabilities = logits.softmax(0)
                teacher = trace['retained_proposal_index']
                plans = np.asarray(trace['configurations'], dtype=np.int64)
                valid = batch['candidate_valid'][0].clone()
                valid[teacher] = False
                available = valid.nonzero().flatten().tolist()
                leading = min(available, key=lambda i: (-float(logits[i]), tuple(plans[i]))) if available else teacher
                # Query the deployment competitor explicitly and use the third
                # branch to cover a different alternative whenever one exists.
                valid[leading] = False
                if valid.any():
                    sampling_logits = torch.from_numpy(transition['sampling_logits'])
                    conditional = sampling_logits.masked_fill(~valid, -torch.inf).softmax(0)
                    uniform = valid.float() / valid.sum()
                    q = ((1 - model.cfg.alternative_uniform_fraction) * conditional
                         + model.cfg.alternative_uniform_fraction * uniform)
                    sampler = torch.distributions.Categorical(probs=q)
                    sampled = int(sampler.sample())
                    sampling_probability = float(sampler.probs[sampled])
                else:
                    sampled, sampling_probability = leading, 1.
                trials = dict(teacher=teacher, leading_alternative=leading, sampled_alternative=sampled)
                action = trials[job['proposal']]
                transition = dict(transition, action=action, retained_index=teacher,
                    leading_alternative=leading, sampled_alternative=sampled,
                    alternative_sampling_probability=sampling_probability,
                    old_probabilities=probabilities.numpy().copy(),
                    old_logp=float(probabilities[action].clamp_min(1e-12).log()),
                    changed=int(action != teacher))
                learned, rules, prior = worker['source'].commands(world.env)
                grid, _ = candidate_grid(prior)
                gate = grid[torch.arange(len(plans[action])), torch.as_tensor(plans[action])]
                force = (750 * (gate * learned + (1 - gate) * rules) + 250).numpy()
                transitions.append(transition)
                intervention_step = steps
                world.rng = np.random.RandomState(job['branch_future_seed'])
            changed += transition['changed']
            events += 1
        result = world.step(force)
        steps += 1
    duration = result['duration'] if result['capture'] else 80.
    reward = (1. if result['success'] else -1.) - .25 * result['collision'] + .02 * result['capture'] - .0005 * duration
    record = dict(completed=True, task=task, policy_seed=job['policy_seed'], steps=steps,
        decision_events=events, changed_events=changed, sampled_events=len(transitions),
        proposal=job['proposal'], future_replica=job['future_replica'], root_group=job['root_group'],
        probe_event_ordinal=probe, intervention_step=intervention_step,
        branch_future_seed=job['branch_future_seed'],
        sampled_action=transitions[0]['action'] if transitions else None,
        leading_alternative=leading, sampled_alternative=sampled,
        alternative_sampling_probability=sampling_probability,
        policy_candidate_count=len(transitions[0]['old_probabilities']) if transitions else None,
        execution='leading_candidate_common_future_tournament',
        reward=reward, reward_duration='capped_capture', **result)
    return dict(completed=True, record=record, transitions=transitions)


def event_examples(results):
    examples, missing = [], 0
    for start in range(0, len(results), 6):
        group = results[start:start + 6]
        present = [bool(x['transitions']) for x in group]
        if not any(present):
            missing += 1
            continue
        if len(group) != 6 or not all(present):
            raise ValueError('Three-action common-future trials reached inconsistent events')
        reference = group[0]['transitions'][0]
        keys = ('retained_index', 'leading_alternative', 'sampled_alternative')
        if any(any(x['transitions'][0][k] != reference[k] for k in keys) for x in group):
            raise ValueError('Tournament branches must share the same current candidate choices')
        observed = {}
        for x in group:
            observed.setdefault(x['transitions'][0]['action'], []).append(x['record']['reward'])
        indices = list(observed)
        returns = [float(np.mean(observed[i])) for i in indices]
        probabilities = reference['old_probabilities']
        teacher = reference['retained_index']
        examples.append(dict(inputs=reference['inputs'], probabilities=probabilities,
            advantage=np.zeros_like(probabilities),
            teacher_index=teacher, leading_alternative_index=reference['leading_alternative'],
            alternative_index=reference['sampled_alternative'],
            sampling_probability=reference['alternative_sampling_probability'],
            alternative_count=int(reference['inputs']['candidate_valid'].sum()) - 1,
            observed_candidate_indices=indices, observed_returns=returns,
            teacher_return=float(np.mean([x['record']['reward'] for x in group[:2]])),
            comparison_return=float(np.mean([x['record']['reward'] for x in group[2:]]))))
    return examples, missing


def tournament_objective(logits, examples):
    losses = []
    for row, x in enumerate(examples):
        indices = torch.tensor(x['observed_candidate_indices'], device=logits.device)
        returns = logits.new_tensor(x['observed_returns'])
        i, j = torch.triu_indices(len(indices), len(indices), 1, device=logits.device)
        if len(i):
            gap = logits[row, indices[i]] - logits[row, indices[j]]
            difference = returns[i] - returns[j]
            losses.append((difference.abs() * F.softplus(-difference.sign() * gap)).mean())
        else:
            losses.append(logits[row, x['teacher_index']] * 0.)
    return torch.stack(losses).mean()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        training.ModelConfig = TournamentConfig
        training.COMPARISON_PROPOSALS = ('teacher', 'leading_alternative', 'sampled_alternative')
        training.initialize = initialize
        training.rollout = rollout
        training.event_examples = event_examples
        training.ACTOR_OBJECTIVE = tournament_objective
        training.PROTOCOL_OVERRIDES = dict(
            objective='mean-return-weighted logistic ranking of all distinct observed action pairs in each current three-action tournament',
            action_support='all original joint proposals at deployment; the leading alternative is explicitly compared with retained and exploratory proposals',
            sampling='retained plus highest deployment-scored non-retained candidate plus90percent conditional/10percent uniform draw over remaining alternatives',
            future_averaging='each of the three current proposals uses the same two future seeds; average returns by action before all-pair ranking',
            importance='no inverse-propensity correction: objective targets the recorded leading/exploratory tournament, not a uniform mean over all bank pairs',
            optimization='same optimizer and physical replay schedule as current-only ranking; primary experiment has no historical task-comparison replay',
            budget='same320scheduled root/prefix groups,80updates and optimizer counts; three proposals by two futures gives1920realepisodes rather than1280; additional candidate examples increase compute',
            target='same candidate score differences as full-bank greedy deployment; compare the actual leading alternative directly',
            claim_boundary='sampling coverage and sampled-pair objective change together; labels use collection-time greedy continuation; no unbiased latest-policy gradient, CV, critic or terminal-return prediction claim')
        return training.train_main()
    common.CommonFutureConfig = TournamentConfig
    return common.main()


if __name__ == '__main__':
    main()
