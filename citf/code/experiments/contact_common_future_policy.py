"""Full-candidate current-event learning with stratified common-future trials.

The retained proposal is evaluated explicitly. An alternative is sampled from
the entire remaining joint bank, with importance weighting for its sampling
probability. Returns from two shared futures are averaged before actor updates.
Online selection always scores the complete joint candidate bank. Sampling
normalizes total alternative prior mass; deployment uses a separate margin
relative to the best alternative, without a candidate-count penalty.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import torch

from citf.bridge import World
from citf.contracts import Protocol
from citf.search import candidate_grid
from experiments import contact_event_policy as contact
from experiments import contact_enumerated_pair_policy as training
from experiments import contact_retained_mass_policy as retained


@dataclass(frozen=True)
class CommonFutureConfig(retained.RetainedMassConfig):
    revision: str = 'CITF-full-bank-decoupled-common-future-policy-v2'
    training_action_sampling: str = 'retained proposal and full-support alternative; two common futures each'
    alternative_uniform_fraction: float = .1
    sampling_retained_logodds: float = 1.5
    deployment_retained_margin: float = 1.5


class CommonFutureController(contact.ContactEventController):
    @torch.no_grad()
    def act(self, world):
        stochastic = self.stochastic
        self.stochastic = False
        try:
            force, trace = super().act(world)
        finally:
            self.stochastic = stochastic
        if self.transition is None:
            return force, trace
        values = self.transition['inputs']
        plans = np.asarray(trace['configurations'], dtype=np.int64)
        teacher = next(i for i, plan in enumerate(plans)
                       if plan.tolist() == trace['retained_teacher_indices'])
        source_logits = values['teacher_logits'].copy()
        alternative = np.arange(len(plans)) != teacher
        deployment_prior = np.zeros(len(plans), dtype=np.float32)
        sampling_prior = np.zeros(len(plans), dtype=np.float32)
        if alternative.any():
            scores = torch.from_numpy(source_logits[alternative])
            # The best alternative needs only the declared deployment margin,
            # regardless of how many other alternatives share its score.
            deployment_prior[alternative] = (scores - scores.max()
                - self.model.cfg.deployment_retained_margin).numpy()
            sampling_prior[alternative] = (scores - torch.logsumexp(scores, dim=0)
                - self.model.cfg.sampling_retained_logodds).numpy()
        values['teacher_logits'] = deployment_prior
        batch = {k: torch.from_numpy(v[None]).to(self.device) for k, v in values.items()}
        logits = self.model(batch)['logits'][0]
        residual = logits - torch.as_tensor(deployment_prior, device=logits.device)
        sampling_logits = residual + torch.as_tensor(sampling_prior, device=logits.device)
        deployment_distribution = torch.distributions.Categorical(logits=logits)
        sampling_distribution = torch.distributions.Categorical(logits=sampling_logits)
        choice = int(sampling_distribution.sample()) if stochastic else min(
            range(len(plans)), key=lambda i: (-float(logits[i]), i != teacher, tuple(plans[i])))
        phases = self.transition['phases']
        self.transition = dict(inputs={k: v.copy() for k, v in values.items()},
            action=choice, old_logp=float(deployment_distribution.log_prob(
                torch.tensor(choice, device=logits.device))),
            changed=int(choice != teacher), entropy=float(deployment_distribution.entropy()),
            sampling_logits=sampling_logits.cpu().numpy().copy(), phases=phases)
        learned, rules, prior = self.source.commands(world.env)
        grid, _ = candidate_grid(prior)
        gate = grid[torch.arange(len(plans[choice])), torch.as_tensor(plans[choice])]
        force = (750 * (gate * learned + (1 - gate) * rules) + 250).numpy()
        trace.update(scores=(-logits).cpu().tolist(), selected_indices=plans[choice].tolist(),
            gates=gate.tolist(), physical_thrust=force.tolist(), teacher_logits=deployment_prior.tolist(),
            source_teacher_logits=source_logits.tolist(), sampling_prior_logits=sampling_prior.tolist(),
            sampling_logits=sampling_logits.cpu().tolist(), retained_proposal_index=teacher,
            deployment_retained_margin=self.model.cfg.deployment_retained_margin,
            sampling_retained_logodds=self.model.cfg.sampling_retained_logodds,
            selection='all original proposals; candidate-count-independent deployment margin; separate sampling prior')
        return force, trace


_initialize_training_worker = training.initialize


def initialize(project, source, config):
    training.ModelConfig = CommonFutureConfig
    _initialize_training_worker(project, source, config)


@torch.no_grad()
def rollout(job):
    worker = training.worker
    model = worker['model']
    model.load_state_dict({k: torch.from_numpy(v) for k, v in job['state'].items()})
    torch.manual_seed(job['policy_seed'])
    task = job['task']
    world = World(worker['project'], Protocol(**job['protocol']), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    controller = CommonFutureController(worker['source'], model, teacher=worker['teacher'])
    probe = int(np.random.default_rng(int(task['future_seed']) ^ 0x5EED9101).integers(4))
    transitions, steps, events, changed = [], 0, 0, 0
    intervention_step, sampled_alternative, sampling_probability = None, None, None
    while not world.done:
        if steps % 20 == 0 and (time.time() >= job['deadline'] or Path(job['stop_file']).exists()):
            return dict(completed=False, task=task, steps=steps, reason='deadline or cooperative stop')
        force, trace = controller.act(world)
        transition = controller.transition
        if transition is not None:
            if job['training'] and events == probe:
                inputs = transition['inputs']
                batch = {k: torch.from_numpy(v[None]) for k, v in inputs.items()}
                logits = model(batch)['logits'][0]
                probabilities = logits.softmax(0)
                teacher = trace['retained_proposal_index']
                valid = batch['candidate_valid'][0].clone()
                valid[teacher] = False
                if valid.any():
                    sampling_logits = torch.from_numpy(transition['sampling_logits'])
                    conditional = sampling_logits.masked_fill(~valid, -torch.inf).softmax(0)
                    uniform = valid.float() / valid.sum()
                    q = ((1 - model.cfg.alternative_uniform_fraction) * conditional
                         + model.cfg.alternative_uniform_fraction * uniform)
                    sampler = torch.distributions.Categorical(probs=q)
                    sampled_alternative = int(sampler.sample())
                    sampling_probability = float(sampler.probs[sampled_alternative])
                else:
                    sampled_alternative, sampling_probability = teacher, 1.
                action = teacher if job['proposal'] == 'teacher' else sampled_alternative
                transition = dict(transition, action=action, retained_index=teacher,
                    sampled_alternative=sampled_alternative,
                    alternative_sampling_probability=sampling_probability,
                    old_probabilities=probabilities.numpy().copy(),
                    old_logp=float(probabilities[action].clamp_min(1e-12).log()),
                    changed=int(action != teacher))
                plans = np.asarray(trace['configurations'], dtype=np.int64)
                learned, rules, prior = worker['source'].commands(world.env)
                grid, _ = candidate_grid(prior)
                gate = grid[torch.arange(len(plans[action])), torch.as_tensor(plans[action])]
                force = (750 * (gate * learned + (1 - gate) * rules) + 250).numpy()
                transitions.append(transition)
                intervention_step = steps
                # Future noise is available only after current inputs and action.
                world.rng = np.random.RandomState(job['branch_future_seed'])
            changed += transition['changed']
            events += 1
        result = world.step(force)
        steps += 1
    duration = result['duration'] if result['capture'] else 80.
    reward = (1. if result['success'] else -1.) - .25 * result['collision'] + .02 * result['capture'] - .0005 * duration
    record = dict(completed=True, task=task, policy_seed=job['policy_seed'], steps=steps,
        decision_events=events, changed_events=changed, sampled_events=len(transitions),
        proposal=job.get('proposal'), future_replica=job.get('future_replica'),
        root_group=job.get('root_group'), probe_event_ordinal=probe if job['training'] else None,
        intervention_step=intervention_step,
        branch_future_seed=job.get('branch_future_seed') if job['training'] else None,
        sampled_action=transitions[0]['action'] if transitions else None,
        sampled_alternative=sampled_alternative, alternative_sampling_probability=sampling_probability,
        policy_candidate_count=len(transitions[0]['old_probabilities']) if transitions else None,
        execution='full_support_decoupled_common_future_event' if job['training'] else 'greedy',
        reward=reward, reward_duration='capped_capture', **result)
    return dict(completed=True, record=record, transitions=transitions)


def event_examples(results):
    examples, missing = [], 0
    for start in range(0, len(results), 4):
        group = results[start:start + 4]
        present = [bool(x['transitions']) for x in group]
        if not any(present):
            missing += 1
            continue
        if not all(present):
            raise ValueError('Common greedy prefixes reached inconsistent events')
        reference = group[0]['transitions'][0]
        selected = reference['sampled_alternative']
        teacher = reference['retained_index']
        if any(x['transitions'][0]['sampled_alternative'] != selected or
               x['transitions'][0]['retained_index'] != teacher for x in group):
            raise ValueError('Common-future branches must share the current action draw')
        probabilities = reference['old_probabilities']
        teacher_return = float(np.mean([x['record']['reward'] for x in group[:2]]))
        comparison_return = float(np.mean([x['record']['reward'] for x in group[2:]]))
        advantage = np.zeros_like(probabilities)
        weights = np.zeros_like(probabilities)
        if selected != teacher:
            advantage[selected] = comparison_return - teacher_return
            weights[selected] = probabilities[selected] / reference['alternative_sampling_probability']
        # E_q[(pi_old(a)/q(a))*(pi_new(a)/pi_old(a))*(R_a-R_teacher)]
        # covers every non-teacher action. The teacher has zero relative return;
        # its probability changes through the same full-bank softmax denominator.
        examples.append(dict(inputs=reference['inputs'], probabilities=probabilities,
            objective_weights=weights, advantage=advantage,
            teacher_return=teacher_return, comparison_return=comparison_return))
    return examples, missing


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        training.ModelConfig = CommonFutureConfig
        training.COMPARISON_PROPOSALS = ('teacher', 'sampled_alternative')
        training.PROTOCOL_OVERRIDES = dict(
            objective='full-bank current-event actor learning; stratified reference/alternative trials with pi_old/q importance weights; complete returns averaged over shared futures before clipping',
            action_support='every original joint candidate has positive training sampling support; all candidates participate in online selection',
            reference_role='actual teacher-action return is a sampled action-independent baseline for the actor update; no return or difference prediction head is fit',
            alternative_sampling='90percent current conditional policy plus10percent uniform over all non-retained proposals',
            sampling_deployment_separation='sampling alternative prior subtracts logsumexp and sampling_retained_logodds; deployment subtracts maximum alternative source score and deployment_retained_margin; legacy retained_prior_logodds is not used',
            stratification='one teacher proposal and one sampled alternative are deliberately evaluated per root; this 1:1 trial allocation is not the stochastic replacement probability or the greedy deployment policy',
            target_policy='importance-corrected actor surrogate uses softmax of deployment scores; sampling scores define only the alternative draw distribution',
            claim_boundary='current-action improvement surrogate under current greedy prefix/continuation; stable net benefit with collision cost, not zero harmed tasks')
        training.rollout = rollout
        training.event_examples = event_examples
        # The spawned initializer must install the config in its own process.
        training.initialize = initialize
        return training.train_main()
    contact.ContactEventConfig = CommonFutureConfig
    contact.ContactEventController = CommonFutureController
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'], common_future=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    contact.main()


if __name__ == '__main__':
    main()
