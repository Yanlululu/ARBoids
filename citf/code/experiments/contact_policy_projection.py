"""Finite task-policy improvement over the actual compared joint actions.

The three observed actions redistribute their existing probability mass by a
relative-entropy policy-improvement step. Unobserved actions retain their old
probabilities in the target; they receive no invented return label. The actor
fits this complete candidate distribution with KL, using actual common-future
returns only to construct a detached policy target, never a value prediction.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch

from citf.train import atomic_save
from experiments import contact_actuation_policy as actuation
from experiments import contact_tournament_policy as tournament

contact = actuation.contact
common = tournament.common
training = tournament.training
projection_diagnostics = []


@dataclass(frozen=True)
class ProjectionConfig(actuation.ActuationConfig):
    revision: str = 'CITF-contact-observed-policy-projection-v1'
    policy_projection_temperature: float = .5
    unobserved_promotion_weight: float = 0.
    policy_update: str = 'finite relative-entropy improvement on observed actions, then full-bank KL projection'


def initialize(project, source, config):
    contact.ContactEventModel = actuation.ActuationModel
    training.ModelConfig = ProjectionConfig
    common._initialize_training_worker(project, source, config)


@torch.no_grad()
def rollout(job):
    result = tournament.rollout(job)
    if job['training'] and result.get('completed') and result['transitions']:
        result['transitions'][0]['policy_projection_temperature'] = (
            training.worker['model'].cfg.policy_projection_temperature)
    return result


def event_examples(results):
    examples, missing = tournament.event_examples(results)
    index = 0
    for start in range(0, len(results), 6):
        if not results[start]['transitions']:
            continue
        examples[index]['policy_projection_temperature'] = (
            results[start]['transitions'][0]['policy_projection_temperature'])
        index += 1
    return examples, missing


def projection_objective(logits, examples):
    losses, shifts, greedy_changes, observed_mass = [], [], [], []
    for row, example in enumerate(examples):
        count = len(example['probabilities'])
        valid = torch.as_tensor(example['inputs']['candidate_valid'],
                                dtype=torch.bool, device=logits.device)
        observed = torch.as_tensor(example['observed_candidate_indices'],
                                   dtype=torch.long, device=logits.device)
        old = logits.new_tensor(example['probabilities'])
        old = old.masked_fill(~valid, 0.)
        old = old / old.sum()
        temperature = example['policy_projection_temperature']
        if temperature <= 0:
            raise ValueError('Policy projection temperature must be positive')
        mass = old[observed].sum()
        returns = logits.new_tensor(example['observed_returns'])
        # This solves the finite relative-entropy improvement problem on the
        # observed simplex. Subtracting max(return) only stabilizes softmax.
        improved = (old[observed].clamp_min(1e-12).log()
                    + (returns - returns.max()) / temperature).softmax(-1)
        target = old.clone()
        target[observed] = mass * improved
        target = target.detach()
        log_policy = logits[row, :count][valid].log_softmax(-1)
        target_valid = target[valid]
        kl = (target_valid * (target_valid.clamp_min(1e-12).log() - log_policy)).sum()
        # Normalize by queried mass so a larger unqueried bank does not dilute
        # the strength of the same observed comparison. The optimum is unchanged.
        losses.append(kl / mass.clamp_min(1e-12))
        shifts.append(float((target - old).abs().sum()))
        greedy_changes.append(int(target.argmax() != old.argmax()))
        observed_mass.append(float(mass))
    loss = torch.stack(losses).mean()
    projection_diagnostics.append(dict(loss=float(loss.detach()), roots=len(examples),
        target_l1_mean=float(np.mean(shifts)), target_greedy_changes=sum(greedy_changes),
        queried_mass_mean=float(np.mean(observed_mass))))
    return loss


def initialize_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    value = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg = ProjectionConfig(**dict(value['config'], revision=ProjectionConfig().revision,
        unobserved_promotion_weight=0.,
        policy_projection_temperature=ProjectionConfig().policy_projection_temperature,
        policy_update=ProjectionConfig().policy_update))
    protocol = dict(value['protocol'], policy_projection_initialization=dict(
        source=args.pretrained, weights='unchanged actuation-readout initialization',
        additional_physical_pretraining=0, temperature=cfg.policy_projection_temperature,
        objective='observed-action relative-entropy improvement and full-candidate KL projection',
        unobserved_target='old collection-time probabilities, without outcome labels',
        removed='weighted unbounded pairwise ranking and maximum unobserved-promotion penalty'))
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(config=asdict(cfg), state=value['state'], protocol=protocol,
        seed=value['seed'], update=0, training_episodes=0), target)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'initialize':
        sys.argv.pop(1)
        return initialize_main()
    contact.ContactEventModel = actuation.ActuationModel
    if mode == 'train':
        sys.argv.pop(1)
        out = Path(sys.argv[sys.argv.index('--out') + 1])
        projection_diagnostics.clear()
        training.ModelConfig = ProjectionConfig
        training.COMPARISON_PROPOSALS = ('teacher', 'leading_alternative', 'sampled_alternative')
        training.initialize = initialize
        training.rollout = rollout
        training.event_examples = event_examples
        training.ACTOR_OBJECTIVE = projection_objective
        training.PROTOCOL_OVERRIDES = dict(
            objective=ProjectionConfig().policy_update,
            policy_target='q(a)=m*softmax(log pi_old(a)+mean_actual_return(a)/0.5) for observed actions; q(a)=pi_old(a) otherwise',
            observed_mass='m is the total collection-time deployment probability of the observed distinct actions',
            projection='KL(q||pi_theta) divided by m, averaged across current roots; detached targets are fixed during the update',
            temperature=ProjectionConfig().policy_projection_temperature,
            initialization='identical zero-output actuation readout and inherited response encoder; no additional physical pretraining',
            unobserved_labels='no return or preference target; preserve their collection-time policy probabilities in the projection target',
            action_support='all original proposals remain selectable; no action mask or new deployment margin',
            sampling='unchanged retained, leading alternative and full-support exploratory alternative, each with two common futures',
            optimization='same current-root optimizer and physical-replay schedule; no historical task replay, entropy term or maximum-promotion penalty',
            budget='same320 root schedule,80updates,1920realepisodes,477optimizer/replay schedule and original64model selection',
            claim_boundary='finite sampled policy-improvement target under collection-time greedy continuation; noisy returns remain; no guaranteed global improvement,CV,critic or terminal-value prediction')
        result = training.train_main()
        if projection_diagnostics:
            diagnostic = dict(calls=len(projection_diagnostics),
                scope='optimizer-call diagnostics; repeated roots are not independent observations',
                mean_loss=float(np.mean([x['loss'] for x in projection_diagnostics])),
                mean_target_l1=float(np.mean([x['target_l1_mean'] for x in projection_diagnostics])),
                mean_queried_mass=float(np.mean([x['queried_mass_mean'] for x in projection_diagnostics])),
                target_greedy_changes=sum(x['target_greedy_changes'] for x in projection_diagnostics),
                repeated_root_count=sum(x['roots'] for x in projection_diagnostics))
            path = out / 'protocol.json'
            protocol = json.loads(path.read_text())
            protocol['policy_projection_optimization'] = diagnostic
            path.write_text(json.dumps(protocol, indent=2) + '\n')
            print(json.dumps(dict(policy_projection_optimization=diagnostic)), flush=True)
        return result
    common.CommonFutureConfig = ProjectionConfig
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'],
            actuation_readout=file_hash(actuation.__file__), policy_projection=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    return common.main()


if __name__ == '__main__':
    main()
