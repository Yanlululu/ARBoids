"""Fit finite action preferences from the actual paired future comparisons.

Each observed future contributes a smooth preference between two current
actions. Averaging these preferences preserves disagreement between futures.
The actor remains a current-action ranker, with no complete-return head.
Unobserved candidates have no invented labels or score-promotion penalty.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from citf.train import atomic_save
from experiments import contact_actuation_policy as actuation
from experiments import contact_tournament_policy as tournament

contact = actuation.contact
common = actuation.common
training = actuation.training
optimization_diagnostics = []


@dataclass(frozen=True)
class SoftPreferenceConfig(actuation.ActuationConfig):
    revision: str = 'CITF-current-actuation-paired-soft-preference-v1'
    paired_return_temperature: float = .5
    unobserved_promotion_weight: float = 0.
    preference_target: str = 'mean sigmoid of actual paired return differences; not a calibrated success probability'


def initialize(project, source, config):
    contact.ContactEventModel = actuation.ActuationModel
    training.ModelConfig = SoftPreferenceConfig
    common._initialize_training_worker(project, source, config)


def event_examples(results):
    examples, missing = tournament.event_examples(results)
    index = 0
    for start in range(0, len(results), 6):
        group = results[start:start + 6]
        if not group[0]['transitions']:
            continue
        example = examples[index]
        by_action, future_seeds = {}, {}
        for trial in group:
            record = trial['record']
            action = trial['transitions'][0]['action']
            replica = record['future_replica']
            seed = record['branch_future_seed']
            if replica in future_seeds and future_seeds[replica] != seed:
                raise ValueError('Preferences require shared future seeds across actions')
            future_seeds[replica] = seed
            by_action.setdefault(action, {}).setdefault(replica, []).append(record['reward'])
        replicas = sorted(future_seeds)
        if len(replicas) != 2 or any(set(by_action[a]) != set(replicas)
                                    for a in example['observed_candidate_indices']):
            raise ValueError('Every observed action must have both common futures')
        example['observed_future_returns'] = [
            [float(np.mean(by_action[action][replica])) for replica in replicas]
            for action in example['observed_candidate_indices']]
        example['paired_return_temperature'] = SoftPreferenceConfig().paired_return_temperature
        index += 1
    return examples, missing


def soft_preference_objective(logits, examples):
    losses, preferences = [], []
    for row, example in enumerate(examples):
        indices = torch.tensor(example['observed_candidate_indices'], device=logits.device)
        returns = logits.new_tensor(example['observed_future_returns'])
        i, j = torch.triu_indices(len(indices), len(indices), 1, device=logits.device)
        if not len(i):
            losses.append(logits[row, example['teacher_index']] * 0.)
            continue
        paired_difference = returns[i] - returns[j]
        target = (paired_difference / example['paired_return_temperature']).sigmoid().mean(-1)
        weight = paired_difference.abs().mean(-1)
        gap = logits[row, indices[i]] - logits[row, indices[j]]
        # The total deployment score difference is fitted directly. A tie in
        # both real futures has zero weight, rather than a fabricated label.
        loss = F.binary_cross_entropy_with_logits(gap, target, reduction='none')
        losses.append((weight * loss).mean())
        preferences.extend(target.detach().cpu().tolist())
    result = torch.stack(losses).mean()
    optimization_diagnostics.append(dict(loss=float(result.detach()),
        observed_pairs=len(preferences),
        mean_target_confidence=float(np.mean([max(x, 1 - x) for x in preferences]))
            if preferences else None))
    return result


def prepare_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    value = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg = SoftPreferenceConfig(**dict(value['config'],
        revision=SoftPreferenceConfig().revision, unobserved_promotion_weight=0.))
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    value['config'] = asdict(cfg)
    value['protocol'] = dict(value['protocol'], soft_preference_initialization=dict(
        source=args.pretrained, parameters='all unchanged; same zero-output actuation initialization',
        change='replace hard ranking plus unobserved max-promotion penalty with finite paired preference loss',
        additional_pretraining=False))
    atomic_save(value, target)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'prepare':
        sys.argv.pop(1)
        return prepare_main()
    contact.ContactEventModel = actuation.ActuationModel
    if mode == 'train':
        sys.argv.pop(1)
        out = Path(sys.argv[sys.argv.index('--out') + 1])
        optimization_diagnostics.clear()
        training.ModelConfig = SoftPreferenceConfig
        training.COMPARISON_PROPOSALS = ('teacher', 'leading_alternative', 'sampled_alternative')
        training.initialize = initialize
        training.rollout = tournament.rollout
        training.event_examples = event_examples
        training.ACTOR_OBJECTIVE = soft_preference_objective
        training.PROTOCOL_OVERRIDES = dict(
            objective='observed-pair cross entropy with mean sigmoid paired-return targets',
            target='mean_f sigmoid((R_i_f-R_j_f)/0.5), using the same two futures for each observed action',
            pair_weight='mean_f absolute(R_i_f-R_j_f); average over distinct observed pairs and then roots',
            preference_temperature=SoftPreferenceConfig().paired_return_temperature,
            target_interpretation='smooth utility preference, not a calibrated probability of benefit or success',
            update_change='finite observed preferences replace hard-direction ranking and the unobserved max-promotion penalty together',
            unobserved_candidates='no labels or loss on unobserved candidates; all remain available at deployment',
            unchanged='same actuation readout, inherited response representation, teacher, all-candidate bank, sampling and1.5deployment margin',
            budget='same320root schedule,80updates,1920realepisodes,477optimizer and physical-replay batches; no collection-time anchor forward pass',
            future_averaging='average smooth paired-future preferences before updating; conflicting effects can cancel',
            selection='unchanged original64 validation and fixed common768 endpoint',
            claim_boundary='combined update-rule experiment, not isolated attribution to soft labels; no critic,return head,CV or guarantee from two futures')
        result = training.train_main()
        if optimization_diagnostics:
            path = out / 'protocol.json'
            protocol = json.loads(path.read_text())
            confidence = [x['mean_target_confidence'] for x in optimization_diagnostics
                          if x['mean_target_confidence'] is not None]
            protocol['soft_preference_optimization'] = dict(
                objective_calls=len(optimization_diagnostics),
                mean_loss=float(np.mean([x['loss'] for x in optimization_diagnostics])),
                mean_target_confidence=float(np.mean(confidence)) if confidence else None,
                scope='optimizer calls in this process segment; reused comparisons are not independent observations',
                after_update=protocol.get('resumptions', [{}])[-1].get('update', 0))
            path.write_text(json.dumps(protocol, indent=2) + '\n')
        return result
    common.CommonFutureConfig = SoftPreferenceConfig
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'],
            soft_preference=file_hash(__file__), actuation_readout=file_hash(actuation.__file__)))
        return value

    evaluation.runtime_identity = identity
    return common.main()


if __name__ == '__main__':
    main()
