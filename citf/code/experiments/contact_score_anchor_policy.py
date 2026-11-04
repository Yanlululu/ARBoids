"""Regularize unobserved score promotion during current-action ranking.

Observed proposals retain the three-action common-future ranking objective.
The remaining proposals receive no outcome label. A soft penalty discourages
their largest increase relative to the retained action and collection-time
scores. All original proposals remain available to the unchanged controller.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch

from experiments import contact_tournament_policy as tournament

common = tournament.common
training = tournament.training
optimization_diagnostics = []


@dataclass(frozen=True)
class ScoreAnchorConfig(tournament.TournamentConfig):
    revision: str = 'CITF-common-future-unobserved-score-anchor-v1'
    unobserved_promotion_weight: float = .25


def initialize(project, source, config):
    training.ModelConfig = ScoreAnchorConfig
    common._initialize_training_worker(project, source, config)


@torch.no_grad()
def rollout(job):
    result = tournament.rollout(job)
    if job['training'] and result.get('completed') and result['transitions']:
        transition = result['transitions'][0]
        model = training.worker['model']
        batch = {k: torch.from_numpy(v[None]) for k, v in transition['inputs'].items()}
        # Inputs are the saved pre-action observation. The worker's weights
        # remain fixed throughout this rollout; no future outcome is an input.
        logits = model(batch)['logits'][0]
        relative = logits - logits[transition['retained_index']]
        transition['collection_relative_logits'] = relative.cpu().numpy().copy()
        transition['unobserved_promotion_weight'] = model.cfg.unobserved_promotion_weight
    return result


def event_examples(results):
    examples, missing = tournament.event_examples(results)
    index = 0
    for start in range(0, len(results), 6):
        if not results[start]['transitions']:
            continue
        reference = results[start]['transitions'][0]
        examples[index].update(
            collection_relative_logits=reference['collection_relative_logits'],
            unobserved_promotion_weight=reference['unobserved_promotion_weight'])
        index += 1
    return examples, missing


def anchored_objective(logits, examples):
    ranking_loss = tournament.tournament_objective(logits, examples)
    penalties, promotions = [], []
    for row, example in enumerate(examples):
        old = logits.new_tensor(example['collection_relative_logits'])
        unobserved = torch.tensor(example['inputs']['candidate_valid'],
                                  device=logits.device, dtype=torch.bool)
        unobserved[example['observed_candidate_indices']] = False
        relative = logits[row, :len(old)] - logits[row, example['teacher_index']]
        if unobserved.any():
            promotion = (relative[unobserved] - old[unobserved]).max().clamp_min(0.)
        else:
            promotion = relative[example['teacher_index']] * 0.
        promotions.append(promotion)
        penalties.append(example['unobserved_promotion_weight'] * promotion.square())
    penalty = torch.stack(penalties).mean()
    optimization_diagnostics.append(dict(
        ranking_loss=float(ranking_loss.detach()),
        promotion_penalty=float(penalty.detach()),
        maximum_unobserved_promotion=float(torch.stack(promotions).max().detach()),
        mean_unobserved_promotion=float(torch.stack(promotions).mean().detach()),
        roots=len(examples)))
    return ranking_loss + penalty


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        out = Path(sys.argv[sys.argv.index('--out') + 1])
        optimization_diagnostics.clear()
        training.ModelConfig = ScoreAnchorConfig
        training.COMPARISON_PROPOSALS = ('teacher', 'leading_alternative', 'sampled_alternative')
        training.initialize = initialize
        training.rollout = rollout
        training.event_examples = event_examples
        training.ACTOR_OBJECTIVE = anchored_objective
        training.PROTOCOL_OVERRIDES = dict(
            objective='three-action mean-return-weighted logistic ranking plus soft unobserved relative-score promotion penalty',
            score_anchor='collection-time candidate logit minus retained-action logit at the same saved pre-action observation',
            promotion_penalty='0.25 times squared positive maximum increase of an unobserved teacher-relative score; average over current roots',
            promotion_weight=ScoreAnchorConfig().unobserved_promotion_weight,
            unobserved_labels='no inferred return, failure or preference label; lowering unobserved scores is not penalized',
            action_support='all original joint candidates remain selectable online; no action mask, residual cap or new deployment margin',
            sampling='same retained, leading alternative and full-support exploratory alternative, each with two common futures',
            future_averaging='average actual returns by action before the unchanged observed-pair ranking loss',
            optimization='same current-root and physical-replay optimizer schedule; no historical task-comparison replay',
            budget='same320 scheduled root/prefix groups,80updates and1920realepisodes; score-anchor forward pass and gradients add compute',
            target='reduce unconstrained changes in the remaining deployment competitors while fitting observed action rankings',
            claim_boundary='soft update regularization,not evidence that an unobserved action is worse; no guaranteed net gain,zero-harm constraint,CV,critic or outcome prediction')
        result = training.train_main()
        if optimization_diagnostics:
            fields = ('ranking_loss', 'promotion_penalty',
                      'maximum_unobserved_promotion', 'mean_unobserved_promotion')
            diagnostic = dict(
                objective_calls=len(optimization_diagnostics),
                scope='pre-gradient diagnostics from optimizer calls; repeated roots are not independent observations',
                mean={k: float(np.mean([x[k] for x in optimization_diagnostics])) for k in fields},
                maximum={k: float(max(x[k] for x in optimization_diagnostics)) for k in fields},
                positive_promotion_calls=sum(x['maximum_unobserved_promotion'] > 1e-6
                                             for x in optimization_diagnostics))
            path = out / 'protocol.json'
            protocol = json.loads(path.read_text())
            protocol['score_anchor_optimization_diagnostic'] = diagnostic
            path.write_text(json.dumps(protocol, indent=2) + '\n')
            print(json.dumps(dict(score_anchor_optimization_diagnostic=diagnostic)), flush=True)
        return result
    common.CommonFutureConfig = ScoreAnchorConfig
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'],
            score_anchor=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    return common.main()


if __name__ == '__main__':
    main()
