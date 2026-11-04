"""Train the retained-relative ordering used by full-bank greedy deployment.

Actual complete returns under common futures supply ranking weights. There is
no fitted return head. A full-support alternative sampler remains independent
of the deployment score, and its propensity corrects the candidate-pair loss.
"""
from dataclasses import dataclass
import sys

import torch
from torch.nn import functional as F

from experiments import contact_common_future_policy as common

training = common.training


@dataclass(frozen=True)
class RankConfig(common.CommonFutureConfig):
    revision: str = 'CITF-full-bank-common-future-task-ranking-v1'
    decision_domain: str = 'all original joint proposals; retained-relative greedy ranking'


def initialize(project, source, config):
    training.ModelConfig = RankConfig
    common._initialize_training_worker(project, source, config)


def event_examples(results):
    examples, missing = common.event_examples(results)
    index = 0
    for start in range(0, len(results), 4):
        group = results[start:start + 4]
        if not group[0]['transitions']:
            continue
        transition = group[0]['transitions'][0]
        examples[index].update(
            teacher_index=transition['retained_index'],
            alternative_index=transition['sampled_alternative'],
            sampling_probability=transition['alternative_sampling_probability'],
            alternative_count=int(transition['inputs']['candidate_valid'].sum()) - 1)
        index += 1
    return examples, missing


def ranking_objective(logits, examples):
    rows = torch.arange(len(examples), device=logits.device)
    teacher = torch.tensor([x['teacher_index'] for x in examples], device=logits.device)
    alternative = torch.tensor([x['alternative_index'] for x in examples], device=logits.device)
    gap = logits[rows, alternative] - logits[rows, teacher]
    difference = logits.new_tensor([x['comparison_return'] - x['teacher_return'] for x in examples])
    # q has at least 10% uniform alternative mass, so these weights are <=10.
    importance = logits.new_tensor([
        1. / (x['alternative_count'] * x['sampling_probability'])
        if x['alternative_count'] and x['alternative_index'] != x['teacher_index'] else 0.
        for x in examples])
    return (importance * difference.abs() * F.softplus(-difference.sign() * gap)).mean()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        training.ModelConfig = RankConfig
        training.COMPARISON_PROPOSALS = ('teacher', 'sampled_alternative')
        training.initialize = initialize
        training.rollout = common.rollout
        training.event_examples = event_examples
        training.ACTOR_OBJECTIVE = ranking_objective
        training.PROTOCOL_OVERRIDES = dict(
            objective='mean-return-weighted retained-relative logistic ranking loss; inverse candidate-pair sampling propensity',
            action_support='all original joint proposals at sampling and deployment',
            sampling='one teacher and one full-support alternative, each under two common futures; same root and future schedule as v2',
            future_averaging='average actual complete returns per action before forming the ranking weight and sign',
            importance='1/(K*q(a)) targets a uniform mean over the K available alternative comparisons at each sampled root; q includes10percent uniform mass',
            optimization='same optimizer and physical replay count schedule; no actor entropy term or PPO clipping; exploration remains in the separate full-support sampler',
            kl_diagnostic='reported KL compares full-bank softmax only for monitoring; it is not the optimized ranking loss',
            target='current greedy deployment compares candidate scores directly with the retained score; ranking also uses that same pairwise difference',
            claim_boundary='sampled task-weighted ranking objective under current greedy continuation; not an unbiased complete-trajectory policy gradient and not a fitted CV, critic, or terminal-return model')
        return training.train_main()
    return common.main()


if __name__ == '__main__':
    main()
