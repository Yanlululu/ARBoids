"""Retain observed task comparisons while learning current-event ordering.

Historical roots are sampled uniformly, including equal and adverse outcomes.
Their labels describe the actual continuation at collection time. The online
controller and its complete joint candidate domain stay unchanged.
"""
from dataclasses import dataclass
import sys

from experiments import contact_rank_policy as ranking

common = ranking.common
training = ranking.training


@dataclass(frozen=True)
class RankReplayConfig(ranking.RankConfig):
    revision: str = 'CITF-full-bank-common-future-task-ranking-replay-v1'


def initialize(project, source, config):
    training.ModelConfig = RankReplayConfig
    common._initialize_training_worker(project, source, config)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        training.ModelConfig = RankReplayConfig
        training.COMPARISON_PROPOSALS = ('teacher', 'sampled_alternative')
        training.initialize = initialize
        training.rollout = common.rollout
        training.event_examples = ranking.event_examples
        training.ACTOR_OBJECTIVE = ranking.ranking_objective
        training.PROTOCOL_OVERRIDES = dict(
            objective='mean-return-weighted retained-relative logistic ranking; current roots mixed with uniform historical task-comparison replay',
            action_support='all original joint proposals at sampling and deployment',
            sampling='teacher and one full-support alternative under two common futures; unchanged root and future schedule',
            future_averaging='mean actual complete return per action before ranking weight and sign',
            importance='each stored comparison retains its original 1/(K*q(a)) candidate-sampling weight',
            optimization='same optimizer and physical replay schedule; additional task-comparison examples use an independent RNG',
            target='same retained-relative score difference used by greedy deployment',
            claim_boundary='historical preference labels use their collection-time continuation; no assertion of an unbiased current-policy gradient or of solved forgetting')
        return training.train_main()
    common.CommonFutureConfig = RankReplayConfig
    return common.main()


if __name__ == '__main__':
    main()
