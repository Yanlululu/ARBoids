"""One KL-calibrated initialization, followed by a bounded training-only comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from parallel_mappo import ParallelRollouts, summarize
from policy.mappo import PredictiveMAPPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-seed', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.first_seed < 100000 or args.episodes < 1 or not 1 <= args.workers <= 6:
        parser.error('Use ordinary training seeds, positive episodes and at most six workers.')
    if args.output.exists():
        parser.error('Do not overwrite a candidate checkpoint.')
    torch.set_num_threads(1)
    agent = PredictiveMAPPO.from_checkpoint(args.checkpoint)
    if agent.settings.compatibility_prior or not agent.settings.theta_space_gate:
        parser.error('Start from a theta-space checkpoint without the physical prior.')
    reference = PredictiveMAPPO.from_checkpoint(args.checkpoint)
    agility = float(agent.config.get('training', {}).get('agility', 2.))
    with ParallelRollouts(agent.config, args.workers, 4) as pool:
        rollout = pool.run(agent, np.arange(args.first_seed, args.first_seed+args.episodes), agility)
        print('[CALIBRATION_BATCH] '+json.dumps(summarize(rollout.summaries)), flush=True)
        initialization = agent.calibrate_compatibility_prior(rollout)
        del rollout
        print('[INITIALIZATION] '+json.dumps(initialization), flush=True)
        comparison_start = args.first_seed+max(10000, args.episodes)
        seeds = np.arange(comparison_start, comparison_start+args.episodes)
        before = pool.run(reference, seeds, agility, deterministic=True, retain=False)
        after = pool.run(agent, seeds, agility, deterministic=True, retain=False)
    a, b = summarize(before), summarize(after)
    passed = (b['success_rate'] >= a['success_rate'] and b['collision_rate'] < a['collision_rate']
              and b['breach_rate'] <= a['breach_rate'])
    result = dict(before=a, after=b, local_improvement=passed, initialization=initialization,
        calibration_first_seed=args.first_seed, comparison_first_seed=comparison_start, episodes=args.episodes,
        collision_budget_met=b['collision_rate'] <= agent.settings.collision_budget,
        performance_qualified=False,
        paired={event: dict(resolved=sum(x[event] and not y[event] for x,y in zip(before,after)),
                            introduced=sum(y[event] and not x[event] for x,y in zip(before,after)))
                for event in ('collision','breach')})
    print('[DECISION] '+json.dumps(result), flush=True)
    if passed:
        agent.training_state.update(performance_qualified=False, stable_count=0,
            prior_calibration_training_comparison=result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        agent.save(args.output)
        print('[CANDIDATE] '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
