"""Bounded paired development comparison using training scenarios only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from policy.mappo import PredictiveMAPPO
from policy.interaction_prediction import InteractionPredictor, PredictionConfig
from parallel_mappo import ParallelRollouts, summarize
from train_mappo_performance import write_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--points', type=int, choices=(3,5,9), default=5)
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--task-prediction', action='store_true')
    parser.add_argument('--terminal-prediction', action='store_true',
                        help='Use buffered nominal capture forecasts to shorten the collision window')
    parser.add_argument('--task-penalty', type=float,
                        help='Explicit weight for preserving predicted interception opportunities')
    parser.add_argument('--gain', type=float, default=1.)
    parser.add_argument('--margin', type=float, default=7.)
    parser.add_argument('--priority-temperature', type=float,
                        help='Distance scale of the existing nearest-interceptor priority')
    parser.add_argument('--horizon', type=float)
    parser.add_argument('--first-seed', type=int, default=2400000)
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--reference-rows', type=Path,
                        help='Reuse a verified comparison of the same source checkpoint and training seeds')
    args = parser.parse_args()
    if args.first_seed < 100000 or args.episodes < 1 or args.output_dir.exists():
        parser.error('Use training seeds, a positive episode count and a new output directory.')
    if not math.isfinite(args.gain) or not 0 < args.gain < 100 or args.workers < 1:
        parser.error('Use a finite gain in (0,100) and positive worker count.')
    if args.task_penalty is not None and (not args.task_prediction or
            not math.isfinite(args.task_penalty) or args.task_penalty <= 0):
        parser.error('A positive finite task penalty requires task prediction.')
    torch.set_num_threads(1)
    before = PredictiveMAPPO.from_checkpoint(args.checkpoint)
    after = PredictiveMAPPO.from_checkpoint(args.checkpoint)
    after.enable_mixture_compatibility(args.points, args.joint, args.task_prediction, args.terminal_prediction)
    after.settings.compatibility_margin = after.actor.compatibility_margin = args.margin
    after.config['mappo']['compatibility_margin'] = args.margin
    if args.priority_temperature is not None:
        after.settings.compatibility_priority_temperature = args.priority_temperature
        after.actor.compatibility_priority_temperature = args.priority_temperature
        after.config['mappo']['compatibility_priority_temperature'] = args.priority_temperature
    after.settings.compatibility_gain = after.config['mappo']['compatibility_gain'] = args.gain
    if args.task_penalty is not None:
        after.settings.compatibility_task_penalty = after.actor.compatibility_task_penalty = args.task_penalty
        after.config['mappo']['compatibility_task_penalty'] = args.task_penalty
    if args.horizon is not None:
        after.config['prediction']['horizon'] = args.horizon
        after.prediction_config = PredictionConfig(**after.config['prediction'])
        after.predictor = InteractionPredictor(after.prediction_config)
        after.actor.prediction_horizon = after.prediction_config.horizon
    with torch.no_grad():
        after.actor.compatibility_gain_raw.fill_(math.log(math.expm1(args.gain)))
    after.actor_optimizer.state.pop(after.actor.compatibility_gain_raw, None)
    after.settings.__post_init__()
    args.output_dir.mkdir(parents=True)
    candidate = args.output_dir/'candidate.pth'
    after.save(candidate)
    meta = dict(episodes=args.episodes, first_seed=args.first_seed, source=str(args.checkpoint.resolve()),
        source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        purpose='training_scenario_probe', performance_qualified=False, extra_optimizer_updates=0,
        joint=args.joint, task_prediction=args.task_prediction, points=args.points, margin=args.margin, gain=args.gain,
        terminal_prediction=args.terminal_prediction, terminal_capture_radius=after.prediction_config.terminal_capture_radius,
        terminal_buffer=after.prediction_config.terminal_buffer,
        horizon=after.prediction_config.horizon, task_penalty=after.settings.compatibility_task_penalty)
    meta['priority_temperature'] = after.settings.compatibility_priority_temperature
    seeds = np.arange(args.first_seed, args.first_seed+args.episodes)
    cached_reference = None
    if args.reference_rows is not None:
        reference_meta = json.loads(args.reference_rows.with_suffix('.json').read_text(encoding='utf-8'))
        if reference_meta['source_sha256'] != meta['source_sha256']:
            raise ValueError('The cached reference belongs to another source checkpoint.')
        with args.reference_rows.open(newline='', encoding='utf-8') as file:
            cached_reference = [{key:float(value) for key,value in row.items()} for row in csv.DictReader(file)]
        if [row['seed'] for row in cached_reference] != seeds.tolist():
            raise ValueError('The cached reference must contain exactly the same ordered training seeds.')
        meta['reference_rows'] = str(args.reference_rows.resolve())
    start = time.monotonic()
    with ParallelRollouts(after.config, args.workers) as pool:
        left = (cached_reference if cached_reference is not None else
                pool.run(before, seeds, 2., deterministic=True, retain=False))
        print('[REFERENCE] '+json.dumps(summarize(left)), flush=True)
        if cached_reference is None:
            write_evaluation(args.output_dir, 'training-reference', summarize(left), left, **meta)
        right = pool.run(after, seeds, 2., deterministic=True, retain=False)
        print('[CANDIDATE] '+json.dumps(summarize(right)), flush=True)
    write_evaluation(args.output_dir, 'training-candidate', summarize(right), right,
                     wall_seconds=time.monotonic()-start, **meta)
    print('[PAIRED] '+json.dumps({event: {f'{x}{y}': sum(a[event]==x and b[event]==y for a,b in zip(left,right))
        for x in (0,1) for y in (0,1)} for event in ('success','collision','breach')}), flush=True)


if __name__ == '__main__':
    main()
