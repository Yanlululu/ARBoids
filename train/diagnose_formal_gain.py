"""Measure learned-gain behavior and residual failures on development scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from envs.TADgame import TADEnv
from policy.interaction_prediction import InteractionPredictor, PredictionConfig
from policy.mappo import PredictiveMAPPO


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantiles(values):
    if not len(values):
        return None
    return dict(zip(('minimum', 'p05', 'median', 'p95', 'maximum'),
                    map(float, np.quantile(values, [0., .05, .5, .95, 1.]))))


def grid_clearance(edges, gates, points, distance_scale):
    """The actor's bilinear nominal-distance lookup, not an event label."""
    grid = np.linspace(0., 1., points)
    weights = np.maximum(0., 1. - np.abs(np.clip(gates, 0., 1.)[:, None] - grid) * (points - 1))
    n = len(gates)
    table = edges[..., 18:18 + points * points].reshape(n, n, points, points) * distance_scale
    distances = np.einsum('ip,ijpq,jq->ij', weights, table, weights)
    return float(distances[np.triu_indices(n, 1)].min())


class GainObserver:
    def __init__(self, actor):
        self.actor = actor
        self.original_signal = actor.compatibility_signal
        self.anchor = self.suggestion = self.gain = None
        self.context_hook = actor.compatibility_context_head.register_forward_hook(self.context)
        actor.compatibility_signal = self.signal

    def context(self, module, inputs, output):
        gain = torch.nn.functional.softplus(self.actor.compatibility_gain_raw) * (2. * output.sigmoid())
        self.gain = gain.detach().cpu().numpy()[0, :, 0].copy()

    def signal(self, edges, mask, obs=None, anchor=None):
        value = self.original_signal(edges, mask, obs, anchor)
        self.anchor = anchor.detach().cpu().numpy()[0, :, 0].copy()
        self.suggestion = self.anchor + value.detach().cpu().numpy()[0, :, 0]
        return value

    def close(self):
        self.context_hook.remove()
        self.actor.compatibility_signal = self.original_signal


def configure_candidate(agent, margin=None, horizon=None, terminal_prediction=False):
    if terminal_prediction:
        agent.enable_mixture_compatibility(agent.settings.compatibility_mixture_points,
            joint=True, task_prediction=agent.settings.compatibility_task_prediction, terminal_prediction=True)
    if margin is not None:
        agent.settings.compatibility_margin = agent.actor.compatibility_margin = margin
        agent.config['mappo']['compatibility_margin'] = margin
    if horizon is not None:
        agent.config['prediction']['horizon'] = horizon
        agent.prediction_config = PredictionConfig(**agent.config['prediction'])
        agent.predictor = InteractionPredictor(agent.prediction_config)
        agent.actor.prediction_horizon = horizon


def policy_configuration(agent):
    return dict(horizon_seconds=agent.prediction_config.horizon,
        prediction_dt_seconds=agent.prediction_config.dt,
        terminal_prediction=agent.settings.compatibility_terminal_prediction,
        terminal_capture_radius_metres=agent.prediction_config.terminal_capture_radius,
        terminal_buffer_seconds=agent.prediction_config.terminal_buffer,
        nominal_margin_metres=agent.actor.compatibility_margin,
        mixture_points=agent.actor.compatibility_mixture_points)


def run_chunk(arguments):
    checkpoint, seeds, margin, horizon, terminal_prediction = arguments
    torch.set_num_threads(1)
    agent = PredictiveMAPPO.from_checkpoint(checkpoint, 'cpu')
    if not (agent.settings.compatibility_joint_mixture and agent.settings.compatibility_context_gain):
        raise ValueError('This diagnosis requires the full joint-prediction, contextual-gain model.')
    configure_candidate(agent, margin, horizon, terminal_prediction)
    agent.actor.eval()
    env = TADEnv(3, True, True, protocol='paper-parameters-v1', total_time=60.)
    observer = None
    episodes, gains, corrections, failures = [], [], [], []
    counts = dict(decisions=0, active_correction_decisions=0, original_nominal_collision=0,
                  learned_nominal_collision=0, suggested_nominal_collision=0,
                  gain_loses_nominal_clearance=0, predicted_safe_next_step_collisions=0)
    try:
        for index, seed in enumerate(seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)
            observation, _ = env.reset(2., noisy_agility=False)
            if observer is None:
                inputs = (observation, env.prediction_snapshot(), env.thrust_to_action(env.boids_actions), env.Current_T)
                expected = agent.act(*inputs, deterministic=True)[0]
                observer = GainObserver(agent.actor)
                actual = agent.act(*inputs, deterministic=True)[0]
                if not np.array_equal(expected, actual):
                    raise RuntimeError('Observation instrumentation changed the policy action.')
            collision = False
            recent = []
            for step in range(300):
                action, record = agent.act(observation, env.prediction_snapshot(),
                    env.thrust_to_action(env.boids_actions), env.Current_T, deterministic=True)
                if not np.isfinite(action).all():
                    raise FloatingPointError('Non-finite action.')
                points = agent.actor.compatibility_mixture_points
                scale = agent.actor.prediction_distance_scale
                anchor_distance = grid_clearance(record['edges'], observer.anchor, points, scale)
                suggested_distance = grid_clearance(record['edges'], observer.suggestion, points, scale)
                actual_distance = grid_clearance(record['edges'], action[:, 2], points, scale)
                delta = np.abs(action[:, 2] - np.clip(observer.anchor, 0., 1.))
                counts['decisions'] += 1
                counts['active_correction_decisions'] += int(delta.max() > 1e-5)
                counts['original_nominal_collision'] += int(anchor_distance <= 5.)
                counts['learned_nominal_collision'] += int(actual_distance <= 5.)
                counts['suggested_nominal_collision'] += int(suggested_distance <= 5.)
                counts['gain_loses_nominal_clearance'] += int(actual_distance <= 5. < suggested_distance)
                gains.extend(observer.gain.tolist())
                corrections.extend(delta.tolist())
                observation, _, outcome, _ = env.step(action, 'AdaRes')
                events = env.physical_events()
                collision |= events['collision']
                counts['predicted_safe_next_step_collisions'] += int(events['collision'] and actual_distance > 5.)
                recent.append(dict(step=step, original_nominal_clearance=anchor_distance,
                    suggested_nominal_clearance=suggested_distance, executed_nominal_clearance=actual_distance,
                    gain=observer.gain.tolist(), maximum_gate_change=float(delta.max())))
                if outcome:
                    break
            else:
                raise RuntimeError('Episode exceeded the unchanged 300-decision horizon.')
            row = dict(seed=seed, success=int(outcome > 2), collision=int(collision),
                       breach=int(events['breach']), outcome_code=int(outcome), steps=step + 1)
            episodes.append(row)
            if not row['success']:
                failures.append(dict(**row, final_decisions=recent[-10:]))
            if (index + 1) % 16 == 0:
                print(json.dumps(dict(chunk_first_seed=seeds[0], completed=index + 1, total=len(seeds),
                    success=sum(r['success'] for r in episodes), collision=sum(r['collision'] for r in episodes),
                    breach=sum(r['breach'] for r in episodes))), flush=True)
    finally:
        if observer is not None:
            observer.close()
    return dict(episodes=episodes, gains=gains, corrections=corrections, failures=failures, counts=counts,
                effective_policy=policy_configuration(agent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-seed', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--margin', type=float, help='Optional full-model development intervention, in metres.')
    parser.add_argument('--horizon', type=float, help='Optional nominal forecast horizon, in seconds.')
    parser.add_argument('--terminal-prediction', action='store_true',
                        help='Use the existing buffered nominal-capture cutoff for joint risk.')
    args = parser.parse_args()
    if not (900000000 <= args.first_seed < args.first_seed + args.episodes < 2**31):
        parser.error('Use the development seed block starting at 900000000.')
    if not 1 <= args.episodes <= 1024 or not 1 <= args.workers <= 4 or args.output.exists():
        parser.error('Use 1-1024 episodes, 1-4 workers, and a new result directory.')
    if args.margin is not None and (not np.isfinite(args.margin) or args.margin <= 5.):
        parser.error('The optional nominal margin must be finite and above the actual 5 m collision threshold.')
    if args.horizon is not None and (not np.isfinite(args.horizon) or args.horizon <= 0
                                    or not np.isclose(args.horizon / .05, round(args.horizon / .05))):
        parser.error('Use a positive forecast horizon with a whole number of 0.05 s steps.')
    source_hash = digest(args.checkpoint)
    groups = [list(map(int, group)) for group in np.array_split(
        np.arange(args.first_seed, args.first_seed + args.episodes), args.workers) if len(group)]
    started = time.monotonic()
    arguments = [(str(args.checkpoint.resolve()), group, args.margin, args.horizon,
                  args.terminal_prediction) for group in groups]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        chunks = list(pool.map(run_chunk, arguments))
    rows = sorted([row for chunk in chunks for row in chunk['episodes']], key=lambda row: row['seed'])
    if [row['seed'] for row in rows] != list(range(args.first_seed, args.first_seed + args.episodes)):
        raise RuntimeError('Missing or duplicated development episodes.')
    if digest(args.checkpoint) != source_hash:
        raise RuntimeError('Source checkpoint changed during diagnosis.')
    if not all(chunk['effective_policy'] == chunks[0]['effective_policy'] for chunk in chunks):
        raise RuntimeError('Workers evaluated different policy configurations.')
    counters = {key: sum(chunk['counts'][key] for chunk in chunks) for key in chunks[0]['counts']}
    summary = dict(completed=True, passed=True, purpose='full_model_development_diagnosis',
        first_seed=args.first_seed, episodes=len(rows), checkpoint_sha256=source_hash,
        intervention_nominal_margin=args.margin, intervention_horizon=args.horizon,
        intervention_terminal_prediction=args.terminal_prediction, actor_updated=False,
        effective_policy=chunks[0]['effective_policy'],
        observer_action_identity_verified=True, wall_seconds=time.monotonic() - started,
        success_count=sum(row['success'] for row in rows), collision_count=sum(row['collision'] for row in rows),
        breach_count=sum(row['breach'] for row in rows), counters=counters,
        effective_gain=quantiles([gain for chunk in chunks for gain in chunk['gains']]),
        absolute_gate_correction=quantiles([value for chunk in chunks for value in chunk['corrections']]),
        failure_cases=[failure for chunk in chunks for failure in chunk['failures']],
        nominal_distance_scope='Full-horizon bilinear grid distances, without a terminal cutoff; actual outcomes come from the environment.')
    args.output.mkdir(parents=True)
    with (args.output / 'episodes.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'failure_cases'}), flush=True)


if __name__ == '__main__':
    main()
