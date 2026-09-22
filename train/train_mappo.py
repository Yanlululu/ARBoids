"""Train prediction-conditioned, two-stage MAPPO on complete team episodes."""
import argparse
from contextlib import contextmanager
import copy
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

from envs.mappo_env import MAPPOEnv
from policy.mappo import PredictiveMAPPO
from policy.rollout_buffer import RolloutBuffer
from utils.manager import ExperimentManager, set_seed

ROOT = Path(__file__).resolve().parent


def make_env(config, duration=None):
    options = dict(config.get('environment', {}))
    if duration is not None:
        options['total_time'] = duration
    agent = config['agent']
    return MAPPOEnv(defender_num=agent['defender_num'], boid_state=agent.get('boid_state', True),
                    form_reward=agent.get('form_reward', True), **options)


def play_episode(agent, env, agility, noisy_agility=False, deterministic=False):
    observation = env.reset(agility, noisy_agility)
    transitions, gates = [], []
    max_steps = int(np.ceil(env.env.Total_T / env.env.Action_T)) + 1
    agent.actor.eval()
    agent.critic.eval()
    for _ in range(max_steps):
        state = env.env.centralized_state()
        value_r, value_c = agent.values(state)
        action, record = agent.act(observation, env.env.prediction_snapshot(),
                                   env.env.thrust_to_action(env.env.boids_actions),
                                   env.env.Current_T, deterministic)
        step = env.step(action)
        record.update(state=state, value_r=value_r, value_c=value_c,
                      reward=step.reward, cost=step.cost, terminated=step.terminated)
        transitions.append(record)
        gates.extend(action[:, 2].tolist())
        observation = step.observation
        if step.terminated:
            summary = dict(success=int(step.outcome > 2), capture=int(step.outcome == 3),
                           timeout=int(step.outcome == 4), breach=int(step.events['breach']),
                           collision=int(env.collision_seen), outcome_code=step.outcome,
                           steps=len(transitions), task_return=sum(t['reward'] for t in transitions),
                           gate_mean=float(np.mean(gates)))
            return transitions, summary
    raise RuntimeError('Environment exceeded its finite task horizon.')


def collect_episodes(agent, env, episodes, agility, noisy_agility=False):
    if episodes <= 0:
        raise ValueError('At least one complete episode is required.')
    rollout = RolloutBuffer()
    for _ in range(episodes):
        transitions, summary = play_episode(agent, env, agility, noisy_agility)
        rollout.add_episode(transitions, summary)
    return rollout


@contextmanager
def evaluation_rng():
    """Evaluation must not change subsequent training resets/exploration."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate(agent, episodes=20, first_seed=10000, agility=2., duration=None):
    if episodes <= 0:
        raise ValueError('Evaluation episodes must be positive.')
    rows = []
    with evaluation_rng():
        env = make_env(agent.config, duration)
        for episode in range(episodes):
            seed = first_seed + episode
            np.random.seed(seed)
            _, row = play_episode(agent, env, agility, deterministic=True)
            rows.append(dict(episode=episode, seed=seed, **row))
    metrics = dict(success_rate=float(np.mean([r['success'] for r in rows])),
                   collision_rate=float(np.mean([r['collision'] for r in rows])),
                   capture_rate=float(np.mean([r['capture'] for r in rows])),
                   timeout_rate=float(np.mean([r['timeout'] for r in rows])),
                   breach_rate=float(np.mean([r['breach'] for r in rows])),
                   mean_task_return=float(np.mean([r['task_return'] for r in rows])))
    return metrics, rows


def run_training(config, exp, device='cpu', agent=None, max_updates=None):
    training = config['training']
    for name in ('total_steps', 'episodes_per_update', 'eval_interval', 'eval_episodes'):
        if not isinstance(training[name], int) or training[name] <= 0:
            raise ValueError(f'training.{name} must be a positive integer.')
    if not np.isfinite(training['agility']) or training['agility'] <= 0:
        raise ValueError('Attacker agility must be finite and positive.')
    agent = agent or PredictiveMAPPO(config, device)
    agent.config = config
    env = make_env(config)
    if training.get('noisy_agility', False) and training['agility'] <= env.env.agility_noise_half_width:
        raise ValueError('Agility noise must not produce zero or negative agility.')
    started = time.monotonic()
    print(f'[MAPPO] factor_clipping=True cost_gamma=1 synchronous_messages=True device={device}', flush=True)
    while agent.total_steps < training['total_steps'] and (max_updates is None or agent.updates < max_updates):
        rollout = collect_episodes(agent, env, training['episodes_per_update'], training['agility'],
                                   training.get('noisy_agility', False))
        metrics = agent.update(rollout)
        metrics['train_success_rate'] = float(np.mean([r['success'] for r in rollout.summaries]))
        metrics['train_task_return'] = float(np.mean([r['task_return'] for r in rollout.summaries]))
        last = agent.total_steps >= training['total_steps'] or (max_updates is not None and agent.updates >= max_updates)
        if agent.updates % training['eval_interval'] == 0 or last:
            evaluation, _ = evaluate(agent, training['eval_episodes'], agility=training['agility'])
            metrics.update({'eval_' + k: v for k, v in evaluation.items()})
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError('Training produced non-finite metrics.')
        exp.record_metrics(**metrics)
        exp.save_model(agent, training.get('model_name', 'predictive-mappo.pth'))
        print(f"[MAPPO] update={agent.updates} steps={agent.total_steps} collision={rollout.collision_rate:.3f} "
              f"lambda={metrics['lambda_used']:.3f}->{agent.lagrange.value:.3f} elapsed={time.monotonic()-started:.1f}s", flush=True)
    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-updates', type=int, default=None, help='Stop at this total update count, including resumed updates')
    parser.add_argument('--total-steps', type=int, default=None)
    parser.add_argument('--episodes-per-update', type=int, default=None)
    parser.add_argument('--eval-episodes', type=int, default=None)
    parser.add_argument('--collision-budget', type=float, default=None)
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'experiments')
    parser.add_argument('--run-id', default=None)
    args = parser.parse_args()
    if args.threads <= 0 or (args.max_updates is not None and args.max_updates <= 0):
        parser.error('threads and max-updates must be positive')
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    agent = PredictiveMAPPO.from_checkpoint(args.resume, args.device, restore_rng=True) if args.resume else None
    if agent is not None and args.config is not None:
        parser.error('Resume uses the checkpoint config; omit --config.')
    if agent is not None:
        config = copy.deepcopy(agent.config)
    else:
        path = args.config or ROOT/'configs/mappo-prediction.yaml'
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
    for argument, key in ((args.total_steps, 'total_steps'), (args.episodes_per_update, 'episodes_per_update'),
                           (args.eval_episodes, 'eval_episodes')):
        if argument is not None:
            config['training'][key] = argument
    if args.collision_budget is not None:
        if agent is not None:
            parser.error('Keep the declared collision budget fixed when resuming.')
        config['mappo']['collision_budget'] = args.collision_budget
    if config.get('algorithm') != PredictiveMAPPO.algorithm:
        parser.error('Use a predictive-mappo configuration.')
    exp = ExperimentManager(config, base_dir=str(args.output_dir), run_id=args.run_id)
    run_training(config, exp, args.device, agent, args.max_updates)


if __name__ == '__main__':
    main()
