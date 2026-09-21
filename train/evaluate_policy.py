"""Evaluate a saved policy with independent seeds in the original 2D environment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from envs.TADgame import TADEnv
from policy.SAC import SAC
from utils.config import load_config
from utils.manager import set_seed
from utils.protocol import environment_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'configs/train.yaml')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--agility', type=float, default=2.0)
    parser.add_argument('--duration', type=float, default=60., help='Evaluation horizon in simulation seconds; paper default is 60')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.episodes <= 0 or args.agility <= 0 or args.duration <= 0:
        parser.error('episodes, agility and duration must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = load_config(str(args.config))
    env = TADEnv(cfg.agent.defender_num, cfg.agent.boid_state, cfg.agent.form_reward, **environment_kwargs(cfg))
    env.Total_T = args.duration
    action_dim = env.action_dim + int(cfg.agent.adaptive)
    agent = SAC(cfg, env.feature1_dim, env.feature2_dim, action_dim,
                adaptive=cfg.agent.adaptive, device=torch.device(args.device))
    agent.load(str(args.checkpoint))
    agent.actor.eval()
    if not all(torch.isfinite(p).all().item() for p in agent.actor.parameters()):
        raise FloatingPointError('Non-finite checkpoint weights')
    controller = 'AdaRes' if cfg.agent.adaptive else ('Res' if cfg.agent.residual else 'RL')
    rows = []
    started = time.monotonic()
    for episode in range(args.episodes):
        seed = args.seed + episode
        set_seed(seed)
        state, _ = env.reset(args.agility, noisy_agility=False)
        done = steps = 0
        while not done:
            action = agent.choose_action(state, True).reshape(env.defender_num, -1)
            if not np.isfinite(action).all():
                raise FloatingPointError('Non-finite policy action')
            state, reward, done, _ = env.step(action, controller)
            if not np.isfinite(state).all() or not np.isfinite(reward).all():
                raise FloatingPointError('Non-finite environment transition')
            steps += 1
            if steps > 10000:
                raise RuntimeError('Environment did not terminate')
        rows.append(dict(episode=episode, seed=seed, success=int(done > 2),
                         outcome_code=int(done), steps=steps, reward=float(env.Rewards.mean())))
        if (episode + 1) % 10 == 0:
            print(f'[EVAL] {episode+1}/{args.episodes} success_rate={np.mean([r["success"] for r in rows]):.3f}', flush=True)
    with (args.output_dir / 'episodes.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successes = sum(row['success'] for row in rows)
    n, z = args.episodes, 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    summary = dict(passed=True, episodes=n, successes=successes, success_rate=p,
                   wilson_95_interval=[center-half, center+half], agility=args.agility,
                   protocol=env.protocol, early_attacker_win=env.early_attacker_win,
                   agility_noise_half_width=env.agility_noise_half_width,
                   config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   duration_limit=env.Total_T, capture_radius=env.Defend_R,
                   target_radius=env.Target_R, collision_radius=env.Collision_R,
                   first_seed=args.seed, checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   mean_reward=float(np.mean([r['reward'] for r in rows])),
                   wall_seconds=time.monotonic()-started)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
