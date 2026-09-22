import torch
import numpy as np
import argparse
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
from policy.SAC import SAC, ReplayBuffer
from envs.TADgame import TADEnv
from utils.config import load_config
from utils.manager import ExperimentManager, set_seed
from utils.protocol import environment_kwargs, apply_adapter_exploration

def evaluate(agent, 
             defender_num=3, 
             agility=2.0,
             boid_state=True,
             controller='Res',
             episodes=50,
             env_options=None):
    with torch.no_grad():
        env = TADEnv(defender_num,
                    boid_state,
                    **(env_options or {}))
        
        def_win_num = 0
        reward = 0.0
        n = episodes
        for _ in range(n):
            s, _ = env.reset(agility, noisy_agility=False)
            done = False
            while not done:
                a = agent.choose_action(s, True)
                a = a.reshape(env.defender_num, -1)
                s_, r, done, _ = env.step(a, controller)
                s = s_
            if done > 2:
                def_win_num += 1
            reward += env.Rewards.mean() / n
    return def_win_num / n, reward

def main(cfg, exp: ExperimentManager, device=torch.device('cpu')):

    if getattr(cfg, 'algorithm', None) == 'predictive-mappo':
        from train_mappo import run_training
        from utils.config import _namespace_to_dict
        return run_training(_namespace_to_dict(cfg), exp, device)

    defender_num = cfg.agent.defender_num
    adaptive = cfg.agent.adaptive
    residual = cfg.agent.residual
    boid_state = cfg.agent.boid_state
    form_reward = cfg.agent.form_reward
    curriculum = cfg.agent.curriculum

    model_name = cfg.training.model_name
    warm_steps = cfg.training.warm_steps
    total_steps = cfg.training.total_steps
    eval_interval = cfg.training.eval_interval
    eval_episodes = getattr(cfg.training, "eval_episodes", 50)
    log_interval = getattr(cfg.training, "log_interval", 1000)
    if not (0 < warm_steps < total_steps and eval_interval > 0 and eval_episodes > 0 and log_interval > 0):
        raise ValueError("Require 0 < warm_steps < total_steps and positive evaluation/log intervals.")

    # Curriculum learning
    init_agility = cfg.curriculum.init_agility
    eva_agility = cfg.curriculum.eva_agility
    ind_agility = cfg.curriculum.ind_agility

    env_options = environment_kwargs(cfg)
    env = TADEnv(defender_num,
                 boid_state,
                 form_reward,
                 **env_options)
    print(f"[PROTOCOL] environment={env_options or {'protocol': 'source'}}; adapter_noise={getattr(cfg.training, 'adapter_noise_distribution', 'uniform')}", flush=True)

    state_dim = env.state_dim
    action_dim = env.action_dim
    if adaptive:
        action_dim += 1

    agent = SAC(cfg, env.feature1_dim, env.feature2_dim, action_dim, adaptive=adaptive, device=device)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Controller type
    if residual:
        if adaptive:
            controller = 'AdaRes'
        else:
            controller = 'Res'
    else:
        controller = 'RL'
    
    print('[INFO] Controller type is', controller)

    train_steps, eval_num = 0, 0
    started = time.perf_counter()
    print(f"[INFO] device={device}, steps={total_steps}, warm_steps={warm_steps}, batch_size={cfg.rl.batch_size}", flush=True)
    while train_steps < total_steps:
        if curriculum:
            agility = int(4 * train_steps / total_steps) * ind_agility + init_agility
            s, _ = env.reset(agility, noisy_agility=True)
        else:
            agility = eva_agility
            s, _ = env.reset(agility, noisy_agility=False)
        done = False
        while not done:

            if train_steps < warm_steps:
                action = np.random.uniform(-1.0, 1.0, (env.defender_num, action_dim))
                if adaptive:
                    action[:, -1] = action[:, -1] * 0.5 + 0.5
            else:
                action = agent.choose_action(s, False)
                action = action.reshape(env.defender_num, action_dim)
                if adaptive:
                    action = apply_adapter_exploration(action, cfg.training)
            
            s_, r, done, _ = env.step(action, controller)

            for i in range(env.defender_num): 
                replay_buffer.store(s[i], action[i], r[i], s_[i], bool(done))
            s = s_

            train_steps += 1
            if train_steps >= warm_steps:
                agent.learn(replay_buffer)

                if train_steps % eval_interval == 0 or train_steps == total_steps:
                    eval_num += 1
                    def_sr, reward = evaluate(agent, defender_num, eva_agility, boid_state, controller, eval_episodes, env_options)
                    if not np.isfinite([def_sr, reward]).all():
                        raise FloatingPointError("Evaluation returned non-finite metrics.")
                    exp.record_metrics(num=eval_num, step=train_steps, def_sr=def_sr, reward=reward)
                    print(f"[EVAL] step={train_steps} success_rate={def_sr:.3f} reward={reward:.3f}", flush=True)
                    exp.save_model(agent, model_name)
            
            if train_steps % log_interval == 0 or train_steps == total_steps:
                updates = max(0, train_steps - warm_steps + 1)
                print(f"[PROGRESS] step={train_steps}/{total_steps} updates={updates} elapsed={time.perf_counter() - started:.1f}s", flush=True)

            if train_steps >= total_steps:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ARBoids Model")
    parser.add_argument("--config", type=str, default=str(SCRIPT_DIR / "configs/train.yaml"), help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use (e.g., cpu, cuda:0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--run-id", default=None, help="Experiment folder name; default: a new timestamped run")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "experiments"), help="Parent experiment directory")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.seed is not None:
        set_seed(args.seed)

    exp = ExperimentManager(
                    cfg, 
                    base_dir=args.output_dir,
                    run_id=args.run_id,
                    repeat_idx=1
                    )
    device = torch.device(args.device)

    main(cfg, exp, device)

