"""Train/evaluate the two-stage channel-fusion policy using the existing boat model."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from envs.TADgame import TADEnv
from RL.MAPPO import MAPPO
from RL.control import to_thrust, blend_thrust
from RL.deployment import DeployedPolicy
from RL.guidance import preserve_rng, search_gates
from RL.observations import tensor_frame
from RL.rollout import TeamRollout
from utils.config import load_config, _namespace_to_dict
from utils.manager import ExperimentManager, set_seed


def make_env(config, defenders=None, duration=None):
    options = dict(config.get("environment", {}))
    if duration is not None:
        options["total_time"] = duration
    agent = config.get("agent", {})
    return TADEnv(defenders or agent.get("defender_num", 3), boid_state=True,
                  form_reward=agent.get("form_reward", True), **options)


def minimum_spacing(env):
    return float(env.def_def_dists.min()) if env.defender_num > 1 else 0.


@torch.no_grad()
def evaluate_episodes(policy, config, episodes, seed=10000, defenders=None,
                      agility=2., duration=None, teacher=None):
    """Main evaluation is actor-only. Teacher search is a separately labeled diagnostic."""
    rows = []
    with preserve_rng():
        for episode in range(episodes):
            np.random.seed(seed + episode)
            torch.manual_seed(seed + episode)
            env = make_env(config, defenders, duration)
            env.reset(agility, noisy_agility=False)
            outcome = steps = q_evaluations = 0
            episode_return = 0.
            spacing = minimum_spacing(env)
            gate_values = []
            while not outcome:
                frame = env.structured_frame(policy.actor.config["motion_features"])
                decision = policy.act(frame)
                action = decision["executed"]
                if teacher is not None:
                    f = tensor_frame(frame, teacher.device)
                    opts = teacher.guidance.options
                    result = search_gates(teacher.q, f, f["boids"],
                                          to_thrust(torch.as_tensor(decision["candidate"], device=teacher.device).unsqueeze(0)),
                                          torch.as_tensor(decision["gates"], device=teacher.device).unsqueeze(0),
                                          fusion=teacher.actor.fusion, step=opts.get("search_step", .15),
                                          radius=opts.get("search_radius", .25), sweeps=opts.get("search_sweeps", 1),
                                          penalty=opts.get("action_penalty", .1), mode=opts.get("search_mode", "joint"))
                    action = result["executed"][0].cpu().numpy()
                    q_evaluations += result["q_evaluations"]
                _, rewards, outcome, info = env.step_thrust(action)
                if not np.allclose(action, info["executed_thrust"], atol=1e-5, rtol=0.):
                    raise AssertionError("Deployment command differs from simulated actuator input")
                episode_return += float(np.mean(rewards))
                spacing = min(spacing, minimum_spacing(env))
                gate_values.append(decision["gates"].mean(0))
                steps += 1
            gates = np.mean(gate_values, axis=0)
            rows.append(dict(episode=episode, seed=seed + episode, defenders=env.defender_num,
                             success=int(outcome in (3, 4)), outcome_code=int(outcome), steps=steps,
                             reward=episode_return, mean_step_reward=episode_return / steps,
                             min_spacing=spacing, gate_c=float(gates[0]), gate_d=float(gates[-1]),
                             q_search_evaluations=q_evaluations))
    return rows


def summarize(rows):
    n = len(rows)
    successes = sum(r["success"] for r in rows)
    p, z = successes / n, 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(passed=True, episodes=n, successes=successes, success_rate=p,
                wilson_95_interval=[center - half, center + half],
                breaches=sum(r["outcome_code"] == 1 for r in rows),
                collisions=sum(r["outcome_code"] == 2 for r in rows),
                captures=sum(r["outcome_code"] == 3 for r in rows),
                timeout_denials=sum(r["outcome_code"] == 4 for r in rows),
                mean_reward=float(np.mean([r["reward"] for r in rows])),
                min_spacing=float(min(r["min_spacing"] for r in rows)))


def evaluate_checkpoint(args, config):
    policy = DeployedPolicy.load(args.checkpoint, args.device)
    teacher = MAPPO.load(args.checkpoint, args.device) if getattr(args, "teacher_search", False) else None
    started = time.perf_counter()
    rows = evaluate_episodes(policy, config, args.episodes, args.seed,
                             getattr(args, "defenders", None), args.agility, args.duration, teacher)
    with (args.output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = summarize(rows)
    result.update(algorithm="channel_mappo", evaluation="teacher_search" if teacher else "decentralized_actor",
                  checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  protocol=config.get("environment", {}).get("protocol", "source"),
                  defenders=rows[0]["defenders"], duration_limit=args.duration,
                  q_search_evaluations=sum(r["q_search_evaluations"] for r in rows),
                  wall_seconds=time.perf_counter() - started)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def main(cfg, exp, device="cpu"):
    config = _namespace_to_dict(cfg)
    training = config["training"]
    total = int(training["total_steps"])
    horizon = int(training.get("rollout_steps", 128))
    evaluation_interval = int(training.get("eval_interval", 4096))
    if min(total, horizon, evaluation_interval, int(training.get("eval_episodes", 10))) < 1:
        raise ValueError("Training, rollout and evaluation sizes must be positive")
    torch.set_num_threads(int(training.get("cpu_threads", 1)))
    agent = MAPPO(config, device)
    counts = config.get("agent", {}).get("training_team_sizes", [config.get("agent", {}).get("defender_num", 3)])
    if not counts or any(int(n) != n or n < 1 for n in counts):
        raise ValueError("training_team_sizes must contain positive integers")
    curriculum = config.get("curriculum", {})

    def reset_env():
        environment = make_env(config, int(np.random.choice(counts)))
        enabled = config.get("agent", {}).get("curriculum", False)
        agility = curriculum.get("eva_agility", 2.)
        if enabled:
            agility = curriculum.get("init_agility", 2.) + min(3, int(4 * agent.training_steps / total)) * curriculum.get("ind_agility", .25)
        environment.reset(agility, noisy_agility=enabled)
        return environment

    env = reset_env()
    completed_episodes = 0
    next_eval = evaluation_interval
    started = time.perf_counter()
    # Log exactly the commands accepted by the simulator, plus their causes.
    fields = ["step", "time", "agent", "team_size", "boids_0", "boids_1", "learned_0", "learned_1",
              "gate_c", "gate_d", "executed_0", "executed_1", "logp_candidate", "logp_gate", "outcome_code"]
    with (Path(exp.exp_dir) / "actions.csv").open("w", newline="", encoding="utf-8") as file:
        action_log = csv.DictWriter(file, fields)
        action_log.writeheader()
        while agent.training_steps < total:
            rollout = TeamRollout()
            length = min(horizon, total - agent.training_steps)
            guidance = config.get("guidance", {})
            sample_count = min(length, int(guidance.get("samples_per_rollout", 4))) if guidance.get("enabled", True) else 0
            snapshot_indices = set(np.linspace(0, length - 1, sample_count, dtype=int))
            projection_count = command_count = 0
            for t in range(length):
                frame_np = env.structured_frame(agent.actor.config["motion_features"])
                frame = tensor_frame(frame_np, agent.device)
                stamp = env.Current_T
                snapshot = env.snapshot() if t in snapshot_indices else None
                with torch.no_grad():
                    decision = agent.actor.act(frame)
                    value = agent.value(frame).item()
                    raw = blend_thrust(frame["boids"], to_thrust(decision["candidate"]), decision["gates"], agent.actor.fusion)
                    projection_count += int((raw - decision["executed"]).abs().gt(1e-4).any(-1).sum())
                executed = decision["executed"][0].cpu().numpy()
                _, rewards, outcome, info = env.step_thrust(executed)
                if not np.allclose(executed, info["executed_thrust"], atol=1e-5, rtol=0.):
                    raise AssertionError("Recorded policy command does not match actuator input")
                decision["executed"] = torch.as_tensor(info["executed_thrust"], device=agent.device, dtype=torch.float32).unsqueeze(0)
                next_frame = env.structured_frame(agent.actor.config["motion_features"])
                with torch.no_grad():
                    next_value = agent.value(tensor_frame(next_frame, agent.device)).item() if not outcome else 0.
                rollout.append(frame_np, next_frame, decision, rewards, outcome, value, next_value, snapshot)
                rollout.rows[-1]["timestamp"] = stamp
                agent.training_steps += 1
                command_count += env.defender_num
                values = {key: value[0].detach().cpu().numpy() for key, value in decision.items()}
                learned = values["candidate"] * 750. + 250.
                for i in range(env.defender_num):
                    action_log.writerow(dict(zip(fields, [agent.training_steps, stamp, i, env.defender_num,
                                                          *frame_np["boids"][i], *learned[i],
                                                          values["gates"][i, 0], values["gates"][i, -1],
                                                          *info["executed_thrust"][i],
                                                          values["logp_candidate"][i, 0], values["logp_gate"][i, 0], outcome])))
                if outcome:
                    completed_episodes += 1
                    env = reset_env()
            metrics = agent.update(rollout, env)
            metrics.update(step=agent.training_steps, episodes=completed_episodes,
                           projection_fraction=projection_count / max(1, command_count),
                           elapsed_seconds=time.perf_counter() - started)
            if agent.training_steps >= next_eval or agent.training_steps == total:
                rows = evaluate_episodes(DeployedPolicy(agent.actor, device), config,
                                         int(training.get("eval_episodes", 10)),
                                         int(training.get("eval_seed", 10000)),
                                         agility=curriculum.get("eva_agility", 2.))
                summary = summarize(rows)
                metrics.update(def_sr=summary["success_rate"], eval_return=summary["mean_reward"],
                               eval_captures=summary["captures"], eval_timeout_denials=summary["timeout_denials"],
                               eval_collisions=summary["collisions"], eval_breaches=summary["breaches"])
                exp.save_model(agent, training.get("model_name", "channel-mappo.pth"))
                next_eval = (agent.training_steps // evaluation_interval + 1) * evaluation_interval
            exp.record_metrics(**metrics)
            file.flush()
            print(f'[MAPPO] step={agent.training_steps}/{total} ppo={metrics["ppo_loss"]:.4f} '
                  f'KL={metrics["approx_kl"]:.5f} guide={metrics["guidance_weight"]:.3f} '
                  f'branches={metrics["branch_steps"]}', flush=True)
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/channel-mappo.yaml"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "experiments"))
    args = parser.parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    experiment = ExperimentManager(cfg, args.output_dir, args.run_id, repeat_idx=1)
    main(cfg, experiment, args.device)
