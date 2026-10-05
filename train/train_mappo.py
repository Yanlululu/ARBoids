"""Train/evaluate the two-stage channel-fusion policy using the existing boat model."""
import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from envs.TADgame import TADEnv
from RL.MAPPO import MAPPO
from RL.control import to_thrust
from RL.collector import TeamCollector
from RL.deployment import DeployedPolicy
from RL.guidance import preserve_rng, search_gates
from RL.observations import tensor_frame, collate_frames
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
def evaluate_episodes_serial(policy, config, episodes, seed=10000, defenders=None,
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


@torch.no_grad()
def evaluate_episodes(policy, config, episodes, seed=10000, defenders=None,
                      agility=2., duration=None, teacher=None):
    """Batch deterministic actors; keep every evaluation world's RNG independent."""
    parallel = int(config.get("training", {}).get("eval_num_envs", 1))
    if parallel < 1 or episodes < 1:
        raise ValueError("Evaluation environment/episode counts must be positive")
    if parallel == 1 or teacher is not None:
        return evaluate_episodes_serial(policy, config, episodes, seed, defenders, agility, duration, teacher)
    rows, active, next_episode = [], [], 0
    with preserve_rng():
        while len(rows) < episodes:
            while len(active) < parallel and next_episode < episodes:
                np.random.seed(seed + next_episode)
                env = make_env(config, defenders, duration)
                env.reset(agility, noisy_agility=False)
                active.append(dict(env=env, episode=next_episode, rng=np.random.get_state(),
                                   steps=0, reward=0., spacing=minimum_spacing(env), gates=[]))
                next_episode += 1
            frames = [item["env"].structured_frame(policy.actor.config["motion_features"]) for item in active]
            decision = policy.actor.act(collate_frames(frames, policy.device), deterministic=True)
            commands, gates = decision["executed"].cpu().numpy(), decision["gates"].cpu().numpy()
            remaining = []
            for index, item in enumerate(active):
                env = item["env"]
                np.random.set_state(item["rng"])
                command = commands[index, :env.defender_num]
                _, rewards, outcome, info = env.step_thrust(command)
                item["rng"] = np.random.get_state()
                if not np.allclose(command, info["executed_thrust"], atol=1e-5, rtol=0.):
                    raise AssertionError("Deployment command differs from simulated actuator input")
                item["steps"] += 1
                item["reward"] += float(np.mean(rewards))
                item["spacing"] = min(item["spacing"], minimum_spacing(env))
                item["gates"].append(gates[index, :env.defender_num].mean(0))
                if not outcome:
                    remaining.append(item)
                    continue
                mean_gates = np.mean(item["gates"], axis=0)
                rows.append(dict(episode=item["episode"], seed=seed + item["episode"], defenders=env.defender_num,
                                 success=int(outcome in (3, 4)), outcome_code=int(outcome), steps=item["steps"],
                                 reward=item["reward"], mean_step_reward=item["reward"] / item["steps"],
                                 min_spacing=item["spacing"], gate_c=float(mean_gates[0]), gate_d=float(mean_gates[-1]),
                                 q_search_evaluations=0))
            active = remaining
    return sorted(rows, key=lambda row: row["episode"])


def summarize(rows):
    n = len(rows)
    successes = sum(r["success"] for r in rows)
    p, z = successes / n, 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    collisions = sum(r["outcome_code"] == 2 for r in rows)
    collision_rate = collisions / n
    collision_center = (collision_rate + z * z / (2 * n)) / (1 + z * z / n)
    collision_half = z * np.sqrt(collision_rate * (1 - collision_rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(passed=True, episodes=n, successes=successes, success_rate=p,
                wilson_95_interval=[center - half, center + half],
                breaches=sum(r["outcome_code"] == 1 for r in rows),
                collisions=collisions, collision_rate=collision_rate,
                collision_wilson_95_interval=[collision_center - collision_half, collision_center + collision_half],
                captures=sum(r["outcome_code"] == 3 for r in rows),
                timeout_denials=sum(r["outcome_code"] == 4 for r in rows),
                mean_reward=float(np.mean([r["reward"] for r in rows])),
                min_spacing=float(min(r["min_spacing"] for r in rows)))


def selection_score(summary, training):
    """Minimize collisions among policies meeting an explicit success floor."""
    objective = training.get("selection_objective", "success")
    success, collisions, reward = summary["success_rate"], summary["collisions"], summary["mean_reward"]
    if objective == "success":
        return success, -collisions, reward
    if objective != "collision" or "selection_success_floor" not in training:
        raise ValueError("Collision selection requires an explicit success floor")
    floor = float(training["selection_success_floor"])
    if not 0. <= floor <= 1.:
        raise ValueError("Selection success floor must be between zero and one")
    if success < floor:
        return 0., success, -collisions, reward
    return 1., -collisions, success, reward


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


def main(cfg, exp, device="cpu", *, resume=None, initialize_actor=None, initialize_model=None,
         max_steps=None, max_seconds=None, stop_file=None):
    config = _namespace_to_dict(cfg)
    training = config["training"]
    action_log_interval = int(training.get("action_log_interval", 1))
    if action_log_interval < 1:
        raise ValueError("action_log_interval must be positive")
    total = int(training["total_steps"])
    horizon = int(training.get("rollout_steps", 128))
    evaluation_interval = int(training.get("eval_interval", 4096))
    if min(total, horizon, evaluation_interval, int(training.get("eval_episodes", 10))) < 1:
        raise ValueError("Training, rollout and evaluation sizes must be positive")
    if (max_steps is not None and max_steps < 1) or (max_seconds is not None and max_seconds <= 0):
        raise ValueError("Run limits must be positive")
    torch.set_num_threads(int(training.get("cpu_threads", 1)))
    if sum(x is not None for x in (resume, initialize_actor, initialize_model)) > 1:
        raise ValueError("Resume and initialization are mutually exclusive")
    agent = MAPPO.load(resume, device) if resume else MAPPO(config, device)
    if initialize_actor or initialize_model:
        source_path = initialize_actor or initialize_model
        source = MAPPO.load(source_path, "cpu")
        if source.actor.config != agent.actor.config:
            raise ValueError("Actor initialization requires the same policy architecture")
        agent.actor.load_state_dict(source.actor.state_dict())
        if initialize_model:
            for key in ("value", "q", "value_target"):
                getattr(agent, key).load_state_dict(getattr(source, key).state_dict())
            if agent.cost_value:
                if source.cost_value:
                    agent.cost_value.load_state_dict(source.cost_value.state_dict())
                else:
                    # Reuse the trained physical representation, keeping a new
                    # probability output; task V itself is unchanged.
                    weights = {k: v for k, v in source.value.state_dict().items()
                               if k in agent.cost_value.state_dict()}
                    head = copy.deepcopy(agent.cost_value.readout[-1].state_dict())
                    agent.cost_value.load_state_dict(weights, strict=False)
                    agent.cost_value.readout[-1].load_state_dict(head)
            agent.terminal_calibration.load_state_dict(source.terminal_calibration.state_dict())
            if (training.get("collision_revisit_probability", 0.)
                    and source.config.get("training", {}).get("collision_revisit_probability")
                    == training["collision_revisit_probability"]
                    and "collision_curriculum" in source.trainer_state):
                agent.trainer_state["collision_curriculum"] = copy.deepcopy(source.trainer_state["collision_curriculum"])
        agent.trainer_state["initialization"] = dict(checkpoint=str(Path(source_path).resolve()),
                                                     source_training_steps=source.training_steps)
        agent.trainer_state["source_trainer_state"] = source.trainer_state
        print(f"[INITIALIZE] {'actor and critics' if initialize_model else 'actor'} from step={source.training_steps}; fresh optimizers", flush=True)
        del source
    if resume:
        for key in ("policy", "mappo", "guidance", "agent", "curriculum", "environment"):
            if agent.config.get(key) != config.get(key):
                raise ValueError(f"Resume configuration mismatch: {key}")
        for key in ("total_steps", "eval_seed", "eval_episodes", "collision_revisit_probability",
                    "collision_revisit_capacity", "collision_revisit_solved_after",
                    "selection_objective", "selection_success_floor"):
            if agent.config["training"].get(key) != training.get(key):
                raise ValueError(f"Keep training.{key} fixed on resume for comparable curriculum/selection")
        agent.restore_training_rng()
        print(f"[RESUME] step={agent.training_steps}; new episode and on-policy rollout", flush=True)
    if agent.training_steps >= total:
        raise ValueError("Checkpoint has already reached total_steps")
    agent.config = config
    stop_step = min(total, agent.training_steps + max_steps) if max_steps is not None else total
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

    collector = TeamCollector(agent, reset_env, int(training.get("num_envs", 1)))
    completed_episodes = int(agent.trainer_state.get("completed_episodes", 0))
    previous_elapsed = float(agent.trainer_state.get("elapsed_seconds", 0.))
    best_score = tuple(agent.trainer_state.get("best_score", (-1., -float("inf"), -float("inf"))))
    next_eval = (agent.training_steps // evaluation_interval + 1) * evaluation_interval
    started = time.perf_counter()
    # Log exactly the commands accepted by the simulator, plus their causes.
    fields = ["step", "time", "agent", "team_size", "boids_0", "boids_1", "learned_0", "learned_1",
              "gate_c", "gate_d", "executed_0", "executed_1", "logp_candidate", "logp_gate", "outcome_code"]
    action_path = Path(exp.exp_dir) / "actions.csv"
    if action_path.exists() and not resume:
        raise FileExistsError(f"Training output already exists: {action_path}; use --resume or a new run-id")
    # Upgrade early single-environment logs in place before appending vector data.
    fields.append("env_id")
    if action_path.exists() and action_path.stat().st_size:
        with action_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if "env_id" not in reader.fieldnames:
                temporary = action_path.with_suffix(".csv.tmp")
                with temporary.open("w", newline="", encoding="utf-8") as destination:
                    writer = csv.DictWriter(destination, fields)
                    writer.writeheader()
                    for row in reader:
                        writer.writerow(dict(row, env_id=0))
        if action_path.with_suffix(".csv.tmp").exists():
            action_path.with_suffix(".csv.tmp").replace(action_path)
    write_header = not action_path.exists() or action_path.stat().st_size == 0
    with action_path.open("a", newline="", encoding="utf-8") as file:
        action_log = csv.DictWriter(file, fields)
        if write_header:
            action_log.writeheader()
        while agent.training_steps < stop_step:
            first_step = agent.training_steps
            length = min(horizon, stop_step - agent.training_steps)
            guidance = config.get("guidance", {})
            sample_count = min(length, int(guidance.get("samples_per_rollout", 4))) if guidance.get("enabled", True) else 0
            snapshot_indices = set(np.linspace(0, length - 1, sample_count, dtype=int))
            collect_started = time.perf_counter()
            priority_count = int(guidance.get("priority_samples_per_rollout", 0)) if guidance.get("enabled", True) else 0
            rollout, collected = collector.collect(length, snapshot_indices, priority_count,
                                                   float(guidance.get("priority_lead_time", 0.)))
            collect_seconds = time.perf_counter() - collect_started
            completed_episodes += collected["completed_episodes"]
            for t, row in enumerate(rollout.rows):
                if (first_step + t) % action_log_interval and not row["outcome"]:
                    continue
                values = row["decision"]
                learned = values["candidate"] * 750. + 250.
                n = len(row["frame"]["mask"])
                for i in range(n):
                    action_log.writerow(dict(zip(fields, [first_step + t + 1, row["timestamp"], i, n,
                                                          *row["frame"]["boids"][i], *learned[i],
                                                          values["gates"][i, 0], values["gates"][i, -1],
                                                          *values["executed"][i],
                                                          values["logp_candidate"][i, 0], values["logp_gate"][i, 0],
                                                          row["outcome"], row["stream_id"]])))
            update_started = time.perf_counter()
            metrics = agent.update(rollout, collector.envs[0])
            elapsed = time.perf_counter() - started
            stopping = (agent.training_steps >= stop_step or (max_seconds is not None and elapsed >= max_seconds)
                        or (stop_file is not None and Path(stop_file).exists()))
            metrics.update(step=agent.training_steps, episodes=completed_episodes,
                           projection_fraction=collected["projection_fraction"],
                           priority_snapshots=collected["priority_snapshots"],
                           collision_revisit_episodes=collected["collision_revisit_episodes"],
                           collision_start_cases=collected["collision_start_cases"],
                           collect_seconds=collect_seconds, update_seconds=time.perf_counter() - update_started,
                           elapsed_seconds=previous_elapsed + elapsed)
            agent.trainer_state.update(completed_episodes=completed_episodes,
                                       elapsed_seconds=previous_elapsed + elapsed)
            if agent.training_steps >= next_eval or stopping:
                rows = evaluate_episodes(DeployedPolicy(agent.actor, device), config,
                                         int(training.get("eval_episodes", 10)),
                                         int(training.get("eval_seed", 10000)),
                                         agility=curriculum.get("eva_agility", 2.))
                summary = summarize(rows)
                metrics["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
                agent.trainer_state["elapsed_seconds"] = metrics["elapsed_seconds"]
                metrics.update(def_sr=summary["success_rate"], eval_return=summary["mean_reward"],
                               eval_captures=summary["captures"], eval_timeout_denials=summary["timeout_denials"],
                               eval_collisions=summary["collisions"], eval_breaches=summary["breaches"])
                score = selection_score(summary, training)
                if score > best_score:
                    best_score = score
                    agent.trainer_state.update(best_score=list(score), best_step=agent.training_steps)
                    exp.save_model(agent, "channel-mappo-best.pth")
                print(f'[EVAL] step={agent.training_steps} success={summary["successes"]}/{summary["episodes"]} '
                      f'collisions={summary["collisions"]} breaches={summary["breaches"]} '
                      f'best_step={agent.trainer_state["best_step"]}', flush=True)
                exp.save_model(agent, training.get("model_name", "channel-mappo.pth"))
                next_eval = (agent.training_steps // evaluation_interval + 1) * evaluation_interval
            exp.record_metrics(**metrics)
            file.flush()
            print(f'[MAPPO] step={agent.training_steps}/{total} ppo={metrics["ppo_loss"]:.4f} '
                  f'KL={metrics["approx_kl"]:.5f} guide={metrics["guidance_weight"]:.3f} '
                  f'branches={metrics["branch_steps"]}', flush=True)
            if stopping:
                break
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/channel-mappo.yaml"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "experiments"))
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path, default=None, help="Continue weights/optimizers at a new episode boundary")
    initialization.add_argument("--initialize-actor", type=Path, default=None, help="Initialize actor weights; start fresh critics/optimizers in a new run")
    initialization.add_argument("--initialize-model", type=Path, default=None, help="Initialize actor and critics; start fresh optimizers in a new run")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum additional steps for this invocation")
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop after a rollout, then evaluate and save")
    parser.add_argument("--stop-file", type=Path, default=None, help="If this file appears, finish the rollout, evaluate and save")
    args = parser.parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg.seed = args.seed
    if args.initialize_actor or args.initialize_model:
        source_path = args.initialize_actor or args.initialize_model
        cfg.initialization = dict(checkpoint=str(source_path.resolve()),
                                  sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                                  include_critics=bool(args.initialize_model))
    if args.resume:
        cfg.resumption = dict(checkpoint=str(args.resume.resolve()),
                              sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest())
    if args.run_id and not args.resume and (Path(args.output_dir) / args.run_id / "actions.csv").exists():
        parser.error("Run already contains training data; use --resume or a new --run-id")
    experiment = ExperimentManager(cfg, args.output_dir, args.run_id, repeat_idx=1)
    main(cfg, experiment, args.device, resume=args.resume, initialize_actor=args.initialize_actor, initialize_model=args.initialize_model,
         max_steps=args.max_steps, max_seconds=args.max_seconds, stop_file=args.stop_file)
