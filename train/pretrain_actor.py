"""Optional ARBoids demonstration initialization; never inserts data into PPO.

This changes the training initialization and must be shared by formal controls.
The deployed model remains the two-stage ChannelActor, without the SAC teacher.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from RL.MAPPO import MAPPO
from RL.Networks import ActorAdap
from RL.control import fuse_thrust, to_thrust, THRUST_SCALE
from RL.deployment import DeployedPolicy
from RL.observations import collate_frames, select_frame, tensor_frame
from train_mappo import make_env, evaluate_episodes, summarize
from utils.config import load_config, _namespace_to_dict
from utils.manager import ExperimentManager, set_seed


@torch.no_grad()
def demonstration(teacher, observations, boids):
    action = teacher(torch.as_tensor(observations, dtype=torch.float32), True, False)[0]
    return fuse_thrust(torch.as_tensor(boids, dtype=torch.float32),
                       to_thrust(action[:, :2]), action[:, 2:3], "scalar").numpy()


@torch.no_grad()
def collect_demonstrations(teacher, student, config, steps, teacher_probability, seed):
    """Label teacher actions on expert/student state distributions (data aggregation)."""
    inference = copy.deepcopy(student).cpu().eval()
    np.random.seed(seed)
    env = make_env(config)
    obs, _ = env.reset(config.get("curriculum", {}).get("eva_agility", 2.), noisy_agility=False)
    frames, targets, episodes = [], [], 0
    for _ in range(steps):
        frame = env.structured_frame(student.config["motion_features"])
        target = demonstration(teacher, obs, frame["boids"])
        frames.append(frame)
        targets.append(target)
        action = target
        if np.random.random() >= teacher_probability:
            action = inference.act(tensor_frame(frame), deterministic=True)["executed"][0].numpy()
        obs, _, outcome, _ = env.step_thrust(action)
        if outcome:
            episodes += 1
            obs, _ = env.reset(config.get("curriculum", {}).get("eva_agility", 2.), noisy_agility=False)
    return frames, targets, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--steps-per-round", type=int, default=6000)
    parser.add_argument("--updates-per-round", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if min(args.rounds, args.steps_per_round, args.updates_per_round, args.batch_size) < 1:
        parser.error("Collection and optimization sizes must be positive")
    directory = args.output_dir / args.run_id
    if directory.exists():
        parser.error("Use a new run-id for demonstration initialization")
    torch.set_num_threads(1)
    set_seed(args.seed)
    cfg = load_config(args.config)
    config = _namespace_to_dict(cfg)
    config["seed"] = args.seed
    config["initialization"] = dict(
        method="arboids_action_imitation", teacher_checkpoint=str(args.teacher.resolve()),
        teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
        data_seed=730000 + args.seed, rounds=args.rounds, steps_per_round=args.steps_per_round,
        updates_per_round=args.updates_per_round, batch_size=args.batch_size, learning_rate=.0003)
    experiment = ExperimentManager(config, str(args.output_dir), args.run_id, repeat_idx=1)
    agent = MAPPO(config, args.device)
    weights = torch.load(args.teacher, map_location="cpu", weights_only=True)
    hidden = weights["l2.weight"].shape[0]
    teacher = ActorAdap(6, 8, 3, hidden).eval()
    teacher.load_state_dict(weights)
    teacher.requires_grad_(False)
    # Modest stochastic exploration for subsequent PPO, independent of the
    # deterministic imitation objective. PPO can subsequently learn these rows.
    with torch.no_grad():
        for head, dim, std in ((agent.actor.proposal_head[-1], 2, .15),
                               (agent.actor.gate_head[-1], agent.actor.gate_dim, .25)):
            head.weight[dim:].zero_()
            head.bias[dim:].fill_(math.log(std))
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=.0003, eps=1e-5)
    frames, targets, rows = [], [], []
    best_score = (-1., -float("inf"), -float("inf"))
    started = time.perf_counter()
    for round_index in range(args.rounds):
        probability = 1. if round_index == 0 else (.3 if round_index == 1 else 0.)
        new_frames, new_targets, episodes = collect_demonstrations(
            teacher, agent.actor, config, args.steps_per_round, probability,
            config["initialization"]["data_seed"] + round_index * 10000)
        frames.extend(new_frames)
        targets.extend(new_targets)
        data = collate_frames(frames, args.device)
        labels = torch.as_tensor(np.asarray(targets), device=args.device, dtype=torch.float32)
        print(f"[IMITATION] round={round_index + 1} transitions={len(frames)} new_episodes={episodes}", flush=True)
        for update in range(args.updates_per_round):
            indices = torch.randint(len(frames), (args.batch_size,), device=args.device)
            decision = agent.actor.act(select_frame(data, indices), deterministic=True)
            loss = ((decision["executed"] - labels[indices]) / THRUST_SCALE).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite imitation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if (update + 1) % 400 == 0:
                print(f"[IMITATION] updates={update + 1} action_mse={loss.item():.6f}", flush=True)
        evaluation = summarize(evaluate_episodes(
            DeployedPolicy(agent.actor, args.device), config, config["training"]["eval_episodes"],
            config["training"]["eval_seed"], agility=config["curriculum"]["eva_agility"]))
        score = (evaluation["success_rate"], -evaluation["collisions"], evaluation["mean_reward"])
        agent.trainer_state["pretraining"] = dict(transitions=len(frames),
            updates=(round_index + 1) * args.updates_per_round, wall_seconds=time.perf_counter() - started,
            teacher=str(args.teacher.resolve()))
        if score > best_score:
            best_score = score
            agent.trainer_state.update(best_score=list(score), best_step=0)
            experiment.save_model(agent, "channel-mappo-best.pth")
        experiment.save_model(agent, "channel-mappo-init.pth")
        row = dict(round=round_index + 1, transitions=len(frames), action_mse=loss.item(),
                   **evaluation, elapsed_seconds=time.perf_counter() - started)
        rows.append(row)
        with (directory / "imitation_metrics.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
