"""Optional gate-only initialization from successful simulator demonstrations.

The corrective teacher is used only while collecting training demonstrations.
Deployment and subsequent on-policy MAPPO use the unmodified ChannelActor.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from RL.MAPPO import MAPPO
from RL.control import fuse_thrust, to_thrust, THRUST_SCALE
from RL.deployment import DeployedPolicy
from RL.observations import collate_frames, select_frame
from train_mappo import make_env, evaluate_episodes, summarize, selection_score
from utils.manager import ExperimentManager, set_seed


def approaching_vessels(frame, horizon=4., clearance=7.):
    """Training query heuristic; each pair uses the observer's body frame."""
    position = frame["edges"][..., :2] * 60.
    velocity = frame["edges"][..., 2:4] * 5.
    dot = (position * velocity).sum(-1)
    speed2 = (velocity * velocity).sum(-1)
    closest_time = np.clip(-dot / np.maximum(speed2, 1e-8), 0., horizon)
    distance = np.linalg.norm(position + closest_time[..., None] * velocity, axis=-1)
    risk = np.exp(-.5 * (distance / 6.) ** 2 - closest_time / horizon)
    pairs = frame["neighbors"] & (dot < 0.) & (distance < clearance)
    return pairs.any(-1) if (risk * pairs).max(initial=0.) > .25 else np.zeros(len(position), dtype=bool)


@torch.no_grad()
def collect_episodes(actor, config, seeds, *, corrective=False, num_envs=8, device="cpu"):
    """Run complete episodes with independent environment RNG and frozen actor."""
    pending, next_episode, rows = [], 0, []
    while next_episode < len(seeds) or pending:
        while len(pending) < num_envs and next_episode < len(seeds):
            seed = int(seeds[next_episode])
            np.random.seed(seed)
            env = make_env(config)
            env.reset(config["curriculum"]["eva_agility"], noisy_agility=True)
            pending.append(dict(seed=seed, env=env, rng=np.random.get_state(), data=[]))
            next_episode += 1
        frames = [item["env"].structured_frame(actor.config["motion_features"]) for item in pending]
        batch = collate_frames(frames, device)
        decision = actor.act(batch, deterministic=True)
        gates = decision["gates"].clone()
        changed = []
        for index, frame in enumerate(frames):
            active = approaching_vessels(frame) if corrective else np.zeros(len(frame["mask"]), dtype=bool)
            if corrective:
                gates[index, :len(active), 0][torch.as_tensor(active, device=device)] = 1.
            changed.append(active)
        actions = fuse_thrust(batch["boids"], to_thrust(decision["candidate"]), gates, actor.fusion).cpu().numpy()
        proposals = decision["candidate"].cpu().numpy()
        targets = gates.cpu().numpy()
        remaining = []
        for index, item in enumerate(pending):
            env = item["env"]
            n = env.defender_num
            if len(item["data"]) > math.ceil(env.Total_T / env.Action_T):
                raise RuntimeError("A collection episode did not terminate within its task time limit")
            item["data"].append(dict(frame=frames[index], candidate=proposals[index, :n].copy(),
                                     gates=targets[index, :n].copy(), action=actions[index, :n].copy(),
                                     changed=changed[index].copy()))
            np.random.set_state(item["rng"])
            _, _, outcome, info = env.step_thrust(actions[index, :n])
            item["rng"] = np.random.get_state()
            if not np.allclose(actions[index, :n], info["executed_thrust"], atol=1e-5, rtol=0.):
                raise AssertionError("Teacher command differs from executed thrust")
            if outcome:
                rows.append(dict(seed=item["seed"], outcome=int(outcome), data=item["data"]))
            else:
                remaining.append(item)
        pending = remaining
    return rows


def successful_corrections(original, corrected):
    """Only supervise corrections that turn a matched collision into a success."""
    outcomes = {row["seed"]: row["outcome"] for row in original}
    return [row for row in corrected if outcomes[row["seed"]] == 2 and row["outcome"] in (3, 4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "experiments")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-episodes", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.00003)
    args = parser.parse_args()
    if min(args.random_episodes, args.updates, args.eval_every, args.batch_size, args.learning_rate) <= 0:
        parser.error("Collection and optimization sizes must be positive")
    directory = args.output_dir / args.run_id
    if directory.exists():
        parser.error("Use a new run-id")
    torch.set_num_threads(1)
    set_seed(args.seed)
    agent = MAPPO.load(args.checkpoint, args.device)
    if agent.actor.fusion != "channels":
        parser.error("This teacher requires physical channel fusion")
    teacher = copy.deepcopy(agent.actor).requires_grad_(False).eval()
    config = copy.deepcopy(agent.config)
    source_state = copy.deepcopy(agent.trainer_state)
    source_step = agent.training_steps
    # Distinct training stream; validation/qualification episodes never provide labels.
    data_seed = 830000 + args.seed * 10000
    seeds = list(dict.fromkeys([int(x[0]) for x in source_state.get("collision_curriculum", {}).get("cases", [])]
                              + list(range(data_seed, data_seed + args.random_episodes))))
    eval_seed, eval_count = config["training"]["eval_seed"], config["training"]["eval_episodes"]
    if any(eval_seed <= seed < eval_seed + eval_count or 630000 <= seed < 650000 for seed in seeds):
        raise ValueError("Training seeds overlap reserved evaluation seeds")
    config["gate_refinement"] = dict(method="successful_simulator_gate_demonstrations",
                                    source_checkpoint=str(args.checkpoint.resolve()),
                                    source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                    source_training_steps=source_step, training_seeds=seeds,
                                    updates=args.updates, learning_rate=args.learning_rate,
                                    batch_size=args.batch_size, teacher_horizon=4., teacher_clearance=7.)
    agent.config = config
    experiment = ExperimentManager(config, str(args.output_dir), args.run_id, repeat_idx=1)
    started = time.perf_counter()
    original = collect_episodes(teacher, config, seeds, device=args.device)
    collision_seeds = [row["seed"] for row in original if row["outcome"] == 2]
    print(json.dumps(dict(phase="collect", episodes=len(original), collisions=len(collision_seeds))), flush=True)
    corrected = collect_episodes(teacher, config, collision_seeds, corrective=True, device=args.device)
    accepted = successful_corrections(original, corrected)
    interventions = [d for row in accepted for d in row["data"] if d["changed"].any()]
    anchors = [d for row in original if row["outcome"] in (3, 4) for d in row["data"]]
    anchors += [d for row in accepted for d in row["data"] if not d["changed"].any()]
    if not interventions or not anchors:
        raise RuntimeError("No paired successful corrections or no successful anchor trajectories")
    print(json.dumps(dict(phase="paired_teacher", trials=len(corrected), improved=len(accepted),
                          intervention_frames=len(interventions), anchor_frames=len(anchors))), flush=True)

    data = interventions + anchors
    frames = collate_frames([row["frame"] for row in data], args.device)
    candidates = torch.tensor(np.asarray([row["candidate"] for row in data]), device=args.device)
    actions = torch.tensor(np.asarray([row["action"] for row in data]), device=args.device)
    targets = torch.tensor(np.asarray([row["gates"] for row in data]), device=args.device)
    messages = teacher.exchange(frames, candidates)
    with torch.no_grad():
        reference, _ = teacher.gate_distribution(frames, candidates, messages)
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.learning_rate, eps=1e-5)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(),
        lr=agent.options.get("learning_rate", 1e-4), eps=1e-5)
    agent.training_steps = 0
    agent.trainer_state = dict(source_trainer_state=source_state,
                              gate_refinement=dict(source_training_steps=source_step,
                                  training_episodes=len(original), corrected_episodes=len(corrected),
                                  accepted_seeds=[row["seed"] for row in accepted],
                                  simulator_steps=sum(len(row["data"]) for row in original + corrected),
                                  intervention_frames=len(interventions), anchor_frames=len(anchors)))
    best_score = None
    checks = []
    for update in range(1, args.updates + 1):
        # Equal mass on scarce improvements and broad successful behavior.
        indices = torch.cat((torch.randint(len(interventions), (args.batch_size,), device=args.device),
                             torch.randint(len(interventions), len(data), (args.batch_size,), device=args.device)))
        frame = select_frame(frames, indices)
        distribution, _ = agent.actor.gate_distribution(frame, candidates[indices], messages[indices], detach_features=True)
        gates = distribution.mean.sigmoid()
        predicted = fuse_thrust(frame["boids"], to_thrust(candidates[indices]), gates, agent.actor.fusion)
        imitation = ((predicted - actions[indices]) / THRUST_SCALE).square().mean()
        gate_loss = (gates - targets[indices]).square().mean()
        # Preserve exploration variance and turn behavior; only mean propulsion
        # is intentionally corrected on accepted interventions.
        variance_loss = (distribution.scale.log() - reference.scale[indices].log()).square().mean()
        loss = imitation + .1 * gate_loss + .01 * variance_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite gate-refinement objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if update % args.eval_every and update != args.updates:
            continue
        summary = summarize(evaluate_episodes(DeployedPolicy(agent.actor, args.device), config,
                                             eval_count, eval_seed, agility=config["curriculum"]["eva_agility"]))
        score = selection_score(summary, config["training"])
        agent.trainer_state["gate_refinement"].update(updates=update, elapsed_seconds=time.perf_counter() - started)
        if best_score is None or score > best_score:
            best_score = score
            experiment.save_model(agent, "channel-mappo-best.pth")
        experiment.save_model(agent, "channel-mappo-init.pth")
        checks.append(dict(update=update, loss=float(loss.detach()), **summary,
                           elapsed_seconds=time.perf_counter() - started))
        (directory / "refinement_metrics.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        print(json.dumps(checks[-1]), flush=True)


if __name__ == "__main__":
    main()
