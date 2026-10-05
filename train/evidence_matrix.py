"""Fixed-budget, independently initialized, paired-seed research experiments.

This runner never tunes hyperparameters from test outcomes. Test evaluation is a
separate command, permitted only after every requested training job has finished.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from evidence_eval import ENVIRONMENT, PhysicalPolicy, sha256
from RL.MAPPO import MAPPO
from RL.control import fuse_thrust, to_thrust, THRUST_SCALE
from RL.deployment import DeployedPolicy
from RL.observations import collate_frames, select_frame
from refine_actor import approaching_vessels
from train_mappo import make_env, evaluate_episodes, summarize, selection_score, main as train
from utils.config import load_config, _namespace_to_dict
from utils.manager import ExperimentManager, set_seed


SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)
ARMS = ("full", "scalar", "thrusters", "no_candidates", "mean_relations",
        "no_guidance", "independent_guidance", "no_motion", "no_safety_demonstrations")
EXCLUDED = ((610000, 650000), (1980000, 2200000), (9000000, 9100000))


def config_for(arm, seed):
    if arm not in ARMS:
        raise ValueError("Unknown experiment arm")
    config = _namespace_to_dict(load_config(Path(__file__).parent / "configs/channel-mappo-gate-stable.yaml"))
    config["seed"] = int(seed)
    config["environment"] = dict(ENVIRONMENT)
    config["training"].update(total_steps=1000000, rollout_steps=8192, num_envs=8,
                              eval_interval=65536, eval_episodes=128, eval_seed=610000,
                              action_log_interval=100, excluded_seed_ranges=[list(r) for r in EXCLUDED])
    config["experiment"] = dict(protocol="channel-evidence-v1", arm=arm,
                                 training_seed=seed, inference="actor_only")
    if arm in ("scalar", "thrusters"):
        config["policy"]["fusion"] = arm
    if arm == "no_candidates":
        config["policy"]["share_candidates"] = False
    if arm == "mean_relations":
        config["policy"]["aggregation"] = "mean"
    if arm == "no_guidance":
        config["guidance"]["enabled"] = False
    if arm == "independent_guidance":
        config["guidance"]["search_mode"] = "independent"
    if arm == "no_motion":
        config["policy"]["motion_features"] = False
    return config


def mask_motion(data):
    output = dict(data)
    output["local"] = data["local"].clone()
    output["edges"] = data["edges"].clone()
    output["local"][..., 6:9] = 0.
    output["edges"][..., 2:6] = 0.
    return output


@torch.no_grad()
def collect_teacher(teacher, seeds, override=None, parallel=8):
    """Matched complete training episodes; override only the common-thrust gate."""
    config = dict(environment=ENVIRONMENT)
    pending, results, cursor = [], [], 0
    while cursor < len(seeds) or pending:
        while cursor < len(seeds) and len(pending) < parallel:
            np.random.seed(int(seeds[cursor]))
            env = make_env(config)
            obs, _ = env.reset(2., noisy_agility=True)
            pending.append(dict(env=env, observation=obs, seed=int(seeds[cursor]),
                                rng=np.random.get_state(), data=[]))
            cursor += 1
        frames = [row["env"].structured_frame() for row in pending]
        batch = collate_frames(frames)
        obs = np.asarray([row["observation"] for row in pending])
        raw = teacher.actor(torch.as_tensor(obs.reshape(-1, obs.shape[-1]), dtype=torch.float32),
                            True, False)[0].reshape(len(pending), 3, 3)
        gates = raw[..., 2:3].expand(-1, -1, 2).clone()
        changed = []
        for i, frame in enumerate(frames):
            active = approaching_vessels(frame) if override is not None else np.zeros(3, dtype=bool)
            if override is not None:
                gates[i, torch.as_tensor(active), 0] = override
            changed.append(bool(active.any()))
        commands = fuse_thrust(batch["boids"], to_thrust(raw[..., :2]), gates, "channels").numpy()
        remaining = []
        for i, row in enumerate(pending):
            row["data"].append((frames[i], commands[i].copy(), changed[i]))
            np.random.set_state(row["rng"])
            row["observation"], _, outcome, _ = row["env"].step_thrust(commands[i])
            row["rng"] = np.random.get_state()
            if len(row["data"]) > 301:
                raise RuntimeError("Teacher collection exceeded its task horizon")
            if outcome:
                results.append(dict(seed=row["seed"], outcome=int(outcome), data=row["data"]))
            else:
                remaining.append(row)
        pending = remaining
    return sorted(results, key=lambda r: r["seed"])


def build_dataset(teacher_path, output, episodes=512):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    teacher = PhysicalPolicy("arboids", teacher_path)
    seeds = list(range(1500000, 1500000 + episodes))
    started = time.perf_counter()
    base = collect_teacher(teacher, seeds)
    failures = [r["seed"] for r in base if r["outcome"] == 2]
    print(f"[DATA] base_episodes={episodes} collisions={len(failures)}", flush=True)
    corrected = {}
    query_steps = 0
    # The search grid and its order are fixed before collection. A later option
    # never replaces an earlier successful correction of the same initial state.
    for gate in (1., 0.):
        attempts = collect_teacher(teacher, failures, override=gate) if failures else []
        query_steps += sum(len(r["data"]) for r in attempts)
        for row in attempts:
            if row["outcome"] in (3, 4) and row["seed"] not in corrected:
                corrected[row["seed"]] = row
        print(f"[DATA] corrective_gate={gate} successful_corrections={len(corrected)}", flush=True)
    anchors = [sample for row in base if row["outcome"] in (3, 4) for sample in row["data"]]
    special = [sample for row in corrected.values() for sample in row["data"]]
    if not anchors:
        raise RuntimeError("No successful base demonstrations")
    data = anchors + special
    metadata = dict(protocol="channel-evidence-v1", teacher=teacher.metadata(),
                    seed_start=1500000, episodes=episodes, base_steps=sum(len(r["data"]) for r in base),
                    corrective_query_steps=query_steps, anchor_samples=len(anchors),
                    corrective_samples=len(special), corrective_episodes=len(corrected),
                    base_collisions=len(failures), corrective_gate_grid=[1., 0.],
                    base_outcomes=[dict(seed=r["seed"], outcome=r["outcome"]) for r in base],
                    corrected_seeds=list(corrected), wall_seconds=time.perf_counter() - started)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(frames=collate_frames([x[0] for x in data]),
                    targets=torch.as_tensor(np.asarray([x[1] for x in data]), dtype=torch.float32),
                    metadata=metadata), output)
    output.with_suffix(".json").write_text(json.dumps(dict(metadata, sha256=sha256(output)), indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("base_outcomes", "teacher", "corrected_seeds")}), flush=True)


def pretrain(config, data_path, output, device, updates=3600):
    set_seed(config["seed"])
    torch.set_num_threads(1)
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    frames = {key: value.to(device) for key, value in data["frames"].items()}
    if not config["policy"]["motion_features"]:
        frames = mask_motion(frames)
    targets = data["targets"].to(device)
    anchor_n = data["metadata"]["anchor_samples"]
    special_n = data["metadata"]["corrective_samples"]
    use_special = config["experiment"]["arm"] != "no_safety_demonstrations" and special_n > 0
    agent = MAPPO(config, device)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in agent.actor.parameters())).hexdigest()
    with torch.no_grad():
        for head, dim, std in ((agent.actor.proposal_head[-1], 2, .15),
                               (agent.actor.gate_head[-1], agent.actor.gate_dim, .25)):
            head.weight[dim:].zero_()
            head.bias[dim:].fill_(math.log(std))
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=.0003, eps=1e-5)
    started = time.perf_counter()
    for update in range(updates):
        if use_special:
            index = torch.cat((torch.randint(anchor_n, (192,), device=device),
                               anchor_n + torch.randint(special_n, (64,), device=device)))
        else:
            index = torch.randint(anchor_n, (256,), device=device)
        result = agent.actor.act(select_frame(frames, index), deterministic=True)
        loss = ((result["executed"] - targets[index]) / THRUST_SCALE).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite demonstration loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if (update + 1) % 600 == 0:
            print(f"[PRETRAIN] update={update + 1}/{updates} mse={loss.item():.6f}", flush=True)
    # Fixed final imitation update; no best-round choice or per-arm extra tuning.
    agent.trainer_state["pretraining"] = dict(dataset_sha256=sha256(data_path),
        updates=updates, samples_per_update=256, initialization_sha256=initial_hash,
        training_seed=config["seed"], safety_demonstrations=use_special,
        dataset=data["metadata"], wall_seconds=time.perf_counter() - started)
    agent.save(output)
    return output


def run_one(arm, seed, root, data_path, device, steps=None, pretrain_updates=3600):
    config = config_for(arm, seed)
    run_id = f"{arm}-seed{seed}"
    directory = root / run_id
    if (directory / "complete.json").exists():
        raise FileExistsError(f"Completed experiment: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.yaml"
    resume = directory / "channel-mappo1.pth"
    init = directory / "initialization.pth"
    if resume.exists():
        old = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if old != config:
            raise RuntimeError("Saved experiment configuration differs from the locked protocol")
    else:
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        if not init.exists():
            pretrain(config, data_path, init, device, pretrain_updates)
    # Reset every training seed after the fixed initialization budget. Initial
    # critic weights and stochastic training streams are independent per seed.
    set_seed(seed)
    experiment = ExperimentManager(config, str(root), run_id, repeat_idx=1)
    agent = train(config, experiment, device, resume=resume if resume.exists() else None,
                  initialize_actor=None if resume.exists() else init, max_steps=steps)
    if agent.training_steps >= config["training"]["total_steps"]:
        model = directory / "channel-mappo-best1.pth"
        completion = dict(passed=True, arm=arm, seed=seed, steps=agent.training_steps,
                          selected_step=agent.trainer_state["best_step"], sha256=sha256(model),
                          initialization_sha256=sha256(init), dataset_sha256=sha256(data_path),
                          wall_seconds=agent.trainer_state["elapsed_seconds"], torch=torch.__version__,
                          python=sys.version, device=str(device))
        (directory / "complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")


def schedule(root, data_path, jobs, device):
    if jobs < 1:
        raise ValueError("Concurrency must be positive")
    root.mkdir(parents=True, exist_ok=True)
    order = [(arm, seed) for seed in SEEDS for arm in ARMS]
    rng = np.random.default_rng(481516)
    # Balance seed blocks and randomize arm order within each block.
    order = [(arm, seed) for seed in SEEDS for arm in rng.permutation(ARMS)]
    queue, active, failures = list(order), [], []
    all_started = time.time()
    while queue or active:
        limit_file = root / "concurrency.json"
        limit = int(json.loads(limit_file.read_text())["jobs"]) if limit_file.exists() else jobs
        if not 1 <= limit <= 12:
            raise ValueError("Explicit concurrency setting must be between 1 and 12")
        while queue and len(active) < limit:
            arm, seed = queue.pop(0)
            directory = root / f"{arm}-seed{seed}"
            if (directory / "complete.json").exists():
                continue
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "training.log").open("a", encoding="utf-8")
            command = [sys.executable, "-X", "utf8", "-u", __file__, "run", "--arm", arm,
                       "--seed", str(seed), "--root", str(root), "--dataset", str(data_path), "--device", device]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg"))
            active.append((arm, seed, process, log))
            print(f"[START] {arm} seed={seed} pid={process.pid}", flush=True)
        remaining = []
        for arm, seed, process, log in active:
            code = process.poll()
            if code is None:
                remaining.append((arm, seed, process, log))
            else:
                log.close()
                complete = root / f"{arm}-seed{seed}" / "complete.json"
                if code or not complete.exists():
                    failures.append(dict(arm=arm, seed=seed, returncode=code))
                    # Do not turn implementation/infrastructure errors into
                    # silent failed seeds or continue consuming the full budget.
                    queue.clear()
                print(f"[END] {arm} seed={seed} returncode={code} complete={complete.exists()}", flush=True)
        active = remaining
        status = dict(protocol="channel-evidence-v1", total=len(order), queued=len(queue),
                      concurrency=limit,
                      active=[dict(arm=a, seed=s, pid=p.pid) for a, s, p, _ in active],
                      completed=sum((root / f"{a}-seed{s}" / "complete.json").exists() for a, s in order),
                      failures=failures, started_unix=all_started, updated_unix=time.time())
        temporary = root / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(root / "status.json")
        if active:
            time.sleep(10)
    if failures:
        raise RuntimeError(f"Experiment jobs failed: {failures}")
    (root / "training-complete.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("dataset")
    collect.add_argument("--teacher", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--episodes", type=int, default=512)
    single = sub.add_parser("run")
    single.add_argument("--arm", choices=ARMS, required=True)
    single.add_argument("--seed", type=int, required=True)
    single.add_argument("--steps", type=int)
    single.add_argument("--pretrain-updates", type=int, default=3600)
    matrix = sub.add_parser("schedule")
    matrix.add_argument("--jobs", type=int, default=4)
    for cmd in (single, matrix):
        cmd.add_argument("--root", type=Path, required=True)
        cmd.add_argument("--dataset", type=Path, required=True)
        cmd.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "dataset":
        build_dataset(args.teacher, args.output, args.episodes)
    elif args.command == "run":
        run_one(args.arm, args.seed, args.root, args.dataset, args.device, args.steps, args.pretrain_updates)
    else:
        schedule(args.root, args.dataset, args.jobs, args.device)


if __name__ == "__main__":
    main()
