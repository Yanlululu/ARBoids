"""Independent ARBoids SAC reproduction with a common development/test protocol.

SAC architecture and optimization retain the published-parameter preset. This
control is trained from scratch and receives no demonstration initialization.
It is distinct from the matched MAPPO scalar-fusion architecture control.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from evidence_eval import PhysicalPolicy, evaluate, ENVIRONMENT, sha256
from evidence_matrix import SEEDS, EXCLUDED
from policy.SAC import SAC, ReplayBuffer
from RL.collector import CollisionStarts
from RL.guidance import preserve_rng
from train_mappo import make_env, selection_score
from utils.config import load_config, _namespace_to_dict, _dict_to_namespace
from utils.manager import set_seed
from utils.protocol import apply_adapter_exploration


def run(seed, output, device, total=1000000):
    directory = output / f"arboids-seed{seed}"
    if directory.exists():
        raise FileExistsError(f"SAC requires a fresh run directory; prior results are retained: {directory}")
    directory.mkdir(parents=True)
    torch.set_num_threads(1)
    set_seed(seed)
    config = _namespace_to_dict(load_config(Path(__file__).parent / "configs/paper-parameters.yaml"))
    config["environment"] = dict(ENVIRONMENT)
    config["seed"] = seed
    config["training"].update(total_steps=total, eval_interval=65536, eval_episodes=128, eval_seed=610000,
                              selection_objective="collision", selection_success_floor=.9375,
                              excluded_seed_ranges=[list(r) for r in EXCLUDED])
    cfg = _dict_to_namespace(config)
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    agent = SAC(cfg, 6, 8, 3, adaptive=True, device=torch.device(device))
    buffer = ReplayBuffer(18, 3)
    steps, episodes = 0, 0
    def reset():
        env = make_env(config)
        agility = 2. + min(3, int(4 * steps / total)) * .25
        env.reset(agility, noisy_agility=True)
        return env
    starts = CollisionStarts(reset, excluded_seed_ranges=EXCLUDED)
    env, _, _ = starts.reset()
    obs, _ = env._get_obs()
    best, metrics = None, []
    elapsed = time.perf_counter()
    last = directory / "adares-final.pth"
    selected = directory / "adares-best.pth"
    while steps < total:
        if steps < cfg.training.warm_steps:
            action = np.random.uniform(-1., 1., (3, 3))
            action[:, 2] = .5 * action[:, 2] + .5
        else:
            action = agent.choose_action(obs, False).reshape(3, 3)
            action = apply_adapter_exploration(action, cfg.training)
        if not np.isfinite(action).all():
            raise FloatingPointError("SAC produced a non-finite action")
        next_obs, reward, outcome, _ = env.step(action, "AdaRes")
        for i in range(3):
            buffer.store(obs[i], action[i], reward[i], next_obs[i], bool(outcome))
        obs = next_obs
        steps += 1
        if steps >= cfg.training.warm_steps:
            agent.learn(buffer)
        if outcome:
            episodes += 1
            env, _, _ = starts.reset()
            obs, _ = env._get_obs()
        if steps % 10000 == 0:
            print(f"[SAC] seed={seed} step={steps}/{total} wall={time.perf_counter()-elapsed:.1f}s", flush=True)
        if steps % cfg.training.eval_interval == 0 or steps == total:
            if not all(torch.isfinite(p).all().item() for p in agent.actor.parameters()):
                raise FloatingPointError("Non-finite SAC checkpoint")
            agent.save(last)
            with preserve_rng():
                policy = PhysicalPolicy("arboids", last, "cpu")
                result = evaluate(policy, range(610000, 610128), parallel=8)
            score = selection_score(result["summary"], config["training"])
            if best is None or score > best:
                best = score
                agent.save(selected)
                selected_step = steps
            row = dict(step=steps, training_episodes=episodes, updates=max(0, steps-cfg.training.warm_steps+1),
                       elapsed_seconds=time.perf_counter()-elapsed, selected_step=selected_step,
                       **result["summary"])
            metrics.append(row)
            with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(metrics)
            print(f'[SAC EVAL] seed={seed} step={steps} success={row["successes"]}/128 collisions={row["collisions"]} breaches={row["breaches"]}', flush=True)
    completion = dict(passed=True, seed=seed, steps=steps, selected_step=selected_step,
                      sha256=sha256(selected), final_sha256=sha256(last),
                      wall_seconds=time.perf_counter()-elapsed, demonstrations=0,
                      algorithm="ARBoids SAC; paper numerical preset with shared environment and development protocol",
                      source_sha256=sha256(__file__), torch=str(torch.__version__))
    (directory / "complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")


def schedule(output, device):
    output.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        directory = output / f"arboids-seed{seed}"
        if (directory / "complete.json").exists():
            continue
        if directory.exists():
            raise RuntimeError(f"An interrupted SAC run requires diagnosis: {directory}")
        command = [sys.executable, "-X", "utf8", "-u", __file__, "run", "--seed", str(seed),
                   "--output", str(output), "--device", device]
        with (output / f"arboids-seed{seed}.log").open("w", encoding="utf-8") as log:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        if done.returncode:
            raise RuntimeError(f"ARBoids seed {seed} failed with exit code {done.returncode}")
        print(f"[SAC COMPLETE] seed={seed}", flush=True)
    (output / "training-complete.json").write_text(json.dumps(dict(passed=True, seeds=list(SEEDS))), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "schedule"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000000)
    args = parser.parse_args()
    if args.command == "run":
        if args.seed is None:
            parser.error("run requires --seed")
        run(args.seed, args.output, args.device, args.steps)
    else:
        schedule(args.output, args.device)
