"""Common physical-command evaluation for channel policies, ARBoids and Boids.

Evaluation seed streams are local to each world, independent of batch size.
The return is the sum of per-step mean team reward, excluding reset rewards.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from RL.Networks import ActorAdap
from RL.control import fuse_thrust, to_thrust
from RL.deployment import DeployedPolicy
from RL.guidance import preserve_rng
from RL.observations import collate_frames
from train_mappo import make_env, minimum_spacing, summarize


ENVIRONMENT = dict(protocol="paper-parameters-v1", total_time=60.,
                   agility_noise_half_width=.5, initial_min_spacing=6.,
                   canonical_agent_order=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PhysicalPolicy:
    def __init__(self, kind, checkpoint=None, device="cpu"):
        self.kind, self.device = kind, torch.device(device)
        self.checkpoint = str(Path(checkpoint).resolve()) if checkpoint else None
        self.motion_features = True
        if kind == "channel":
            deployed = DeployedPolicy.load(checkpoint, device)
            self.actor = deployed.actor
            self.motion_features = self.actor.config["motion_features"]
        elif kind == "arboids":
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.actor = ActorAdap(6, 8, 3, weights["l2.weight"].shape[0]).to(device).eval()
            self.actor.load_state_dict(weights)
        elif kind == "boids":
            self.actor = None
        else:
            raise ValueError(f"Unknown physical policy: {kind}")

    @torch.no_grad()
    def act(self, frames, observations):
        if self.kind == "channel":
            return self.actor.act(collate_frames(frames, self.device), deterministic=True)["executed"].cpu().numpy()
        boids = np.asarray([f["boids"] for f in frames])
        if self.kind == "boids":
            return boids
        count, n, width = np.asarray(observations).shape
        raw = self.actor(torch.as_tensor(np.asarray(observations).reshape(-1, width),
                                        dtype=torch.float32, device=self.device), True, False)[0]
        raw = raw.reshape(count, n, 3)
        return fuse_thrust(torch.as_tensor(boids, device=self.device), to_thrust(raw[..., :2]),
                           raw[..., 2:3], "scalar").cpu().numpy()

    def metadata(self):
        return dict(kind=self.kind, checkpoint=self.checkpoint,
                    checkpoint_sha256=sha256(self.checkpoint) if self.checkpoint else None,
                    parameters=sum(p.numel() for p in self.actor.parameters()) if self.actor else 0,
                    device=str(self.device))


def configure_current(env, scale):
    original = env.generate_random_current
    env.generate_random_current = lambda: scale * original()


@torch.no_grad()
def evaluate(policy, seeds, *, defenders=3, agility=2., parallel=8,
             current_scale=1., delay_steps=0, duration=60.):
    seeds = [int(seed) for seed in seeds]
    if len(seeds) != len(set(seeds)) or not seeds or min(defenders, parallel) < 1:
        raise ValueError("Use unique seeds and positive episode, team and batch sizes")
    if delay_steps < 0 or current_scale < 0:
        raise ValueError("Delay and current scale must be nonnegative")
    config = dict(environment=dict(ENVIRONMENT, total_time=duration))
    pending, rows, next_episode = [], [], 0
    inference_seconds, inference_calls = 0., 0
    started = time.perf_counter()
    with preserve_rng():
        while len(rows) < len(seeds):
            while len(pending) < parallel and next_episode < len(seeds):
                seed = seeds[next_episode]
                np.random.seed(seed)
                env = make_env(config, defenders)
                configure_current(env, current_scale)
                observation, _ = env.reset(agility, noisy_agility=False)
                initial = np.r_[env.attacker.pos, env.attacker.theta,
                                np.asarray([d.pos for d in env.defender_list]).ravel(),
                                [d.theta for d in env.defender_list]]
                pending.append(dict(env=env, observation=observation, seed=seed,
                                    rng=np.random.get_state(), steps=0, reward=0.,
                                    spacing=minimum_spacing(env), energy=0.,
                                    initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                                    delay=deque([np.zeros((defenders, 2)) for _ in range(delay_steps)])))
                next_episode += 1
            frames = [r["env"].structured_frame(policy.motion_features) for r in pending]
            tick = time.perf_counter()
            commands = policy.act(frames, [r["observation"] for r in pending])
            inference_seconds += time.perf_counter() - tick
            inference_calls += 1
            remaining = []
            for index, item in enumerate(pending):
                env = item["env"]
                np.random.set_state(item["rng"])
                item["delay"].append(commands[index, :defenders].copy())
                command = item["delay"].popleft()
                observation, rewards, outcome, info = env.step_thrust(command)
                if not np.allclose(command, info["executed_thrust"], atol=1e-5, rtol=0.):
                    raise AssertionError("Policy command and actuator input differ")
                item["rng"], item["observation"] = np.random.get_state(), observation
                item["steps"] += 1
                item["reward"] += float(np.mean(rewards))
                item["energy"] += float(np.mean(command ** 2)) * env.Action_T
                item["spacing"] = min(item["spacing"], minimum_spacing(env))
                if item["steps"] > int(np.ceil(duration / env.Action_T)) + 1:
                    raise RuntimeError("Evaluation exceeded the episode horizon")
                if not outcome:
                    remaining.append(item)
                    continue
                rows.append(dict(seed=item["seed"], defenders=defenders, agility=agility,
                                 current_scale=current_scale, delay_steps=delay_steps,
                                 outcome_code=int(outcome), success=int(outcome in (3, 4)),
                                 steps=item["steps"], time_seconds=env.Current_T,
                                 reward=item["reward"], mean_step_reward=item["reward"] / item["steps"],
                                 min_spacing=item["spacing"], thrust_squared_integral=item["energy"],
                                 initial_sha256=item["initial_sha256"]))
            pending = remaining
    rows.sort(key=lambda r: r["seed"])
    return dict(passed=True, policy=policy.metadata(), rows=rows, summary=summarize(rows),
                environment=config["environment"], wall_seconds=time.perf_counter() - started,
                batched_inference_seconds=inference_seconds, inference_calls=inference_calls,
                parallel=parallel, reward_definition="sum_t mean_i r_it; reset reward excluded",
                energy_definition="mean squared thrust integrated over seconds; not electrical energy")


def suites(cohort="frozen"):
    base = 2000000 if cohort == "frozen" else 9000000
    result = [dict(name="nominal", seed=base, episodes=2048)]
    for index, agility in enumerate((1.5, 1.75, 2.25, 2.5, 2.75, 3.)):
        result.append(dict(name=f"agility-{agility}", seed=base + 10000 + index * 1000,
                           episodes=256, agility=agility))
    for index, defenders in enumerate((2, 4, 5, 6, 8)):
        result.append(dict(name=f"team-{defenders}", seed=base + 20000 + index * 1000,
                           episodes=256, defenders=defenders))
    for index, scale in enumerate((2., 3.)):
        result.append(dict(name=f"current-{scale}", seed=base + 30000 + index * 1000,
                           episodes=256, current_scale=scale))
    for index, delay in enumerate((1, 2)):
        result.append(dict(name=f"delay-{delay}", seed=base + 40000 + index * 1000,
                           episodes=256, delay_steps=delay))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("channel", "arboids", "boids"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cohort", choices=("frozen", "matrix"), default="frozen")
    parser.add_argument("--suite", default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = PhysicalPolicy(args.kind, args.checkpoint, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    available = suites(args.cohort)
    selected = available if args.suite == "all" else [s for s in available if s["name"] == args.suite]
    if not selected:
        parser.error("Unknown evaluation suite")
    for spec in selected:
        path = args.output / f'{spec["name"]}.json'
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("passed") and old["policy"] == policy.metadata() and old.get("suite") == spec:
                print(f"[CACHED] {path}", flush=True)
                continue
            raise RuntimeError(f"Existing evaluation does not match frozen inputs: {path}")
        options = {k: v for k, v in spec.items() if k not in ("name", "seed", "episodes")}
        result = evaluate(policy, range(spec["seed"], spec["seed"] + spec["episodes"]),
                          parallel=args.parallel, **options)
        result["suite"] = spec
        result["evaluator_sha256"] = sha256(__file__)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(dict(suite=spec["name"], **result["summary"], wall_seconds=result["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
