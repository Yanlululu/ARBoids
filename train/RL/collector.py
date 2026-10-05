"""Batched policy inference over independent simulator instances."""
import heapq
import numpy as np
import torch
from .control import blend_thrust, to_thrust
from .observations import collate_frames
from .rollout import TeamRollout, DECISION_KEYS


class CollisionStarts:
    """Revisit training initial conditions, always collecting a new policy rollout."""
    def __init__(self, reset_env, probability=0., capacity=128, solved_after=3, state=None,
                 excluded_seed_ranges=()):
        if not 0. <= probability <= 1. or capacity < 1 or solved_after < 1:
            raise ValueError("Invalid collision-start curriculum settings")
        self.reset_env, self.probability = reset_env, probability
        self.capacity, self.solved_after = capacity, solved_after
        self.excluded_seed_ranges = tuple((int(lo), int(hi)) for lo, hi in excluded_seed_ranges)
        if any(not 0 <= lo < hi <= 2 ** 31 for lo, hi in self.excluded_seed_ranges):
            raise ValueError("Invalid excluded training seed interval")
        self.cases = dict((int(seed), int(streak)) for seed, streak in (state or {}).get("cases", []))
        if any(self.excluded(seed) for seed in self.cases):
            raise ValueError("A held-out seed is present in the training curriculum")
        while len(self.cases) > capacity:
            self.cases.pop(next(iter(self.cases)))

    def excluded(self, seed):
        return any(lo <= seed < hi for lo, hi in self.excluded_seed_ranges)

    def reset(self):
        if not self.probability and not self.excluded_seed_ranges:
            return self.reset_env(), None, False
        revisit = bool(self.cases) and np.random.random() < self.probability
        seed = int(np.random.choice(list(self.cases))) if revisit else int(np.random.randint(2 ** 31))
        while self.excluded(seed):
            seed = int(np.random.randint(2 ** 31))
        # Only initialization is repeated. Future currents and policy samples
        # use the continuing training RNG, rather than the previous trajectory.
        current_rng = np.random.get_state()
        try:
            np.random.seed(seed)
            env = self.reset_env()
        finally:
            np.random.set_state(current_rng)
        return env, seed, revisit

    def observe(self, seed, outcome):
        if seed is None or not self.probability:
            return
        if outcome == 2:
            self.cases[seed] = 0
            while len(self.cases) > self.capacity:
                self.cases.pop(next(iter(self.cases)))
        elif seed in self.cases:
            # A breach is not a solved collision case. Require consecutive
            # successful defenses before removing it from the difficult starts.
            self.cases[seed] = self.cases[seed] + 1 if outcome in (3, 4) else 0
            if self.cases[seed] >= self.solved_after:
                del self.cases[seed]

    def state_dict(self):
        return {"cases": list(self.cases.items())}


def encounter_priority(frame, horizon=3., lead_time=0., collision_radius=5.):
    """Rank approaching pairs for training queries, without changing actions."""
    position = frame["edges"][..., :2] * 60.
    velocity = frame["edges"][..., 2:4] * 5.
    dot = (position * velocity).sum(-1)
    speed2 = (velocity * velocity).sum(-1)
    time = np.clip(-dot / np.maximum(speed2, 1e-8), 0., horizon)
    closest = np.linalg.norm(position + time[..., None] * velocity, axis=-1)
    score = np.exp(-.5 * (closest / 6.) ** 2 - time / horizon)
    if lead_time:
        if not 0. < lead_time < horizon:
            raise ValueError("Priority lead time must lie inside the prediction horizon")
        # Closest approach occurs after entry into the collision radius. Rank
        # queries by time to that entry, giving the slow hull time to react.
        # This only selects simulator queries; it never filters policy actions.
        offset = (position * position).sum(-1) - collision_radius ** 2
        discriminant = dot * dot - speed2 * offset
        entry = (-dot - np.sqrt(np.maximum(discriminant, 0.))) / np.maximum(speed2, 1e-8)
        event_time = np.where((discriminant >= 0.) & (entry >= 0.), entry, time)
        score = np.exp(-.5 * (closest / 6.) ** 2 - .5 * ((event_time - lead_time) / .75) ** 2)
    score *= frame["neighbors"] & (dot < 0.)
    return float(score.max(initial=0.))


class TeamCollector:
    def __init__(self, agent, reset_env, num_envs=1):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.agent, self.reset_env = agent, reset_env
        training = agent.config.get("training", {})
        self.starts = CollisionStarts(reset_env, training.get("collision_revisit_probability", 0.),
                                      int(training.get("collision_revisit_capacity", 128)),
                                      int(training.get("collision_revisit_solved_after", 3)),
                                      agent.trainer_state.get("collision_curriculum"),
                                      training.get("excluded_seed_ranges", ()))
        initial = [self.starts.reset() for _ in range(num_envs)]
        self.envs, self.episode_seeds = [x[0] for x in initial], [x[1] for x in initial]

    @torch.no_grad()
    def collect(self, steps, snapshot_indices=(), priority_snapshots=0, priority_lead_time=0.):
        agent = self.agent
        rollout = TeamRollout()
        projected = commands = episodes = revisits = 0
        motion = agent.actor.config["motion_features"]
        if priority_snapshots < 0:
            raise ValueError("priority_snapshots must be nonnegative")
        priority = []
        while len(rollout.rows) < steps:
            active = self.envs[:min(len(self.envs), steps - len(rollout.rows))]
            frames = [env.structured_frame(motion) for env in active]
            frame = collate_frames(frames, agent.device)
            decision = agent.actor.act(frame)
            values = agent.value(frame).flatten().cpu().numpy()
            cost_values = agent.collision_probability(frame).flatten().cpu().numpy() if agent.cost_value else None
            raw = blend_thrust(frame["boids"], to_thrust(decision["candidate"]), decision["gates"], agent.actor.fusion)
            projected += int(((raw - decision["executed"]).abs().gt(1e-4).any(-1) & frame["mask"]).sum())
            # One transfer per tensor for the whole batch, not per vessel/env.
            decisions = {key: decision[key].detach().cpu() for key in DECISION_KEYS}
            pending = []
            for i, env in enumerate(active):
                n = env.defender_num
                timestamp = env.Current_T
                snapshot = env.snapshot() if len(rollout.rows) + i in snapshot_indices else None
                index = len(rollout.rows) + i
                if priority_snapshots and snapshot is None:
                    risk = encounter_priority(frames[i], lead_time=priority_lead_time,
                                              collision_radius=env.Collision_R)
                    if risk > 0 and (len(priority) < priority_snapshots or risk > priority[0][0]):
                        item = (risk, index, env.snapshot())
                        if len(priority) < priority_snapshots:
                            heapq.heappush(priority, item)
                        else:
                            heapq.heapreplace(priority, item)
                data = {key: value[i:i + 1, :n, :n] if key == "messages" else value[i:i + 1, :n]
                        for key, value in decisions.items()}
                executed = data["executed"][0].numpy()
                _, rewards, outcome, info = env.step_thrust(executed)
                if not np.allclose(executed, info["executed_thrust"], atol=1e-5, rtol=0.):
                    raise AssertionError("Recorded policy command does not match actuator input")
                data["executed"] = torch.as_tensor(info["executed_thrust"], dtype=torch.float32).unsqueeze(0)
                pending.append((data, rewards, outcome, snapshot, timestamp))
                commands += n
            next_frames = [env.structured_frame(motion) for env in active]
            next_frame = collate_frames(next_frames, agent.device)
            next_values = agent.value(next_frame).flatten().cpu().numpy()
            next_cost_values = agent.collision_probability(next_frame).flatten().cpu().numpy() if agent.cost_value else None
            for i, (data, rewards, outcome, snapshot, timestamp) in enumerate(pending):
                rollout.append(frames[i], next_frames[i], data, rewards, outcome, values[i],
                               0. if outcome else next_values[i], snapshot, stream_id=i)
                rollout.rows[-1]["timestamp"] = timestamp
                if cost_values is not None:
                    rollout.rows[-1].update(cost_value=float(cost_values[i]),
                                           next_cost_value=0. if outcome else float(next_cost_values[i]))
                agent.training_steps += 1
                if outcome:
                    episodes += 1
                    self.starts.observe(self.episode_seeds[i], outcome)
                    self.envs[i], self.episode_seeds[i], revisited = self.starts.reset()
                    revisits += int(revisited)
        for _, index, snapshot in priority:
            rollout.rows[index]["snapshot"] = snapshot
        if self.starts.probability:
            agent.trainer_state["collision_curriculum"] = self.starts.state_dict()
        return rollout, dict(projection_fraction=projected / max(1, commands), completed_episodes=episodes,
                             priority_snapshots=len(priority), collision_revisit_episodes=revisits,
                             collision_start_cases=len(self.starts.cases))
