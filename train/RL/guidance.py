"""Bounded joint gate improvement, simulator validation and separate Q labels."""
import copy
import itertools
import random
from collections import deque
from contextlib import contextmanager
import numpy as np
import torch
from .control import fuse_thrust, to_thrust, THRUST_SCALE
from .observations import tensor_frame


@contextmanager
def preserve_rng():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def search_gates(q, frame, boids, learned, initial, *, fusion="channels", step=0.15,
                 radius=0.25, sweeps=1, penalty=0.1, mode="joint"):
    """Search one state; every trial is scored as a complete joint action.

    Independent ablation chooses each coordinate against the same original
    plan. The joint variant evaluates later coordinates against accepted edits.
    Ordering is geometric, never by arbitrary vessel index.
    """
    if frame["mask"].shape[0] != 1 or mode not in ("joint", "independent"):
        raise ValueError("Search expects one team and joint/independent mode")
    if step <= 0 or radius <= 0 or sweeps < 1 or penalty < 0:
        raise ValueError("Invalid bounded-search settings")
    original = initial.clone()
    current = original.clone()
    u0 = fuse_thrust(boids, learned, original, fusion)
    valid = frame["mask"]
    count = valid.sum().clamp_min(1) * 2
    q_calls = 0

    def score(gates):
        nonlocal q_calls
        trials = gates.shape[0]
        repeated = {k: v.expand(trials, *v.shape[1:]) for k, v in frame.items()}
        u = fuse_thrust(boids, learned, gates, fusion)
        values = q(repeated, u).squeeze(-1)
        distance = ((((u - u0) / THRUST_SCALE) ** 2) * valid.unsqueeze(-1)).sum((-2, -1)) / count
        q_calls += trials
        return values - penalty * distance, values, u

    initial_score, initial_q, _ = score(original)
    positions = frame["nodes"][0, :, :2].cpu().numpy()
    order = sorted(torch.where(valid[0])[0].tolist(), key=lambda i: tuple(positions[i]))
    offsets = torch.tensor(list(itertools.product((-step, 0., step), repeat=original.shape[-1])),
                           device=original.device, dtype=original.dtype)
    lower, upper = (original - radius).clamp(0., 1.), (original + radius).clamp(0., 1.)
    for _ in range(sweeps if mode == "joint" else 1):
        for i in order:
            base = current if mode == "joint" else original
            trials = base.expand(len(offsets), -1, -1).clone()
            trials[:, i] = torch.maximum(torch.minimum(base[:, i] + offsets, upper[:, i]), lower[:, i])
            # Include the exact unchanged team first, preserving it on score ties.
            trials = torch.cat((base, trials), dim=0)
            scores, _, _ = score(trials)
            best = int(scores.argmax())
            if mode == "joint":
                current = trials[best:best + 1].clone()
            else:
                current[:, i] = trials[best, i]
    final_score, final_q, result = score(current)
    return dict(gates=current, executed=result, baseline=u0,
                predicted_gain=float(final_q - initial_q),
                objective_gain=float(final_score - initial_score), q_evaluations=q_calls)


@torch.no_grad()
def branch_return(env, snapshot, first_action, actor, value, gamma, horizon, device):
    """First action is intervened on; subsequent actions use the frozen policy."""
    env.restore(snapshot)
    total, steps, terminal = 0., 0, False
    initial_frame = env.structured_frame(actor.config["motion_features"])
    actual_first = None
    for k in range(horizon):
        if k == 0:
            action = first_action
        else:
            frame = tensor_frame(env.structured_frame(actor.config["motion_features"]), device)
            action = actor.act(frame)["executed"][0].cpu().numpy()
        _, reward, outcome, info = env.step_thrust(action)
        if k == 0:
            actual_first = info["executed_thrust"].copy()
        total += gamma ** k * float(np.mean(reward))
        steps += 1
        if outcome:
            terminal = True
            break
    if not terminal:
        final_frame = tensor_frame(env.structured_frame(actor.config["motion_features"]), device)
        total += gamma ** steps * float(value(final_frame).item())
    return total, steps, (initial_frame, actual_first, total)


class TeamGuidance:
    def __init__(self, options):
        self.options = options
        self.validation_history = deque(maxlen=int(options.get("validation_window", 32)))

    def prepare(self, agent, rollout, batch, env):
        opts = self.options
        targets = batch["decision"]["executed"].detach().clone()
        weights = torch.zeros(len(rollout.rows), device=agent.device)
        stats = dict(teacher_trials=0, teacher_predicted_gain=0., teacher_branch_gain=0.,
                     teacher_accepted=0, branch_steps=0, q_search_evaluations=0,
                     guidance_suspended=0)
        calibration = []
        if not opts.get("enabled", True) or agent.training_steps < int(opts.get("warm_steps", 4096)):
            return targets, weights, calibration, stats
        eligible = [(i, r) for i, r in enumerate(rollout.rows) if r["snapshot"] is not None]
        scratch = copy.deepcopy(env)
        for index, row in eligible:
            frame = tensor_frame(row["frame"], agent.device)
            decision = {k: torch.as_tensor(v, device=agent.device).unsqueeze(0)
                        for k, v in row["decision"].items()}
            with torch.no_grad():
                g0, _ = agent.actor.representative(frame, decision["candidate"], decision["messages"])
                result = search_gates(agent.q, frame, frame["boids"], to_thrust(decision["candidate"]), g0,
                                      fusion=agent.actor.fusion, step=opts.get("search_step", .15),
                                      radius=opts.get("search_radius", .25), sweeps=opts.get("search_sweeps", 1),
                                      penalty=opts.get("action_penalty", .1), mode=opts.get("search_mode", "joint"))
            stats["q_search_evaluations"] += result["q_evaluations"]
            if result["objective_gain"] <= opts.get("min_predicted_gain", 0.):
                continue
            stats["teacher_trials"] += 1
            stats["teacher_predicted_gain"] += result["predicted_gain"]
            returns = []
            # Each context starts with the same torch RNG. restore(snapshot)
            # also gives both branches identical exogenous NumPy disturbances.
            for action in (result["baseline"], result["executed"]):
                with preserve_rng():
                    value, steps, label = branch_return(
                        scratch, row["snapshot"], action[0].cpu().numpy(), agent.actor,
                        agent.value_target, agent.gamma, int(opts.get("branch_horizon", 3)), agent.device)
                returns.append(value)
                stats["branch_steps"] += steps
                calibration.append(label)
            gain = returns[1] - returns[0]
            stats["teacher_branch_gain"] += gain
            accepted = gain > opts.get("min_branch_gain", 0.)
            self.validation_history.append(accepted)
            if accepted:
                n = len(row["frame"]["mask"])
                targets[index, :n] = result["executed"][0]
                weights[index] = 1.
                stats["teacher_accepted"] += 1
        trials = max(1, stats["teacher_trials"])
        stats["teacher_predicted_gain"] /= trials
        stats["teacher_branch_gain"] /= trials
        if (len(self.validation_history) >= int(opts.get("min_validations", 8))
                and np.mean(self.validation_history) < opts.get("min_validation_rate", .5)):
            weights.zero_()
            stats["guidance_suspended"] = 1
        return targets.detach(), weights.detach(), calibration, stats
