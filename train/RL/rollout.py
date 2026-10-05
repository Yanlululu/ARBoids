"""On-policy team trajectories. Counterfactual branches never enter this buffer."""
import numpy as np
import torch
from .observations import collate_frames


DECISION_KEYS = ("candidate_z", "candidate", "gate_z", "gates", "messages",
                 "logp_candidate", "logp_gate", "executed")


def team_mean(values, mask):
    if values.ndim == 3:
        values = values.squeeze(-1)
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def compute_gae(rewards, values, next_values, terminals, gamma, lam):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        live = 1.0 - float(terminals[i])
        delta = rewards[i] + gamma * live * next_values[i] - values[i]
        carry = delta + gamma * lam * live * carry
        advantages[i] = carry
    return advantages, advantages + np.asarray(values, dtype=np.float32)


class TeamRollout:
    def __init__(self):
        self.rows = []

    def append(self, frame, next_frame, decision, rewards, outcome, value, next_value, snapshot=None, stream_id=0):
        self.rows.append(dict(frame=frame, next_frame=next_frame,
                              decision={k: decision[k].detach().cpu().numpy()[0].copy() for k in DECISION_KEYS},
                              rewards=np.asarray(rewards).copy(), reward=float(np.mean(rewards)),
                              terminal=bool(outcome), outcome=int(outcome), value=float(value),
                              next_value=float(next_value), snapshot=snapshot, stream_id=int(stream_id)))

    def batch(self, device, gamma, lam, cost_lam=.99):
        frames = collate_frames([r["frame"] for r in self.rows], device)
        next_frames = collate_frames([r["next_frame"] for r in self.rows], device)
        max_n = frames["mask"].shape[1]
        decisions = {}
        for key in DECISION_KEYS:
            example = self.rows[0]["decision"][key]
            shape = list(example.shape)
            shape[0] = max_n
            if key == "messages":
                shape[1] = max_n
            array = np.zeros((len(self.rows), *shape), dtype=np.float32)
            for i, row in enumerate(self.rows):
                value = row["decision"][key]
                array[(i, *(slice(0, size) for size in value.shape))] = value
            decisions[key] = torch.as_tensor(array, device=device)
        rewards = [r["reward"] for r in self.rows]
        terminals = [r["terminal"] for r in self.rows]
        advantage, returns = np.zeros(len(self.rows), dtype=np.float32), np.zeros(len(self.rows), dtype=np.float32)
        cost_advantage, cost_returns = np.zeros_like(advantage), np.zeros_like(returns)
        # Vector environments are interleaved in the buffer. GAE must follow
        # each environment's own time axis, including its rollout bootstrap.
        streams = np.asarray([r.get("stream_id", 0) for r in self.rows])
        for stream in np.unique(streams):
            indices = np.flatnonzero(streams == stream)
            selected = [self.rows[i] for i in indices]
            advantage[indices], returns[indices] = compute_gae(
                [r["reward"] for r in selected], [r["value"] for r in selected],
                [r["next_value"] for r in selected], [r["terminal"] for r in selected], gamma, lam)
            cost_advantage[indices], cost_returns[indices] = compute_gae(
                [float(r["outcome"] == 2) for r in selected], [r.get("cost_value", 0.) for r in selected],
                [r.get("next_cost_value", 0.) for r in selected], [r["terminal"] for r in selected], 1., cost_lam)
        advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-8)
        return dict(frame=frames, next_frame=next_frames, decision=decisions,
                    reward=torch.tensor(rewards, device=device).unsqueeze(-1),
                    terminal=torch.tensor(terminals, device=device).unsqueeze(-1),
                    advantage=torch.tensor(advantage, device=device),
                    cost_advantage=torch.tensor(cost_advantage, device=device),
                    cost_returns=torch.tensor(cost_returns, device=device).unsqueeze(-1),
                    returns=torch.tensor(returns, device=device).unsqueeze(-1))
