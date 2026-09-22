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

    def append(self, frame, next_frame, decision, rewards, outcome, value, next_value, snapshot=None):
        self.rows.append(dict(frame=frame, next_frame=next_frame,
                              decision={k: decision[k].detach().cpu().numpy()[0].copy() for k in DECISION_KEYS},
                              rewards=np.asarray(rewards).copy(), reward=float(np.mean(rewards)),
                              terminal=bool(outcome), outcome=int(outcome), value=float(value),
                              next_value=float(next_value), snapshot=snapshot))

    def batch(self, device, gamma, lam):
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
        advantage, returns = compute_gae(rewards, [r["value"] for r in self.rows],
                                        [r["next_value"] for r in self.rows], terminals, gamma, lam)
        advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-8)
        return dict(frame=frames, next_frame=next_frames, decision=decisions,
                    reward=torch.tensor(rewards, device=device).unsqueeze(-1),
                    terminal=torch.tensor(terminals, device=device).unsqueeze(-1),
                    advantage=torch.tensor(advantage, device=device),
                    returns=torch.tensor(returns, device=device).unsqueeze(-1))
