"""Observed terminal returns for Q calibration; never a policy rollout buffer."""
from collections import deque
import random
import torch
from .observations import collate_frames


class TerminalCalibration:
    def __init__(self, capacity=128):
        if capacity < 1:
            raise ValueError("Terminal calibration capacity must be positive")
        self.groups = {outcome: deque(maxlen=capacity) for outcome in (1, 2, 3, 4)}

    def add(self, rows):
        for row in rows:
            if not row["terminal"]:
                continue
            # This observed terminal return is r with no bootstrap. In a noisy
            # simulator it is a sample, not the exact expected Q(s, action).
            # Outcome-balanced replay is a calibration objective, not an
            # unbiased replacement for the main on-policy Bellman regression.
            self.groups[row["outcome"]].append(dict(
                frame={key: torch.as_tensor(value).cpu().clone() for key, value in row["frame"].items()},
                executed=torch.as_tensor(row["decision"]["executed"]).cpu().clone(), reward=row["reward"]))

    @staticmethod
    def _batch(entries, device):
        if not entries:
            return None
        frame = collate_frames([entry["frame"] for entry in entries], device)
        actions = torch.zeros((*frame["mask"].shape, 2), device=device)
        for i, entry in enumerate(entries):
            actions[i, :len(entry["executed"])] = entry["executed"].to(device)
        rewards = torch.tensor([entry["reward"] for entry in entries], device=device).unsqueeze(-1)
        return frame, actions, rewards

    def sample(self, device, per_outcome=8):
        entries = [entry for group in self.groups.values() if group
                   for entry in random.choices(list(group), k=per_outcome)]
        return self._batch(entries, device)

    def probe(self, device):
        """Evaluate every stored label, giving each observed outcome equal mass."""
        groups = [list(group) for group in self.groups.values() if group]
        if not groups:
            return None
        batch = self._batch([entry for group in groups for entry in group], device)
        weights = torch.tensor([1. / (len(groups) * len(group)) for group in groups for _ in group],
                               device=device).unsqueeze(-1)
        return (*batch, weights)

    def state_dict(self):
        return {outcome: list(group) for outcome, group in self.groups.items()}

    def load_state_dict(self, state):
        for outcome, entries in state.items():
            self.groups[int(outcome)].clear()
            # Checkpoint loading may map tensors to CUDA; storage remains CPU.
            for entry in entries:
                self.groups[int(outcome)].append(dict(
                    frame={key: value.cpu() for key, value in entry["frame"].items()},
                    executed=entry["executed"].cpu(), reward=entry["reward"]))

    def __len__(self):
        return sum(len(group) for group in self.groups.values())
