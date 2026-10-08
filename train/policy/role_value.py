"""Conditional task values for residual capture maneuvers around an intact prior."""
from functools import lru_cache
from itertools import combinations

import numpy as np
import torch
from torch import nn

from policy.role_residual import RoleResidualPolicy, public_target


@lru_cache(maxsize=12)
def compositions(n):
    masks = [np.zeros(n, dtype=np.float32)]
    for count in (1, 2):
        for group in combinations(range(n), count):
            mask = np.zeros(n, dtype=np.float32)
            mask[list(group)] = 1.
            masks.append(mask)
    return np.stack(masks)


def features(packet, controls):
    state = np.asarray(packet['motion'])
    target, velocity = public_target(packet)
    radial = target / max(np.linalg.norm(target), 1e-9)
    frame = np.column_stack((radial, [-radial[1], radial[0]]))
    heading = np.arctan2(radial[1], radial[0])
    prior, pursuit, eta = controls
    return np.column_stack(((state[:, :2]-target)@frame/30.,
        state[:, 3:5]@frame/5., np.cos(state[:, 2]-heading), np.sin(state[:, 2]-heading),
        state[:, 5], np.broadcast_to(velocity@frame/5., (len(state), 2)),
        np.full(len(state), np.linalg.norm(target)/60.), prior[:, :2], pursuit[:, :2],
        eta/20., np.full(len(state), len(state)/8.))).astype(np.float32)


class ConditionalRoleValue(nn.Module):
    """A shared unary value plus symmetric conditional pair contributions."""
    def __init__(self, interaction=True):
        super().__init__()
        self.interaction = interaction
        self.encoder = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.baseline = nn.Sequential(nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
        self.unary = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.pair = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x, masks):
        h = self.encoder(x)
        global_h = h.mean(-2)
        unary = self.unary(torch.cat((h, global_h[:, None].expand_as(h)), -1)).squeeze(-1)
        value = self.baseline(global_h) + torch.einsum('bn,kn->bk', unary, masks)
        if self.interaction:
            i, j = torch.triu_indices(x.shape[1], x.shape[1], 1, device=x.device)
            pair = self.pair(torch.cat((h[:, i]+h[:, j], torch.abs(h[:, i]-h[:, j])), -1)).squeeze(-1)
            value = value + torch.einsum('bp,kp->bk', pair, masks[:, i]*masks[:, j])
        return value


class LearnedRoleResidual(RoleResidualPolicy):
    def __init__(self, model, minimum_gain=1., block_steps=10):
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.minimum_gain, self.block_steps = minimum_gain, block_steps
        self.remaining, self.coefficients = 0, None
        self.decisions, self.interventions = 0, 0

    @torch.no_grad()
    def choose_action(self, packet, deterministic=True):
        controls = self.candidate_controls(packet)
        if not self.remaining:
            masks = compositions(len(packet['obs']))
            values = self.model(torch.from_numpy(features(packet, controls))[None],
                                torch.from_numpy(masks))[0].numpy()
            choice = int(np.argmax(values))
            if (values[choice]-values[0])*20. < self.minimum_gain:
                choice = 0
            self.coefficients = masks[choice]
            self.remaining = self.block_steps
            self.decisions += 1
            self.interventions += int(choice != 0)
        self.remaining -= 1
        prior, pursuit, _ = controls
        return (prior+self.coefficients[:, None]*(pursuit-prior)).astype(np.float32), 0.
