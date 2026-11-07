"""Actual online continuation with a polynomial-tail bounded endpoint map.

The source endpoint is heavily tanh-saturated on actual replay. Softsign keeps
the same admissible thrust interval and derivative through the true action,
with slower tail-gradient decay. No auxiliary reward or straight-through
gradient is introduced. Reuse the matched ordinary online continuation.
"""
import torch
from torch import nn
from torch.nn import functional as F

from train_shared_team_interaction import main
from unified.networks import (CITF, Endpoint, JointGenerator, ModelConfig, TeamCritic,
                              features, nearest_other, relative_geometry)


class TrainableEndpoint(Endpoint):
    def map_logits(self, logits):
        return F.softsign(logits)

    def forward(self, x, rule_source):
        n = rule_source.shape[1]
        pair = relative_geometry(x)[:, :n, :n]
        neighbor = self.neighbor_encoder(pair)
        diagonal = torch.eye(n, dtype=torch.bool, device=x.device)[None, :, :, None]
        if n > self.neighbors+1:
            blocked = ~nearest_other(pair, self.neighbors)[..., None]
            neighbor_mean = neighbor.masked_fill(blocked, 0.).sum(2)/self.neighbors
            neighbor_max = neighbor.masked_fill(blocked, -torch.inf).amax(2)
        else:
            neighbor_mean = neighbor.masked_fill(diagonal, 0.).sum(2)/(n-1)
            neighbor_max = neighbor.masked_fill(diagonal, -torch.inf).amax(2)
        relative_heading = x[:, -1:, 2]-x[:, :-1, 2]
        attacker_turn = x[:, -1:, 5:6].expand(-1, rule_source.shape[1], -1)/2
        local = torch.cat((features(x)[:, :-1], rule_source/1000,
                           relative_heading.cos()[..., None],
                           relative_heading.sin()[..., None], attacker_turn,
                           neighbor_mean, neighbor_max), -1)
        previous = x[:, :-1, 6:8]
        low = (previous-self.delta).clamp(min=-500.)
        high = (previous+self.delta).clamp(max=1000.)
        return (low+high)/2+(high-low)/2*self.map_logits(self.head(local))


class TrainableEndpointCITF(CITF):
    def __init__(self, cfg=None):
        nn.Module.__init__(self)
        self.cfg = cfg or ModelConfig()
        self.endpoint = TrainableEndpoint(self.cfg)
        self.generator = JointGenerator(self.cfg)
        self.q1, self.q2 = TeamCritic(self.cfg, 1), TeamCritic(self.cfg, 2)


if __name__ == '__main__':
    main(candidate_type=TrainableEndpointCITF, candidate_name='softsign_endpoint',
         candidate_definition='replace endpoint tanh(z) by z/(1+abs(z)) in both model and target; same parameters, bounds and true action derivatives; ordinary r25 critic aggregation retained')
