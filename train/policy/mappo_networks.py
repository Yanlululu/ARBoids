"""Shared two-stage actor and a separate centralized reward/cost critic."""
import math
import torch
from torch import nn
from torch.distributions import Normal


def initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, math.sqrt(2.))
        nn.init.zeros_(module.bias)


class PredictiveActor(nn.Module):
    def __init__(self, hidden_dim=512, relation_dim=64, coordination_dim=16, edge_dim=18):
        super().__init__()
        h, r, c = hidden_dim, relation_dim, coordination_dim
        if h < 16 or h % 4 or min(r, c) < 1:
            raise ValueError('hidden_dim must be a multiple of four >=16; head dimensions must be positive.')
        self.register_buffer('local_scale', torch.tensor(
            [60., math.pi, 60., math.pi, 5., math.pi,
             5., math.pi, 5., math.pi, 60., math.pi, 1., 1.]))
        self.register_buffer('neighbor_scale', torch.tensor([60., math.pi]))
        self.f1 = nn.Linear(6, h // 2)
        self.f2 = nn.Linear(8, h // 2)
        self.teammate = nn.Linear(2, h // 4)
        self.encoder = nn.Sequential(nn.Linear(h + h // 4, h), nn.Tanh(),
                                     nn.Linear(h, h), nn.Tanh())
        self.proposal_mean = nn.Linear(h, 2)
        self.proposal_log_std = nn.Linear(h, 2)
        self.action_encoder = nn.Sequential(nn.Linear(4, h // 4), nn.Tanh())
        self.relation_encoder = nn.Sequential(nn.Linear(edge_dim, r), nn.Tanh(),
                                              nn.Linear(r, r), nn.Tanh())
        self.relation_query = nn.Linear(h, r)
        self.priority_head = nn.Sequential(nn.Linear(edge_dim, r), nn.Tanh(), nn.Linear(r, 1))
        self.coordination_encoder = nn.Sequential(nn.Linear(r + 1, c), nn.Tanh())
        self.adapter = nn.Sequential(nn.Linear(h + h // 4 + r + c, h // 2), nn.Tanh())
        self.gate_mean = nn.Linear(h // 2, 1)
        self.gate_log_std = nn.Linear(h // 2, 1)
        self.apply(initialize)
        for output in (self.proposal_mean, self.proposal_log_std, self.gate_mean, self.gate_log_std):
            nn.init.orthogonal_(output.weight, .01)
        nn.init.constant_(self.proposal_log_std.bias, -.5)
        nn.init.constant_(self.gate_log_std.bias, -.5)

    def encode(self, obs):
        local = obs[..., :14] / self.local_scale
        neighbors = obs[..., 14:].reshape(*obs.shape[:-1], -1, 2) / self.neighbor_scale
        if neighbors.shape[-2]:
            teammate = torch.tanh(self.teammate(neighbors)).mean(dim=-2)
        else:
            teammate = obs.new_zeros(*obs.shape[:-1], self.teammate.out_features)
        return self.encoder(torch.cat((torch.tanh(self.f1(local[..., :6])),
                                       torch.tanh(self.f2(local[..., 6:])), teammate), dim=-1))

    def proposal_distribution(self, obs):
        h = self.encode(obs)
        return Normal(self.proposal_mean(h), self.proposal_log_std(h).clamp(-5., 1.).exp())

    def relation_features(self, h, edges, mask):
        embedded = self.relation_encoder(edges)
        score = (self.relation_query(h).unsqueeze(-2) * embedded).sum(-1) / math.sqrt(embedded.shape[-1])
        # A masked softmax that also works for a boat with no valid neighbors.
        weights = torch.softmax(score.masked_fill(~mask, -1e9), dim=-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        z_pred = (weights.unsqueeze(-1) * embedded).sum(-2)
        priority = self.priority_head(edges).squeeze(-1)
        eta = torch.sigmoid(priority - priority.transpose(-1, -2))
        coordinate = self.coordination_encoder(torch.cat((embedded, (eta - .5).unsqueeze(-1)), dim=-1))
        z_coord = (weights.unsqueeze(-1) * coordinate).sum(-2)
        return z_pred, z_coord, eta

    def gate_distribution(self, obs, proposals, boids, edges, mask):
        h = self.encode(obs)
        z_pred, z_coord, eta = self.relation_features(h, edges, mask)
        actions = self.action_encoder(torch.cat((proposals, boids), dim=-1))
        features = self.adapter(torch.cat((h, actions, z_pred, z_coord), dim=-1))
        distribution = Normal(self.gate_mean(features), self.gate_log_std(features).clamp(-5., 1.).exp())
        return distribution, eta

    def sample_proposals(self, obs, deterministic=False):
        distribution = self.proposal_distribution(obs)
        raw = distribution.mean if deterministic else distribution.sample()
        return torch.tanh(raw), raw, distribution.log_prob(raw).sum(-1)

    def sample_gates(self, obs, proposals, boids, edges, mask, deterministic=False):
        distribution, eta = self.gate_distribution(obs, proposals, boids, edges, mask)
        raw = distribution.mean if deterministic else distribution.sample()
        return torch.sigmoid(raw), raw, distribution.log_prob(raw).sum(-1), eta

    def evaluate_actions(self, obs, proposals, boids, edges, mask, proposal_raw, gate_raw):
        # Saved samples/messages are constants; current encodings stay differentiable.
        proposal = self.proposal_distribution(obs)
        gate, _ = self.gate_distribution(obs, proposals, boids, edges, mask)
        return (proposal.log_prob(proposal_raw).sum(-1), gate.log_prob(gate_raw).sum(-1),
                proposal.entropy().sum(-1) + gate.entropy().sum(-1))


class CentralizedCritic(nn.Module):
    def __init__(self, global_dim, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)
        self.apply(initialize)

    def forward(self, state):
        h = self.encoder(state)
        return self.reward_head(h).squeeze(-1), self.cost_head(h).squeeze(-1)
