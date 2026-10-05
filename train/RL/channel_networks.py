"""Two-stage decentralized policy and permutation-invariant team critics."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from .control import candidate_features, fuse_thrust, to_thrust, to_action
from .observations import LOCAL_DIM, EDGE_DIM, MESSAGE_DIM, NODE_DIM, GLOBAL_DIM


def mlp(inputs, hidden, outputs, layer_norm=False):
    layers = [nn.Linear(inputs, hidden)]
    if layer_norm:
        layers.append(nn.LayerNorm(hidden))
    return nn.Sequential(*layers, nn.Tanh(), nn.Linear(hidden, outputs))


def masked_mean(values, mask, dim):
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim) / weights.sum(dim).clamp_min(1.)


def transformed_log_prob(dist, latent, transform):
    if transform == "tanh":
        jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
    else:
        jacobian = F.logsigmoid(latent) + F.logsigmoid(-latent)
    return (dist.log_prob(latent) - jacobian).sum(-1, keepdim=True)


class RelationAttention(nn.Module):
    def __init__(self, hidden, aggregation="attention"):
        super().__init__()
        self.aggregation = aggregation
        self.encoder = mlp(EDGE_DIM + MESSAGE_DIM, hidden, hidden)
        self.query = nn.Linear(hidden + 4, hidden)
        self.key = nn.Linear(hidden, hidden)

    def forward(self, query, relations, mask):
        encoded = self.encoder(relations)
        if self.aggregation == "mean":
            weights = mask.to(encoded.dtype)
        else:
            logits = (self.query(query).unsqueeze(-2) * self.key(encoded)).sum(-1) / math.sqrt(encoded.shape[-1])
            # Finite sentinel and explicit mask also handle an entirely empty neighborhood.
            weights = torch.softmax(logits.masked_fill(~mask, -1e9), -1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return (weights.unsqueeze(-1) * encoded).sum(-2), weights


class ChannelActor(nn.Module):
    def __init__(self, hidden=128, fusion="channels", aggregation="attention",
                 share_candidates=True, motion_features=True):
        super().__init__()
        if fusion not in ("scalar", "channels", "thrusters") or aggregation not in ("attention", "mean"):
            raise ValueError("Invalid fusion or relationship aggregation")
        self.config = dict(hidden=hidden, fusion=fusion, aggregation=aggregation,
                           share_candidates=share_candidates, motion_features=motion_features)
        self.fusion = fusion
        self.gate_dim = 1 if fusion == "scalar" else 2
        self.encoder = mlp(LOCAL_DIM, hidden, hidden)
        self.proposal_neighbors = mlp(EDGE_DIM, hidden, hidden)
        self.proposal_head = mlp(2 * hidden, hidden, 4)
        self.surge_attention = RelationAttention(hidden, aggregation)
        self.turn_attention = RelationAttention(hidden, aggregation)
        self.gate_head = mlp(3 * hidden + 5, hidden, 2 * self.gate_dim)
        for head in (self.proposal_head[-1], self.gate_head[-1]):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.bias[head.out_features // 2:] = -0.5

    @staticmethod
    def distribution(parameters):
        mean, log_std = parameters.chunk(2, -1)
        return Normal(mean, log_std.clamp(-5., 1.).exp())

    def proposal_distribution(self, frame):
        h = self.encoder(frame["local"])
        neighbors = masked_mean(self.proposal_neighbors(frame["edges"]), frame["neighbors"], -2)
        return self.distribution(self.proposal_head(torch.cat((h, neighbors), -1)))

    def propose(self, frame, deterministic=False):
        dist = self.proposal_distribution(frame)
        latent = dist.mean if deterministic else dist.sample()
        return latent, torch.tanh(latent), transformed_log_prob(dist, latent, "tanh")

    def exchange(self, frame, candidate):
        """One synchronous round. Messages contain proposals, never final gates."""
        features = candidate_features(frame["boids"], candidate)
        batch, n, _ = features.shape
        received = features.unsqueeze(1).expand(batch, n, n, 4)
        if not self.config["share_candidates"]:
            received = torch.zeros_like(received)
        age = torch.zeros(batch, n, n, 1, device=features.device, dtype=features.dtype)
        return torch.cat((received, age), -1).detach()

    def gate_distribution(self, frame, candidate, messages, detach_features=False):
        h = self.encoder(frame["local"])
        if detach_features:
            h = h.detach()
        own = candidate_features(frame["boids"].detach(), candidate.detach())
        query = torch.cat((h, own), -1)
        relations = torch.cat((frame["edges"], messages.detach()), -1)
        z_c, attention_c = self.surge_attention(query, relations, frame["neighbors"])
        z_d, attention_d = self.turn_attention(query, relations, frame["neighbors"])
        count = frame["neighbors"].sum(-1, keepdim=True).to(h.dtype) / 8.
        parameters = self.gate_head(torch.cat((h, own, z_c, z_d, count), -1))
        return self.distribution(parameters), (attention_c, attention_d)

    def act(self, frame, deterministic=False):
        z_l, candidate, log_l = self.propose(frame, deterministic)
        messages = self.exchange(frame, candidate)
        dist, attention = self.gate_distribution(frame, candidate, messages)
        z_g = dist.mean if deterministic else dist.sample()
        gates = torch.sigmoid(z_g)
        return dict(candidate_z=z_l, candidate=candidate, gate_z=z_g, gates=gates,
                    logp_candidate=log_l, logp_gate=transformed_log_prob(dist, z_g, "sigmoid"),
                    messages=messages, executed=fuse_thrust(frame["boids"], to_thrust(candidate), gates, self.fusion),
                    attention_c=attention[0], attention_d=attention[1])

    def evaluate_actions(self, frame, data, entropy=True):
        # Both stages condition on the same historical random decisions/messages.
        proposal_dist = self.proposal_distribution(frame)
        gate_dist, _ = self.gate_distribution(frame, data["candidate"], data["messages"])
        log_l = transformed_log_prob(proposal_dist, data["candidate_z"], "tanh")
        log_g = transformed_log_prob(gate_dist, data["gate_z"], "sigmoid")
        ent = torch.zeros_like(log_l)
        if entropy:
            ent = -transformed_log_prob(proposal_dist, proposal_dist.rsample(), "tanh")
            ent = ent - transformed_log_prob(gate_dist, gate_dist.rsample(), "sigmoid")
        return log_l + log_g, ent

    def representative(self, frame, candidate, messages, adapter_only=False):
        dist, _ = self.gate_distribution(frame, candidate, messages, adapter_only)
        gates = torch.sigmoid(dist.mean)
        return gates, fuse_thrust(frame["boids"].detach(), to_thrust(candidate.detach()), gates, self.fusion)


class TeamCritic(nn.Module):
    def __init__(self, hidden=128, action_conditioned=False, normalize_value=False, layer_norm=False):
        super().__init__()
        self.action_conditioned = action_conditioned
        self.node = mlp(NODE_DIM + (2 if action_conditioned else 0), hidden, hidden, layer_norm)
        self.message = mlp(2 * hidden + EDGE_DIM, hidden, hidden, layer_norm)
        self.update = mlp(2 * hidden, hidden, hidden, layer_norm)
        self.global_encoder = mlp(GLOBAL_DIM, hidden, hidden, layer_norm)
        self.readout = mlp(2 * hidden + 1, hidden, 1, layer_norm)
        self.normalize_value = normalize_value
        if normalize_value:
            self.register_buffer("output_mean", torch.zeros(()))
            self.register_buffer("output_scale", torch.ones(()))
            self.register_buffer("second_moment", torch.ones(()))
            self.register_buffer("normalization_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def rebase_output(self, mean, scale):
        """Change normalized coordinates while preserving every raw value."""
        head = self.readout[-1]
        head.weight.mul_(self.output_scale / scale)
        head.bias.copy_((self.output_scale * head.bias + self.output_mean - mean) / scale)
        self.output_mean.copy_(mean)
        self.output_scale.copy_(scale)

    @torch.no_grad()
    def update_normalization(self, targets, decay=.99):
        if not self.normalize_value:
            return
        weight = decay if self.normalization_updates.item() else 0.
        mean = weight * self.output_mean + (1 - weight) * targets.mean()
        second = weight * self.second_moment + (1 - weight) * targets.square().mean()
        scale = (second - mean.square()).clamp_min(1e-4).sqrt()
        self.rebase_output(mean, scale)
        self.second_moment.copy_(second)
        self.normalization_updates.add_(1)

    def normalize_targets(self, targets):
        return (targets - self.output_mean) / self.output_scale if self.normalize_value else targets

    @torch.no_grad()
    def copy_normalization_from(self, other):
        if self.normalize_value:
            self.rebase_output(other.output_mean, other.output_scale)
            self.second_moment.copy_(other.second_moment)
            self.normalization_updates.copy_(other.normalization_updates)

    def forward(self, frame, executed=None, normalized=False):
        nodes = frame["nodes"]
        if self.action_conditioned:
            if executed is None:
                raise ValueError("Team Q requires actual projected actuator commands")
            nodes = torch.cat((nodes, to_action(executed)), -1)
        h = self.node(nodes)
        n = h.shape[-2]
        hi = h.unsqueeze(-2).expand(-1, -1, n, -1)
        hj = h.unsqueeze(-3).expand(-1, n, -1, -1)
        messages = self.message(torch.cat((hi, hj, frame["edges"]), -1))
        related = masked_mean(messages, frame["neighbors"], -2)
        h = h + self.update(torch.cat((h, related), -1))
        pooled = masked_mean(h, frame["mask"], -2)
        count = frame["mask"].sum(-1, keepdim=True).to(h.dtype) / 8.
        prediction = self.readout(torch.cat((pooled, self.global_encoder(frame["global_state"]), count), -1))
        if self.normalize_value and not normalized:
            prediction = prediction * self.output_scale + self.output_mean
        return prediction
