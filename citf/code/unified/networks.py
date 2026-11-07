"""Joint fusion generator and candidate-action Transformer critics."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Beta, Categorical

from .environment import boids, execute, fusion


@dataclass
class ModelConfig:
    width: int = 128
    heads: int = 4
    layers: int = 2
    samples: int = 8
    search_steps: int = 2
    search_rate: float = .1
    thrust_delta: float = 300.
    action_interaction: bool = True
    feedback: float = 1.
    neighbors: int = 2

    def __post_init__(self):
        if self.samples < 2:
            raise ValueError('The leave-one-out generator objective needs at least two independent samples.')
        if self.heads < 1 or self.width < 1 or self.width % self.heads or self.layers < 1:
            raise ValueError('Positive model dimensions and width divisible by heads are required.')
        if self.search_steps < 0 or self.search_rate < 0 or self.thrust_delta <= 0 or self.feedback < 0:
            raise ValueError('Invalid search, actuator or feedback configuration.')
        if self.neighbors < 1:
            raise ValueError('The spatial neighborhood needs at least one teammate.')


@contextmanager
def frozen(module):
    states = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(module.parameters(), states):
            p.requires_grad_(flag)


def features(x):
    b, entities = x.shape[:2]
    # Thrust is expressed in each vessel's body frame. Give the learned
    # controller the same frame, without an artificial generation-slot role.
    c, s = x[..., 2].cos(), x[..., 2].sin()
    def body(vector):
        return torch.stack((c*vector[..., 0]+s*vector[..., 1],
                            -s*vector[..., 0]+c*vector[..., 1]), -1)
    position, velocity = x[..., :2], x[..., 3:5]
    relative = position[:, -1:]-position
    role = x.new_ones(b, entities, 1)
    role[:, -1] = 0
    return torch.cat((body(-position)/60, body(relative)/60, body(velocity)/5,
                      body(velocity[:, -1:]-velocity)/5, x[..., 5:6]/2,
                      x[..., 6:8]*role/1000, x[..., 8:9], role,
                      position.norm(dim=-1, keepdim=True)/60,
                      relative.norm(dim=-1, keepdim=True)/60), -1)


def relative_geometry(x):
    """Current pair geometry in the receiving vessel's body frame."""
    position, velocity, heading = x[..., :2], x[..., 3:5], x[..., 2]
    dp = position[:, None]-position[:, :, None]
    dv = velocity[:, None]-velocity[:, :, None]
    c, s = heading.cos()[:, :, None], heading.sin()[:, :, None]
    distance = dp.norm(dim=-1)
    angle = heading[:, None]-heading[:, :, None]
    return torch.stack(((c*dp[..., 0]+s*dp[..., 1])/15,
                        (-s*dp[..., 0]+c*dp[..., 1])/15,
                        (c*dv[..., 0]+s*dv[..., 1])/5,
                        (-s*dv[..., 0]+c*dv[..., 1])/5,
                        distance/15, (dp*dv).sum(-1)/distance.clamp_min(1e-6)/5,
                        angle.cos(), angle.sin()), -1)


def controlled_pair_effect(heading, command):
    """Input-induced acceleration components, not a future-state rollout.

    The attacker has no defender-controlled input. Its token therefore has
    zero command; all of its measured motion remains in relative_geometry.
    """
    surge = command.sum(-1)/380
    yaw = 1.25*(command[..., 0]-command[..., 1])/1430
    c, s = heading.cos(), heading.sin()
    acceleration = surge[..., None]*torch.stack((c, s), -1)
    difference = acceleration[:, None]-acceleration[:, :, None]
    c, s = c[:, :, None], s[:, :, None]
    rows = yaw[:, :, None].expand(-1, -1, heading.shape[1])
    columns = yaw[:, None].expand(-1, heading.shape[1], -1)
    return torch.stack(((c*difference[..., 0]+s*difference[..., 1])/5,
                        (-s*difference[..., 0]+c*difference[..., 1])/5,
                        rows/2, columns/2), -1)


def nearest_other(pair, count):
    """Nearest physical teammates; independent of the candidate commands."""
    batch, n = pair.shape[:2]
    diagonal = torch.eye(n, dtype=torch.bool, device=pair.device)[None]
    distance = pair[..., 4].masked_fill(diagonal, torch.inf)
    selected = distance.topk(min(count, n-1), dim=-1, largest=False).indices
    return torch.zeros(batch, n, n, dtype=torch.bool, device=pair.device).scatter_(-1, selected, True)


def entity_neighborhood(pair, count):
    """Keep the same local context size as the three-defender training task."""
    batch, length = pair.shape[:2]
    n = length-1
    allowed = torch.zeros(batch, length, length, dtype=torch.bool, device=pair.device)
    allowed[:, :n, :n] = nearest_other(pair[:, :n, :n], count) | torch.eye(
        n, dtype=torch.bool, device=pair.device)[None]
    allowed[:, :n, n] = True
    selected = pair[:, n, :n, 4].topk(count+1, dim=-1, largest=False).indices
    allowed[:, n, :n].scatter_(-1, selected, True)
    allowed[:, n, n] = True
    return allowed


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, causal=False, edge_dim=0, neighbors=2):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.neighbors = neighbors
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.projection = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.SiLU(), nn.Linear(2*width, width))
        self.edge_bias = (nn.Sequential(nn.Linear(edge_dim, width//2), nn.SiLU(), nn.Linear(width//2, heads))
                          if edge_dim else None)
        self.edge_value = nn.Linear(edge_dim, width, bias=False) if edge_dim else None

    def forward(self, x, edge=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        score = q @ k.transpose(-1, -2)/math.sqrt(width//self.heads)
        if self.edge_bias is not None:
            score = score+self.edge_bias(edge).permute(0, 3, 1, 2)
        if self.causal:
            mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            score = score.masked_fill(mask, -torch.inf)
        if edge is not None and length > self.neighbors+2:
            score = score.masked_fill(~entity_neighborhood(edge, self.neighbors)[:, None], -torch.inf)
        weights = score.softmax(-1)
        context = weights @ v
        if self.edge_value is not None:
            edge_values = self.edge_value(edge).reshape(batch, length, length, self.heads, width//self.heads)
            context = context+torch.einsum('bhij,bijhd->bhid', weights, edge_values)
        context = context.transpose(1, 2).reshape(batch, length, width)
        x = x+self.projection(context)
        return x+self.ff(self.norm2(x))


class LocalBlock(nn.Module):
    """Per-vessel value features, with no cross-current-action mixing."""
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.SiLU(), nn.Linear(4*width, width))

    def forward(self, x):
        return x+self.ff(self.norm(x))


class PrefixAttentionBlock(nn.Module):
    """Query vessel i reads the actual selected actions of vessels j < i.

    State and action remain attached to the same physical vessel. A learned
    empty-prefix token keeps the first conditional distribution well defined.
    """
    def __init__(self, width, heads, neighbors=2):
        super().__init__()
        self.heads = heads
        self.neighbors = neighbors
        self.query_norm, self.memory_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2*width)
        self.empty_prefix = nn.Parameter(torch.zeros(1, 1, width))
        self.edge_bias = nn.Sequential(nn.Linear(8, width//2), nn.SiLU(), nn.Linear(width//2, heads))
        self.edge_value = nn.Linear(8, width, bias=False)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.SiLU(), nn.Linear(2*width, width))

    def forward(self, query, memory, pair):
        batch, length, width = query.shape
        memory = torch.cat((memory, self.empty_prefix.expand(batch, -1, -1)), 1)
        q = self.query(self.query_norm(query)).reshape(batch, length, self.heads, width//self.heads).transpose(1, 2)
        k, v = self.key_value(self.memory_norm(memory)).reshape(
            batch, length+1, 2, self.heads, width//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        pair = torch.cat((pair, pair.new_zeros(batch, length, 1, 8)), 2)
        score = q @ k.transpose(-1, -2)/math.sqrt(width//self.heads)
        score = score+self.edge_bias(pair).permute(0, 3, 1, 2)
        # No current/future selected action enters a conditional likelihood.
        future = torch.ones(length, length, dtype=torch.bool, device=query.device).triu(0)
        if length > self.neighbors+1:
            distance = pair[:, :, :length, 4].masked_fill(future, torch.inf)
            selected = distance.topk(self.neighbors, dim=-1, largest=False).indices
            allowed = torch.zeros_like(distance, dtype=torch.bool).scatter_(-1, selected, True) & ~future
            mask = torch.cat((~allowed, allowed.new_zeros(batch, length, 1)), 2)[:, None]
        else:
            mask = torch.cat((future, future.new_zeros(length, 1)), 1)
        weights = score.masked_fill(mask, -torch.inf).softmax(-1)
        edge_values = self.edge_value(pair).reshape(batch, length, length+1, self.heads, width//self.heads)
        context = weights @ v+torch.einsum('bhij,bijhd->bhid', weights, edge_values)
        query = query+self.projection(context.transpose(1, 2).reshape(batch, length, width))
        return query+self.ff(self.norm2(query))


class EntityEncoder(nn.Module):
    def __init__(self, cfg, extra=0):
        super().__init__()
        self.embed = nn.Linear(16+extra, cfg.width)
        self.layers = nn.ModuleList([AttentionBlock(cfg.width, cfg.heads, edge_dim=8,
                                                    neighbors=cfg.neighbors) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, x, extra=None, pair=None):
        f = features(x)
        # The continuation policy uses the episode-fixed autoregressive order.
        # Retain that control context in Q as well as the generator, without
        # assigning a permanent identity to any physical vessel.
        n = x.shape[1]-1
        position = x.new_full((len(x), n+1, 1), -1.)
        position[:, :n, 0] = torch.linspace(0., 1., n, device=x.device, dtype=x.dtype)
        f = torch.cat((f, position), -1)
        pair = relative_geometry(x) if pair is None else pair
        if extra is not None:
            extra = torch.cat((extra, torch.zeros_like(extra[:, :1])), 1)
            f = torch.cat((f, extra), -1)
        z = self.embed(f)
        for layer in self.layers:
            z = layer(z, pair)
        return self.norm(z)


class Endpoint(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = 2*cfg.width
        self.delta = cfg.thrust_delta
        self.neighbors = cfg.neighbors
        neighbor_width = cfg.width//2
        self.neighbor_encoder = nn.Sequential(nn.Linear(8, neighbor_width), nn.SiLU(),
                                              nn.Linear(neighbor_width, neighbor_width), nn.SiLU())
        # A shared per-vessel control source. Team-level candidate dependence
        # belongs to the Adapter rather than a second centralized thrust policy.
        self.head = nn.Sequential(nn.Linear(20+2*neighbor_width, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        nn.init.normal_(self.head[-1].weight, std=.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, rule_source):
        n = rule_source.shape[1]
        pair = relative_geometry(x)[:, :n, :n]
        neighbor = self.neighbor_encoder(pair)
        diagonal = torch.eye(n, dtype=torch.bool, device=x.device)[None, :, :, None]
        # Keep individual relative-motion information that a two-thrust Boids
        # summary cannot recover. Both reductions are independent of vessel IDs.
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
        return (low+high)/2+(high-low)/2*self.head(local).tanh()


class JointGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = EntityEncoder(cfg, extra=4)
        self.prefix = nn.Linear(4, cfg.width)
        self.decoder = nn.ModuleList([PrefixAttentionBlock(cfg.width, cfg.heads, cfg.neighbors)
                                      for _ in range(cfg.layers)])
        self.head = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, 5))
        nn.init.normal_(self.head[-1].weight, std=.005)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            self.head[-1].bias[3:] = math.log(math.expm1(1.9))

    def encode(self, x, a, b, pair=None):
        return self.encoder(x, torch.cat((a, b), -1)/1000, pair=pair)[:, :-1]

    def parameters_at(self, z, prefix, pair):
        h = z
        memory = z+self.prefix(prefix)
        for layer in self.decoder:
            h = layer(h, memory, pair)
        value = self.head(h)
        return value[..., :3], F.softplus(value[..., 3:])+.1

    @torch.no_grad()
    def sample(self, x, a, b, count):
        batch, n = a.shape[:2]
        pair = relative_geometry(x)
        z = self.encode(x, a, b, pair)[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, -1)
        pair = pair[:, None, :n, :n].expand(-1, count, -1, -1, -1).reshape(batch*count, n, n, 8)
        xx = x[:, None].expand(-1, count, -1, -1).reshape(batch*count, n+1, -1)
        aa = a[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, 2)
        bb = b[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, 2)
        prefix, g = z.new_zeros(batch*count, n, 4), z.new_zeros(batch*count, n)
        for j in range(n):
            logits, shape = self.parameters_at(z, prefix, pair)
            kind = Categorical(logits=logits[:, j]).sample()
            # Keep the sampled interior value: a 1e-6 clip introduces extra
            # boundary masses that are absent from the fitted Beta likelihood.
            limits = torch.finfo(shape.dtype)
            interior = Beta(shape[:, j, 0], shape[:, j, 1]).sample().clamp(limits.tiny, 1-limits.eps/2)
            coefficient = torch.where(kind == 0, 0., torch.where(kind == 1, 1., interior))
            g[:, j] = coefficient
            u = fusion(xx, aa, bb, g, self.cfg.thrust_delta)[:, j]
            prefix[:, j] = torch.cat((coefficient[:, None], u/1000, torch.ones_like(coefficient[:, None])), -1)
        return g.reshape(batch, count, n)

    def log_prob(self, x, a, b, g):
        a, b, g = a.detach(), b.detach(), g.detach()
        batch, count, n = g.shape
        pair = relative_geometry(x)
        z = self.encode(x, a, b, pair)[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, -1)
        pair = pair[:, None, :n, :n].expand(-1, count, -1, -1, -1).reshape(batch*count, n, n, 8)
        u = fusion(x, a, b, g, self.cfg.thrust_delta)
        tokens = torch.cat((g[..., None], u/1000, torch.ones_like(g[..., None])), -1)
        prefix = tokens.reshape(batch*count, n, 4)
        logits, shape = self.parameters_at(z, prefix, pair)
        logs = logits.log_softmax(-1).reshape(batch, count, n, 3)
        shape = shape.reshape(batch, count, n, 2)
        # Do not perturb interior teacher labels. Endpoints use probability
        # masses; 0.5 merely avoids evaluating an unused infinite Beta density.
        inside = (g > 0) & (g < 1)
        interior = Beta(shape[..., 0], shape[..., 1]).log_prob(torch.where(inside, g, .5))
        per_boat = torch.where(g == 0, logs[..., 0], torch.where(g == 1, logs[..., 1], logs[..., 2]+interior))
        return per_boat.sum(-1)


class TeamCritic(nn.Module):
    def __init__(self, cfg, critic_index=1):
        super().__init__()
        self.action_interaction = cfg.action_interaction
        self.thrust_delta = cfg.thrust_delta
        self.neighbors = cfg.neighbors
        self.encoder = EntityEncoder(cfg)
        self.action = nn.Linear(2, cfg.width)
        self.layers = nn.ModuleList([LocalBlock(cfg.width) for _ in range(cfg.layers)])
        self.value = nn.Sequential(nn.Linear(cfg.width, cfg.width), nn.SiLU(), nn.Linear(cfg.width, 1))
        self.readout = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.width), nn.SiLU(), nn.Linear(cfg.width, 1))
        if self.action_interaction:
            # Preserve the same initialization stream for the common local
            # paths in the full method and its no-interaction ablation.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(torch.initial_seed()+104729*critic_index)
                self.interaction_layers = nn.ModuleList([
                    AttentionBlock(cfg.width, cfg.heads, edge_dim=12, neighbors=cfg.neighbors)
                    for _ in range(cfg.layers)])
                pair_width = cfg.width//2
                self.interaction_pair = nn.Sequential(nn.Linear(12, pair_width), nn.SiLU(),
                                                      nn.Linear(pair_width, pair_width), nn.SiLU())
                self.interaction_readout = nn.Sequential(nn.LayerNorm(cfg.width+pair_width),
                                                         nn.Linear(cfg.width+pair_width, cfg.width//2),
                                                         nn.SiLU(), nn.Linear(cfg.width//2, 2))
                nn.init.zeros_(self.interaction_readout[-1].weight)
                nn.init.zeros_(self.interaction_readout[-1].bias)

    def encode(self, x):
        pair = relative_geometry(x)
        rule = execute(boids(x), x[:, :-1, 6:8], self.thrust_delta)
        return self.encoder(x, pair=pair), pair, x[..., 2], rule

    def from_encoded(self, encoded, u, individual=False):
        z, pair, heading, rule = encoded
        multiple = u.ndim == 4
        if multiple:
            batch, count, n = u.shape[:3]
            z = z[:, None].expand(-1, count, -1, -1).reshape(batch*count, n+1, -1)
            pair = pair[:, None].expand(-1, count, -1, -1, -1).reshape(batch*count, n+1, n+1, 8)
            heading = heading[:, None].expand(-1, count, -1).reshape(batch*count, n+1)
            rule = rule[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, 2)
            u = u.reshape(batch*count, n, 2)
        n = u.shape[1]
        context = self.value(z[:, :n]).squeeze(-1)
        command = torch.cat((u, torch.zeros_like(u[:, :1])), 1)
        action_state = z+self.action(command/1000)
        h = action_state
        for layer in self.layers:
            h = layer(h)
        q = context+self.readout(h[:, :n]).squeeze(-1)
        if self.action_interaction:
            edge = torch.cat((pair, controlled_pair_effect(heading, command)), -1)
            h = action_state
            for layer in self.interaction_layers:
                h = layer(h, edge)
            # Each directed pair describes the receiving vessel's return
            # response to the teammate's actual departure from its executable
            # rule command. Transformer context makes the response depend on
            # the complete candidate, including the receiving vessel's action.
            pair_response = self.interaction_pair(edge[:, :n, :n])
            receiver = h[:, :n, None].expand(-1, -1, n, -1)
            response = self.interaction_readout(torch.cat((receiver, pair_response), -1))
            departure = (u-rule)/1000.
            pair_value = (response*departure[:, None]).sum(-1)
            diagonal = torch.eye(n, dtype=torch.bool, device=u.device)[None]
            allowed = (nearest_other(edge[:, :n, :n], self.neighbors)
                       if n > self.neighbors+1 else ~diagonal)
            interaction = pair_value.masked_fill(~allowed, 0.).sum(-1)/min(self.neighbors, n-1)
            # This branch is zero when the relevant teammates keep their
            # rule commands. It remains active when this receiving vessel
            # itself keeps Boids while another vessel changes its command.
            # Only actual-return TD supervision is used; there are no extra
            # intervention rollouts, return-difference labels or value heads.
            q = q+interaction
        if individual:
            return q.reshape(batch, count, n) if multiple else q
        q = q.mean(-1)
        return q.reshape(batch, count) if multiple else q

    def forward(self, x, u, individual=False):
        return self.from_encoded(self.encode(x), u, individual=individual)


class CITF(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.endpoint = Endpoint(self.cfg)
        self.generator = JointGenerator(self.cfg)
        self.q1, self.q2 = TeamCritic(self.cfg, 1), TeamCritic(self.cfg, 2)

    def encode_value(self, x):
        return self.q1.encode(x), self.q2.encode(x)

    def value_from_encoded(self, encoded, u):
        first = self.q1.from_encoded(encoded[0], u, individual=True)
        second = self.q2.from_encoded(encoded[1], u, individual=True)
        # Clip each vessel's return before the team mean. With the action
        # interaction removed this remains an additive function of own actions.
        return torch.minimum(first, second).mean(-1)

    def select(self, x, endpoint_noise_std=0., exploration_generator=None):
        """Search current coefficients only; no differentiation through the search."""
        with torch.no_grad():
            rule_source = boids(x)
            b = execute(rule_source, x[:, :-1, 6:8], self.cfg.thrust_delta)
            a = self.endpoint(x, rule_source)
            # Off-policy exploration covers BOTH endpoint thrust coordinates.
            # A single noisy endpoint is shared by all candidates this cycle.
            # Evaluation, Bellman targets and policy updates use noise_std=0.
            if endpoint_noise_std > 0:
                noise = torch.randn(a.shape, device=a.device, dtype=a.dtype,
                                    generator=exploration_generator)*endpoint_noise_std
                a = execute(a+noise, x[:, :-1, 6:8], self.cfg.thrust_delta)
            initial = self.generator.sample(x, a, b, self.cfg.samples)
            anchors = torch.stack((torch.zeros_like(initial[:, 0]), torch.ones_like(initial[:, 0])), 1)
            g = torch.cat((initial, anchors), 1)
            encoded = self.encode_value(x)
            best_g = g.clone()
            best_q = self.value_from_encoded(encoded, fusion(x, a, b, g, self.cfg.thrust_delta))
            initial_pool_scores = best_q.clone()
            initial_scores = best_q[:, :self.cfg.samples].clone()
            chain_index = torch.arange(g.shape[1], device=x.device)[None].expand(len(x), -1)
            best_order = chain_index.clone()
        with frozen(self.q1), frozen(self.q2), torch.enable_grad():
            for step in range(self.cfg.search_steps):
                g = g.detach().requires_grad_(True)
                score = self.value_from_encoded(encoded, fusion(x, a, b, g, self.cfg.thrust_delta))
                grad = torch.autograd.grad(score.sum(), g)[0]
                g = (g+self.cfg.search_rate*grad/(grad.abs().amax(-1, keepdim=True)+1e-8)).clamp(0, 1).detach()
                with torch.no_grad():
                    new_q = self.value_from_encoded(encoded, fusion(x, a, b, g, self.cfg.thrust_delta))
                    improve = new_q > best_q
                    best_g = torch.where(improve[..., None], g, best_g)
                    best_order = torch.where(improve, (step+1)*g.shape[1]+chain_index, best_order)
                    best_q = torch.maximum(best_q, new_q)
        with torch.no_grad():
            if not bool(torch.isfinite(best_q).all()):
                raise FloatingPointError('Non-finite candidate value or search gradient.')
            # Tie order is generation time across the whole pool, not chain ID.
            tied = best_q == best_q.amax(1, keepdim=True)
            index = best_order.masked_fill(~tied, torch.iinfo(best_order.dtype).max).argmin(1)
            selected = best_g[torch.arange(len(x), device=x.device), index]
            action = fusion(x, a, b, selected, self.cfg.thrust_delta)
        return dict(action=action, g=selected, candidates=best_g, scores=best_q,
                    initial=initial, initial_scores=initial_scores,
                    initial_pool_scores=initial_pool_scores, a=a, b=b, rule_source=rule_source)
