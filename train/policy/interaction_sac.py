"""Interaction-conditioned residual SAC and the matched legacy continuation.

Policy inputs are public measurements. The centralized dynamics state is used
only by the critics. One replay row is one complete team decision.
"""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from policy.networks import ActorAdap
from policy.SAC import SAC
from utils.config import _dict_to_namespace


PACKET_KEYS = ('obs', 'motion', 'central')


def tensor_packet(packet, device='cpu'):
    return {k: torch.as_tensor(packet[k], dtype=torch.float32, device=device)
            for k in PACKET_KEYS}


def transformed_normal(mean, log_std, noise=None, deterministic=False, gate=False):
    std = log_std.exp()
    raw = mean if deterministic else mean + std * (torch.randn_like(mean) if noise is None else noise)
    logp = Normal(mean, std).log_prob(raw)
    logp = logp - 2 * (math.log(2.) - raw - F.softplus(-2 * raw))
    action = raw.tanh()
    if gate:
        action = (action + 1.) * .5
        logp = logp + math.log(2.)
    return action, logp.sum(-1, keepdim=True)


def masked_mean(x, mask, dim):
    weight = mask.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim) / weight.sum(dim).clamp_min(1.)


class InteractionActor(nn.Module):
    def __init__(self, hidden=512, relation=128):
        super().__init__()
        self.hidden, self.relation = hidden, relation
        self.base = ActorAdap(6, 8, 3, hidden)
        self.relations = nn.Sequential(nn.Linear(15, relation), nn.LeakyReLU(),
                                       nn.Linear(relation, relation), nn.LeakyReLU())
        self.relation_gate = nn.Linear(relation, 1, bias=False)
        self.gate_log_std = nn.Linear(hidden + relation, 1)
        nn.init.zeros_(self.relation_gate.weight)
        nn.init.zeros_(self.gate_log_std.weight)
        nn.init.constant_(self.gate_log_std.bias, math.log(.1))

    def initialize_source(self, weights):
        self.base.load_state_dict(weights, strict=True)

    def freeze_proposals(self, freeze=True):
        for name, parameter in self.base.named_parameters():
            if name.split('.')[0] in ('f1', 'f2', 't1', 'l1', 'l2', 'mean_layer', 'log_std_layer'):
                parameter.requires_grad_(not freeze)

    def encode(self, obs, mask):
        b, n, d = obs.shape
        if d != 14 + 2 * (n - 1) or n < 2:
            raise ValueError('Expected the released observation [B,N,14+2*(N-1)], N>=2.')
        net, act = self.base, self.base.activation
        neighbor_mask = mask[:, None, :].expand(b, n, n)
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=obs.device)
        neighbor_mask = neighbor_mask[:, off_diagonal].reshape(b, n, n - 1)
        neighbors = act(net.t1(obs[..., 14:].reshape(b, n, n - 1, 2)))
        h = torch.cat((act(net.f1(obs[..., :6])), act(net.f2(obs[..., 6:14])),
                       masked_mean(neighbors, neighbor_mask, -2)), -1)
        return act(net.l2(act(net.l1(h))))

    def relation_features(self, motion, proposals, boids):
        n = motion.shape[-2]
        delta = motion.unsqueeze(-3) - motion.unsqueeze(-2)
        heading = motion[..., 2].unsqueeze(-1)
        c, s = heading.cos(), heading.sin()
        p = torch.stack((c * delta[..., 0] + s * delta[..., 1],
                         -s * delta[..., 0] + c * delta[..., 1]), -1) / 60.
        v = torch.stack((c * delta[..., 3] + s * delta[..., 4],
                         -s * delta[..., 3] + c * delta[..., 4]), -1) / 5.
        relation = torch.cat((p, delta[..., 2:3].sin(), delta[..., 2:3].cos(),
                              v, delta[..., 5:6]), -1)
        candidates = torch.cat((proposals, boids), -1)
        return torch.cat((relation, candidates.unsqueeze(-2).expand(-1, -1, n, -1),
                          candidates.unsqueeze(-3).expand(-1, n, -1, -1)), -1)

    def forward(self, obs, motion, mask=None, deterministic=False, noise=None):
        b, n = obs.shape[:2]
        if mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=obs.device)
        h = self.encode(obs, mask)
        proposals, log_l = transformed_normal(self.base.mean_layer(h),
            self.base.log_std_layer(h).clamp(-20, 2),
            None if noise is None else noise[..., :2], deterministic)
        boids = obs[..., 12:14]
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pair_mask &= ~torch.eye(n, dtype=torch.bool, device=obs.device)
        edges = self.relations(self.relation_features(motion, proposals, boids))
        message = masked_mean(edges, pair_mask, -2)
        own = self.base.activation(self.base.a1(torch.cat((proposals, boids), -1)))
        mean = self.base.adap_layer(torch.cat((own, h), -1)) + self.relation_gate(message)
        log_std = self.gate_log_std(torch.cat((h, message), -1)).clamp(-5, 2)
        gates, log_g = transformed_normal(mean, log_std,
            None if noise is None else noise[..., 2:3], deterministic, gate=True)
        action = torch.cat((proposals, gates), -1)
        logp = ((log_l + log_g) * mask.unsqueeze(-1)).sum(-2)
        return action, logp


class TeamValue(nn.Module):
    def __init__(self, hidden=512, relation=128):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(12, relation), nn.LeakyReLU(),
                                  nn.Linear(relation, relation), nn.LeakyReLU())
        self.pair = nn.Sequential(nn.Linear(2 * relation + 1, relation), nn.LeakyReLU(),
                                  nn.Linear(relation, relation), nn.LeakyReLU())
        self.output = nn.Sequential(nn.Linear(2 * relation + 10, hidden), nn.LeakyReLU(),
                                    nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1))

    def forward(self, packet, action, mask=None):
        b, n = action.shape[:2]
        if mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=action.device)
        central = packet['central']
        if central.shape[-1] != 7 * (n + 1) + 2:
            raise ValueError('Central state does not match the team size.')
        state = central[..., :7*n].reshape(b, n, 7)
        nodes = self.node(torch.cat((state, action, packet['obs'][..., 12:14]), -1))
        left, right = nodes.unsqueeze(-2), nodes.unsqueeze(-3)
        distance = (state[..., :2].unsqueeze(-2) - state[..., :2].unsqueeze(-3)).norm(dim=-1, keepdim=True)
        pairs = self.pair(torch.cat((left + right, (left - right).abs(), distance), -1))
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pair_mask &= ~torch.eye(n, dtype=torch.bool, device=action.device)
        pair_pool = masked_mean(pairs.flatten(1, 2), pair_mask.flatten(1, 2), 1)
        task = torch.cat((central[..., 7*n:], mask.sum(-1, keepdim=True).to(action.dtype) / 7.), -1)
        return self.output(torch.cat((masked_mean(nodes, mask, 1), pair_pool, task), -1))


class TwinTeamCritic(nn.Module):
    def __init__(self, hidden=512, relation=128):
        super().__init__()
        self.q1 = TeamValue(hidden, relation)
        self.q2 = TeamValue(hidden, relation)

    def forward(self, packet, action, mask=None):
        return self.q1(packet, action, mask), self.q2(packet, action, mask)


class JointReplay:
    def __init__(self, capacity=1_000_000):
        self.capacity, self.count, self.size = int(capacity), 0, 0
        self.arrays = {}

    def store(self, before, action, executed, individual_reward, after, terminated):
        record = {**before, **{'next_' + k: v for k, v in after.items()},
                  'action': action, 'executed': executed,
                  'individual_reward': individual_reward,
                  'reward': np.asarray([np.mean(individual_reward)]),
                  'done': np.asarray([float(terminated)])}
        for key, value in record.items():
            value = np.asarray(value, dtype=np.float32)
            if not np.isfinite(value).all():
                raise FloatingPointError('Non-finite replay field: ' + key)
            if key not in self.arrays:
                self.arrays[key] = np.empty((self.capacity, *value.shape), dtype=np.float32)
            if self.arrays[key].shape[1:] != value.shape:
                raise ValueError('Cannot mix incompatible team dimensions in one replay.')
            self.arrays[key][self.count] = value
        self.count = (self.count + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device='cpu'):
        if not self.size:
            raise ValueError('Empty replay.')
        index = np.random.randint(self.size, size=batch_size)
        return {k: torch.as_tensor(v[index], device=device) for k, v in self.arrays.items()}

    def state_dict(self):
        return dict(capacity=self.capacity, count=self.count, size=self.size,
                    arrays={k: v[:self.size].copy() for k, v in self.arrays.items()})

    def load_state_dict(self, state):
        self.capacity, self.count, self.size = state['capacity'], state['count'], state['size']
        self.arrays = {}
        for k, v in state['arrays'].items():
            self.arrays[k] = np.empty((self.capacity, *v.shape[1:]), dtype=np.float32)
            self.arrays[k][:self.size] = v


def current_and_next(batch):
    return ({k: batch[k] for k in PACKET_KEYS}, {k: batch['next_' + k] for k in PACKET_KEYS})


class InteractionSAC:
    legacy = False

    def __init__(self, config, device='cpu'):
        self.config, self.device = copy.deepcopy(config), torch.device(device)
        rl = config['rl']
        self.gamma, self.tau = rl['GAMMA'], rl['TAU']
        self.batch_size = rl['batch_size']
        self.actor = InteractionActor(rl['hidden_dim'], config['interaction']['relation_dim']).to(self.device)
        self.critic = TwinTeamCritic(rl['hidden_dim'], config['interaction']['relation_dim']).to(self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=rl['learning_rate'], eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=rl['learning_rate'], eps=1e-5)
        self.log_alpha = torch.tensor(math.log(.2), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=rl['learning_rate'])
        self.set_stage('gate')

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def set_stage(self, stage):
        if stage not in ('gate', 'joint'):
            raise ValueError('Unknown training stage.')
        self.stage = stage
        self.actor.freeze_proposals(stage == 'gate')

    @torch.no_grad()
    def choose_action(self, packet, deterministic=False, noise=None):
        obs = torch.as_tensor(packet['obs'], dtype=torch.float32, device=self.device).unsqueeze(0)
        motion = torch.as_tensor(packet['motion'], dtype=torch.float32, device=self.device).unsqueeze(0)
        if noise is not None:
            noise = torch.as_tensor(noise, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp = self.actor(obs, motion, deterministic=deterministic, noise=noise)
        return action[0].cpu().numpy(), float(logp.item())

    def learn(self, replay, auxiliary=None):
        batch = replay.sample(self.batch_size, self.device)
        packet, following = current_and_next(batch)
        with torch.no_grad():
            next_action, next_logp = self.actor(following['obs'], following['motion'])
            q1, q2 = self.target(following, next_action)
            target = batch['reward'] + self.gamma * (1. - batch['done']) * (torch.minimum(q1, q2) - self.alpha * next_logp)
        q1, q2 = self.critic(packet, batch['action'])
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        auxiliary_loss = torch.zeros((), device=self.device)
        kind = self.config['interaction']['auxiliary']
        if auxiliary is not None and kind != 'none':
            ap = tensor_packet(auxiliary, self.device)
            a = torch.as_tensor(auxiliary['action'], device=self.device)
            cf = torch.as_tensor(auxiliary['counterfactual'], device=self.device)
            g = torch.as_tensor(auxiliary['return'], device=self.device).detach()
            gc = torch.as_tensor(auxiliary['counterfactual_return'], device=self.device).detach()
            qa, qb = self.critic(ap, a), self.critic(ap, cf)
            for left, right in zip(qa, qb):
                if kind == 'difference':
                    auxiliary_loss = auxiliary_loss + F.mse_loss(left - right, g - gc)
                elif kind == 'value':
                    auxiliary_loss = auxiliary_loss + .5 * (F.mse_loss(left, g) + F.mse_loss(right, gc))
                else:
                    raise ValueError('Unknown auxiliary objective.')
        critic_loss = td_loss + self.config['interaction']['auxiliary_weight'] * auxiliary_loss
        if not torch.isfinite(critic_loss):
            raise FloatingPointError('Non-finite critic loss.')
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        action, logp = self.actor(packet['obs'], packet['motion'])
        q1, q2 = self.critic(packet, action)
        actor_loss = (self.alpha * logp - torch.minimum(q1, q2)).mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError('Non-finite policy loss.')
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        if self.stage == 'joint':
            target_entropy = -3 * action.shape[-2]
            alpha_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        return dict(td_loss=float(td_loss.detach()), auxiliary_loss=float(auxiliary_loss.detach()),
                    actor_loss=float(actor_loss.detach()), alpha=float(self.alpha))

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), target=self.target.state_dict(),
                    actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                    log_alpha=self.log_alpha.detach(), alpha_optimizer=self.alpha_optimizer.state_dict(), stage=self.stage)

    def load_state_dict(self, state):
        for name in ('actor', 'critic', 'target', 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
        self.set_stage(state['stage'])


class LegacyCBFSAC:
    """Released individual-critic SAC, with the common physical safety layer."""
    legacy = True

    def __init__(self, config, device='cpu'):
        self.config, self.device = copy.deepcopy(config), torch.device(device)
        self.sac = SAC(_dict_to_namespace(config), 6, 8, 3, adaptive=True, device=self.device)
        self.actor = self.sac.actor
        with torch.no_grad():
            self.sac.log_alpha.fill_(math.log(.2))
        self.sac.alpha = self.sac.log_alpha.exp().detach().to(self.device)
        self.set_stage('gate')

    def set_stage(self, stage):
        self.stage = stage
        self.sac.adaptive_alpha = stage == 'joint'
        for name, parameter in self.actor.named_parameters():
            if name.split('.')[0] in ('f1', 'f2', 't1', 'l1', 'l2', 'mean_layer', 'log_std_layer'):
                parameter.requires_grad_(stage == 'joint')

    def choose_action(self, packet, deterministic=False, noise=None):
        action = self.sac.choose_action(packet['obs'], deterministic).reshape(-1, 3)
        if not deterministic:
            action[:, 2] = np.clip(action[:, 2] + np.random.normal(0., .1, len(action)), 0., 1.)
        return action, 0.

    def learn(self, replay, auxiliary=None):
        class LocalView:
            def sample(self, batch_size, device):
                n = replay.arrays['action'].shape[1]
                index = np.random.randint(replay.size * n, size=batch_size)
                row, boat = index // n, index % n
                fields = replay.arrays
                data = (fields['obs'][row, boat], fields['action'][row, boat],
                        fields['individual_reward'][row, boat, None], fields['next_obs'][row, boat],
                        fields['done'][row])
                return tuple(torch.as_tensor(value, device=device) for value in data)
        self.sac.learn(LocalView())
        return dict(alpha=float(self.sac.alpha.detach()))

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.sac.critic.state_dict(),
                    target=self.sac.critic_target.state_dict(), actor_optimizer=self.sac.actor_optimizer.state_dict(),
                    critic_optimizer=self.sac.critic_optimizer.state_dict(), log_alpha=self.sac.log_alpha.detach(),
                    alpha_optimizer=self.sac.alpha_optimizer.state_dict(), stage=self.stage)

    def load_state_dict(self, state):
        mapping = dict(actor='actor', critic='critic', target='critic_target', actor_optimizer='actor_optimizer',
                       critic_optimizer='critic_optimizer', alpha_optimizer='alpha_optimizer')
        for source, target in mapping.items():
            getattr(self.sac, target).load_state_dict(state[source])
        with torch.no_grad():
            self.sac.log_alpha.copy_(state['log_alpha'].cpu())
        self.sac.alpha = self.sac.log_alpha.exp().detach().to(self.device)
        self.set_stage(state['stage'])


def make_agent(config, device='cpu'):
    return (LegacyCBFSAC if config['interaction']['arm'] == 'arboids_cbf' else InteractionSAC)(config, device)
