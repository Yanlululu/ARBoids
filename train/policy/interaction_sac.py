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


PACKET_KEYS = ('obs', 'motion', 'central')


def bootstrap_value(q1, q2, estimator='min'):
    if estimator == 'mean':
        return .5 * (q1 + q2)
    if estimator == 'min':
        return torch.minimum(q1, q2)
    raise ValueError('Unknown bootstrap estimator.')


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
    def __init__(self, hidden=512, relation=128, peer_candidates=True, entropy_objective='joint-sum-v1'):
        super().__init__()
        self.hidden, self.relation = hidden, relation
        self.peer_candidates = bool(peer_candidates)
        if entropy_objective not in ('joint-sum-v1', 'stage-mean-v2', 'proposal-mean-v3', 'task-return-v4'):
            raise ValueError('Unknown entropy objective.')
        self.entropy_objective, self.stage = entropy_objective, 'gate'
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
        self.stage = 'gate' if freeze else 'joint'
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
        peers = candidates if self.peer_candidates else torch.zeros_like(candidates)
        return torch.cat((relation, candidates.unsqueeze(-2).expand(-1, -1, n, -1),
                          peers.unsqueeze(-3).expand(-1, n, -1, -1)), -1)

    def forward(self, obs, motion, mask=None, deterministic=False, noise=None):
        motion = motion.to(dtype=obs.dtype)
        b, n = obs.shape[:2]
        if mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=obs.device)
        h = self.encode(obs, mask)
        proposals, log_l = transformed_normal(self.base.mean_layer(h),
            self.base.log_std_layer(h).clamp(-20, 2),
            None if noise is None else noise[..., :2], deterministic)
        boids = obs[..., 12:14]
        gates, log_g = self.gates(h, motion, proposals, boids, mask, deterministic,
                                None if noise is None else noise[..., 2:3])
        action = torch.cat((proposals, gates), -1)
        if self.entropy_objective == 'task-return-v4':
            logp = torch.zeros((b, 1), dtype=obs.dtype, device=obs.device)
        elif self.entropy_objective == 'proposal-mean-v3':
            # Blending noise receives no intrinsic reward. Gate-only learning
            # optimizes the physical task; joint learning regularizes only
            # the trainable proposal distribution, as in the source policy.
            logp = (masked_mean(log_l, mask, -2) if self.stage == 'joint'
                    else torch.zeros((b, 1), dtype=obs.dtype, device=obs.device))
        elif self.entropy_objective == 'stage-mean-v2':
            # Team reward is a mean. Match its fleet-size scale, and in the
            # gate-only MDP treat frozen proposals as exogenous candidates.
            # Their state-dependent density must not become a gate reward.
            logp = masked_mean(log_g if self.stage == 'gate' else log_l + log_g, mask, -2)
        else:
            logp = ((log_l + log_g) * mask.unsqueeze(-1)).sum(-2)
        return action, logp

    def gates(self, h, motion, proposals, boids, mask, deterministic=False, noise=None):
        """Gate evaluation with explicit candidates, also used by controlled diagnostics."""
        n = proposals.shape[-2]
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pair_mask &= ~torch.eye(n, dtype=torch.bool, device=proposals.device)
        edges = self.relations(self.relation_features(motion, proposals, boids))
        message = masked_mean(edges, pair_mask, -2)
        own = self.base.activation(self.base.a1(torch.cat((proposals, boids), -1)))
        mean = self.base.adap_layer(torch.cat((own, h), -1)) + self.relation_gate(message)
        log_std = self.gate_log_std(torch.cat((h, message), -1)).clamp(-5, 2)
        return transformed_normal(mean, log_std, noise, deterministic, gate=True)


class TeamValue(nn.Module):
    def __init__(self, hidden=512, relation=128, control_coordinates='raw-v1'):
        super().__init__()
        if control_coordinates not in ('raw-v1','nominal-thrust-v2'):
            raise ValueError('Unknown critic control coordinates.')
        self.control_coordinates = control_coordinates
        if control_coordinates != 'raw-v1':
            self.register_buffer('control_coordinates_version',torch.tensor(2,dtype=torch.int64))
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
        boids = packet['obs'][..., 12:14]
        controls = action
        if self.control_coordinates == 'nominal-thrust-v2':
            # Physical thrust is 750 * nominal + 250. CBF receives this
            # composition only; its arbitrary proposal/gate factorization
            # cannot change a root action value. Root entropy is outside Q.
            nominal = action[..., 2:3] * action[..., :2] + (1.-action[..., 2:3]) * boids
            controls = torch.cat((nominal, nominal[..., :1]-nominal[..., 1:2]), -1)
        nodes = self.node(torch.cat((state, controls, boids), -1))
        left, right = nodes.unsqueeze(-2), nodes.unsqueeze(-3)
        distance = (state[..., :2].unsqueeze(-2) - state[..., :2].unsqueeze(-3)).norm(dim=-1, keepdim=True)
        pairs = self.pair(torch.cat((left + right, (left - right).abs(), distance), -1))
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pair_mask &= ~torch.eye(n, dtype=torch.bool, device=action.device)
        pair_pool = masked_mean(pairs.flatten(1, 2), pair_mask.flatten(1, 2), 1)
        task = torch.cat((central[..., 7*n:], mask.sum(-1, keepdim=True).to(action.dtype) / 7.), -1)
        return self.output(torch.cat((masked_mean(nodes, mask, 1), pair_pool, task), -1))


class TwinTeamCritic(nn.Module):
    def __init__(self, hidden=512, relation=128, control_coordinates='raw-v1'):
        super().__init__()
        self.q1 = TeamValue(hidden, relation, control_coordinates)
        self.q2 = TeamValue(hidden, relation, control_coordinates)

    def forward(self, packet, action, mask=None):
        return self.q1(packet, action, mask), self.q2(packet, action, mask)


class JointReplay:
    def __init__(self, capacity=1_000_000):
        self.capacity, self.count, self.size = int(capacity), 0, 0
        self.arrays = {}

    def store(self, before, action, executed, individual_reward, after, terminated, *, policy_cost=None):
        record = {**before, **{'next_' + k: v for k, v in after.items()},
                  'action': action, 'executed': executed,
                  'individual_reward': individual_reward,
                  'reward': np.asarray([np.mean(individual_reward)]),
                  'done': np.asarray([float(terminated)])}
        if policy_cost is not None:
            record['policy_cost'] = np.asarray([policy_cost])
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

    def complete_returns(self, gamma):
        """Warmup Q targets from complete real episodes, excluding root entropy."""
        if self.count != self.size or 'policy_cost' not in self.arrays:
            raise ValueError('Complete real-return warmup requires intact on-policy history and policy costs.')
        terminal = np.flatnonzero(self.arrays['done'][:self.size, 0])
        if not len(terminal):
            raise ValueError('No completed real episode is available for critic warmup.')
        stop = int(terminal[-1]) + 1
        returns = np.empty(stop, dtype=np.float64)
        for i in range(stop - 1, -1, -1):
            returns[i] = float(self.arrays['reward'][i, 0])
            if not self.arrays['done'][i, 0]:
                returns[i] += gamma * (returns[i + 1] - self.arrays['policy_cost'][i + 1, 0])
        return returns.astype(np.float32)

    def state_dict(self):
        # Adjacent transitions share the same physical state. Store next-state
        # exceptions at episode/ring boundaries, without reducing precision.
        arrays, exceptions = {}, {}
        for key, value in self.arrays.items():
            if not key.startswith('next_'):
                arrays[key] = value[:self.size].copy()
                continue
            expected = np.roll(self.arrays[key[5:]][:self.size], -1, axis=0)
            different = np.any(value[:self.size] != expected, axis=tuple(range(1, value.ndim)))
            index = np.flatnonzero(different)
            exceptions[key] = dict(index=index, values=value[index].copy())
        return dict(capacity=self.capacity, count=self.count, size=self.size,
                    arrays=arrays, next_exceptions=exceptions)

    def load_state_dict(self, state):
        self.capacity, self.count, self.size = state['capacity'], state['count'], state['size']
        self.arrays = {}
        for k, v in state['arrays'].items():
            self.arrays[k] = np.empty((self.capacity, *v.shape[1:]), dtype=np.float32)
            self.arrays[k][:self.size] = v
        for key, values in state.get('next_exceptions', {}).items():
            original = self.arrays[key[5:]]
            self.arrays[key] = np.empty_like(original)
            self.arrays[key][:self.size] = np.roll(original[:self.size], -1, axis=0)
            self.arrays[key][values['index']] = values['values']


def current_and_next(batch):
    return ({k: batch[k] for k in PACKET_KEYS}, {k: batch['next_' + k] for k in PACKET_KEYS})


class InteractionSAC:
    legacy = False

    def __init__(self, config, device='cpu'):
        self.config, self.device = copy.deepcopy(config), torch.device(device)
        rl = config['rl']
        self.gamma, self.tau = rl['GAMMA'], rl['TAU']
        self.batch_size = rl['batch_size']
        self.entropy_objective = config['interaction'].get('entropy_objective', 'joint-sum-v1')
        self.actor = InteractionActor(rl['hidden_dim'], config['interaction']['relation_dim'],
            peer_candidates=config['interaction'].get('peer_candidates', True),
            entropy_objective=self.entropy_objective).to(self.device)
        self.critic_control_coordinates = config['interaction'].get('critic_control_coordinates','raw-v1')
        self.critic = TwinTeamCritic(rl['hidden_dim'], config['interaction']['relation_dim'],
                                   self.critic_control_coordinates).to(self.device)
        if config['training'].get('critic_warmup_target') == 'complete_real_state_return':
            # First fit a state-value prior, rather than initializing control
            # sensitivities from one realized trajectory at each state.
            with torch.no_grad():
                for head in (self.critic.q1,self.critic.q2):
                    head.node[0].weight[:,7:10].zero_()
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        # Historical checkpoints used the supervised critic as its own teacher.
        # Keep that interpretation readable, but new runs bootstrap exclusively
        # from a separate critic fitted to real transitions. Auxiliary gradients
        # must never reach this critic or its Polyak target.
        self.bootstrap_source = config['interaction'].get('bootstrap_source', 'coupled')
        if self.bootstrap_source not in ('coupled', 'real_td'):
            raise ValueError('Unknown bootstrap source.')
        self.bootstrap_estimator = config['interaction'].get('bootstrap_estimator', 'min')
        if self.bootstrap_estimator not in ('min', 'mean'):
            raise ValueError('Unknown bootstrap estimator.')
        self.bootstrap_critic = copy.deepcopy(self.critic) if self.bootstrap_source == 'real_td' else None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=rl.get('actor_learning_rate',rl['learning_rate']), eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=rl['learning_rate'], eps=1e-5)
        self.bootstrap_optimizer = (torch.optim.Adam(self.bootstrap_critic.parameters(),
            lr=rl['learning_rate'], eps=1e-5) if self.bootstrap_critic is not None else None)
        self.log_alpha = torch.tensor(math.log(.2), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=rl.get('temperature_learning_rate',rl['learning_rate']))
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

    def paired_gate_loss(self, auxiliary):
        """Distill better executed gate interventions with fixed root candidates.

        Return gaps weight improvements; non-focal gates stay near the behavior
        that generated this paired comparison. Neither critic supplies targets.
        """
        if auxiliary is None:
            raise ValueError('Paired gate improvement requires intervention returns.')
        ap = tensor_packet(auxiliary, self.device)
        action = torch.as_tensor(auxiliary['action'], device=self.device).detach()
        alternative = torch.as_tensor(auxiliary['counterfactual'], device=self.device).detach()
        gap = (torch.as_tensor(auxiliary['return'], device=self.device) -
               torch.as_tensor(auxiliary['counterfactual_return'], device=self.device)).detach()
        changed = (action[..., 2] - alternative[..., 2]).abs() > 1e-6
        single = self.config['interaction'].get('intervention_scope', 'single') == 'single'
        if (single and (changed.sum(-1) > 1).any()) or not torch.equal(action[..., :2], alternative[..., :2]):
            raise ValueError('Gate improvement requires one focal gate and shared candidates.')
        mask = torch.ones(action.shape[:2], dtype=torch.bool, device=self.device)
        h = self.actor.encode(ap['obs'], mask)
        prediction, _ = self.actor.gates(h, ap['motion'].to(h.dtype), action[..., :2],
                                         ap['obs'][..., 12:14], mask, deterministic=True)
        prediction = prediction.squeeze(-1)
        target = torch.where(gap >= 0., action[..., 2], alternative[..., 2])
        weights = gap.abs().clamp(max=10.) * changed
        improvement = (weights * (prediction - target).square()).sum() / weights.sum().clamp_min(1e-8)
        anchor = ((prediction - action[..., 2]).square() * ~changed).sum() / (~changed).sum().clamp_min(1)
        return improvement + self.config['interaction'].get('gate_anchor_weight', .05) * anchor

    def learn(self, replay, auxiliary=None, *, diagnostics=False, update_actor=True):
        if (self.stage == 'gate' and
                self.config['interaction'].get('gate_objective', 'critic-v1') == 'paired-improvement-v1'):
            # Complete Monte Carlo improvements need neither a value tail nor
            # a critic gradient. Spend this phase on control improvement.
            loss = self.paired_gate_loss(auxiliary) if update_actor else torch.zeros((), device=self.device)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite paired gate loss.')
            self.actor_optimizer.zero_grad(set_to_none=True)
            checks = {}
            if update_actor:
                loss.backward()
                gradients = [p.grad for p in self.actor.parameters() if p.grad is not None]
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError('Missing or non-finite paired gate gradients.')
                if diagnostics:
                    checks['actor_gradient_norm'] = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
                self.actor_optimizer.step()
            return dict(actor_loss=float(loss.detach()), actor_updated=bool(update_actor),
                        critic_updated=False, bootstrap_updated=False, alpha=float(self.alpha), **checks)
        batch = replay.sample(self.batch_size, self.device)
        packet, following = current_and_next(batch)
        with torch.no_grad():
            next_action, next_logp = self.actor(following['obs'], following['motion'])
            q1, q2 = self.target(following, next_action)
            next_value = bootstrap_value(q1, q2, self.bootstrap_estimator)
            target = batch['reward'] + self.gamma * (1. - batch['done']) * (next_value - self.alpha * next_logp)
            bootstrap_disagreement = (q1 - q2).abs().mean()
            bootstrap_value_mean = next_value.mean()
        bootstrap_loss = self.fit_bootstrap(packet, batch['action'], target)
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
        checks = {}
        def gradient_norm(module):
            gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError('Missing or non-finite training gradients.')
            return float(torch.sqrt(sum(g.square().sum() for g in gradients)))
        if diagnostics:
            checks.update(critic_gradient_norm=gradient_norm(self.critic),
                weighted_auxiliary_loss=float((self.config['interaction']['auxiliary_weight'] * auxiliary_loss).detach()),
                td_target_abs_max=float(target.abs().max()), bootstrap_disagreement=float(bootstrap_disagreement),
                bootstrap_value_mean=float(bootstrap_value_mean), entropy_cost_mean=float((self.alpha * next_logp).mean()))
            if bootstrap_loss is not None:
                checks['bootstrap_td_loss'] = bootstrap_loss
                checks['bootstrap_gradient_norm'] = gradient_norm(self.bootstrap_critic)
            if auxiliary is not None:
                checks['model_target_abs_max'] = float(max(abs(auxiliary['return']).max(),
                                                         abs(auxiliary['counterfactual_return']).max()))
                if 'difference_standard_error' in auxiliary:
                    checks['auxiliary_difference_standard_error']=float(np.mean(auxiliary['difference_standard_error']))
        self.critic_optimizer.step()
        if diagnostics and self.bootstrap_critic is not None:
            storage=lambda optimizer:{(str(v.device),v.data_ptr()) for values in optimizer.state.values()
                for v in values.values() if isinstance(v,torch.Tensor)}
            aliases=len(storage(self.critic_optimizer)&storage(self.bootstrap_optimizer))
            checks['bootstrap_optimizer_aliases']=aliases
            if aliases:
                raise RuntimeError('Main and bootstrap optimizers must have independent state storage.')
            if kind=='none':
                delta=max(float((left-right).detach().abs().max()) for left,right in
                          zip(self.critic.parameters(),self.bootstrap_critic.parameters()))
                checks['same_info_critic_max_difference']=delta
                if delta!=0.:
                    raise RuntimeError('Same-info main and bootstrap critics diverged despite identical objectives.')
        self.critic.requires_grad_(False)
        if not update_actor:
            actor_loss = torch.zeros((), device=self.device)
        else:
            action, logp = self.actor(packet['obs'], packet['motion'])
            q1, q2 = self.critic(packet, action)
            actor_loss = (self.alpha * logp - torch.minimum(q1, q2)).mean()
        if not torch.isfinite(actor_loss):
            raise FloatingPointError('Non-finite policy loss.')
        self.actor_optimizer.zero_grad(set_to_none=True)
        if update_actor:
            actor_loss.backward()
        if diagnostics and update_actor:
            checks['actor_gradient_norm'] = gradient_norm(self.actor)
            checks['adapter_gradient_norm'] = gradient_norm(self.actor.base.adap_layer)
            checks['candidate_gate_gradient_norm'] = gradient_norm(self.actor.relation_gate)
        if update_actor:
            self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        if (update_actor and self.entropy_objective != 'task-return-v4' and
                (self.stage == 'joint' or self.entropy_objective == 'stage-mean-v2')):
            target_entropy = (-2 if self.entropy_objective == 'proposal-mean-v3' else
                ((-1 if self.stage == 'gate' else -3)
                 if self.entropy_objective == 'stage-mean-v2' else -3 * action.shape[-2]))
            alpha_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            source = self.bootstrap_critic if self.bootstrap_critic is not None else self.critic
            for target_parameter, parameter in zip(self.target.parameters(), source.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        return dict(td_loss=float(td_loss.detach()), auxiliary_loss=float(auxiliary_loss.detach()),
                    actor_loss=float(actor_loss.detach()), actor_updated=bool(update_actor), critic_updated=True,
                    bootstrap_updated=self.bootstrap_critic is not None,
                    alpha=float(self.alpha), **checks)

    def initialize_critic_from_bootstrap(self):
        if self.bootstrap_critic is None:
            raise ValueError('A separate bootstrap critic is required.')
        self.critic.load_state_dict(self.bootstrap_critic.state_dict())
        # Adam load_state_dict can retain same-device tensor storage, including
        # CPU step counters for CUDA optimizers. Values must match, storage must not.
        self.critic_optimizer.load_state_dict(copy.deepcopy(self.bootstrap_optimizer.state_dict()))

    def fit_bootstrap(self, packet, action, target, *, state_only=False):
        """Only real-transition Bellman targets may update the bootstrap critic."""
        if self.bootstrap_critic is None:
            return None
        q1, q2 = self.bootstrap_critic(packet, action)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite real-transition bootstrap loss.')
        self.bootstrap_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if state_only:
            for head in (self.bootstrap_critic.q1,self.bootstrap_critic.q2):
                head.node[0].weight.grad[:,7:10].zero_()
        self.bootstrap_optimizer.step()
        return float(loss.detach())

    def learn_bootstrap(self, replay):
        """Recondition a legacy teacher with fixed actor and real replay only."""
        if self.bootstrap_critic is None:
            raise ValueError('A separate real-transition bootstrap critic is required.')
        batch = replay.sample(self.batch_size, self.device)
        packet, following = current_and_next(batch)
        with torch.no_grad():
            action, logp = self.actor(following['obs'], following['motion'])
            q1, q2 = self.target(following, action)
            next_value = bootstrap_value(q1, q2, self.bootstrap_estimator)
            target = batch['reward'] + self.gamma * (1. - batch['done']) * (next_value - self.alpha * logp)
        loss = self.fit_bootstrap(packet, batch['action'], target)
        with torch.no_grad():
            for slow, current in zip(self.target.parameters(), self.bootstrap_critic.parameters()):
                slow.lerp_(current, self.tau)
        return dict(bootstrap_td_loss=loss, bootstrap_disagreement=float((q1-q2).abs().mean()),
                    bootstrap_value_mean=float(next_value.mean()), td_target_abs_max=float(target.abs().max()))

    def learn_bootstrap_returns(self, replay, returns):
        """Initialize values before policy learning, without a learned value tail."""
        index = np.random.randint(len(returns), size=self.batch_size)
        packet = {k: torch.as_tensor(replay.arrays[k][index], device=self.device) for k in PACKET_KEYS}
        action = torch.as_tensor(replay.arrays['action'][index], device=self.device)
        target = torch.as_tensor(returns[index, None], device=self.device)
        loss = self.fit_bootstrap(packet, action, target,
            state_only=self.config['training'].get('critic_warmup_target') == 'complete_real_state_return')
        with torch.no_grad():
            for slow, current in zip(self.target.parameters(), self.bootstrap_critic.parameters()):
                slow.lerp_(current, self.tau)
        return dict(bootstrap_mc_loss=loss, complete_return_samples=len(returns),
                    real_return_target_mean=float(target.mean()), real_return_target_abs_max=float(target.abs().max()))

    def state_dict(self):
        state = dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), target=self.target.state_dict(),
                    actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                    log_alpha=self.log_alpha.detach(), alpha_optimizer=self.alpha_optimizer.state_dict(), stage=self.stage,
                    critic_control_coordinates=self.critic_control_coordinates,
                    entropy_objective=self.entropy_objective, bootstrap_estimator=self.bootstrap_estimator)
        if self.bootstrap_critic is not None:
            state.update(bootstrap_critic=self.bootstrap_critic.state_dict(),
                         bootstrap_optimizer=self.bootstrap_optimizer.state_dict())
        return state

    def load_state_dict(self, state):
        if state.get('bootstrap_estimator', 'min') != self.bootstrap_estimator:
            raise ValueError('Bootstrap estimator changed; a fresh matched training run is required.')
        if state.get('entropy_objective', 'joint-sum-v1') != self.entropy_objective:
            raise ValueError('Entropy objective changed; a fresh matched training run is required.')
        if state.get('critic_control_coordinates','raw-v1') != self.critic_control_coordinates:
            raise ValueError('Critic control coordinates changed; a fresh matched training run is required.')
        if (self.bootstrap_critic is not None) != ('bootstrap_critic' in state):
            raise ValueError('Bootstrap protocol changed; use explicit checkpoint reconditioning.')
        for name in ('actor', 'critic', 'target', 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
            value=copy.deepcopy(state[name]) if name.endswith('optimizer') else state[name]
            getattr(self, name).load_state_dict(value)
        if self.bootstrap_critic is not None:
            self.bootstrap_critic.load_state_dict(state['bootstrap_critic'])
            self.bootstrap_optimizer.load_state_dict(copy.deepcopy(state['bootstrap_optimizer']))
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
        self.set_stage(state['stage'])


class LegacyCBFSAC:
    """Released individual-critic SAC, with the common physical safety layer."""
    legacy = True

    def __init__(self, config, device='cpu'):
        from policy.SAC import SAC
        from utils.config import _dict_to_namespace
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
