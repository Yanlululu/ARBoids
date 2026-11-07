"""Actual-transition off-policy learning for the complete deployed controller."""
import copy
from dataclasses import asdict
import torch

from .environment import fusion
from .networks import frozen


class Replay:
    def __init__(self, capacity, n, device, shapes=None):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive.')
        self.capacity, self.size, self.cursor = capacity, 0, 0
        shapes = shapes or dict(x=(n+1, 9), u=(n, 2), r=(), ri=(n,), next_x=(n+1, 9), done=(),
                                order=(n,), inverse_order=(n,), mask=(n+1,), event=(), seconds=())
        integer = {'order', 'inverse_order', 'event'}
        boolean = {'done', 'mask'}
        self.data = {key: torch.empty((capacity,)+tuple(shape), device=device,
                                     dtype=torch.long if key in integer else
                                     torch.bool if key in boolean else torch.float32)
                     for key, shape in shapes.items()}
        self._terminal_pools = None

    @torch.no_grad()
    def add(self, **values):
        if values.keys() != self.data.keys():
            raise ValueError('Every stored transition must supply the complete replay schema.')
        first = next(iter(values.values()))
        count = len(first)
        # Retain the last capacity rows without duplicate advanced-index writes.
        start = max(0, count-self.capacity)
        index = (torch.arange(start, count, device=first.device)+self.cursor) % self.capacity
        for key, tensor in values.items():
            self.data[key][index] = tensor[start:].detach().to(self.data[key].dtype)
        self.cursor = (self.cursor+count) % self.capacity
        self.size = min(self.capacity, self.size+count)
        self._terminal_pools = None

    def sample(self, count, generator=None):
        index = torch.randint(self.size, (count,), device=next(iter(self.data.values())).device, generator=generator)
        return {key: tensor[index] for key, tensor in self.data.items()}

    def sample_value(self, count, generator=None, terminal_fraction=.5, importance_exponent=.5):
        """Stratify actual terminal events for value fitting; policy states stay uniform.

        The mixture includes uniform replay, so every stored transition keeps
        positive sampling probability. Partial importance weights temper the
        emphasis on rare terminal targets without changing their rewards.
        """
        terminal_count = int(count*terminal_fraction)
        if not terminal_count:
            return self.sample(count, generator)
        if self._terminal_pools is None:
            event, done = self.data['event'][:self.size], self.data['done'][:self.size]
            pools = {kind: ((event == kind) & done).nonzero().flatten() for kind in (1, 2, 3, 4)}
            self._terminal_pools = {kind: ids for kind, ids in pools.items() if len(ids)}
        pools = self._terminal_pools
        if not pools:
            return self.sample(count, generator)
        uniform_count = count-terminal_count
        device = self.data['r'].device
        parts = [torch.randint(self.size, (uniform_count,), device=device, generator=generator)]
        masses = {}
        for position, (kind, pool) in enumerate(pools.items()):
            draws = terminal_count//len(pools)+(position < terminal_count % len(pools))
            parts.append(pool[torch.randint(len(pool), (draws,), device=device, generator=generator)])
            masses[kind] = draws/count/len(pool)
        index = torch.cat(parts)
        uniform_mass = uniform_count/count/self.size
        probability = torch.full((count,), uniform_mass, device=device)
        for kind, mass in masses.items():
            probability += (self.data['event'][index] == kind)*mass
        batch = {key: tensor[index] for key, tensor in self.data.items()}
        batch['value_weight'] = (uniform_mass/probability).pow(importance_exponent)
        return batch

    def state_dict(self):
        return dict(capacity=self.capacity, size=self.size, cursor=self.cursor,
                    data={key: tensor[:self.size].detach().cpu().clone() for key, tensor in self.data.items()})

    @torch.no_grad()
    def load_state_dict(self, saved):
        if saved['capacity'] != self.capacity or saved['data'].keys() != self.data.keys():
            raise ValueError('The replay capacity and schema must match the saved training run.')
        self.size, self.cursor = saved['size'], saved['cursor']
        if not (0 <= self.size <= self.capacity and 0 <= self.cursor < self.capacity):
            raise ValueError('Invalid replay cursor or size in checkpoint.')
        if self.size < self.capacity and self.cursor != self.size:
            raise ValueError('A partially filled replay must end at its saved cursor.')
        for key, tensor in saved['data'].items():
            if tensor.shape != self.data[key][:self.size].shape:
                raise ValueError(f'Replay shape mismatch for {key}.')
            self.data[key][:self.size].copy_(tensor)
        self._terminal_pools = None


class Learner:
    def __init__(self, model, lr=1e-4, gamma=.99, tau=.005, policy_delay=2):
        if lr <= 0 or not 0 <= gamma <= 1 or not 0 < tau <= 1 or policy_delay < 1:
            raise ValueError('Invalid learner learning rate, discount, target rate or policy delay.')
        self.model, self.target = model, copy.deepcopy(model)
        self.target.requires_grad_(False)
        self.gamma, self.tau, self.policy_delay = gamma, tau, policy_delay
        self.q_parameters = list(model.q1.parameters())+list(model.q2.parameters())
        self.q_optimizer = torch.optim.Adam(self.q_parameters, lr=lr)
        self.endpoint_optimizer = torch.optim.Adam(model.endpoint.parameters(), lr=lr)
        self.generator_optimizer = torch.optim.Adam(model.generator.parameters(), lr=lr)
        self.updates = 0

    def set_policy_learning_rate(self, learning_rate):
        for optimizer in (self.endpoint_optimizer, self.generator_optimizer):
            for group in optimizer.param_groups:
                group['lr'] = learning_rate

    def update(self, batch, temperature=1., feedback_scale=1., policy_batch=None):
        if temperature <= 0:
            raise ValueError('Configuration weights need a positive temperature.')
        if not 0 <= feedback_scale <= 1:
            raise ValueError('The feedback ramp must stay between zero and one.')
        self.updates += 1
        x, u, individual_reward, next_x, done = (batch[key] for key in ('x', 'u', 'ri', 'next_x', 'done'))
        # The target search needs derivatives of G even with frozen parameters.
        with torch.no_grad():
            target = individual_reward.clone()
            continuing = ~done.bool()
            if bool(continuing.any()):
                next_state = next_x[continuing]
                next_u = self.target.select(next_state)['action']
                first = self.target.q1(next_state, next_u, individual=True)
                second = self.target.q2(next_state, next_u, individual=True)
                # The same per-vessel clipped value drives deployment, target
                # control and all policy objectives.
                continuation = torch.minimum(first, second)
                target[continuing] += self.gamma*continuation
        q1 = self.model.q1(x, u, individual=True)
        q2 = self.model.q2(x, u, individual=True)
        per_transition_loss = ((q1-target).square()+(q2-target).square()).mean(-1)
        if 'value_weight' in batch:
            q_loss = (batch['value_weight']*per_transition_loss).mean()
        else:
            q_loss = per_transition_loss.mean()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_parameters, 100., error_if_nonfinite=True)
        self.q_optimizer.step()
        report = dict(q_loss=q_loss.detach(), q_mean=q1.detach().mean(), target_mean=target.mean())
        if self.updates % self.policy_delay:
            return report

        if policy_batch is not None:
            x = policy_batch['x']
        result = self.model.select(x)
        g, scores = result['candidates'], result['scores']
        count = g.shape[1]
        # Each optimization chain contributes its best configuration once.
        duplicates = (g[:, :, None] == g[:, None, :]).all(-1)
        previous = torch.ones(count, count, dtype=torch.bool, device=x.device).tril(-1)
        keep = ~(duplicates & previous).any(-1)
        weights = (scores/temperature).masked_fill(~keep, -torch.inf).softmax(-1).detach()

        self.endpoint_optimizer.zero_grad(set_to_none=True)
        self.generator_optimizer.zero_grad(set_to_none=True)
        with frozen(self.model.q1), frozen(self.model.q2):
            # Generator conditions and both optimizers use the SAME endpoint snapshot.
            a = self.model.endpoint(x, result['rule_source'])
            with torch.no_grad():
                encoded = self.model.encode_value(x)
            actions = fusion(x, a, result['b'], g, self.model.cfg.thrust_delta)
            value = self.model.value_from_encoded(encoded, actions)
            endpoint_loss = -(weights*value/100.).sum(-1).mean()
            endpoint_loss.backward()

        initial = result['initial']
        base_scores = result['initial_scores']/100.
        baseline = (base_scores.sum(-1, keepdim=True)-base_scores)/(self.model.cfg.samples-1)
        advantage = (base_scores-baseline).detach()
        base_logp = self.model.generator.log_prob(x, result['a'], result['b'], initial)
        base_loss = -(advantage*base_logp).mean()
        feedback_loss = base_loss.new_zeros(())
        feedback_strength = self.model.cfg.feedback*feedback_scale
        if feedback_strength:
            feedback_logp = self.model.generator.log_prob(x, result['a'], result['b'], g)
            feedback_loss = -(weights*feedback_logp).sum(-1).mean()
        generator_loss = base_loss+feedback_strength*feedback_loss
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.endpoint.parameters(), 10., error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), 10., error_if_nonfinite=True)
        self.endpoint_optimizer.step()
        self.generator_optimizer.step()
        self.soft_update()
        report.update(endpoint_loss=endpoint_loss.detach(), base_loss=base_loss.detach(),
                      feedback_loss=feedback_loss.detach(), feedback_weight=feedback_strength,
                      coefficient=result['g'].mean(),
                      candidate_value_gain=(scores.amax(-1)-result['initial_pool_scores'].amax(-1)).mean())
        return report

    @torch.no_grad()
    def soft_update(self):
        for target_parameter, parameter in zip(self.target.parameters(), self.model.parameters()):
            target_parameter.lerp_(parameter, self.tau)

    def payload(self):
        return dict(model=self.model.state_dict(), target=self.target.state_dict(),
                    config=asdict(self.model.cfg), updates=self.updates,
                    optimizers=[self.q_optimizer.state_dict(), self.endpoint_optimizer.state_dict(),
                                self.generator_optimizer.state_dict()])

    def restore(self, payload):
        self.model.load_state_dict(payload['model'])
        self.target.load_state_dict(payload['target'])
        self.updates = payload['updates']
        for optimizer, state in zip((self.q_optimizer, self.endpoint_optimizer, self.generator_optimizer), payload['optimizers']):
            optimizer.load_state_dict(state)
