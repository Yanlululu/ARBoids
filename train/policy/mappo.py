"""Factor-clipped, two-stage MAPPO with a projected collision multiplier."""
from dataclasses import dataclass, asdict
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F

from policy.interaction_prediction import PredictionConfig, InteractionPredictor, exchange_candidates
from policy.mappo_networks import PredictiveActor, CentralizedCritic


@dataclass
class MAPPOConfig:
    hidden_dim: int = 512
    relation_dim: int = 64
    coordination_dim: int = 16
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    ppo_epochs: int = 5
    minibatch_size: int = 128
    clip_ratio: float = .2
    entropy_coef: float = .001
    max_grad_norm: float = .5
    reward_gamma: float = 1.
    cost_gamma: float = 1.
    gae_lambda: float = .95
    collision_budget: float = .05
    lagrange_init: float = 1.
    lagrange_lr: float = 1.

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('All MAPPO settings must be finite.')
        for name in ('hidden_dim', 'relation_dim', 'coordination_dim', 'ppo_epochs', 'minibatch_size'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if min(self.actor_lr, self.critic_lr, self.max_grad_norm, self.lagrange_lr) <= 0:
            raise ValueError('Learning rates and gradient bound must be positive.')
        if not 0 < self.clip_ratio < 1 or not 0 < self.reward_gamma <= 1:
            raise ValueError('Invalid PPO clip ratio or reward discount.')
        if self.cost_gamma != 1.:
            raise ValueError('Binary episode collision probability requires cost_gamma=1.')
        if not 0 <= self.gae_lambda <= 1 or not 0 <= self.collision_budget <= 1:
            raise ValueError('GAE lambda and collision budget must be in [0,1].')
        if self.lagrange_init < 0 or self.entropy_coef < 0:
            raise ValueError('Multiplier and entropy coefficient cannot be negative.')


class LagrangeMultiplier:
    def __init__(self, value, learning_rate, budget):
        self.value, self.learning_rate, self.budget = float(value), float(learning_rate), float(budget)

    def update(self, episode_collision_rate):
        if not np.isfinite(episode_collision_rate) or not 0 <= episode_collision_rate <= 1:
            raise ValueError('Expected the binary collision rate of complete episodes.')
        self.value = max(0., self.value + self.learning_rate * (episode_collision_rate - self.budget))
        return self.value


class PredictiveMAPPO:
    algorithm = 'predictive-mappo'
    format_version = 1

    def __init__(self, config, device='cpu'):
        self.config = config
        self.settings = MAPPOConfig(**config.get('mappo', {}))
        self.prediction_config = PredictionConfig(**config.get('prediction', {}))
        self.device = torch.device(device)
        self.defender_num = int(config['agent']['defender_num'])
        if self.defender_num < 2:
            raise ValueError('At least two defenders are required.')
        s = self.settings
        self.actor = PredictiveActor(s.hidden_dim, s.relation_dim, s.coordination_dim).to(self.device)
        self.critic = CentralizedCritic(7 * (self.defender_num + 1) + 2, s.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=s.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=s.critic_lr, eps=1e-5)
        self.predictor = InteractionPredictor(self.prediction_config)
        self.lagrange = LagrangeMultiplier(s.lagrange_init, s.lagrange_lr, s.collision_budget)
        self.total_steps = 0
        self.updates = 0

    def tensor(self, value, boolean=False):
        return torch.as_tensor(value, dtype=torch.bool if boolean else torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, observation, states, boids, timestamp, deterministic=False):
        """The only deployment path: propose -> exchange -> predict -> gate."""
        observation = np.asarray(observation, dtype=np.float32)
        n = self.defender_num
        if observation.shape != (n, 14 + 2 * (n - 1)) or not np.isfinite(observation).all():
            raise ValueError('Wrong observation shape or non-finite observation.')
        obs = self.tensor(observation).unsqueeze(0)
        proposal, raw, logp_l = self.actor.sample_proposals(obs, deterministic)
        proposals = proposal[0].cpu().numpy()
        messages = exchange_candidates(states, boids, proposals, timestamp)
        if len(messages.states) != n:
            raise ValueError('Message count must match the checkpoint defender count.')
        edges, mask = self.predictor.features(messages)
        gates, gate_raw, logp_g, _ = self.actor.sample_gates(
            obs, proposal, self.tensor(np.array(messages.boids)).unsqueeze(0),
            self.tensor(edges).unsqueeze(0), self.tensor(mask, True).unsqueeze(0), deterministic)
        action = np.concatenate((proposals, gates[0].cpu().numpy()), axis=-1)
        record = dict(obs=observation.copy(), proposals=proposals.copy(), boids=np.array(messages.boids),
                      edges=edges, mask=mask, proposal_raw=raw[0].cpu().numpy(),
                      gate_raw=gate_raw[0].cpu().numpy(), old_logp_l=logp_l[0].cpu().numpy(),
                      old_logp_g=logp_g[0].cpu().numpy(), message_states=np.array(messages.states),
                      timestamp=float(messages.timestamps[0]))
        if not np.isfinite(action).all():
            raise FloatingPointError('Non-finite action.')
        return action, record

    @torch.no_grad()
    def values(self, state):
        r, c = self.critic(self.tensor(state).unsqueeze(0))
        return float(r.item()), float(c.item())

    def evaluate_batch(self, batch):
        return self.actor.evaluate_actions(batch['obs'], batch['proposals'], batch['boids'],
                                           batch['edges'], batch['mask'], batch['proposal_raw'], batch['gate_raw'])

    def update(self, rollout):
        cfg = self.settings
        data = rollout.tensors(cfg, self.device)
        if any(not torch.isfinite(value).all() for value in data.values()):
            raise FloatingPointError('Non-finite rollout data.')
        fixed_lambda = self.lagrange.value
        advantage = data['advantage_r'] - fixed_lambda * data['advantage_c']
        advantage = ((advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)).detach()
        count = len(advantage)
        totals, weight = {}, 0
        self.actor.train()
        self.critic.train()
        for _ in range(cfg.ppo_epochs):
            indices = torch.randperm(count, device=self.device)
            for start in range(0, count, cfg.minibatch_size):
                index = indices[start:start + cfg.minibatch_size]
                batch = {key: value[index] for key, value in data.items()}
                logp_l, logp_g, entropy = self.evaluate_batch(batch)
                delta_l, delta_g = logp_l - batch['old_logp_l'], logp_g - batch['old_logp_g']
                ratio_l, ratio_g = delta_l.exp(), delta_g.exp()
                adv = advantage[index, None]  # One team advantage, broadcast to both stages/all boats.
                def surrogate(ratio):
                    return torch.minimum(ratio * adv, ratio.clamp(1-cfg.clip_ratio, 1+cfg.clip_ratio) * adv)
                actor_loss = -(surrogate(ratio_l) + surrogate(ratio_g)).mean() - cfg.entropy_coef * entropy.mean()
                value_r, value_c = self.critic(batch['state'])
                reward_loss = F.mse_loss(value_r, batch['return_r'])
                cost_loss = F.mse_loss(value_c, batch['return_c'])
                critic_loss = reward_loss + cost_loss
                if not torch.isfinite(actor_loss + critic_loss):
                    raise FloatingPointError('Non-finite PPO loss.')
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm,
                                                            error_if_nonfinite=True)
                gradient_metrics = {}
                for name in ('proposal_mean', 'relation_encoder', 'priority_head', 'coordination_encoder', 'adapter', 'gate_mean'):
                    parameters = getattr(self.actor, name).parameters()
                    gradient_metrics['grad_' + name] = sum(float(p.grad.norm()) for p in parameters if p.grad is not None)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm,
                                                             error_if_nonfinite=True)
                self.critic_optimizer.step()
                metrics = dict(actor_loss=float(actor_loss.detach()), reward_value_loss=float(reward_loss.detach()),
                               cost_value_loss=float(cost_loss.detach()), latent_entropy=float(entropy.mean().detach()),
                               actor_grad_norm=float(actor_norm), critic_grad_norm=float(critic_norm),
                               kl_proposal=float((ratio_l - 1 - delta_l).mean().detach()),
                               kl_gate=float((ratio_g - 1 - delta_g).mean().detach()),
                               clip_fraction=float(((ratio_l - 1).abs() > cfg.clip_ratio).float().mean().detach()),
                               **gradient_metrics)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + value * len(index)
                weight += len(index)
        self.total_steps += rollout.steps
        self.updates += 1
        self.lagrange.update(rollout.collision_rate)  # Exactly once, after all epochs.
        return dict({key: value / weight for key, value in totals.items()},
                    lambda_used=fixed_lambda, lambda_next=self.lagrange.value,
                    collision_rate=rollout.collision_rate, steps=self.total_steps, update=self.updates)

    def save(self, path):
        state = np.random.get_state()
        rng = dict(numpy_kind=state[0], numpy_keys=torch.tensor(state[1].astype(np.int64)),
                   numpy_pos=state[2], numpy_has_gauss=state[3], numpy_cached=state[4],
                   python=random.getstate(), torch=torch.get_rng_state())
        if torch.cuda.is_available():
            rng['cuda'] = torch.cuda.get_rng_state_all()
        torch.save(dict(algorithm=self.algorithm, format_version=self.format_version, config=self.config,
                        actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                        lagrange=self.lagrange.value, total_steps=self.total_steps, updates=self.updates, rng=rng), Path(path))

    @classmethod
    def from_checkpoint(cls, path, device='cpu', restore_rng=False):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('algorithm') != cls.algorithm or saved.get('format_version') != cls.format_version:
            raise ValueError('Not a supported predictive MAPPO checkpoint.')
        agent = cls(saved['config'], device)
        agent.actor.load_state_dict(saved['actor'])
        agent.critic.load_state_dict(saved['critic'])
        agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        agent.lagrange.value = float(saved['lagrange'])
        agent.total_steps, agent.updates = int(saved['total_steps']), int(saved['updates'])
        if not np.isfinite(agent.lagrange.value) or agent.lagrange.value < 0:
            raise ValueError('Invalid checkpoint multiplier.')
        if any(not torch.isfinite(p).all() for net in (agent.actor, agent.critic) for p in net.parameters()):
            raise ValueError('Checkpoint has non-finite parameters.')
        if restore_rng:
            rng = saved['rng']
            np.random.set_state((rng['numpy_kind'], rng['numpy_keys'].numpy().astype(np.uint32),
                                 rng['numpy_pos'], rng['numpy_has_gauss'], rng['numpy_cached']))
            random.setstate(rng['python'])
            torch.set_rng_state(rng['torch'])
            if torch.cuda.is_available() and 'cuda' in rng:
                torch.cuda.set_rng_state_all(rng['cuda'])
        agent.actor.eval()
        agent.critic.eval()
        return agent
