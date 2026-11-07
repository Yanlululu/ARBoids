"""ARBoids/RP/SAC using the author's network definitions in the common task."""
import copy
import importlib.util
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F

from .environment import boids, boids_components, execute
from .networks import frozen


def author_networks():
    candidates = [Path(__file__).resolve().parents[1]/'arboids_networks.py',
                  Path(__file__).resolve().parents[3]/'third_party/arboids_release/train/policy/networks.py']
    source = next(path for path in candidates if path.exists())
    spec = importlib.util.spec_from_file_location('arboids_author_networks', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def polar(vector, yaw, distance_scale):
    bearing = torch.atan2(vector[..., 1], vector[..., 0])-yaw
    bearing = torch.atan2(bearing.sin(), bearing.cos())
    return torch.stack((vector.norm(dim=-1)/distance_scale, bearing/math.pi), -1)


def local_observation(x, include_boids=True):
    p, yaw = x[:, :-1, :2], x[:, :-1, 2]
    components = boids_components(x)
    b = boids(x, components)
    first = torch.cat((polar(-p, yaw, 60), polar(x[:, -1:, :2]-p, yaw, 60),
                       polar(x[:, -1:, 3:5].expand_as(p), yaw, 5),
                       (x[:, :-1, 6:8]-250)/750, x[:, :-1, 8:9]), -1)
    delta = p[:, :, None]-p[:, None, :]
    if include_boids:
        separation, alignment, cohesion = components
        first = torch.cat((first, polar(separation, yaw, 20), polar(alignment, yaw, 1),
                           polar(cohesion, yaw, 3), (b-250)/750), -1)
    n = p.shape[1]
    off_diagonal = ~torch.eye(n, device=x.device, dtype=torch.bool)
    mates = polar(-delta, yaw[:, :, None].expand(-1, -1, n), 60)[:, off_diagonal].reshape(len(x), n, 2*(n-1))
    return torch.cat((first, mates), -1), b


class PublicBaseline(nn.Module):
    def __init__(self, method='arboids', width=512, delta=300.):
        super().__init__()
        if method not in ('arboids', 'rp', 'sac') or width < 4 or width % 4 or delta <= 0:
            raise ValueError('Invalid public baseline method, hidden width or actuator limit.')
        networks = author_networks()
        self.method, self.delta = method, delta
        self.action_dim = 3 if method == 'arboids' else 2
        actor = networks.ActorAdap if method == 'arboids' else networks.ActorSAC
        f2 = 0 if method == 'sac' else 8
        # Original six local coordinates, own previous command and remaining time.
        self.actor = actor(9, f2, self.action_dim, width)
        self.critic = networks.CriticSAC(9, f2, self.action_dim, width)

    def observe(self, x):
        return local_observation(x, self.method != 'sac')

    def command(self, x, latent, b):
        learned = 250+750*latent[..., :2]
        if self.method == 'arboids':
            nominal = b+latent[..., 2:3]*(learned-b)
        elif self.method == 'rp':
            nominal = learned+b
        else:
            nominal = learned
        return execute(nominal, x[:, :-1, 6:8], self.delta)

    @torch.no_grad()
    def select(self, x, deterministic=True, exploration_generator=None, uniform=False):
        observation, b = self.observe(x)
        if uniform:
            latent = 2*torch.rand(len(x), x.shape[1]-1, self.action_dim, device=x.device,
                                  generator=exploration_generator)-1
            if self.method == 'arboids':
                latent[..., 2] = (latent[..., 2]+1)/2
        else:
            latent, _ = self.actor(observation.flatten(0, 1), deterministic=deterministic, with_logprob=False)
            latent = latent.reshape(len(x), x.shape[1]-1, -1)
        if self.method == 'arboids' and not deterministic and not uniform:
            noise = torch.randn(latent[..., 2].shape, device=latent.device,
                                dtype=latent.dtype, generator=exploration_generator)
            latent[..., 2] = (latent[..., 2]+.1*noise).clamp(0, 1)
        return dict(action=self.command(x, latent, b), latent=latent, observation=observation)


class BaselineLearner:
    def __init__(self, model, lr=1e-4, gamma=.99, tau=.005, target_entropy=None):
        self.model, self.target = model, copy.deepcopy(model.critic)
        self.gamma, self.tau = gamma, tau
        self.target_entropy = -float(model.action_dim) if target_entropy is None else target_entropy
        if lr <= 0 or not 0 <= gamma <= 1 or not 0 < tau <= 1 or not math.isfinite(self.target_entropy):
            raise ValueError('Invalid baseline learning recipe.')
        self.target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=lr, eps=1e-5)
        self.log_alpha = torch.zeros((), device=next(model.parameters()).device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.updates = 0

    def update(self, s, a, reward, next_s, done):
        self.updates += 1
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_a, next_logp = self.model.actor(next_s)
            q1, q2 = self.target(next_s, next_a)
            target = reward[:, None]+self.gamma*(1-done.float()[:, None])*(torch.minimum(q1, q2)-alpha*next_logp)
        q1, q2 = self.model.critic(s, a)
        critic_loss = F.mse_loss(q1, target)+F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), float('inf'), error_if_nonfinite=True)
        self.critic_optimizer.step()
        with frozen(self.model.critic):
            policy_a, logp = self.model.actor(s)
            q1, q2 = self.model.critic(s, policy_a)
            actor_loss = (alpha*logp-torch.minimum(q1,q2)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), float('inf'), error_if_nonfinite=True)
            self.actor_optimizer.step()
        # Author recipe: ARBoids -3, RP/SAC -2, with alpha-space optimization.
        alpha_loss = -(self.log_alpha.exp()*(logp.detach()+self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.model.critic.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        return dict(q_loss=float(critic_loss.detach()), actor_loss=float(actor_loss.detach()),
                    alpha_loss=float(alpha_loss.detach()), alpha=float(self.log_alpha.exp().detach()))

    def payload(self):
        return dict(model=self.model.state_dict(), target=self.target.state_dict(),
                    log_alpha=self.log_alpha.detach(), updates=self.updates,
                    learner_recipe=dict(gamma=self.gamma, tau=self.tau, target_entropy=self.target_entropy),
                    optimizers=[self.actor_optimizer.state_dict(), self.critic_optimizer.state_dict(),
                                self.alpha_optimizer.state_dict()])

    def restore(self, payload):
        if payload.get('learner_recipe') != dict(gamma=self.gamma, tau=self.tau, target_entropy=self.target_entropy):
            raise ValueError('The saved baseline entropy, discount and target-update recipe must match.')
        self.model.load_state_dict(payload['model'])
        self.target.load_state_dict(payload['target'])
        with torch.no_grad():
            self.log_alpha.copy_(payload['log_alpha'])
        self.updates = payload['updates']
        for optimizer, state in zip((self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer), payload['optimizers']):
            optimizer.load_state_dict(state)
