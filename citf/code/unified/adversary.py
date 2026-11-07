"""SAC opponent for the prescribed alternating defender/attacker experiment."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from .environment import execute
from .networks import frozen


def observation(x, agility):
    """Current kinematics and the attacker's own previous command only."""
    attacker = x[:, -1]
    c, s = attacker[:, 2].cos(), attacker[:, 2].sin()

    def rotate(v):
        return torch.stack((c[:, None]*v[..., 0]+s[:, None]*v[..., 1],
                            -s[:, None]*v[..., 0]+c[:, None]*v[..., 1]), -1)

    position = rotate(x[:, :-1, :2]-attacker[:, None, :2])/60
    velocity = rotate(x[:, :-1, 3:5]-attacker[:, None, 3:5])/5
    heading = x[:, :-1, 2]-attacker[:, None, 2]
    defenders = torch.cat((position, velocity, heading.cos()[..., None],
                           heading.sin()[..., None], x[:, :-1, 5:6]/2), -1)
    own = torch.cat((rotate(-attacker[:, None, :2]).squeeze(1)/60,
                     rotate(attacker[:, None, 3:5]).squeeze(1)/5,
                     attacker[:, 5:6]/2, (attacker[:, 6:8]/agility[:, None]-250)/750,
                     attacker[:, 8:9]), -1)
    return torch.cat((own, defenders.flatten(1)), -1)


def mlp(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, outputs))


class Attacker(nn.Module):
    def __init__(self, defenders=3, width=128, delta=300.):
        super().__init__()
        self.defenders, self.width, self.delta = defenders, width, delta
        dimension = 8+7*defenders
        self.actor = mlp(dimension, 4, width)
        self.q1, self.q2 = mlp(dimension+2, 1, width), mlp(dimension+2, 1, width)

    def sample(self, state, deterministic=False):
        mean, log_std = self.actor(state).chunk(2, -1)
        distribution = Normal(mean, log_std.clamp(-20, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        latent = raw.tanh()
        logp = (distribution.log_prob(raw)-2*(math.log(2)-raw-F.softplus(-2*raw))).sum(-1)
        return latent, logp

    def command(self, x, latent, agility):
        return execute(250+750*latent, x[:, -1, 6:8]/agility[:, None], self.delta)*agility[:, None]

    @torch.no_grad()
    def select(self, x, agility, deterministic=True):
        state = observation(x, agility)
        latent, _ = self.sample(state, deterministic)
        return dict(action=self.command(x, latent, agility), latent=latent, state=state)

    def value(self, state, latent):
        inputs = torch.cat((state, latent), -1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)


class AttackerLearner:
    def __init__(self, model, lr=1e-4):
        self.model, self.target = model, copy.deepcopy(model)
        self.target.requires_grad_(False)
        self.q_parameters = list(model.q1.parameters())+list(model.q2.parameters())
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q_parameters, lr=lr)
        self.log_alpha = torch.zeros((), device=next(model.parameters()).device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.updates = 0

    def update(self, batch):
        state, latent, reward, next_state, done = (batch[key] for key in ('s', 'a', 'r', 'next_s', 'done'))
        self.updates += 1
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            target = reward.clone()
            continuing = ~done.bool()
            if bool(continuing.any()):
                next_a, logp = self.model.sample(next_state[continuing])
                q1, q2 = self.target.value(next_state[continuing], next_a)
                target[continuing] += .99*(torch.minimum(q1, q2)-alpha*logp)
        q1, q2 = self.model.value(state, latent)
        q_loss = F.mse_loss(q1, target)+F.mse_loss(q2, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_parameters, 100., error_if_nonfinite=True)
        self.q_optimizer.step()
        with frozen(self.model.q1), frozen(self.model.q2):
            sampled, logp = self.model.sample(state)
            q1, q2 = self.model.value(state, sampled)
            actor_loss = (alpha*logp-torch.minimum(q1, q2)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), 10., error_if_nonfinite=True)
            self.actor_optimizer.step()
        alpha_loss = -(self.log_alpha*(logp.detach()-2.)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.model.parameters()):
                target_parameter.lerp_(parameter, .005)
        return dict(q_loss=q_loss.detach(), actor_loss=actor_loss.detach(), alpha=alpha)

    def payload(self):
        return dict(model=self.model.state_dict(), target=self.target.state_dict(),
                    defenders=self.model.defenders, width=self.model.width, delta=self.model.delta,
                    log_alpha=self.log_alpha.detach(), updates=self.updates,
                    optimizers=[self.actor_optimizer.state_dict(), self.q_optimizer.state_dict(),
                                self.alpha_optimizer.state_dict()])

    def restore(self, payload):
        self.model.load_state_dict(payload['model'])
        self.target.load_state_dict(payload['target'])
        self.updates = payload['updates']
        with torch.no_grad():
            self.log_alpha.copy_(payload['log_alpha'])
        for optimizer, saved in zip((self.actor_optimizer, self.q_optimizer, self.alpha_optimizer), payload['optimizers']):
            optimizer.load_state_dict(saved)


def attacker_reward(before, after, outcomes, target_radius):
    progress = (before[:, -1, :2].norm(dim=-1)-after[:, -1, :2].norm(dim=-1))/target_radius
    return torch.where(outcomes == 1, 100., torch.where((outcomes == 3) | (outcomes == 4), -100.,
                       torch.where(outcomes == 2, 0., progress)))
