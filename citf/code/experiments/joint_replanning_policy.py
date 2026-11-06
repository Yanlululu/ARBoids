"""Observed-history Transformer with full authority over joint physical control.

The same decision function is used at every environment tick in collection,
danger-state continuation and deployment. There is no program bank, reserved
guard, source actor, online teacher or control projection in this policy.
"""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, LowRankMultivariateNormal, Normal

from citf.bridge import Controller
from citf.contracts import Protocol
from citf.model import Attention


@dataclass(frozen=True)
class JointConfig:
    revision: str = 'CITF-joint-continuous-replanning-v1'
    history: int = 8
    width: int = 96
    heads: int = 4
    layers: int = 2
    joint_context: bool = True
    exploration_rank: int = 3
    initial_std: float = .20

    def __post_init__(self):
        if min(self.history, self.width, self.heads, self.layers) < 1 or self.width % self.heads:
            raise ValueError('Positive dimensions and a head-divisible width are required')
        if self.exploration_rank < 0 or not math.isfinite(self.initial_std) or not .025 <= self.initial_std <= .8:
            raise ValueError('Invalid joint exploration configuration')


def observation(world, previous_force, history):
    """Only measured states, known remaining time and the previous applied force."""
    if world.done:
        raise ValueError('A terminal world has no next control decision')
    measured, positions, valid, _ = world.observation()
    if history > measured.shape[1]:
        raise ValueError('World history is shorter than the policy history')
    previous = np.asarray(previous_force, dtype=np.float32)
    if previous.shape != (world.env.defender_num, 2) or not np.isfinite(previous).all():
        raise ValueError('Previous physical command must contain both thrusters of every boat')
    return dict(history=measured[:, -history:], history_valid=valid[:, -history:],
                positions=positions, previous_force=torch.from_numpy(previous.copy()),
                remaining=torch.tensor([(world.protocol.time_limit-world.env.Current_T)/world.protocol.time_limit],
                                       dtype=torch.float32))


def batch_observations(rows, device):
    return {key: torch.stack([row[key] for row in rows]).to(device) for key in rows[0]}


class JointReplanningTransformer(nn.Module):
    def __init__(self, cfg=JointConfig()):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.history_embed = nn.Linear(8, w)
        self.time_embed = nn.Parameter(torch.randn(cfg.history, w)*.01)
        self.temporal = nn.ModuleList([Attention(w, cfg.heads) for _ in range(2)])
        # Attacker-relative position/velocity, time, role and previous thrust.
        self.control_state = nn.Sequential(nn.Linear(8, w), nn.GELU(), nn.Linear(w, w))
        self.joint = nn.ModuleList([Attention(w, cfg.heads, relational=True) for _ in range(cfg.layers)])
        self.mean = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 2))
        self.scale = nn.Linear(w, 2)
        self.factor = nn.Linear(w, 2*cfg.exploration_rank) if cfg.exploration_rank else None
        self.value = nn.Sequential(nn.LayerNorm(3*w+1), nn.Linear(3*w+1, w), nn.GELU(), nn.Linear(w, 2))
        nn.init.normal_(self.mean[-1].weight, std=.01)
        nn.init.constant_(self.mean[-1].bias, math.atanh(-1/3))
        nn.init.zeros_(self.scale.weight)
        nn.init.constant_(self.scale.bias, math.log(cfg.initial_std))
        if self.factor is not None:
            nn.init.normal_(self.factor.weight, std=.005)
            nn.init.zeros_(self.factor.bias)

    def forward(self, batch):
        h = batch['history']
        b, entities, length, features = h.shape
        if length != self.cfg.history or features != 8 or entities < 3:
            raise ValueError('Expected measured [batch, defenders+attacker, history, 8]')
        valid = batch['history_valid']
        if valid.shape != h.shape[:3] or not valid[..., -1].all():
            raise ValueError('Every real entity needs a current history observation')
        z = (self.history_embed(h)+self.time_embed).reshape(b*entities, length, -1)
        for layer in self.temporal:
            z = layer(z, valid.reshape(b*entities, length), causal=True)
        z = z[:, -1].reshape(b, entities, -1)
        position = batch['positions']
        velocity = h[:, :, -1, 2:4]*5
        previous = (batch['previous_force']-250)/750
        previous = torch.cat((previous, previous.new_zeros(b, 1, 2)), 1)
        role = h[:, :, -1, 7:8]
        state = torch.cat(((position-position[:, -1:])/30,
                           (velocity-velocity[:, -1:])/5,
                           batch['remaining'][:, None].expand(-1, entities, -1), role, previous), -1)
        z = z+self.control_state(state)
        delta = position[:, :, None]-position[:, None, :]
        dv = velocity[:, :, None]-velocity[:, None, :]
        distance = delta.norm(dim=-1)
        closing = -(delta*dv).sum(-1)/distance.clamp_min(.1)
        closest_time = (-(delta*dv).sum(-1)/dv.square().sum(-1).clamp_min(.01)).clamp(0, 2)
        closest = (delta+closest_time[..., None]*dv).norm(dim=-1)
        edge = torch.cat((delta/20, (distance/20)[..., None], (closing/5)[..., None],
                          (closest/10)[..., None]), -1)
        available = torch.ones(b, entities, dtype=torch.bool, device=z.device)
        if self.cfg.joint_context:
            for layer in self.joint:
                z = layer(z, available, edge)
        else:
            # Identical measured inputs and per-token transforms; remove only
            # communication between entities, including off-diagonal geometry.
            diagonal = torch.arange(entities, device=z.device)
            own_edge = edge[:, diagonal, diagonal].reshape(b*entities, 1, 1, 5)
            for layer in self.joint:
                z = layer(z.reshape(b*entities, 1, -1), available.reshape(b*entities, 1),
                          own_edge).reshape(b, entities, -1)
        boats = z[:, :-1]
        mean = self.mean(boats).flatten(1)
        std = self.scale(boats).clamp(math.log(.025), math.log(.8)).exp().flatten(1)
        if self.factor is None:
            distribution = Independent(Normal(mean, std), 1)
        else:
            factor = .35*self.factor(boats).tanh().reshape(b, -1, self.cfg.exploration_rank)
            distribution = LowRankMultivariateNormal(mean, factor, std.square())
        pooled = torch.cat((boats.mean(1), boats.amax(1), z[:, -1], batch['remaining']), -1)
        value = self.value(pooled)
        return dict(distribution=distribution, mean=mean,
                    value=value[:, 0], cost_value=value[:, 1].sigmoid())

    @staticmethod
    def force(latent):
        # A bijection into the entire actuator range. No clipping of a teacher
        # residual, which would merge distinct sampled actions at saturation.
        return 250+750*latent.tanh().reshape(len(latent), -1, 2)

    def validate_runtime(self, source, world):
        if world is None:
            return
        if self.cfg.history > world.protocol.history_steps:
            raise ValueError('Policy history does not fit the deployment buffer')
        trained = getattr(self, 'physical_protocol', None)
        if trained is not None:
            before, after = asdict(Protocol(**trained)), asdict(world.protocol)
            ignored = {'history_steps', 'prediction_steps', 'risk_weight', 'risk_clearance', 'prior_scale'}
            if any(before[key] != value for key, value in after.items() if key not in ignored):
                raise ValueError('Joint policy training and deployment physical task differ')


@torch.no_grad()
def decide(model, world, previous_force, *, stochastic=False, device='cpu'):
    """Single collection/deployment path, replanned at every real control tick."""
    observed = observation(world, previous_force, model.cfg.history)
    output = model(batch_observations([observed], device))
    distribution = output['distribution']
    latent = distribution.sample() if stochastic else output['mean']
    force = model.force(latent)[0].cpu().numpy()
    # The tanh transform has no learned parameters. Its Jacobian cancels in
    # the PPO ratio, so retain the exact joint density of the pre-tanh action.
    record = dict(observation=observed, latent=latent[0].cpu(),
                  logp=float(distribution.log_prob(latent)[0]),
                  value=float(output['value'][0]), cost_value=float(output['cost_value'][0]))
    return force, record


def load_policy(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = JointReplanningTransformer(JointConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['state'])
    model.physical_protocol = payload['physical_protocol']
    return model.eval()


class JointReplanningController(Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous = None
        self.last_time = None

    def act(self, world):
        start = time.perf_counter()
        now = world.env.Current_T
        if self.previous is None or self.last_time is not None and now <= self.last_time:
            self.previous = np.zeros((world.env.defender_num, 2), np.float32)
        force, record = decide(self.model, world, self.previous, device=self.device)
        self.previous, self.last_time = force.copy(), now
        return force, dict(revision=self.model.cfg.revision, physical_thrust=force.tolist(),
            joint_latent=record['latent'].tolist(), evaluated_unique=1, forward_batches=1,
            replan_seconds=world.protocol.control_dt, teacher_used_online=False,
            source_used_online=False, reserved_guard=False, projection_applied=False,
            compute_seconds=time.perf_counter()-start)


def main():
    import citf.evaluate as evaluation
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = original(*args, **kwargs)
        checkpoint = args[2] if len(args) > 2 else kwargs['checkpoint']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        result.update(method='CITF-joint-continuous-replanning', revision=payload['config']['revision'])
        result['controller'] = dict(configuration=payload['config'],
            action='Transformer joint continuous left/right thrust for every boat,[-500,1000]N',
            control='same decide() at every real control tick in training and deployment',
            search='direct learned joint distribution; no fixed candidate bank',
            teacher_used_online=False, source_used_online=False, reserved_guard=False,
            projection_applied=False, training_update=payload.get('update'),
            training_environment_steps=payload.get('environment_steps'),
            learning='fresh complete rollouts and danger-state continuations under the current policy',
            warmstart_environment_steps=payload.get('warmstart_environment_steps', 0),
            supported_selections=['joint'])
        return result

    evaluation.SOURCE_REQUIRED = False
    evaluation.runtime_identity = identity
    evaluation.load_behavior = lambda path, source, world, device: load_policy(path, device)
    evaluation.Controller = JointReplanningController
    evaluation.main()


if __name__ == '__main__':
    main()
