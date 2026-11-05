"""Joint continuous control with an observed persistent residual state.

The previous executed latent is part of both actor and critic input. Correlated
exploration therefore uses an explicit Markov state, with the same conditional
policy during collection, optimization and deterministic deployment.
"""
from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from citf.contracts import canonical_hash, file_hash
from experiments.continuous_actor_critic import (
    ContinuousConfig, ContinuousActorCritic, ContinuousController, observed_history)


@dataclass(frozen=True)
class EpisodeConfig(ContinuousConfig):
    revision: str = 'CITF-complete-episode-persistent-actor-critic-v1'
    correlation: float = .8

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.correlation < 1:
            raise ValueError('Persistent-noise correlation must lie in [0, 1)')


class EpisodeActorCritic(ContinuousActorCritic):
    def __init__(self, cfg=EpisodeConfig()):
        super().__init__(cfg)
        self.previous = nn.Linear(2, cfg.width, bias=False)
        nn.init.zeros_(self.previous.weight)

    def forward(self, history, remaining, reference, previous):
        position = history[:, -1, :, :2]
        radius = position.norm(dim=-1)
        previous_nodes = torch.cat((previous, previous.new_zeros((len(previous), 1, 2))), 1)
        encoded = self.encoder(history, remaining) + self.previous(previous_nodes)
        n = position.shape[1] - 1
        i, j = torch.triu_indices(n, n, 1, device=position.device)
        geometry = torch.stack((
            (position[:, :-1] - position[:, -1:]).norm(dim=-1).amin(1) / 60,
            (position[:, i] - position[:, j]).norm(dim=-1).amin(1) / 20,
            (radius[:, -1] - radius[:, :-1].amin(1)) / 30), -1)
        pooled = torch.cat((encoded[:, :-1].mean(1), encoded[:, :-1].amax(1),
            encoded[:, -1], geometry), -1)
        value = 1.05 * self.critic(pooled).squeeze(-1).tanh()
        command = torch.cat(((reference - 250) / 750,
            reference.new_zeros((len(reference), 1, 2))), 1)
        angle = torch.atan2(-position[:, -1, 1], -position[:, -1, 0])
        c, s = angle.cos(), angle.sin()
        canonical = torch.stack((c[:, None] * position[..., 0] + s[:, None] * position[..., 1],
            -s[:, None] * position[..., 0] + c[:, None] * position[..., 1]), -1)
        delta = canonical[:, :, None] - canonical[:, None, :]
        pair = delta.norm(dim=-1)
        relative = torch.cat((delta / 20, (pair / 20)[..., None], (1 / (pair + 1))[..., None],
            ((radius[:, :, None] - radius[:, None, :]) / 20)[..., None]), -1)
        nodes = self.action_context(encoded + self.command(command), relative, self.cfg.context)
        desired = self.actor(nodes[:, :-1])
        mean = self.cfg.correlation * previous + (1 - self.cfg.correlation) * desired
        std = self.log_std.clamp(-5., -1.).exp() * math.sqrt(1 - self.cfg.correlation ** 2)
        return dict(mean=mean, desired=desired, std=std.expand_as(mean), value=value)

    def distribution(self, history, remaining, reference, previous):
        output = self(history, remaining, reference, previous)
        return Normal(output['mean'], output['std']), output['value']


def load_policy(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = EpisodeActorCritic(EpisodeConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['state'])
    model.training_protocol=payload.get('physical_protocol',payload.get('protocol',{}).get('physical_protocol'))
    return model.eval()


class EpisodeController(ContinuousController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous = None
        self.last_time = None

    @torch.no_grad()
    def act(self, world):
        started = time.perf_counter()
        reference, trace = self.teacher.act(world)
        now = world.env.Current_T
        if self.previous is None or self.last_time is not None and now <= self.last_time:
            self.previous = np.zeros_like(reference, dtype=np.float32)
        history = torch.as_tensor(observed_history(world,self.model.cfg.history), device=self.device)[None]
        remaining = torch.tensor([world.protocol.time_limit - now], device=self.device)
        reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=self.device)[None]
        previous = torch.as_tensor(self.previous, device=self.device)[None]
        output = self.model(history, remaining, reference_tensor, previous)
        force = self.model.force(reference_tensor, output['mean'])[0].cpu().numpy()
        trace.update(revision=self.model.cfg.revision, physical_thrust=force.tolist(),
            reference_physical_thrust=reference.tolist(), previous_latent=self.previous.tolist(),
            latent_residual=output['mean'][0].cpu().tolist(), physical_residual=(force-reference).tolist(),
            control_period=world.protocol.control_dt, correlation=self.model.cfg.correlation,
            compute_seconds=time.perf_counter()-started)
        trace['forward_batches'] += 1
        self.previous = output['mean'][0].cpu().numpy()
        self.last_time = now
        return force, trace


def main():
    import citf.evaluate as evaluation
    import experiments.continuous_actor_critic as base
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = original(*args, **kwargs)
        checkpoint = args[2] if len(args) > 2 else kwargs.get('checkpoint')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = payload['config']
        result.update(method='CITF-persistent-complete-episode-actor-critic-development', revision=cfg['revision'])
        result['controller'].update(action='reference+1500*tanh(joint latent),common[-500,1000]N',
            control='every0.2s; previous executed latent is an observed policy state',
            correlation=cfg['correlation'], learning='actual complete-episode MonteCarlo returns and PPO',
            reference_mode=cfg.get('reference_mode','teacher'),
            teacher_used_online=cfg.get('reference_mode','teacher')=='teacher',
            source_used_online=cfg.get('reference_mode','teacher')!='direct',
            teacher_warmstart_environment_steps=payload.get('warmstart_environment_steps',0),
            teacher_checkpoint_sha256=(file_hash(cfg['teacher_checkpoint']) if cfg.get('reference_mode','teacher')=='teacher'
                                       else payload.get('teacher_sha256')),
            encoder_initialization_sha256=payload.get('encoder_initialization_sha256'),
            training_update=payload.get('update'), training_environment_steps=payload.get('environment_steps'),
            reference_has_learning=cfg.get('reference_mode','teacher')!='direct', expanded_control_domain=True)
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            policy=file_hash(__file__), base=file_hash(base.__file__)))
        return result

    evaluation.runtime_identity = identity
    evaluation.load_behavior = lambda checkpoint, source, world, device: load_policy(checkpoint, device)
    evaluation.Controller = EpisodeController
    evaluation.main()


if __name__ == '__main__':
    main()
