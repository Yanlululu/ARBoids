"""Condition current configuration choice on the last executed command.

The existing candidate-interaction policy is retained as an exact warm start.
A zero-output task readout receives both its hidden interaction features and
candidate thrust changes relative to the observed previous command. Task
returns train this readout, the existing actor and the geometry encoder with
the same learned-policy reference and stopped-geometry replay. No fixed
switching penalty or action exclusion is introduced.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from citf.contracts import canonical_hash, file_hash
from experiments import reference_configuration_policy as reference

joint = reference.joint
BasePolicy = reference.ReferenceConfigurationPolicy
BaseController = joint.JointPolicyController
base_inputs = joint.policy.inputs
base_worker = reference.initialize_worker


@dataclass(frozen=True)
class ActuatorContextConfig(reference.ReferencePolicyConfig):
    revision: str = 'CITF-executed-actuator-context-policy-v1'
    action_features: str = 'candidate-interaction features and current thrust changes from the last actually executed command'
    training_signal: str = 'ordinary task policy gradient with executed-command context, frozen learned-policy KL and stopped-geometry replay'


class ActuatorContextActor(nn.Module):
    def __init__(self, base, width):
        super().__init__()
        self.base = base
        self.context_width = 2 * width
        self.context_tokens = nn.Sequential(nn.Linear(width + 9, width), nn.GELU(), nn.Linear(width, width))
        self.context = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, features):
        hidden = self.base[:-1](features[..., :-self.context_width])
        return self.base[-1](hidden) + self.context(torch.cat((hidden, features[..., -self.context_width:]), -1))

    def encode_commands(self, cover, commands, valid):
        tokens = self.context_tokens(torch.cat((cover, commands), -1))
        mask = valid[:, None, :, None]
        mean = (tokens * mask).sum(-2) / valid.sum(-1)[:, None, None].clamp_min(1)
        maximum = tokens.masked_fill(~mask, -torch.inf).amax(-2)
        return torch.cat((mean, maximum), -1)


class ActuatorContextPolicy(BasePolicy):
    def __init__(self, cfg=ActuatorContextConfig()):
        super().__init__(cfg)
        self.actor = ActuatorContextActor(self.actor, cfg.width)
        self._capture_command_tokens = False
        self._command_tokens = None

    def _capture_cover(self, module, arguments):
        super()._capture_cover(module, arguments)
        if getattr(self, '_capture_command_tokens', False):
            self._command_tokens = arguments[0]

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'actor.0.weight' in state_dict:
            # The source is the declared original joint policy. Only the new
            # zero-output readout retains its initialization on this load.
            mapped = {('actor.base.' + key[6:] if key.startswith('actor.') else key): value
                      for key, value in state_dict.items()}
            mapped.update({key: value for key, value in self.state_dict().items()
                           if key.startswith('actor.context')})
            state_dict = mapped
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def features(self, batch, history_cache=None):
        self._capture_command_tokens = True
        try:
            features, baseline = super().features(batch, history_cache)
        finally:
            self._capture_command_tokens = False
        current = batch['candidate'][:, :, :-1, 6:8]
        roots, candidates, defenders, _ = current.shape
        cover = self._command_tokens.reshape(roots, candidates, defenders, self.cfg.width)
        self._command_tokens = None
        previous = batch['previous_command'][:, None].expand_as(current)
        available = batch['previous_command_valid'].to(current.dtype)
        valid = batch['entity_valid'][:, :-1]
        delta = (current - previous) * available[:, None, None, None]
        commands = torch.cat((current, previous, delta, delta.abs(),
            available[:, None, None, None].expand(roots, candidates, defenders, 1)), -1)
        context = self.actor.encode_commands(cover, commands, valid)
        return torch.cat((features, context), -1), baseline


def inputs(world, factors, plans):
    batch, gates, rotation = base_inputs(world, factors, plans)
    previous = getattr(world, '_citf_previous_command', None)
    if previous is None:
        previous = np.zeros((world.env.defender_num, 2), dtype=np.float32)
        available = False
    else:
        available = True
    batch['previous_command'] = torch.as_tensor(previous, dtype=torch.float32)[None]
    batch['previous_command_valid'] = torch.tensor([available], dtype=torch.bool)
    return batch, gates, rotation


class ActuatorContextController(BaseController):
    @torch.no_grad()
    def act(self, world):
        if world.env.Current_T == 0:
            world._citf_previous_command = None
        previous = getattr(world, '_citf_previous_command', None)
        force, trace = super().act(world)
        trace['previous_executed_command'] = previous.tolist() if previous is not None else None
        trace['command_context_role'] = 'observed previous normalized actuator command; first-step mask'
        world._citf_previous_command = ((force - 250.) / 750.).copy()
        return force, trace


def bind():
    reference.ReferencePolicyConfig = ActuatorContextConfig
    reference.ReferenceConfigurationPolicy = ActuatorContextPolicy
    joint.JointPolicyController = ActuatorContextController
    joint.policy.inputs = inputs
    joint.inputs = inputs


def initialize_worker(*args):
    bind()
    base_worker(*args)


def main():
    bind()
    reference.initialize_worker = initialize_worker
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        save = reference.atomic_save
        implementation = file_hash(__file__)
        recorded = False

        def save_checkpoint(value, path):
            nonlocal recorded
            protocol = dict(value['protocol'])
            protocol['executed_command_context'] = dict(
                inputs='previous normalized executed thrust and availability; no future actions',
                readout='each conditional defender token bound to current and previous command, signed and absolute changes, availability; learned token embedding before invariant pooling and zero-output task readout',
                initial='existing policy weights copied; additional readout has zero output',
                fixed_switching_penalty=False, all_current_configurations_available=True)
            protocol['runtime_code'] = canonical_hash(dict(core=protocol['runtime_code'],
                actuator_context_configuration_policy=implementation))
            if not recorded:
                (Path(path).parent / 'protocol.json').write_text(json.dumps(protocol, indent=2))
                recorded = True
            save(dict(value, protocol=protocol), path)

        reference.atomic_save = save_checkpoint
        reference.main()
        return
    import citf.evaluate as evaluation
    previous_identity = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous_identity(*args, **kwargs)
        result['controller']['executed_command_context'] = ActuatorContextConfig().action_features
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            actuator_context_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    reference.main()


if __name__ == '__main__':
    main()
