"""Refine a learned current policy with task queries over its interactions.

The existing action head and geometric encoder form an exact warm start.
Two learned queries read every candidate-conditioned defender and pair token;
a zero-output additional head learns which joint interactions matter for the
current action. The matched reference-policy objective and budget are kept.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch
from torch import nn

from citf.contracts import canonical_hash, file_hash
from experiments import reference_configuration_policy as reference

joint = reference.joint
BasePolicy = reference.ReferenceConfigurationPolicy
base_worker = reference.initialize_worker


@dataclass(frozen=True)
class ReadoutPolicyConfig(reference.ReferencePolicyConfig):
    revision: str = 'CITF-warm-query-readout-current-configuration-policy-v2'
    action_features: str = 'existing action features plus two task queries over every conditional defender and pair token'
    training_signal: str = 'ordinary task policy gradient with learned task-query readout, frozen learned-policy KL and stopped-geometry replay'


class TaskReadoutActor(nn.Module):
    def __init__(self, base, width, heads):
        super().__init__()
        self.base = base
        self.context_width = 2 * width
        self.query = nn.Parameter(torch.randn(2, width) * .02)
        self.role = nn.Parameter(torch.randn(2, width) * .02)
        self.norm = nn.LayerNorm(width)
        self.readout = nn.MultiheadAttention(width, heads, batch_first=True)
        self.context = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def encode(self, cover, pairs, cover_valid, pair_valid):
        roots, candidates, _, width = cover.shape
        tokens = torch.cat((cover + self.role[0], pairs + self.role[1]), -2)
        valid = torch.cat((cover_valid, pair_valid), -1)
        tokens = self.norm(tokens.reshape(roots * candidates, -1, width))
        mask = valid[:, None].expand(roots, candidates, -1).reshape(roots * candidates, -1)
        query = self.query[None].expand(roots * candidates, -1, -1)
        value = self.readout(query, tokens, tokens, key_padding_mask=~mask, need_weights=False)[0]
        return value.reshape(roots, candidates, self.context_width)

    def forward(self, features):
        hidden = self.base[:-1](features[..., :-self.context_width])
        return self.base[-1](hidden) + self.context(torch.cat((hidden, features[..., -self.context_width:]), -1))


class ReadoutConfigurationPolicy(BasePolicy):
    def __init__(self, cfg=ReadoutPolicyConfig()):
        super().__init__(cfg)
        self.actor = TaskReadoutActor(self.actor, cfg.width, cfg.heads)
        self._capture_query_tokens = False
        self._query_cover, self._query_pairs, self._query_valid = None, None, None
        self.encoder.register_forward_hook(self._capture_query_valid)

    def _capture_cover(self, module, arguments):
        super()._capture_cover(module, arguments)
        if getattr(self, '_capture_query_tokens', False):
            self._query_cover = arguments[0]

    def _capture_pairs(self, module, arguments):
        super()._capture_pairs(module, arguments)
        if getattr(self, '_capture_query_tokens', False):
            self._query_pairs = arguments[0]

    def _capture_query_valid(self, module, arguments, output):
        if self._capture_query_tokens:
            self._query_valid = output['all_pair_valid']

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if 'actor.0.weight' in state_dict:
            mapped = {('actor.base.' + key[6:] if key.startswith('actor.') else key): value
                      for key, value in state_dict.items()}
            mapped.update({key: value for key, value in self.state_dict().items()
                           if key.startswith('actor.') and not key.startswith('actor.base.')})
            state_dict = mapped
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def features(self, batch, history_cache=None):
        self._capture_query_tokens = True
        try:
            features, baseline = super().features(batch, history_cache)
        finally:
            self._capture_query_tokens = False
        roots, candidates = batch['candidate_valid'].shape
        cover = self._query_cover.reshape(roots, candidates, -1, self.cfg.width)
        pairs = self._query_pairs.reshape(roots, candidates, -1, self.cfg.width)
        readout = self.actor.encode(cover, pairs, batch['entity_valid'][:, :-1], self._query_valid)
        self._query_cover, self._query_pairs, self._query_valid = None, None, None
        return torch.cat((features, readout), -1), baseline


def bind():
    reference.ReferencePolicyConfig = ReadoutPolicyConfig
    reference.ReferenceConfigurationPolicy = ReadoutConfigurationPolicy


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
            protocol['task_query_readout'] = dict(
                inputs=ReadoutPolicyConfig().action_features,
                initial='existing trained encoder and actor copied; additional task-query action head has zero output',
                objective='matched reference-policy task return and stopped-geometry replay',
                no_value_or_future_plan=True)
            protocol['runtime_code'] = canonical_hash(dict(core=protocol['runtime_code'],
                readout_configuration_policy=implementation))
            if not recorded:
                (Path(path).parent / 'protocol.json').write_text(json.dumps(protocol, indent=2))
                recorded = True
            save(dict(value, protocol=protocol), path)

        reference.atomic_save = save_checkpoint
        reference.main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['task_query_readout'] = ReadoutPolicyConfig().action_features
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            readout_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    reference.main()


if __name__ == '__main__':
    main()
