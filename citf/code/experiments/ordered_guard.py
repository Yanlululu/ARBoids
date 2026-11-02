"""Expose the observed traversal order of the released APF response rule.

The archived attacker reduces its obstacle influence radius as it traverses
the defender list. Consequently its response is not invariant to arbitrary
list permutations. This predictor adds only that known list position, never
the hidden agility or future actions, to the candidate-interaction features.
"""
from dataclasses import dataclass
import sys

import torch
from torch import nn

from citf.contracts import canonical_hash, file_hash
from experiments import guard_focus as focus
from experiments import guard_prediction as guard
from experiments import guard_envelope


@dataclass(frozen=True)
class OrderedConfig(focus.FocusConfig):
    revision: str = 'CITF-ordered-stopped-guard-envelope-v1'
    observed_order: str = 'released-APF-defender-list-position'


class OrderedPredictor(guard.GuardPredictor):
    def __init__(self, cfg=OrderedConfig()):
        super().__init__(cfg)
        self.order_embed = nn.Linear(3, cfg.width, bias=False)
        nn.init.zeros_(self.order_embed.weight)

    def encode_history(self, b):
        history = super().encode_history(b)
        entities = history.shape[1]
        index = torch.arange(entities, dtype=history.dtype, device=history.device)
        defender = index < entities - 1
        code = torch.stack((index / max(1, entities - 2), index / 8,
                            defender.to(history.dtype)), -1) * defender[:, None]
        return history + self.order_embed(code)[None] * b['entity_valid'][..., None]


def main():
    original = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original(path), 'ordered_guard': file_hash(__file__)})
    guard.GuardPredictor = OrderedPredictor
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        focus.FocusConfig = OrderedConfig
        focus.main()
    else:
        import citf.evaluate as evaluation
        identity_original = evaluation.runtime_identity

        def identity(*positional, **kwargs):
            result = identity_original(*positional, **kwargs)
            result['controller']['observed_order'] = OrderedConfig().observed_order
            result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'ordered_guard': file_hash(__file__)})
            return result

        evaluation.runtime_identity = identity
        guard.GuardConfig = OrderedConfig
        guard_envelope.main()


if __name__ == '__main__':
    main()
