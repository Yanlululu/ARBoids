"""Matched online learning with balanced receiver and action-relation inputs.

The original readout normalizes the concatenated receiver and pair code once.
Their observed scale disparity grows during training. Normalize each existing
stream before the original affine readout normalization; add no parameters,
features, rewards, candidate chains or auxiliary learning targets.
"""
from torch import nn
from torch.nn import functional as F
import torch

from train_shared_team_interaction import main
from unified.networks import CITF


class BalancedInteractionNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, receiver_width, eps=1e-5):
        super().__init__(normalized_shape, eps=eps)
        self.receiver_width = receiver_width

    def forward(self, value):
        receiver, relation = value[..., :self.receiver_width], value[..., self.receiver_width:]
        receiver = F.layer_norm(receiver, (receiver.shape[-1],), eps=self.eps)
        relation = F.layer_norm(relation, (relation.shape[-1],), eps=self.eps)
        return super().forward(torch.cat((receiver, relation), -1))


class BalancedInteractionCITF(CITF):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        for critic in (self.q1, self.q2):
            if critic.action_interaction:
                original = critic.interaction_readout[0]
                critic.interaction_readout[0] = BalancedInteractionNorm(
                    original.normalized_shape, self.cfg.width, original.eps)


if __name__ == '__main__':
    main(candidate_type=BalancedInteractionCITF, candidate_name='balanced_interaction',
         candidate_definition='parameter-free separate normalization of the existing Transformer receiver and pair-action relation before their original affine LayerNorm; original tanh endpoint, r25 TD loss, local/global aggregation and generator objectives retained')
