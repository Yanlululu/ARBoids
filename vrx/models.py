"""Deployment actors share the training architecture and forward computation."""
from pathlib import Path
import sys

import torch

TRAIN = Path(__file__).resolve().parents[1]/'train'
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from policy.networks import ActorSAC as TrainingActorSAC, ActorAdap as TrainingActorAdap


class CheckpointActor:
    @classmethod
    def from_checkpoint(cls, modelname, device='cpu'):
        weights = torch.load(modelname, map_location='cpu', weights_only=True)
        feature1_dim = weights['f1.weight'].shape[1]
        feature2_dim = weights['f2.weight'].shape[1]
        adaptive = issubclass(cls, TrainingActorAdap)
        if feature1_dim != 6 or feature2_dim not in (0, 8) or (adaptive and feature2_dim != 8):
            raise ValueError('Unsupported defender observation layout in checkpoint.')
        action_dim = weights['mean_layer.weight'].shape[0] + int(adaptive)
        if action_dim != 2 + int(adaptive):
            raise ValueError('Defender checkpoints require two thrust channels.')
        actor = cls(feature1_dim, feature2_dim, action_dim, weights['l1.weight'].shape[0]).to(device)
        actor.load_state_dict(weights)
        actor.eval()
        return actor

    def load(self, modelname):
        self.load_state_dict(torch.load(modelname, map_location='cpu', weights_only=True))


class ActorSAC(CheckpointActor, TrainingActorSAC):
    pass


class ActorAdap(CheckpointActor, TrainingActorAdap):
    pass
