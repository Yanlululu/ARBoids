"""Deployment actors share the training architecture and forward computation."""
from pathlib import Path
import sys

import torch

TRAIN = Path(__file__).resolve().parents[1]/'train'
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from policy.networks import ActorSAC as TrainingActorSAC, ActorAdap as TrainingActorAdap


class CheckpointActor:
    def load(self, modelname):
        self.load_state_dict(torch.load(modelname, map_location='cpu', weights_only=True))


class ActorSAC(CheckpointActor, TrainingActorSAC):
    pass


class ActorAdap(CheckpointActor, TrainingActorAdap):
    pass
