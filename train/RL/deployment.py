"""Actor-only inference and versioned checkpoint loading, shared with VRX."""
import torch
from .channel_networks import ChannelActor
from .observations import FEATURE_VERSION, tensor_frame

CHECKPOINT_FORMAT = "arboids-channel-mappo-v1"


class DeployedPolicy:
    def __init__(self, actor, device="cpu", config=None):
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        self.config = config or {}

    @classmethod
    def load(cls, path, device="cpu"):
        data = torch.load(path, map_location=device, weights_only=True)
        if data.get("format") != CHECKPOINT_FORMAT or data.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Checkpoint is not a compatible ChannelMAPPO model")
        actor = ChannelActor(**data["actor_config"])
        actor.load_state_dict(data["actor"])
        if not all(torch.isfinite(p).all().item() for p in actor.parameters()):
            raise FloatingPointError("Checkpoint contains non-finite actor parameters")
        return cls(actor, device, data.get("config", {}))

    @torch.no_grad()
    def act(self, frame, deterministic=True):
        decision = self.actor.act(tensor_frame(frame, self.device), deterministic)
        return {key: value[0].cpu().numpy() for key, value in decision.items()}
