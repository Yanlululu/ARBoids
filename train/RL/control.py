"""Physical actuator convention shared by training, search and deployment."""
import torch

MIN_THRUST = -500.0
MAX_THRUST = 1000.0
THRUST_CENTER = 250.0
THRUST_SCALE = 750.0


def to_thrust(action):
    return action * THRUST_SCALE + THRUST_CENTER


def to_action(thrust):
    return (thrust - THRUST_CENTER) / THRUST_SCALE


def to_channels(thrust):
    return torch.stack(((thrust[..., 0] + thrust[..., 1]) / 2,
                        (thrust[..., 1] - thrust[..., 0]) / 2), dim=-1)


def from_channels(channels):
    return torch.stack((channels[..., 0] - channels[..., 1],
                        channels[..., 0] + channels[..., 1]), dim=-1)


def candidate_features(boids, learned_action):
    # Applying the channel transform to normalized thrust gives
    # [(c - 250) / 750, d / 750], including the asymmetric thrust offset.
    return torch.cat((to_channels(to_action(boids)),
                      to_channels(learned_action)), dim=-1)


def blend_thrust(boids, learned, gates, mode="channels"):
    """Unprojected command, for projection diagnostics only."""
    if mode == "channels":
        b, l = to_channels(boids), to_channels(learned)
        raw = from_channels(b + gates * (l - b))
    elif mode in ("scalar", "thrusters"):
        raw = boids + gates * (learned - boids)
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
    return raw


def fuse_thrust(boids, learned, gates, mode="channels"):
    """Candidates are in N; output follows the training thruster order.

    Clipping the reconstructed thrusters is exactly Euclidean projection
    in (common, differential) coordinates, since A.T @ A = 2 I.
    """
    return blend_thrust(boids, learned, gates, mode).clamp(MIN_THRUST, MAX_THRUST)
