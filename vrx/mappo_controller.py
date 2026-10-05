"""ROS-independent deployment adapter using the exact training actor/predictor."""
from pathlib import Path
import sys
import numpy as np

TRAIN = Path(__file__).resolve().parents[1] / 'train'
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))
from policy.mappo import PredictiveMAPPO
from policy.message_stress import MessageStress


class MAPPOController:
    def __init__(self, checkpoint, device='cpu', delay_steps=0, drop_probability=0., communication_seed=0):
        self.agent = PredictiveMAPPO.from_checkpoint(checkpoint, device)
        self.communication = MessageStress(delay_steps, drop_probability, communication_seed)
        self.last_action = None

    def control(self, observations, states, boids_thrust, timestamp):
        boids_action = (np.asarray(boids_thrust) - 250.) / 750.
        action = self.communication.act(self.agent, observations, states, boids_action, timestamp)
        self.last_action = action.copy()
        learned_thrust = 750. * action[:, :2] + 250.
        return action[:, 2:3] * learned_thrust + (1. - action[:, 2:3]) * boids_thrust


def synchronize_kinematics(positions, yaws, velocities, yaw_rates, timestamps):
    """Extrapolate feedback to one timestamp using measured constant velocity.

    The runner records skew/compute time; this is a local synchronous message
    emulation, not a network-delay or packet-loss solution.
    """
    arrays = [np.asarray(x) for x in (positions, yaws, velocities, yaw_rates, timestamps)]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('Cannot synchronize missing or non-finite vessel feedback.')
    positions, yaws, velocities, yaw_rates, timestamps = arrays
    stamp = float(timestamps.max())
    delta = stamp - timestamps
    states = np.column_stack((positions + velocities * delta[:, None],
                              yaws + yaw_rates * delta, velocities, yaw_rates))
    return states, stamp, float(delta.max())
