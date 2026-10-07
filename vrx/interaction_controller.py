"""The IA-CRRL training policy and common CBF, without an online rollout."""
from pathlib import Path
import sys
import numpy as np

TRAIN = Path(__file__).resolve().parents[1] / 'train'
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))
import study_runtime
from interaction_rollout import DeploymentPolicy, safety_controller


class InteractionController(DeploymentPolicy):
    def __init__(self, checkpoint, device='cpu', defenders=3):
        super().__init__(checkpoint, device)
        # Compile the fixed-size CBF before the simulator begins the trial.
        angles = np.arange(defenders) * 2 * np.pi / defenders
        state = np.zeros((defenders, 6))
        state[:, :2] = 20 * np.column_stack((np.cos(angles), np.sin(angles)))
        action = np.zeros((defenders, 3))
        action[:, 2] = .5
        safety_controller(defenders).control(state, action, np.zeros((defenders, 2)))
