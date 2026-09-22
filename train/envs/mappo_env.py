"""On-policy interface; the legacy SAC and attacker step API stays intact."""
from dataclasses import dataclass
import numpy as np

from envs.TADgame import TADEnv


@dataclass
class Transition:
    observation: np.ndarray
    reward: float
    cost: float
    terminated: bool
    outcome: int
    events: dict
    attacker_observation: np.ndarray


class MAPPOEnv:
    def __init__(self, env=None, **kwargs):
        self.env = env if env is not None else TADEnv(**kwargs)
        if self.env.protocol != 'paper-parameters-v1' or self.env.LearningSide != 'Def':
            raise ValueError('Predictive MAPPO requires the paper-parameters-v1 defender task.')
        if not self.env.boid_state or self.env.defender_num < 2:
            raise ValueError('Require Boids observations and at least two defenders.')
        self.global_dim = 7 * (self.env.defender_num + 1) + 2
        self.collision_seen = False
        self.finished = True

    def reset(self, agility=2., noisy_agility=False):
        observation, _ = self.env.reset(agility, noisy_agility)
        self.collision_seen = False
        self.finished = False
        return observation

    def step(self, action, attacker_action=None):
        if self.finished:
            raise RuntimeError('Reset before stepping a completed episode.')
        action = np.asarray(action)
        if action.shape != (self.env.defender_num, 3) or not np.isfinite(action).all():
            raise ValueError('Expected finite [defenders, 3] action.')
        if np.any(np.abs(action[:, :2]) > 1.) or np.any((action[:, 2] < 0.) | (action[:, 2] > 1.)):
            raise ValueError('Proposal or gate is outside its bounds.')
        obs, _, done, att_obs = self.env.step(action, 'AdaRes', attacker_action)
        main, formation, _ = self.env.paper_reward_components()
        events = self.env.physical_events()
        cost = float(events['collision'] and not self.collision_seen)
        self.collision_seen |= events['collision']
        self.finished = bool(done)
        # done=4 is the actual finite task deadline, not a rollout truncation.
        return Transition(obs, float((main + formation).mean()), cost,
                          bool(done), int(done), events, att_obs)
