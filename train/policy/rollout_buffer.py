"""Complete episodes collected under one frozen behavior policy."""
import numpy as np
import torch


def gae(rewards, values, gamma, gae_lambda):
    """A complete finite-horizon episode always has terminal bootstrap zero."""
    rewards, values = np.asarray(rewards), np.asarray(values)
    if rewards.shape != values.shape or not len(rewards):
        raise ValueError('A nonempty complete episode is required.')
    advantages = np.zeros(len(rewards), dtype=np.float32)
    following_value, following_advantage = 0., 0.
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * following_value - values[t]
        following_advantage = delta + gamma * gae_lambda * following_advantage
        advantages[t] = following_advantage
        following_value = values[t]
    return advantages, advantages + values


class RolloutBuffer:
    def __init__(self):
        self.episodes = []
        self.summaries = []

    def add_episode(self, transitions, summary):
        if not transitions or not transitions[-1]['terminated']:
            raise ValueError('Only complete episodes may enter an update.')
        if any(t['terminated'] for t in transitions[:-1]):
            raise ValueError('A rollout episode crosses a terminal boundary.')
        if not np.isclose(sum(t['cost'] for t in transitions), float(summary['collision'])):
            raise ValueError('Episode cost must equal its binary collision event.')
        self.episodes.append(transitions)
        self.summaries.append(dict(summary))

    def tensors(self, config, device):
        if not self.episodes:
            raise ValueError('Cannot update with an empty rollout.')
        rows = []
        for episode in self.episodes:
            adv_r, ret_r = gae([t['reward'] for t in episode], [t['value_r'] for t in episode],
                               config.reward_gamma, config.gae_lambda)
            adv_c, ret_c = gae([t['cost'] for t in episode], [t['value_c'] for t in episode],
                               config.cost_gamma, config.gae_lambda)
            for i, transition in enumerate(episode):
                rows.append(dict(transition, advantage_r=adv_r[i], advantage_c=adv_c[i],
                                 return_r=ret_r[i], return_c=ret_c[i]))
        batch = {}
        for key in rows[0]:
            dtype = torch.bool if key in ('mask', 'terminated') else torch.float32
            batch[key] = torch.as_tensor(np.stack([row[key] for row in rows]), dtype=dtype, device=device)
        return batch

    @property
    def steps(self):
        return sum(len(episode) for episode in self.episodes)

    @property
    def collision_rate(self):
        if not self.summaries:
            raise ValueError('Collision rates require complete episodes.')
        return float(np.mean([row['collision'] for row in self.summaries]))
