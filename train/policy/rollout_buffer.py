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
        groups, group_keys = {}, []
        for episode, summary in zip(self.episodes, self.summaries):
            key = None
            if (config.curriculum_cost_baseline or config.curriculum_reward_baseline) and summary.get('curriculum', False):
                key = (summary['source_seed'], summary['source_step'], episode[0].get('gate_scale', 1.),
                       summary.get('gate_steps', 0))
                groups.setdefault(key, []).append(dict(summary, initial_task_return=sum(t['reward'] for t in episode)))
            group_keys.append(key)
        for summaries in groups.values():
            if len({row['seed'] for row in summaries}) != len(summaries):
                raise ValueError('Recovery baselines require independent, distinct continuation seeds.')
        rows = []
        for episode, summary, group_key in zip(self.episodes, self.summaries, group_keys):
            adv_r, ret_r = gae([t['reward'] for t in episode], [t['value_r'] for t in episode],
                               config.reward_gamma, config.gae_lambda)
            adv_c, ret_c = gae([t['cost'] for t in episode], [t['value_c'] for t in episode],
                               config.cost_gamma, config.cost_gae_lambda)
            if config.cost_monte_carlo_targets:
                # Keep the cost critic's physical event target independent of
                # the actor's GAE bias/variance tradeoff. Complete episodes and
                # cost_gamma=1 make this the undiscounted remaining event cost.
                ret_c = np.cumsum([t['cost'] for t in episode][::-1])[::-1].copy()
            prefix_rewards = np.concatenate(([0.], np.cumsum([t['reward'] for t in episode[:-1]])))
            own_task_return = sum(t['reward'] for t in episode)
            for i, transition in enumerate(episode):
                # Latent densities and endpoint masses use different measures.
                # A mode switch requires fresh collection even if actions match.
                mode = transition.get('gate_likelihood_censored', False)
                if mode not in (False, True) or mode != config.censored_gate_likelihood:
                    raise ValueError('Gate probability mode changed; collect fresh on-policy episodes.')
                row = dict(transition, advantage_r=adv_r[i], advantage_c=adv_c[i],
                           return_r=ret_r[i], return_c=ret_c[i], is_curriculum=bool(summary.get('curriculum', False)),
                           gate_likelihood_censored=bool(mode))
                if config.curriculum_cost_baseline:
                    group = groups.get(group_key, ())
                    weight = max(0., 1. - i / config.curriculum_baseline_steps) if len(group) > 1 else 0.
                    baseline = transition['value_c']
                    if weight:
                        # Exclude this trajectory's own outcome. The other
                        # independent continuations estimate the start state's
                        # cost; decay toward the state critic as paths diverge.
                        others = (sum(s['collision'] for s in group) - summary['collision']) / (len(group) - 1)
                        baseline = weight * others + (1. - weight) * baseline
                        row['advantage_c'] = ret_c[i] - baseline
                    row.update(group_cost_weight=weight, group_cost_baseline=baseline)
                if config.curriculum_reward_baseline:
                    group = groups.get(group_key, ())
                    weight = max(0., 1. - i / config.curriculum_baseline_steps) if len(group) > 1 else 0.
                    baseline = transition['value_r']
                    if weight:
                        # Other futures estimate the start return. Subtract only
                        # this path's already observed rewards, never its future.
                        others = (sum(s['initial_task_return'] for s in group) - own_task_return) / (len(group)-1)
                        baseline = weight*(others-prefix_rewards[i]) + (1.-weight)*baseline
                        row['advantage_r'] = ret_r[i] - baseline
                    row.update(group_reward_weight=weight, group_reward_baseline=baseline)
                if config.minimize_collision:
                    # Terminal events are actual complete-task outcomes. Their
                    # undiscounted remaining return is constant until termination.
                    for suffix, name in (('s', 'success'), ('b', 'breach')):
                        row['return_' + suffix] = float(summary[name])
                        baseline = transition['value_' + suffix]
                        group = groups.get(group_key, ())
                        weight = (max(0., 1.-i/config.curriculum_baseline_steps)
                                  if config.curriculum_cost_baseline and len(group)>1 else 0.)
                        if weight:
                            others = (sum(s[name] for s in group)-summary[name])/(len(group)-1)
                            baseline = weight*others+(1.-weight)*baseline
                        row['advantage_' + suffix] = row['return_' + suffix]-baseline
                        if suffix == 'b':
                            row['group_breach_baseline'] = baseline
                elif config.breach_constraint:
                    # Actual complete-episode breach, including an event that
                    # coincides with a collision. No predicted risk is a label.
                    row['return_b'] = float(summary['breach'])
                    baseline = transition['value_b']
                    group = groups.get(group_key, ())
                    weight = (max(0., 1.-i/config.curriculum_baseline_steps)
                              if config.curriculum_cost_baseline and len(group)>1 else 0.)
                    if weight:
                        others = (sum(s['breach'] for s in group)-summary['breach'])/(len(group)-1)
                        baseline = weight*others+(1.-weight)*baseline
                    row['advantage_b'] = row['return_b']-baseline
                    row['group_breach_baseline'] = baseline
                rows.append(row)
        batch = {}
        for key in rows[0]:
            dtype = torch.bool if key in ('mask', 'terminated', 'is_curriculum') else torch.float32
            batch[key] = torch.as_tensor(np.stack([row[key] for row in rows]), dtype=dtype, device=device)
        return batch

    @property
    def steps(self):
        return sum(len(episode) for episode in self.episodes)

    @property
    def collision_rate(self):
        return self.ordinary_event_rate('collision')

    @property
    def success_rate(self):
        return self.ordinary_event_rate('success')

    @property
    def breach_rate(self):
        return self.ordinary_event_rate('breach')

    def ordinary_event_rate(self, name):
        if not self.summaries:
            raise ValueError('Collision rates require complete episodes.')
        # Failure-conditioned starts deliberately oversample risk. Only full
        # tasks from the original reset distribution estimate the dual budget.
        ordinary = [row for row in self.summaries if not row.get('curriculum', False)]
        if not ordinary:
            raise ValueError('Curriculum starts alone cannot estimate the task collision budget.')
        return float(np.mean([row[name] for row in ordinary]))
