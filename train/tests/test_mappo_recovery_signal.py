"""Independent recovery baselines and transactional task-preserving PPO."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.mappo import PredictiveMAPPO
from train_mappo import collect_episodes, make_env
from parallel_mappo import breach_branch_cases


class RecoverySignalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(841)
        np.random.seed(841)
        self.config = dict(agent=dict(defender_num=3, boid_state=True, form_reward=True),
            environment=dict(protocol='paper-parameters-v1', total_time=.6), prediction=dict(horizon=.1),
            mappo=dict(hidden_dim=32, relation_dim=8, coordination_dim=4, ppo_epochs=2,
                       minibatch_size=64, cost_gae_lambda=1., cost_monte_carlo_targets=True,
                       curriculum_cost_baseline=True, curriculum_baseline_steps=2,
                       full_batch_guard=True, guard_task_surrogate=True, target_kl=.01, joint_ratio=True))

    def test_baseline_excludes_own_outcome_and_decays_without_changing_critic_targets(self):
        agent = PredictiveMAPPO(self.config)
        rollout = collect_episodes(agent, make_env(agent.config), 4, 2.)
        for i, (episode, summary) in enumerate(zip(rollout.episodes, rollout.summaries)):
            summary.update(collision=int(i in (2, 3)), seed=100000+i)
            if i:
                summary.update(curriculum=True, source_seed=110000, source_step=10, gate_steps=20)
            for row in episode:
                row.update(value_c=.1, cost=0.)
            episode[-1]['cost'] = float(summary['collision'])
        data = rollout.tensors(agent.settings, 'cpu')
        start = len(rollout.episodes[0])
        torch.testing.assert_close(data['group_cost_baseline'][start:start+3], torch.tensor([1., .55, .1]))
        torch.testing.assert_close(data['return_c'][start:start+3], torch.zeros(3))
        self.assertEqual(rollout.collision_rate, 0.)
        # Changing only this trajectory's own outcome must not change its baseline.
        rollout.summaries[1]['collision'] = 1
        rollout.episodes[1][-1]['cost'] = 1.
        changed = rollout.tensors(agent.settings, 'cpu')
        torch.testing.assert_close(changed['group_cost_baseline'][start:start+3],
                                   data['group_cost_baseline'][start:start+3])
        torch.testing.assert_close(changed['return_c'][start:start+3], torch.ones(3))

    def test_better_combined_loss_cannot_commit_a_task_regression(self):
        agent = PredictiveMAPPO(self.config)
        rollout = collect_episodes(agent, make_env(agent.config), 3, 2.)
        before = copy.deepcopy(agent.actor.state_dict())
        stats = [dict(surrogate=0., kl_joint=0., kl_limit=0., task_surrogate=0., cost_surrogate=0.),
                 dict(surrogate=1., kl_joint=.001, kl_limit=.001, task_surrogate=-.01, cost_surrogate=.1)]
        with patch.object(agent, '_policy_statistics', side_effect=stats):
            result = agent.update(rollout)
        self.assertEqual(result['actor_accepted_steps'], 0)
        self.assertEqual(result['task_surrogate_gain'], 0.)
        self.assertEqual(result['cost_surrogate_gain'], 0.)
        self.assertEqual(len(agent.actor_optimizer.state), 0)
        for name, value in before.items():
            torch.testing.assert_close(agent.actor.state_dict()[name], value, rtol=0, atol=0)

    def test_task_baseline_excludes_own_future_and_subtracts_only_observed_prefix(self):
        config = copy.deepcopy(self.config)
        config['mappo'].update(gae_lambda=1., curriculum_reward_baseline=True)
        agent = PredictiveMAPPO(config)
        rollout = collect_episodes(agent, make_env(agent.config), 4, 2.)
        for i, (episode, summary) in enumerate(zip(rollout.episodes, rollout.summaries)):
            summary.update(seed=100000+i)
            if i:
                summary.update(curriculum=True, source_seed=110000, source_step=10, gate_steps=20)
            for row, reward in zip(episode, ([1.,1.,1.], [1.,1.,2.], [2.,2.,2.], [4.,3.,1.])[i]):
                row.update(reward=reward, value_r=.1)
        start = len(rollout.episodes[0])
        data = rollout.tensors(agent.settings, 'cpu')
        torch.testing.assert_close(data['group_reward_baseline'][start:start+3], torch.tensor([7.,3.05,.1]))
        torch.testing.assert_close(data['return_r'][start:start+3], torch.tensor([4.,3.,2.]))
        rollout.episodes[1][-1]['reward'] += 10.
        changed = rollout.tensors(agent.settings, 'cpu')
        torch.testing.assert_close(changed['group_reward_baseline'][start:start+3], data['group_reward_baseline'][start:start+3])
        torch.testing.assert_close(changed['return_r'][start:start+3], data['return_r'][start:start+3]+10.)

    def test_breach_sampler_precedes_first_effective_intervention_and_excludes_audit_seeds(self):
        config = copy.deepcopy(self.config)
        config['environment']['total_time'] = 1.2
        config['mappo'].update(theta_space_gate=True)
        agent = PredictiveMAPPO(config)
        agent.enable_compatibility_prior()
        rollout = collect_episodes(agent, make_env(agent.config), 1, 2.)
        episode, summary = rollout.episodes[0], rollout.summaries[0]
        summary.update(breach=1, seed=100321)
        mean = torch.zeros(len(episode), 3, 1)
        mean[3:] = .3
        for i, row in enumerate(episode):
            row['gate_raw'] = mean[i].numpy()
        with patch.object(agent.actor, 'gate_distribution', return_value=(
                torch.distributions.Normal(mean, torch.ones_like(mean)), None, torch.zeros_like(mean))):
            cases = breach_branch_cases(agent, rollout, 8, 931, lead_steps=2)
        self.assertEqual(cases[0]['source_step'], 1)
        self.assertEqual(len(cases[0]['prefix']), 1)
        self.assertEqual(len(set(cases[0]['seeds'])), 8)
        self.assertEqual(cases[0]['recovery_kind'], 'breach')
        summary['seed'] = 60000
        with self.assertRaises(ValueError):
            breach_branch_cases(agent, rollout, 8, 931)


if __name__ == '__main__':
    unittest.main()
