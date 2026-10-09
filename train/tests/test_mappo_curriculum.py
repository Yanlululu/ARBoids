"""Critical-state continuation preserves physics, likelihoods and the dual budget."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parallel_mappo
from parallel_mappo import ParallelRollouts, collision_branch_cases, restore_branch_start
from policy.mappo import PredictiveMAPPO
from train_mappo import make_env, play_episode, continue_episode, collect_episodes


def configuration():
    return dict(agent=dict(defender_num=3, boid_state=True, form_reward=True),
                environment=dict(protocol='paper-parameters-v1', total_time=1.),
                mappo=dict(hidden_dim=32, relation_dim=8, coordination_dim=4,
                           ppo_epochs=2, minibatch_size=4, theta_space_gate=True),
                prediction=dict(horizon=.1))


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        np.random.seed(8)

    def source(self, agent):
        seed = 123456
        np.random.seed(seed)
        torch.manual_seed(seed)
        episode, _ = play_episode(agent, make_env(agent.config), 2., True)
        index = 2
        return dict(source_seed=seed, source_step=index,
                    prefix=[row['executed_action'].copy() for row in episode[:index]],
                    obs=episode[index]['obs'].copy(), state=episode[index]['state'].copy(),
                    seeds=[345000, 345001])

    def test_replay_restores_exact_state_and_retains_original_deadline(self):
        agent = PredictiveMAPPO(configuration())
        case = self.source(agent)
        env = make_env(agent.config)
        obs = restore_branch_start(env, case, 2., True)
        np.testing.assert_array_equal(env.env.centralized_state(), case['state'])
        np.testing.assert_array_equal(obs.astype(np.float32), case['obs'])
        self.assertAlmostEqual(env.env.Current_T, .4)
        episode, summary = continue_episode(agent, env, obs)
        self.assertAlmostEqual(episode[0]['timestamp'], .4)
        self.assertTrue(episode[-1]['terminated'])
        self.assertLessEqual(summary['steps'], 3)
        self.assertAlmostEqual(env.env.Current_T, 1.)
        case['state'][0] += .01
        with self.assertRaisesRegex(ValueError, 'reproduce'):
            restore_branch_start(make_env(agent.config), case, 2., True)

    def test_parallel_branches_are_new_on_policy_samples(self):
        agent = PredictiveMAPPO(configuration())
        agent.config['training'] = dict(branch_gate_scale=4.)
        case = self.source(agent)
        with ParallelRollouts(agent.config, 2) as pool:
            rollout, restored = pool.resume_cases(agent, [case], 2., True)
        self.assertEqual(restored, len(case['prefix']))
        self.assertEqual(len(rollout.episodes), 2)
        for episode, summary in zip(rollout.episodes, rollout.summaries):
            self.assertTrue(summary['curriculum'])
            self.assertEqual(summary['source_seed'], case['source_seed'])
            np.testing.assert_array_equal(episode[0]['state'], case['state'])
            self.assertAlmostEqual(sum(row['cost'] for row in episode), summary['collision'])
            self.assertTrue(all(row['gate_scale'] == 4. for row in episode))
        self.assertFalse(np.array_equal(rollout.episodes[0][0]['gate_raw'], rollout.episodes[1][0]['gate_raw']))
        batch = rollout.tensors(agent.settings, 'cpu')
        logp_l, logp_g, _ = agent.evaluate_batch(batch)
        torch.testing.assert_close(logp_l, batch['old_logp_l'], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(logp_g, batch['old_logp_g'], atol=2e-5, rtol=2e-5)
        with self.assertRaisesRegex(ValueError, 'alone'):
            _ = rollout.collision_rate

    def test_conditional_costs_train_values_but_do_not_change_dual_estimate(self):
        agent = PredictiveMAPPO(configuration())
        rollout = collect_episodes(agent, make_env(agent.config), 2, 2.)
        self.assertEqual(rollout.collision_rate, 0.)
        # A synthetic terminal collision isolates cost accounting from dynamics.
        episode = copy.deepcopy(rollout.episodes[0])
        episode[-1]['cost'] = 1.
        summary = dict(rollout.summaries[0], collision=1, curriculum=True,
                       success=0, capture=0, timeout=0, outcome_code=2)
        rollout.add_episode(episode, summary)
        self.assertEqual(rollout.collision_rate, 0.)
        before = agent.lagrange.value
        metrics = agent.update(rollout)
        self.assertEqual(metrics['collision_rate'], 0.)
        self.assertGreater(metrics['cost_value_loss'], 0.)
        self.assertAlmostEqual(agent.lagrange.value, before-agent.settings.lagrange_lr*.05)

    def test_paired_reference_preserves_sampled_futures_and_matches_independent_replay(self):
        agent = PredictiveMAPPO(configuration())
        case = self.source(agent)
        actor = {key: value.detach().numpy().copy() for key, value in agent.actor.state_dict().items()}
        critic = {key: value.detach().numpy().copy() for key, value in agent.critic.state_dict().items()}
        config = copy.deepcopy(agent.config)
        parallel_mappo._initialize(config)
        plain, _ = parallel_mappo._branches((config, actor, critic, [case], 2., True))
        paired_config = copy.deepcopy(config)
        paired_config['training'] = dict(branch_paired_reference=True)
        paired, _ = parallel_mappo._branches((paired_config, actor, critic, [case], 2., True))
        for (before, old), (after, summary) in zip(plain, paired):
            self.assertEqual(old, {k: v for k, v in summary.items() if not k.startswith('paired_reference_')})
            self.assertEqual(len(before), len(after))
            for left, right in zip(before, after):
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key], err_msg=key)
            env = make_env(agent.config)
            observation = restore_branch_start(env, case, 2., True)
            np.random.seed(summary['seed'])
            _, reference = continue_episode(agent, env, observation, deterministic=True)
            for key in ('success','capture','timeout','breach','collision','task_return','steps'):
                self.assertEqual(summary['paired_reference_'+key], reference[key])

    def test_failure_sampling_is_bounded_and_excludes_reserved_seeds(self):
        row = dict(executed_action=np.zeros((3, 3)), obs=np.zeros((3, 18)), state=np.zeros(30))
        source = SimpleNamespace(episodes=[[row]*5, [row]*6],
                                 summaries=[dict(collision=1, seed=123456), dict(collision=0, seed=654321)])
        cases = collision_branch_cases(source, 7, 2, 44)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]['source_step'], 3)
        self.assertEqual(len(cases[0]['seeds']), 7)
        self.assertTrue(all(seed >= 100000 for seed in cases[0]['seeds']))
        self.assertEqual(cases[0]['seeds'], collision_branch_cases(source, 7, 2, 44)[0]['seeds'])
        self.assertEqual(collision_branch_cases(source, 0, 2, 44), [])
        source.summaries[0]['seed'] = 20000
        with self.assertRaisesRegex(ValueError, 'training seeds'):
            collision_branch_cases(source, 7, 2, 44)


if __name__ == '__main__':
    unittest.main()
