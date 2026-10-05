"""Actual-event objectives, constraint migration and batched worker integration."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parallel_mappo import ParallelRollouts
from policy.mappo import PredictiveMAPPO
from train_mappo import collect_episodes, make_env, play_episodes_batched, continue_episode
from policy.rollout_buffer import RolloutBuffer
from train_mappo_performance import (initialize_collision_minimization, fit_initial_values,
                                     qualifies, validation_result, protected_breach_reference)


def configuration():
    return dict(agent=dict(defender_num=3, boid_state=True, form_reward=True),
                environment=dict(protocol='paper-parameters-v1', total_time=.6),
                mappo=dict(hidden_dim=32, relation_dim=8, coordination_dim=4,
                           ppo_epochs=2, minibatch_size=64, joint_ratio=True),
                prediction=dict(horizon=.1))


class EventObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(831)
        np.random.seed(831)

    def agent(self):
        agent = PredictiveMAPPO(configuration())
        agent.update(collect_episodes(agent, make_env(agent.config), 2, 2.))
        return agent

    def test_migration_preserves_actor_optimizer_and_existing_critic_then_round_trips(self):
        agent = self.agent()
        actor = copy.deepcopy(agent.actor.state_dict())
        critic = copy.deepcopy(agent.critic.state_dict())
        optimizer = agent.actor_optimizer
        old_moments = {p: copy.deepcopy(s) for p, s in agent.critic_optimizer.state.items()}
        initialize_collision_minimization(agent, .93, .00375)
        self.assertIs(agent.actor_optimizer, optimizer)
        for key, value in actor.items():
            torch.testing.assert_close(agent.actor.state_dict()[key], value, rtol=0, atol=0)
        for key, value in critic.items():
            torch.testing.assert_close(agent.critic.state_dict()[key], value, rtol=0, atol=0)
        for p, state in old_moments.items():
            for key, value in state.items():
                torch.testing.assert_close(agent.critic_optimizer.state[p][key], value, rtol=0, atol=0)
        self.assertEqual(agent.lagrange.value, 0.)
        self.assertAlmostEqual(float(agent.critic.success_head.bias.detach()), .93, places=6)
        with self.assertRaises(ValueError):
            initialize_collision_minimization(agent, .93, .00375)
        rollout = collect_episodes(agent, make_env(agent.config), 2, 2.)
        fit_initial_values(agent, rollout, epochs=1)
        for key, value in actor.items():
            torch.testing.assert_close(agent.actor.state_dict()[key], value, rtol=0, atol=0)
        agent.success_lagrange.value, agent.breach_lagrange.value = 2.3, 5.7
        with tempfile.TemporaryDirectory(prefix='arboids-objective-') as directory:
            path = Path(directory)/'model.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
        self.assertTrue(restored.settings.minimize_collision)
        self.assertEqual(restored.success_lagrange.value, 2.3)
        self.assertEqual(restored.breach_lagrange.value, 5.7)
        for key, value in agent.critic.state_dict().items():
            torch.testing.assert_close(restored.critic.state_dict()[key], value, rtol=0, atol=0)

    def test_actual_terminal_labels_and_undiscounted_cost_ignore_prediction_and_bootstrap(self):
        agent = self.agent()
        initialize_collision_minimization(agent, .93, .00375)
        rollout = collect_episodes(agent, make_env(agent.config), 2, 2.)
        for i, (episode, summary) in enumerate(zip(rollout.episodes, rollout.summaries)):
            summary.update(success=1-i, breach=i, collision=i)
            for t in episode:
                t.update(cost=0., value_c=7., value_s=.2, value_b=.3)
            episode[-1]['cost'] = float(i)
        data = rollout.tensors(agent.settings, 'cpu')
        n = len(rollout.episodes[0])
        expected = torch.cat((torch.zeros(n), torch.ones(len(rollout.episodes[1]))))
        torch.testing.assert_close(data['return_c'], expected)
        torch.testing.assert_close(data['return_b'], expected)  # Both breach and collision count.
        torch.testing.assert_close(data['return_s'], 1.-expected)
        torch.testing.assert_close(data['advantage_c'], expected-7.)
        torch.testing.assert_close(data['advantage_s'], 1.-expected-.2)
        torch.testing.assert_close(data['advantage_b'], expected-.3)

    def test_success_floor_migration_preserves_breach_head_and_named_adam_state(self):
        agent = self.agent()
        agent.enable_breach_constraint(.00375)
        agent.update(collect_episodes(agent, make_env(agent.config), 2, 2.))
        before = copy.deepcopy(agent.critic.state_dict())
        actor = copy.deepcopy(agent.actor.state_dict())
        breach = agent.critic.breach_head
        initialize_collision_minimization(agent, .99, .0025)
        self.assertIs(agent.critic.breach_head, breach)
        self.assertFalse(agent.settings.breach_constraint)
        self.assertEqual(agent.settings.success_floor, .99)
        self.assertAlmostEqual(agent.success_lagrange.budget,.01)
        for name,value in before.items():
            torch.testing.assert_close(agent.critic.state_dict()[name],value,rtol=0,atol=0)
        rollout = collect_episodes(agent,make_env(agent.config),2,2.)
        fit_initial_values(agent,rollout,epochs=1)
        for name,value in actor.items():
            torch.testing.assert_close(agent.actor.state_dict()[name],value,rtol=0,atol=0)
        moments = {name:copy.deepcopy(agent.critic_optimizer.state[parameter])
                   for name,parameter in agent.critic.named_parameters()}
        with tempfile.TemporaryDirectory(prefix='arboids-success-floor-') as directory:
            path=Path(directory)/'model.pth'
            agent.save(path)
            restored=PredictiveMAPPO.from_checkpoint(path)
        for name,parameter in restored.critic.named_parameters():
            for key,value in moments[name].items():
                torch.testing.assert_close(restored.critic_optimizer.state[parameter][key],value,rtol=0,atol=0)
        self.assertEqual(restored.settings.success_floor,.99)

    def test_event_recovery_baselines_exclude_own_success_and_breach(self):
        agent = self.agent()
        initialize_collision_minimization(agent,.99,.0025)
        agent.settings.curriculum_cost_baseline=True
        rollout=collect_episodes(agent,make_env(agent.config),2,2.)
        roots=[]
        for i,success in enumerate((1,0,1)):
            roots.append(rollout.steps)
            episode=copy.deepcopy(rollout.episodes[0])
            for transition in episode:
                transition.update(cost=0.,value_s=.2,value_b=.3)
            summary=dict(rollout.summaries[0],curriculum=True,seed=900001+i,
                source_seed=800001,source_step=0,success=success,breach=1-success,collision=0)
            rollout.add_episode(episode,summary)
        data=rollout.tensors(agent.settings,'cpu')
        torch.testing.assert_close(data['return_s'][roots],torch.tensor([1.,0.,1.]))
        torch.testing.assert_close(data['advantage_s'][roots],torch.tensor([.5,-1.,.5]))
        torch.testing.assert_close(data['advantage_b'][roots],torch.tensor([-.5,1.,-.5]))

    def test_zero_collision_dual_still_trains_safety_and_all_policy_stages(self):
        agent = self.agent()
        initialize_collision_minimization(agent, .93, .00375)
        rollout = collect_episodes(agent, make_env(agent.config), 2, 2.)
        for i, (episode, summary) in enumerate(zip(rollout.episodes, rollout.summaries)):
            summary.update(success=1-i, breach=0, collision=i)
            for t in episode:
                t['cost'] = 0.
            episode[-1]['cost'] = float(i)
        agent.success_lagrange.value = agent.breach_lagrange.value = 0.
        before = agent.actor.gate_mean.weight.detach().clone()
        result = agent.update(rollout)
        self.assertEqual(result['lambda_used'], 0.)
        self.assertEqual(result['collision_objective_weight'], 1.)
        self.assertFalse(torch.equal(before, agent.actor.gate_mean.weight))
        for name in ('proposal_mean', 'relation_encoder', 'priority_head', 'adapter', 'gate_mean'):
            self.assertGreater(result['grad_' + name], 0.)

    def test_all_duals_exclude_failure_conditioned_curriculum_and_update_once(self):
        agent = self.agent()
        initialize_collision_minimization(agent, .93, .00375)
        rollout = collect_episodes(agent, make_env(agent.config), 2, 2.)
        for episode, summary in zip(rollout.episodes, rollout.summaries):
            summary.update(success=1, breach=0, collision=0)
            for t in episode:
                t['cost'] = 0.
        branch = copy.deepcopy(rollout.episodes[0])
        branch[-1]['cost'] = 1.
        rollout.add_episode(branch, dict(success=0, breach=1, collision=1, curriculum=True))
        result = agent.update(rollout)
        self.assertEqual(result['lambda_next'], 0.)
        self.assertAlmostEqual(result['success_multiplier_used'], 1.)
        self.assertAlmostEqual(result['success_multiplier_next'], .3)
        self.assertAlmostEqual(result['breach_multiplier_next'], 4.9625)

    def test_qualification_rejects_breach_regression_and_preserves_protected_subset(self):
        metrics = dict(success_rate=.94, collision_rate=.02, breach_rate=.006)
        self.assertTrue(qualifies(metrics, .93, .05, .065))
        self.assertFalse(qualifies(metrics, .93, .05, .065, .005))
        baseline = dict(success_rate=.90, collision_rate=.10, breach_rate=.05,
                        protected_reference=dict(seeds=list(range(20)), success_rate=.90,
                                                 collision_rate=.10, breach_rate=0.))
        rows = [dict(seed=i, success=int(i != 0), capture=int(i != 0), timeout=0,
                     breach=int(i == 0), collision=0, task_return=0., gate_mean=.5) for i in range(100)]
        _, passed, _ = validation_result(rows, baseline, .05, preserve_breach=True)
        self.assertFalse(passed)  # Overall breach rate passes; protected subset does not.
        with tempfile.TemporaryDirectory(prefix='arboids-baseline-') as directory:
            path = Path(directory)/'episodes.csv'
            path.write_text('seed,breach\n0,0\n1,1\n2,0\n', encoding='utf-8')
            reference = dict(protected_reference=dict(seeds=[0, 1]))
            protected_breach_reference(reference, path)
            self.assertEqual(reference['protected_reference']['breach_rate'], .5)

    def test_batched_workers_preserve_event_values_and_on_policy_likelihoods(self):
        agent = self.agent()
        initialize_collision_minimization(agent, .93, .00375)
        seeds = [741001, 741002, 741003, 741004]
        with ParallelRollouts(agent.config, workers=2, environments_per_worker=2) as pool:
            rollout = pool.run(agent, seeds, 2., True)
        expected = (play_episodes_batched(agent, seeds[:2], 2., True, 2) +
                    play_episodes_batched(agent, seeds[2:], 2., True, 2))
        for episode, (reference, _) in zip(rollout.episodes, expected):
            for actual, wanted in zip(episode, reference):
                for key in ('state', 'value_s', 'value_b', 'gate_raw', 'proposal_raw'):
                    np.testing.assert_allclose(actual[key], wanted[key], rtol=1e-6, atol=1e-6)
        batch = rollout.tensors(agent.settings, 'cpu')
        lp, lg, _ = agent.evaluate_batch(batch)
        torch.testing.assert_close(lp, batch['old_logp_l'], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lg, batch['old_logp_g'], rtol=1e-5, atol=1e-5)

    def test_curriculum_exploration_context_is_part_of_both_old_and_new_likelihoods(self):
        agent = self.agent()
        initialize_collision_minimization(agent, .93, .00375)
        rollout = RolloutBuffer()
        for scale in (1., 4.):
            env = make_env(agent.config)
            episode, summary = continue_episode(agent, env, env.reset(2.), gate_scale=scale)
            summary['curriculum'] = scale != 1.
            rollout.add_episode(episode, summary)
        data = rollout.tensors(agent.settings, 'cpu')
        proposal, gate, _ = agent.evaluate_batch(data)
        torch.testing.assert_close(proposal, data['old_logp_l'], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(gate, data['old_logp_g'], rtol=1e-5, atol=1e-5)
        wrong = dict(data, gate_scale=torch.ones_like(data['gate_scale']))
        _, wrong_gate, _ = agent.evaluate_batch(wrong)
        self.assertGreater(float((wrong_gate-gate).abs().max().detach()), .1)
        agent.settings.full_batch_guard = True
        agent.settings.target_kl = .1
        result = agent.update(rollout)
        self.assertTrue(np.isfinite(result['actor_loss']))
        self.assertGreater(result['actor_accepted_steps'], 0)

    def test_exploration_context_does_not_change_deterministic_actions_or_proposals(self):
        agent = self.agent()
        env = make_env(agent.config)
        obs = env.reset(2.)
        args = obs, env.env.prediction_snapshot(), env.env.thrust_to_action(env.env.boids_actions), 0.
        expected, _ = agent.act(*args, deterministic=True)
        actual, _ = agent.act(*args, deterministic=True, gate_scale=4.)
        np.testing.assert_array_equal(expected, actual)
        torch.manual_seed(170)
        _, original = agent.act(*args)
        torch.manual_seed(170)
        _, broad = agent.act(*args, gate_scale=4.)
        np.testing.assert_array_equal(original['proposal_raw'], broad['proposal_raw'])
        np.testing.assert_array_equal(original['edges'], broad['edges'])
        with self.assertRaises(ValueError):
            agent.act(*args, gate_scale=0.)


if __name__ == '__main__':
    unittest.main()
