"""Behavioral invariants of the two-stage policy, prediction and constraints."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'train'))
sys.path.insert(0, str(ROOT / 'vrx'))
from envs.modules import WAMV
from policy.interaction_prediction import InteractionPredictor, PredictionConfig, exchange_candidates
from policy.mappo import PredictiveMAPPO, MAPPOConfig, LagrangeMultiplier
from policy.rollout_buffer import RolloutBuffer, gae
from train_mappo import make_env, collect_episodes, evaluate
from mappo_controller import MAPPOController, synchronize_kinematics


def small_config():
    return dict(algorithm='predictive-mappo', agent=dict(defender_num=3, boid_state=True, form_reward=True),
                environment=dict(protocol='paper-parameters-v1', total_time=2., agility_noise_half_width=.5),
                mappo=dict(hidden_dim=32, relation_dim=16, coordination_dim=8, ppo_epochs=2, minibatch_size=8),
                prediction=dict(horizon=.4, dt=.05, distance_scale=5., speed_scale=5.))


class PredictionTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        torch.manual_seed(3)
        torch.set_num_threads(1)

    def test_vectorized_prediction_matches_original_dynamics_at_every_substep(self):
        predictor = InteractionPredictor(PredictionConfig(horizon=.4))
        rng = np.random.default_rng(22)
        states = rng.normal(size=(6, 6))
        states[:, :2] *= 20
        states[:, 5] *= .1
        thrusts = rng.uniform(-700., 1200., size=(6, 2))  # includes actuator clipping
        vectorized = predictor.trajectories(states, thrusts)
        for i, state in enumerate(states):
            boat = WAMV()
            boat.reset(state[:2].copy(), state[2])
            boat.velocity_r = state[3:].copy()
            boat.N = 1
            expected = [boat.pos.copy()]
            for _ in range(8):
                boat.step(thrusts[i], np.zeros(3))
                expected.append(boat.pos.copy())
            np.testing.assert_allclose(vectorized[i], expected, rtol=1e-11, atol=1e-11)

    def test_exchange_and_prediction_have_no_live_state_or_rng_side_effects(self):
        env = make_env(small_config())
        env.reset()
        before = copy.deepcopy(env.env.__dict__)
        state = env.env.prediction_snapshot()
        rng_before = np.random.get_state()
        messages = exchange_candidates(state, env.env.thrust_to_action(env.env.boids_actions), np.zeros((3, 2)), 0.)
        edges, mask = InteractionPredictor(PredictionConfig(horizon=.4)).features(messages)
        np.testing.assert_array_equal(state, env.env.prediction_snapshot())
        np.testing.assert_array_equal(before['Pos_Def'], env.env.Pos_Def)
        self.assertEqual(before['Current_T'], env.env.Current_T)
        np.testing.assert_array_equal(rng_before[1], np.random.get_state()[1])
        self.assertEqual(rng_before[2:], np.random.get_state()[2:])
        self.assertEqual(edges.shape, (3, 3, 18))
        self.assertFalse(mask.diagonal().any())
        np.testing.assert_allclose(edges[..., 0:3], edges.transpose(1, 0, 2)[..., 0:3])
        np.testing.assert_allclose(edges[..., 3:6], edges.transpose(1, 0, 2)[..., 6:9])
        with self.assertRaises(ValueError):
            exchange_candidates(state, np.zeros((3, 2)), np.zeros((3, 2)), [0., .1, 0.])

    def test_message_permutation_preserves_equivariant_predictions(self):
        states = np.random.normal(size=(3, 6))
        boids, proposals = np.random.uniform(-1, 1, size=(2, 3, 2))
        predictor = InteractionPredictor(PredictionConfig(horizon=.4))
        edges, _ = predictor.features(exchange_candidates(states, boids, proposals, 0.))
        order = [2, 0, 1]
        permuted, _ = predictor.features(exchange_candidates(states[order], boids[order], proposals[order], 0.))
        np.testing.assert_allclose(permuted, edges[order][:, order])


class LearningTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(8)
        torch.manual_seed(8)
        torch.set_num_threads(1)
        self.config = small_config()
        self.agent = PredictiveMAPPO(self.config)

    def sample(self):
        env = make_env(self.config)
        obs = env.reset()
        action, record = self.agent.act(obs, env.env.prediction_snapshot(),
                                        env.env.thrust_to_action(env.env.boids_actions), 0.)
        batch = {key: self.agent.tensor(value, key == 'mask').unsqueeze(0) for key, value in record.items()}
        return env, action, record, batch

    def test_old_samples_have_unit_ratios_without_resampling(self):
        _, _, _, batch = self.sample()
        with patch.object(self.agent.actor, 'sample_proposals', side_effect=AssertionError('resampled')):
            new_l, new_g, _ = self.agent.evaluate_batch(batch)
        torch.testing.assert_close((new_l - batch['old_logp_l']).exp(), torch.ones_like(new_l))
        torch.testing.assert_close((new_g - batch['old_logp_g']).exp(), torch.ones_like(new_g))

    def test_prediction_and_coordination_affect_gate_and_receive_gradients(self):
        _, _, _, b = self.sample()
        actor = self.agent.actor
        distribution, eta = actor.gate_distribution(b['obs'], b['proposals'], b['boids'], b['edges'], b['mask'])
        torch.testing.assert_close(eta + eta.transpose(-1, -2), torch.ones_like(eta))
        changed, _ = actor.gate_distribution(b['obs'], b['proposals'], b['boids'], b['edges'] * 0., b['mask'])
        self.assertGreater(float((distribution.mean - changed.mean).abs().max().detach()), 1e-7)
        _, logp, _ = self.agent.evaluate_batch(b)
        (-logp.mean()).backward()
        for name in ('relation_encoder', 'priority_head', 'coordination_encoder', 'adapter', 'gate_mean'):
            norm = sum(float(p.grad.abs().sum()) for p in getattr(actor, name).parameters() if p.grad is not None)
            self.assertGreater(norm, 0., name)
        # No pathwise proposal-action gradient is used to train the proposal head.
        self.assertIsNone(actor.proposal_mean.weight.grad)
        no_neighbors = torch.zeros_like(b['mask'])
        empty, _ = actor.gate_distribution(b['obs'], b['proposals'], b['boids'], b['edges'], no_neighbors)
        self.assertTrue(torch.isfinite(empty.mean).all())

    def test_full_update_trains_both_stages_and_updates_multiplier_once(self):
        rollout = collect_episodes(self.agent, make_env(self.config), 2, 2.)
        before = self.agent.actor.proposal_mean.weight.detach().clone()
        result = self.agent.update(rollout)
        self.assertTrue(np.isfinite(list(result.values())).all())
        self.assertFalse(torch.equal(before, self.agent.actor.proposal_mean.weight))
        for name in ('proposal_mean', 'relation_encoder', 'priority_head', 'coordination_encoder', 'adapter', 'gate_mean'):
            self.assertGreater(result['grad_' + name], 0.)
        expected = max(0., result['lambda_used'] + (rollout.collision_rate - .05))
        self.assertAlmostEqual(self.agent.lagrange.value, expected)

    def test_checkpoint_resume_and_vrx_use_same_deterministic_policy(self):
        env, _, _, _ = self.sample()
        self.agent.update(collect_episodes(self.agent, make_env(self.config), 1, 2.))
        obs, states = env.env._get_obs()[0], env.env.prediction_snapshot()
        boids = env.env.boids_actions.copy()
        action, _ = self.agent.act(obs, states, env.env.thrust_to_action(boids), 0., True)
        with tempfile.TemporaryDirectory(prefix='arboids-mappo-') as directory:
            filename = Path(directory) / 'model.pth'
            self.agent.save(filename)
            expected_np, expected_torch = np.random.rand(), torch.rand(3)
            loaded = PredictiveMAPPO.from_checkpoint(filename, restore_rng=True)
            self.assertEqual(np.random.rand(), expected_np)
            torch.testing.assert_close(torch.rand(3), expected_torch)
            self.assertEqual(loaded.updates, self.agent.updates)
            self.assertEqual(loaded.total_steps, self.agent.total_steps)
            self.assertEqual(loaded.lagrange.value, self.agent.lagrange.value)
            self.assertEqual(len(loaded.actor_optimizer.state), len(self.agent.actor_optimizer.state))
            reloaded, _ = loaded.act(obs, states, env.env.thrust_to_action(boids), 0., True)
            np.testing.assert_array_equal(action, reloaded)
            deployed = MAPPOController(filename).control(obs, states, boids, 0.)
            expected = action[:, 2:3] * env.env.action_to_thrust(action[:, :2]) + (1-action[:, 2:3]) * boids
            np.testing.assert_allclose(deployed, expected, rtol=1e-6, atol=1e-4)

    def test_evaluation_preserves_training_rng(self):
        np_state, torch_state = np.random.get_state(), torch.get_rng_state()
        evaluate(self.agent, episodes=1, first_seed=901, duration=.4)
        np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
        self.assertEqual(np_state[2:], np.random.get_state()[2:])
        torch.testing.assert_close(torch_state, torch.get_rng_state())


class ConstraintTests(unittest.TestCase):
    def test_undiscounted_cost_and_task_deadline_bootstrap(self):
        advantages, returns = gae([0., 0., 1.], [0., 0., 0.], 1., 1.)
        np.testing.assert_array_equal(returns, [1., 1., 1.])
        _, returns = gae([0., 0.], [5., 7.], 1., 1.)
        np.testing.assert_array_equal(returns, [0., 0.])
        with self.assertRaises(ValueError):
            MAPPOConfig(cost_gamma=.99)
        with self.assertRaises(ValueError):
            RolloutBuffer().add_episode([dict(terminated=False)], {})

    def test_collision_is_independent_of_outcome_and_counted_once_per_team(self):
        config = small_config()
        config['agent']['form_reward'] = False
        env = make_env(config)
        obs = env.reset()
        env.env.attacker.pos[:] = [1., 0.]
        for boat, pos in zip(env.env.defender_list, ([10., 0.], [11., 0.], [12., 0.])):
            boat.pos[:] = pos
        env.env._get_obs()
        self.assertEqual(env.env._isTerminate(), 1)
        with patch.object(env.env, 'step', side_effect=[(obs, np.zeros(3), 0, np.zeros(8)),
                                                        (obs, np.zeros(3), 1, np.zeros(8))]):
            first = env.step(np.zeros((3, 3)))
            second = env.step(np.zeros((3, 3)))
        self.assertEqual((first.cost, second.cost), (1., 0.))
        self.assertEqual(first.reward, -100.)  # No fixed collision penalty in the task reward.
        self.assertTrue(second.events['breach'] and second.events['collision'])
        self.assertTrue(second.terminated)

    def test_gate_endpoints_execute_correct_thrust(self):
        for gate in (0., 1.):
            env = make_env(small_config())
            env.reset()
            boids = env.env.boids_actions.copy()
            action = np.tile([.2, -.7, gate], (3, 1))
            expected = boids if gate == 0 else env.env.action_to_thrust(action[:, :2])
            with patch.object(env.env.defender_list[0], 'step', wraps=env.env.defender_list[0].step) as step:
                env.step(action)
            np.testing.assert_allclose(step.call_args.args[0], expected[0])

    def test_multiplier_moves_in_both_directions_and_projects_at_zero(self):
        multiplier = LagrangeMultiplier(1., 2., .1)
        self.assertAlmostEqual(multiplier.update(.3), 1.4)
        self.assertAlmostEqual(multiplier.update(0.), 1.2)
        multiplier.value = .05
        self.assertEqual(multiplier.update(0.), 0.)

    def test_vrx_feedback_alignment_uses_yaw_rate_and_timestamp(self):
        states, timestamp, skew = synchronize_kinematics(
            [[0., 0.], [1., 2.]], [0., 1.], [[2., 0.], [0., 0.]], [.2, 0.], [1., 1.1])
        np.testing.assert_allclose(states[0, :3], [.2, 0., .02])
        self.assertAlmostEqual(timestamp, 1.1)
        self.assertAlmostEqual(skew, .1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
