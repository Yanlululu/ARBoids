"""Causality, navigation preservation, and task-time prediction contracts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np

from attacker_motion_observer import AttackerMotionObserver
from feedback_joint_control import NominalEnvironment, observe, snapshot
from predictive_interception_v2 import TaskTimePredictiveController, reachable_capture_time


def policy(obs):
    result = np.empty((len(obs), 3), dtype=np.float32)
    result[:, 0] = .65
    result[:, 1] = .2 + .15*np.tanh(obs[:, 3])
    result[:, 2] = .5 + .15*np.tanh(obs[:, 2]/30.)
    return result


class TaskTimeContracts(unittest.TestCase):
    def scene(self, agility=2.25, n=3):
        np.random.seed(8813)
        env = NominalEnvironment(defender_num=n)
        obs, _ = env.reset(agility=agility, noisy_agility=False)
        return env, obs

    def test_reachability_distinguishes_escaping_and_approaching_targets(self):
        state = np.array([[0., 0., 0., 0., 0., 0.]])
        escaping = np.array([20., 0., 0., 5., 0., 0.])
        approaching = np.array([20., 0., np.pi, -5., 0., 0.])
        self.assertEqual(reachable_capture_time(state, escaping)[0], 80.)
        self.assertLess(reachable_capture_time(state, approaching)[0], 3.)

    def test_direct_guidance_preserves_original_boids(self):
        env, obs = self.scene(n=6)
        rule = TaskTimePredictiveController(6, policy)
        np.testing.assert_array_equal(rule.guidance(observe(env, obs), 'direct'), env.boids_actions)

    def test_observer_identifies_from_past_motion_without_hidden_agility(self):
        for actual in (1.5, 2.25, 3.):
            env, obs = self.scene(actual)
            observer = AttackerMotionObserver()
            self.assertEqual(observer.update(observe(env, obs)), 2.25)
            for _ in range(30):
                obs, _, _, _ = env.step(np.zeros((3, 2)), 'RL')
                observer.update(observe(env, obs))
            self.assertLess(abs(observer.estimate-actual), .08)

    def test_prediction_does_not_update_observer_or_rng(self):
        env, obs = self.scene()
        rule = TaskTimePredictiveController(3, policy)
        before = np.random.get_state()
        rule.predict(observe(env, obs), 'lead')
        self.assertEqual(rule.observer.updates, 0)
        self.assertIsNone(rule.observer.previous)
        for a, b in zip(before, np.random.get_state()):
            np.testing.assert_array_equal(a, b)

    def test_conservative_forecast_matches_executed_prefix(self):
        env, obs = self.scene(n=6)
        rule = TaskTimePredictiveController(6, policy, block_steps=20, adaptive_attacker=False)
        planned = 0
        for _ in range(40):
            command, info = rule.control(observe(env, obs))
            planned += int(info['planned'])
            np.testing.assert_allclose(command, rule.plan['commands'][info['plan_offset']], atol=.003, rtol=0.)
            obs, _, done, _ = env.step(env.thrust_to_action(command), 'RL')
            np.testing.assert_allclose(snapshot(env)[:, :3], info['predicted_next_state'][:, :3], atol=2e-6, rtol=0.)
            if done:
                break
        self.assertEqual(planned, 2)


if __name__ == '__main__':
    unittest.main()
