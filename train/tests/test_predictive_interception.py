"""Behavioral contracts for the mission-level predictive controller."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np

from feedback_joint_control import NominalEnvironment, observe, snapshot
from predictive_interception import PredictiveInterceptionController, constant_turn_displacement


def policy(obs):
    result = np.empty((len(obs), 3), dtype=np.float32)
    result[:, 0] = .65
    result[:, 1] = .2 + .15 * np.tanh(obs[:, 3])
    result[:, 2] = .5 + .15 * np.tanh(obs[:, 2] / 30.)
    return result


class MissionPredictionContracts(unittest.TestCase):
    def scene(self, n=3):
        np.random.seed(17318)
        env = NominalEnvironment(defender_num=n)
        obs, _ = env.reset(agility=2.25, noisy_agility=False)
        return env, obs

    def test_constant_turn_includes_zero_rate_limit(self):
        v = np.array([2., 0.])
        np.testing.assert_array_equal(constant_turn_displacement(v, 0., 2.), [4., 0.])
        np.testing.assert_allclose(constant_turn_displacement(v, 1., np.pi/2), [2., 2.], atol=1e-12)

    def test_baseline_feedback_is_unchanged_cbf(self):
        env, obs = self.scene(6)
        rule = PredictiveInterceptionController(6, policy)
        m = observe(env, obs)
        expected = rule.safety.control(m.defenders, policy(obs), m.boids)[1]
        actual = rule.feedback(m, 'baseline')[0]
        np.testing.assert_array_equal(expected, actual)

    def test_fixed_ablation_performs_no_trajectory_prediction(self):
        env, obs = self.scene()
        rule = PredictiveInterceptionController(3, policy, fixed_template='lead_guard')
        with patch.object(rule, 'predict', side_effect=AssertionError('Fixed policy must not forecast.')):
            for _ in range(6):
                force, info = rule.control(observe(env, obs))
                self.assertEqual(info['candidate_policies'], 0)
                self.assertIsNone(info['predicted_next_state'])
                obs, _, _, _ = env.step(env.thrust_to_action(force), 'RL')

    def test_feedback_predictions_match_physical_execution_across_blocks(self):
        for n in (3, 6):
            env, obs = self.scene(n)
            rule = PredictiveInterceptionController(n, policy, block_steps=20)
            plans = 0
            for _ in range(40):
                force, info = rule.control(observe(env, obs))
                if info['planned']:
                    plans += 1
                    self.assertLessEqual(rule.plan['score'], rule.plan['baseline_score'])
                np.testing.assert_allclose(force, rule.plan['commands'][info['plan_offset']], atol=.003, rtol=0.)
                obs, _, done, _ = env.step(env.thrust_to_action(force), 'RL')
                np.testing.assert_allclose(snapshot(env)[:, :3], info['predicted_next_state'][:, :3], atol=2e-6, rtol=0.)
                if done:
                    break
            self.assertEqual(plans, 2)

    def test_prediction_does_not_consume_randomness(self):
        env, obs = self.scene()
        rule = PredictiveInterceptionController(3, policy)
        before = np.random.get_state()
        rule.predict(observe(env, obs), 'lead_guard')
        after = np.random.get_state()
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
