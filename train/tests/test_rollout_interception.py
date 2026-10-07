"""Contracts separating the executed prefix from the continuation value."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np

from feedback_joint_control import NominalEnvironment, observe, snapshot
from predictive_interception_v2 import TaskTimePredictiveController, TEMPLATES
from rollout_interception import RolloutInterceptionController
from test_task_time_prediction import policy


class ContinuationContracts(unittest.TestCase):
    def scene(self, n=6):
        np.random.seed(1138)
        env = NominalEnvironment(defender_num=n)
        obs, _ = env.reset(agility=2.25, noisy_agility=False)
        return env, obs

    def test_tail_does_not_change_any_candidate_execution_prefix(self):
        env, obs = self.scene()
        short = TaskTimePredictiveController(6, policy)
        long = RolloutInterceptionController(6, policy, tail_steps=40, blend=.5)
        measurement = observe(env, obs)
        for template in TEMPLATES:
            a, b = short.predict(measurement, template), long.predict(measurement, template)
            for field in ('positions', 'attackers', 'commands'):
                np.testing.assert_array_equal(a[field], b[field])
            self.assertLessEqual(len(b['commands']), 10)
            self.assertEqual(b['outcome'], a['outcome'])

    def test_continuation_uses_stated_baseline_and_no_randomness(self):
        env, obs = self.scene()
        rule = RolloutInterceptionController(6, policy, tail_steps=40)
        before = np.random.get_state()
        with patch.object(rule, 'feedback', wraps=rule.feedback) as feedback:
            plan = rule.predict(observe(env, obs), 'lead')
        calls = [call.args[1] for call in feedback.call_args_list]
        self.assertEqual(calls[:10], ['lead']*10)
        self.assertTrue(len(calls)>10)
        self.assertEqual(calls[10:], ['baseline']*(len(calls)-10))
        self.assertEqual(plan['continuation_steps'], len(calls)-10)
        self.assertEqual(rule.observer.updates, 0)
        for a, b in zip(before, np.random.get_state()):
            np.testing.assert_array_equal(a, b)

    def test_chosen_prefix_matches_execution_across_replans(self):
        env, obs = self.scene()
        rule = RolloutInterceptionController(6, policy, tail_steps=40, adaptive_attacker=False)
        plans = 0
        for _ in range(20):
            force, info = rule.control(observe(env, obs))
            plans += int(info['planned'])
            np.testing.assert_allclose(force, rule.plan['commands'][info['plan_offset']], atol=.003, rtol=0.)
            obs, _, done, _ = env.step(env.thrust_to_action(force), 'RL')
            np.testing.assert_allclose(snapshot(env)[:, :3], info['predicted_next_state'][:, :3], atol=2e-6, rtol=0.)
            if done:
                break
        self.assertEqual(plans, 2)


if __name__ == '__main__':
    unittest.main()
