"""Numerical, observation and causal contracts for batched prediction."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import torch

from array_rollout import ArrayNominalEnvironment, ArrayRolloutController, BatchedSourcePolicy
from compiled_source_policy import CompiledSourcePolicy
from evaluate_feedback_joint import CHECKPOINT
from feedback_joint_control import nominal_scene, observe, snapshot
from feedback_joint_fast import FastNominalEnvironment
from rollout_interception import RolloutInterceptionController
from source_arboids import SourcePolicy, numerical_source


class ArrayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source = SourcePolicy(CHECKPOINT)
        cls.batched = BatchedSourcePolicy(cls.source)
        cls.compiled = CompiledSourcePolicy(cls.source)

    def scene(self, n):
        np.random.seed(7720 + n)
        env = numerical_source().TADEnv(defender_num=n)
        obs, _ = env.reset(agility=2.25, noisy_agility=False)
        return env, obs

    def test_vectorized_environment_matches_source_observations_and_dynamics(self):
        for n in (2, 3, 6):
            env, obs = self.scene(n)
            measurement = observe(env, obs)
            a, b = nominal_scene(measurement), nominal_scene(measurement)
            a.__class__, b.__class__ = FastNominalEnvironment, ArrayNominalEnvironment
            for tick in range(12):
                force = np.random.default_rng(100 + tick).uniform(-400., 950., (n, 2))
                oa, _, da, _ = a.step(a.thrust_to_action(force), 'RL')
                ob, _, db, _ = b.step(b.thrust_to_action(force), 'RL')
                np.testing.assert_allclose(snapshot(a), snapshot(b), atol=1e-12, rtol=0.)
                np.testing.assert_allclose(oa, ob, atol=1e-10, rtol=0.)
                np.testing.assert_allclose(a.boids_actions, b.boids_actions, atol=1e-9, rtol=0.)
                self.assertEqual(da, db)

    def test_batched_actor_is_within_float32_tolerance(self):
        for n in (2, 3, 4, 5, 6):
            env, obs = self.scene(n)
            for _ in range(12):
                expected = self.source(obs)
                np.testing.assert_allclose(self.batched(obs), expected, atol=3e-5, rtol=0.)
                obs, _, done, _ = env.step(expected, 'AdaRes')
                if done:
                    break

    def test_full_rollout_preserves_execution_prefix_with_tolerance(self):
        env, obs = self.scene(6)
        measurement = observe(env, obs)
        settings = dict(block_steps=10, blend=1., tail_steps=100)
        reference = RolloutInterceptionController(6, self.compiled, **settings)
        batched = ArrayRolloutController(6, self.source, prediction_policy=self.batched, **settings)
        rng = np.random.get_state()
        for template in ('baseline', 'learned', 'direct', 'lead', 'lead_guard'):
            a, b = reference.predict(measurement, template), batched.predict(measurement, template)
            np.testing.assert_allclose(a['positions'], b['positions'], atol=5e-5, rtol=0.)
            np.testing.assert_allclose(a['commands'], b['commands'], atol=.05, rtol=0.)
        self.assertIs(batched.policy, self.source)
        self.assertEqual(batched.observer.updates, 0)
        for a, b in zip(rng, np.random.get_state()):
            np.testing.assert_array_equal(a, b)

    def test_forecast_qp_matches_full_diagnostic_qp(self):
        env, obs = self.scene(6)
        measurement = observe(env, obs)
        rule = ArrayRolloutController(6, self.batched, prediction_policy=self.batched, blend=1.)
        for template in ('baseline', 'learned', 'direct', 'lead', 'lead_guard'):
            expected, _, _ = rule.feedback(measurement, template)
            np.testing.assert_array_equal(rule._forecast_force(measurement, template), expected)

    def test_failed_candidates_prefer_later_loss_not_earlier_loss(self):
        env, obs = self.scene(6)
        measurement = observe(env, obs)

        class FailureEnvironment:
            def __init__(self, failure_step):
                self.failure_step, self.steps = failure_step, 0

            def thrust_to_action(self, force):
                return force

            def step(self, action, mode):
                self.steps += 1
                return obs, 0., int(self.steps >= self.failure_step), None

        scores = {}
        for ranking in ('legacy', 'delay'):
            rule = ArrayRolloutController(6, self.source, tail_steps=0, failure_cost=ranking)
            for failure_step in (2, 8):
                fake = FailureEnvironment(failure_step)
                with patch('array_rollout.nominal_scene', return_value=fake), \
                     patch('array_rollout.ArrayNominalEnvironment', FailureEnvironment), \
                     patch('array_rollout.observe', return_value=measurement), \
                     patch('array_rollout.reachable_capture_time', return_value=np.full(6, 80.)), \
                     patch.object(rule, '_forecast_force', return_value=np.zeros((6, 2))):
                    scores[ranking, failure_step] = rule.predict(measurement, 'baseline')['score']
        self.assertLess(scores['legacy', 2], scores['legacy', 8])
        self.assertLess(scores['delay', 8], scores['delay', 2])

    def test_zero_continuation_ablation_has_identical_candidate_prefix(self):
        env, obs = self.scene(6)
        measurement = observe(env, obs)
        settings = dict(blend=1., block_steps=5, prediction_policy=self.batched, failure_cost='delay')
        short = ArrayRolloutController(6, self.source, tail_steps=0, **settings)
        long = ArrayRolloutController(6, self.source, tail_steps=100, **settings)
        for template in ('baseline', 'lead_guard'):
            a, b = short.predict(measurement, template), long.predict(measurement, template)
            for key in ('positions', 'attackers', 'commands'):
                np.testing.assert_array_equal(a[key], b[key])
            self.assertEqual(a['continuation_steps'], 0)


if __name__ == '__main__':
    unittest.main()
