"""Numerical equivalence of compiled model integration and task prediction."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import torch

from array_rollout import ArrayNominalEnvironment, ArrayRolloutController, BatchedSourcePolicy
from evaluate_feedback_joint import CHECKPOINT
from feedback_joint_control import nominal_scene, observe, snapshot
from jit_nominal_environment import JitNominalEnvironment
from jit_rollout import JitRolloutController
from source_arboids import SourcePolicy, numerical_source


class CompiledModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policy = SourcePolicy(CHECKPOINT)
        cls.batched = BatchedSourcePolicy(cls.policy)

    def scene(self, n):
        np.random.seed(7740 + n)
        env = numerical_source().TADEnv(defender_num=n)
        obs, _ = env.reset(agility=2.25, noisy_agility=False)
        return env, obs

    def test_compiled_integration_preserves_source_state_and_observations(self):
        for n in (2, 3, 4, 5, 6):
            env, obs = self.scene(n)
            a, b = nominal_scene(observe(env, obs)), nominal_scene(observe(env, obs))
            a.__class__, b.__class__ = ArrayNominalEnvironment, JitNominalEnvironment
            for tick in range(60):
                force = np.random.default_rng(448 + tick).uniform(-450., 1000., (n, 2))
                oa, _, da, _ = a.step(a.thrust_to_action(force), 'RL')
                ob, _, db, _ = b.step(b.thrust_to_action(force), 'RL')
                np.testing.assert_allclose(snapshot(a), snapshot(b), atol=1e-8, rtol=0.)
                np.testing.assert_allclose(oa, ob, atol=1e-7, rtol=0.)
                np.testing.assert_allclose(a.boids_actions, b.boids_actions, atol=1e-6, rtol=0.)
                self.assertEqual(da, db)
                if da:
                    break

    def test_closed_loop_scores_and_prefix_match_array_backend(self):
        for n in (2, 6):
            env, obs = self.scene(n)
            measurement = observe(env, obs)
            settings = dict(prediction_policy=self.batched, block_steps=10, blend=1.,
                            tail_policy='candidate', tail_steps=100, failure_cost='delay')
            a = ArrayRolloutController(n, self.policy, **settings)
            b = JitRolloutController(n, self.policy, **settings)
            for template in ('baseline', 'learned', 'direct', 'lead', 'lead_guard'):
                reference, compiled = a.predict(measurement, template), b.predict(measurement, template)
                for key in ('positions', 'attackers'):
                    np.testing.assert_allclose(reference[key], compiled[key], atol=1e-6, rtol=0.)
                np.testing.assert_allclose(reference['commands'], compiled['commands'], atol=.005, rtol=0.)
                np.testing.assert_allclose(reference['score'], compiled['score'], atol=.001, rtol=0.)
                self.assertEqual(reference['continuation_outcome'], compiled['continuation_outcome'])
            self.assertIs(b.policy, self.policy)
            self.assertEqual(b.observer.updates, 0)


if __name__ == '__main__':
    unittest.main()
