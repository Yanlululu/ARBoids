"""Source fidelity and mechanism-isolation checks for the contribution study."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import torch

from source_arboids import numerical_source, verify_source, SOURCE_ROOT, source_mixture, verify_source_config
from gate_study_control import NominalModel, GateController


class ContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source = numerical_source()

    def test_original_files_and_class_identity(self):
        self.assertIn('train/envs/TADgame.py', verify_source()['files'])
        self.assertEqual(self.source.TADEnv.__module__, '_author_environment')
        self.assertEqual(self.source.WAMV.__module__, '_author_dynamics')
        self.assertEqual(self.source.ActorAdap.__module__, '_author_networks')
        self.assertEqual(self.source.TADEnv().Total_T, 80.)
        self.assertNotIn('protocol', self.source.TADEnv.__init__.__code__.co_varnames)

    def test_changed_training_protocol_is_rejected(self):
        verify_source_config(SOURCE_ROOT/'train/configs/train.yaml')
        with self.assertRaises(ValueError):
            verify_source_config(Path(__file__).resolve().parents[1]/'configs/paper-parameters.yaml')

    def test_predictor_matches_untouched_integrator(self):
        rng = np.random.default_rng(103)
        states = rng.normal(size=(12, 6))
        states[:, :2] *= 10
        states[:, 5] *= .15
        thrust = rng.uniform(-500., 1000., (12, 2))
        predicted = NominalModel().trajectories(states, thrust, 2.)
        for i, state in enumerate(states):
            boat = self.source.WAMV()
            boat.reset(state[:2].copy(), state[2])
            boat.velocity_r = state[3:].copy()
            boat.update_velocity(np.zeros(3))
            for step in range(10):
                boat.step(thrust[i], np.zeros(3))
                np.testing.assert_allclose(boat.pos, predicted[i, (step+1)*4], rtol=0., atol=1e-10)

    def test_baseline_never_calls_added_predictor(self):
        action = np.array([[.5, -.3, .19], [-.4, .6, .71]], dtype=np.float32)
        boids = np.array([[200., 450.], [980., -200.]])
        with patch.object(NominalModel, 'trajectories', side_effect=AssertionError('Baseline prediction')):
            actual, thrust, info = GateController('original').control(np.zeros((2, 6)), action, boids)
        np.testing.assert_array_equal(actual, action)
        np.testing.assert_array_equal(thrust, source_mixture(action, boids))
        self.assertFalse(info['warning'])

    def test_no_warning_keeps_original_command(self):
        state = np.array([[0., 0., 0., 0., 0., 0.], [100., 100., 0., 0., 0., 0.]])
        action = np.array([[.5, -.3, .19], [-.4, .6, .71]], dtype=np.float32)
        boids = np.array([[200., 450.], [980., -200.]])
        for method in ('reactive_independent','reactive_joint','predictive_independent','predictive_joint'):
            actual, thrust, info = GateController(method).control(state, action, boids)
            np.testing.assert_array_equal(actual, action)
            np.testing.assert_array_equal(thrust, source_mixture(action, boids))
            self.assertFalse(info['warning'])

    def test_reactive_method_does_not_roll_out_trajectories(self):
        state = np.array([[-3., 0., 0., 1., 0., 0.], [3., 0., np.pi, -1., 0., 0.]])
        action = np.array([[1., 1., .4], [1., 1., .6]], dtype=np.float32)
        boids = np.full((2, 2), -500.)
        with patch.object(NominalModel, 'trajectories', side_effect=AssertionError('Reactive rollout')):
            _, _, info = GateController('reactive_joint').control(state, action, boids)
        self.assertTrue(info['warning'])

    def test_candidate_original_is_exact(self):
        rng = np.random.default_rng(281)
        action = rng.uniform(-1., 1., (5, 3)).astype(np.float32)
        action[:, 2] = rng.uniform(0., 1., 5)
        boids = rng.uniform(-500., 1000., (5, 2))
        _, thrusts = GateController().candidates(action, boids)
        np.testing.assert_array_equal(thrusts[:, 0], source_mixture(action, boids))

    def test_baseline_full_trajectory_equals_plain_source_loop(self):
        from evaluate_gate_contribution import rollout, direct_original
        def policy(observation):
            return np.tile(np.array([.6, .1, .35], dtype=np.float32), (len(observation), 1))
        for n in (2, 4):
            scene = dict(cell='test', scene_seed=17561+n, noise_seed=28700+n, defenders=n, agility=2.25)
            actual = rollout(scene, 'original', policy)
            self.assertEqual(actual['trajectory_sha256'], direct_original(scene, policy))

    def test_cbf_and_gate_model_use_same_source_dynamics(self):
        from cbf_source_baseline import USVBarrierConfig
        import jax.numpy as jnp
        rng = np.random.default_rng(874)
        state = rng.normal(size=(3, 6))
        thrust = rng.uniform(-500., 1000., (3, 2))
        config = USVBarrierConfig(3)
        z = jnp.asarray(state.reshape(-1))
        derivative = np.asarray(config.f(z)+config.g(z) @ jnp.asarray(thrust.reshape(-1)/1000.)).reshape(3, 6)
        expected = NominalModel().acceleration(state, thrust)
        np.testing.assert_allclose(derivative[:, :3], state[:, 3:6], rtol=0., atol=1e-12)
        np.testing.assert_allclose(derivative[:, 3:6], expected, rtol=0., atol=1e-12)

    def test_official_cbf_filter_finite_and_bounded(self):
        from cbf_source_baseline import CBFController, dependency_provenance
        state = np.array([[-5., 0., 0., 1., 0., 0.], [5., 0., np.pi, -1., 0., 0.]])
        action = np.array([[1., 1., .4], [1., 1., .6]], dtype=np.float32)
        _, thrust, info = CBFController(2).control(state, action, np.full((2, 2), 900.))
        self.assertTrue(np.isfinite(thrust).all())
        self.assertLessEqual(thrust.max(), 1000.)
        self.assertGreaterEqual(thrust.min(), -500.)
        self.assertEqual(dependency_provenance()['versions']['cbfpy'], '0.0.4')
        self.assertLess(info['cbf_constraint_violation'], 1e-3)

    def test_joint_selection_dominates_unilateral_combination_on_same_objective(self):
        rng = np.random.default_rng(918)
        for risk in ('reactive', 'predictive'):
            state = rng.normal(size=(3, 6))
            state[:, :2] *= 4.
            state[:, 5] *= .1
            action = rng.uniform(-1., 1., (3, 3)).astype(np.float32)
            action[:, 2] = rng.uniform(0., 1., 3)
            boids = rng.uniform(-500., 1000., (3, 2))
            joint, independent = GateController(risk+'_joint'), GateController(risk+'_independent')
            tj, _ = joint.select(state, action, boids)
            ti, _ = independent.select(state, action, boids)
            options, thrusts = joint.candidates(action, boids)
            pairs, scores = joint.pair_scores(state, thrusts)
            deviation = np.mean(((thrusts-thrusts[:, :1])/1500.)**2, axis=-1)
            def objective(theta):
                indices = np.array([np.flatnonzero(options[i]==theta[i])[0] for i in range(3)])
                return max(scores[p, indices[i], indices[j]] for p, (i,j) in enumerate(pairs)) + .02*deviation[np.arange(3),indices].mean()
            self.assertLessEqual(objective(tj), objective(ti)+1e-12)
            self.assertLessEqual(objective(tj), objective(action[:, 2])+1e-12)


if __name__ == '__main__':
    unittest.main()
