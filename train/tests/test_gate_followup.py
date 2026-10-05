"""Matched-action forecasts, finite rolling windows and paired noise checks."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.TADgame import TADEnv
from envs.snapshot import seed_random
from gate_diagnostics import align_environment_noise, gate_options, run_branch
from gate_followup import (choose_rolling_gate, held_forecasts, prefix_metrics,
                           run_controlled, nominal_environment)
from test_gate_diagnostics import make_case, policy


class GateFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_nominal_held_model_matches_same_control_endpoint_trajectory(self):
        case, _ = make_case(total_time=2.)
        options = gate_options(case.theta, (0., 1.))
        paths, _, thrusts = held_forecasts(case.snapshot.environment, case.action, options, 1.)
        for i in (0, 1, 8):
            trace = run_controlled(case, policy, mode='hold', theta=options[i], nominal=True,
                                   active_steps=5, stop_after=5)
            metrics = prefix_metrics(paths[i], trace)
            self.assertLess(metrics['maximum_position_error'], 1e-9)
            np.testing.assert_array_equal(trace.thrusts, np.tile(thrusts[i], (len(trace.thrusts), 1, 1)))

    def test_pulse_forecast_rejoins_baseline_instead_of_holding(self):
        case, _ = make_case(total_time=1.4)
        trace = run_controlled(case, policy, mode='pulse', theta=np.ones(3), nominal=True, stop_after=5)
        np.testing.assert_array_equal(trace.gates[0], np.ones(3))
        np.testing.assert_array_equal(trace.gates[1:], np.tile(case.theta, (len(trace.gates)-1, 1)))
        baseline = run_controlled(case, policy, nominal=True)
        zero = run_controlled(case, policy, mode='pulse', theta=case.theta, nominal=True)
        self.assertEqual(baseline.summary['trajectory_sha256'], zero.summary['trajectory_sha256'])

    def test_no_override_matches_legacy_branch_with_absolute_noise_pairing(self):
        case, _ = make_case()
        old = run_branch(case, policy, case.theta, future_seed=99, align_future_step=True)
        new = run_controlled(case, policy, future_seed=99).summary
        self.assertEqual(old['trajectory_sha256'], new['trajectory_sha256'])

    def test_absolute_noise_alignment_matches_later_steps_of_same_stream(self):
        case, _ = make_case()
        env = case.snapshot.environment
        def draw():
            return np.concatenate([env._apf_force_noise(), *[env.generate_random_current() for _ in range(4)]])
        seed_random(91)
        draws = [draw() for _ in range(6)]
        seed_random(91)
        align_environment_noise(env, 5)
        np.testing.assert_array_equal(draw(), draws[5])

    def test_rolling_replans_and_stops_exactly_at_active_window(self):
        case, _ = make_case(total_time=1.6)
        observed = []
        def choose(env, action, **kwargs):
            observed.append(env.Current_T)
            return np.full(3, .9), dict(warning=True)
        with patch('gate_followup.choose_rolling_gate', choose):
            trace = run_controlled(case, policy, mode='rolling', active_steps=3, future_seed=5)
        self.assertEqual(len(observed), 3)
        np.testing.assert_allclose(np.diff(observed), .2)
        np.testing.assert_allclose(trace.gates[:3], .9)
        np.testing.assert_array_equal(trace.gates[3:], np.tile(case.theta, (len(trace.gates)-3, 1)))

    def test_no_prediction_warning_preserves_original_gate(self):
        case, _ = make_case()
        env = nominal_environment(case.snapshot)
        # A zero-length forecast is invalid; a brief valid forecast at these
        # well-separated starting positions must leave the original gate exact.
        theta, info = choose_rolling_gate(env, case.action, horizon=.2, safe_distance=5.01)
        self.assertFalse(info['warning'])
        np.testing.assert_array_equal(theta, case.theta)

    def test_prefix_comparison_does_not_score_unobserved_future(self):
        case, _ = make_case(total_time=.6)
        paths, _, _ = held_forecasts(case.snapshot.environment, case.action, case.theta[None], 2.)
        trace = run_controlled(case, policy, mode='hold', active_steps=10, nominal=True, stop_after=10)
        metrics = prefix_metrics(paths[0], trace)
        self.assertLess(metrics['common_prefix_seconds'], 2.)
        self.assertLess(metrics['maximum_position_error'], 1e-9)


if __name__ == '__main__':
    unittest.main()
