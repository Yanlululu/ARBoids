"""Causal/replay checks for frozen-candidate gate interventions."""
from pathlib import Path
import pickle
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.TADgame import TADEnv
from envs.snapshot import RandomState, SimulationSnapshot, seed_random
from gate_diagnostics import (GateCase, collect_cases, corrected_gates, diagnose_case,
                              gate_options, minimum_distance, predict_options, run_branch,
                              select_candidate, summarize_cases, trajectory_digest)
from policy.interaction_prediction import InteractionPredictor


def policy(observation):
    return np.tile(np.array([-.35, .65, .43], dtype=np.float32), (len(observation), 1))


def make_case(seed=36, total_time=.8, prefix=1):
    seed_random(seed)
    env = TADEnv(protocol='paper-parameters-v1', total_time=total_time)
    obs, _ = env.reset()
    for _ in range(prefix):
        obs, _, done, _ = env.step(policy(obs), 'AdaRes')
        assert not done
    case = GateCase(seed, prefix, 'normal', SimulationSnapshot.capture(env), obs.copy(), policy(obs))
    done = 0
    while not done:
        obs, _, done, _ = env.step(policy(obs), 'AdaRes')
    case.source_trajectory_sha256 = trajectory_digest(env)
    return case, env


def assert_nested_equal(test, left, right):
    if isinstance(left, np.ndarray):
        test.assertEqual(left.dtype, right.dtype)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(test, a, b)
    elif hasattr(left, '__dict__'):
        test.assertIs(type(left), type(right))
        assert_nested_equal(test, vars(left), vars(right))
    else:
        test.assertEqual(left, right)


class CollisionAtDeadline(TADEnv):
    """Controlled terminal event for testing collision-lead sample indexing."""
    def _isTerminate(self):
        return 2 if self.Current_T >= .8 - 1e-8 else 0

    def physical_events(self):
        return dict(collision=self.Current_T >= .8 - 1e-8, breach=False, capture=False)


class GateDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_snapshot_roundtrip_restores_every_field_and_rng(self):
        case, _ = make_case()
        snapshot = pickle.loads(pickle.dumps(case.snapshot))
        first = snapshot.restore()
        obs1, reward1, done1, _ = first.step(case.action, 'AdaRes')
        draws1 = (np.random.random(), random.random(), torch.rand(3))
        second = snapshot.restore()
        obs2, reward2, done2, _ = second.step(case.action, 'AdaRes')
        draws2 = (np.random.random(), random.random(), torch.rand(3))
        assert_nested_equal(self, first, second)
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(reward1, reward2)
        self.assertEqual(done1, done2)
        self.assertEqual(draws1[:2], draws2[:2])
        torch.testing.assert_close(draws1[2], draws2[2], rtol=0, atol=0)
        first.defender_list[0].pos[:] = 999
        first.boids_actions[:] = 0
        restored = snapshot.restore()
        assert_nested_equal(self, restored, case.snapshot.environment)

    def test_zero_correction_reproduces_original_full_trajectory(self):
        case, original = make_case()
        zero = corrected_gates(case.theta, np.zeros_like(case.theta))
        np.testing.assert_array_equal(zero, case.theta)
        result = run_branch(case, policy, zero)
        self.assertEqual(result['trajectory_sha256'], trajectory_digest(original))
        self.assertEqual(result['outcome_code'], original._isTerminate())

    def test_actual_collision_penalty_is_in_the_team_return(self):
        seed_random(18)
        env = TADEnv(protocol='paper-parameters-v1', total_time=.4)
        env.reset()
        for boat, pos, velocity in zip(env.defender_list,
                ((0., 0.), (5.3, 0.), (0., 20.)), ((2., 0., 0.), (-2., 0., 0.), (0., 0., 0.))):
            boat.reset(np.array(pos), 0.)
            boat.velocity_r = np.array(velocity)
        env.boids_actions[:] = 0.
        observation, _ = env._get_obs()
        self.assertGreater(minimum_distance(env), env.Collision_R)
        case = GateCase(18, 0, 'collision', SimulationSnapshot.capture(env), observation, policy(observation))
        _, rewards, done, _ = env.step(case.action, 'AdaRes')
        self.assertEqual(done, 2)
        main, formation, collision = env.paper_reward_components()
        self.assertLess(collision.mean(), 0.)
        result = run_branch(case, policy, case.theta)
        self.assertEqual(result['steps'], 1)
        self.assertEqual(result['collision'], 1)
        self.assertEqual(result['team_return'], float(rewards.mean()))
        self.assertAlmostEqual(result['team_return'], float((main + formation + collision).mean()))

    def test_branch_order_and_outer_rng_are_unchanged(self):
        case, _ = make_case()
        outer = RandomState.capture()
        a = run_branch(case, policy, np.zeros(3), future_seed=9)
        run_branch(case, policy, np.ones(3), future_seed=10)
        b = run_branch(case, policy, np.zeros(3), future_seed=9)
        self.assertEqual(a, b)
        actual = (np.random.random(), random.random(), torch.rand(3))
        outer.restore()
        expected = (np.random.random(), random.random(), torch.rand(3))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_intervention_is_one_cycle_and_attacker_uses_each_branch_scene(self):
        case, _ = make_case(total_time=1.2, prefix=0)
        original_step, original_current = TADEnv.step, TADEnv.generate_random_current
        original_apf = TADEnv._APF_navi_step
        traces, currents, scenes = [], [], []
        def track(env, action, controller='Boids', att_action=None):
            self.assertIsNone(att_action)
            result = original_step(env, action, controller, att_action)
            traces.append((action.copy(), env.att_action.copy()))
            return result
        def current(env):
            value = original_current(env)
            currents.append(value.copy())
            return value
        def apf(env, position, goal, obstacles, phi, robot='att'):
            scenes.append(np.array([obstacle.pos for obstacle in obstacles]).copy())
            return original_apf(env, position, goal, obstacles, phi, robot)
        with (patch.object(TADEnv, 'step', track), patch.object(TADEnv, 'generate_random_current', current),
              patch.object(TADEnv, '_APF_navi_step', apf)):
            run_branch(case, policy, np.zeros(3), future_seed=20)
            first, first_currents, first_scenes = traces[:], currents[:], scenes[:]
            traces.clear()
            currents.clear()
            scenes.clear()
            run_branch(case, policy, np.ones(3), future_seed=20)
        np.testing.assert_array_equal(first[0][0][:, :2], case.action[:, :2])
        np.testing.assert_array_equal(first[0][0][:, 2], np.zeros(3))
        for action, _ in first[1:]:
            np.testing.assert_array_equal(action, case.action)
        np.testing.assert_array_equal(first_currents, currents)
        np.testing.assert_array_equal(first[0][1], traces[0][1])
        self.assertEqual(len(first_scenes), len(first))
        np.testing.assert_array_equal(first_scenes[0], scenes[0])
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(first_scenes[1:], scenes[1:])))

    def test_five_point_grid_includes_exact_original_and_limits_growth(self):
        theta = np.array([.1, .2, .3], dtype=np.float32)
        options = gate_options(theta)
        self.assertEqual(options.shape, (126, 3))
        np.testing.assert_array_equal(options[0], theta)
        self.assertEqual(options.dtype, theta.dtype)
        self.assertEqual(len(gate_options(np.zeros(3, dtype=np.float32))), 125)
        with self.assertRaisesRegex(ValueError, 'limit'):
            gate_options(np.zeros(10, dtype=np.float32))

    def test_prediction_uses_actual_thrust_units_and_does_not_change_state(self):
        case, _ = make_case()
        options = gate_options(case.theta, (0., 1.))
        predictor_step = InteractionPredictor.trajectories
        captured = []
        def track(predictor, states, thrusts):
            captured.append(thrusts.copy())
            return predictor_step(predictor, states, thrusts)
        before = pickle.dumps(case.snapshot.environment)
        with patch.object(InteractionPredictor, 'trajectories', track):
            predictions = predict_options(case, options, horizon=.4, safe_distance=7.)
        env = case.snapshot.restore()
        env.step(case.action, 'AdaRes')
        actual = np.array([[boat.left_thrust, boat.right_thrust] for boat in env.defender_list])
        np.testing.assert_array_equal(np.clip(captured[0][:3], -500., 1000.), actual)
        self.assertEqual(before, pickle.dumps(case.snapshot.environment))
        self.assertEqual(len(predictions), len(options))

    def test_search_has_paired_reference_and_separate_validation(self):
        case, _ = make_case(total_time=.6)
        summary, rows = diagnose_case(case, policy, [101, 102], [201, 202], grid=(0., 1.), horizon=.4)
        self.assertTrue(summary['source_replay_verified'])
        self.assertEqual(summary['options'], 9)
        self.assertEqual(len(rows), 9 * 2 + 2 * 2)
        for row in rows:
            reference = next(r for r in rows if r['phase'] == row['phase'] and
                             r['continuation_seed'] == row['continuation_seed'] and r['role'] == 'reference')
            self.assertEqual(row['team_return_delta'], row['team_return'] - reference['team_return'])
        self.assertEqual({r['candidate_id'] for r in rows if r['phase'] == 'validation'},
                         {0, summary['selected_candidate_id']})
        self.assertEqual({r['continuation_seed'] for r in rows if r['phase'] == 'validation'}, {201, 202})
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            diagnose_case(case, policy, [1], [1])

    def test_source_mismatch_is_rejected(self):
        case, _ = make_case()
        case.source_trajectory_sha256 = 'incorrect'
        with self.assertRaisesRegex(ValueError, 'source trajectory'):
            diagnose_case(case, policy, [1], [2], grid=(0., 1.))

    def test_selection_rejects_collision_to_breach_trade(self):
        def row(success, collision, breach):
            return dict(success=success, collision=collision, breach=breach,
                        team_return=0., short_min_distance=10.)
        options = np.array([[.5, .5], [0., 0.], [1., 1.]])
        averages = [row(.5, .5, 0.), row(.5, 0., .5), row(1., 0., 0.)]
        self.assertEqual(select_candidate(averages, options), 2)
        self.assertEqual(select_candidate(averages[:2], options[:2]), 0)

    def test_collection_samples_success_and_preserves_source_trajectories(self):
        factory = lambda: TADEnv(protocol='paper-parameters-v1', total_time=.6)
        cases, sources = collect_cases(policy, [30, 31, 32], env_factory=factory,
                                       collision_episodes=0, normal_episodes=2)
        self.assertEqual(len(cases), 2)
        for case in cases:
            self.assertEqual(case.kind, 'normal')
            self.assertEqual(run_branch(case, policy, case.theta)['trajectory_sha256'], case.source_trajectory_sha256)
        for row in sources:
            _, original = make_case(row['source_seed'], total_time=.6, prefix=0)
            self.assertEqual(trajectory_digest(original), row['trajectory_sha256'])

    def test_collision_lookback_uses_time_before_terminal_event(self):
        cases, _ = collect_cases(policy, [30],
            env_factory=lambda: CollisionAtDeadline(protocol='paper-parameters-v1', total_time=.8),
            lookbacks=(.2, .4), collision_episodes=1, normal_episodes=0)
        self.assertEqual([case.source_step for case in cases], [2, 3])
        for case in cases:
            self.assertAlmostEqual(case.snapshot.environment.Current_T + case.lookback_seconds, .8)
            self.assertEqual(case.kind, 'collision')

    def test_summary_weights_source_episodes_not_number_of_states(self):
        def row(seed, delta):
            return dict(kind='collision', source_seed=seed, search_observed_improvement=False,
                        validation_observed_improvement=False,
                        **{f'validation_{key}_delta': delta for key in
                           ('success', 'collision', 'breach', 'team_return', 'short_min_distance')})
        result = summarize_cases([row(1, 0.), row(1, 0.), row(2, 1.)])['collision']
        self.assertEqual(result['source_episodes'], 2)
        self.assertEqual(result['source_mean_validation_deltas']['collision'], .5)


if __name__ == '__main__':
    unittest.main()
