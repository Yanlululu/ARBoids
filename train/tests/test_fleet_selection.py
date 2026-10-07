"""Contracts for the new penalty and the newly included six-boat dimension."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np

from feedback_joint_control import FeedbackJointController, NominalEnvironment, observe, snapshot
from feedback_joint_fast import FastFeedbackJointController
from fleet_selection_control import FleetSelectionController, penalize_prediction


def policy(obs):
    values = np.empty((len(obs), 3), dtype=np.float32)
    values[:, 0] = .7
    values[:, 1] = .2 + .15 * np.tanh(obs[:, 3])
    values[:, 2] = .45 + .15 * np.tanh(obs[:, 2] / 30.)
    return values


class FleetSelectionContracts(unittest.TestCase):
    def scene(self):
        np.random.seed(4419)
        env = NominalEnvironment(defender_num=6)
        obs, _ = env.reset(agility=2.25, noisy_agility=False)
        return env, obs

    def test_penalty_requires_improvement_without_changing_outcome_priority(self):
        base = dict(template='baseline', score=(0, 0, 0, .3))
        close = dict(template='more_boids', score=(0, 0, 0, .29))
        better = dict(template='more_learned', score=(0, 0, -1, .9))
        scored = [penalize_prediction(x, .03) for x in (base, close, better)]
        self.assertLess(scored[0]['score'], scored[1]['score'])
        self.assertLess(scored[2]['score'], scored[0]['score'])
        self.assertEqual(base['score'], scored[0]['score'])
        self.assertEqual(close['score'][-1], .29)

    def test_invalid_penalty_is_rejected(self):
        for penalty in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                FleetSelectionController(6, policy, departure_penalty=penalty)

    def test_zero_penalty_matches_frozen_fast_controller(self):
        env, obs = self.scene()
        reference = FastFeedbackJointController(6, policy, 5, force_planning=True)
        wrapped = FleetSelectionController(6, policy, 5, 0., force_planning=True)
        for _ in range(11):
            a, ia = reference.control(observe(env, obs))
            b, ib = wrapped.control(observe(env, obs))
            np.testing.assert_array_equal(a, b)
            self.assertEqual(ia['template'], ib['template'])
            np.testing.assert_array_equal(reference.plan['positions'], wrapped.plan['positions'])
            obs, _, _, _ = env.step(env.thrust_to_action(a), 'RL')

    def test_six_boat_prediction_matches_reference_and_physical_execution(self):
        for steps in (5, 10):
            env, obs = self.scene()
            reference = FeedbackJointController(6, policy, steps, force_planning=True)
            fast = FleetSelectionController(6, policy, steps, .01, force_planning=True)
            state = observe(env, obs)
            for template in ('baseline', 'more_learned', 'more_boids', 'interceptor_learned', 'interceptor_boids'):
                a, b = reference.predict(state, template), fast.predict(state, template)
                np.testing.assert_allclose(a['positions'], b['positions'], atol=1e-6, rtol=0.)
                np.testing.assert_allclose(a['commands'], b['commands'], atol=.002, rtol=0.)
            for tick in range(3 * steps):
                command, info = fast.control(observe(env, obs))
                expected = fast.plan['commands'][info['plan_offset']]
                np.testing.assert_allclose(command, expected, atol=.002, rtol=0.)
                obs, _, done, _ = env.step(env.thrust_to_action(command), 'RL')
                np.testing.assert_allclose(snapshot(env)[:, :3], info['predicted_next_state'][:, :3],
                                           atol=1e-6, rtol=0.)
                if done:
                    break
            self.assertEqual(fast.plan_count, 3)


if __name__ == '__main__':
    unittest.main()
