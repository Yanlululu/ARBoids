import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from citf.bridge import World, factorize, inputs
from citf.collect import sample_root
from citf.contracts import Protocol
from citf.opponents import APFOpponent
from citf.risk import pair_indices
from citf.search import candidate_grid
from experiments.branch_diagnosis import (candidate_pool, feedback_distances,
                                         score_comparison, third_vessel_effect)


PROJECT = Path(__file__).resolve().parents[2] / 'source_archive'


class FixedSource:
    sha256 = 'fixed-source'; implementation_sha256 = 'fixed-implementation'

    def commands(self, env):
        n = env.defender_num
        return torch.full((n, 2), .6), torch.full((n, 2), .2), torch.full((n, 2), .5)


class BranchDiagnosis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pool_keeps_actual_search_and_deduplicates(self):
        prior = torch.full((6, 2), .5); grid, valid = candidate_grid(prior)
        f = dict(prior=prior, grid=grid, valid=valid,
                 unary=((grid - prior[:, None]) / .4).square().mean(-1),
                 pair=torch.zeros(6, 6, 9, 9))
        pool, meta = candidate_pool(f, Protocol(), 96, np.random.default_rng(17))
        other, _ = candidate_pool(f, Protocol(), 96, np.random.default_rng(17))
        np.testing.assert_array_equal(pool, other)
        self.assertEqual(len({tuple(x) for x in pool}), 96)
        self.assertIn(tuple(meta['source_center']), {tuple(x) for x in pool})
        self.assertTrue({tuple(x) for x in meta['physics_search']['configurations']} <= {tuple(x) for x in pool})
        with self.assertRaises(ValueError):
            candidate_pool(f, Protocol(), 1, np.random.default_rng(17))

    def test_crossfit_does_not_reuse_optimistic_oracle_future(self):
        # Candidate 1 wins only in-sample: it looks perfect in one future and
        # collides in the other. Cross-fitting must expose the negative gain.
        plans = np.array([[0, 0], [1, 1]])
        real = np.array([[[6.], [6.]], [[7.], [5.]]])
        metric = score_comparison(plans, np.zeros(2), np.array([[7.], [6.]]),
                                  np.array([[7.], [6.]]), real, weight=1.)
        self.assertEqual(metric['nominal']['regret'], 0.)
        self.assertLess(metric['oracle_leave_one_future_out_gain'], 0.)

    def test_third_vessel_intervention_preserves_pair_candidates(self):
        plans = np.array([[4, 4, 4], [4, 4, 0]])
        nominal = np.array([[6., 8., 9.], [6., 7., 8.]])
        feedback = nominal.copy(); feedback[1, 0] = 5.
        real = np.repeat(feedback[:, None], 3, axis=1)
        metric = third_vessel_effect(plans, nominal, feedback, real)
        self.assertEqual(metric['matched_pair_interventions'], 1)
        self.assertEqual(metric['nominal_mean_abs_change_m'], 0.)
        self.assertEqual(metric['real_mean_abs_change_m'], 1.)

    def test_feedback_uses_observable_velocity_and_preserves_root(self):
        world = World(PROJECT, Protocol(prediction_steps=1), 3)
        world.reset(71001, 81001, 2.); source = FixedSource()
        f = factorize(world, source); plans = np.array([[4, 4, 4], [0, 8, 4]])
        before = world.state().copy(); rng = world.rng.get_state()
        distance, steps = feedback_distances(world, source, f, plans, APFOpponent())
        b, _, _ = inputs(world, f, plans); i, j = pair_indices(3)
        expected = (b['nominal'][0, :, i] - b['nominal'][0, :, j]).norm(dim=-1).amin(-1).numpy()
        np.testing.assert_allclose(distance, expected, atol=2e-6)
        self.assertEqual(steps, 2); np.testing.assert_array_equal(world.state(), before)
        self.assertEqual(world.env.Current_T, 0.); np.testing.assert_array_equal(world.rng.get_state()[1], rng[1])
        altered = world.clone()
        for boat in [*altered.env.defender_list, altered.env.attacker]:
            boat.velocity_r = boat.velocity_r + np.array([10., -8., 2.])
            boat.current_velocity = boat.current_velocity - np.array([10., -8., 2.])
        other, _ = feedback_distances(altered, source, f, plans, APFOpponent())
        np.testing.assert_array_equal(distance, other)

    def test_resource_stop_prevents_uncommitted_real_branches(self):
        world = World(PROJECT, Protocol(prediction_steps=1), 3); world.reset(71001, 81001, 2.)
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp) / 'stop'; stop.touch()
            with self.assertRaises(InterruptedError):
                sample_root(world, FixedSource(), np.array([[4, 4, 4]]), [1, 2], {}, stop_file=stop)
        self.assertEqual(world.env.Current_T, 0.)


if __name__ == '__main__':
    unittest.main()
