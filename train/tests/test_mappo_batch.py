"""Independent-world RNG, prediction and on-policy likelihoods under batching."""
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parallel_mappo
from parallel_mappo import restore_branch_start
from policy.interaction_prediction import InteractionPredictor, PredictionConfig, exchange_candidates
from policy.mappo import PredictiveMAPPO
from policy.rollout_buffer import RolloutBuffer
from train_mappo import make_env, play_episode, play_episodes_batched


def configuration(theta_space=False):
    return dict(agent=dict(defender_num=3, boid_state=True, form_reward=True),
                environment=dict(protocol='paper-parameters-v1', total_time=1.),
                mappo=dict(hidden_dim=32, relation_dim=8, coordination_dim=4,
                           theta_space_gate=theta_space, minibatch_size=8, ppo_epochs=1),
                prediction=dict(horizon=.1))


class BatchedSamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        np.random.seed(71)

    def test_prediction_worlds_are_independent_and_match_serial(self):
        predictor = InteractionPredictor(PredictionConfig(horizon=.4))
        rng = np.random.default_rng(49)
        messages = [exchange_candidates(rng.normal(size=(3, 6)), rng.uniform(-1, 1, (3, 2)),
                                       rng.uniform(-1, 1, (3, 2)), i) for i in range(5)]
        edges, masks = predictor.features_batch(messages)
        for i, message in enumerate(messages):
            expected, mask = predictor.features(message)
            np.testing.assert_allclose(edges[i], expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(masks[i], mask)
        changed = [*messages]
        changed[0] = exchange_candidates(np.ones((3, 6)), np.ones((3, 2)), -np.ones((3, 2)), 0.)
        actual, _ = predictor.features_batch(changed)
        np.testing.assert_array_equal(actual[1:], edges[1:])

    def test_single_slot_preserves_the_existing_seeded_sampling_path(self):
        for theta_space in (False, True):
            with self.subTest(theta_space=theta_space):
                agent = PredictiveMAPPO(configuration(theta_space))
                seeds = [701001, 701002]
                expected = []
                for seed in seeds:
                    np.random.seed(seed)
                    random.seed(seed)
                    torch.manual_seed(seed)
                    expected.append(play_episode(agent, make_env(agent.config), 2., True))
                actual = play_episodes_batched(agent, seeds, 2., True, batch_size=1)
                for (original, original_summary), (batched, summary) in zip(expected, actual):
                    self.assertEqual(original_summary, {k: v for k, v in summary.items() if k != 'seed'})
                    for a, b in zip(original, batched):
                        for key in a:
                            np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_multiple_slots_preserve_rng_prefixes_and_recomputable_likelihoods(self):
        agent = PredictiveMAPPO(configuration(True))
        seeds = [702001, 702002, 702003]
        global_rng = torch.get_rng_state().clone()
        episodes = play_episodes_batched(agent, seeds, 2., True, batch_size=2)
        torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
        reordered = play_episodes_batched(agent, seeds[::-1], 2., True, batch_size=3)[::-1]
        buffer = RolloutBuffer()
        for (episode, summary), (reference, other_summary) in zip(episodes, reordered):
            self.assertEqual(summary['seed'], other_summary['seed'])
            self.assertEqual(summary['outcome_code'], other_summary['outcome_code'])
            self.assertEqual(len(episode), len(reference))
            for a, b in zip(episode, reference):
                for key in ('proposal_raw', 'gate_raw', 'state', 'executed_action'):
                    np.testing.assert_allclose(a[key], b[key], rtol=2e-5, atol=2e-5, err_msg=key)
            # The curriculum must still restore the original physical state
            # exactly, even though network matmuls have batched rounding.
            start = 2
            case = dict(source_seed=summary['seed'], prefix=[t['executed_action'] for t in episode[:start]],
                        state=episode[start]['state'], obs=episode[start]['obs'])
            restored = restore_branch_start(make_env(agent.config), case, 2., True)
            np.testing.assert_array_equal(restored.astype(np.float32), episode[start]['obs'])
            buffer.add_episode(episode, summary)
        batch = buffer.tensors(agent.settings, 'cpu')
        lp, lg, _ = agent.evaluate_batch(batch)
        torch.testing.assert_close(lp, batch['old_logp_l'], rtol=2e-5, atol=1e-4)
        torch.testing.assert_close(lg, batch['old_logp_g'], rtol=2e-5, atol=1e-4)

    def test_deterministic_checks_always_use_the_original_path(self):
        args = ({}, {}, {}, [700001], 2., False, True, True, 4)
        with patch.object(parallel_mappo, '_load_policy'), \
                patch.object(parallel_mappo, 'play_episodes_batched') as batched, \
                patch.object(parallel_mappo, 'play_episode', return_value=([], {})) as serial:
            result = parallel_mappo._episodes(args)
        batched.assert_not_called()
        serial.assert_called_once()
        self.assertEqual(result[0][1]['seed'], 700001)


if __name__ == '__main__':
    unittest.main()
