"""Evidence integrity: physical policy equivalence, holdouts and independent units."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_eval import PhysicalPolicy, evaluate, ENVIRONMENT
from evidence_matrix import config_for, mask_motion, SEEDS, ARMS
from evidence_stats import exact_sign_flip, holm, paired_fixed
from RL.collector import CollisionStarts
from RL.Networks import ActorAdap
from RL.observations import tensor_frame
from train_mappo import make_env


class EvidenceTests(unittest.TestCase):
    def test_common_evaluator_batch_invariance_and_finite_horizon(self):
        policy = PhysicalPolicy("boids")
        a = evaluate(policy, [150, 151, 152], parallel=1, duration=.4)["rows"]
        b = evaluate(policy, [150, 151, 152], parallel=3, duration=.4)["rows"]
        self.assertEqual(a, b)
        self.assertEqual([r["steps"] for r in a], [2, 2, 2])
        self.assertTrue(all(r["outcome_code"] == 4 for r in a))

    def test_legacy_policy_uses_exact_physical_fusion_and_step_rewards(self):
        actor = ActorAdap(6, 8, 3, 32).eval()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            checkpoint = Path(directory) / "actor.pth"
            torch.save(actor.state_dict(), checkpoint)
            policy = PhysicalPolicy("arboids", checkpoint)
            np.random.seed(918)
            env = make_env(dict(environment=dict(ENVIRONMENT, total_time=.4)))
            obs, _ = env.reset(2., noisy_agility=False)
            rewards, steps = 0., 0
            while True:
                with torch.no_grad():
                    action = actor(torch.as_tensor(obs, dtype=torch.float32), True, False)[0].numpy()
                rng = np.random.get_state()
                frame = env.structured_frame()
                physical = policy.act([frame], [obs])[0]
                self.assertTrue(np.allclose(physical, env.action_to_thrust(action[:, :2]) * action[:, 2:3]
                                            + env.boids_actions * (1 - action[:, 2:3]), atol=2e-4))
                np.random.set_state(rng)
                obs, reward, outcome, _ = env.step(action, controller="AdaRes")
                rewards += float(np.mean(reward))
                steps += 1
                if outcome:
                    break
            result = evaluate(policy, [918], parallel=1, duration=.4)["rows"][0]
            self.assertAlmostEqual(result["reward"], rewards, places=4)
            self.assertEqual(result["steps"], steps)

    def test_no_motion_demonstrations_match_actual_no_motion_observations(self):
        env = make_env(dict(environment=ENVIRONMENT))
        env.reset()
        actual = tensor_frame(env.structured_frame(False))
        masked = mask_motion(tensor_frame(env.structured_frame(True)))
        for key in actual:
            self.assertTrue(torch.equal(actual[key], masked[key]), key)

    def test_training_never_resets_on_a_reserved_seed(self):
        seen = []
        starts = CollisionStarts(lambda: seen.append(np.random.get_state()[1][0]),
                                 probability=.5, excluded_seed_ranges=[(10, 20)])
        with patch("numpy.random.randint", side_effect=[15, 21]):
            _, seed, _ = starts.reset()
        self.assertEqual(seed, 21)
        self.assertEqual(seen, [21])
        with self.assertRaises(ValueError):
            CollisionStarts(lambda: None, state={"cases": [(12, 0)]}, excluded_seed_ranges=[(10, 20)])

    def test_controls_have_same_budget_and_only_declared_factors_change(self):
        full = config_for("full", SEEDS[0])
        self.assertEqual(len(set(SEEDS)), 8)
        for arm in ARMS:
            config = config_for(arm, SEEDS[0])
            for key in ("environment", "mappo", "training", "curriculum", "agent"):
                self.assertEqual(config[key], full[key])

    def test_inference_uses_training_seeds_and_checks_pairing(self):
        self.assertEqual(exact_sign_flip([0.] * 8), 1.)
        self.assertEqual(exact_sign_flip([-1.] * 8), 2 / 256)
        self.assertEqual(holm([.02, .01, .4]), [.04, .03, .4])
        with self.assertRaises(ValueError):
            paired_fixed([dict(seed=1, outcome_code=2, initial_sha256="a")],
                         [dict(seed=1, outcome_code=3, initial_sha256="b")], 2)


if __name__ == "__main__":
    unittest.main()
