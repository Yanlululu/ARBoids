"""Physics parity and independent random streams for the learned-opponent audit."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_eval import PhysicalPolicy, evaluate
from evidence_adversary import evaluate_duel, load_attacker
from policy.networks import ActorAtt


class AdversarialEvidenceTests(unittest.TestCase):
    def test_attacker_checkpoint_loading_preserves_actions(self):
        original = ActorAtt(2, 6, 2, 32).eval()
        observation = torch.randn(5, 8)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'attacker.pth'
            torch.save(original.state_dict(), path)
            restored = load_attacker(path)
        self.assertTrue(torch.equal(original(observation, True, False)[0], restored(observation, True, False)[0]))

    def test_apf_physics_and_defender_reward_are_unchanged(self):
        policy = PhysicalPolicy('boids')
        standard = evaluate(policy, [150, 151, 152], parallel=3, duration=.4)['rows']
        duel = evaluate_duel(policy, None, [150, 151, 152], parallel=1, duration=.4)['rows']
        for a, b in zip(standard, duel):
            for key in ('seed', 'outcome_code', 'success', 'steps', 'initial_sha256'):
                self.assertEqual(a[key], b[key], key)
            for key in ('reward', 'min_spacing', 'thrust_squared_integral'):
                self.assertAlmostEqual(a[key], b[key], places=6, msg=key)

    def test_learned_attacker_does_not_couple_world_random_streams(self):
        torch.set_num_threads(1)
        torch.manual_seed(81)
        attacker = ActorAtt(2, 6, 2, 32).eval()
        policy = PhysicalPolicy('boids')
        a = evaluate_duel(policy, attacker, [915, 916, 917], parallel=1, duration=.4)['rows']
        b = evaluate_duel(policy, attacker, [915, 916, 917], parallel=3, duration=.4)['rows']
        for x, y in zip(a, b):
            self.assertEqual(x['initial_sha256'], y['initial_sha256'])
            self.assertEqual(x['outcome_code'], y['outcome_code'])
            self.assertAlmostEqual(x['attacker_reward'], y['attacker_reward'], places=5)
            self.assertAlmostEqual(x['reward'], y['reward'], places=5)


if __name__ == '__main__':
    unittest.main()
