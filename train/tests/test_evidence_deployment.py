"""The link audit must preserve ideal actions and cannot deliver future packets."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evidence_deployment import CandidateLink, evaluate_messages
from evidence_eval import PhysicalPolicy, evaluate
from RL.channel_networks import ChannelActor


class DeploymentEvidenceTests(unittest.TestCase):
    def test_ideal_link_matches_original_physical_policy(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        policy = PhysicalPolicy('boids')
        policy.kind, policy.actor = 'channel', ChannelActor(hidden=32).eval()
        original = evaluate(policy, [311, 312], parallel=1, duration=.4)['rows']
        linked = evaluate_messages(policy, [311, 312], duration=.4)['rows']
        for a, b in zip(original, linked):
            for key in ('initial_sha256', 'outcome_code', 'steps', 'success'):
                self.assertEqual(a[key], b[key])
            for key in ('reward', 'min_spacing', 'thrust_squared_integral'):
                self.assertAlmostEqual(a[key], b[key], places=6, msg=key)

    def test_delayed_packets_use_generation_age_and_never_future_values(self):
        link = CandidateLink(0., 2, seed=19)
        for step in range(1, 5):
            packet = torch.full((1, 3, 3, 5), float(step))
            actual = link.transmit(packet)
            self.assertTrue(torch.all(actual[..., :4] == max(0, step - 2)))
            self.assertTrue(torch.allclose(actual[..., 4], torch.full((1, 3, 3), min(step, 2) * .2)))


if __name__ == '__main__':
    unittest.main()
