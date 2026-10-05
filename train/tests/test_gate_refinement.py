"""Contracts for optional simulator demonstrations, separate from PPO."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RL.channel_networks import ChannelActor
from RL.observations import tensor_frame
from refine_actor import approaching_vessels, successful_corrections, collect_episodes
from test_channel_mappo import environment, permute_frame


class GateRefinementTests(unittest.TestCase):
    def test_collection_removes_finished_worlds_and_keeps_scenario_rng_independent(self):
        actor = ChannelActor(hidden=16)
        config = dict(agent=dict(defender_num=3), curriculum=dict(eva_agility=2.),
                      environment=dict(protocol="paper-parameters-v1", total_time=.4,
                                       initial_min_spacing=6., canonical_agent_order=True))
        serial = collect_episodes(actor, config, [11, 12, 13], num_envs=1)
        batched = collect_episodes(actor, config, [11, 12, 13], num_envs=2)
        self.assertEqual(sorted(row["seed"] for row in batched), [11, 12, 13])
        for before, after in zip(sorted(serial, key=lambda r: r["seed"]), sorted(batched, key=lambda r: r["seed"])):
            self.assertEqual(before["outcome"], after["outcome"])
            self.assertEqual(len(after["data"]), 2)
            for x, y in zip(before["data"], after["data"]):
                np.testing.assert_allclose(x["action"], y["action"], atol=.001, rtol=0.)

    def test_pair_priority_is_permutation_equivariant_and_handles_no_neighbors(self):
        frame = environment(3).structured_frame()
        frame["edges"][..., :4] = 0.
        frame["edges"][0, 1, :4] = [10 / 60, 0., -3 / 5, 0.]
        frame["edges"][1, 0, :4] = [-10 / 60, 0., 3 / 5, 0.]
        np.testing.assert_array_equal(approaching_vessels(frame), [True, True, False])
        permutation = [2, 0, 1]
        np.testing.assert_array_equal(approaching_vessels(permute_frame(frame, permutation)),
                                      approaching_vessels(frame)[permutation])
        frame["neighbors"][:] = False
        self.assertFalse(approaching_vessels(frame).any())

    def test_only_successful_matched_collision_corrections_are_accepted(self):
        before = [dict(seed=i, outcome=o) for i, o in enumerate([2, 2, 2, 3])]
        after = [dict(seed=i, outcome=o) for i, o in enumerate([3, 1, 2, 3])]
        self.assertEqual(successful_corrections(before, after), [after[0]])

    def test_gate_supervision_leaves_candidate_parameters_frozen(self):
        actor = ChannelActor(hidden=16)
        frame = tensor_frame(environment().structured_frame())
        with torch.no_grad():
            initial = actor.act(frame, deterministic=True)
        distribution, _ = actor.gate_distribution(frame, initial["candidate"], initial["messages"], detach_features=True)
        (distribution.mean.sigmoid() - .9).square().mean().backward()
        for component in (actor.encoder, actor.proposal_neighbors, actor.proposal_head):
            self.assertTrue(all(p.grad is None for p in component.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.gate_head.parameters()))


if __name__ == "__main__":
    unittest.main()
