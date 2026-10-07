"""Integration checks for real CBF transitions, replay storage, and resumption."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'train'))
import study_runtime
import numpy as np
import torch
import yaml

from envs.TADgame import TADEnv
from envs.snapshot import RandomState, seed_random
from interaction_rollout import (public_packet, execute, safety_controller, SnapshotPool,
                                 InterventionSampler, DeploymentPolicy)
from policy.interaction_sac import JointReplay, make_agent
from train_interaction import arm_config, run_training, export_policy
from interaction_review import check_endpoint, validation_calibration


def small_config(arm='full'):
    c = arm_config(yaml.safe_load((ROOT/'train/configs/interaction-aware-sac.yaml').read_text()), arm)
    c['rl'].update(hidden_dim=32, batch_size=4)
    c['training'].update(gate_steps=6, joint_steps=6, warm_steps=2, replay_capacity=16,
        eval_interval=12, eval_episodes=1, checkpoint_interval=6, log_interval=12,
        allow_random_initialization=True)
    c['interaction'].update(relation_dim=16, workers=0, pairs_per_batch=2, horizon_steps=2,
        intervention_interval=4, snapshot_interval=2, snapshot_capacity=8)
    return c


def equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a: equal(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b): equal(test, x, y)
    else:
        test.assertEqual(a, b)


class Pipeline(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_random(42)

    def test_cold_cbf_does_not_consume_training_rng(self):
        safety_controller.cache_clear()
        state = RandomState.capture()
        safety_controller(3)
        actual = np.random.rand(5), torch.randn(5)
        state.restore()
        equal(self, actual, (np.random.rand(5), torch.randn(5)))

    def test_lossless_replay_ring_and_reset_boundaries(self):
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25)
        packet = public_packet(env)
        for capacity in (4, 20):
            replay = JointReplay(capacity)
            for i in range(8):
                following = {k: v + i for k, v in packet.items()}
                replay.store(packet, np.zeros((3,3)), np.zeros((3,2)), np.ones(3), following, i%3==0)
                packet = following if i%3 else {k: v*2 for k,v in following.items()}
            restored = JointReplay(1)
            restored.load_state_dict(replay.state_dict())
            for key in replay.arrays:
                equal(self, replay.arrays[key][:replay.size], restored.arrays[key][:restored.size])

    def test_exact_resume_including_stage_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'continuous', Path(directory)/'resumed'
            c = small_config()
            run_training(c, output=a)
            run_training(c, output=b, max_steps=3)
            run_training(c, output=b, stop_after_stage='gate')
            gate = torch.load(b/'gate-endpoint.pth', weights_only=True)
            self.assertEqual((gate['step'],gate['stage']), (6,'gate'))
            again = run_training(c, output=b, stop_after_stage='gate')
            self.assertEqual(again['step'], 6)
            before = (b/'gate-endpoint.pth').read_bytes()
            safety_controller.cache_clear()
            run_training(c, output=b)
            self.assertEqual((b/'gate-endpoint.pth').read_bytes(), before)
            with self.assertRaises(ValueError):
                run_training(c, output=b, stop_after_stage='gate')
            left = torch.load(a/'resume.pth', weights_only=False)
            right = torch.load(b/'resume.pth', weights_only=False)
            for key in ('agent', 'replay', 'auxiliary', 'step', 'done', 'simulated_steps'):
                equal(self, left[key], right[key])
            for key in ('numpy','python','torch_cpu','torch_cuda'):
                equal(self, getattr(left['random'],key), getattr(right['random'],key))

    def test_review_detects_changed_frozen_proposals(self):
        agent = make_agent(small_config())
        source = {k:v.clone() for k,v in agent.actor.base.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            export_policy(agent,directory,6,endpoint=True)
            saved = torch.load(Path(directory)/'gate-endpoint.pth',weights_only=True)
            self.assertTrue(check_endpoint(saved,source,'gate',6)['frozen_proposals_equal'])
            saved['actor']['base.mean_layer.bias'][0] += .01
            with self.assertRaisesRegex(ValueError,'frozen proposal'):
                check_endpoint(saved,source,'gate',6)

    def test_validation_calibration_is_finite_and_isolates_rng(self):
        from interaction_rollout import frozen_payload
        agent = make_agent(small_config())
        state = RandomState.capture()
        rows = validation_calibration(frozen_payload(agent,online_critic=True),agent,42,states=2,repetitions=2)
        actual = np.random.rand(4),torch.randn(4)
        state.restore()
        equal(self,actual,(np.random.rand(4),torch.randn(4)))
        self.assertEqual([r['scene_seed'] for r in rows],[330000000,330000001])
        for row in rows:
            self.assertTrue(all(np.isfinite(value) for value in row.values()))
            self.assertGreater(row['simulated_steps'],0)

    def test_attacker_environment_keeps_defender_team_reward(self):
        env = TADEnv(3, protocol='paper-parameters-v1', LearningSide='Att')
        env.reset(2.25)
        packet = public_packet(env)
        _, reward, _, _, info, _ = execute(env, packet, np.full((3,3), .5), attacker_action=np.zeros(2))
        self.assertEqual(reward.shape, (3,))
        equal(self, reward, sum(env.paper_reward_components()))
        self.assertTrue(np.isfinite(info['attacker_reward']))

    def test_parallel_labels_match_serial_and_value_control(self):
        c = small_config()
        c['interaction']['pairs_per_batch'] = 4
        agent = make_agent(c)
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25)
        pool = SnapshotPool(2); pool.add(env)
        serial = InterventionSampler(0)
        parallel = InterventionSampler(2)
        try:
            a, ca = serial.generate(agent,pool,step=1000)
            b, cb = parallel.generate(agent,pool,step=1000)
            # Workers return their interleaved chunks in fixed worker order.
            order = [0,2,1,3]
            for key in a: np.testing.assert_allclose(a[key][order],b[key],rtol=1e-6,atol=1e-6)
            self.assertEqual(ca,cb)
            agent.config['interaction']['auxiliary'] = 'value'
            same, used = serial.generate(agent,pool,step=1000)
            equal(self, a, same)
            self.assertEqual(used,ca)
        finally:
            parallel.close()

    def test_vrx_and_numerical_physical_command_agree(self):
        sys.path.insert(0,str(ROOT/'vrx'))
        from interaction_controller import InteractionController
        agent = make_agent(small_config())
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25)
        p = public_packet(env)
        with tempfile.TemporaryDirectory() as directory:
            export_policy(agent,directory,0)
            vrx = InteractionController(Path(directory)/'policy.pth')
            numerical = DeploymentPolicy(Path(directory)/'policy.pth')
            a = vrx.control(p['obs'],p['motion'],env.boids_actions)
            b = numerical.control(p['obs'],p['motion'],env.boids_actions)
            equal(self,a,b)
            equal(self,vrx.last_action,numerical.last_action)


if __name__ == '__main__':
    unittest.main()
