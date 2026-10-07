"""Behavioral invariants of interaction-conditioned SAC and interventions."""
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
from torch.distributions import Normal, TransformedDistribution, TanhTransform, AffineTransform
import yaml

from envs.TADgame import TADEnv
from envs.snapshot import RandomState, seed_random
from policy.interaction_sac import InteractionActor, InteractionSAC, JointReplay, TwinTeamCritic, transformed_normal
from interaction_rollout import (public_packet, compact_snapshot, FrozenPolicy, frozen_payload,
    branch_return, intervention_pair, SnapshotPool, InterventionSampler, DeploymentPolicy)
from train_interaction import export_policy


def config():
    c = yaml.safe_load((ROOT / 'train/configs/interaction-aware-sac.yaml').read_text())
    c['rl'].update(hidden_dim=32, batch_size=4)
    c['interaction'].update(relation_dim=16, workers=0, pairs_per_batch=2, horizon_steps=2, safety=False)
    return c


def scene(n=3):
    env = TADEnv(n, protocol='paper-parameters-v1')
    env.reset(2.25)
    return env, public_packet(env)


class Networks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_random(42)

    def test_density_includes_affine_gate_jacobian(self):
        mean = torch.tensor([[.2], [-1.1]], dtype=torch.float64)
        log_std = torch.full_like(mean, -.5)
        theta, logp = transformed_normal(mean, log_std, torch.full_like(mean, .3), gate=True)
        dist = TransformedDistribution(Normal(mean, log_std.exp()), [TanhTransform(), AffineTransform(.5, .5)])
        torch.testing.assert_close(logp, dist.log_prob(theta).sum(-1, keepdim=True))

    def test_source_initialization_preserves_deterministic_controls(self):
        _, p = scene()
        actor = InteractionActor(32, 16)
        obs, motion = torch.as_tensor(p['obs']), torch.as_tensor(p['motion'])
        source, _ = actor.base(obs, True)
        actual, _ = actor(obs[None], motion[None], deterministic=True)
        torch.testing.assert_close(actual[0], source)

    def test_permutation_and_fleet_sizes(self):
        actor, critic = InteractionActor(32, 16), TwinTeamCritic(32, 16)
        torch.nn.init.normal_(actor.relation_gate.weight, std=.1)
        for n in range(2, 8):
            env, p = scene(n)
            packet = {k: torch.as_tensor(v)[None] for k, v in p.items()}
            original, _ = actor(packet['obs'], packet['motion'], deterministic=True)
            expected = critic(packet, original)
            order = np.arange(n)[::-1].copy()
            env.defender_list = [env.defender_list[i] for i in order]
            env.boids_states = env.boids_states[order]
            env.boids_actions = env.boids_actions[order]
            permuted = {k: torch.as_tensor(v)[None] for k, v in public_packet(env).items()}
            action, _ = actor(permuted['obs'], permuted['motion'], deterministic=True)
            torch.testing.assert_close(action, original[:, order], rtol=2e-5, atol=2e-6)
            for q, wanted in zip(critic(permuted, action), expected):
                torch.testing.assert_close(q, wanted, rtol=2e-5, atol=2e-6)

    def test_empty_masks_are_finite_and_peer_candidate_path_has_gradient(self):
        _, p = scene()
        actor = InteractionActor(32, 16)
        torch.nn.init.normal_(actor.relation_gate.weight)
        obs, motion = torch.as_tensor(p['obs'])[None], torch.as_tensor(p['motion'])[None]
        action, logp = actor(obs, motion, mask=torch.zeros(1, 3, dtype=torch.bool))
        self.assertTrue(torch.isfinite(action).all())
        self.assertEqual(float(logp), 0.)
        action, _ = actor(obs, motion, deterministic=True)
        action[..., 2].sum().backward()
        self.assertGreater(float(actor.relations[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(actor.base.mean_layer.weight.grad.abs().sum()), 0.)

    def test_freeze_unfreeze_and_auxiliary_updates(self):
        agent = InteractionSAC(config())
        env, packet = scene()
        replay = JointReplay(16)
        for _ in range(5):
            action, _ = agent.choose_action(packet)
            replay.store(packet, action, env.boids_actions, np.ones(3), packet, False)
        before = agent.actor.base.mean_layer.weight.detach().clone()
        agent.learn(replay)
        torch.testing.assert_close(agent.actor.base.mean_layer.weight, before, rtol=0, atol=0)
        agent.set_stage('joint')
        agent.learn(replay)
        self.assertFalse(torch.equal(agent.actor.base.mean_layer.weight, before))
        pool = SnapshotPool(2)
        pool.add(env)
        auxiliary, _ = InterventionSampler(0).generate(agent, pool)
        result = agent.learn(replay, auxiliary)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))

    def test_deployment_matches_training_without_critic(self):
        agent = InteractionSAC(config())
        _, packet = scene(6)
        with tempfile.TemporaryDirectory() as directory:
            export_policy(agent, directory, 12)
            policy = DeploymentPolicy(Path(directory) / 'policy.pth')
            wanted, _ = agent.choose_action(packet, True)
            got, _ = policy.choose_action(packet)
            np.testing.assert_array_equal(got, wanted)
            self.assertFalse(hasattr(policy, 'critic'))


class Interventions(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_random(19)
        self.agent = InteractionSAC(config())
        self.env, self.packet = scene()
        self.snapshot = compact_snapshot(self.env)
        self.policy = FrozenPolicy(frozen_payload(self.agent))

    def test_first_step_only_and_rng_isolation(self):
        state = RandomState.capture()
        env_before = self.env.defender_list[0].pos.copy()
        result = intervention_pair(self.policy, self.snapshot, 123, 456, 1, 0., 3, trace=True)
        np.testing.assert_array_equal(result['action'][[0, 2]], result['counterfactual'][[0, 2]])
        np.testing.assert_array_equal(result['action'][:, :2], result['counterfactual'][:, :2])
        self.assertEqual(result['counterfactual'][1, 2], 0.)
        traces = result['trace']
        self.assertEqual(traces[0][0]['logp'], 0.)
        self.assertEqual(traces[1][0]['logp'], 0.)
        self.assertNotEqual(traces[1][1]['action'][1, 2], 0.)
        np.testing.assert_array_equal(self.env.defender_list[0].pos, env_before)
        after = np.random.rand(5), torch.randn(5)
        state.restore()
        np.testing.assert_array_equal(after[0], np.random.rand(5))
        torch.testing.assert_close(after[1], torch.randn(5), rtol=0, atol=0)

    def test_terminal_deadline_does_not_bootstrap(self):
        self.snapshot.environment.Current_T = 59.8
        action, _ = self.agent.choose_action(self.packet, True)
        self.policy.value = lambda *args: self.fail('Terminal tail was bootstrapped')
        value, steps, trace = branch_return(self.policy, self.snapshot, action, 44, 55, 100, trace=True)
        self.assertEqual(steps, 1)
        self.assertEqual(trace[-1]['outcome'], 4)
        self.assertAlmostEqual(value, trace[0]['reward'])

    def test_identical_intervention_has_identical_continuation(self):
        action, _ = self.agent.choose_action(self.packet, True)
        a = branch_return(self.policy, self.snapshot, action, 32, 45, 5)
        b = branch_return(self.policy, self.snapshot, action.copy(), 32, 45, 5)
        self.assertEqual(a[:2], b[:2])

    def test_sampler_preserves_rng_and_absolute_labels(self):
        pool = SnapshotPool(2)
        pool.add(self.env)
        state = RandomState.capture()
        labels, count = InterventionSampler(0).generate(self.agent, pool)
        got = np.random.rand(3), torch.randn(3)
        state.restore()
        np.testing.assert_array_equal(got[0], np.random.rand(3))
        torch.testing.assert_close(got[1], torch.randn(3), rtol=0, atol=0)
        self.assertEqual(labels['return'].shape, (2, 1))
        self.assertEqual(count, 8)


if __name__ == '__main__':
    unittest.main()
