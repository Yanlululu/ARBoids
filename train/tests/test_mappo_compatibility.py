"""Physical candidate contrasts, actual gate probabilities and checkpoint migration."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.mappo import PredictiveMAPPO
from train_mappo import collect_episodes, make_env
from train_mappo_performance import baseline_reference
from policy.networks import ActorAdap
from policy.interaction_prediction import exchange_candidates
from refine_mappo_prior import propose_gain_step, constrained_direction


def configuration():
    return dict(agent=dict(defender_num=3, boid_state=True, form_reward=True),
        environment=dict(protocol='paper-parameters-v1', total_time=.6), prediction=dict(horizon=.1),
        mappo=dict(hidden_dim=32, relation_dim=8, coordination_dim=4, theta_space_gate=True,
                   baseline_initialization=True, ppo_epochs=1, minibatch_size=64, prediction_lr=1e-4))


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(865)
        np.random.seed(865)

    def test_scalar_prior_cached_likelihood_matches_full_censored_policy(self):
        config = configuration()
        config['mappo'].update(compatibility_prior=True, compatibility_gain=.04,
            compatibility_compact_risk=True, censored_gate_likelihood=True)
        agent = PredictiveMAPPO(config)
        env = make_env(config)
        observation = env.reset(2.)
        _, record = agent.act(observation, env.env.prediction_snapshot(),
                             env.env.thrust_to_action(env.env.boids_actions), 0.)
        batch = {k: agent.tensor(v[None], k=='mask') for k, v in record.items()
                 if k in ('obs', 'proposals', 'boids', 'edges', 'mask', 'proposal_raw', 'gate_raw')}
        batch['edges'][..., [0, 3, 6, 9]] = torch.tensor([3., 4., 8., 9.])/5.
        batch['gate_raw'][0, :, 0] = torch.tensor([-1., .5, 2.])
        with torch.no_grad():
            captured = []
            handle = agent.actor.gate_mean.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
            distribution, _, anchor = agent.actor.gate_distribution(batch['obs'], batch['proposals'],
                batch['boids'], batch['edges'], batch['mask'], return_anchor=True)
            handle.remove()
            signal = agent.actor.compatibility_signal(batch['edges'], batch['mask'], batch['obs'], anchor)
            gain = float(torch.nn.functional.softplus(agent.actor.compatibility_gain_raw))
            delta = torch.randn_like(agent.actor.gate_mean.weight)*.01
            cached = torch.distributions.Normal(distribution.mean+.03*signal+captured[0]@delta.T+.002, distribution.stddev)
            expected = agent.actor.gate_log_probability(cached, batch['gate_raw'])
            agent.actor.compatibility_gain_raw.fill_(float(np.log(np.expm1(gain+.03))))
            agent.actor.gate_mean.weight.add_(delta)
            agent.actor.gate_mean.bias.add_(.002)
            _, actual, _ = agent.evaluate_batch(batch)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.isfinite(actual).all())

    def test_scalar_prior_diagnosis_is_read_only_and_rejects_stale_samples(self):
        config = configuration()
        config['mappo'].update(compatibility_prior=True, compatibility_gain=.04,
            censored_gate_likelihood=True, breach_constraint=True, joint_ratio=True,
            cost_monte_carlo_targets=True, target_kl=.01)
        agent = PredictiveMAPPO(config)
        rollout = collect_episodes(agent, make_env(config), 4, 2.)
        for i, summary in enumerate(rollout.summaries):
            summary['seed'] = 2900000+i
        actor = copy.deepcopy(agent.actor.state_dict())
        counters = agent.updates, agent.total_steps, agent.lagrange.value, agent.breach_lagrange.value
        result = propose_gain_step(agent, rollout)
        self.assertIn('gradients', result)
        result = propose_gain_step(agent, rollout, gate_head=True)
        self.assertTrue(result['gate_head'])
        for name, value in actor.items():
            torch.testing.assert_close(agent.actor.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(counters, (agent.updates, agent.total_steps, agent.lagrange.value, agent.breach_lagrange.value))
        with torch.no_grad():
            agent.actor.gate_mean.bias.add_(.1)
        with self.assertRaisesRegex(ValueError, 'stale'):
            propose_gain_step(agent, rollout)

    def test_gate_head_projection_respects_constraints_and_zero_rows(self):
        fisher = torch.eye(2)
        direction = constrained_direction(fisher, torch.tensor([2., 1.]),
                                          torch.tensor([[1., 0.], [0., -1.], [0., 0.]]))
        torch.testing.assert_close(direction, torch.tensor([-.01*np.sqrt(5.)/1.001, 1./1.001], dtype=torch.float64))
        self.assertGreater(float(direction@torch.tensor([2., 1.], dtype=torch.float64)), 0.)
        unconstrained = constrained_direction(fisher, torch.tensor([2., 1.]), torch.zeros(3, 2))
        torch.testing.assert_close(unconstrained, torch.tensor([2./1.001, 1./1.001], dtype=torch.float64))

    def test_candidate_order_controls_direction_and_no_neighbors_gives_zero(self):
        agent = PredictiveMAPPO(configuration())
        agent.enable_compatibility_prior(task_priority=False, peer_intent=False)
        edges = torch.randn(2, 3, 3, 18)
        mask = ~torch.eye(3, dtype=torch.bool).expand(2, -1, -1)
        edges[..., [0, 3, 6, 9]] = torch.tensor([3., 4., 8., 9.]) / 5.
        positive = agent.actor.compatibility_signal(edges, mask)
        self.assertTrue((positive > 0).all())  # L has more compatible predicted pairs.
        reversed_edges = edges.clone()
        reversed_edges[..., [0, 3, 6, 9]] = edges[..., [6, 9, 0, 3]]
        torch.testing.assert_close(agent.actor.compatibility_signal(reversed_edges, mask), -positive)
        torch.testing.assert_close(agent.actor.compatibility_signal(edges, torch.zeros_like(mask)),
                                   torch.zeros_like(positive))
        equivalent = edges.clone()
        equivalent[..., [6, 9]] = equivalent[..., [0, 3]]
        torch.testing.assert_close(agent.actor.compatibility_signal(equivalent, mask), torch.zeros_like(positive))
        safe = edges.clone()
        safe[..., [0, 3, 6, 9]] = torch.tensor([7., 8., 12., 15.]) / 5.
        torch.testing.assert_close(agent.actor.compatibility_signal(safe, mask), torch.zeros_like(positive))

    def test_prior_enters_actual_gate_likelihood_and_receives_score_gradient(self):
        agent = PredictiveMAPPO(configuration())
        env = make_env(agent.config)
        obs = agent.tensor(env.reset(2.))[None]
        action, record = agent.act(obs[0].numpy(), env.env.prediction_snapshot(),
                                   env.env.thrust_to_action(env.env.boids_actions), 0., True)
        proposals = agent.tensor(record['proposals'])[None]
        boids = agent.tensor(record['boids'])[None]
        edges = agent.tensor(record['edges'])[None]
        edges[..., [0, 3, 6, 9]] = torch.tensor([3., 4., 8., 9.]) / 5.
        mask = agent.tensor(record['mask'], True)[None]
        before, _ = agent.actor.gate_distribution(obs, proposals, boids, edges, mask)
        agent.enable_compatibility_prior()
        after, _ = agent.actor.gate_distribution(obs, proposals, boids, edges, mask)
        torch.testing.assert_close(after.mean-before.mean, agent.actor.compatibility_signal(edges, mask, obs, before.mean))
        # Saved latent samples stay constants; the gradient is d log pi, never dQ/da.
        raw = after.mean.detach() + .1
        loss = -after.log_prob(raw).mean()
        loss.backward()
        self.assertGreater(float(agent.actor.compatibility_gain_raw.grad.abs()), 0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in agent.actor.relation_encoder.parameters()), 0.)
        self.assertLess(float(agent.actor.compatibility_gain_raw.grad), 0.)

    def test_migration_checkpoint_and_baseline_keep_their_own_architecture(self):
        agent = PredictiveMAPPO(configuration())
        agent.update(collect_episodes(agent, make_env(agent.config), 2, 2.))
        weights = copy.deepcopy(agent.actor.state_dict())
        moments = {p: copy.deepcopy(state) for p, state in agent.actor_optimizer.state.items()}
        agent.enable_compatibility_prior(peer_intent=True)
        for name, value in weights.items():
            torch.testing.assert_close(agent.actor.state_dict()[name], value, rtol=0, atol=0)
        for parameter, state in moments.items():
            for key, value in state.items():
                torch.testing.assert_close(agent.actor_optimizer.state[parameter][key], value, rtol=0, atol=0)
        rollout = collect_episodes(agent, make_env(agent.config), 2, 2.)
        for episode in rollout.episodes:
            for transition in episode:
                np.testing.assert_array_equal(transition['obs'], transition['message_observations'])
        data = rollout.tensors(agent.settings, 'cpu')
        lp, lg, _ = agent.evaluate_batch(data)
        torch.testing.assert_close(lp, data['old_logp_l'], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lg, data['old_logp_g'], rtol=1e-5, atol=1e-5)
        with tempfile.TemporaryDirectory(prefix='arboids-compatibility-') as directory:
            path = Path(directory)/'model.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
        for name, value in agent.actor.state_dict().items():
            torch.testing.assert_close(restored.actor.state_dict()[name], value, rtol=0, atol=0)
        restored.update(collect_episodes(restored, make_env(restored.config), 2, 2.))
        reference = baseline_reference(agent.config, ActorAdap(6, 8, 3, 32).state_dict())
        self.assertIsNone(reference.actor.compatibility_gain_raw)
        self.assertFalse(reference.actor.theta_space_gate)
        self.assertFalse(reference.settings.compatibility_peer_intent)

    def test_legacy_risk_kernel_is_restored_without_changing_its_policy(self):
        agent = PredictiveMAPPO(configuration())
        agent.enable_compatibility_prior(compact_risk=False, task_priority=False, peer_intent=False)
        agent.config['mappo'].pop('compatibility_compact_risk')
        agent.config['mappo'].pop('compatibility_task_priority')
        agent.config['mappo'].pop('compatibility_priority_temperature')
        agent.config['mappo'].pop('compatibility_peer_intent')
        with tempfile.TemporaryDirectory(prefix='arboids-legacy-risk-') as directory:
            path = Path(directory)/'model.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
        edges = torch.zeros(1, 3, 3, 18)
        edges[..., [0, 3, 6, 9]] = torch.tensor([7., 8., 12., 15.]) / 5.
        mask = ~torch.eye(3, dtype=torch.bool)[None]
        self.assertFalse(restored.actor.compatibility_compact_risk)
        torch.testing.assert_close(agent.actor.compatibility_signal(edges, mask),
                                   restored.actor.compatibility_signal(edges, mask), rtol=0, atol=0)
        self.assertTrue((restored.actor.compatibility_signal(edges, mask) > 0).all())

    def test_interception_responsibility_is_reciprocal_and_locally_computable(self):
        agent = PredictiveMAPPO(configuration())
        agent.enable_compatibility_prior()
        # Boat 0 at (0,0), heading 0; boat 1 at (10,0), heading pi/2.
        # Attacker at (2,0): each boat reconstructs the other's distance locally.
        obs = torch.zeros(1, 2, 16)
        obs[0, :, 2] = torch.tensor([2., 8.])
        obs[0, 1, 3] = torch.pi / 2
        edges = torch.zeros(1, 2, 2, 18)
        edges[0, 0, 1, 12:14] = torch.tensor([2., 0.])
        edges[0, 1, 0, 12:14] = torch.tensor([0., 2.])
        responsibility = agent.actor.compatibility_responsibility(obs, edges)
        self.assertLess(float(responsibility[0, 0, 1]), 1.)
        self.assertGreater(float(responsibility[0, 1, 0]), 1.)
        torch.testing.assert_close(responsibility[0, 0, 1] + responsibility[0, 1, 0], torch.tensor(2.))
        changed_peer_obs = obs.clone()
        changed_peer_obs[0, 1, 2:4] = torch.tensor([100., -1.])
        torch.testing.assert_close(agent.actor.compatibility_responsibility(changed_peer_obs, edges)[0, 0, 1],
                                   responsibility[0, 0, 1], rtol=0, atol=0)

    def test_peer_intent_uses_the_other_boat_and_keeps_its_score_gradient(self):
        agent = PredictiveMAPPO(configuration())
        agent.enable_compatibility_prior(task_priority=False, peer_intent=True)
        edges = torch.zeros(1, 2, 2, 18)
        edges[..., [0, 3, 6, 9]] = torch.tensor([10., 5., 6., 10.]) / 5.
        mask = ~torch.eye(2, dtype=torch.bool)[None]
        zero, one = torch.zeros(1, 2, 1), torch.ones(1, 2, 1)
        self.assertTrue((agent.actor.compatibility_signal(edges, mask, anchor=zero) < 0).all())
        self.assertTrue((agent.actor.compatibility_signal(edges, mask, anchor=one) > 0).all())
        anchor = torch.full((1, 2, 1), .5, requires_grad=True)
        agent.actor.compatibility_signal(edges, mask, anchor=anchor)[0, 0].backward()
        self.assertEqual(float(anchor.grad[0, 0]), 0.)
        self.assertGreater(float(anchor.grad[0, 1]), 0.)

    def test_exchanged_actor_context_is_a_validated_immutable_raw_copy(self):
        obs = np.zeros((3, 18), np.float32)
        messages = exchange_candidates(np.zeros((3, 6)), np.zeros((3, 2)), np.zeros((3, 2)),
                                       0., actor_observations=obs)
        obs[:] = 1.
        np.testing.assert_array_equal(messages.actor_observations, np.zeros((3, 18), np.float32))
        self.assertFalse(messages.actor_observations.flags.writeable)
        with self.assertRaises(ValueError):
            exchange_candidates(np.zeros((3, 6)), np.zeros((3, 2)), np.zeros((3, 2)),
                                0., actor_observations=np.zeros((3, 19)))

    def test_batched_intents_keep_worlds_separate_and_recompute_saved_probabilities(self):
        agent = PredictiveMAPPO(configuration())
        agent.enable_compatibility_prior(peer_intent=True)
        observations, states, boids = [], [], []
        for seed in (9101, 9102):
            np.random.seed(seed)
            env = make_env(agent.config)
            observations.append(env.reset(2.))
            states.append(env.env.prediction_snapshot())
            boids.append(env.env.thrust_to_action(env.env.boids_actions))
        actions, records = agent.act_batch(observations, states, boids, [0., 0.],
            [torch.Generator().manual_seed(9101), torch.Generator().manual_seed(9102)], deterministic=True)
        for i, record in enumerate(records):
            single, _ = agent.act(observations[i], states[i], boids[i], 0., deterministic=True)
            np.testing.assert_allclose(actions[i], single, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(record['message_observations'], record['obs'])
        batch = {key: agent.tensor(np.asarray([r[key] for r in records]), key=='mask')
                 for key in ('obs','proposals','boids','edges','mask','proposal_raw','gate_raw')}
        lp, lg, _ = agent.evaluate_batch(batch)
        torch.testing.assert_close(lp, agent.tensor(np.asarray([r['old_logp_l'] for r in records])), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lg, agent.tensor(np.asarray([r['old_logp_g'] for r in records])), rtol=1e-5, atol=1e-5)

    def test_calibrated_prior_bounds_team_kl_and_preserves_the_trained_actor(self):
        agent = PredictiveMAPPO(configuration())
        rollout = collect_episodes(agent, make_env(agent.config), 2, 2.)
        for i, (episode, summary) in enumerate(zip(rollout.episodes, rollout.summaries)):
            summary['seed'] = 100100+i
            for row in episode:
                # A controlled candidate contrast tests the analytical bound;
                # these synthetic edges are not an environment performance test.
                row['edges'][..., [0,3,6,9]] = np.array([3.,4.,8.,9.])/5.
                with torch.no_grad():
                    _, raw, lp, _ = agent.actor.sample_gates(*[
                        agent.tensor(row[key], key=='mask')[None]
                        for key in ('obs','proposals','boids','edges','mask')])
                row['gate_raw'], row['old_logp_g'] = raw[0].numpy(), lp[0].numpy()
        batch = rollout.tensors(agent.settings, 'cpu')
        args = [batch[key] for key in ('obs','proposals','boids','edges','mask')]
        old, _ = agent.actor.gate_distribution(*args)
        weights = copy.deepcopy(agent.actor.state_dict())
        result = agent.calibrate_compatibility_prior(rollout, target_kl=1e-4)
        new, _ = agent.actor.gate_distribution(*args)
        kl = torch.distributions.kl_divergence(old, new).sum((-1,-2)).mean()
        self.assertLess(result['gain'], 1.)
        self.assertAlmostEqual(float(kl.detach()), 1e-4, delta=1e-7)
        self.assertAlmostEqual(result['initialized_joint_kl'], 1e-4)
        for name, value in weights.items():
            torch.testing.assert_close(agent.actor.state_dict()[name], value, rtol=0, atol=0)
        (-new.log_prob(batch['gate_raw']).sum()).backward()
        self.assertGreater(float(agent.actor.compatibility_gain_raw.grad.abs()), 0.)

    def test_calibration_rejects_evaluation_and_stale_samples_without_mutation(self):
        agent = PredictiveMAPPO(configuration())
        rollout = collect_episodes(agent, make_env(agent.config), 1, 2.)
        rollout.summaries[0]['seed'] = 20000
        with self.assertRaisesRegex(ValueError, 'ordinary training'):
            agent.calibrate_compatibility_prior(rollout, 1e-4)
        self.assertIsNone(agent.actor.compatibility_gain_raw)
        rollout.summaries[0]['seed'] = 100100
        rollout.episodes[0][0]['old_logp_g'] += 1.
        with self.assertRaisesRegex(ValueError, 'fresh calibration'):
            agent.calibrate_compatibility_prior(rollout, 1e-4)
        self.assertIsNone(agent.actor.compatibility_gain_raw)


if __name__ == '__main__':
    unittest.main()
