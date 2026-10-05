"""Regression checks for transfer, update limits and the performance gate."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.mappo import PredictiveMAPPO, clipped_policy_loss
from policy.networks import ActorAdap
from parallel_mappo import ParallelRollouts
from train_mappo import collect_episodes, make_env, play_episode, run_training
from qualify_mappo import main as qualify_main, completed_development
from train_mappo_performance import (qualifies, reset_gate_exploration, restore_proposal_exploration,
                                     configure_batch, initialize_bounded_gate, initialize_new_gate,
                                     initialize_theta_space_gate, baseline_reference, validation_result,
                                     require_fresh_audit)


def configuration():
    return dict(algorithm='predictive-mappo', agent=dict(defender_num=3, boid_state=True, form_reward=True),
                environment=dict(protocol='paper-parameters-v1', total_time=.6),
                mappo=dict(hidden_dim=32, relation_dim=8, coordination_dim=4, ppo_epochs=2,
                           minibatch_size=4, baseline_initialization=True, reward_value_scale=100.),
                prediction=dict(horizon=.1))


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        np.random.seed(7)

    def test_audit_reuse_rejects_partial_overlap_in_another_checkout(self):
        with tempfile.TemporaryDirectory(prefix='arboids-audit-') as directory:
            root = Path(directory)
            previous = root/'original'/'old-run'
            previous.mkdir(parents=True)
            (previous/'audit-candidate.json').write_text('{"first_seed":60000,"episodes":2000}')
            roots = [root/'current', root/'original']
            with self.assertRaisesRegex(ValueError, 'already been used'):
                require_fresh_audit([61999, 62000], roots)
            require_fresh_audit([62000, 62001], roots)
            # An explicit non-contiguous manifest takes precedence over its
            # first-seed/count metadata, and every actual used seed is blocked.
            (previous/'audit-candidate.csv').write_text('seed\n60001\n90001\n')
            with self.assertRaisesRegex(ValueError, 'already been used'):
                require_fresh_audit([90001], roots)
            require_fresh_audit([60002], roots)

    def test_development_only_and_nonqualifying_reports_cannot_bypass_acceptance(self):
        def row(seed, success):
            return dict(seed=seed,success=success,capture=success,timeout=0,breach=0,
                        collision=1-success,task_return=0.,gate_mean=.5)
        with tempfile.TemporaryDirectory(prefix='arboids-measure-') as directory:
            root=Path(directory)
            candidate,reference,manifest=root/'input.pth',root/'reference.pth',root/'summary.json'
            agent=PredictiveMAPPO(configuration())
            agent.save(candidate)
            torch.save(ActorAdap(6,8,3,32).state_dict(),reference)
            manifest.write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                protocol='paper-parameters-v1',duration=.6,agility=2.,episodes=2,seeds=[20000,20001],
                success_rate=.5,collision_rate=.5,breach_rate=0.)))
            for flag in ('--development-only','--report-nonqualifying'):
                with self.subTest(flag=flag):
                    output=root/flag[2:]
                    pool=Mock()
                    development=[row(20000,1),row(20001,int(flag=='--development-only'))]
                    audit_before=[row(70000+i,int(i>=20)) for i in range(200)]
                    audit_after=[row(70000+i,1) for i in range(200)]
                    pool.run.side_effect=[development,audit_before,audit_after]
                    argv=['qualify_mappo.py','--checkpoint',str(candidate),'--baseline',str(reference),
                          '--baseline-validation',str(manifest),'--output-dir',str(output),
                          '--audit-seed','70000','--audit-episodes','200',
                          '--report-batch-size','100',
                          '--min-success-rate','.99','--max-collision-rate','.01',flag]
                    with patch.object(sys,'argv',argv), patch('qualify_mappo.ParallelRollouts') as factory, \
                            patch('qualify_mappo.require_fresh_audit') as fresh, contextlib.redirect_stdout(io.StringIO()):
                        factory.return_value.__enter__.return_value=pool
                        qualify_main()
                    result=json.loads((output/'qualification.json').read_text())
                    self.assertFalse(result['performance_passed'])
                    self.assertEqual(result['development_passed'],flag=='--development-only')
                    self.assertFalse((output/'qualified.pth').exists())
                    self.assertEqual(result['criteria']['source_checkpoint_sha256'],hashlib.sha256(candidate.read_bytes()).hexdigest())
                    if flag=='--development-only':
                        self.assertEqual(pool.run.call_count,1)
                        fresh.assert_not_called()
                        self.assertFalse((output/'audit-candidate.json').exists())
                        baseline_hash=hashlib.sha256(reference.read_bytes()).hexdigest()
                        manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest()
                        rows,provenance=completed_development(output,candidate,baseline_hash,manifest_hash,[20000,20001])
                        self.assertEqual([row['seed'] for row in rows],[20000,20001])
                        self.assertEqual(provenance['episodes'],2)
                        with self.assertRaisesRegex(ValueError,'exact ordered'):
                            completed_development(output,candidate,baseline_hash,manifest_hash,[20001,20000])
                        with self.assertRaisesRegex(ValueError,'evaluation manifest'):
                            completed_development(output,candidate,baseline_hash,'0'*64,[20000,20001])
                        different=root/'different.pth'
                        with torch.no_grad():
                            agent.actor.proposal_mean.bias.add_(.1)
                        agent.save(different)
                        with self.assertRaisesRegex(ValueError,'different checkpoint'):
                            completed_development(output,different,baseline_hash,manifest_hash,[20000,20001])
                        reused_argv=list(argv)
                        reused_argv[reused_argv.index('--output-dir')+1]=str(root/'reused')
                        reused_argv+=['--development-results',str(output)]
                        with patch.object(sys,'argv',reused_argv), patch('qualify_mappo.ParallelRollouts') as factory, \
                                patch('qualify_mappo.require_fresh_audit') as fresh, contextlib.redirect_stdout(io.StringIO()):
                            qualify_main()
                            factory.return_value.__enter__.return_value.run.assert_not_called()
                            fresh.assert_not_called()
                        reused=json.loads((root/'reused'/'qualification.json').read_text())
                        self.assertTrue(reused['development_passed'])
                        self.assertFalse(reused['performance_passed'])
                        self.assertEqual(reused['criteria']['development_reuse']['rows_sha256'],provenance['rows_sha256'])
                        completed=json.loads((output/'validation.json').read_text())
                        (output/'validation.json').write_text(json.dumps(dict(completed,complete=False)))
                        with self.assertRaisesRegex(ValueError,'completed frozen'):
                            completed_development(output,candidate,baseline_hash,manifest_hash,[20000,20001])
                    else:
                        self.assertEqual(pool.run.call_count,3)
                        fresh.assert_called_once()
                        audit=json.loads((output/'audit-candidate.json').read_text())
                        self.assertFalse(audit['performance_passed'])
                        self.assertEqual([b['first_seed'] for b in audit['report_batches']],[70000,70100])
                        self.assertEqual(sum(b['event_counts']['candidate']['success'] for b in audit['report_batches']),200)
                        self.assertEqual(set(audit['confidence_intervals_95']),{'success','collision','breach'})
                        self.assertEqual(sum(b['event_counts']['reference']['collision'] for b in audit['report_batches']),20)
    def test_transfer_preserves_proposals_and_sigmoid_gate(self):
        agent = PredictiveMAPPO(configuration())
        teacher = ActorAdap(6, 8, 3, 32)
        # Nontrivial adapter logits exercise the tanh/sigmoid conversion.
        with torch.no_grad():
            teacher.adap_layer.weight.normal_(0., .2)
            teacher.adap_layer.bias.fill_(.3)
        agent.actor.initialize_from_baseline(teacher.state_dict())
        env = make_env(configuration())
        obs = env.reset()
        with torch.no_grad():
            expected, _ = teacher(torch.tensor(obs, dtype=torch.float32), True, False)
        actual, _ = agent.act(obs, env.env.prediction_snapshot(),
                              env.env.thrust_to_action(env.env.boids_actions), 0., True)
        np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=2e-6)
        self.assertEqual(agent.actor.gate_mean.weight.count_nonzero(), 0)

    def test_value_scale_only_changes_value_representation(self):
        config = configuration()
        agent = PredictiveMAPPO(config)
        env = make_env(config)
        env.reset()
        state = env.env.centralized_state()
        with torch.no_grad():
            r, c = agent.critic(agent.tensor(state)[None])
        reward_value, cost_value = agent.values(state)
        self.assertAlmostEqual(reward_value, float(r)*100.)
        self.assertAlmostEqual(cost_value, float(c))

    def test_new_gate_preserves_proposals_and_clears_only_reset_momentum(self):
        agent = PredictiveMAPPO(configuration())
        agent.update(collect_episodes(agent, make_env(configuration()), 2, 2.))
        env = make_env(configuration())
        obs = env.reset()
        before = copy.deepcopy(agent.actor.state_dict())
        preserved_momentum = {parameter: copy.deepcopy(state) for parameter, state
                              in agent.actor_optimizer.state.items()
                              if all(parameter is not reset for layer in
                                     (agent.actor.base_gate, agent.actor.gate_mean)
                                     for reset in layer.parameters())}
        agent.training_state['stable_count'] = 2
        initialize_new_gate(agent)
        _, record = agent.act(obs, env.env.prediction_snapshot(),
            env.env.thrust_to_action(env.env.boids_actions), 0., True)
        np.testing.assert_array_equal(record['gate_raw'], 0.)
        self.assertEqual(agent.training_state['stable_count'], 0)
        self.assertEqual(agent.actor.baseline_gate_bound, 0.)
        for key, value in before.items():
            if not key.startswith(('base_gate.', 'gate_mean.')):
                torch.testing.assert_close(agent.actor.state_dict()[key], value, rtol=0, atol=0)
        for layer in (agent.actor.base_gate, agent.actor.gate_mean):
            for parameter in layer.parameters():
                self.assertNotIn(parameter, agent.actor_optimizer.state)
        for parameter, state in preserved_momentum.items():
            for key, value in state.items():
                torch.testing.assert_close(agent.actor_optimizer.state[parameter][key], value)
        agent.update(collect_episodes(agent, env, 2, 2.))
        self.assertGreater(float(agent.actor.gate_mean.weight.detach().abs().sum()), 0.)

    def test_theta_space_transfer_preserves_trained_policy_and_other_momentum(self):
        config = configuration()
        config['mappo']['prediction_lr'] = .001
        agent = PredictiveMAPPO(config)
        agent.update(collect_episodes(agent, make_env(config), 2, 2.))
        env = make_env(config)
        obs = env.reset()
        args = (obs, env.env.prediction_snapshot(), env.env.thrust_to_action(env.env.boids_actions), 0., True)
        expected, _ = agent.act(*args)
        old_head = copy.deepcopy(agent.actor.gate_mean.state_dict())
        reset = [p for layer in (agent.actor.gate_mean, agent.actor.gate_log_std) for p in layer.parameters()]
        moments = {p: copy.deepcopy(s) for p, s in agent.actor_optimizer.state.items()
                   if all(p is not q for q in reset)}
        initialize_theta_space_gate(agent)
        actual, _ = agent.act(*args)
        np.testing.assert_array_equal(actual, expected)
        for key, value in old_head.items():
            torch.testing.assert_close(agent.actor.prior_gate_mean.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(agent.settings.theta_space_gate)
        self.assertFalse(any(p.requires_grad for p in agent.actor.prior_gate_mean.parameters()))
        for p in reset:
            self.assertNotIn(p, agent.actor_optimizer.state)
        for p, state in moments.items():
            for key, value in state.items():
                torch.testing.assert_close(agent.actor_optimizer.state[p][key], value, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            initialize_theta_space_gate(agent)
        with self.assertRaises(ValueError):
            initialize_new_gate(agent)
        with self.assertRaises(ValueError):
            initialize_bounded_gate(agent, 2.)

    def test_theta_space_endpoint_exploration_uses_unclipped_sample_probabilities(self):
        agent = PredictiveMAPPO(configuration())
        with torch.no_grad():
            agent.actor.base_gate.weight.zero_()
            agent.actor.base_gate.bias.fill_(-12.)
            agent.actor.gate_mean.weight.zero_()
            agent.actor.gate_mean.bias.zero_()
        initialize_theta_space_gate(agent, .1)
        env = make_env(agent.config)
        obs = env.reset()
        _, record = agent.act(obs, env.env.prediction_snapshot(),
                              env.env.thrust_to_action(env.env.boids_actions), 0., True)
        batch = {key: agent.tensor(value, key == 'mask')[None].expand(1024, *np.shape(value))
                 for key, value in record.items() if key in
                 ('obs', 'proposals', 'boids', 'edges', 'mask', 'proposal_raw')}
        args = [batch[key] for key in ('obs', 'proposals', 'boids', 'edges', 'mask')]
        with torch.no_grad():
            gates, raw, old_logp, _ = agent.actor.sample_gates(*args)
        self.assertGreater(float((gates > .1).float().mean()), .11)
        self.assertLess(float((gates > .1).float().mean()), .20)
        self.assertTrue(torch.equal(gates, raw.clamp(0., 1.)))
        self.assertGreater(int((raw < 0.).count_nonzero()), 100)
        batch['gate_raw'] = raw
        _, logp, _ = agent.evaluate_batch(batch)
        torch.testing.assert_close(logp, old_logp, rtol=0, atol=0)
        # Reward increasing physical theta: the score gradient can now move
        # the residual despite the inherited -12 logit. No dQ/dtheta is used.
        loss = -(logp * gates.squeeze(-1)).mean()
        loss.backward()
        self.assertLess(float(agent.actor.gate_mean.bias.grad), -.1)

    def test_theta_space_checkpoint_restores_optimizer_and_actor_gradients(self):
        agent = PredictiveMAPPO(configuration())
        initialize_theta_space_gate(agent)
        agent.update(collect_episodes(agent, make_env(agent.config), 2, 2.))
        with tempfile.TemporaryDirectory(prefix='arboids-theta-space-') as directory:
            path = Path(directory)/'checkpoint.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
        self.assertTrue(restored.actor.theta_space_gate)
        for key, value in agent.actor.state_dict().items():
            torch.testing.assert_close(restored.actor.state_dict()[key], value, rtol=0, atol=0)
        before = restored.actor.gate_mean.weight.detach().clone()
        metrics = restored.update(collect_episodes(restored, make_env(restored.config), 2, 2.))
        self.assertFalse(torch.equal(before, restored.actor.gate_mean.weight))
        self.assertGreater(metrics['grad_relation_encoder'], 0.)
        self.assertGreater(metrics['grad_priority_head'], 0.)
        self.assertFalse(any(p.requires_grad for p in restored.actor.prior_gate_mean.parameters()))

    def test_expanded_validation_preserves_original_success_and_collision_gates(self):
        rows = [dict(seed=i, success=int(i >= 5), capture=int(i >= 5), timeout=0,
                     breach=0, collision=int(i < 5), task_return=0., gate_mean=.5)
                for i in range(100)]
        baseline = dict(success_rate=.90, collision_rate=.10,
                        protected_reference=dict(seeds=list(range(20)),
                                                 success_rate=.90, collision_rate=.10))
        metrics, passed, _ = validation_result(rows, baseline, .05)
        self.assertEqual(metrics['success_rate'], .95)
        self.assertFalse(passed)  # Overall improvement cannot hide the original subset's failure.
        for row in rows:
            row.update(success=1, capture=1, collision=int(row['seed'] in (0, 30, 50, 70)))
        _, passed, score = validation_result(rows, baseline, .05)
        self.assertTrue(passed)
        rows[30]['collision'] = 0
        _, passed, lower_collision_score = validation_result(rows, baseline, .05)
        self.assertTrue(passed)
        self.assertGreater(lower_collision_score, score)
        with self.assertRaises(ValueError):
            validation_result(rows[1:], baseline, .05)

    def test_parallel_collection_matches_serial_with_fixed_episode_seeds(self):
        for theta_space in (False, True):
            with self.subTest(theta_space=theta_space):
                config = configuration()
                agent = PredictiveMAPPO(config)
                if theta_space:
                    initialize_theta_space_gate(agent)
                seeds = [20001, 20002]
                with ParallelRollouts(config, workers=2) as workers:
                    parallel = workers.run(agent, seeds)
                for seed, episode in zip(seeds, parallel.episodes):
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    serial, _ = play_episode(agent, make_env(config), 2.)
                    self.assertEqual(len(episode), len(serial))
                    for actual, expected in zip(episode, serial):
                        for key in ('obs', 'proposal_raw', 'gate_raw', 'edges', 'reward', 'cost'):
                            np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, rtol=1e-6)

    def test_initialized_checkpoint_restores_architecture_and_training_state(self):
        agent = PredictiveMAPPO(configuration())
        teacher = ActorAdap(6, 8, 3, 32)
        agent.actor.initialize_from_baseline(teacher.state_dict())
        agent.training_state = dict(stable_count=2, best_score=[1, .95, -.04])
        with tempfile.TemporaryDirectory(prefix='arboids-transfer-') as directory:
            path = Path(directory)/'checkpoint.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
        self.assertEqual(restored.training_state, agent.training_state)
        self.assertEqual(restored.settings.reward_value_scale, 100.)
        for key, value in agent.actor.state_dict().items():
            torch.testing.assert_close(restored.actor.state_dict()[key], value)

    def test_parallel_reference_does_not_inherit_candidate_gate_bound(self):
        config = configuration()
        config['mappo']['baseline_gate_bound'] = 2.
        teacher = ActorAdap(6, 8, 3, 32)
        with torch.no_grad():
            teacher.adap_layer.weight.zero_()
            teacher.adap_layer.bias.fill_(-4.)
        candidate = PredictiveMAPPO(config)
        candidate.actor.initialize_from_baseline(teacher.state_dict())
        reference = baseline_reference(config, teacher.state_dict())
        with ParallelRollouts(config, workers=2) as workers:
            bounded = workers.run(candidate, [20001, 20002], deterministic=True)
            original = workers.run(reference, [20001, 20002], deterministic=True)
            bounded_again = workers.run(candidate, [20001, 20002], deterministic=True)
        np.testing.assert_allclose(bounded.episodes[0][0]['gate_raw'], -2., atol=1e-6)
        np.testing.assert_allclose(original.episodes[0][0]['gate_raw'], -8., atol=1e-6)
        for first, last in zip(bounded.episodes, bounded_again.episodes):
            for a, b in zip(first, last):
                np.testing.assert_array_equal(a['gate_raw'], b['gate_raw'])
        for seed, episode in zip([20001, 20002], original.episodes):
            np.random.seed(seed)
            torch.manual_seed(seed)
            serial, _ = play_episode(reference, make_env(reference.config), 2., deterministic=True)
            self.assertEqual(len(episode), len(serial))
            for a, b in zip(episode, serial):
                for key in ('proposals', 'gate_raw', 'reward', 'cost'):
                    np.testing.assert_array_equal(a[key], b[key])

    def test_kl_limit_stops_actor_but_still_fits_values(self):
        config = configuration()
        config['mappo']['target_kl'] = .001
        agent = PredictiveMAPPO(config)
        rollout = collect_episodes(agent, make_env(config), 2, 2.)
        before_actor = copy.deepcopy(agent.actor.state_dict())
        before_critic = copy.deepcopy(agent.critic.state_dict())
        evaluate = agent.evaluate_batch
        def shifted(batch):
            proposal, gate, entropy = evaluate(batch)
            return proposal + 1., gate + 1., entropy
        with patch.object(agent, 'evaluate_batch', side_effect=shifted):
            metrics = agent.update(rollout)
        self.assertEqual(metrics['actor_early_stopped'], 1.)
        for key, value in before_actor.items():
            torch.testing.assert_close(value, agent.actor.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, agent.critic.state_dict()[key])
                            for key, value in before_critic.items()))

    def test_full_batch_guard_rejects_overshoot_and_restores_adam(self):
        config = configuration()
        config['mappo']['joint_ratio'] = True
        agent = PredictiveMAPPO(config)
        agent.update(collect_episodes(agent, make_env(config), 4, 2.))
        agent.set_actor_learning_rates(actor_lr=.02)
        agent.settings.full_batch_guard = config['mappo']['full_batch_guard'] = True
        agent.settings.target_kl = config['mappo']['target_kl'] = 1e-5
        rollout = collect_episodes(agent, make_env(config), 4, 2.)
        before_actor = copy.deepcopy(agent.actor.state_dict())
        before_optimizer = copy.deepcopy(agent.actor_optimizer.state_dict())
        before_critic = copy.deepcopy(agent.critic.state_dict())
        with patch.object(agent.lagrange, 'update', wraps=agent.lagrange.update) as dual_update:
            result = agent.update(rollout)
        dual_update.assert_called_once_with(rollout.collision_rate)
        self.assertGreater(result['actor_optimizer_steps'], 0)
        self.assertEqual(result['actor_accepted_steps'], 0)
        self.assertEqual(result['actor_rollback'], 1.)
        self.assertLessEqual(result['policy_kl_after'], agent.settings.target_kl)
        for key, value in before_actor.items():
            torch.testing.assert_close(value, agent.actor.state_dict()[key], rtol=0, atol=0)
        after_optimizer = agent.actor_optimizer.state_dict()
        self.assertEqual(before_optimizer['param_groups'], after_optimizer['param_groups'])
        for parameter, state in before_optimizer['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(value, after_optimizer['state'][parameter][key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, agent.critic.state_dict()[key])
                            for key, value in before_critic.items()))

    def test_full_batch_guard_accepts_learning_and_round_trips(self):
        config = configuration()
        config['mappo'].update(joint_ratio=True, full_batch_guard=True, target_kl=.03, ppo_epochs=4,
                               actor_lr=1e-5, minibatch_size=64)
        agent = PredictiveMAPPO(config)
        before = copy.deepcopy(agent.actor.state_dict())
        rollout = collect_episodes(agent, make_env(config), 8, 2.)
        result = agent.update(rollout)
        self.assertGreater(result['policy_surrogate_gain'], 0.)
        self.assertGreater(result['actor_accepted_steps'], 0)
        self.assertLessEqual(result['actor_accepted_steps'], result['actor_optimizer_steps'])
        self.assertLessEqual(result['policy_joint_kl_after'], .03)
        self.assertTrue(any(not torch.equal(value, agent.actor.state_dict()[key])
                            for key, value in before.items()))
        with tempfile.TemporaryDirectory(prefix='arboids-ppo-guard-') as directory:
            path = Path(directory) / 'checkpoint.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
        self.assertTrue(restored.settings.full_batch_guard)
        for key, value in agent.actor.state_dict().items():
            torch.testing.assert_close(restored.actor.state_dict()[key], value, rtol=0, atol=0)

    def test_full_batch_guard_keeps_best_epoch_instead_of_last(self):
        config = configuration()
        config['mappo'].update(joint_ratio=True, full_batch_guard=True, target_kl=1.)
        agent = PredictiveMAPPO(config)
        rollout = collect_episodes(agent, make_env(config), 4, 2.)
        snapshots = []
        def scored(data, advantage):
            snapshots.append((copy.deepcopy(agent.actor.state_dict()),
                              copy.deepcopy(agent.actor_optimizer.state_dict())))
            return dict(surrogate=(0., 1., .5)[len(snapshots)-1], kl_joint=.001, kl_limit=.001)
        with patch.object(agent, '_policy_statistics', side_effect=scored):
            result = agent.update(rollout)
        self.assertEqual(result['policy_epoch_selected'], 1)
        self.assertEqual(result['actor_rollback'], 1.)
        for key, value in snapshots[1][0].items():
            torch.testing.assert_close(agent.actor.state_dict()[key], value, rtol=0, atol=0)
        actual = agent.actor_optimizer.state_dict()
        for parameter, state in snapshots[1][1]['state'].items():
            for key, value in state.items():
                torch.testing.assert_close(actual['state'][parameter][key], value, rtol=0, atol=0)

    def test_separate_rates_preserve_adam_state_and_round_trip(self):
        config = configuration()
        agent = PredictiveMAPPO(config)
        agent.update(collect_episodes(agent, make_env(config), 2, 2.))
        parameter = agent.actor.gate_mean.weight
        momentum = agent.actor_optimizer.state[parameter]['exp_avg'].clone()
        agent.set_actor_learning_rates(1e-6, 1e-4)
        self.assertEqual([g['lr'] for g in agent.actor_optimizer.param_groups], [1e-6, 1e-4])
        torch.testing.assert_close(momentum, agent.actor_optimizer.state[parameter]['exp_avg'])
        agent.update(collect_episodes(agent, make_env(config), 2, 2.))
        with tempfile.TemporaryDirectory(prefix='arboids-rates-') as directory:
            path = Path(directory)/'checkpoint.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
        self.assertEqual(len(restored.actor_optimizer.param_groups), 2)
        torch.testing.assert_close(agent.actor_optimizer.state[parameter]['exp_avg'],
                                   restored.actor_optimizer.state[restored.actor.gate_mean.weight]['exp_avg'])

    def test_gate_rejects_safety_or_success_regression(self):
        self.assertTrue(qualifies(dict(success_rate=.93, collision_rate=.05), .93, .05, .065))
        self.assertFalse(qualifies(dict(success_rate=.925, collision_rate=.01), .93, .05, .065))
        self.assertFalse(qualifies(dict(success_rate=.99, collision_rate=.055), .93, .05, .065))
        self.assertFalse(qualifies(dict(success_rate=float('nan'), collision_rate=0.), .93, .05, .065))
        self.assertFalse(qualifies(dict(success_rate=.96, collision_rate=.04), .93, .05, .04))
        self.assertTrue(qualifies(dict(success_rate=.94, collision_rate=.035), .93, .05, .04))
        self.assertFalse(qualifies(dict(success_rate=.94, collision_rate=.055), .93, .05, .08))

    def test_joint_ratio_has_same_initial_score_gradient_and_clips_team_change(self):
        advantage = torch.tensor([.5, -1.2])
        factors = [torch.zeros(2, 3, requires_grad=True) for _ in range(2)]
        factor_loss = clipped_policy_loss(*factors, advantage, .1)
        expected = torch.autograd.grad(factor_loss, factors)
        joint_loss = clipped_policy_loss(*factors, advantage, .1, joint=True)
        actual = torch.autograd.grad(joint_loss, factors)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
        # Each of six factors changes by only 2%, but the team ratio exceeds 1.1.
        factors = [torch.full((1, 3), float(np.log(1.02)), requires_grad=True) for _ in range(2)]
        loss = clipped_policy_loss(*factors, torch.ones(1), .1, joint=True)
        self.assertAlmostEqual(float(loss.detach()), -1.1/3, places=6)
        for gradient in torch.autograd.grad(loss, factors):
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_joint_kl_checks_all_boats_and_both_stages(self):
        config = configuration()
        config['mappo'].update(joint_ratio=True, target_kl=.002)
        agent = PredictiveMAPPO(config)
        rollout = collect_episodes(agent, make_env(config), 2, 2.)
        before = {name: value.clone() for name, value in agent.actor.state_dict().items()}
        evaluate = agent.evaluate_batch
        def shifted(batch):
            proposal, gate, entropy = evaluate(batch)
            return proposal + .03, gate + .03, entropy
        with patch.object(agent, 'evaluate_batch', side_effect=shifted):
            result = agent.update(rollout)
        self.assertLess(result['kl_gate'], .002)
        self.assertLess(result['kl_proposal'], .002)
        self.assertGreater(result['kl_joint'], .002)
        self.assertEqual(result['actor_optimizer_steps'], 0)
        for name, value in before.items():
            torch.testing.assert_close(value, agent.actor.state_dict()[name], rtol=0, atol=0)

    def test_joint_update_trains_both_stages_and_restores_clipping_rule(self):
        config = configuration()
        config['mappo']['joint_ratio'] = True
        agent = PredictiveMAPPO(config)
        result = agent.update(collect_episodes(agent, make_env(config), 2, 2.))
        for name in ('proposal_mean', 'relation_encoder', 'priority_head', 'adapter', 'gate_mean'):
            self.assertGreater(result['grad_' + name], 0.)
        with tempfile.TemporaryDirectory(prefix='arboids-joint-ratio-') as directory:
            path = Path(directory)/'checkpoint.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
        self.assertTrue(restored.settings.joint_ratio)
        self.assertEqual(restored.updates, agent.updates)

    def test_gate_variance_rate_migrates_optimizer_without_changing_policy(self):
        config = configuration()
        config['mappo']['prediction_lr'] = 1e-3
        agent = PredictiveMAPPO(config)
        agent.update(collect_episodes(agent, make_env(config), 2, 2.))
        parameter = agent.actor.gate_log_std.weight
        momentum = agent.actor_optimizer.state[parameter]['exp_avg'].clone()
        before = {name: value.clone() for name, value in agent.actor.state_dict().items()}
        agent.set_actor_learning_rates(gate_std_lr=1e-4)
        self.assertEqual([group['name'] for group in agent.actor_optimizer.param_groups],
                         ['base', 'prediction', 'gate_variance'])
        grouped = [p for group in agent.actor_optimizer.param_groups for p in group['params']]
        self.assertEqual(len(grouped), len(set(map(id, grouped))))
        self.assertEqual(set(map(id, grouped)), set(map(id, agent.actor.parameters())))
        torch.testing.assert_close(agent.actor_optimizer.state[parameter]['exp_avg'], momentum)
        for name, value in before.items():
            torch.testing.assert_close(value, agent.actor.state_dict()[name], rtol=0, atol=0)
        metrics = agent.update(collect_episodes(agent, make_env(config), 2, 2.))
        self.assertGreater(metrics['grad_gate_log_std'], 0.)
        self.assertFalse(torch.equal(parameter, before['gate_log_std.weight']))
        with tempfile.TemporaryDirectory(prefix='arboids-gate-variance-') as directory:
            path = Path(directory)/'checkpoint.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
        self.assertEqual(restored.settings.gate_std_lr, 1e-4)
        self.assertEqual(len(restored.actor_optimizer.param_groups), 3)
        torch.testing.assert_close(restored.actor.gate_log_std.weight, parameter)
        torch.testing.assert_close(restored.actor_optimizer.state[restored.actor.gate_log_std.weight]['exp_avg'],
                                   agent.actor_optimizer.state[parameter]['exp_avg'])

    def test_shared_encoding_preserves_two_stage_probabilities_and_gradients(self):
        agent = PredictiveMAPPO(configuration())
        rollout = collect_episodes(agent, make_env(configuration()), 2, 2.)
        batch = rollout.tensors(agent.settings, 'cpu')
        actor = agent.actor
        proposal = actor.proposal_distribution(batch['obs'])
        gate, _ = actor.gate_distribution(batch['obs'], batch['proposals'], batch['boids'],
                                          batch['edges'], batch['mask'])
        reference = (proposal.log_prob(batch['proposal_raw']).sum(-1),
                     gate.log_prob(batch['gate_raw']).sum(-1),
                     proposal.entropy().sum(-1) + gate.entropy().sum(-1))
        sum(value.mean() for value in reference).backward()
        expected_gradients = {name: p.grad.clone() for name, p in actor.named_parameters()
                              if p.grad is not None}
        actor.zero_grad(set_to_none=True)
        with patch.object(actor, 'encode', wraps=actor.encode) as encode:
            actual = agent.evaluate_batch(batch)
            self.assertEqual(encode.call_count, 1)
        for expected, result in zip(reference, actual):
            torch.testing.assert_close(expected, result, rtol=0, atol=0)
        sum(value.mean() for value in actual).backward()
        for name, p in actor.named_parameters():
            if name in expected_gradients:
                torch.testing.assert_close(p.grad, expected_gradients[name], rtol=2e-5, atol=1e-6)
        env = make_env(configuration())
        obs = env.reset()
        with patch.object(actor, 'encode', wraps=actor.encode) as encode:
            agent.act(obs, env.env.prediction_snapshot(),
                      env.env.thrust_to_action(env.env.boids_actions), 0., deterministic=True)
            self.assertEqual(encode.call_count, 1)

    def test_batch_override_preserves_policy_and_resets_stability(self):
        config = configuration()
        config['training'] = dict(episodes_per_update=32, eval_episodes=200)
        agent = PredictiveMAPPO(config)
        before = {key: value.clone() for key, value in agent.actor.state_dict().items()}
        agent.training_state['stable_count'] = 2
        configure_batch(agent, 128, 1024)
        self.assertEqual(agent.config['training']['episodes_per_update'], 128)
        self.assertEqual(agent.settings.minibatch_size, 1024)
        self.assertEqual(agent.config['mappo']['minibatch_size'], 1024)
        self.assertEqual(agent.config['training']['eval_episodes'], 200)
        self.assertEqual(agent.training_state['stable_count'], 0)
        for key, value in before.items():
            torch.testing.assert_close(agent.actor.state_dict()[key], value)
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                configure_batch(agent, value)

    def test_bounded_prior_does_not_bound_learned_gate_or_cut_its_gradients(self):
        agent = PredictiveMAPPO(configuration())
        with torch.no_grad():
            agent.actor.base_gate.weight.zero_()
            agent.actor.base_gate.bias.fill_(-20.)
        initialize_bounded_gate(agent, 2.)
        env = make_env(configuration())
        obs = env.reset()
        _, record = agent.act(obs, env.env.prediction_snapshot(),
            env.env.thrust_to_action(env.env.boids_actions), 0., True)
        args = (agent.tensor(obs)[None], agent.tensor(record['proposals'])[None],
                agent.tensor(record['boids'])[None], agent.tensor(record['edges'])[None],
                agent.tensor(record['mask'], True)[None])
        distribution, _ = agent.actor.gate_distribution(*args)
        torch.testing.assert_close(distribution.mean, torch.full_like(distribution.mean, -2.))
        with torch.no_grad():
            agent.actor.gate_mean.weight.fill_(.1)
            agent.actor.gate_mean.bias.fill_(4.)
        distribution, _ = agent.actor.gate_distribution(*args)
        self.assertGreater(float(distribution.mean.detach().max()), 0.)
        distribution.log_prob(distribution.mean.detach() + .3).sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in agent.actor.relation_encoder.parameters()), 0.)
        with tempfile.TemporaryDirectory(prefix='arboids-bounded-gate-') as directory:
            path = Path(directory)/'checkpoint.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
            self.assertEqual(restored.actor.baseline_gate_bound, 2.)
            actual, _ = restored.actor.gate_distribution(*args)
            torch.testing.assert_close(actual.mean, distribution.mean)

    def test_reference_ignores_candidate_bound_and_preserves_training_rng(self):
        config = configuration()
        config['mappo']['baseline_gate_bound'] = 2.
        teacher = ActorAdap(6, 8, 3, 32)
        with torch.no_grad():
            teacher.adap_layer.weight.zero_()
            teacher.adap_layer.bias.fill_(-4.)
        rng = torch.get_rng_state()
        reference = baseline_reference(config, teacher.state_dict())
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertEqual(config['mappo']['baseline_gate_bound'], 2.)
        self.assertEqual(reference.actor.baseline_gate_bound, 0.)
        env = make_env(config)
        obs = env.reset()
        with torch.no_grad():
            expected, _ = teacher(torch.tensor(obs, dtype=torch.float32), True, False)
        actual, _ = reference.act(obs, env.env.prediction_snapshot(),
            env.env.thrust_to_action(env.env.boids_actions), 0., True)
        np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=2e-6)

    def test_full_episode_cost_targets_do_not_depend_on_critic_errors(self):
        agent = PredictiveMAPPO(configuration())
        rollout = collect_episodes(agent, make_env(configuration()), 2, 2.)
        agent.settings.cost_gae_lambda = 1.
        for index, episode in enumerate(rollout.episodes):
            rollout.summaries[index]['collision'] = index
            for t, row in enumerate(episode):
                row['cost'] = float(index == 1 and t == len(episode)-1)
                row['value_c'] = 7. + t
        batch = rollout.tensors(agent.settings, 'cpu')
        expected = torch.cat([torch.full((len(episode),), float(index))
                              for index, episode in enumerate(rollout.episodes)])
        torch.testing.assert_close(batch['return_c'], expected)

    def test_cost_monte_carlo_targets_keep_actor_gae_separate(self):
        agent = PredictiveMAPPO(configuration())
        rollout = collect_episodes(agent, make_env(configuration()), 2, 2.)
        for index, episode in enumerate(rollout.episodes):
            rollout.summaries[index]['collision'] = index
            for t, row in enumerate(episode):
                row['cost'] = float(index == 1 and t == len(episode)-1)
                row['value_c'] = 7. + t
        agent.settings.cost_gae_lambda = .95
        original = rollout.tensors(agent.settings, 'cpu')
        agent.settings.cost_monte_carlo_targets = True
        corrected = rollout.tensors(agent.settings, 'cpu')
        torch.testing.assert_close(corrected['advantage_c'], original['advantage_c'])
        expected = torch.cat([torch.full((len(episode),), float(index))
                              for index, episode in enumerate(rollout.episodes)])
        torch.testing.assert_close(corrected['return_c'], expected)
        self.assertFalse(torch.allclose(original['return_c'], expected))
        agent.settings.cost_gae_lambda = 1.
        monte_carlo = rollout.tensors(agent.settings, 'cpu')
        torch.testing.assert_close(monte_carlo['return_c'], corrected['return_c'])
        self.assertFalse(torch.allclose(monte_carlo['advantage_c'], corrected['advantage_c']))

    def test_prediction_epsilon_preserves_moments_and_other_groups(self):
        config = configuration()
        config['mappo']['prediction_lr'] = .001
        agent = PredictiveMAPPO(config)
        agent.update(collect_episodes(agent, make_env(config), 2, 2.))
        before = {p: copy.deepcopy(s) for p, s in agent.actor_optimizer.state.items()}
        agent.set_actor_learning_rates(prediction_eps=1e-8)
        self.assertEqual([g['eps'] for g in agent.actor_optimizer.param_groups], [1e-5, 1e-8])
        for parameter, state in before.items():
            for key, value in state.items():
                torch.testing.assert_close(agent.actor_optimizer.state[parameter][key], value)
        with tempfile.TemporaryDirectory(prefix='arboids-prediction-eps-') as directory:
            path = Path(directory)/'checkpoint.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
            self.assertEqual(restored.settings.prediction_eps, 1e-8)
            self.assertEqual([g['eps'] for g in restored.actor_optimizer.param_groups], [1e-5, 1e-8])
            restored.update(collect_episodes(restored, make_env(config), 2, 2.))
        with self.assertRaises(ValueError):
            PredictiveMAPPO(configuration()).set_actor_learning_rates(prediction_eps=1e-8)

    def test_training_accepts_undefined_explained_variance_for_zero_cost_batch(self):
        config = configuration()
        config['training'] = dict(total_steps=100, episodes_per_update=2, eval_interval=1,
                                  eval_episodes=1, agility=2.)
        config['mappo']['cost_monte_carlo_targets'] = True
        experiment = Mock()
        run_training(config, experiment, max_updates=1)
        metrics = experiment.record_metrics.call_args.kwargs
        self.assertIsNone(metrics['explained_variance_c'])
        self.assertEqual(metrics['update'], 1)

    def test_exploration_reset_preserves_deterministic_actions(self):
        agent = PredictiveMAPPO(configuration())
        env = make_env(configuration())
        obs = env.reset()
        states, boids = env.env.prediction_snapshot(), env.env.thrust_to_action(env.env.boids_actions)
        before, _ = agent.act(obs, states, boids, 0., True)
        reset_gate_exploration(agent, .25)
        after, record = agent.act(obs, states, boids, 0., True)
        np.testing.assert_array_equal(before, after)
        distribution, _ = agent.actor.gate_distribution(agent.tensor(obs)[None],
            agent.tensor(record['proposals'])[None], agent.tensor(boids)[None],
            agent.tensor(record['edges'])[None], agent.tensor(record['mask'], True)[None])
        torch.testing.assert_close(distribution.stddev, torch.full_like(distribution.stddev, .25))
        self.assertEqual(agent.training_state['stable_count'], 0)
        teacher = ActorAdap(6, 8, 3, 32)
        restore_proposal_exploration(agent, teacher.state_dict())
        proposal_restored, _ = agent.act(obs, states, boids, 0., True)
        np.testing.assert_array_equal(before, proposal_restored)
        torch.testing.assert_close(agent.actor.proposal_log_std.weight, teacher.log_std_layer.weight)


if __name__ == '__main__':
    unittest.main()
