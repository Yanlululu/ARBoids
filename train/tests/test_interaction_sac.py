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
from train_interaction import export_policy, arm_config


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

    def test_paired_gate_improvement_uses_returns_and_preserves_candidates(self):
        c = arm_config(config(), 'full')
        c['interaction'].update(entropy_objective='task-return-v4', gate_objective='paired-improvement-v1')
        agent = InteractionSAC(c)
        _, packet = scene()
        action, _ = agent.choose_action(packet, deterministic=True)
        alternative = action.copy(); alternative[0, 2] = 1.
        auxiliary = {key: value[None] for key, value in packet.items()}
        auxiliary.update(action=action[None], counterfactual=alternative[None],
            **{'return': np.array([[-20.]], dtype=np.float32),
               'counterfactual_return': np.array([[-10.]], dtype=np.float32)})
        initial = float(agent.paired_gate_loss(auxiliary).detach())
        swapped = dict(auxiliary, action=alternative[None], counterfactual=action[None])
        swapped['return'], swapped['counterfactual_return'] = auxiliary['counterfactual_return'], auxiliary['return']
        self.assertAlmostEqual(initial, float(agent.paired_gate_loss(swapped).detach()))
        proposals = copy.deepcopy(agent.actor.base.mean_layer.state_dict())
        agent.critic.forward = lambda *args: self.fail('Critic supplied a direct gate target')
        for _ in range(20):
            agent.actor_optimizer.zero_grad(set_to_none=True)
            agent.paired_gate_loss(auxiliary).backward()
            agent.actor_optimizer.step()
        self.assertLess(float(agent.paired_gate_loss(auxiliary).detach()), initial)
        after, _ = agent.choose_action(packet, deterministic=True)
        self.assertGreater(after[0, 2], action[0, 2])
        np.testing.assert_array_equal(after[:, :2], action[:, :2])
        for key, value in agent.actor.base.mean_layer.state_dict().items():
            torch.testing.assert_close(value, proposals[key], rtol=0, atol=0)
        bad = copy.deepcopy(auxiliary); bad['counterfactual'][0, 1, 2] += .1
        with self.assertRaisesRegex(ValueError, 'one focal gate'):
            agent.paired_gate_loss(bad)
        bootstrap_before = copy.deepcopy(agent.bootstrap_critic.state_dict())
        result = agent.learn(None, auxiliary, diagnostics=True)
        self.assertFalse(result['critic_updated']); self.assertFalse(result['bootstrap_updated'])
        self.assertTrue(result['actor_updated'])
        for key, value in agent.bootstrap_critic.state_dict().items():
            torch.testing.assert_close(value, bootstrap_before[key], rtol=0, atol=0)
        self.assertEqual(len(agent.critic_optimizer.state), 0)

    def test_task_return_has_no_entropy_reward_or_temperature_update(self):
        c = arm_config(config(), 'same_info'); c['interaction']['entropy_objective'] = 'task-return-v4'
        agent = InteractionSAC(c); env, packet = scene(); replay = JointReplay(4)
        for stage in ('gate', 'joint'):
            agent.set_stage(stage)
            action, logp = agent.choose_action(packet)
            self.assertEqual(logp, 0.)
            for _ in range(4): replay.store(packet, action, env.boids_actions, np.ones(3), packet, False)
            alpha = float(agent.alpha)
            agent.learn(replay)
            self.assertEqual(float(agent.alpha), alpha)
            self.assertEqual(len(agent.alpha_optimizer.state), 0)

    def test_team_intervention_preserves_candidates_and_supervises_the_joint_configuration(self):
        c = arm_config(config(), 'full')
        c['interaction'].update(intervention_scope='team', gate_objective='paired-improvement-v1',
                                entropy_objective='task-return-v4')
        agent = InteractionSAC(c); env, packet = scene()
        policy = FrozenPolicy(frozen_payload(agent))
        row = intervention_pair(policy, compact_snapshot(env), 930100, 131, None, (0., .5, 1.), 2)
        np.testing.assert_array_equal(row['counterfactual'][:, :2], row['action'][:, :2])
        np.testing.assert_array_equal(row['counterfactual'][:, 2], [0., .5, 1.])
        auxiliary = {k: np.asarray(row[k])[None] for k in
            ('obs', 'motion', 'central', 'action', 'counterfactual', 'return', 'counterfactual_return')}
        auxiliary['return'][:] = -30.; auxiliary['counterfactual_return'][:] = -10.
        before = float(agent.paired_gate_loss(auxiliary).detach())
        for _ in range(10):
            agent.actor_optimizer.zero_grad(set_to_none=True)
            agent.paired_gate_loss(auxiliary).backward(); agent.actor_optimizer.step()
        self.assertLess(float(agent.paired_gate_loss(auxiliary).detach()), before)
        with self.assertRaisesRegex(ValueError, 'legal reference per vessel'):
            intervention_pair(policy, compact_snapshot(env), 930100, 131, None, (0., 1.), 2)

    def test_skipped_actor_update_keeps_policy_and_temperature_exact(self):
        c = arm_config(config(), 'same_info'); c['interaction']['entropy_objective'] = 'stage-mean-v2'
        agent = InteractionSAC(c); env, packet = scene(); replay = JointReplay(4)
        action, _ = agent.choose_action(packet)
        for _ in range(4): replay.store(packet, action, env.boids_actions, np.ones(3), packet, False)
        before = copy.deepcopy(agent.actor.state_dict()); alpha = float(agent.alpha)
        result = agent.learn(replay, update_actor=False, diagnostics=True)
        self.assertFalse(result['actor_updated'])
        for key, value in agent.actor.state_dict().items(): torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(float(agent.alpha), alpha)
        self.assertEqual(len(agent.actor_optimizer.state), 0)
        self.assertGreater(result['bootstrap_gradient_norm'], 0.)

    def test_complete_task_rollout_is_negative_capped_time(self):
        c = config(); c['rl']['GAMMA'] = 1.
        c['interaction'].update(entropy_objective='task-return-v4', reward_objective='capped-time-v1',
                                deterministic_rollouts=True)
        policy = FrozenPolicy(frozen_payload(InteractionSAC(c)))
        env, packet = scene()
        action, logp = policy.action(packet, torch.Generator().manual_seed(13))
        other, _ = policy.action(packet, torch.Generator().manual_seed(71))
        np.testing.assert_array_equal(action, other); self.assertEqual(logp, 0.)
        value, steps, trace = branch_return(policy, compact_snapshot(env), action, 932113, 13, 300, trace=True)
        self.assertNotEqual(trace[-1]['outcome'], 0)
        expected = steps * .2 if trace[-1]['outcome'] == 3 else 60.
        self.assertAlmostEqual(value, -expected, places=4)

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

    def test_bootstrap_reduction_matches_real_targets_and_frozen_tails(self):
        env, packet = scene()
        for estimator, expected in (('min', 1.), ('mean', 2.)):
            c = config()
            c['interaction']['bootstrap_estimator'] = estimator
            agent = InteractionSAC(c)
            def constant_target():
                with torch.no_grad():
                    for parameter in agent.target.parameters(): parameter.zero_()
                    agent.target.q1.output[-1].bias.fill_(1.)
                    agent.target.q2.output[-1].bias.fill_(3.)
            for terminal in (False, True):
                replay = JointReplay(4)
                for _ in range(4):
                    action, _ = agent.choose_action(packet)
                    replay.store(packet, action, env.boids_actions, np.ones(3), packet, terminal)
                for update in ('learn', 'learn_bootstrap'):
                    constant_target()
                    obs = torch.as_tensor(packet['obs'])[None].expand(4, -1, -1)
                    motion = torch.as_tensor(packet['motion'])[None].expand(4, -1, -1)
                    torch.manual_seed(319)
                    with torch.no_grad(): _, logp = agent.actor(obs, motion)
                    target = 1. + agent.gamma * (1.-terminal) * (expected - agent.alpha * logp)
                    torch.manual_seed(319)
                    result = (agent.learn(replay, diagnostics=True) if update == 'learn'
                              else agent.learn_bootstrap(replay))
                    self.assertAlmostEqual(result['bootstrap_value_mean'], expected)
                    self.assertAlmostEqual(result['td_target_abs_max'], float(target.abs().max()), places=5)
            constant_target()
            frozen = FrozenPolicy(frozen_payload(agent))
            _, logp = frozen.action(packet, torch.Generator().manual_seed(741))
            self.assertAlmostEqual(frozen.value(packet, torch.Generator().manual_seed(741)),
                                   expected - frozen.alpha * logp, places=5)

    def test_bootstrap_estimator_change_requires_fresh_training(self):
        old = InteractionSAC(config()).state_dict()
        old.pop('bootstrap_estimator')  # Historical checkpoints retain minimum reduction.
        InteractionSAC(config()).load_state_dict(old)
        c = config(); c['interaction']['bootstrap_estimator'] = 'mean'
        with self.assertRaisesRegex(ValueError, 'Bootstrap estimator changed'):
            InteractionSAC(c).load_state_dict(old)
        c['interaction']['bootstrap_estimator'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Unknown bootstrap estimator'):
            InteractionSAC(c)

    def test_mean_entropy_is_fleet_invariant_and_excludes_frozen_proposals(self):
        actor = InteractionActor(32, 16, entropy_objective='stage-mean-v2')
        obs, motion = torch.zeros(1, 3, 18), torch.zeros(1, 3, 6)
        actor.freeze_proposals(True)
        action, gate_entropy = actor(obs, motion, deterministic=True)
        with torch.no_grad():
            actor.base.log_std_layer.bias.add_(.5)
        changed, same_entropy = actor(obs, motion, deterministic=True)
        torch.testing.assert_close(action, changed, rtol=0, atol=0)
        torch.testing.assert_close(gate_entropy, same_entropy, rtol=0, atol=0)
        _, large_entropy = actor(torch.zeros(1, 6, 24), torch.zeros(1, 6, 6), deterministic=True)
        torch.testing.assert_close(large_entropy, gate_entropy)
        actor.freeze_proposals(False)
        _, joint_entropy = actor(obs, motion, deterministic=True)
        with torch.no_grad():
            actor.base.log_std_layer.bias.sub_(.5)
        _, original_joint = actor(obs, motion, deterministic=True)
        torch.testing.assert_close(original_joint - joint_entropy, torch.ones_like(joint_entropy))

    def test_proposal_entropy_optimizes_task_only_gates_and_excludes_gate_noise(self):
        c = config(); c['interaction']['entropy_objective'] = 'proposal-mean-v3'
        agent = InteractionSAC(c)
        obs, motion = torch.zeros(1, 3, 18), torch.zeros(1, 3, 6)
        for stage in ('gate', 'joint'):
            agent.set_stage(stage)
            action, first = agent.actor(obs, motion, deterministic=True)
            with torch.no_grad():
                agent.actor.gate_log_std.bias.add_(1.)
                agent.actor.base.adap_layer.bias.add_(.5)
            changed, second = agent.actor(obs, motion, deterministic=True)
            self.assertFalse(torch.equal(action[..., 2], changed[..., 2]))
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            if stage == 'gate': self.assertEqual(float(second), 0.)
            else:
                with torch.no_grad(): agent.actor.base.log_std_layer.bias.sub_(.5)
                _, third = agent.actor(obs, motion, deterministic=True)
                torch.testing.assert_close(third-second, torch.ones_like(third))
            env, packet = scene()
            frozen = FrozenPolicy(frozen_payload(agent))
            noise = torch.randn((3, 3), generator=torch.Generator().manual_seed(123))
            actual, logp = agent.choose_action(packet, noise=noise)
            copied, copied_logp = frozen.action(packet, torch.Generator().manual_seed(123))
            np.testing.assert_array_equal(actual, copied)
            self.assertEqual(logp, copied_logp)
            replay = JointReplay(4)
            for _ in range(4): replay.store(packet, actual, env.boids_actions, np.ones(3), packet, False)
            before = float(agent.alpha)
            result = agent.learn(replay, diagnostics=True)
            if stage == 'gate':
                self.assertEqual(float(agent.alpha), before)
                self.assertEqual(result['entropy_cost_mean'], 0.)
                self.assertEqual(len(agent.alpha_optimizer.state), 0)
            else: self.assertNotEqual(float(agent.alpha), before)

    def test_new_entropy_is_consistent_in_rollout_stages_and_temperature_learning(self):
        c = arm_config(config(), 'same_info')
        c['interaction']['entropy_objective'] = 'stage-mean-v2'
        agent = InteractionSAC(c)
        env, packet = scene()
        replay = JointReplay(8)
        for _ in range(8):
            action, _ = agent.choose_action(packet)
            replay.store(packet, action, env.boids_actions, np.ones(3), packet, False)
        for stage in ('gate', 'joint'):
            agent.set_stage(stage)
            frozen = FrozenPolicy(frozen_payload(agent))
            noise = torch.randn((3, 3), generator=torch.Generator().manual_seed(123))
            action, logp = agent.choose_action(packet, noise=noise)
            copied, copied_logp = frozen.action(packet, torch.Generator().manual_seed(123))
            np.testing.assert_array_equal(copied, action)
            self.assertEqual(copied_logp, logp)
            self.assertEqual(frozen.actor.stage, stage)
            before = float(agent.alpha)
            agent.learn(replay)
            self.assertNotEqual(float(agent.alpha), before)
            restored = InteractionSAC(c)
            restored.load_state_dict(copy.deepcopy(agent.state_dict()))
            self.assertEqual(restored.actor.stage, stage)
            self.assertEqual(float(restored.alpha), float(agent.alpha))

    def test_entropy_revision_cannot_reinterpret_existing_values(self):
        old = InteractionSAC(config())
        c = config()
        c['interaction']['entropy_objective'] = 'stage-mean-v2'
        new = InteractionSAC(c)
        with self.assertRaisesRegex(ValueError, 'Entropy objective changed'):
            new.load_state_dict(old.state_dict())
        with self.assertRaisesRegex(ValueError, 'Entropy objective changed'):
            old.load_state_dict(new.state_dict())

    def test_complete_real_returns_exclude_root_entropy_and_unfinished_episodes(self):
        env, packet = scene()
        action = np.zeros((3, 3), dtype=np.float32)
        replay = JointReplay(8)
        for reward, done, cost in ((1.,False,50.),(2.,True,.5),(3.,True,100.),(4.,False,999.)):
            replay.store(packet,action,env.boids_actions,np.full(3,reward),packet,done,policy_cost=cost)
        np.testing.assert_allclose(replay.complete_returns(.9),[2.35,2.,3.])
        restored=JointReplay(1);restored.load_state_dict(replay.state_dict())
        np.testing.assert_array_equal(replay.complete_returns(.9),restored.complete_returns(.9))
        unfinished=JointReplay(8)
        unfinished.store(packet,action,env.boids_actions,np.ones(3),packet,False,policy_cost=1.)
        with self.assertRaisesRegex(ValueError,'No completed real episode'):
            unfinished.complete_returns(.9)

    def test_real_return_warmup_never_uses_a_bootstrap_tail_or_updates_the_actor(self):
        agent=InteractionSAC(config());env,packet=scene();replay=JointReplay(8)
        for index in range(4):
            action,logp=agent.choose_action(packet)
            replay.store(packet,action,env.boids_actions,np.ones(3),packet,index%2==1,
                         policy_cost=float(agent.alpha)*logp)
        returns=replay.complete_returns(agent.gamma)
        actor=copy.deepcopy(agent.actor.state_dict());critic=copy.deepcopy(agent.critic.state_dict())
        def reject_tail(*args): self.fail('Real-return initializer read a learned tail')
        agent.target.forward=reject_tail
        for _ in range(3): checks=agent.learn_bootstrap_returns(replay,returns)
        self.assertEqual(checks['complete_return_samples'],4)
        self.assertTrue(np.isfinite(checks['bootstrap_mc_loss']))
        for name,wanted in (('actor',actor),('critic',critic)):
            for key,value in getattr(agent,name).state_dict().items():
                torch.testing.assert_close(value,wanted[key],rtol=0,atol=0)

    def test_state_return_initialization_cannot_invent_control_sensitivities(self):
        c=config();c['training']['critic_warmup_target']='complete_real_state_return'
        c['interaction']['critic_control_coordinates']='nominal-thrust-v2'
        agent=InteractionSAC(c);env,packet=scene();replay=JointReplay(8)
        for i in range(4):
            action,logp=agent.choose_action(packet)
            replay.store(packet,action,env.boids_actions,np.full(3,i+1),packet,i%2==1,policy_cost=.2*logp)
        returns=replay.complete_returns(agent.gamma)
        p={k:torch.as_tensor(v,dtype=torch.float32)[None] for k,v in packet.items()}
        left=torch.zeros((1,3,3));right=torch.ones_like(left)
        for _ in range(3):
            agent.learn_bootstrap_returns(replay,returns)
            for a,b in zip(agent.bootstrap_critic(p,left),agent.bootstrap_critic(p,right)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        agent.initialize_critic_from_bootstrap()
        pool=SnapshotPool(1);pool.add(env)
        auxiliary,_=InterventionSampler(0).generate(agent,pool)
        agent.learn(replay,auxiliary)
        self.assertGreater(float(agent.critic.q1.node[0].weight[:,7:10].abs().max()),0.)
        self.assertGreater(float(agent.bootstrap_critic.q1.node[0].weight[:,7:10].abs().max()),0.)

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
        self.assertEqual(float(logp.detach()), 0.)
        action, _ = actor(obs, motion, deterministic=True)
        action[..., 2].sum().backward()
        self.assertGreater(float(actor.relations[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(actor.base.mean_layer.weight.grad.abs().sum()), 0.)

    def test_no_peer_ablation_removes_only_peer_candidate_dependence(self):
        _, p = scene()
        obs, motion = torch.as_tensor(p['obs'])[None], torch.as_tensor(p['motion'], dtype=torch.float32)[None]
        full = InteractionActor(32, 16)
        torch.nn.init.normal_(full.relation_gate.weight, std=.2)
        masked = InteractionActor(32, 16, peer_candidates=False)
        masked.load_state_dict(full.state_dict(), strict=True)
        self.assertEqual(sum(x.numel() for x in full.parameters()), sum(x.numel() for x in masked.parameters()))
        mask = torch.ones((1, 3), dtype=torch.bool)
        noise = torch.randn((1, 3, 1))
        proposals = torch.randn((1, 3, 2)).tanh().requires_grad_()
        boids = obs[..., 12:14].clone().requires_grad_()
        for actor, expect_peer in ((full, True), (masked, False)):
            h = actor.encode(obs, mask)
            gates, _ = actor.gates(h, motion, proposals, boids, mask, noise=noise)
            gradients = torch.autograd.grad(gates[0, 0], (proposals, boids), retain_graph=True)
            self.assertGreater(sum(float(g[0, 0].abs().sum()) for g in gradients), 0.)
            peer_gradient = sum(float(g[0, 1:].abs().sum()) for g in gradients)
            self.assertEqual(peer_gradient > 0., expect_peer)
            changed, changed_boids = proposals.detach().clone(), boids.detach().clone()
            changed[:, 1:] *= -1
            changed_boids[:, 1:] += .2
            other, _ = actor.gates(h, motion, changed, changed_boids, mask, noise=noise)
            self.assertEqual(bool(torch.equal(gates[:, 0], other[:, 0])), not expect_peer)

    def test_no_peer_training_rollout_and_deployment_use_the_same_mask(self):
        from policy.interaction_sac import make_agent
        full_config = arm_config(config(), 'full')
        masked_config = arm_config(config(), 'no_peer')
        expected = copy.deepcopy(full_config)
        expected['interaction'].update(arm='no_peer', peer_candidates=False)
        self.assertEqual(masked_config, expected)
        agent = make_agent(masked_config)
        torch.nn.init.normal_(agent.actor.relation_gate.weight, std=.1)
        frozen = FrozenPolicy(frozen_payload(agent))
        self.assertFalse(frozen.actor.peer_candidates)
        _, packet = scene()
        with tempfile.TemporaryDirectory() as directory:
            export_policy(agent, directory, 0)
            deployment = DeploymentPolicy(Path(directory)/'policy.pth')
            self.assertFalse(deployment.actor.peer_candidates)
            wanted, _ = agent.choose_action(packet, deterministic=True)
            got, _ = deployment.choose_action(packet)
            np.testing.assert_array_equal(wanted, got)
        noise = torch.randn((1, 3, 3), generator=torch.Generator().manual_seed(123))
        wanted, _ = agent.choose_action(packet, noise=noise[0])
        got, _ = frozen.action(packet, torch.Generator().manual_seed(123))
        np.testing.assert_array_equal(wanted, got)

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

    def test_auxiliary_labels_cannot_update_the_bootstrap_teacher(self):
        c = config()
        c['interaction']['bootstrap_source'] = 'real_td'
        c['interaction']['bootstrap_estimator'] = 'mean'
        left, right = InteractionSAC(c), InteractionSAC(c)
        env, packet = scene()
        replay = JointReplay(8)
        for _ in range(8):
            action, _ = left.choose_action(packet)
            replay.store(packet, action, env.boids_actions, np.ones(3), packet, False)
        for _ in range(3): left.learn_bootstrap(replay)
        left.initialize_critic_from_bootstrap()
        right.load_state_dict(copy.deepcopy(left.state_dict()))
        pool = SnapshotPool(1)
        pool.add(env)
        auxiliary, _ = InterventionSampler(0).generate(left, pool)
        changed = copy.deepcopy(auxiliary)
        changed['counterfactual_return'] += 1000.
        actor = copy.deepcopy(left.actor.state_dict())
        for _ in range(3):
            # Hold the continuation policy fixed to isolate the direct label path.
            left.actor.load_state_dict(actor)
            right.actor.load_state_dict(actor)
            state = RandomState.capture()
            left.learn(replay, auxiliary)
            state.restore()
            right.learn(replay, changed)
            for name in ('bootstrap_critic', 'target'):
                for key, value in getattr(left, name).state_dict().items():
                    torch.testing.assert_close(value, getattr(right, name).state_dict()[key], rtol=0, atol=0)
            for key,values in left.bootstrap_optimizer.state_dict()['state'].items():
                for field,value in values.items():
                    torch.testing.assert_close(value,right.bootstrap_optimizer.state_dict()['state'][key][field],rtol=0,atol=0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(left.critic.parameters(), right.critic.parameters())))

    def test_same_info_real_td_teacher_preserves_the_original_update(self):
        c = arm_config(config(), 'same_info')
        c['interaction']['bootstrap_source'] = 'real_td'
        new = InteractionSAC(c)
        old_config = copy.deepcopy(c)
        old_config['interaction']['bootstrap_source'] = 'coupled'
        old = InteractionSAC(old_config)
        state = copy.deepcopy(new.state_dict())
        del state['bootstrap_critic'], state['bootstrap_optimizer']
        old.load_state_dict(state)
        env, packet = scene()
        replay = JointReplay(8)
        for _ in range(8):
            action, _ = new.choose_action(packet)
            replay.store(packet, action, env.boids_actions, np.ones(3), packet, False)
        for _ in range(3):
            random = RandomState.capture()
            new.learn(replay)
            random.restore()
            old.learn(replay)
            for name in ('actor', 'critic', 'target'):
                for key, value in getattr(new, name).state_dict().items():
                    torch.testing.assert_close(value, getattr(old, name).state_dict()[key], rtol=0, atol=0)

    def test_restore_breaks_historical_optimizer_state_aliases(self):
        c=arm_config(config(),'same_info');agent=InteractionSAC(c)
        env,packet=scene();replay=JointReplay(8)
        for _ in range(8):
            action,_=agent.choose_action(packet)
            replay.store(packet,action,env.boids_actions,np.ones(3),packet,False)
        for _ in range(3): agent.learn_bootstrap(replay)
        agent.initialize_critic_from_bootstrap()
        saved=agent.state_dict()
        saved['bootstrap_optimizer']=saved['critic_optimizer']
        restored=InteractionSAC(c);restored.load_state_dict(saved)
        checks=restored.learn(replay,diagnostics=True)
        self.assertEqual(checks['bootstrap_optimizer_aliases'],0)
        self.assertEqual(checks['same_info_critic_max_difference'],0.)

    def test_bootstrap_protocol_cannot_silently_reinterpret_old_checkpoints(self):
        c = config()
        c['interaction']['bootstrap_source'] = 'coupled'
        old = InteractionSAC(c)
        c['interaction']['bootstrap_source'] = 'real_td'
        new = InteractionSAC(c)
        with self.assertRaisesRegex(ValueError, 'explicit checkpoint reconditioning'):
            new.load_state_dict(old.state_dict())

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

    def test_composed_critic_obeys_physical_action_aliases_and_zero_gate_effect(self):
        from policy.interaction_sac import tensor_packet
        packet={k:v.unsqueeze(0) for k,v in tensor_packet(self.packet).items()}
        boids=packet['obs'][...,12:14]
        first=torch.cat((.5*boids,torch.ones_like(boids[...,:1])),dim=-1)
        second=torch.cat((torch.zeros_like(boids),.5*torch.ones_like(boids[...,:1])),dim=-1)
        critic=TwinTeamCritic(32,16,'nominal-thrust-v2')
        for left,right in zip(critic(packet,first),critic(packet,second)):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        agreed=torch.cat((boids,.4*torch.ones_like(boids[...,:1])),dim=-1).requires_grad_()
        critic(packet,agreed)[0].sum().backward()
        torch.testing.assert_close(agreed.grad[...,2],torch.zeros_like(agreed.grad[...,2]),rtol=0,atol=0)
        raw=TwinTeamCritic(32,16)
        with self.assertRaises(RuntimeError): critic.load_state_dict(raw.state_dict())
        c=config();c['interaction']['critic_control_coordinates']='nominal-thrust-v2'
        revised=InteractionSAC(c)
        with self.assertRaisesRegex(ValueError,'fresh matched training'):
            revised.load_state_dict(self.agent.state_dict())
        restored=FrozenPolicy(frozen_payload(revised))
        self.assertEqual(restored.critic.q1.control_coordinates,'nominal-thrust-v2')

    def test_repeated_labels_fix_root_actions_and_pair_independent_futures(self):
        from unittest.mock import patch
        from interaction_rollout import repeated_intervention_pair
        calls=[]
        def consequence(policy,snapshot,action,future,noise,steps,**kw):
            calls.append((action.copy(),future,noise))
            return float(action[:,2].sum())+float(noise%37),steps,[]
        rng=RandomState.capture()
        with patch('interaction_rollout.branch_return',side_effect=consequence):
            row=repeated_intervention_pair(self.policy,self.snapshot,123,456,1,0.,3,4)
        after=np.random.rand(3),torch.randn(3);rng.restore()
        np.testing.assert_array_equal(after[0],np.random.rand(3))
        torch.testing.assert_close(after[1],torch.randn(3),rtol=0,atol=0)
        self.assertEqual(row['simulated_steps'],24)
        self.assertEqual(len({x[1:] for x in calls}),4)
        for i in range(0,8,2):
            self.assertEqual(calls[i][1:],calls[i+1][1:])
            np.testing.assert_array_equal(calls[i][0],calls[0][0])
            np.testing.assert_array_equal(calls[i+1][0],calls[1][0])
        self.assertAlmostEqual(float(row['return'][0]-row['counterfactual_return'][0]),float(row['action'][1,2]),places=5)
        self.assertAlmostEqual(float(row['difference_standard_error'][0]),0.,places=6)


if __name__ == '__main__':
    unittest.main()
