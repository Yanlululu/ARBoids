"""Behavioral and integration contracts for physical two-stage MAPPO."""
import ast
import copy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
from envs.TADgame import TADEnv
from RL.channel_networks import ChannelActor, TeamCritic
from RL.control import fuse_thrust, to_channels, from_channels, to_action, to_thrust
from RL.deployment import DeployedPolicy
from RL.guidance import preserve_rng, search_gates, branch_return
from RL.MAPPO import MAPPO
from RL.collector import TeamCollector, CollisionStarts, encounter_priority
from RL.calibration import TerminalCalibration
from RL.observations import collate_frames, tensor_frame
from RL.rollout import TeamRollout, compute_gae

torch.set_num_threads(1)


def environment(n=3, **kwargs):
    env = TADEnv(n, protocol="paper-parameters-v1", initial_min_spacing=6., canonical_agent_order=True, **kwargs)
    env.reset(2., noisy_agility=False)
    return env


def permute_frame(frame, permutation):
    result = {}
    for key, value in frame.items():
        if key == "global_state":
            result[key] = value.copy()
        elif key in ("edges", "neighbors"):
            result[key] = value[permutation][:, permutation].copy()
        else:
            result[key] = value[permutation].copy()
    return result


class FusionTests(unittest.TestCase):
    def test_scalar_reduction_equal_candidates_limits_and_example(self):
        torch.manual_seed(6)
        b, l = torch.rand(1000, 2) * 1500 - 500, torch.rand(1000, 2) * 1500 - 500
        g = torch.rand(1000, 1)
        torch.testing.assert_close(fuse_thrust(b, l, g.expand(-1, 2)), b + g * (l - b), atol=2e-4, rtol=1e-5)
        torch.testing.assert_close(fuse_thrust(b, b, torch.rand_like(b)), b, atol=1e-4, rtol=1e-5)
        result = fuse_thrust(b, l, torch.rand_like(b))
        self.assertTrue(bool(((result >= -500) & (result <= 1000)).all()))
        actual = fuse_thrust(torch.tensor([420., 780.]), torch.tensor([420., 180.]), torch.tensor([0., 1.]))
        torch.testing.assert_close(actual, torch.tensor([720., 480.]))

    def test_projection_is_nearest_feasible_point(self):
        b = torch.tensor([1000., 1000.], dtype=torch.float64)
        l = torch.tensor([-500., 1000.], dtype=torch.float64)
        g = torch.tensor([0., 1.], dtype=torch.float64)
        result = fuse_thrust(b, l, g)
        raw_cd = to_channels(b) + g * (to_channels(l) - to_channels(b))
        projected_distance = (to_channels(result) - raw_cd).square().sum()
        candidates = torch.rand(5000, 2, dtype=torch.float64) * 1500 - 500
        self.assertTrue(bool(((to_channels(candidates) - raw_cd).square().sum(-1) >= projected_distance - 1e-8).all()))

    def test_asymmetric_normalization(self):
        u = torch.tensor([[-500., 1000.], [720., 480.]])
        torch.testing.assert_close(to_thrust(to_action(u)), u)
        torch.testing.assert_close(from_channels(to_channels(u)), u)
        self.assertEqual(float(to_thrust(torch.tensor(0.))), 250.)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(21)
        torch.manual_seed(21)
        self.env = environment()
        self.frame = self.env.structured_frame()
        self.actor = ChannelActor(hidden=32)

    def test_historical_likelihood_and_gate_policy_gradient(self):
        f = tensor_frame(self.frame)
        with torch.no_grad():
            decision = self.actor.act(f)
        logp, _ = self.actor.evaluate_actions(f, decision, entropy=False)
        torch.testing.assert_close(logp, decision["logp_candidate"] + decision["logp_gate"])
        loss = -(logp * torch.tensor([[[1.], [-1.], [2.]]])).mean()
        loss.backward()
        self.assertGreater(float(self.actor.gate_head[-1].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.actor.proposal_head[-1].weight.grad.abs().sum()), 0.)

    def test_guide_updates_adapter_only(self):
        f = tensor_frame(self.frame)
        with torch.no_grad():
            decision = self.actor.act(f)
        _, u = self.actor.representative(f, decision["candidate"], decision["messages"], adapter_only=True)
        (u / 750.).square().mean().backward()
        for module in (self.actor.encoder, self.actor.proposal_neighbors, self.actor.proposal_head):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertGreater(float(self.actor.gate_head[-1].weight.grad.abs().sum()), 0.)

    def test_actor_equivariance_and_critic_invariance(self):
        permutation = np.array([2, 0, 1])
        f, p = tensor_frame(self.frame), tensor_frame(permute_frame(self.frame, permutation))
        with torch.no_grad():
            a, b = self.actor.act(f, True), self.actor.act(p, True)
            torch.testing.assert_close(a["executed"][:, permutation], b["executed"], atol=2e-4, rtol=1e-5)
            for conditioned in (False, True):
                critic = TeamCritic(32, conditioned)
                torch.testing.assert_close(critic(f, a["executed"]), critic(p, b["executed"]), atol=1e-6, rtol=1e-5)

    def test_single_agent_padding_and_no_central_state_leak(self):
        single = environment(1).structured_frame()
        combined = collate_frames([single, self.frame])
        with torch.no_grad():
            alone = self.actor.act(tensor_frame(single), True)
            padded = self.actor.act(combined, True)
            torch.testing.assert_close(alone["executed"][0, 0], padded["executed"][0, 0], atol=2e-4, rtol=1e-5)
            self.assertTrue(torch.isfinite(padded["executed"]).all())
            self.assertEqual(float(padded["attention_c"][0].sum()), 0.)
            altered = tensor_frame(self.frame)
            original = self.actor.act(altered, True)["executed"]
            altered["nodes"].fill_(123.)
            altered["global_state"].fill_(-456.)
            torch.testing.assert_close(original, self.actor.act(altered, True)["executed"])

    def test_candidate_exchange_is_explicit_and_fixed(self):
        f = tensor_frame(self.frame)
        with torch.no_grad():
            decision = self.actor.act(f)
        messages = decision["messages"].clone()
        self.actor.proposal_head[-1].bias.data[:2] += .5
        self.actor.evaluate_actions(f, decision, entropy=False)
        torch.testing.assert_close(messages, decision["messages"])
        _, modified, _ = self.actor.propose(f, True)
        self.assertFalse(torch.allclose(messages, self.actor.exchange(f, modified)))


class EnvironmentTests(unittest.TestCase):
    def test_actual_actions_snapshot_restore_and_true_timeout(self):
        np.random.seed(12)
        env = environment(total_time=.2)
        snap = env.snapshot()
        action = np.array([[1800., -800.], [250., 300.], [500., 250.]])
        first = env.step_thrust(action)
        first_state = env.structured_frame()
        env.restore(snap)
        second = env.step_thrust(action)
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)
        for key in first_state:
            np.testing.assert_array_equal(first_state[key], env.structured_frame()[key])
        np.testing.assert_array_equal(second[3]["executed_thrust"], np.clip(action, -500., 1000.))
        self.assertEqual(second[2], 4)
        self.assertTrue(second[3]["terminated"])
        self.assertFalse(second[3]["truncated"])

    def test_ten_agents_have_legal_initial_spacing(self):
        self.assertGreaterEqual(environment(10).def_def_dists.min(), 6. - 1e-10)

    def test_canonical_environment_permutation(self):
        env = environment()
        snap = env.snapshot()
        action = np.array([[120., 200.], [250., 230.], [300., 450.]])
        env.step_thrust(action)
        expected = np.array([d.pos for d in env.defender_list])
        expected_attacker = env.attacker.pos.copy()
        env.restore(snap)
        permutation = [2, 0, 1]
        env.defender_list = [env.defender_list[i] for i in permutation]
        env.step_thrust(action[permutation])
        np.testing.assert_allclose(np.array([d.pos for d in env.defender_list]), expected[permutation], atol=1e-10)
        np.testing.assert_allclose(env.attacker.pos, expected_attacker, atol=1e-10)

    def test_branch_does_not_touch_main_environment_or_rng(self):
        env = environment()
        state = env.snapshot()
        actor = ChannelActor(hidden=32)
        value = TeamCritic(32)
        with torch.no_grad():
            action = actor.act(tensor_frame(env.structured_frame()), True)["executed"][0].numpy()
        before_np, before_torch = np.random.get_state(), torch.get_rng_state().clone()
        with preserve_rng():
            result = branch_return(copy.deepcopy(env), state, action, actor, value, .99, 3, "cpu")
        self.assertEqual(result[1], 3)
        np.testing.assert_array_equal(before_np[1], np.random.get_state()[1])
        self.assertEqual(before_np[2:], np.random.get_state()[2:])
        torch.testing.assert_close(before_torch, torch.get_rng_state())
        self.assertEqual(env.Current_T, state["state"]["Current_T"])
        np.testing.assert_array_equal(env.attacker.pos, state["state"]["attacker"].pos)


class TrainingTests(unittest.TestCase):
    def test_collision_cost_uses_episode_outcomes_and_keeps_task_rewards_separate(self):
        config = dict(policy=dict(hidden=32), mappo=dict(epochs=1, minibatch_steps=8,
                      collision_cost_coefficient=20., cost_gae_lambda=1.), guidance=dict(enabled=False))
        agent = MAPPO(config)
        calls = [0]

        def reset():
            env = environment(total_time=.2)
            calls[0] += 1
            if calls[0] % 2:
                env.Collision_R = 1000.
            return env

        collector = TeamCollector(agent, reset, 2)
        rollout, _ = collector.collect(8)
        before = rollout.batch("cpu", .99, .95, cost_lam=1.)
        expected = torch.tensor([[float(r["outcome"] == 2)] for r in rollout.rows])
        torch.testing.assert_close(before["cost_returns"], expected)
        self.assertTrue(all(r["next_cost_value"] == 0. for r in rollout.rows))
        self.assertGreater(float(before["cost_returns"].sum()), 0.)
        self.assertLess(float(before["cost_returns"].sum()), len(rollout.rows))
        metrics = agent.update(rollout, collector.envs[0])
        self.assertGreater(metrics["cost_value_loss"], 0.)
        self.assertEqual(metrics["collision_cost_coefficient"], 20.)
        torch.testing.assert_close(before["reward"], rollout.batch("cpu", .99, .95)["reward"], rtol=0., atol=0.)
        checkpoint = io.BytesIO()
        agent.save(checkpoint)
        checkpoint.seek(0)
        loaded = MAPPO.load(checkpoint)
        torch.testing.assert_close(agent.collision_probability(before["frame"]),
                                   loaded.collision_probability(before["frame"]))
        live, _ = TeamCollector(loaded, lambda: environment(total_time=10.)).collect(1)
        values = live.batch("cpu", .99, .95, cost_lam=1.)
        self.assertAlmostEqual(float(values["cost_returns"][0]), live.rows[0]["next_cost_value"], places=6)

    def test_collision_start_curriculum_uses_new_on_policy_actions_and_retires_solved_cases(self):
        config = dict(policy=dict(hidden=32), mappo=dict(epochs=1, minibatch_steps=8),
                      training=dict(collision_revisit_probability=1., collision_revisit_capacity=2),
                      guidance=dict(enabled=False))
        agent = MAPPO(config)

        def collision_env():
            env = environment()
            env.Collision_R = 1000.  # Force a real collision terminal on the first test step.
            return env

        collector = TeamCollector(agent, collision_env)
        initial_positions = np.array([v.pos.copy() for v in collector.envs[0].defender_list])
        seed = collector.episode_seeds[0]
        rollout, stats = collector.collect(8)
        self.assertEqual(stats["collision_revisit_episodes"], 8)
        self.assertTrue(all(row["outcome"] == 2 for row in rollout.rows))
        np.testing.assert_array_equal(initial_positions, [v.pos for v in collector.envs[0].defender_list])
        self.assertFalse(np.array_equal(rollout.rows[0]["decision"]["candidate"],
                                       rollout.rows[1]["decision"]["candidate"]))
        self.assertLess(agent.update(rollout, collector.envs[0])["likelihood_error"], 1e-4)
        loaded = CollisionStarts(collision_env, probability=1., state=collector.starts.state_dict())
        self.assertEqual(loaded.cases, collector.starts.cases)
        loaded.observe(seed, 3)
        loaded.observe(seed, 1)
        loaded.observe(seed, 4)
        loaded.observe(seed, 3)
        self.assertIn(seed, loaded.cases)  # Breach broke the successful streak.
        loaded.observe(seed, 3)
        self.assertNotIn(seed, loaded.cases)
        for extra in (10, 11, 12):
            collector.starts.observe(extra, 2)
        self.assertEqual(list(collector.starts.cases), [11, 12])

    def test_terminal_q_labels_are_exact_bounded_and_separate_from_ppo(self):
        config = dict(policy=dict(hidden=32), mappo=dict(epochs=1, minibatch_steps=8,
                      value_normalization=True, critic_layer_norm=True, terminal_q_updates=2,
                      terminal_q_interleave_weight=.25, terminal_q_full_probe=True),
                      guidance=dict(enabled=False))
        agent = MAPPO(config)
        collector = TeamCollector(agent, lambda: environment(total_time=.4), 2)
        rollout, _ = collector.collect(8)
        before = rollout.batch("cpu", .99, .95)
        metrics = agent.update(rollout, collector.envs[0])
        after = rollout.batch("cpu", .99, .95)
        for key in ("returns", "advantage", "reward", "terminal"):
            torch.testing.assert_close(before[key], after[key], rtol=0., atol=0.)
        self.assertEqual(metrics["terminal_q_samples"], 4)
        self.assertEqual(metrics["terminal_q_updates"], 2)
        probe = agent.terminal_calibration.probe("cpu")
        self.assertAlmostEqual(float(probe[3].sum()), 1.)
        self.assertEqual(len(probe[2]), 4)
        with torch.no_grad():
            exact_rmse = ((agent.q(probe[0], probe[1]) - probe[2]).square() * probe[3]).sum().sqrt()
        self.assertAlmostEqual(metrics["terminal_q_rmse_after"], float(exact_rmse), places=5)
        self.assertFalse(agent.terminal_q_ready)  # Denials alone do not certify capture/collision estimates.
        replay = TerminalCalibration(capacity=2)
        replay.add(rollout.rows)
        self.assertEqual(len(replay), 2)
        for entry, original in zip(replay.groups[4], [row for row in rollout.rows if row["terminal"]][-2:]):
            self.assertEqual(entry["reward"], original["reward"])
            np.testing.assert_array_equal(entry["executed"].numpy(), original["decision"]["executed"])
        collision = copy.deepcopy(next(row for row in rollout.rows if row["terminal"]))
        collision.update(outcome=2, reward=-33.)
        replay.add([collision])
        mixed_probe = replay.probe("cpu")
        # One collision and two denials each receive half the probe mass by
        # outcome, so repeated common terminal labels cannot hide rare errors.
        self.assertAlmostEqual(float(mixed_probe[3][mixed_probe[2] == -33.].sum()), .5)
        self.assertAlmostEqual(float(mixed_probe[3][mixed_probe[2] != -33.].sum()), .5)
        checkpoint = io.BytesIO()
        agent.save(checkpoint)
        checkpoint.seek(0)
        loaded = MAPPO.load(checkpoint)
        self.assertEqual(len(loaded.terminal_calibration), 4)
        for old, new in zip(agent.terminal_calibration.groups[4], loaded.terminal_calibration.groups[4]):
            self.assertEqual(old["reward"], new["reward"])
            torch.testing.assert_close(old["executed"], new["executed"])

    def test_priority_queries_preserve_on_policy_trajectory_and_bound_storage(self):
        agent = MAPPO(dict(policy=dict(hidden=32), guidance=dict(enabled=False)))
        with preserve_rng():
            ordinary, _ = TeamCollector(agent, lambda: environment(total_time=2.), 3).collect(36, {0, 35})
        with preserve_rng():
            prioritized, stats = TeamCollector(agent, lambda: environment(total_time=2.), 3).collect(36, {0, 35}, 5)
        self.assertLessEqual(stats["priority_snapshots"], 5)
        self.assertGreater(stats["priority_snapshots"], 0)
        for a, b in zip(ordinary.rows, prioritized.rows):
            for key in ("reward", "outcome", "value", "next_value"):
                self.assertEqual(a[key], b[key])
            for key in a["decision"]:
                np.testing.assert_array_equal(a["decision"][key], b["decision"][key])
        selected = [encounter_priority(row["frame"]) for i, row in enumerate(prioritized.rows)
                    if i not in (0, 35) and row["snapshot"] is not None]
        other = [encounter_priority(row["frame"]) for i, row in enumerate(prioritized.rows)
                 if i not in (0, 35) and row["snapshot"] is None]
        self.assertGreaterEqual(min(selected), max(other))
        frame = environment().structured_frame()
        frame["edges"][..., 2:4] = frame["edges"][..., :2]
        self.assertEqual(encounter_priority(frame), 0.)
        frame["edges"][..., 2:4] *= -1
        self.assertGreater(encounter_priority(frame), 0.)
        self.assertAlmostEqual(encounter_priority(frame), encounter_priority(permute_frame(frame, [2, 0, 1])))

    def test_priority_lead_time_selects_actionable_approach_before_collision(self):
        near = environment(2).structured_frame()
        near["edges"].fill(0.)
        near["edges"][0, 1, :4] = [5.3 / 60., 0., -3. / 5., 0.]
        near["edges"][1, 0, :4] = [-5.3 / 60., 0., 3. / 5., 0.]
        early = {key: value.copy() for key, value in near.items()}
        early["edges"][..., :2] *= 9.5 / 5.3
        self.assertGreater(encounter_priority(near), encounter_priority(early))
        self.assertGreater(encounter_priority(early, lead_time=1.5),
                           encounter_priority(near, lead_time=1.5))
        self.assertAlmostEqual(encounter_priority(early, lead_time=1.5),
                               encounter_priority(permute_frame(early, [1, 0]), lead_time=1.5))
        early["edges"][..., 2:4] *= -1
        self.assertEqual(encounter_priority(early, lead_time=1.5), 0.)

    def test_validated_guidance_batch_is_used_with_sparse_targets(self):
        config = dict(policy=dict(hidden=32), mappo=dict(epochs=1, minibatch_steps=4),
                      guidance=dict(enabled=False, batch_mode="validated"))
        agent = MAPPO(config)
        collector = TeamCollector(agent, lambda: environment(total_time=2.), 2)
        rollout, _ = collector.collect(8)

        def labels(agent, rollout, batch, env):
            target = (batch["decision"]["executed"] + 100.).clamp(-500., 1000.)
            weight = torch.zeros(8)
            weight[-1] = 1.
            return target, weight, [], {}

        agent.guidance.prepare = labels
        metrics = agent.update(rollout, collector.envs[0])
        self.assertGreater(metrics["guide_loss"], 0.)
        self.assertEqual(metrics["guidance_weight"], .125)
        self.assertLess(metrics["likelihood_error"], 1e-4)

    def test_acceptance_rejects_tied_collisions_or_lost_successes(self):
        from qualify_policy import compare_paired

        def outcomes(codes):
            return [dict(seed=100 + i, outcome_code=code, success=int(code in (3, 4)))
                    for i, code in enumerate(codes)]

        baseline = outcomes([3, 4, 2, 2])
        self.assertFalse(compare_paired(outcomes([3, 4, 2, 2]), baseline)["passed"])
        self.assertFalse(compare_paired(outcomes([3, 1, 1, 2]), baseline)["passed"])
        self.assertTrue(compare_paired(outcomes([3, 4, 1, 2]), baseline)["passed"])
        with self.assertRaises(ValueError):
            compare_paired(outcomes([3, 4, 2]), baseline)

    def test_collision_selection_preserves_baseline_success_before_reducing_collisions(self):
        from train_mappo import selection_score
        options = dict(selection_objective="collision", selection_success_floor=.9375)
        below_floor = dict(success_rate=.93, collisions=0, mean_reward=100.)
        safe = dict(success_rate=.9375, collisions=0, mean_reward=-10.)
        fast = dict(success_rate=.99, collisions=1, mean_reward=100.)
        self.assertGreater(selection_score(fast, options), selection_score(below_floor, options))
        self.assertGreater(selection_score(safe, options), selection_score(fast, options))
        self.assertGreater(selection_score(fast, {}), selection_score(safe, {}))
        with self.assertRaises(ValueError):
            selection_score(safe, dict(selection_objective="collision"))

    def test_oversized_actor_update_restores_weights_and_adam_state(self):
        config = dict(policy=dict(hidden=32), mappo=dict(epochs=1, minibatch_steps=4, actor_minibatch_steps=8,
                      learning_rate=.1, target_kl=1e-10, max_kl_backtracks=0),
                      guidance=dict(enabled=False))
        agent = MAPPO(config)
        # Populate Adam moments, so the rollback checks more than empty state.
        agent._optimize(sum(p.square().sum() for p in agent.actor.parameters()),
                        agent.actor_optimizer, agent.actor.parameters())
        collector = TeamCollector(agent, lambda: environment(total_time=2.), 2)
        rollout, _ = collector.collect(8)
        before = copy.deepcopy(agent.actor.state_dict())
        optimizer_before = copy.deepcopy(agent.actor_optimizer.state_dict())
        metrics = agent.update(rollout, collector.envs[0])
        self.assertEqual(metrics["actor_updates_rejected"], 1)
        # The entire rollout is used even when the first actor step hits KL;
        # critic minibatches remain independent of the actor batch size.
        self.assertEqual(metrics["actor_minibatches"], 1)
        self.assertEqual(metrics["actor_samples_processed"], len(rollout.rows))
        self.assertLessEqual(metrics["max_actor_kl"], 1e-10)
        for key, value in before.items():
            torch.testing.assert_close(agent.actor.state_dict()[key], value, rtol=0., atol=0.)
        restored = agent.actor_optimizer.state_dict()
        self.assertEqual(restored["param_groups"], optimizer_before["param_groups"])
        for key, state in optimizer_before["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(restored["state"][key][name], value, rtol=0., atol=0.)
        agent.options.update(target_kl=.001, max_kl_backtracks=14)
        metrics = agent.update(rollout, collector.envs[0])
        self.assertGreater(metrics["actor_kl_backtracks"], 0)
        self.assertEqual(metrics["actor_updates_rejected"], 0)
        self.assertLessEqual(metrics["max_actor_kl"], .001)
        self.assertTrue(any(not torch.equal(agent.actor.state_dict()[key], value)
                            for key, value in before.items()))
        for key, state in optimizer_before["state"].items():
            self.assertEqual(agent.actor_optimizer.state_dict()["state"][key]["step"], state["step"] + 1)

    def test_demonstrations_match_legacy_execution_and_train_both_actor_stages(self):
        from pretrain_actor import demonstration
        from RL.Networks import ActorAdap
        env = environment()
        obs, _ = env._get_obs()
        teacher = ActorAdap(6, 8, 3, 32).eval()
        frame = env.structured_frame()
        target = demonstration(teacher, obs, frame["boids"])
        with torch.no_grad():
            legacy = teacher(torch.as_tensor(obs, dtype=torch.float32), True, False)[0].numpy()
        env.step(legacy, "AdaRes")
        np.testing.assert_allclose(target, env.last_executed_thrust, atol=1e-4, rtol=0.)
        student = ChannelActor(hidden=32)
        predicted = student.act(tensor_frame(frame), deterministic=True)["executed"]
        loss = ((predicted - torch.as_tensor(target)[None]) / 750.).square().mean()
        loss.backward()
        for head in (student.proposal_head[-1], student.gate_head[-1]):
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.)

    def test_value_normalization_preserves_raw_values_and_target_coordinates(self):
        frame = collate_frames([environment().structured_frame() for _ in range(5)])
        value = TeamCritic(32, normalize_value=True, layer_norm=True)
        target = copy.deepcopy(value)
        before = value(frame).detach()
        value.update_normalization(torch.tensor([[-80.], [50.], [-20.], [10.], [-40.]]))
        torch.testing.assert_close(value(frame), before, atol=1e-5, rtol=1e-5)
        target.copy_normalization_from(value)
        torch.testing.assert_close(target(frame), before, atol=1e-5, rtol=1e-5)
        self.assertGreater(float(value.output_scale), 1.)
        with torch.no_grad():
            value.readout[-1].bias.add_(.2)
        source_prediction = value(frame).detach()
        with torch.no_grad():
            for source, destination in zip(value.parameters(), target.parameters()):
                destination.lerp_(source, .1)
        torch.testing.assert_close(target(frame), .9 * before + .1 * source_prediction, atol=1e-5, rtol=1e-5)

    def test_vector_gae_keeps_environment_time_axes_separate(self):
        env = environment()
        frame = env.structured_frame()
        actor = ChannelActor(hidden=32)
        with torch.no_grad():
            decision = actor.act(tensor_frame(frame))
        rollout = TeamRollout()
        for stream, reward, terminal in [(0, 1., False), (1, 10., False), (0, 2., True), (1, 20., True)]:
            rollout.append(frame, frame, decision, [reward] * 3, terminal, 0., 0., stream_id=stream)
        batch = rollout.batch("cpu", .9, 1.)
        np.testing.assert_allclose(batch["returns"].numpy().ravel(), [2.8, 28., 2., 20.])

    def test_vector_collector_handles_resets_partial_batches_and_mixed_teams(self):
        config = dict(policy=dict(hidden=32), mappo=dict(epochs=1, minibatch_steps=8, value_normalization=True), guidance=dict(enabled=False))
        agent = MAPPO(config)
        counts = iter([1, 3, 4, 4, 1, 3, 1, 3, 4])
        collector = TeamCollector(agent, lambda: environment(next(counts), total_time=.4), num_envs=3)
        rollout, stats = collector.collect(7, {0, 3, 6})
        self.assertEqual(agent.training_steps, 7)
        self.assertEqual(stats["completed_episodes"], 3)
        self.assertEqual([row["stream_id"] for row in rollout.rows], [0, 1, 2, 0, 1, 2, 0])
        self.assertEqual([i for i, row in enumerate(rollout.rows) if row["snapshot"] is not None], [0, 3, 6])
        for i in range(3):
            np.testing.assert_array_equal(rollout.rows[i]["next_frame"]["local"], rollout.rows[i + 3]["frame"]["local"])
        self.assertEqual([len(row["frame"]["mask"]) for row in rollout.rows], [1, 3, 4, 1, 3, 4, 4])
        self.assertLess(agent.update(rollout, collector.envs[0])["likelihood_error"], 1e-4)

    def test_batched_evaluation_preserves_each_scenario_rng_and_outcome(self):
        from train_mappo import evaluate_episodes
        policy = DeployedPolicy(ChannelActor(hidden=32))
        config = dict(environment=dict(protocol="paper-parameters-v1", canonical_agent_order=True), agent=dict(defender_num=3))
        state = np.random.get_state()
        serial = evaluate_episodes(policy, config, 4, seed=651000, duration=1.)
        config["training"] = dict(eval_num_envs=3)
        batched = evaluate_episodes(policy, config, 4, seed=651000, duration=1.)
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])
        for a, b in zip(serial, batched):
            self.assertEqual((a["seed"], a["outcome_code"], a["steps"]), (b["seed"], b["outcome_code"], b["steps"]))
            for key in ("reward", "min_spacing", "gate_c", "gate_d"):
                self.assertAlmostEqual(a[key], b[key], places=4)

    def test_gae_bootstraps_rollout_boundary_but_not_denial(self):
        a, _ = compute_gae([1.], [2.], [10.], [False], .9, .95)
        b, _ = compute_gae([1.], [2.], [10.], [True], .9, .95)
        np.testing.assert_allclose(a, [8.])
        np.testing.assert_allclose(b, [-1.])

    def test_joint_search_reuses_updated_team_and_preserves_neighborhood(self):
        f = tensor_frame(environment(2).structured_frame())
        b, l = torch.zeros(1, 2, 2), torch.full((1, 2, 2), 1000.)
        g = torch.full((1, 2, 1), .5)

        def q(frame, actions):
            return -((actions[..., 0].sum(-1, keepdim=True) - 1200.) / 1000.).square()

        args = dict(fusion="scalar", step=.2, radius=.2, sweeps=2, penalty=0.)
        joint = search_gates(q, f, b, l, g, **args)
        independent = search_gates(q, f, b, l, g, mode="independent", **args)
        self.assertGreater(joint["objective_gain"], independent["objective_gain"])
        self.assertLessEqual(float((joint["gates"] - g).abs().max()), .200001)
        self.assertGreaterEqual(joint["objective_gain"], 0.)
        torch.testing.assert_close(g, torch.full_like(g, .5))

    def test_mixed_team_rollout_update_and_checkpoint(self):
        config = dict(policy=dict(hidden=32), mappo=dict(epochs=1, minibatch_steps=8, value_normalization=True,
                                                       critic_layer_norm=True, critic_learning_rate=.0003), guidance=dict(enabled=False))
        agent = MAPPO(config)
        rollout = TeamRollout()
        for n in (1, 3, 4):
            env = environment(n, total_time=.4)
            for _ in range(2):
                f = env.structured_frame()
                with torch.no_grad():
                    decision = agent.actor.act(tensor_frame(f))
                    value = agent.value(tensor_frame(f)).item()
                _, rewards, outcome, info = env.step_thrust(decision["executed"][0].numpy())
                nf = env.structured_frame()
                with torch.no_grad():
                    nv = agent.value(tensor_frame(nf)).item() if not outcome else 0.
                rollout.append(f, nf, decision, rewards, outcome, value, nv)
        before = [r["decision"]["executed"].copy() for r in rollout.rows]
        metrics = agent.update(rollout, env)
        self.assertLess(metrics["likelihood_error"], 1e-4)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        for r, original in zip(rollout.rows, before):
            np.testing.assert_array_equal(r["decision"]["executed"], original)
        agent.training_steps = 123
        agent.trainer_state = dict(best_score=[.8, -2, 3.], best_step=100)
        agent.guidance.validation_history.extend([True, False, True])
        checkpoint = io.BytesIO()
        agent.save(checkpoint)
        expected_numpy, expected_torch = np.random.random(3), torch.rand(3)
        checkpoint.seek(0)
        deployed = DeployedPolicy.load(checkpoint)
        checkpoint.seek(0)
        restored = MAPPO.load(checkpoint)
        self.assertEqual(restored.training_steps, 123)
        self.assertEqual(restored.trainer_state, agent.trainer_state)
        torch.testing.assert_close(restored.value.output_mean, agent.value.output_mean)
        torch.testing.assert_close(restored.q.output_scale, agent.q.output_scale)
        self.assertEqual(list(restored.guidance.validation_history), [True, False, True])
        self.assertEqual(restored.actor_optimizer.state_dict()["param_groups"],
                         agent.actor_optimizer.state_dict()["param_groups"])
        restored.restore_training_rng()
        np.testing.assert_array_equal(np.random.random(3), expected_numpy)
        torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
        f = env.structured_frame()
        np.testing.assert_allclose(deployed.act(f)["executed"], DeployedPolicy(restored.actor).act(f)["executed"])

    def test_all_fusion_ablation_modes_have_valid_likelihoods(self):
        frame = tensor_frame(environment().structured_frame())
        for fusion in ("scalar", "channels", "thrusters"):
            actor = ChannelActor(hidden=32, fusion=fusion, aggregation="mean", share_candidates=False, motion_features=False)
            with torch.no_grad():
                decision = actor.act(frame)
            logp, _ = actor.evaluate_actions(frame, decision, entropy=False)
            torch.testing.assert_close(logp, decision["logp_candidate"] + decision["logp_gate"])
            self.assertEqual(decision["gates"].shape[-1], 1 if fusion == "scalar" else 2)


class VrxContractTests(unittest.TestCase):
    def test_vrx_control_uses_shared_policy_and_thruster_mapping(self):
        # Execute the actual controller method without requiring ROS imports.
        legacy_tree = ast.parse((ROOT / "vrx/tad_vrx_experiment.py").read_text(encoding="utf-8"))
        functions = [node for node in legacy_tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("force_to_thruster", "APF_navi_control", "Boids_navi_control")]
        from RL.observations import build_frame
        scope = dict(np=np, build_frame=build_frame, Float64=lambda data: data)
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<vrx-functions>", "exec"), scope)
        tree = ast.parse((ROOT / "vrx/run_experiment.py").read_text(encoding="utf-8"))
        trial = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trial")
        control = next(node for node in trial.body if isinstance(node, ast.FunctionDef) and node.name == "control")
        exec(compile(ast.Module(body=[control], type_ignores=[]), "<vrx-control>", "exec"), scope)
        env = environment()
        policy = DeployedPolicy(ChannelActor(hidden=32))
        vessels = [env.attacker, *env.defender_list]
        published = []
        trial_state = SimpleNamespace(num_robots=4, agility=2., total_time=60.,
                                      curr_pos=np.array([v.pos for v in vessels]),
                                      curr_vel=np.array([v.vel for v in vessels]),
                                      curr_phi=np.array([v.theta for v in vessels]),
                                      curr_yaw_rate=np.array([v.velocity[2] for v in vessels]),
                                      last_stamp=np.ones(4), channel_policy=policy, rows=[],
                                      publishers=[SimpleNamespace(publish=published.append) for _ in range(8)],
                                      args=SimpleNamespace(controller="ChannelMAPPO", action_period=.2))
        scope["control"](trial_state, 0.)
        expected = policy.act(env.structured_frame())["executed"]
        np.testing.assert_allclose(trial_state.rows[0]["DefAct"].reshape(3, 2), expected, atol=1e-4)
        np.testing.assert_allclose(np.array(published).reshape(4, 2)[1:], expected[:, ::-1], atol=1e-4)
        self.assertIn("CandidateMessages", trial_state.rows[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
