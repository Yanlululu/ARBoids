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
        config = dict(policy=dict(hidden=32), mappo=dict(epochs=1, minibatch_steps=8), guidance=dict(enabled=False))
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
        checkpoint = io.BytesIO()
        agent.save(checkpoint)
        checkpoint.seek(0)
        deployed = DeployedPolicy.load(checkpoint)
        checkpoint.seek(0)
        restored = MAPPO.load(checkpoint)
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
