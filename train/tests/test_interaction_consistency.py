"""Independent reproductions of math/control inconsistencies, before learning."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'train'))
import study_runtime
import numpy as np
import torch
import yaml

from envs.TADgame import TADEnv
from envs.modules import Obstacle
from envs.snapshot import seed_random
from interaction_rollout import public_packet, safety_controller, FrozenPolicy, frozen_payload
from policy.interaction_sac import (InteractionActor, InteractionSAC, JointReplay,
                                   TwinTeamCritic, tensor_packet)


def current_config():
    config = yaml.safe_load((ROOT / 'train/configs/interaction-aware-sac.yaml').read_text())
    config['rl'].update(hidden_dim=32, batch_size=4)
    config['interaction'].update(relation_dim=16, workers=0,
        critic_control_coordinates='nominal-thrust-v2', entropy_objective='proposal-mean-v3',
        bootstrap_estimator='mean')
    return config


def batch(packet):
    return {key: value[None] for key, value in tensor_packet(packet).items()}


class ExecutionStateReproductions(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_random(42)
        self.env = TADEnv(3, protocol='paper-parameters-v1')
        self.env.reset(2.25, noisy_agility=False)

    def test_cbf_velocity_is_part_of_the_conditioned_value_state(self):
        self.env.attacker.reset(np.array([50., 0.]), np.pi)
        for boat, position, heading in zip(self.env.defender_list,
                [(0., 0.), (7.2, 0.), (30., 0.)], [0., np.pi, 0.]):
            boat.reset(np.array(position), heading)
        other = copy.deepcopy(self.env)
        for boat, current in zip(other.defender_list, [2., -2., 0.]):
            boat.reset(boat.pos.copy(), boat.theta, np.array([current, 0., 0.]))
        # Two complete environment states with real, recomputed Boids controls;
        # this is not an arbitrary inconsistent edit of an input packet.
        for env in (self.env, other):
            env._Boid_navi_step(np.array([b.pos for b in env.defender_list]),
                np.array([b.vel for b in env.defender_list]),
                np.array([b.theta for b in env.defender_list]), env.attacker.pos)
        left = public_packet(self.env)
        right = public_packet(other)
        np.testing.assert_array_equal(left['central'], right['central'])
        np.testing.assert_array_equal(left['obs'][..., 12:14], right['obs'][..., 12:14])
        action = np.tile([.5, .5, 1.], (3, 1)).astype(np.float32)
        cbf = safety_controller(3)
        u0 = cbf.control(left['motion'], action, self.env.boids_actions)[1]
        u1 = cbf.control(right['motion'], action, other.boids_actions)[1]
        self.assertGreater(np.max(np.abs(u0 - u1)), 100.)
        action = torch.as_tensor(action)[None]
        legacy = TwinTeamCritic(32, 16, 'nominal-thrust-v2')
        torch.testing.assert_close(legacy(batch(left), action)[0],
                                   legacy(batch(right), action)[0], rtol=0, atol=0)
        critic = InteractionSAC(current_config()).critic
        self.assertGreater(abs(float((critic(batch(left), action)[0] -
                                     critic(batch(right), action)[0]).detach())), 1e-7)

    def test_apf_order_is_not_a_symmetry_of_the_released_environment(self):
        self.env.attacker.reset(np.array([20., 0.]), np.pi)
        for boat, position in zip(self.env.defender_list,
                                  [(-5., 10.), (-1., -5.), (-15., 20.)]):
            boat.reset(np.array(position), 0.)
        self.env._Boid_navi_step(np.array([b.pos for b in self.env.defender_list]),
            np.array([b.vel for b in self.env.defender_list]),
            np.array([b.theta for b in self.env.defender_list]), self.env.attacker.pos)
        self.env._apf_force_noise = lambda: np.zeros(2)
        obstacles = [Obstacle(pos=b.pos, radius=self.env.Obs_R) for b in self.env.defender_list]
        u0 = self.env._APF_navi_step(self.env.attacker.pos, np.zeros(2), obstacles, np.pi)
        u1 = self.env._APF_navi_step(self.env.attacker.pos, np.zeros(2), obstacles[::-1], np.pi)
        self.assertGreater(np.max(np.abs(u0 - u1)), 100.)
        left = public_packet(self.env)
        self.env.defender_list.reverse()
        self.env.boids_actions = self.env.boids_actions[::-1].copy()
        self.env.boids_states = self.env.boids_states[::-1].copy()
        right = public_packet(self.env)
        action = torch.tensor([[[.5, .5, 1.]] * 3])
        legacy = TwinTeamCritic(32, 16, 'nominal-thrust-v2')
        torch.testing.assert_close(legacy(batch(left), action)[0],
                                   legacy(batch(right), action)[0], rtol=1e-6, atol=1e-8)
        critic = InteractionSAC(current_config()).critic
        self.assertGreater(abs(float((critic(batch(left), action)[0] -
                                     critic(batch(right), action)[0]).detach())), 1e-7)

    def test_agility_thrust_conversion_is_an_inverse(self):
        action = np.array([-.8, .6])
        for agility in (1., 2.25, 4., 6.):
            thrust = self.env.action_to_thrust(action, agility)
            np.testing.assert_allclose(self.env.thrust_to_action(thrust, agility), action,
                                       rtol=0, atol=1e-14)

    def test_attacker_replay_action_matches_the_command_that_was_executed(self):
        command = np.array([-.8, .6])
        physical = self.env.action_to_thrust(command, self.env.attacker.agility)
        recorded = []
        step = self.env.attacker.step

        def capture(thrust, current):
            recorded.append(thrust.copy())
            return step(thrust, current)

        self.env.attacker.step = capture
        self.env.step(np.zeros((3, 3)), 'AdaRes', command)
        np.testing.assert_allclose(recorded[0], physical, rtol=0, atol=0)
        np.testing.assert_allclose(self.env.att_action, command, rtol=0, atol=1e-14)

    def test_state_version_propagates_to_frozen_targets_and_cannot_silently_resume(self):
        config = current_config()
        agent = InteractionSAC(config)
        packet = public_packet(self.env)
        action, _ = agent.choose_action(packet, deterministic=True)
        restored = InteractionSAC(config)
        restored.load_state_dict(copy.deepcopy(agent.state_dict()))
        policy = FrozenPolicy(frozen_payload(agent))
        self.assertEqual(policy.critic.q1.state_coordinates, 'execution-v2')
        for critic in (restored.critic, policy.critic):
            for got, expected in zip(critic(batch(packet), torch.tensor(action)[None]),
                                     agent.critic(batch(packet), torch.tensor(action)[None])):
                torch.testing.assert_close(got, expected, rtol=0, atol=0)
        legacy_config = copy.deepcopy(config)
        legacy_config['interaction'].pop('critic_state_coordinates')
        legacy = InteractionSAC(legacy_config)
        legacy_state = legacy.state_dict()
        legacy_state.pop('critic_state_coordinates')
        InteractionSAC(legacy_config).load_state_dict(legacy_state)
        FrozenPolicy(frozen_payload(legacy))
        with self.assertRaisesRegex(ValueError, 'Critic state coordinates changed'):
            agent.load_state_dict(legacy_state)


class AnalyticGradientContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_random(194)

    def test_relation_coordinates_and_candidate_order_match_the_formula(self):
        actor = InteractionActor(32, 16).double()
        motion = torch.tensor([[[0., 0., np.pi/2, 1., 1., .1],
                                [2., 4., np.pi, 2., 3., .4]]], dtype=torch.float64)
        proposal = torch.tensor([[[.1, .2], [.3, .4]]], dtype=torch.float64)
        boids = torch.tensor([[[-.1, -.2], [-.3, -.4]]], dtype=torch.float64)
        features = actor.relation_features(motion, proposal, boids)
        expected = torch.tensor([4/60, -2/60, 1., 0., 2/5, -1/5, .3,
                                 .1, .2, -.1, -.2, .3, .4, -.3, -.4], dtype=torch.float64)
        torch.testing.assert_close(features[0, 0, 1], expected, rtol=0, atol=1e-15)

    def test_blending_chain_rule_in_physical_newtons(self):
        raw = torch.tensor([-.4, .2, .7], dtype=torch.float64, requires_grad=True)
        learned = torch.tensor([[.2, -.6], [.1, .7], [-.3, .6]], dtype=torch.float64)
        boids = torch.tensor([[-.2, .4], [.5, -.1], [.8, .2]], dtype=torch.float64)
        coefficients = torch.tensor([[2., -1.], [.3, .9], [-.7, .6]], dtype=torch.float64)
        theta = (1 + raw.tanh()) / 2
        thrust = 750 * (theta[:, None] * learned + (1-theta[:, None]) * boids) + 250
        gradient, = torch.autograd.grad((thrust * coefficients).sum(), raw)
        exact = 375 * (1-raw.detach().tanh().square()) * ((learned-boids)*coefficients).sum(-1)
        torch.testing.assert_close(gradient, exact, rtol=1e-14, atol=1e-13)

    def test_actual_actor_parameter_gradients_match_fixed_noise_finite_differences(self):
        for stage in ('gate', 'joint'):
            actor = InteractionActor(32, 16, entropy_objective='proposal-mean-v3').double()
            actor.freeze_proposals(stage == 'gate')
            torch.nn.init.normal_(actor.relation_gate.weight, std=.2)
            obs, motion = .1*torch.randn(1, 3, 18, dtype=torch.float64), torch.randn(1, 3, 6, dtype=torch.float64)
            noise = .3*torch.randn(1, 3, 3, dtype=torch.float64)
            weight = torch.tensor([[[.3, -.2], [-.4, .8], [.7, .1]]], dtype=torch.float64)

            def objective():
                action, logp = actor(obs, motion, noise=noise)
                nominal = action[..., 2:] * action[..., :2] + (1-action[..., 2:]) * obs[..., 12:14]
                return (nominal * weight).sum() + .2*logp.sum()

            names = ['base.adap_layer.bias', 'relation_gate.weight', 'relations.0.weight', 'gate_log_std.bias']
            if stage == 'joint':
                names += ['base.mean_layer.weight', 'base.log_std_layer.bias']
            parameters = dict(actor.named_parameters())
            grads = torch.autograd.grad(objective(), [parameters[name] for name in names])
            for name, gradient in zip(names, grads):
                parameter = parameters[name]
                direction = torch.randn_like(parameter)
                direction /= direction.norm()
                original = parameter.detach().clone()
                with torch.no_grad():
                    parameter.copy_(original + 1e-6*direction)
                    right = float(objective())
                    parameter.copy_(original - 1e-6*direction)
                    left = float(objective())
                    parameter.copy_(original)
                expected = float((gradient*direction).sum())
                self.assertAlmostEqual((right-left)/2e-6, expected, delta=2e-7,
                                       msg=f'{stage}: {name}')
                self.assertGreater(float(gradient.abs().max()), 1e-12)
            self.assertEqual(actor.base.mean_layer.weight.requires_grad, stage == 'joint')

    def test_corrected_critic_action_derivatives_and_physical_aliases(self):
        env = TADEnv(3, protocol='paper-parameters-v1'); env.reset(2.25)
        packet = {k: v.double() for k, v in batch(public_packet(env)).items()}
        critic = TwinTeamCritic(32, 16, 'nominal-thrust-v2', 'execution-v2').double()
        action = torch.tensor([[[.1, -.2, .4], [.6, -.1, .7], [.3, .2, .5]]],
                              dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a: critic(packet, a), (action,),
                                                eps=1e-6, atol=1e-7, rtol=1e-5))
        boids = packet['obs'][..., 12:14]
        first = torch.cat((.5*boids, torch.ones_like(boids[..., :1])), -1)
        second = torch.cat((torch.zeros_like(boids), .5*torch.ones_like(boids[..., :1])), -1)
        for left, right in zip(critic(packet, first), critic(packet, second)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_td_and_auxiliary_parameter_gradients_match_independent_scalar_oracle(self):
        class ScalarTwins(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([.4, -.2]))
                self.bias = torch.nn.Parameter(torch.tensor([.3, .1]))

            def forward(self, packet, action):
                value = action[:, 0, 2:3] * self.weight + self.bias
                return value[:, :1], value[:, 1:]

        env = TADEnv(3, protocol='paper-parameters-v1'); env.reset(2.25)
        packet = public_packet(env)
        for objective in ('difference', 'value'):
            config = current_config()
            config['interaction'].update(auxiliary=objective, bootstrap_source='coupled')
            agent = InteractionSAC(config)
            agent.critic = ScalarTwins()
            agent.target = copy.deepcopy(agent.critic).requires_grad_(False)
            agent.critic_optimizer = torch.optim.SGD(agent.critic.parameters(), lr=0.)
            replay = JointReplay(4)
            action = np.zeros((3, 3), dtype=np.float32); action[:, 2] = .25
            replay.store(packet, action, env.boids_actions, np.full(3, 2.), packet, True)
            left, right = action[None].copy(), action[None].copy()
            left[:, 0, 2], right[:, 0, 2] = .8, .2
            g, gc = torch.tensor([[2.]], requires_grad=True), torch.tensor([[-1.]], requires_grad=True)
            auxiliary = {k: v[None] for k, v in packet.items()}
            auxiliary.update(action=left, counterfactual=right, **{'return': g, 'counterfactual_return': gc})
            result = agent.learn(replay, auxiliary, update_actor=False)
            w, b = agent.critic.weight.detach(), agent.critic.bias.detach()
            residual = b + .25*w - 2
            expected_w, expected_b = .5*residual, 2*residual
            if objective == 'difference':
                error = .6*w - 3
                expected_w += .1*2*.6*error
                expected_loss = error.square().sum()
            else:
                a, c = b + .8*w - 2, b + .2*w + 1
                expected_w += .1*(.8*a + .2*c)
                expected_b += .1*(a+c)
                expected_loss = .5*(a.square()+c.square()).sum()
            torch.testing.assert_close(agent.critic.weight.grad, expected_w, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(agent.critic.bias.grad, expected_b, rtol=1e-6, atol=1e-6)
            self.assertAlmostEqual(result['auxiliary_loss'], float(expected_loss), places=5)
            self.assertIsNone(g.grad)
            self.assertIsNone(gc.grad)
            self.assertTrue(all(p.grad is None for p in agent.target.parameters()))
            self.assertTrue(all(p.grad is None for p in agent.actor.parameters()))


if __name__ == '__main__':
    unittest.main()
