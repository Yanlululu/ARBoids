"""Clipped-action probabilities, independent integral checks and behavior preservation."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.mappo import PredictiveMAPPO
from train_mappo import collect_episodes, make_env


def configuration():
    return dict(agent=dict(defender_num=3, boid_state=True, form_reward=True),
        environment=dict(protocol='paper-parameters-v1', total_time=.6), prediction=dict(horizon=.1),
        mappo=dict(hidden_dim=32, relation_dim=8, coordination_dim=4, theta_space_gate=True,
                   baseline_initialization=True, ppo_epochs=1, minibatch_size=64, prediction_lr=1e-4))


class CensoredGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(875)
        np.random.seed(875)
        self.agent = PredictiveMAPPO(configuration())
        self.agent.enable_censored_gate_likelihood()

    def test_endpoint_samples_share_mass_and_interior_keeps_density(self):
        actor = self.agent.actor
        mu = torch.tensor([[.2]], dtype=torch.float64, requires_grad=True)
        distribution = Normal(mu, torch.ones_like(mu)*.3)
        left = actor.gate_log_probability(distribution, mu.new_tensor([[-.1]]))
        far_left = actor.gate_log_probability(distribution, mu.new_tensor([[-3.]]))
        torch.testing.assert_close(left, far_left, rtol=0, atol=0)
        torch.testing.assert_close(left.exp(), distribution.cdf(torch.zeros_like(mu)).squeeze(-1))
        right = actor.gate_log_probability(distribution, mu.new_tensor([[1.5]]))
        torch.testing.assert_close(right.exp(), 1.-distribution.cdf(torch.ones_like(mu)).squeeze(-1))
        inside = mu.new_tensor([[.4]])
        torch.testing.assert_close(actor.gate_log_probability(distribution, inside), distribution.log_prob(inside).sum(-1))
        self.assertLess(float(torch.autograd.grad(left.sum(), mu, retain_graph=True)[0]), 0.)
        self.assertGreater(float(torch.autograd.grad(right.sum(), mu)[0]), 0.)

    def test_entropy_matches_quadrature_and_extreme_gradients_are_finite(self):
        mu = torch.tensor([[-.3],[0.],[.5],[1.],[1.3]], dtype=torch.float64, requires_grad=True)
        log_std = torch.full_like(mu, np.log(.1), requires_grad=True)
        distribution = Normal(mu, log_std.exp())
        nodes, weights = np.polynomial.legendre.leggauss(256)
        x = torch.tensor((nodes+1.)/2.)
        log_density = distribution.log_prob(x)
        integral = -(log_density.exp()*log_density*torch.tensor(weights/2.)).sum(-1)
        p0 = distribution.cdf(torch.zeros_like(mu)).squeeze(-1)
        p1 = (1.-distribution.cdf(torch.ones_like(mu))).squeeze(-1)
        reference = integral-torch.special.xlogy(p0,p0)-torch.special.xlogy(p1,p1)
        torch.testing.assert_close(self.agent.actor.gate_entropy(distribution), reference, atol=1e-10, rtol=1e-8)
        extreme = Normal(mu*100., log_std.exp())
        loss = self.agent.actor.gate_entropy(extreme).sum()
        loss += self.agent.actor.gate_log_probability(extreme, mu.detach().clamp(0,1)).sum()
        gradients = torch.autograd.grad(loss, (mu,log_std))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_score_expectation_is_unchanged_and_endpoint_noise_is_removed(self):
        # Deterministic Gaussian quantiles avoid a flaky Monte Carlo assertion.
        z = Normal(0.,1.).icdf((torch.arange(100000,dtype=torch.float64)+.5)/100000)
        mu = torch.tensor(-.2,dtype=torch.float64,requires_grad=True)
        std = .1
        raw = (mu.detach()+std*z).unsqueeze(-1)
        distribution = Normal(mu,torch.tensor(std,dtype=torch.float64))
        probability = self.agent.actor.gate_log_probability(distribution,raw)
        reward = raw.clamp(0,1).squeeze(-1)
        gradient = torch.autograd.grad((probability*reward).mean(),mu)[0]
        # d E[clip(w)] / d mu = P(0 < w < 1).
        expected = torch.special.ndtr((1.-mu.detach())/std)-torch.special.ndtr(-mu.detach()/std)
        torch.testing.assert_close(gradient, expected, atol=3e-5, rtol=1e-3)
        a = -mu.detach()/std
        endpoint_score = -torch.exp(Normal(0.,1.).log_prob(a))/std/torch.special.ndtr(a)
        raw_score = (raw.squeeze(-1)-mu.detach())/std**2
        physical_score = torch.where(raw.squeeze(-1)<=0, endpoint_score, raw_score)
        self.assertLess(float(physical_score.var()), .2*float(raw_score.var()))

    def test_migration_preserves_actions_and_recomputes_saved_probabilities(self):
        legacy = PredictiveMAPPO(configuration())
        env = make_env(legacy.config)
        obs = env.reset(2.)
        args = (obs,env.env.prediction_snapshot(),env.env.thrust_to_action(env.env.boids_actions),0.)
        torch.manual_seed(876)
        before, record = legacy.act(*args)
        optimizer = legacy.actor_optimizer
        legacy.enable_censored_gate_likelihood()
        torch.manual_seed(876)
        after, repeated = legacy.act(*args)
        np.testing.assert_array_equal(before,after)
        np.testing.assert_array_equal(record['gate_raw'],repeated['gate_raw'])
        self.assertIs(legacy.actor_optimizer,optimizer)
        rollout = collect_episodes(legacy,env,2,2.)
        data = rollout.tensors(legacy.settings,'cpu')
        lp,lg,_ = legacy.evaluate_batch(data)
        torch.testing.assert_close(lp,data['old_logp_l'],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(lg,data['old_logp_g'],rtol=1e-5,atol=1e-5)
        legacy.update(rollout)
        with tempfile.TemporaryDirectory(prefix='arboids-censored-') as directory:
            path=Path(directory)/'model.pth';legacy.save(path)
            restored=PredictiveMAPPO.from_checkpoint(path)
        self.assertTrue(restored.actor.censored_gate_likelihood)
        lp,lg,_=restored.evaluate_batch(data)
        current_lp,current_lg,_=legacy.evaluate_batch(data)
        torch.testing.assert_close(lp,current_lp,rtol=0,atol=0)
        torch.testing.assert_close(lg,current_lg,rtol=0,atol=0)

    def test_probability_mode_switch_rejects_stale_or_mixed_rollouts_before_update(self):
        agent = PredictiveMAPPO(configuration())
        env = make_env(agent.config)
        stale = collect_episodes(agent, env, 2, 2.)
        original = {key: value.clone() for key, value in agent.actor.state_dict().items()}
        agent.enable_censored_gate_likelihood()
        with self.assertRaisesRegex(ValueError, 'collect fresh'):
            agent.update(stale)
        for episode in stale.episodes:
            for transition in episode:
                transition.pop('gate_likelihood_censored')
        with self.assertRaisesRegex(ValueError, 'collect fresh'):
            agent.update(stale)
        fresh = collect_episodes(agent, env, 2, 2.)
        fresh.episodes[-1][-1]['gate_likelihood_censored'] = False
        with self.assertRaisesRegex(ValueError, 'collect fresh'):
            agent.update(fresh)
        self.assertEqual(agent.updates, 0)
        self.assertFalse(agent.actor_optimizer.state)
        for key, value in agent.actor.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
