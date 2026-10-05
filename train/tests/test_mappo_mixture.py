"""Mixed-action prediction, likelihood replay, and legacy policy isolation."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.mappo import PredictiveMAPPO
from policy.interaction_prediction import exchange_candidates, InteractionPredictor, PredictionConfig
from train_mappo import make_env, collect_episodes
from train_mappo_performance import baseline_reference
from policy.networks import ActorAdap
from qualify_mappo import meets_targets


def config():
    return dict(agent=dict(defender_num=3, boid_state=True, form_reward=True),
        environment=dict(protocol='paper-parameters-v1', total_time=.6), prediction=dict(horizon=.2),
        mappo=dict(hidden_dim=32, relation_dim=8, coordination_dim=4, theta_space_gate=True,
                   baseline_initialization=True, ppo_epochs=1, minibatch_size=64, prediction_lr=1e-4))


class MixtureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(142)
        np.random.seed(142)
        self.agent = PredictiveMAPPO(config())
        self.agent.enable_compatibility_prior()
        self.agent.enable_mixture_compatibility(5)

    def test_grid_uses_actual_mixed_thrust_and_batch_matches(self):
        env = make_env(self.agent.config)
        env.reset(2.)
        states = env.env.prediction_snapshot()
        boids = env.env.thrust_to_action(env.env.boids_actions)
        proposals = np.random.uniform(-1, 1, (3, 2))
        message = exchange_candidates(states, boids, proposals, 0.)
        predictor = self.agent.predictor
        edges, mask = predictor.features(message)
        legacy, _ = InteractionPredictor(PredictionConfig(horizon=.2)).features(message)
        np.testing.assert_array_equal(edges[..., :18], legacy)
        batched, batch_mask = predictor.features_batch([message, message])
        np.testing.assert_array_equal(edges, batched[0])
        np.testing.assert_array_equal(mask, batch_mask[0])
        grid = edges[..., 18:].reshape(3, 3, 5, 5)*5.
        for i, j, a, b in [(0, 1, 1, 3), (1, 2, 2, 4), (2, 0, 0, 2)]:
            actions = np.asarray([(1-a/4)*boids[i]+a/4*proposals[i],
                                  (1-b/4)*boids[j]+b/4*proposals[j]])
            paths = predictor.trajectories(states[[i,j]], 750.*actions+250.)
            expected = np.linalg.norm(paths[0]-paths[1], axis=-1).min()
            self.assertAlmostEqual(grid[i,j,a,b], expected, places=5)
        np.testing.assert_allclose(grid, grid.transpose(1, 0, 3, 2), atol=1e-5)
        order = [2, 0, 1]
        permuted, _ = predictor.features(exchange_candidates(states[order], boids[order], proposals[order], 0.))
        np.testing.assert_array_equal(permuted, edges[order][:,order])

    def test_safe_intended_mix_is_unchanged_even_with_unsafe_unused_endpoints(self):
        actor = self.agent.actor
        edges = torch.zeros(1, 3, 3, 43)
        distances = torch.full((1, 3, 3, 5, 5), 10.)
        distances[..., 0, 0] = 3.
        edges[..., 18:] = distances.flatten(-2)/5.
        mask = ~torch.eye(3, dtype=torch.bool)[None]
        obs = torch.zeros(1, 3, 18)
        safe = actor.compatibility_signal(edges, mask, obs, torch.ones(1, 3, 1))
        torch.testing.assert_close(safe, torch.zeros_like(safe))
        unsafe = actor.compatibility_signal(edges, mask, obs, torch.zeros(1, 3, 1))
        self.assertTrue((unsafe > 0).all())
        torch.testing.assert_close(actor.compatibility_signal(edges, torch.zeros_like(mask), obs,
                                  torch.zeros(1, 3, 1)), torch.zeros_like(safe))

    def test_likelihood_gradient_update_checkpoint_and_baseline(self):
        agent = self.agent
        agent.enable_mixture_compatibility(9, joint=True, task_prediction=True)
        rollout = collect_episodes(agent, make_env(agent.config), 2, 2.)
        batch = rollout.tensors(agent.settings, 'cpu')
        left, right, _ = agent.evaluate_batch(batch)
        torch.testing.assert_close(left, batch['old_logp_l'], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(right, batch['old_logp_g'], atol=1e-5, rtol=1e-5)
        before = agent.actor.gate_mean.weight.detach().clone()
        agent.update(rollout)
        self.assertFalse(torch.equal(before, agent.actor.gate_mean.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in agent.actor.parameters()))
        with tempfile.TemporaryDirectory(prefix='arboids-mixture-') as directory:
            path = Path(directory)/'model.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
            actual = restored.evaluate_batch(batch)
            expected = agent.evaluate_batch(batch)
            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            self.assertEqual(restored.predictor.config.mixture_points, 9)
            self.assertTrue(restored.settings.compatibility_task_prediction)
        # The reference must never inherit an experimental prediction rule.
        state = ActorAdap(6, 8, 3, 32).state_dict()
        reference = baseline_reference(agent.config, state)
        self.assertEqual(reference.predictor.config.mixture_points, 0)
        self.assertFalse(reference.settings.compatibility_prior)

    def test_joint_choice_keeps_interceptor_and_non_tied_permutation(self):
        actor = self.agent.actor
        actor.compatibility_joint_mixture = True
        edges = torch.zeros(1, 3, 3, 43)
        distance = torch.full((1, 3, 3, 5, 5), 10.)
        distance[:, 0, 1, 0, 0] = distance[:, 1, 0, 0, 0] = 3.
        edges[..., 18:] = distance.flatten(-2)/5.
        mask = ~torch.eye(3, dtype=torch.bool)[None]
        obs = torch.zeros(1, 3, 18)
        obs[0, :, 2] = torch.tensor([5., 10., 15.])
        anchor = torch.zeros(1, 3, 1)
        signal = actor.compatibility_signal(edges, mask, obs, anchor)
        torch.testing.assert_close(signal[0, :, 0], torch.tensor([0., .25, 0.]))
        order = [2, 0, 1]
        other = actor.compatibility_signal(edges[:, order][:, :, order], mask[:, order][:, :, order],
                                            obs[:, order], anchor[:, order])
        torch.testing.assert_close(other, signal[:, order])
        safe = actor.compatibility_signal(edges, mask, obs, torch.ones_like(anchor))
        torch.testing.assert_close(safe, torch.zeros_like(safe))

    def test_task_prediction_preserves_future_interceptor_in_joint_choice(self):
        actor = self.agent.actor
        actor.compatibility_joint_mixture = actor.compatibility_task_prediction = True
        distance = torch.full((1, 3, 3, 5, 5), 10.)
        distance[:, 0, 1, 0, 0] = distance[:, 1, 0, 0, 0] = 3.
        metrics = torch.tensor([[[8.,8.,8.,8.,8.], [4.,7.,8.,9.,10.], [12.,12.,12.,12.,12.]]])
        metrics = metrics.unsqueeze(-2).expand(-1,-1,2,-1).flatten(-2)
        edges = torch.cat((torch.zeros(1,3,3,18), distance.flatten(-2)/5.,
                           metrics[:, :, None].expand(-1,-1,3,-1)/5.), -1)
        mask = ~torch.eye(3,dtype=torch.bool)[None]
        obs = torch.zeros(1,3,18);obs[0,:,2]=torch.tensor([5.,10.,15.])
        signal = actor.compatibility_signal(edges, mask, obs, torch.zeros(1,3,1))
        torch.testing.assert_close(signal[0,:,0], torch.tensor([.25,0.,0.]))

    def test_task_features_use_only_exchanged_current_position_and_velocity(self):
        self.agent.enable_mixture_compatibility(5, joint=True, task_prediction=True)
        states = np.array([[0.,0.,0.,0.,0.,0.], [0.,10.,0.,0.,0.,0.], [10.,0.,0.,0.,0.,0.]])
        target = np.array([20.,20.])
        velocity = np.array([1.,-.5])
        obs = np.zeros((3,18), dtype=np.float32)
        relative = target-states[:,:2]
        obs[:,2]=np.linalg.norm(relative,axis=-1);obs[:,3]=np.arctan2(relative[:,1],relative[:,0])
        obs[:,4]=np.linalg.norm(velocity);obs[:,5]=np.arctan2(velocity[1],velocity[0])
        message = exchange_candidates(states, np.zeros((3,2)), np.ones((3,2)), 0., actor_observations=obs)
        predictor = self.agent.predictor
        edges, _ = predictor.features(message)
        paths = predictor.mixture_paths([message])[0]
        target_path = target + np.arange(paths.shape[-2])[:,None]*.05*velocity
        separation = np.linalg.norm(paths-target_path[None,None], axis=-1)
        expected = np.stack((separation.min(-1), separation[..., -1]), -2).reshape(3,10)
        for i,j in [(0,1),(1,2),(2,0)]:
            np.testing.assert_allclose(edges[i,j,43:]*5., expected[i], rtol=1e-6, atol=1e-5)
        with self.assertRaisesRegex(ValueError,'observations'):
            predictor.features(exchange_candidates(states,np.zeros((3,2)),np.ones((3,2)),0.))

    def test_user_target_is_stricter_than_old_five_percent_budget(self):
        self.assertFalse(meets_targets(dict(success_rate=.98, collision_rate=.01), .99, .01))
        self.assertFalse(meets_targets(dict(success_rate=.99, collision_rate=.02), .99, .01))
        self.assertTrue(meets_targets(dict(success_rate=.99, collision_rate=.01), .99, .01))

    def test_terminal_window_keeps_pre_capture_collisions_and_requires_capture(self):
        agent = self.agent
        agent.enable_mixture_compatibility(5, joint=True)
        actor = agent.actor
        actor.prediction_horizon = 2.
        distance = torch.full((1,3,3,5,5), 10.)
        distance[:,0,1,0,0] = distance[:,1,0,0,0] = 3.
        old_edges = torch.cat((torch.zeros(1,3,3,18), distance.flatten(-2)/5.), -1)
        mask = ~torch.eye(3,dtype=torch.bool)[None]
        obs = torch.zeros(1,3,18)
        anchor = torch.zeros(1,3,1)
        old_signal = actor.compatibility_signal(old_edges,mask,obs,anchor)
        self.assertGreater(float(old_signal.abs().sum()),0.)
        prefixes = torch.full((1,3,3,3,5,5),10.)
        times = torch.full((1,3,3,5),2.)
        edges = torch.cat((old_edges,prefixes.flatten(-3)/5.,times),-1)
        actor.compatibility_terminal_prediction = True
        torch.testing.assert_close(actor.compatibility_signal(edges,mask,obs,anchor),old_signal,rtol=0,atol=0)
        # A capture in 0.2 s + 0.4 s buffer uses the 1 s prefix, never the
        # shorter 0.5 s prefix. Collisions after that window can be ignored.
        edges[:,2,:,-5:] = .1
        torch.testing.assert_close(actor.compatibility_signal(edges,mask,obs,anchor),torch.zeros_like(anchor))
        prefixes[...,1,:,:] = distance
        edges[...,43:118] = prefixes.flatten(-3)/5.
        torch.testing.assert_close(actor.compatibility_signal(edges,mask,obs,anchor),old_signal)
        # An off-grid intent next to a noncapturing action is not certified by
        # interpolating its time with the neighboring early capture.
        edges[...,43:118] = 2.
        edges[:,2,:,-5:] = 2.
        edges[:,2,:,-5] = .1
        anchor[:,2] = .125
        signal = actor.compatibility_signal(edges,mask,obs,anchor)
        self.assertGreater(float(signal.abs().sum()),0.)

    def test_terminal_features_are_nominal_prefixes_and_capture_times(self):
        agent = self.agent
        agent.enable_mixture_compatibility(5,joint=True,task_prediction=True,terminal_prediction=True)
        states = np.array([[0.,0.,0.,0.,0.,0.],[0.,10.,0.,0.,0.,0.],[10.,0.,0.,0.,0.,0.]])
        target = np.array([3.9,0.])
        obs = np.zeros((3,18),np.float32)
        relative = target-states[:,:2]
        obs[:,2]=np.linalg.norm(relative,axis=-1);obs[:,3]=np.arctan2(relative[:,1],relative[:,0])
        message = exchange_candidates(states,np.zeros((3,2)),np.ones((3,2)),0.,actor_observations=obs)
        edges,mask = agent.predictor.features(message)
        batched,_ = agent.predictor.features_batch([message,message])
        np.testing.assert_array_equal(edges,batched[0])
        paths = agent.predictor.mixture_paths([message])[0]
        for index,fraction in enumerate((.25,.5,.75)):
            stop=int(np.ceil(fraction*(paths.shape[-2]-1)))+1
            expected=np.linalg.norm(paths[0,:,None,:stop]-paths[1,None,:,:stop],axis=-1).min(-1)
            np.testing.assert_allclose(edges[0,1,53+25*index:53+25*(index+1)].reshape(5,5)*5.,expected,atol=1e-5)
        np.testing.assert_array_equal(edges[0,1,-5:],np.zeros(5))
        np.testing.assert_array_equal(edges[1,0,-5:],np.full(5,2.))
        np.testing.assert_array_equal(edges[~mask],0.)
        rollout=collect_episodes(agent,make_env(agent.config),2,2.)
        data=rollout.tensors(agent.settings,'cpu')
        lp,lg,_=agent.evaluate_batch(data)
        torch.testing.assert_close(lp,data['old_logp_l'],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(lg,data['old_logp_g'],atol=1e-5,rtol=1e-5)
        agent.update(rollout)
        with tempfile.TemporaryDirectory(prefix='arboids-terminal-') as directory:
            path=Path(directory)/'model.pth'
            agent.save(path)
            restored=PredictiveMAPPO.from_checkpoint(path)
            for before,after in zip(agent.evaluate_batch(data),restored.evaluate_batch(data)):
                torch.testing.assert_close(before,after,atol=0,rtol=0)
        reference=baseline_reference(agent.config,ActorAdap(6,8,3,32).state_dict())
        self.assertFalse(reference.settings.compatibility_terminal_prediction)
        self.assertFalse(reference.prediction_config.terminal_prediction)

    def test_contextual_gain_preserves_policy_adam_and_checkpoint(self):
        agent = self.agent
        agent.enable_mixture_compatibility(9, joint=True)
        rollout = collect_episodes(agent, make_env(agent.config), 2, 2.)
        agent.update(rollout)
        data = rollout.tensors(agent.settings, 'cpu')
        expected = agent.evaluate_batch(data)
        weights = copy.deepcopy(agent.actor.state_dict())
        moments = {p:copy.deepcopy(state) for p,state in agent.actor_optimizer.state.items()}
        agent.enable_contextual_compatibility(context_only=True)
        actual = agent.evaluate_batch(data)
        for old,new in zip(expected,actual):
            torch.testing.assert_close(old,new,rtol=0,atol=0)
        for key,value in weights.items():
            torch.testing.assert_close(agent.actor.state_dict()[key],value,rtol=0,atol=0)
        for parameter,state in moments.items():
            for key,value in state.items():
                torch.testing.assert_close(agent.actor_optimizer.state[parameter][key],value,rtol=0,atol=0)
        with tempfile.TemporaryDirectory(prefix='arboids-context-') as directory:
            path = Path(directory)/'model.pth'
            agent.save(path)
            restored = PredictiveMAPPO.from_checkpoint(path)
            self.assertTrue(restored.settings.compatibility_context_gain)
            self.assertTrue(restored.settings.context_gain_only)
            for old,new in zip(actual,restored.evaluate_batch(data)):
                torch.testing.assert_close(old,new,rtol=0,atol=0)
        reference = baseline_reference(agent.config, ActorAdap(6,8,3,32).state_dict())
        self.assertIsNone(reference.actor.compatibility_context_head)
        with self.assertRaisesRegex(ValueError,'once'):
            agent.enable_contextual_compatibility()

    def test_releasing_context_only_retains_policy_adam_and_reenables_base_updates(self):
        agent=self.agent
        agent.enable_mixture_compatibility(5,joint=True)
        agent.enable_contextual_compatibility(context_only=True)
        rollout=collect_episodes(agent,make_env(agent.config),2,2.)
        agent.update(rollout)
        data=rollout.tensors(agent.settings,'cpu')
        outputs=agent.evaluate_batch(data)
        moments={p:copy.deepcopy(state) for p,state in agent.actor_optimizer.state.items()}
        agent.resume_full_actor_training()
        self.assertFalse(agent.settings.context_gain_only)
        self.assertTrue(all(group['lr']>0 for group in agent.actor_optimizer.param_groups))
        for before,after in zip(outputs,agent.evaluate_batch(data)):
            torch.testing.assert_close(before,after,atol=0,rtol=0)
        for parameter,state in moments.items():
            for name,value in state.items():
                torch.testing.assert_close(agent.actor_optimizer.state[parameter][name],value,atol=0,rtol=0)
        before=agent.actor.proposal_mean.weight.detach().clone()
        agent.update(collect_episodes(agent,make_env(agent.config),2,2.))
        self.assertFalse(torch.equal(before,agent.actor.proposal_mean.weight))
        with tempfile.TemporaryDirectory(prefix='arboids-full-actor-') as directory:
            path=Path(directory)/'model.pth'
            agent.save(path)
            restored=PredictiveMAPPO.from_checkpoint(path)
            self.assertFalse(restored.settings.context_gain_only)
            for before,after in zip(agent.evaluate_batch(data),restored.evaluate_batch(data)):
                torch.testing.assert_close(before,after,atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'not restricted'):
            agent.resume_full_actor_training()

    def test_contextual_gain_gets_policy_gradient_only_when_intervening(self):
        agent = self.agent
        agent.enable_mixture_compatibility(9, joint=True)
        actor = agent.actor
        obs = torch.randn(2,3,18)
        proposals = torch.zeros(2,3,2)
        boids = torch.ones_like(proposals)*.3
        mask = ~torch.eye(3,dtype=torch.bool)[None].expand(2,-1,-1)
        edges = torch.zeros(2,3,3,99)
        grid = torch.linspace(0.,1.,9)
        distance = 2.+2.*(grid[:,None]+grid[None,:])
        edges[...,18:] = distance.flatten()/5.
        before,_ = actor.gate_distribution(obs,proposals,boids,edges,mask)
        agent.enable_contextual_compatibility(context_only=True)
        distribution,_ = actor.gate_distribution(obs,proposals,boids,edges,mask)
        torch.testing.assert_close(distribution.mean,before.mean,rtol=0,atol=0)
        safe_edges = edges.clone()
        safe_edges[...,18:] = 4.
        safe_before,_ = actor.gate_distribution(obs,proposals,boids,safe_edges,mask)
        with torch.no_grad():
            actor.compatibility_context_head.bias.fill_(4.)
        safe_after,_ = actor.gate_distribution(obs,proposals,boids,safe_edges,mask)
        torch.testing.assert_close(safe_before.mean,safe_after.mean,rtol=0,atol=0)
        with torch.no_grad():
            actor.compatibility_context_head.bias.zero_()
        # A real conditional Normal score supplies the context head's gradient;
        # no derivative through the discrete joint argmin is needed.
        distribution,_ = actor.gate_distribution(obs,proposals,boids,edges,mask)
        sample = (distribution.mean+.5*distribution.stddev).detach()
        agent.actor_optimizer.zero_grad(set_to_none=True)
        (-actor.gate_log_probability(distribution,sample).mean()).backward()
        gradient = actor.compatibility_context_head.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()),0.)
        unchanged = {name:value.clone() for name,value in actor.state_dict().items()
                     if not name.startswith('compatibility_context_head.')}
        agent.actor_optimizer.step()
        self.assertGreater(float(actor.compatibility_context_head.weight.detach().norm()),0.)
        for name,value in unchanged.items():
            torch.testing.assert_close(actor.state_dict()[name],value,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
