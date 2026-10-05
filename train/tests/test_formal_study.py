"""Scientific controls: genuine ablations, exact SAC continuation and link faults."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'train'))
from formal_study_training import load_sac, save_sac, rng_state, restore_rng, mappo_config, initialize_matched_agent
from policy.SAC import SAC, ReplayBuffer
from policy.mappo import PredictiveMAPPO
from policy.networks import ActorAdap
from policy.message_stress import MessageStress
from train_mappo import make_env, collect_episodes
from utils.config import _dict_to_namespace


def small(arm='full'):
    config = mappo_config(arm)
    config['mappo'].update(hidden_dim=32, relation_dim=8, coordination_dim=4,
                           ppo_epochs=1, minibatch_size=16, full_batch_backtracking_steps=0)
    config['environment']['total_time'] = .6
    config['prediction']['horizon'] = .1
    return config


class FormalStudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(61)
        np.random.seed(61)

    def test_sac_resume_reproduces_next_update_including_alpha_and_replay(self):
        config = dict(rl=dict(batch_size=8,GAMMA=.99,TAU=.005,learning_rate=1e-4,hidden_dim=32),
                      agent=dict(defender_num=3),environment=dict(protocol='paper-parameters-v1',total_time=.6))
        agent = SAC(_dict_to_namespace(config),6,8,3,adaptive=True)
        replay = ReplayBuffer(18,3)
        # A small circular buffer exercises wraparound without wasting test memory.
        replay.max_size=16
        for name in ('s','a','r','s_','dw'):
            setattr(replay,name,getattr(replay,name)[:16].copy())
        for _ in range(23):
            replay.store(np.random.normal(size=18),np.random.uniform(-1,1,3),1.,np.zeros(18),False)
        agent.learn(replay)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'full.pth'
            save_sac(path,agent,replay,config,dict(steps=52,training_seed=61))
            agent.learn(replay)
            expected={name:copy.deepcopy(getattr(agent,name).state_dict()) for name in ('actor','critic','critic_target')}
            expected_alpha=agent.log_alpha.detach().clone()
            loaded, restored, _, counters=load_sac(path,'cpu')
            self.assertEqual((restored.count,restored.size),(7,16))
            self.assertEqual(counters['steps'],52)
            loaded.learn(restored)
            for name in expected:
                for key,value in expected[name].items():
                    torch.testing.assert_close(getattr(loaded,name).state_dict()[key],value,rtol=0,atol=0)
            torch.testing.assert_close(loaded.log_alpha,expected_alpha,rtol=0,atol=0)

    def test_actor_only_checkpoint_is_rejected_for_sac_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'actor.pth'
            torch.save({'weight':torch.zeros(1)},path)
            with self.assertRaisesRegex(ValueError,'Actor-only'):
                load_sac(path,'cpu')

    def test_no_prediction_never_calls_predictor_or_uses_message_kinematics(self):
        config=small('no_prediction')
        agent=PredictiveMAPPO(config)
        env=make_env(config)
        obs=env.reset()
        states=env.env.prediction_snapshot()
        boids=env.env.thrust_to_action(env.env.boids_actions)
        with patch.object(agent.predictor,'features',side_effect=AssertionError('prediction leaked')):
            first,record=agent.act(obs,states,boids,0.,True)
            second,_=agent.act(obs,states+100.,boids,0.,True)
        np.testing.assert_array_equal(first,second)
        self.assertFalse(record['mask'].any())
        with patch.object(agent.predictor,'features_batch',side_effect=AssertionError('prediction leaked')):
            batch,_=agent.act_batch([obs],[states],[boids],[0.],[torch.Generator().manual_seed(5)],True)
        np.testing.assert_allclose(batch[0],first,rtol=1e-6,atol=1e-6)

    def test_fixed_gain_remains_fixed_through_update_and_checkpoint(self):
        config=small('fixed_gain')
        agent=PredictiveMAPPO(config)
        initial=agent.actor.compatibility_gain_raw.detach().clone()
        rollout=collect_episodes(agent,make_env(config),2,2.)
        agent.update(rollout)
        torch.testing.assert_close(agent.actor.compatibility_gain_raw,initial,rtol=0,atol=0)
        self.assertFalse(agent.actor.compatibility_gain_raw.requires_grad)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'policy.pth'
            agent.save(path)
            other=PredictiveMAPPO.from_checkpoint(path)
            self.assertFalse(other.actor.compatibility_gain_raw.requires_grad)
            torch.testing.assert_close(other.actor.compatibility_gain_raw,initial,rtol=0,atol=0)

    def test_control_recipes_differ_only_in_declared_mechanism(self):
        full=mappo_config('full')
        no_joint=mappo_config('no_joint')
        no_joint['mappo']['compatibility_joint_mixture']=True
        self.assertEqual(full,no_joint)
        fixed=mappo_config('fixed_gain')
        fixed['mappo']['compatibility_context_gain']=True
        fixed['mappo'].pop('fixed_compatibility_gain')
        self.assertEqual(full,fixed)

    def test_shared_initial_weights_and_rng_match_across_ablations(self):
        baseline=ActorAdap(6,8,3,32,1.).state_dict()
        full=initialize_matched_agent(small('full'),baseline,61)
        expected_rng=torch.get_rng_state().clone()
        for arm in ('no_prediction','fixed_gain','no_joint'):
            agent=initialize_matched_agent(small(arm),baseline,61)
            for key,value in agent.actor.state_dict().items():
                torch.testing.assert_close(value,full.actor.state_dict()[key],rtol=0,atol=0)
            for key,value in agent.critic.state_dict().items():
                torch.testing.assert_close(value,full.critic.state_dict()[key],rtol=0,atol=0)
            torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)

    def test_complete_packet_loss_preserves_current_local_packet(self):
        network=MessageStress(drop_probability=1.)
        def packet(t):
            return dict(states=np.full((3,6),t,dtype=float),boids=np.full((3,2),t,dtype=float),
                        proposals=np.full((3,2),t,dtype=float),observations=np.full((3,18),t,dtype=float),
                        timestamps=np.full(3,t,dtype=float))
        network.receive(packet(0.))
        caches=network.receive(packet(.2))
        for i,cache in enumerate(caches):
            self.assertEqual(cache['timestamps'][i],.2)
            self.assertTrue(np.all(np.delete(cache['timestamps'],i)==0.))
        self.assertEqual(network.drops,network.attempts)
        self.assertAlmostEqual(network.maximum_age,.2)

    def test_delay_holds_previous_peer_packet_only(self):
        network=MessageStress(delay_steps=1)
        def packet(t):
            return dict(states=np.full((2,6),t),timestamps=np.full(2,t))
        network.receive(packet(0.))
        network.receive(packet(.2))
        caches=network.receive(packet(.4))
        np.testing.assert_array_equal(caches[0]['timestamps'],[.4,.2])
        np.testing.assert_array_equal(caches[1]['timestamps'],[.2,.4])

    def test_zero_impairment_is_exact_deployment_action(self):
        config=small()
        agent=PredictiveMAPPO(config)
        env=make_env(config)
        obs=env.reset()
        states=env.env.prediction_snapshot()
        boids=env.env.thrust_to_action(env.env.boids_actions)
        action=agent.act(obs,states,boids,0.,True)[0]
        np.testing.assert_array_equal(MessageStress().act(agent,obs,states,boids,0.),action)


if __name__=='__main__':
    unittest.main()
