"""Native task reward, actual breach probabilities and guarded constraint updates."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.mappo import PredictiveMAPPO
from train_mappo import make_env, collect_episodes


def configuration():
    return dict(agent=dict(defender_num=3, boid_state=True, form_reward=True),
        environment=dict(protocol='paper-parameters-v1', total_time=.6), prediction=dict(horizon=.1),
        mappo=dict(hidden_dim=32, relation_dim=8, coordination_dim=4, theta_space_gate=True,
            censored_gate_likelihood=True, baseline_initialization=True, ppo_epochs=2,
            minibatch_size=64, cost_gae_lambda=1., cost_monte_carlo_targets=True, lagrange_lr=10.))


class BreachConstraintTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(885)
        np.random.seed(885)

    def test_migration_retains_policy_values_and_adam_then_restores_checkpoint(self):
        agent=PredictiveMAPPO(configuration())
        agent.update(collect_episodes(agent,make_env(agent.config),2,2.))
        actor=copy.deepcopy(agent.actor.state_dict())
        critic=copy.deepcopy(agent.critic.state_dict())
        actor_optimizer=agent.actor_optimizer
        moments={p:copy.deepcopy(s) for p,s in agent.critic_optimizer.state.items()}
        agent.enable_breach_constraint(.00375)
        self.assertIs(agent.actor_optimizer,actor_optimizer)
        self.assertIsNone(agent.critic.success_head)
        for name,value in actor.items():
            torch.testing.assert_close(agent.actor.state_dict()[name],value,rtol=0,atol=0)
        for name,value in critic.items():
            torch.testing.assert_close(agent.critic.state_dict()[name],value,rtol=0,atol=0)
        for p,state in moments.items():
            for name,value in state.items():
                torch.testing.assert_close(agent.critic_optimizer.state[p][name],value,rtol=0,atol=0)
        with tempfile.TemporaryDirectory(prefix='arboids-breach-') as directory:
            path=Path(directory)/'model.pth';agent.save(path)
            restored=PredictiveMAPPO.from_checkpoint(path)
        self.assertTrue(restored.settings.breach_constraint)
        self.assertFalse(restored.settings.minimize_collision)
        rollout=collect_episodes(restored,make_env(restored.config),2,2.)
        self.assertIn('value_b',rollout.episodes[0][0])
        self.assertNotIn('value_s',rollout.episodes[0][0])
        self.assertIn('breach_value_loss',restored.update(rollout))

    def test_zero_breach_multiplier_retains_the_original_task_policy_gradient(self):
        original=PredictiveMAPPO(configuration())
        constrained=copy.deepcopy(original)
        constrained.enable_breach_constraint(.02,0.)
        rollout=collect_episodes(original,make_env(original.config),3,2.)
        enriched=copy.deepcopy(rollout)
        for episode in enriched.episodes:
            for row in episode:
                row['value_b']=.02
        torch.manual_seed(886)
        original.update(rollout)
        torch.manual_seed(886)
        constrained.update(enriched)
        for name,value in original.actor.state_dict().items():
            torch.testing.assert_close(constrained.actor.state_dict()[name],value,rtol=0,atol=0)

    def test_actual_breach_targets_exclude_own_recovery_outcome_and_dual_uses_normal_episodes(self):
        config=configuration();config['mappo'].update(curriculum_cost_baseline=True,curriculum_baseline_steps=2)
        agent=PredictiveMAPPO(config);agent.enable_breach_constraint(.02,4.)
        rollout=collect_episodes(agent,make_env(agent.config),4,2.)
        for i,(episode,summary) in enumerate(zip(rollout.episodes,rollout.summaries)):
            summary.update(seed=100000+i,breach=int(i>=2),success=int(i<2))
            if i:
                summary.update(curriculum=True,source_seed=110000,source_step=10,gate_steps=20)
        # A simultaneous collision does not erase a breach label or duplicate it.
        rollout.summaries[2]['collision']=1
        rollout.episodes[2][-1]['cost']=1.
        data=rollout.tensors(agent.settings,'cpu');start=len(rollout.episodes[0])
        torch.testing.assert_close(data['group_breach_baseline'][start:start+3],torch.tensor([1.,.51,.02]))
        torch.testing.assert_close(data['return_b'][start:start+3],torch.zeros(3))
        rollout.summaries[1]['breach']=1
        changed=rollout.tensors(agent.settings,'cpu')
        torch.testing.assert_close(data['group_breach_baseline'][start:start+3],changed['group_breach_baseline'][start:start+3])
        torch.testing.assert_close(changed['return_b'][start:start+3],torch.ones(3))
        result=agent.update(rollout)
        self.assertEqual(result['breach_rate'],0.)
        self.assertAlmostEqual(result['breach_multiplier_next'],3.8)

    def test_task_and_collision_gains_cannot_hide_a_breach_surrogate_regression(self):
        config=configuration();config['mappo'].update(full_batch_guard=True,guard_task_surrogate=True,target_kl=.01,joint_ratio=True)
        agent=PredictiveMAPPO(config);agent.enable_breach_constraint(.02,4.)
        rollout=collect_episodes(agent,make_env(agent.config),3,2.)
        before=copy.deepcopy(agent.actor.state_dict())
        stats=[dict(surrogate=0.,kl_joint=0.,kl_limit=0.,task_surrogate=0.,cost_surrogate=0.,breach_surrogate=0.),
               dict(surrogate=1.,kl_joint=.001,kl_limit=.001,task_surrogate=.1,cost_surrogate=.1,breach_surrogate=-.01)]
        with patch.object(agent,'_policy_statistics',side_effect=stats):
            result=agent.update(rollout)
        self.assertEqual(result['actor_accepted_steps'],0)
        self.assertEqual(result['breach_surrogate_gain'],0.)
        self.assertFalse(agent.actor_optimizer.state)
        for name,value in before.items():
            torch.testing.assert_close(agent.actor.state_dict()[name],value,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
