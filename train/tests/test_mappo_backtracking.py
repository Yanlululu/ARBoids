"""Full-rollout retries must make feasible progress or restore every Adam moment."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from policy.mappo import PredictiveMAPPO, MAPPOConfig
from train_mappo import make_env, collect_episodes


class BacktrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(271)
        np.random.seed(271)
        config = dict(agent=dict(defender_num=3, boid_state=True, form_reward=True),
            environment=dict(protocol='paper-parameters-v1',total_time=.8), prediction=dict(horizon=.1),
            mappo=dict(hidden_dim=32,relation_dim=8,coordination_dim=4,actor_lr=.03,
                       minibatch_size=16,ppo_epochs=1,joint_ratio=True,full_batch_guard=True,
                       target_kl=.001,full_batch_backtracking_steps=12))
        self.agent=PredictiveMAPPO(config)
        # Populate Adam before the tested update; retrying must not advance its
        # moments once for each rejected trial.
        sum(p.square().sum()*.0001 for p in self.agent.actor.parameters()).backward()
        self.agent.actor_optimizer.step()
        self.agent.actor_optimizer.zero_grad(set_to_none=True)
        rollout=collect_episodes(self.agent,make_env(config),4,2.)
        self.data=rollout.tensors(self.agent.settings,'cpu')
        advantage=self.data['advantage_r']-self.agent.lagrange.value*self.data['advantage_c']
        self.advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
        self.reference=self.agent._policy_statistics(self.data,self.advantage)

    def test_backtracking_accepts_one_scaled_adam_step_with_actual_kl_improvement(self):
        agent=self.agent
        steps={p:float(s['step']) for p,s in agent.actor_optimizer.state.items()}
        accepted,result,diagnostics=agent._backtrack_full_policy(self.data,self.advantage,self.reference)
        self.assertTrue(accepted)
        self.assertGreater(diagnostics['full_gradient_backtracking_trials'],1)
        self.assertGreater(result['surrogate'],self.reference['surrogate'])
        self.assertLessEqual(result['kl_limit'],agent.settings.target_kl)
        self.assertLess(diagnostics['full_gradient_step_scale'],1.)
        for parameter,state in agent.actor_optimizer.state.items():
            self.assertLessEqual(float(state['step'])-steps[parameter],1.)
        self.assertTrue(any(float(state['step'])>steps[p] for p,state in agent.actor_optimizer.state.items()))
        self.assertEqual(agent.actor_optimizer.param_groups[0]['lr'],.03)

    def test_failed_trials_restore_parameters_and_all_optimizer_tensors(self):
        agent=self.agent
        before=copy.deepcopy(agent.actor.state_dict())
        optimizer=copy.deepcopy(agent.actor_optimizer.state_dict())
        rejected=dict(self.reference,surrogate=self.reference['surrogate']-1.)
        with patch.object(agent,'_policy_statistics',return_value=rejected):
            accepted,_,diagnostics=agent._backtrack_full_policy(self.data,self.advantage,self.reference)
        self.assertFalse(accepted)
        self.assertEqual(diagnostics['full_gradient_backtracking_trials'],12)
        for name,value in before.items():
            torch.testing.assert_close(agent.actor.state_dict()[name],value,rtol=0,atol=0)
        after=agent.actor_optimizer.state_dict()
        self.assertEqual(after['param_groups'],optimizer['param_groups'])
        for index,state in optimizer['state'].items():
            for key,value in state.items():
                torch.testing.assert_close(after['state'][index][key],value,rtol=0,atol=0)

    def test_backtracking_requires_a_guard_and_a_bounded_trial_count(self):
        for options in (dict(full_batch_backtracking_steps=2),
                        dict(full_batch_guard=True,target_kl=.01,full_batch_backtracking_steps=13)):
            with self.assertRaises(ValueError):
                MAPPOConfig(**options)


if __name__=='__main__':
    unittest.main()
