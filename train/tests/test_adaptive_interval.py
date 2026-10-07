"""Causal identification and committed-interval contracts."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import torch

from adaptive_interval_rollout import AdaptiveIntervalController
from array_rollout import BatchedSourcePolicy
from evaluate_feedback_joint import CHECKPOINT
from feedback_joint_control import observe
from jit_rollout import JitRolloutController
from source_arboids import SourcePolicy, numerical_source


class CadenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policy = SourcePolicy(CHECKPOINT)
        cls.batch = BatchedSourcePolicy(cls.policy)

    def scene(self):
        np.random.seed(7746)
        env = numerical_source().TADEnv(defender_num=6)
        obs,_ = env.reset(agility=1.5,noisy_agility=False)
        return env,obs

    def test_disabled_adaptation_is_exactly_the_frozen_one_second_controller(self):
        settings = dict(prediction_policy=self.batch, block_steps=5, blend=1., tail_steps=100,
                        tail_policy='candidate', capture_margin=.5, failure_cost='delay')
        a = JitRolloutController(6,self.policy,**settings)
        b = AdaptiveIntervalController(6,self.policy,agility_threshold=0.,**settings)
        env,obs = self.scene()
        for tick in range(16):
            measurement = observe(env,obs)
            x,ix = a.control(measurement)
            y,iy = b.control(measurement)
            np.testing.assert_array_equal(x,y)
            self.assertEqual(ix['template'],iy['template'])
            self.assertEqual(b.observer.updates,tick)
            self.assertEqual(b.block_steps,5)
            if ix['planned']:
                np.testing.assert_array_equal(a.plan['score'],b.plan['score'])
            obs,_,done,_ = env.step(env.thrust_to_action(x),'RL')
            if done:
                break

    def test_estimate_change_cannot_interrupt_a_committed_interval(self):
        rule = AdaptiveIntervalController(6,self.policy,prediction_policy=self.batch,
            tail_steps=0,tail_policy='candidate',blend=1.,failure_cost='delay')
        class Observer:
            updates = 0
            def update(self,measurement):
                self.updates += 1
                return 1.5 if self.updates==1 else 2.5
        rule.observer = Observer()
        env,obs = self.scene()
        for tick in range(11):
            force,info = rule.control(observe(env,obs))
            self.assertEqual(rule.observer.updates,tick+1)
            self.assertEqual(info['planned'],tick in (0,10))
            self.assertEqual(info['execution_interval_steps'],10 if tick<10 else 5)
            obs,_,done,_ = env.step(env.thrust_to_action(force),'RL')
            self.assertFalse(done)
        self.assertEqual(rule.interval_plan_counts,{5:1,10:1})


if __name__=='__main__':
    unittest.main()
