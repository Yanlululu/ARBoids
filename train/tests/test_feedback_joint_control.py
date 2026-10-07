"""Contracts for feedback-consistent prediction and unconstrained gate geometry."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np

from source_arboids import numerical_source, snapshot
from feedback_joint_control import (Measurement, NominalEnvironment, observe, nominal_scene,
                                     FeedbackJointController, project_thrust)


def policy(observations):
    n=len(observations)
    action=np.empty((n,3),dtype=np.float32)
    action[:,0]=.7
    action[:,1]=.2+.15*np.tanh(observations[:,3])
    action[:,2]=.45+.15*np.tanh(observations[:,2]/30.)
    return action


class FeedbackContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule=FeedbackJointController(3,policy,block_steps=5,force_planning=True)

    def scene(self,seed=491):
        np.random.seed(seed)
        env=NominalEnvironment(defender_num=3)
        obs,_=env.reset(agility=2.25,noisy_agility=False)
        return env,obs

    def setUp(self):
        self.rule.reset()

    def test_predictions_equal_execution_across_replanning_boundaries(self):
        for block_steps in (5,10):
            self.rule.block_steps=block_steps
            self.rule.reset()
            env,obs=self.scene()
            plans=0
            max_command_error=max_position_error=0.
            for tick in range(3*block_steps):
                command,info=self.rule.control(observe(env,obs))
                if info['planned']:
                    plans+=1
                    self.assertEqual(tick % block_steps,0)
                    self.assertLessEqual(self.rule.plan['score'],self.rule.plan['baseline_score'])
                expected=self.rule.plan['commands'][info['plan_offset']]
                max_command_error=max(max_command_error,float(np.max(np.abs(expected-command))))
                obs,_,done,_=env.step(env.thrust_to_action(command),'RL')
                max_position_error=max(max_position_error,float(np.max(np.abs(snapshot(env)[:,:3]-info['predicted_next_state'][:,:3]))))
                if done:
                    break
            self.assertEqual(plans,3)
            self.assertLess(max_command_error,1e-5)
            self.assertLess(max_position_error,1e-8)
        self.rule.block_steps=5

    def test_prediction_neither_consumes_nor_uses_future_randomness(self):
        env,obs=self.scene()
        measurement=observe(env,obs)
        np.random.seed(781)
        before=np.random.get_state()
        first=self.rule.predict(measurement,'baseline')
        after=np.random.get_state()
        for a,b in zip(before,after):
            np.testing.assert_array_equal(a,b)
        np.random.seed(89104)
        second=self.rule.predict(measurement,'baseline')
        np.testing.assert_array_equal(first['positions'],second['positions'])
        np.testing.assert_array_equal(first['commands'],second['commands'])

    def test_each_predicted_cycle_recomputes_feedback(self):
        env,obs=self.scene()
        with patch.object(self.rule,'feedback',wraps=self.rule.feedback) as feedback:
            result=self.rule.predict(observe(env,obs),'more_boids')
        self.assertEqual(feedback.call_count,len(result['commands']))
        self.assertGreater(np.max(np.abs(np.diff(result['commands'],axis=0))),1e-3)

    def test_parameters_are_committed_but_thrust_is_not(self):
        env,obs=self.scene()
        commands=[]
        templates=[]
        for _ in range(5):
            command,info=self.rule.control(observe(env,obs))
            commands.append(command)
            templates.append(info['template'])
            obs,_,_,_=env.step(env.thrust_to_action(command),'RL')
        self.assertEqual(len(set(templates)),1)
        self.assertEqual(self.rule.plan_count,1)
        self.assertEqual(self.rule.remaining,0)
        self.assertGreater(np.max(np.abs(np.diff(commands,axis=0))),1e-3)

    def test_continuous_correction_leaves_blending_segment(self):
        states=np.array([[-3.,0.,0.,1.,0.,0.],[3.,0.,np.pi,-1.,0.,0.],[50.,50.,0.,0.,0.,0.]])
        measurement=Measurement(states,np.array([60.,0.,np.pi,-2.,0.,0.]),
                                np.zeros((3,18)),np.full((3,2),900.))
        fixed=lambda obs:np.tile(np.array([1.,1.,.5],dtype=np.float32),(len(obs),1))
        old=self.rule.policy
        self.rule.policy=fixed
        try:
            force,_,_=self.rule.feedback(measurement,'baseline')
            projected=project_thrust(force,fixed(measurement.observations),measurement.boids)
            self.assertGreater(np.linalg.norm(force-projected),10.)
            self.assertTrue((force>=-500.).all() and (force<=1000.).all())
        finally:
            self.rule.policy=old

    def test_original_apf_only_differs_by_removed_noise(self):
        env,obs=self.scene()
        from types import SimpleNamespace
        obstacles=[SimpleNamespace(pos=b.pos,r=env.Obs_R) for b in env.defender_list]
        original=numerical_source().TADEnv._APF_navi_step
        with patch('numpy.random.normal',return_value=np.zeros(2)):
            expected=original(env,env.attacker.pos,np.zeros(2),obstacles,env.attacker.theta)
        actual=env._APF_navi_step(env.attacker.pos,np.zeros(2),obstacles,env.attacker.theta)
        np.testing.assert_array_equal(expected,actual)

    def test_reset_drops_previous_episode_parameters(self):
        env,obs=self.scene()
        self.rule.control(observe(env,obs))
        self.rule.reset()
        self.assertEqual(self.rule.remaining,0)
        self.assertIsNone(self.rule.previous_thrust)
        self.assertIsNone(self.rule.previous_attacker_thrust)
        self.assertIsNone(self.rule.plan)

    def test_model_input_has_no_hidden_source_state(self):
        self.assertEqual(set(Measurement.__dataclass_fields__),
                         {'defenders','attacker','observations','boids','time','capture_radius','total_time'})
        env,obs=self.scene()
        measurement=observe(env,obs)
        env.attacker.agility=9.
        env.attacker.velocity_r[:]=200.
        env.defender_list[0].velocity_r[:]=-200.
        nominal=nominal_scene(measurement)
        self.assertEqual(nominal.attacker.agility,2.25)
        np.testing.assert_array_equal(nominal.defender_list[0].velocity_r,measurement.defenders[0,3:])


class FastFeedbackContracts(FeedbackContracts):
    @classmethod
    def setUpClass(cls):
        from feedback_joint_fast import FastFeedbackJointController
        cls.rule=FastFeedbackJointController(3,policy,block_steps=5,force_planning=True)


if __name__=='__main__':
    unittest.main()
