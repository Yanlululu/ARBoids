"""Validate whole-block recovery without using method outcomes as acceptance criteria."""
import argparse
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

PATH=Path(__file__).resolve().parents[2]/'vrx/evaluate_gate_contribution.py'
SPEC=importlib.util.spec_from_file_location('_vrx_gate_pairing',PATH)
BATCH=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)

CONTROL_SPEC = importlib.util.spec_from_file_location('_vrx_shared_control', PATH.with_name('tad_vrx_experiment.py'))
CONTROL = importlib.util.module_from_spec(CONTROL_SPEC)
CONTROL_SPEC.loader.exec_module(CONTROL)
MODEL_SPEC = importlib.util.spec_from_file_location('_vrx_deployment_models', PATH.with_name('models.py'))
MODELS = importlib.util.module_from_spec(MODEL_SPEC)
MODEL_SPEC.loader.exec_module(MODELS)


def row(method,offset=0.,success=1):
    return dict(cell='dock-n3',seed=123,method=method,run_id=f'original-{method}',
                initial_poses='same-generated-source-poses',
                first_attacker_position=json.dumps([0.,0.]),
                first_defender_positions=json.dumps([[10.+offset,0.],[20.,0.],[30.,0.]]),
                success=success)


class VRXPairingTests(unittest.TestCase):
    def setUp(self):
        self.scene=dict(cell='dock-n3',seed=123)
        self.args=argparse.Namespace(output_dir=Path('unused-pairing-test-output'),resume=True,workers=2)

    def test_missing_methods_or_different_generated_states_are_rejected(self):
        with self.assertRaises(RuntimeError):
            BATCH.pairing_error([row(m) for m in BATCH.METHODS[:-1]])
        rows=[row(m) for m in BATCH.METHODS]
        rows[-1]['initial_poses']='different'
        with self.assertRaises(RuntimeError):
            BATCH.pairing_error(rows)

    def test_valid_pairs_never_trigger_additional_trials(self):
        rows=[row(m,success=0) for m in BATCH.METHODS]
        with patch.object(BATCH,'trial_job') as job:
            actual,replacements=BATCH.repair_initial_pairing(rows,[self.scene],self.args)
        job.assert_not_called()
        self.assertEqual(actual,rows)
        self.assertEqual(replacements,[])

    def test_all_methods_are_replaced_even_when_new_outcomes_are_worse(self):
        old=[row(m,offset=.008 if m=='cbf' else 0.) for m in BATCH.METHODS]
        def completed(item,args):
            return row(item[1],success=0)
        with patch.object(Path,'mkdir'),patch.object(BATCH,'trial_job',side_effect=completed) as job:
            actual,replacements=BATCH.repair_initial_pairing(old,[self.scene],self.args)
        self.assertEqual(job.call_count,4)
        self.assertEqual({r['method'] for r in actual},set(BATCH.METHODS))
        self.assertTrue(all(r['success']==0 for r in actual))
        self.assertTrue(all(r['replaces_run_id'].startswith('original-') for r in actual))
        self.assertEqual(len(replacements),1)
        self.assertEqual(BATCH.pairing_error(actual),0.)

    def test_persistent_pairing_failure_has_a_fixed_retry_bound(self):
        old=[row(m,offset=.008 if m=='cbf' else 0.) for m in BATCH.METHODS]
        def completed(item,args):
            return row(item[1],offset=.008 if item[1]=='cbf' else 0.)
        with patch.object(Path,'mkdir'),patch.object(BATCH,'trial_job',side_effect=completed) as job:
            with self.assertRaisesRegex(RuntimeError,'two whole-block attempts'):
                BATCH.repair_initial_pairing(old,[self.scene],self.args)
        self.assertEqual(job.call_count,8)


class VRXControlTests(unittest.TestCase):
    def test_deployment_actors_use_the_training_forward_and_checkpoint_layout(self):
        from policy.networks import ActorSAC, ActorAdap
        for training_class, deployment_class, actions in ((ActorSAC, MODELS.ActorSAC, 2),
                                                          (ActorAdap, MODELS.ActorAdap, 3)):
            training = training_class(6, 8, actions, 32)
            deployed = deployment_class(6, 8, actions, 32)
            deployed.load_state_dict(training.state_dict(), strict=True)
            self.assertIs(deployment_class.forward, training_class.forward)
            for defenders in (3, 6):
                observation = torch.randn(defenders, 14+2*(defenders-1))
                torch.testing.assert_close(deployed(observation, True, False)[0],
                                           training(observation, True, False)[0], rtol=0., atol=0.)

    def test_shared_scene_setup_is_ros_independent_and_retains_observation_width(self):
        for defenders in (3, 6):
            manager = CONTROL.ExperimentManager(defenders+1)
            poses = manager.generate_init_info(2., setting=0)
            self.assertEqual(len(poses.split(';')), defenders+1)
            self.assertEqual((manager.defend_r, manager.total_time), (5., 60.))
            obs = manager.get_observations(manager.curr_pos[1:], np.zeros(defenders),
                manager.curr_pos[0], np.zeros(2), np.zeros((defenders, 6)), np.zeros((defenders, 2)))
            self.assertEqual(obs.shape, (defenders, 14+2*(defenders-1)))
            self.assertTrue(np.isfinite(obs).all())

    def test_custom_thruster_limits_affect_conversion_before_residual_mixing(self):
        boids = np.array([[100., 200.]])
        for controller, expected in (('RL', [[300., 550.]]), ('Res', [[400., 750.]]),
                                     ('AdaRes', [[150., 287.5]])):
            values = [[0., .5, .25]] if controller == 'AdaRes' else [[0., .5]]
            actor = lambda *args: (torch.tensor(values), None)
            thrust = CONTROL.RL_navi_control(actor, np.zeros((1, 14)), boids, controller,
                                            min_thrust=-200., max_thrust=800.)
            np.testing.assert_array_equal(thrust, expected)


if __name__=='__main__':
    unittest.main()
