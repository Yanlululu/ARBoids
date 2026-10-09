"""Control parity and information contracts for the stage-one student."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import torch
from torch import nn

from distill_role_value import initialize
from envs.TADgame import TADEnv
from envs.snapshot import seed_random
from evaluate_candidate_roles import adaptive_guard_controller
from feedback_joint_control import observe
from policy.composition_student import (CompositionStudent, CompositionStudentController,
                                        STATE_DIM, student_inputs)
from train_composition_student import main, preference
from analyze_candidate_roles import analyze_learning, analyze_student, verify_frozen_source


class FixedChoice(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, state, response):
        logits = state.new_zeros((len(state), 43))
        logits[:, self.index] = 1.
        return logits


class StudentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize()

    def test_all_compositions_preserve_frozen_nominal_and_cbf_commands(self):
        seed_random(1600000000)
        env = TADEnv(6, protocol='paper-parameters-v1')
        obs, _ = env.reset(6., noisy_agility=False)
        measurement = observe(env, obs)
        teacher = adaptive_guard_controller(6)
        teacher.attacker_agility = teacher.observer.update(measurement)
        state, response = student_inputs(teacher, measurement)
        self.assertEqual(state.shape, (6, STATE_DIM))
        for index, name in enumerate(teacher.options):
            nominal = teacher.nominal(measurement, name)
            np.testing.assert_array_equal(response[index, :, 2:4], nominal[:, :2])
            np.testing.assert_array_equal(response[index, :, 0], teacher.options[name])
            _, expected, _ = teacher.safety.control(measurement.defenders, nominal, measurement.boids)
            student = CompositionStudentController(FixedChoice(index))
            with patch.object(student.operator, 'predict', side_effect=AssertionError('Online forecast forbidden')):
                actual, info = student.control(measurement)
            self.assertEqual(info['template'], name)
            np.testing.assert_array_equal(actual, expected)

    def test_commitment_keeps_feedback_and_causal_history_current(self):
        seed_random(1600000001)
        env = TADEnv(6, protocol='paper-parameters-v1')
        obs, _ = env.reset(4., noisy_agility=False)
        student = CompositionStudentController(FixedChoice(25))
        with patch.object(student.operator, 'predict', side_effect=AssertionError('Online forecast forbidden')):
            for tick in range(11):
                force, info = student.control(observe(env, obs))
                self.assertEqual(info['planned'], tick in (0, 10))
                self.assertEqual(student.operator.observer.updates, tick)
                np.testing.assert_array_equal(student.operator.previous_thrust, force)
                self.assertIsNotNone(student.operator.previous_attacker_thrust)
                obs, _, _, _ = env.step(np.zeros((6, 3)), 'AdaRes', defender_thrust=force)
        self.assertEqual(student.plan_count, 2)

    def test_shared_scorer_conditions_on_actual_joint_response(self):
        torch.manual_seed(42)
        model = CompositionStudent()
        state, response = torch.randn(2, 6, STATE_DIM), torch.randn(2, 43, 6, 6)
        response[..., :2] = (response[..., :2] > 0).float()
        scores = model(state, response)
        order = torch.randperm(43)
        torch.testing.assert_close(model(state, response[:, order]), scores[:, order])
        changed = response.clone()
        changed[:, 7, 3, 2:4] += .5
        difference = model(state, changed)-scores
        self.assertTrue(torch.all(difference[:, 7].abs() > 1e-7))
        torch.testing.assert_close(difference[:, :7], torch.zeros_like(difference[:, :7]))

    def test_teacher_preference_retains_lexicographic_priority(self):
        scores = torch.tensor([[[0., 0., 20.], [0., 0., 20.], [0., 1., 1.], [1., 0., 0.]]])
        torch.testing.assert_close(preference(scores), torch.tensor([[.5, .5, 0., 0.]]))

    def test_invalid_experiment_sizes_and_duplicate_seeds_fail_before_collection(self):
        for options in (['--count', '0'], ['--validation-count', '0'], ['--test-count', '0'],
                        ['--updates', '0'], ['--workers', '0'], ['--training-seeds', '42', '42']):
            with tempfile.TemporaryDirectory() as directory, patch.object(sys, 'argv',
                    ['train_composition_student.py', '--output', str(Path(directory)/'run'),
                     '--source', __file__, *options]), patch('train_composition_student.initialize') as initialize:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                initialize.assert_not_called()

    def test_training_entry_freezes_sources_independently_of_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.pth'
            source.write_bytes(b'checkpoint fixture')
            try:
                os.chdir(root)
                with patch.object(sys, 'argv', ['train_composition_student.py', '--output', 'run',
                        '--source', str(source)]), patch('train_composition_student.initialize'), \
                        patch('train_composition_student.atomic_json', side_effect=RuntimeError('stop at protocol')) as save:
                    with self.assertRaisesRegex(RuntimeError, 'stop at protocol'):
                        main()
                self.assertIn(str(Path(__file__).resolve().parents[1]/'policy/composition_student.py'),
                              save.call_args.args[1]['files'])
            finally:
                os.chdir(previous)

    def test_frozen_source_verification_survives_removal_of_live_files(self):
        source = 'original = 1\n'
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory)/'removed.py')
            for text in (source, source.replace('\n', '\r\n')):
                digest = hashlib.sha256(text.encode()).hexdigest()
                verify_frozen_source(missing, digest, source)
                with self.assertRaises(ValueError):
                    verify_frozen_source(missing, digest, 'altered = 2\n')

    def test_learning_analysis_rejects_missing_registered_training_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'predictive-learning-analysis-protocol.json').write_text(json.dumps(
                dict(training_seeds=[42, 101], runs=['seed42'])), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'exactly one distinct run'):
                analyze_learning(root)

    def test_student_analysis_rejects_wrong_agility_even_when_scene_ids_match(self):
        methods = ['teacher43', 'teacher22', 'prior', 'source', 'flat-42', 'structured-42']
        protocol = dict(stage='structured-student-stage-one-development', files={},
                        arguments=dict(training_seeds=[42]), banks=dict(test=[[1, 4., 'test'], [2, 6., 'test']]))
        rows = [dict(scene_seed=scene, method=method, agility=4.) for scene in (1, 2) for method in methods]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
            (root/'results.json').write_text(json.dumps(dict(complete=True, episodes=rows)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unexpected test episodes'):
                analyze_student(root)


if __name__ == '__main__':
    unittest.main()
