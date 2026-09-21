"""Behavioral checks for the published-parameter protocol and source defaults."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'train'))
from envs.TADgame import TADEnv
from utils.config import load_config
from utils.protocol import environment_kwargs, apply_adapter_exploration


def placed(protocol='paper-parameters-v1', attacker=(30, 0), defenders=((40, 0), (50, 0), (60, 0)), form=False):
    env = TADEnv(protocol=protocol, form_reward=form)
    env.reset()
    env.attacker.pos = np.asarray(attacker, dtype=float)
    for defender, position in zip(env.defender_list, defenders):
        defender.pos = np.asarray(position, dtype=float)
    env._get_obs()
    return env


class PaperProtocolTests(unittest.TestCase):
    def test_source_defaults_and_early_loss_remain_available(self):
        source = placed(protocol='source')
        self.assertEqual((source.Total_T, source.agility_noise_half_width), (80., .25))
        self.assertEqual(source._isTerminate(), 1)
        paper = placed()
        self.assertEqual((paper.Total_T, paper.agility_noise_half_width), (60., .5))
        self.assertEqual(paper._isTerminate(), 0)

    def test_target_and_collision_boundaries(self):
        self.assertEqual(placed(attacker=(15, 0), defenders=((0, 0), (-7, 7), (-7, -7)))._isTerminate(), 1)
        self.assertEqual(placed(attacker=(60, 0), defenders=((0, 0), (3, 4), (20, 0)))._isTerminate(), 2)

    def test_capture_uses_section_ii_strict_radius(self):
        self.assertEqual(placed(defenders=((25, 0), (0, 10), (0, -10)))._isTerminate(), 0)
        self.assertEqual(placed(defenders=((25.01, 0), (0, 10), (0, -10)))._isTerminate(), 3)

    def test_timeout_at_sixty_seconds(self):
        env = placed(attacker=(60, 0), defenders=((0, 0), (0, 10), (0, -10)))
        env.Current_T = 59.8
        self.assertEqual(env._isTerminate(), 0)
        env.Current_T = 60.
        self.assertEqual(env._isTerminate(), 4)

    def test_main_and_helper_reward_with_terminal_formation(self):
        env = placed(defenders=((10, 0), (20, 0), (26, 0)), form=True)
        np.testing.assert_allclose(env._get_rewards(3), [-.5, 49.5, 99.5], atol=1e-12)

    def test_collision_penalty_once_per_agent(self):
        env = placed(defenders=((0, 0), (4, 0), (8, 0)))
        np.testing.assert_array_equal(env._get_rewards(2), [-50., -50., -50.])

    def test_main_and_collision_rewards_are_additive(self):
        env = placed(defenders=((26, 0), (24, 0), (-20, 0)))
        np.testing.assert_array_equal(env._get_rewards(2), [50., 0., 0.])

    def test_no_capture_bonus_for_timeout(self):
        env = placed(attacker=(60, 0), defenders=((70, 0), (55, 5*np.sqrt(3)), (55, -5*np.sqrt(3))))
        env.Current_T = 60.
        self.assertEqual(env._isTerminate(), 4)
        np.testing.assert_array_equal(env._get_rewards(4), [0., 0., 0.])

    def test_curriculum_distribution_and_observation_finiteness(self):
        env = TADEnv(protocol='paper-parameters-v1')
        np.random.seed(707)
        agilities = []
        for _ in range(200):
            state, _ = env.reset(2., noisy_agility=True)
            self.assertTrue(np.isfinite(state).all())
            agilities.append(env.attacker.agility)
        self.assertGreaterEqual(min(agilities), 1.5)
        self.assertLessEqual(max(agilities), 2.5)
        self.assertLess(min(agilities), 1.6)
        self.assertGreater(max(agilities), 2.4)

    def test_adapter_source_draw_is_unchanged_and_paper_is_gaussian(self):
        np.random.seed(123)
        expected = np.full((3, 3), .5, dtype=np.float32)
        expected[:, -1] = np.clip(expected[:, -1]+np.random.uniform(-.1, .1), 0., 1.)
        np.random.seed(123)
        actual = apply_adapter_exploration(np.full((3, 3), .5, dtype=np.float32), SimpleNamespace())
        np.testing.assert_array_equal(actual, expected)
        np.random.seed(456)
        samples = apply_adapter_exploration(np.full((10000, 3), .5), SimpleNamespace(adapter_noise_distribution='normal', adapter_noise_scale=.1))[:, -1]
        self.assertAlmostEqual(samples.mean(), .5, delta=.005)
        self.assertAlmostEqual(samples.std(), .1, delta=.005)
        self.assertGreater(samples.max(), .8)

    def test_training_validation_uses_paper_environment(self):
        spec = importlib.util.spec_from_file_location('train_entry_test', ROOT/'train/train.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = load_config(ROOT/'train/configs/paper-parameters.yaml')
        created = []
        def factory(*args, **kwargs):
            env = TADEnv(*args, **kwargs)
            created.append(env)
            return env
        agent = SimpleNamespace(choose_action=lambda state, deterministic: np.zeros((3, 3)))
        np.random.seed(17)
        with patch.object(module, 'TADEnv', factory):
            score, reward = module.evaluate(agent, controller='AdaRes', episodes=1, env_options=environment_kwargs(cfg))
        self.assertTrue(np.isfinite([score, reward]).all())
        self.assertEqual(created[0].protocol, 'paper-parameters-v1')
        self.assertEqual(created[0].Total_T, 60.)

    def test_vrx_paper_termination_matches_2d_geometry(self):
        tree = ast.parse((ROOT/'vrx/run_experiment.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trial')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'outcome')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<vrx-outcome>', 'exec'), namespace)
        rng = np.random.default_rng(91)
        for index in range(100):
            positions = rng.normal(0, 25, (4, 2))
            env = placed(attacker=positions[0], defenders=positions[1:])
            env.Current_T = 60. if index % 3 == 0 else 20.
            trial = SimpleNamespace(curr_pos=positions, target_r=15., collision_r=5., defend_r=5., total_time=60., args=SimpleNamespace(termination_rule='paper'))
            self.assertEqual(namespace['outcome'](trial, env.Current_T), env._isTerminate())


if __name__ == '__main__':
    unittest.main(verbosity=2)
