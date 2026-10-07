"""Source isolation and provenance guards; synthetic fixtures are not experiments."""

import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('original_runner', ROOT / 'scripts/run_original_arboids.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def temporary_directory():
    directory = tempfile.TemporaryDirectory(dir=ROOT, prefix='.original-runner-test-')
    # Constrain automatic recursive cleanup to a generated workspace directory.
    if Path(directory.name).resolve().parent != ROOT.resolve():
        raise RuntimeError('Test directory is outside the workspace.')
    return directory


class OriginalReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release = runner.original_release()
        cls.source, cls.hashes, cls.modules = cls.release.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.release.__exit__(None, None, None)

    def setUp(self):
        self.cfg = runner.configuration(self.source, self.modules, 'arboids')

    def write_training_fixture(self, directory):
        """Small fake artifacts exercise rejection without running any training."""
        import yaml
        checkpoint = directory / 'adares1.pth'
        checkpoint.write_bytes(b'synthetic test checkpoint, not a model')
        config = self.modules['utils.config']._namespace_to_dict(self.cfg)
        (directory / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        with (directory / 'metrics1.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('num', 'def_sr', 'reward', 'time'))
            writer.writeheader()
            writer.writerows(dict(num=i, def_sr=.5, reward=0, time=i) for i in range(1, 192))
        record = dict(kind='author-release-training', schema_version=1, passed=True,
                      completed=True, initialization='from_scratch', variant='arboids',
                      allowed_agent_changes={}, seed=101, training_steps=1000000,
                      evaluation_count=191, config=config, checkpoint_file=checkpoint.name,
                      checkpoint_sha256=runner.sha256(checkpoint.read_bytes()),
                      config_sha256=runner.sha256((directory / 'config.yaml').read_bytes()),
                      metrics_sha256=runner.sha256((directory / 'metrics1.csv').read_bytes()),
                      **runner.release_identity(self.source))
        (directory / 'completed.json').write_text(json.dumps(record), encoding='utf-8')
        return checkpoint, record

    def validate(self, checkpoint, variant='arboids'):
        cfg = runner.configuration(self.source, self.modules, variant)
        return runner.validate_training_origin(checkpoint, variant, cfg, self.source, self.modules)

    def test_release_imports_have_no_working_tree_sources(self):
        runner.verify_release(self.source, self.hashes, self.modules)
        for name, module in self.modules.items():
            self.assertTrue(module.__file__.startswith('git:' + runner.UPSTREAM_REVISION + ':'))
            self.assertIsInstance(module.__loader__, runner.ReleaseLoader)
        env = self.modules['envs.TADgame'].TADEnv()
        self.assertEqual(env.Total_T, 80.0)
        self.assertFalse(hasattr(env, 'protocol'))

    def test_mutated_source_is_rejected(self):
        changed = dict(self.source)
        changed['train/train.py'] += b'\n# altered\n'
        with self.assertRaisesRegex(RuntimeError, 'Author source was changed'):
            runner.verify_release(changed, self.hashes, self.modules)

    def test_preloaded_training_modules_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'fresh process'):
            with runner.original_release():
                pass

    def test_only_declared_author_switches_change(self):
        utility = self.modules['utils.config']
        original = utility._namespace_to_dict(self.cfg)
        for variant, changes in runner.VARIANTS.items():
            expected = copy.deepcopy(original)
            expected['agent'].update(changes)
            actual = runner.configuration(self.source, self.modules, variant)
            self.assertEqual(utility._namespace_to_dict(actual), expected)
        self.assertEqual(original['rl']['batch_size'], 4096)
        self.assertEqual(original['training']['total_steps'], 1000000)

    def test_validation_schedule_starts_after_warmup(self):
        steps = runner.evaluation_steps(self.cfg)
        self.assertEqual((steps[0], steps[-1], len(steps)), (50000, 1000000, 191))
        small = copy.deepcopy(self.cfg)
        small.training.warm_steps = 11
        small.training.eval_interval = 5
        small.training.total_steps = 25
        self.assertEqual(runner.evaluation_steps(small), [15, 20, 25])

    def test_existing_modified_framework_checkpoint_is_rejected(self):
        with temporary_directory() as temporary:
            checkpoint = Path(temporary) / 'candidate.pth'
            checkpoint.write_bytes(b'non-original actor')
            with self.assertRaisesRegex(ValueError, 'Missing completed.json'):
                self.validate(checkpoint)

    def test_record_and_artifacts_must_agree(self):
        with temporary_directory() as temporary:
            checkpoint, record = self.write_training_fixture(Path(temporary))
            self.assertEqual(self.validate(checkpoint)['seed'], 101)
            for key, value in (('completed', False), ('upstream_tree', 'wrong'),
                               ('variant', 'rp'), ('training_steps', 500000),
                               ('evaluation_count', 200), ('source_sha256', {}), ('seed', None)):
                with self.subTest(key=key):
                    changed = dict(record, **{key: value})
                    (checkpoint.parent / 'completed.json').write_text(json.dumps(changed), encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'provenance'):
                        self.validate(checkpoint)

    def test_checkpoint_and_metrics_tampering_are_rejected(self):
        for filename in ('adares1.pth', 'metrics1.csv', 'config.yaml'):
            with self.subTest(filename=filename), temporary_directory() as temporary:
                checkpoint, _ = self.write_training_fixture(Path(temporary))
                path = checkpoint.parent / filename
                path.write_bytes(path.read_bytes() + b'changed')
                with self.assertRaisesRegex(ValueError, 'hash|artifact'):
                    self.validate(checkpoint)

    def test_incomplete_schedule_and_nonfinite_metrics_are_rejected(self):
        with temporary_directory() as temporary:
            checkpoint, _ = self.write_training_fixture(Path(temporary))
            path = checkpoint.parent / 'metrics1.csv'
            original = path.read_text(encoding='utf-8')
            path.write_text('\n'.join(original.splitlines()[:-1]) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'evaluation schedule'):
                runner.validate_metrics(path, self.cfg)
            path.write_text(original.replace('0.5', 'nan', 1), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
                runner.validate_metrics(path, self.cfg)

    def test_boids_uses_original_environment_end_to_end(self):
        with temporary_directory() as temporary:
            args = SimpleNamespace(variant='boids', output=Path(temporary) / 'eval',
                                   checkpoint=None, device='cpu', first_scene_seed=17,
                                   episodes=2, agility=2.0)
            result = runner.evaluate(args, self.source, self.modules)
            self.assertTrue(result['passed'])
            self.assertEqual(result['evaluation_protocol'], 'author-release-80s')
            self.assertEqual(result['episodes'], 2)
            self.assertEqual(sum(result['counts'][key] for key in ('success', 'collision', 'attacker_win')), 2)
            self.assertEqual(result['counts']['attacker_win'],
                             result['counts']['early_attacker_win'] + result['counts']['physical_breach'])
            self.assertIsNone(result['training_seed'])


if __name__ == '__main__':
    unittest.main()
