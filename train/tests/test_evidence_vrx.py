"""Task outcomes and infrastructure completion have different meanings in VRX."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_vrx_evidence import analyze


class VRXEvidenceTests(unittest.TestCase):
    def fixture(self):
        pairs = []
        for i in range(100):
            codes = (3, 3) if i < 90 else (3, 1) if i < 96 else (1, 3) if i < 98 else (4, 4) if i == 98 else (2, 2)
            pair = dict(seed=2100000+i)
            for arm, code, seconds in zip(('channel', 'reference'), codes, (20., 22.)):
                pair[arm] = dict(passed=True, seed=pair['seed'], outcome_code=code, success=code in (3, 4),
                                 initial_poses=[float(i)], checkpoint_sha256=arm, simulation_seconds=seconds)
            pairs.append(pair)
        return dict(passed=True, protocol=dict(episodes=100, channel=dict(sha256='channel'),
                                               reference=dict(sha256='reference')), pairs=pairs)

    def test_timeout_success_is_separate_from_capture_and_comparison_is_paired(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            path = root / 'paired.json'
            path.write_text(json.dumps(self.fixture()), encoding='utf-8')
            result = analyze(path, root / 'analysis')
        self.assertEqual(result['summary']['channel']['successes'], 97)
        self.assertEqual(result['summary']['channel']['captures'], 96)
        self.assertEqual(result['summary']['channel']['timeout_denials'], 1)
        self.assertEqual(result['comparisons']['success']['candidate_only'], 6)
        self.assertEqual(result['comparisons']['success']['reference_only'], 2)
        self.assertAlmostEqual(result['conditional_capture_time']['mean_difference_seconds'], -2.)
        self.assertEqual(result['conditional_capture_time']['pairs'], 90)

    def test_infrastructure_failure_cannot_be_reported_as_completed_task_evidence(self):
        data = self.fixture()
        data['pairs'][5]['channel']['passed'] = False
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / 'paired.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                analyze(path, Path(folder) / 'analysis')


if __name__ == '__main__':
    unittest.main()
