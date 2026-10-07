"""Validate whole-block recovery without using method outcomes as acceptance criteria."""
import argparse
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


PATH=Path(__file__).resolve().parents[2]/'vrx/evaluate_gate_contribution.py'
SPEC=importlib.util.spec_from_file_location('_vrx_gate_pairing',PATH)
BATCH=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


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


if __name__=='__main__':
    unittest.main()
