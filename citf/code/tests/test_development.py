import json,tempfile,unittest
from pathlib import Path
import numpy as np
from experiments.independent_select import independent_select
from citf.evaluate import committed_rows,atomic_json
from citf.contracts import file_hash


class Development(unittest.TestCase):
    def test_single_common_reference_is_order_independent(self):
        # The combined choice deliberately differs from sequential coordinate descent.
        def score(x):return -x.sum(1)+10*x.prod(1)
        forward,trace=independent_select([0,0],[[0,1],[0,1]],score)
        backward,_=independent_select([0,0],[[0,1],[0,1]],score,[1,0])
        np.testing.assert_array_equal(forward,[1,1]);np.testing.assert_array_equal(forward,backward)
        self.assertEqual(trace['evaluated_unique'],3)
        self.assertEqual(trace['configurations'][0],[0,0])

    def test_independent_unique_budget_and_ties(self):
        selected,trace=independent_select([4]*6,[list(range(9))]*6,lambda x:np.zeros(len(x)))
        self.assertEqual(trace['evaluated_unique'],49);np.testing.assert_array_equal(selected,np.zeros(6))

    def test_resume_only_accepts_committed_matching_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'task_records').mkdir();trace=root/'0.jsonl.gz';trace.write_bytes(b'trace')
            task={'task_id':'one'};row={'task':task,'trace':trace.name,'trace_sha256':file_hash(trace)}
            atomic_json(row,root/'task_records/00000.json')
            self.assertEqual(committed_rows(root,[task]),{0:row})
            (root/'1.jsonl.gz.pending').write_bytes(b'uncommitted')
            self.assertEqual(len(committed_rows(root,[task])),1)
            trace.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):committed_rows(root,[task])

    def test_resume_rejects_changed_task_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'task_records').mkdir();trace=root/'0.jsonl.gz';trace.write_bytes(b'trace')
            atomic_json({'task':{'task_id':'one'},'trace':trace.name,'trace_sha256':file_hash(trace)},root/'task_records/00000.json')
            with self.assertRaises(ValueError):committed_rows(root,[{'task_id':'other'}])


if __name__=='__main__':unittest.main()
