"""Phase barriers must survive restarts and reject stale review decisions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import run_interaction_study as scheduler
from interaction_review import review_evidence, ARMS, METRICS


class StageSchedule(unittest.TestCase):
    def test_review_summary_uses_paired_validation_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            data=study/'reviews/data-gate-42'
            artifacts={}
            for arm in (*ARMS,'initial_cbf'):
                delta = 1 if arm=='full' else 0
                rows=[dict(scene_seed=310420000+i,**{m:delta for m in METRICS}) for i in range(100)]
                calibration=[dict(state=i,absolute_error=delta) for i in range(32)]
                path=data/f'{arm}.json'
                scheduler.write_json(path,dict(summary={},episodes=rows,calibration=calibration))
                artifacts[arm]=scheduler.digest(path)
            scheduler.write_json(data/'completed.json',dict(complete=True,artifacts=artifacts))
            evidence=review_evidence(study,'gate',[42])
            self.assertEqual(evidence['paired_comparisons'][0]['difference'],1.)
            self.assertEqual(evidence['paired_comparisons'][0]['ci95'],[1.,1.])
            self.assertIn('One training seed',evidence['interpretation'])
            (data/'full.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                review_evidence(study,'gate',[42])

    def test_dependencies_block_joint_replication_and_final_tests(self):
        with patch.object(scheduler,'verified_existing_pretrain',return_value=None):
            jobs,_=scheduler.build_jobs(Path('/study'),'python',Path('/activate'))
        graph={j['name']:j for j in jobs}
        self.assertEqual(len(graph),len(jobs))
        self.assertEqual(len([j for j in jobs if j['kind']=='review']),4)
        done=set()
        while len(done)<len(jobs):
            ready={n for n,j in graph.items() if n not in done and set(j['dependencies'])<=done}
            self.assertTrue(ready,'Cyclic or missing dependencies')
            done.update(ready)
        self.assertIn('review-pilot-gate',graph['train-42-full']['dependencies'])
        self.assertIn('review-pilot-joint',graph['pretrain-101']['dependencies'])
        self.assertIn('review-replicated-gate',graph['train-101-full']['dependencies'])
        for name,job in graph.items():
            if name.startswith(('eval-','calibration-','adv-','vrx-')):
                self.assertIn('review-final',job['dependencies'])
        self.assertEqual(graph['gate-42-full']['command'][-2:],['--stop-after-stage','gate'])

    def test_review_cannot_continue_without_exact_evidence_and_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job=dict(name='review-pilot-gate',kind='review',stage='gate',seeds=[42])
            scheduler.write_json(study/'manifest.json',dict(jobs=[job]))
            data=study/'reviews/data-gate-42/completed.json'
            scheduler.write_json(data,dict(complete=True))
            evidence=study/'reviews/review-pilot-gate/evidence.json'
            scheduler.write_json(evidence,dict(technical_checks_passed=True,inputs={'42':scheduler.digest(data)}))
            self.assertFalse(scheduler.review_passed(job,study))
            fingerprint=scheduler.digest(evidence)
            reason='All technical checks passed; the limited one-seed validation warrants joint-stage diagnosis, not a final superiority claim.'
            with self.assertRaises(ValueError):
                scheduler.decide_review(study,job['name'],'continue',reason,'stale')
            with self.assertRaises(ValueError):
                scheduler.decide_review(study,job['name'],'continue','ok',fingerprint)
            scheduler.decide_review(study,job['name'],'hold',reason,fingerprint)
            self.assertFalse(scheduler.review_passed(job,study))
            scheduler.decide_review(study,job['name'],'continue',reason,fingerprint)
            self.assertTrue(scheduler.review_passed(job,study))
            scheduler.write_json(data,dict(complete=True,changed=True))
            self.assertFalse(scheduler.review_passed(job,study))


if __name__=='__main__':
    unittest.main()
