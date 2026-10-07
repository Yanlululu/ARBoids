"""Formal effects preserve seed pairing, failed episodes, and the frozen evidence chain."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'train'))
sys.path.insert(0,str(ROOT/'scripts'))
import interaction_evaluation as evaluation
import run_interaction_study as scheduler


def csv_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fixture(study):
    write=scheduler.write_json
    protocol=dict(formal_evidence=evaluation.formal_evidence_standard())
    write(study/'manifest.json',dict(candidate_protocol=protocol,code={}))
    freeze=study/'reviews/review-protocol-freeze'
    write(freeze/'evidence.json',dict(frozen_protocol=protocol,frozen_source={}))
    write(freeze/'decision.json',dict(decision='continue',assessment='frozen',evidence_sha256=evaluation.digest(freeze/'evidence.json')))
    for seed in evaluation.SEEDS:
        for arm in scheduler.FORMAL_ARMS:
            directory=study/f'training/seed-{seed}/{arm}'
            write(directory/'progress.json',dict(step=1000000,complete=True))
            checkpoint=directory/'policy.pth';checkpoint.write_bytes(f'fixed {seed} {arm}'.encode())
            inputs=dict(checkpoint=str(checkpoint),checkpoint_sha256=evaluation.digest(checkpoint))
            primary=study/f'primary/seed-{seed}/{arm}'
            rows=[]
            for n,a in evaluation.CORE_CELLS:
                for index in range(200):
                    scene=560000000+evaluation.SEEDS.index(seed)*1000000+evaluation.CELLS.index((n,a))*10000+index
                    duration=18. if arm=='full' else 20.
                    rows.append(dict(arm=arm,training_seed=seed,cell=f'n{n}-a{a:g}',scene_seed=scene,
                        capture_time=duration,duration=duration,success=1,capture=1,collision=0,breach=0,timeout=0))
            csv_rows(primary/'episodes.csv',rows)
            write(primary/'completed.json',dict(complete=True,input=inputs,episodes_per_cell=200,
                cells=[list(c) for c in evaluation.CORE_CELLS],episodes_sha256=evaluation.digest(primary/'episodes.csv')))
            if arm=='arboids_cbf':continue
            calibration=study/f'calibration/seed-{seed}/{arm}'
            error=1. if arm=='full' else 3.
            rows=[dict(state=i,scene_seed=520000000+evaluation.SEEDS.index(seed)*1000000+i,
                training_seed=seed,defenders=3 if i<32 else 6,absolute_error=error,
                predicted_difference=1.+error,environment_difference=1.) for i in range(64)]
            csv_rows(calibration/'calibration.csv',rows)
            write(calibration/'completed.json',dict(complete=True,input=inputs,states=64,repetitions=4,
                calibration_sha256=evaluation.digest(calibration/'calibration.csv')))
        mechanism=study/f'mechanism/seed-{seed}'
        inputs=dict(repetitions=4)
        rows=[dict(scene_seed=530000000+evaluation.SEEDS.index(seed)*1000000+i,
            returns={'00':[10.]*4,'10':[9.]*4,'01':[8.]*4,'11':[9.]*4,'permuted':[8.]*4},
            executed_first_thrust={'00':[[1.,0.]],'permuted':[[0.,0.]]},
            trajectory_examples={'00':[{'time':0.}]} if i<2 else {}) for i in range(32)]
        write(mechanism/'states.json',dict(input=inputs,states=rows))
        write(mechanism/'completed.json',dict(complete=True,input=inputs,states_sha256=evaluation.digest(mechanism/'states.json')))


class FormalEvidence(unittest.TestCase):
    def test_registration_preserves_live_learner_and_precedes_formal_results(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            child=dict(name='develop-42-same_info-250000',kind='gpu',command=['python','learner'])
            core=dict(name='review-formal-core',kind='review',dependencies=['primary-202-full'],
                policy=dict(allowed_assessments=['technical_ready']))
            manifest=dict(format='interaction-study-v6',root=str(scheduler.ROOT),jobs=[child,core],
                code={'scripts/run_interaction_study.py':'old','train/train_interaction.py':'unchanged'},
                candidate_protocol=dict(H=100),technical_review_automation=scheduler.technical_review_policy())
            scheduler.write_json(study/'manifest.json',manifest)
            record=dict(name=child['name'],pid=123,state='running',process_identity='456')
            scheduler.write_json(study/'jobs'/f'{child["name"]}.json',record)
            checkpoint=study/'training/seed-42/same_info/resume.pth';checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'uninterrupted training state')
            fcntl=SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
            current=dict(manifest['code'],**{'scripts/run_interaction_study.py':'new'})
            with patch.dict(sys.modules,fcntl=fcntl),patch.object(scheduler,'source_hashes',return_value=current), \
                    patch.object(scheduler,'process_identity',return_value='456'), \
                    patch.object(scheduler,'verify_running_job',return_value='456'):
                scheduler.migrate_evidence_standard(study)
            updated=json.loads((study/'manifest.json').read_text())
            self.assertEqual(updated['jobs'][0],child)
            self.assertEqual(updated['jobs'][1]['dependencies'],['diagnostics-formal-evidence'])
            self.assertEqual(updated['jobs'][-1]['dependencies'],['primary-202-full'])
            self.assertEqual(updated['candidate_protocol']['formal_evidence'],evaluation.formal_evidence_standard())
            self.assertEqual(checkpoint.read_bytes(),b'uninterrupted training state')
            scheduler.write_json(study/'reviews/review-protocol-freeze/evidence.json',{})
            with patch.dict(sys.modules,fcntl=fcntl):
                with self.assertRaisesRegex(RuntimeError,'before protocol freeze'):
                    scheduler.migrate_evidence_standard(study)

    def test_formal_entry_requires_repeated_value_trend_and_explicit_assessments(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job=dict(name='review-protocol-freeze',kind='review',stage='development-v6',freezes_protocol=True,
                policy=dict(allowed_assessments=['frozen']))
            manifest=dict(jobs=[job],candidate_protocol={'standard':'fixed'},code={})
            scheduler.write_json(study/'manifest.json',manifest)
            path=study/'reviews'/job['name']/'evidence.json'
            evidence=dict(technical_checks_passed=True,inputs={},policy=job['policy'],frozen_source={},
                frozen_protocol=manifest['candidate_protocol'],formal_entry_evidence=dict(trends={
                    '42':dict(E_env_full_minus_same_info=-1.),'101':dict(E_env_full_minus_same_info=1.)}))
            scheduler.write_json(path,evidence)
            reason='The fixed development results and all control implementations were inspected before freezing the complete method.'
            findings={k:dict(status='supported',rationale=reason) for k in
                ('implementation','conditional_value','interaction','small_controls','fresh_confirmation','protocol')}
            with self.assertRaisesRegex(ValueError,'six explicit'):
                scheduler.decide_review(study,job['name'],'continue',reason,scheduler.digest(path),'frozen')
            with self.assertRaisesRegex(ValueError,'repeated conditional-value'):
                scheduler.decide_review(study,job['name'],'continue',reason,scheduler.digest(path),'frozen',findings=findings)
            evidence['formal_entry_evidence']['trends']['101']['E_env_full_minus_same_info']=-.2
            scheduler.write_json(path,evidence)
            scheduler.decide_review(study,job['name'],'continue',reason,scheduler.digest(path),'frozen',findings=findings)
            self.assertTrue(scheduler.review_passed(job,study))

    def test_paired_effect_preserves_all_seeds_without_unanimity(self):
        values=np.repeat(np.array([-2.,-2.,-2.,-2.,.1])[:,None],40,axis=1)
        effect=evaluation.paired_effect(values,np.random.default_rng(13),repeats=2000)
        self.assertEqual(len(effect['seed_effects']),5)
        self.assertEqual(effect['negative_seed_count'],4)
        self.assertLess(effect['ci95'][1],0)
        self.assertEqual(len(effect['leave_one_seed_out']),5)
        self.assertAlmostEqual(effect['difference'],-1.58)
        values=np.zeros((5,40));values[:,-1]=-400
        effect=evaluation.paired_effect(values,np.random.default_rng(13),repeats=100)
        self.assertLess(effect['difference'],0)
        self.assertEqual(effect['trimmed_10pct_sensitivity'],0.)

    def test_complete_five_seed_evidence_and_failure_metric_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory);fixture(study)
            interval=evaluation.hierarchical_interval
            with patch.object(evaluation,'hierarchical_interval',side_effect=lambda v,r,repeats=10000:interval(v,r,100)):
                result=evaluation.formal_evidence(study)
            self.assertEqual(result['A']['effects']['capture_time']['difference'],-2.)
            self.assertEqual(result['B']['E_env']['all']['difference'],-2.)
            self.assertEqual(result['C']['strong']['capture_time']['difference'],-2.)
            self.assertEqual(result['D']['configuration_matching']['difference'],2.)
            self.assertEqual(result['D']['conditional_interaction']['difference'],2.)
            self.assertEqual(result['D']['noise_corrected_interaction_second_moment']['difference'],4.)
            self.assertTrue(all(result['statistical_checks'].values()))
            self.assertFalse(result['practical_relevance_assessed'])
            self.assertEqual(len(result['arm_summaries']),12)
            self.assertEqual(len(result['D']['trajectory_examples']),10)
            primary=study/'primary/seed-202/full'
            with (primary/'episodes.csv').open() as f:rows=list(csv.DictReader(f))
            rows[0].update(capture='0',success='0',capture_time='18')
            csv_rows(primary/'episodes.csv',rows)
            record=json.loads((primary/'completed.json').read_text())
            record['episodes_sha256']=evaluation.digest(primary/'episodes.csv')
            scheduler.write_json(primary/'completed.json',record)
            with self.assertRaisesRegex(ValueError,'every noncapture'):
                evaluation.formal_evidence(study)
            rows[0]['capture_time']='60';csv_rows(primary/'episodes.csv',rows)
            record['episodes_sha256']=evaluation.digest(primary/'episodes.csv')
            scheduler.write_json(primary/'completed.json',record)
            with patch.object(evaluation,'hierarchical_interval',side_effect=lambda v,r,repeats=10000:interval(v,r,100)):
                result=evaluation.formal_evidence(study)
            self.assertGreater(result['A']['effects']['capture_time']['difference'],-2.)
            (study/'training/seed-202/full/policy.pth').write_bytes(b'changed model')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):
                evaluation.formal_evidence(study)

    def test_formal_review_requires_practical_assessment_and_statistical_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job=dict(name='review-formal-core',kind='review',stage='development-v6',requires_claim_assessment=True,
                policy=dict(allowed_assessments=['supported']))
            scheduler.write_json(study/'manifest.json',dict(jobs=[job]))
            path=study/'reviews/review-formal-core/evidence.json'
            evidence=dict(technical_checks_passed=True,inputs={},policy=job['policy'],
                formal_evidence=dict(statistical_checks={k:True for k in 'ABCD'}))
            scheduler.write_json(path,evidence)
            rationale='All paired A--D effects and the practical relevance of saved seconds have been inspected in their task context.'
            with self.assertRaisesRegex(ValueError,'explicit A--D'):
                scheduler.decide_review(study,job['name'],'continue',rationale,scheduler.digest(path),'supported')
            findings={k:dict(status='supported',rationale=rationale) for k in 'ABCD'}
            with self.assertRaisesRegex(ValueError,'absolute seconds'):
                scheduler.decide_review(study,job['name'],'continue',rationale,scheduler.digest(path),'supported',findings=findings)
            findings['A']['practical_relevance']=rationale
            scheduler.decide_review(study,job['name'],'continue',rationale,scheduler.digest(path),'supported',findings=findings)
            self.assertTrue(scheduler.review_passed(job,study))
            evidence['formal_evidence']['statistical_checks']['B']=False
            scheduler.write_json(path,evidence)
            with self.assertRaisesRegex(ValueError,'do not support'):
                scheduler.decide_review(study,job['name'],'continue',rationale,scheduler.digest(path),'supported',findings=findings)


if __name__=='__main__':unittest.main()
