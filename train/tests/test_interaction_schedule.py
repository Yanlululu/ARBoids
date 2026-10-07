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
from interaction_review import review_evidence, deployment_evidence, ARMS, METRICS


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
        self.assertEqual(len([j for j in jobs if j['kind']=='review']),7)
        done=set()
        while len(done)<len(jobs):
            ready={n for n,j in graph.items() if n not in done and set(j['dependencies'])<=done}
            self.assertTrue(ready,'Cyclic or missing dependencies')
            done.update(ready)
        def reachable(approved):
            done=set()
            while True:
                ready={n for n,j in graph.items() if n not in done and set(j['dependencies'])<=done
                       and (j['kind']!='review' or n in approved)}
                if not ready: return done
                done.update(ready)
        approved=[]
        early=reachable(approved)
        self.assertEqual({n for n in early if n.startswith('gate-')},
                         {f'gate-42-{a}' for a in scheduler.CORE_ARMS})
        self.assertFalse(any(n.startswith(('train-','eval-','vrx-')) for n in early))
        approved.append('review-core-gate')
        pilot=reachable(approved)
        self.assertEqual({n for n in pilot if n.startswith('train-')},
                         {f'train-42-{a}' for a in scheduler.CORE_ARMS})
        approved.append('review-core-pilot')
        replicated=reachable(approved)
        self.assertEqual({n for n in replicated if n.startswith('train-')},
                         {f'train-{s}-{a}' for s in (42,101) for a in scheduler.CORE_ARMS})
        approved.append('review-core-replication')
        ablated=reachable(approved)
        self.assertEqual({n for n in ablated if n.startswith('train-')},
                         {f'train-{s}-{a}' for s in (42,101) for a in scheduler.ARMS})
        self.assertNotIn('pretrain-202',ablated)
        approved.append('review-evidence')
        deployment=reachable(approved)
        self.assertEqual(len([n for n in deployment if n.startswith('vrx-pilot-')]),60)
        self.assertFalse(any(n.startswith('vrx-') and not n.startswith('vrx-pilot-') for n in deployment))
        approved.append('review-deployment')
        replicated_all=reachable(approved)
        self.assertEqual(len([n for n in replicated_all if n.startswith('train-')]),25)
        self.assertFalse(any(n.startswith(('eval-','calibration-')) for n in replicated_all))
        approved.append('review-submission')
        primary=reachable(approved)
        self.assertIn('figures',primary)
        self.assertFalse(any(n.startswith('adv-') for n in primary))
        self.assertNotIn('figures-extensions',primary)
        self.assertIn('--without-extensions',graph['figures']['command'])
        self.assertNotIn('--without-extensions',graph['figures-extensions']['command'])
        self.assertEqual(graph['gate-42-full']['command'][-2:],['--stop-after-stage','gate'])

    def test_deployment_bank_is_paired_independent_and_keeps_task_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            with patch.object(scheduler,'verified_existing_pretrain',return_value=None):
                jobs,_=scheduler.build_jobs(study,'python',Path('/activate'))
            scheduler.write_json(study/'manifest.json',dict(jobs=jobs))
            final_scenes={int(j['command'][2].split('--seed ')[1].split()[0]) for j in jobs
                          if j['name'].startswith('vrx-') and j.get('phase')=='submission'}
            for job in (j for j in jobs if j.get('phase')=='deployment-validation'):
                spec=job['validation']
                self.assertNotIn(spec['scene_seed'],final_scenes)
                self.assertIn(f'--seed {spec["scene_seed"]}',job['command'][2])
                checkpoint=study/f'training/seed-{spec["seed"]}/{spec["arm"]}/policy.pth'
                checkpoint.parent.mkdir(parents=True,exist_ok=True)
                checkpoint.write_bytes(b'fixed validation policy')
                scheduler.write_json(job['result'],dict(passed=True,outcome_code=1,seed=spec['scene_seed'],
                    setting=spec['setting'],num_robots=spec['defenders']+1,controller='IACRRL',
                    duration_limit=60,termination_rule='paper',agility=2.25,checkpoint_sha256=scheduler.digest(checkpoint),
                    simulation_seconds=20.,defender_collision=False,terminal_positions=[[0.,0.]],target_radius=15.))
            evidence=deployment_evidence(study)
            self.assertEqual(len(evidence['episodes']),60)
            self.assertTrue(evidence['technical_checks_passed'])
            self.assertTrue(all(r['success']==0 and r['capture_time']==60 for r in evidence['episodes']))
            self.assertEqual(len(evidence['seed_contrasts']),8)
            checkpoint.write_bytes(b'changed policy')
            with self.assertRaisesRegex(ValueError,'mismatched'):
                deployment_evidence(study)

    def test_core_evidence_survives_adding_ablations_but_not_editing_cached_results(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            data=study/'reviews/data-joint-42'
            artifacts={}
            for arm in (*scheduler.CORE_ARMS,'initial_cbf'):
                path=data/f'{arm}.json'
                scheduler.write_json(path,dict(summary={},episodes=[dict(scene_seed=310420000+i,
                    **{m:0. for m in METRICS}) for i in range(100)],
                    calibration=[dict(state=i,absolute_error=0.) for i in range(32)]))
                artifacts[arm]=scheduler.digest(path)
            scheduler.write_json(data/'core-completed.json',dict(complete=True,artifacts=artifacts))
            job=dict(name='review-core-pilot',kind='review',stage='joint',scope='core',seeds=[42],
                     policy=dict(allowed_assessments=['promising','inconclusive']))
            scheduler.write_json(study/'manifest.json',dict(jobs=[job]))
            evidence=review_evidence(study,'joint',[42],'core')
            evidence['policy']=job['policy']
            path=study/'reviews'/job['name']/'evidence.json'
            scheduler.write_json(path,evidence)
            reason='One seed remains inconclusive; only the prescribed second seed is warranted, not the full submission matrix.'
            scheduler.decide_review(study,job['name'],'continue',reason,scheduler.digest(path),'inconclusive')
            scheduler.write_json(data/'completed.json',dict(complete=True,artifacts={**artifacts,'short':'new','arboids_cbf':'new'}))
            self.assertTrue(scheduler.review_passed(job,study))
            scheduler.write_json(data/'full.json',dict(changed=True))
            self.assertFalse(scheduler.review_passed(job,study))

    def test_technical_success_cannot_bypass_scientific_hold(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job=dict(name='review-core-replication',kind='review',stage='joint',seeds=[42,101],
                     policy=dict(allowed_assessments=['promising'],next_stage_budget=dict(additional_training_steps=5000000)))
            scheduler.write_json(study/'manifest.json',dict(jobs=[job]))
            data=study/'completed.json'
            scheduler.write_json(data,dict(complete=True))
            path=study/'reviews'/job['name']/'evidence.json'
            scheduler.write_json(path,dict(technical_checks_passed=True,inputs={'data':scheduler.digest(data)},
                input_paths={'data':'completed.json'},policy=job['policy']))
            reason='The mechanism and paired task contrasts must support further spending across both seeds before expanding the budget.'
            for assessment in (None,'technical_ready','inconclusive','unsupported'):
                with self.assertRaisesRegex(ValueError,'assessment'):
                    scheduler.decide_review(study,job['name'],'continue',reason,scheduler.digest(path),assessment)
            scheduler.decide_review(study,job['name'],'hold',reason,scheduler.digest(path),'inconclusive')
            self.assertFalse(scheduler.review_passed(job,study))
            scheduler.decide_review(study,job['name'],'continue',reason,scheduler.digest(path),'promising')
            self.assertTrue(scheduler.review_passed(job,study))
            job['policy']['next_stage_budget']['additional_training_steps']=9999999
            self.assertFalse(scheduler.review_passed(job,study))

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
