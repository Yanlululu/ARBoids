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
from interaction_review import review_evidence, deployment_evidence, ARMS, METRICS, CANDIDATE_RESPONSE_PROTOCOL


def candidate_response(seed=42):
    return dict(protocol=CANDIDATE_RESPONSE_PROTOCOL, resamples=4,
        states=[dict(state=i, scene_seed=350000000+scheduler.SEEDS.index(seed)*1000000+i,
                     perturbations=[{} for _ in range(4)]) for i in range(32)])


def automatic_pair_fixture(study):
    jobs=[]
    for arm,step in (('full',250000),('same_info',100000)):
        probe=study/f'training/seed-42/{arm}/probe-{step}.pth'
        probe.parent.mkdir(parents=True)
        probe.write_bytes(b'fixed development actor and critics')
        path=study/f'reviews/screens/42-{arm}-{step}/completed.json'
        scheduler.write_json(path,dict(complete=True,technical_checks_passed=True,step=step,
            input=dict(seed=42,arm=arm,checkpoint_sha256=scheduler.digest(probe)),
            frozen_proposals=dict(finite=True,frozen_proposals_equal=True),
            learning_checks=dict(td_loss=900.,bootstrap_td_loss=800.,bootstrap_disagreement=10.,
                td_target_abs_max=700.,critic_gradient_norm=100.,actor_gradient_norm=2.,adapter_gradient_norm=0.),
            summary=dict(E_model=30.,E_env=41.,environment_zero_predictor=40.,collision=0.,
                success=.6,capture=.5,capture_time=35.,executed_nonzero_fraction=1.)))
        jobs.append(dict(name=f'diagnostics-screen-42-{arm}-{step}',kind='cpu',
            completion=str(path),command=['python','--checkpoint-step',str(step)]))
    job=dict(name='review-pair-100000',kind='review',stage='development-v6',
        dependencies=[j['name'] for j in jobs],policy=dict(allowed_assessments=['technical_ready']))
    jobs.append(job)
    manifest=dict(format='interaction-study-v6',root=str(scheduler.ROOT),jobs=jobs,
        code={'scripts/run_interaction_study.py':'old'},technical_review_automation=scheduler.technical_review_policy())
    scheduler.write_json(study/'manifest.json',manifest)
    evidence=scheduler.build_signal_first_evidence(job,study,{j['name']:j for j in jobs})
    scheduler.write_json(study/'reviews'/job['name']/'evidence.json',evidence)
    return job,manifest


class StageSchedule(unittest.TestCase):
    def test_automatic_pair_check_passes_without_claiming_efficacy(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job,manifest=automatic_pair_fixture(study)
            self.assertTrue(scheduler.automatic_technical_review(job,study,manifest))
            self.assertTrue(scheduler.review_passed(job,study))
            decision=json.loads((study/'reviews'/job['name']/'decision.json').read_text())
            self.assertEqual(decision['assessment'],'technical_ready')
            self.assertEqual(decision['automatic_technical_review']['failures'],[])
            # A flat adapter gradient alone must not be relabelled as a failed mechanism.
            self.assertEqual(decision['automatic_technical_review']['measurements']['full']['step'],250000)
            self.assertFalse(scheduler.automatic_technical_review(job,study,manifest))

    def test_automatic_pair_check_holds_changed_or_invalid_evidence(self):
        for failure in ('probe','nan','calibration','collision','frozen','missing','over_budget'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as directory:
                study=Path(directory)
                job,manifest=automatic_pair_fixture(study)
                if failure=='probe':
                    (study/'training/seed-42/full/probe-250000.pth').write_bytes(b'changed after evaluation')
                else:
                    path=study/'reviews/screens/42-full-250000/completed.json'
                    data=json.loads(path.read_text())
                    if failure=='nan': data['learning_checks']['td_loss']=float('nan')
                    if failure=='calibration': data['summary']['E_env']=51.
                    if failure=='collision': data['summary']['collision']=1/32
                    if failure=='frozen': data['frozen_proposals']['frozen_proposals_equal']=False
                    if failure=='missing': del data['learning_checks']['bootstrap_td_loss']
                    if failure=='over_budget': data['step']=300000
                    scheduler.write_json(path,data)
                    evidence=scheduler.build_signal_first_evidence(job,study,{j['name']:j for j in manifest['jobs']})
                    scheduler.write_json(study/'reviews'/job['name']/'evidence.json',evidence)
                self.assertTrue(scheduler.automatic_technical_review(job,study,manifest))
                self.assertFalse(scheduler.review_passed(job,study))
                decision=json.loads((study/'reviews'/job['name']/'decision.json').read_text())
                self.assertEqual(decision['decision'],'hold')
                self.assertTrue(decision['automatic_technical_review']['failures'])

    def test_automatic_reviews_preserve_manual_holds_and_scientific_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job,manifest=automatic_pair_fixture(study)
            evidence=study/'reviews'/job['name']/'evidence.json'
            scheduler.decide_review(study,job['name'],'hold',
                'An inspected scientific concern requires diagnosis; automatic checks must never override this decision.',
                scheduler.digest(evidence),'unsupported')
            decision=evidence.with_name('decision.json')
            original=decision.read_bytes()
            self.assertFalse(scheduler.automatic_technical_review(job,study,manifest))
            self.assertEqual(decision.read_bytes(),original)
            for name in ('review-pair-250000','review-small-matrix','review-protocol-freeze'):
                other=dict(job,name=name,policy=dict(allowed_assessments=['promising']))
                self.assertFalse(scheduler.automatic_technical_review(other,study,manifest))
            self.assertFalse(scheduler.automatic_technical_review(job,study,dict(manifest,technical_review_automation=None)))

    def test_technical_review_migration_preserves_live_jobs_and_rejects_learner_edits(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job,manifest=automatic_pair_fixture(study)
            child=dict(name='develop-42-same_info-100000',kind='gpu',command=['python','learner'])
            manifest['jobs'].append(child)
            manifest['code']['train/train_interaction.py']='unchanged'
            scheduler.write_json(study/'manifest.json',manifest)
            record=dict(name=child['name'],pid=123,state='running',process_identity='456')
            record_path=study/'jobs'/f'{child["name"]}.json'
            scheduler.write_json(record_path,record)
            checkpoint=study/'training/seed-42/same_info/resume.pth'
            checkpoint.write_bytes(b'live learner state')
            fcntl=SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
            current=dict(manifest['code'],**{'scripts/run_interaction_study.py':'new'})
            with patch.dict(sys.modules,fcntl=fcntl),patch.object(scheduler,'source_hashes',return_value=current), \
                    patch.object(scheduler,'process_identity',return_value='456'), \
                    patch.object(scheduler,'verify_running_job',return_value='456') as verify:
                scheduler.migrate_technical_reviews(study)
            verify.assert_called_once_with(child,record)
            updated=json.loads((study/'manifest.json').read_text())
            self.assertEqual(updated['jobs'],manifest['jobs'])
            self.assertEqual(updated['technical_review_automation'],scheduler.technical_review_policy())
            self.assertEqual(checkpoint.read_bytes(),b'live learner state')
            self.assertEqual(json.loads(record_path.read_text()),record)
            before=(study/'manifest.json').read_bytes()
            current['train/train_interaction.py']='changed'
            with patch.dict(sys.modules,fcntl=fcntl),patch.object(scheduler,'source_hashes',return_value=current):
                with self.assertRaisesRegex(RuntimeError,'Only scheduler'):
                    scheduler.migrate_technical_reviews(study)
            self.assertEqual((study/'manifest.json').read_bytes(),before)

    def test_technical_review_migration_uses_completed_full_endpoint(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job,manifest=automatic_pair_fixture(study)
            job['dependencies'][0]='diagnostics-screen-42-full-215000'
            scheduler.write_json(study/'manifest.json',manifest)
            evidence=study/'reviews'/job['name']/'evidence.json'
            fcntl=SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
            with patch.dict(sys.modules,fcntl=fcntl),patch.object(scheduler,'source_hashes',return_value=manifest['code']):
                with self.assertRaisesRegex(RuntimeError,'cannot be silently rebound'):
                    scheduler.migrate_technical_reviews(study)
                evidence.unlink()
                scheduler.migrate_technical_reviews(study)
            updated=json.loads((study/'manifest.json').read_text())
            self.assertEqual(updated['jobs'][-1]['dependencies'][0],'diagnostics-screen-42-full-250000')

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
                scheduler.write_json(path,dict(summary={},episodes=rows,calibration=calibration,
                                               candidate_response=candidate_response()))
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
            jobs,_=scheduler.build_legacy_jobs(Path('/study'),'python',Path('/activate'))
        graph={j['name']:j for j in jobs}
        self.assertEqual(len(graph),len(jobs))
        self.assertEqual(len([j for j in jobs if j['kind']=='review']),9)
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
                         {f'gate-42-{a}' for a in (*scheduler.CORE_ARMS,'no_peer')})
        self.assertIn('diagnostics-arm-gate-42-initial_cbf',early)
        self.assertTrue(graph['review-core-gate']['peer_gate_control'])
        self.assertIn('diagnostics-peer-gate-42',graph['review-core-gate']['dependencies'])
        self.assertEqual(graph['review-core-gate']['policy']['next_stage_budget']['additional_training_steps'],4000000)
        self.assertEqual(graph['review-core-replication']['policy']['next_stage_budget']['additional_training_steps'],5000000)
        self.assertFalse(any(n.startswith(('train-','eval-','vrx-')) for n in early))
        approved.append('review-core-gate')
        pilot=reachable(approved)
        self.assertEqual({n for n in pilot if n.startswith('train-')},
                         {f'train-42-{a}' for a in (*scheduler.CORE_ARMS,'no_peer')})
        approved.append('review-core-pilot')
        replicated=reachable(approved)
        self.assertEqual({n for n in replicated if n.startswith('train-')},
                         {f'train-{s}-{a}' for s in (42,101) for a in scheduler.CORE_ARMS} | {'train-42-no_peer'})
        approved.append('review-core-replication')
        ablated=reachable(approved)
        self.assertEqual({n for n in ablated if n.startswith('train-')},
                         {f'train-{s}-{a}' for s in (42,101) for a in scheduler.ARMS} | {'train-42-no_peer'})
        self.assertNotIn('pretrain-202',ablated)
        self.assertNotIn('train-101-no_peer',ablated)
        self.assertNotIn('review-evidence', graph['review-peer-pilot']['dependencies'])
        approved.append('review-peer-pilot')
        candidate_replicated=reachable(approved)
        self.assertIn('train-101-no_peer',candidate_replicated)
        self.assertNotIn('pretrain-202',candidate_replicated)
        approved.append('review-peer-replication')
        approved.append('review-evidence')
        deployment=reachable(approved)
        self.assertEqual(len([n for n in deployment if n.startswith('vrx-pilot-')]),60)
        self.assertFalse(any(n.startswith('vrx-') and not n.startswith('vrx-pilot-') for n in deployment))
        approved.append('review-deployment')
        replicated_all=reachable(approved)
        self.assertEqual(len([n for n in replicated_all if n.startswith('train-')]),27)
        self.assertFalse(any(n.startswith(('eval-','calibration-')) for n in replicated_all))
        approved.append('review-submission')
        primary=reachable(approved)
        self.assertIn('figures',primary)
        self.assertFalse(any(n.startswith('adv-') for n in primary))
        self.assertNotIn('figures-extensions',primary)
        self.assertIn('--without-extensions',graph['figures']['command'])
        self.assertNotIn('--without-extensions',graph['figures-extensions']['command'])
        self.assertEqual(graph['gate-42-full']['command'][-2:],['--stop-after-stage','gate'])
        for name in ('gate-101-no_peer','train-101-no_peer','diagnostics-peer-gate-101',
                     'diagnostics-peer-joint-101','review-peer-replication'):
            self.assertEqual(graph[name]['condition'],dict(review='review-peer-pilot',assessment='inconclusive'))
        self.assertTrue(graph['review-evidence']['peer_control'])

    def test_parallel_slots_use_disjoint_cpu_budgets_and_reuse_freed_slots(self):
        manifest=dict(cpu_affinity=list(range(20)),cpu_per_gpu=4,cpu_per_evaluation=2,
                      limits=dict(gpu=4,cpu=2,vrx=1))
        jobs, active, allocations={}, {}, []
        for kind, count in (('gpu',4),('cpu',2)):
            for i in range(count):
                name=f'{kind}-{i}'
                jobs[name]=dict(kind=kind)
                slot, assigned=scheduler.resource_assignment(manifest,kind,jobs,active)
                self.assertEqual(slot,i)
                self.assertEqual(len(assigned),4 if kind=='gpu' else 2)
                active[name]=(None,None,slot)
                allocations.extend(assigned)
        self.assertEqual(sorted(allocations),list(range(20)))
        del active['gpu-1']
        slot, assigned=scheduler.resource_assignment(manifest,'gpu',jobs,active)
        self.assertEqual((slot,assigned),(1,[4,5,6,7]))

    def test_signal_first_graph_separates_development_freeze_and_formal_work(self):
        with patch.object(scheduler,'verified_existing_pretrain',return_value=None):
            jobs,_=scheduler.build_jobs(Path('/study'),'python',Path('/activate'))
        graph={j['name']:j for j in jobs}
        self.assertEqual(len(graph),len(jobs))
        done=set()
        while len(done)<len(graph):
            ready={n for n,j in graph.items() if n not in done and set(j['dependencies'])<=done}
            self.assertTrue(ready,'Signal-first graph is cyclic or has missing prerequisites')
            done.update(ready)
        def reachable(approvals):
            done=set()
            while True:
                ready={n for n,j in graph.items() if n not in done and set(j['dependencies'])<=done
                       and (j['kind']!='review' or n in approvals)}
                if not ready: return done
                done.update(ready)
        self.assertFalse(any(n.startswith('develop-') for n in reachable([])))
        self.assertIn('diagnostics-signal',reachable([]))
        approved=['review-signal']
        first=reachable(approved)
        self.assertEqual({n for n in first if n.startswith('develop-')},
                         {'develop-42-full-5000','develop-42-same_info-5000'})
        approved.extend(['review-pair-connectivity','review-pair-50000','review-pair-100000'])
        pair=reachable(approved)
        self.assertIn('develop-42-full-250000',pair)
        self.assertNotIn('pretrain-101',pair)
        self.assertFalse(any('no_peer' in n or 'model_value' in n or '-short-' in n for n in pair))
        approved.append('review-pair-250000')
        replicated=reachable(approved)
        self.assertIn('develop-101-no_peer-250000',replicated)
        self.assertNotIn('develop-42-short-250000',replicated)
        approved.append('review-development-replication')
        small=reachable(approved)
        for seed in scheduler.PILOT_SEEDS:
            for arm in scheduler.DEVELOPMENT_ARMS:
                self.assertIn(f'develop-{seed}-{arm}-250000',small)
        self.assertFalse(any(n.startswith('train-') for n in small))
        approved.append('review-small-matrix')
        self.assertEqual(len([n for n in reachable(approved) if n.startswith('diagnostics-confirmation-')]),10)
        approved.append('review-protocol-freeze')
        formal=reachable(approved)
        self.assertEqual(len([n for n in formal if n.startswith('train-')]),30)
        self.assertTrue(set(scheduler.PILOT_SEEDS).isdisjoint(scheduler.FORMAL_SEEDS))
        self.assertFalse(any(n.startswith(('vrx-','eval-','adv-')) for n in formal))
        for j in jobs:
            if j['name'].startswith('train-'):
                self.assertIn('--formal',j['command'])
                self.assertEqual(j['formal_steps'],1000000)
        approved.append('review-formal-core')
        self.assertIn('figures',reachable(approved))
        self.assertNotIn('figures-extensions',reachable(approved))

    def test_existing_development_is_audited_without_recreating_missing_early_checkpoints(self):
        with patch.object(scheduler,'verified_existing_pretrain',return_value=None):
            jobs,_=scheduler.build_jobs(Path('/study'),'python',Path('/activate'),
                {'42/full':210000,'42/same_info':30000,'42/model_value':20000,'42/no_peer':20000})
        graph={j['name']:j for j in jobs}
        self.assertNotIn('develop-42-full-5000',graph)
        self.assertNotIn('develop-42-full-100000',graph)
        self.assertIn('develop-42-same_info-50000',graph)
        self.assertIn('develop-42-full-250000',graph)
        self.assertIn('diagnostics-target-audit-42-full',graph['review-pair-connectivity']['dependencies'])
        self.assertEqual(graph['develop-42-no_peer-250000']['dependencies'][0],'review-pair-250000')

    def test_freeze_is_invalidated_by_protocol_change(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job=dict(name='review-protocol-freeze',stage='development-v6',kind='review',dependencies=[],
                     freezes_protocol=True,policy=dict(allowed_assessments=['frozen']))
            manifest=dict(candidate_protocol={'H':100,'formal_steps':1000000},code={'learner':'fixed'},jobs=[job])
            scheduler.write_json(study/'manifest.json',manifest)
            evidence=scheduler.build_signal_first_evidence(job,study,{job['name']:job})
            path=study/'reviews/review-protocol-freeze/evidence.json'
            scheduler.write_json(path,evidence)
            scheduler.decide_review(study,job['name'],'continue',
                'The complete matched development matrix and fresh confirmation justify freezing every listed comparison setting.',
                scheduler.digest(path),'frozen')
            self.assertTrue(scheduler.review_passed(job,study))
            manifest['candidate_protocol']['H']=10
            scheduler.write_json(study/'manifest.json',manifest)
            self.assertFalse(scheduler.review_passed(job,study))

    def test_signal_migration_preserves_checkpoint_bytes_and_rejects_live_jobs(self):
        import torch
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            with patch.object(scheduler,'verified_existing_pretrain',return_value=None):
                jobs,_=scheduler.build_legacy_jobs(study,'python',Path('/activate'))
            manifest=dict(format='interaction-study-v5',root=str(scheduler.ROOT),jobs=jobs,
                code={'scripts/run_interaction_study.py':'old'},cpu_affinity=list(range(20)))
            scheduler.write_json(study/'manifest.json',manifest)
            record=study/'jobs/gate-42-full.json'
            scheduler.write_json(record,dict(name='gate-42-full',state='paused_protocol_revision',pid=123))
            checkpoint=study/'training/seed-42/full/resume.pth'
            checkpoint.parent.mkdir(parents=True)
            config=dict(training=dict(gate_steps=250000,joint_steps=1000000,warm_steps=5000))
            torch.save(dict(step=210000,agent=dict(stage='gate'),config=config,
                            simulated_steps=1700000,elapsed=10000.),checkpoint)
            before=checkpoint.read_bytes()
            scheduler.write_json(checkpoint.parent/'progress.json',dict(step=213000))
            fcntl=SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
            with patch.dict(sys.modules,fcntl=fcntl),patch.object(scheduler,'process_identity',return_value='live'):
                with self.assertRaisesRegex(RuntimeError,'Stop owned jobs'):
                    scheduler.migrate_signal_first(study)
            with patch.dict(sys.modules,fcntl=fcntl),patch.object(scheduler,'process_identity',return_value=None), \
                 patch.object(scheduler,'source_hashes',return_value={'scripts/run_interaction_study.py':'new'}), \
                 patch.object(scheduler,'verified_existing_pretrain',return_value=None):
                scheduler.migrate_signal_first(study)
            updated=json.loads((study/'manifest.json').read_text())
            self.assertEqual(updated['format'],'interaction-study-v6')
            self.assertEqual(checkpoint.read_bytes(),before)
            self.assertEqual(updated['execution_revisions'][-1]['preserved_checkpoints']['42/full']['uncheckpointed_steps'],3000)
            self.assertIn('gate-42-full',updated['retired_jobs'])
            self.assertEqual(updated['development_resume_steps']['42/full'],210000)

    def test_explicit_scientific_hold_stops_idle_scheduler(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            job=dict(name='review-pair-connectivity',kind='review',stage='development-v6',dependencies=[],command=[],
                     policy=dict(allowed_assessments=['technical_ready']))
            scheduler.write_json(study/'manifest.json',dict(format='interaction-study-v6',root=str(scheduler.ROOT),
                code={},inherited_pretraining={},jobs=[job],limits=dict(gpu=2,cpu=2,vrx=1)))
            scheduler.write_json(study/'reviews'/job['name']/'evidence.json',
                dict(technical_checks_passed=True,inputs={},policy=job['policy']))
            scheduler.write_json(study/'reviews'/job['name']/'decision.json',dict(decision='hold'))
            fcntl=SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
            with patch.dict(sys.modules,fcntl=fcntl),patch.object(scheduler,'blocking_processes',return_value=[]), \
                 patch.object(scheduler.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)), \
                 patch.object(scheduler.subprocess,'Popen') as launch:
                scheduler.run(study)
            launch.assert_not_called()
            self.assertEqual(json.loads((study/'status.json').read_text())['state'],'held')

    def test_bootstrap_migration_retains_checkpoints_and_binds_recovery_probes(self):
        import torch
        import yaml
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            with patch.object(scheduler,'verified_existing_pretrain',return_value=None):
                jobs,_=scheduler.build_jobs(study,'python',Path('/activate'))
            manifest=dict(format='interaction-study-v6',root=str(scheduler.ROOT),jobs=jobs,
                code={'scripts/run_interaction_study.py':'old'},limits=dict(gpu=2,cpu=2,vrx=1),retired_jobs=[])
            scheduler.write_json(study/'manifest.json',manifest)
            originals={}
            for arm,step in (('full',215000),('same_info',35000),('no_peer',20000)):
                path=study/f'training/seed-42/{arm}'
                path.mkdir(parents=True)
                mode='coupled' if arm=='no_peer' else 'real_td'
                config=dict(interaction=dict(bootstrap_source=mode))
                (path/'config.yaml').write_text(yaml.safe_dump(config))
                torch.save(dict(step=step,config=config,agent=dict(stage='gate')),path/'resume.pth')
                originals[path/'resume.pth']=(path/'resume.pth').read_bytes()
                if arm=='no_peer': continue
                repair_step=step-5000
                (path/f'probe-{step}.pth').write_bytes(b'fixed post-update checkpoint')
                (path/f'probe-{repair_step}.pth').write_bytes(b'fixed recovery checkpoint')
                scheduler.write_json(study/f'reviews/bootstrap-repair/42-{arm}/completed.json',dict(
                    complete=True,input=dict(seed=42,arm=arm),passed=True,actor_unchanged=True,step=repair_step,
                    bootstrap_updates=10000,learning_wall_seconds=2.,calibration=dict(summary={},wall_seconds=1.),
                    repaired_probe_sha256=scheduler.digest(path/f'probe-{repair_step}.pth')))
            fcntl=SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
            scheduler.write_json(study/'status.json',dict(active=['still running']))
            with patch.dict(sys.modules,fcntl=fcntl):
                with self.assertRaisesRegex(RuntimeError,'active validation jobs'):
                    scheduler.migrate_bootstrap(study)
            scheduler.write_json(study/'status.json',dict(active=[]))
            with patch.dict(sys.modules,fcntl=fcntl),patch.object(scheduler,'process_identity',return_value=None), \
                 patch.object(scheduler,'source_hashes',return_value={'scripts/run_interaction_study.py':'new'}), \
                 patch.object(scheduler,'verified_existing_pretrain',return_value=None):
                scheduler.migrate_bootstrap(study)
            for path,raw in originals.items(): self.assertEqual(path.read_bytes(),raw)
            updated=json.loads((study/'manifest.json').read_text())
            graph={j['name']:j for j in updated['jobs']}
            self.assertIn('diagnostics-screen-42-full-215000',graph)
            self.assertIn('diagnostics-bootstrap-check-42-full-215000',graph)
            self.assertIn('diagnostics-bootstrap-repair-42-full',graph['review-pair-connectivity-bootstrap']['dependencies'])
            self.assertIn('review-pair-connectivity',updated['retired_jobs'])
            self.assertIn('--recondition-bootstrap-if-needed',graph['develop-42-no_peer-250000']['command'])
            self.assertNotIn('--recondition-bootstrap-if-needed',graph['develop-42-full-250000']['command'])
            self.assertEqual(graph['develop-42-full-250000']['dependencies'][0],'review-pair-connectivity-bootstrap')
            self.assertEqual(graph['develop-42-same_info-50000']['dependencies'][0],'review-pair-connectivity-bootstrap')
            self.assertIn('diagnostics-screen-42-full-250000',graph['review-pair-100000']['dependencies'])

    def test_recovery_review_detects_a_changed_fixed_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            probe=study/'training/seed-42/full/probe-210000.pth'
            probe.parent.mkdir(parents=True);probe.write_bytes(b'repaired critic')
            completion=study/'reviews/bootstrap-repair/42-full/completed.json'
            scheduler.write_json(completion,dict(complete=True,input=dict(seed=42,arm='full'),step=210000,
                repaired_probe_sha256=scheduler.digest(probe),passed=True,actor_unchanged=True,
                bootstrap_updates=10000,calibration=dict(summary={})))
            parent=dict(name='diagnostics-bootstrap-repair-42-full',kind='gpu',command=[],completion=str(completion))
            job=dict(name='review-pair-connectivity-bootstrap',kind='review',stage='development-v6',
                dependencies=[parent['name']],policy=dict(allowed_assessments=['technical_ready']))
            evidence=scheduler.build_signal_first_evidence(job,study,{parent['name']:parent})
            self.assertTrue(scheduler.evidence_inputs_match(job,study,evidence))
            probe.write_bytes(b'changed critic')
            self.assertFalse(scheduler.evidence_inputs_match(job,study,evidence))

    def test_adoption_checks_command_owner_and_process_identity(self):
        job=dict(command=['python','train/train_interaction.py','--output','/study/full'])
        record=dict(pid=123,process_identity='456')
        with patch.object(scheduler,'process_identity',return_value='456'), \
             patch.object(Path,'read_bytes',return_value=b'python\0train/train_interaction.py\0--output\0/study/full\0'), \
             patch.object(Path,'resolve',return_value=scheduler.ROOT), \
             patch.object(scheduler.os,'getpgid',return_value=123,create=True):
            self.assertEqual(scheduler.verify_running_job(job,record),'456')
            with self.assertRaisesRegex(RuntimeError,'does not match'):
                scheduler.verify_running_job(job,dict(pid=123,process_identity='reused-pid'))
            with self.assertRaisesRegex(RuntimeError,'does not match'):
                scheduler.verify_running_job(dict(command=['unrelated-work']),record)
        process=scheduler.AdoptedProcess(123,'456')
        with patch.object(scheduler,'process_identity',side_effect=['456',None,'other-start-time']):
            self.assertIsNone(process.poll())
            self.assertEqual(process.poll(),'unobserved')
            self.assertEqual(process.poll(),'unobserved')

    def test_early_migration_preserves_live_training_and_rejects_learner_changes(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            with patch.object(scheduler,'verified_existing_pretrain',return_value=None), \
                 patch.object(scheduler,'advance_candidate_screen',side_effect=lambda jobs,*args:jobs):
                jobs,_=scheduler.build_legacy_jobs(study,'python',Path('/activate'))
            old=dict(format='interaction-study-v4',root=str(scheduler.ROOT),jobs=jobs,cpu_affinity=list(range(20)),
                limits=dict(gpu=1,cpu=2,vrx=1,rollout_workers=4),
                code={'scripts/run_interaction_study.py':'old','train/train_interaction.py':'learner'})
            scheduler.write_json(study/'manifest.json',old)
            record=study/'jobs/gate-42-full.json'
            scheduler.write_json(record,dict(name='gate-42-full',pid=123,state='running'))
            checkpoint=study/'training/seed-42/full/resume.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'live checkpoint, never rewritten by migration')
            fcntl=SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
            with patch.dict(sys.modules,fcntl=fcntl), patch.object(scheduler,'process_identity',return_value='456'), \
                 patch.object(scheduler,'verify_running_job',return_value='456'), \
                 patch.object(scheduler,'source_hashes',return_value={**old['code'],'train/train_interaction.py':'unreviewed'}):
                with self.assertRaisesRegex(RuntimeError,'frozen learner'):
                    scheduler.migrate_early_screen(study)
                self.assertEqual(json.loads((study/'manifest.json').read_text()),old)
            with patch.dict(sys.modules,fcntl=fcntl), patch.object(scheduler,'process_identity',return_value='456'), \
                 patch.object(scheduler,'verify_running_job',return_value='456'), \
                 patch.object(scheduler,'source_hashes',return_value={**old['code'],'scripts/run_interaction_study.py':'new'}):
                scheduler.migrate_early_screen(study)
            updated=json.loads((study/'manifest.json').read_text())
            self.assertEqual(updated['format'],'interaction-study-v5')
            self.assertEqual(updated['limits']['gpu'],4)
            self.assertEqual(updated['execution_revisions'][-1]['preserved_running'],
                             [dict(name='gate-42-full',pid=123,process_identity='456')])
            self.assertEqual(json.loads(record.read_text())['state'],'running')
            self.assertEqual(checkpoint.read_bytes(),b'live checkpoint, never rewritten by migration')
            graph={j['name']:j for j in updated['jobs']}
            self.assertEqual(graph['gate-42-no_peer']['dependencies'],['pretrain-42'])
            self.assertEqual(graph['gate-42-full']['command'],next(j['command'] for j in jobs if j['name']=='gate-42-full'))

    def test_deployment_bank_is_paired_independent_and_keeps_task_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            with patch.object(scheduler,'verified_existing_pretrain',return_value=None):
                jobs,_=scheduler.build_legacy_jobs(study,'python',Path('/activate'))
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
                    calibration=[dict(state=i,absolute_error=0.) for i in range(32)],
                    candidate_response=candidate_response()))
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

    def test_candidate_replication_requires_the_exact_reviewed_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            parent=dict(name='review-peer-pilot',kind='review',stage='joint',seeds=[42],scope='peer',
                        policy=dict(allowed_assessments=['promising','inconclusive']))
            child=dict(name='gate-101-no_peer', condition=dict(review=parent['name'],assessment='inconclusive'))
            jobs={parent['name']:parent,child['name']:child}
            scheduler.write_json(study/'manifest.json',dict(jobs=list(jobs.values())))
            path=study/'reviews'/parent['name']/'evidence.json'
            scheduler.write_json(path,dict(technical_checks_passed=True,inputs={},policy=parent['policy']))
            with self.assertRaisesRegex(RuntimeError,'current reviewed'):
                scheduler.conditional_selection(child,study,jobs)
            rationale='Paired candidate-control pilot evidence was inspected; this decision releases only the stated bounded budget.'
            scheduler.decide_review(study,parent['name'],'continue',rationale,scheduler.digest(path),'promising')
            enabled, first_hash=scheduler.conditional_selection(child,study,jobs)
            self.assertFalse(enabled)
            scheduler.decide_review(study,parent['name'],'continue',rationale,scheduler.digest(path),'inconclusive')
            enabled, second_hash=scheduler.conditional_selection(child,study,jobs)
            self.assertTrue(enabled)
            self.assertNotEqual(first_hash,second_hash)
            scheduler.write_json(path,dict(technical_checks_passed=False))
            with self.assertRaisesRegex(RuntimeError,'current reviewed'):
                scheduler.conditional_selection(child,study,jobs)

    def test_skipped_replication_survives_resume_and_rejects_changed_branch(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            parent=dict(name='review-peer-pilot',kind='review',stage='joint',seeds=[42],scope='peer',
                dependencies=[],command=[],policy=dict(allowed_assessments=['promising','inconclusive']))
            children=[]
            previous=parent['name']
            for name, kind in (('gate-101-no_peer','gpu'),('train-101-no_peer','gpu'),
                               ('diagnostics-peer-joint-101','cpu'),('review-peer-replication','review')):
                children.append(dict(name=name,kind=kind,command=[],dependencies=[previous],
                    condition=dict(review=parent['name'],assessment='inconclusive')))
                previous=name
            scheduler.write_json(study/'manifest.json',dict(root=str(scheduler.ROOT),code={},jobs=[parent]+children,
                inherited_pretraining={},limits=dict(gpu=1,cpu=2,vrx=1)))
            path=study/'reviews'/parent['name']/'evidence.json'
            scheduler.write_json(path,dict(technical_checks_passed=True,inputs={},policy=parent['policy']))
            reason='This pilot supports bounded expansion with one candidate-control seed; replication is not requested by this assessment.'
            scheduler.decide_review(study,parent['name'],'continue',reason,scheduler.digest(path),'promising')
            scheduler.write_json(study/'jobs'/f'{parent["name"]}.json',dict(name=parent['name'],state='completed'))
            fcntl=SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
            with patch.dict(sys.modules,fcntl=fcntl), patch.object(scheduler,'blocking_processes',return_value=[]), \
                 patch.object(scheduler.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)), \
                 patch.object(scheduler.time,'sleep'), patch.object(scheduler.subprocess,'Popen') as launch:
                scheduler.run(study)
                self.assertEqual(json.loads((study/'status.json').read_text())['state'],'complete')
                for child in children:
                    self.assertEqual(json.loads((study/'jobs'/f'{child["name"]}.json').read_text())['state'],'skipped')
                scheduler.run(study)
                launch.assert_not_called()
                scheduler.decide_review(study,parent['name'],'continue',reason,scheduler.digest(path),'inconclusive')
                with self.assertRaisesRegex(RuntimeError,'conditional job'):
                    scheduler.run(study)

    def test_missing_candidate_response_cannot_be_reused_as_complete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            data=study/'reviews/data-gate-42'
            artifacts={}
            for arm in (*scheduler.CORE_ARMS,'initial_cbf'):
                path=data/f'{arm}.json'
                scheduler.write_json(path,dict(summary={},episodes=[dict(scene_seed=310420000+i,
                    **{m:0. for m in METRICS}) for i in range(100)],
                    calibration=[dict(state=i,absolute_error=0.) for i in range(32)]))
                artifacts[arm]=scheduler.digest(path)
            scheduler.write_json(data/'core-completed.json',dict(complete=True,artifacts=artifacts))
            with self.assertRaisesRegex(ValueError,'candidate response'):
                review_evidence(study,'gate',[42],'core')

    def test_review_costs_count_shared_results_once_and_include_candidate_work(self):
        import build_interaction_figures as figures
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            for stage in ('gate','joint'):
                data=study/f'reviews/data-{stage}-42'
                artifacts={}
                for arm in ('full','no_peer','initial_cbf'):
                    path=data/f'{arm}.json'
                    scheduler.write_json(path,dict(validation_steps=100,calibration_steps=20,
                        candidate_response_steps=7,wall_seconds=3.,candidate_response=dict(cbf_evaluations=512)))
                    artifacts[arm]=scheduler.digest(path)
                scheduler.write_json(data/'core-completed.json',dict(complete=True,
                    artifacts={a:h for a,h in artifacts.items() if a!='no_peer'}))
                scheduler.write_json(data/'peer-completed.json',dict(complete=True,artifacts=artifacts))
            with patch.object(figures,'SEEDS',(42,)):
                costs=figures.stage_review_costs(study,'interaction-study-v4')
                self.assertEqual(len(costs),2)
                for row in costs:
                    self.assertEqual(row['validation_steps'],300)
                    self.assertEqual(row['calibration_steps'],60)
                    self.assertEqual(row['candidate_response_steps'],21)
                    self.assertEqual(row['candidate_response_cbf_evaluations'],1536)
                    self.assertEqual(row['wall_seconds'],9.)
                (data/'no_peer.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'cost artifact changed'):
                    figures.stage_review_costs(study,'interaction-study-v4')

    def test_expansion_evidence_binds_the_selected_candidate_control(self):
        import interaction_review
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            parent=dict(name='review-peer-pilot',kind='review',stage='joint',scope='peer',seeds=[42],
                        policy=dict(allowed_assessments=['promising','inconclusive']))
            expansion=dict(name='review-evidence',stage='joint',scope='full',seeds=[42,101],peer_control=True,
                           policy=dict(allowed_assessments=['promising']))
            jobs={j['name']:j for j in (parent,expansion)}
            scheduler.write_json(study/'manifest.json',dict(jobs=list(jobs.values())))
            path=study/'reviews'/parent['name']/'evidence.json'
            scheduler.write_json(path,dict(technical_checks_passed=True,inputs={},policy=parent['policy']))
            reason='The paired candidate-control pilot supports further bounded validation; one seed does not establish robustness.'
            scheduler.decide_review(study,parent['name'],'continue',reason,scheduler.digest(path),'promising')
            def reports(study, stage, seeds, scope):
                filename=f'{stage}-{scope}.json'
                scheduler.write_json(study/filename,dict(seeds=list(seeds)))
                return dict(technical_checks_passed=True,stage=stage,seeds=list(seeds),
                    inputs={'fixture':scheduler.digest(study/filename)},input_paths={'fixture':filename},artifact_inputs={})
            with patch.object(interaction_review,'review_evidence',side_effect=reports):
                evidence=scheduler.build_review_evidence(expansion,study,jobs)
            self.assertEqual(evidence['peer_candidate_control']['seeds'],[42])
            self.assertEqual(evidence['peer_candidate_gate_stage']['stage'],'gate')
            self.assertTrue(scheduler.evidence_inputs_match(expansion,study,evidence))
            scheduler.write_json(study/'gate-peer.json',dict(changed=True))
            self.assertFalse(scheduler.evidence_inputs_match(expansion,study,evidence))

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

    def test_submission_exports_finish_without_adversarial_data(self):
        import build_interaction_figures as figures
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            study=root/'study'
            output=root/'figures'
            output.mkdir()
            manuscript=root/'paper.tex'
            manuscript.write_text('%% BEGIN GENERATED RESULTS\n%% END GENERATED RESULTS',encoding='utf-8')
            (root/'results.md').write_text('',encoding='utf-8')
            metrics=dict(success=1.,capture=1.,collision=0.,breach=0.,timeout=0.,capture_time=20.)
            effect=dict(difference=-2.,ci95=[-3.,-1.],seed_effects={str(s):-2. for s in scheduler.FORMAL_SEEDS})
            formal=dict(A=dict(effects=dict(capture_time=effect)),B=dict(E_env=dict(all=effect)),
                C=dict(strong=dict(capture_time=effect)),D=dict(configuration_matching=effect,
                    conditional_interaction=effect,noise_corrected_interaction_second_moment=effect),
                arm_summaries=[dict(arm='full',cell='n3-a2.25',mean=metrics,
                    ci95={k:[v,v] for k,v in metrics.items()},seed_standard_deviation={k:0. for k in metrics},
                    seed_means={k:{str(s):v for s in scheduler.FORMAL_SEEDS} for k,v in metrics.items()})])
            scheduler.write_json(study/'analysis.json',dict(complete=True,
                formal_evidence=formal,
                summaries=[dict(arm='full',cell='n3-a2.25',**metrics)],
                paired_comparisons=[dict(reference='arboids_cbf',cell='n3-a2.25',metric='capture_time',difference=0.,ci95=[0.,0.])]))
            scheduler.write_json(study/'manifest.json',dict(format='interaction-study-v2',jobs=[],
                inherited_pretraining={f'pretrain-{s}':dict(provenance='fixture') for s in scheduler.FORMAL_SEEDS}))
            scheduler.write_json(study/'runtime/completed.json',dict(complete=True,
                summaries=[dict(arm='full',defenders=3,median=.001,p95=.002)]))
            for seed in scheduler.FORMAL_SEEDS:
                for arm in scheduler.FORMAL_ARMS:
                    scheduler.write_json(study/f'training/seed-{seed}/{arm}/progress.json',
                        dict(step=1000000,complete=True,simulated_steps=100,elapsed=10.,updates=995001))
                    if arm!='arboids_cbf':
                        path=study/f'calibration/seed-{seed}/{arm}/calibration.csv'
                        path.parent.mkdir(parents=True)
                        path.write_text('absolute_error\n'+'1.0\n'*64,encoding='utf-8')
                for stage in ('gate','joint'):
                    scheduler.write_json(study/f'reviews/data-{stage}-{seed}/completed.json',
                        dict(validation_steps=100,calibration_steps=100,wall_seconds=1.))
            vrx=[dict(arm=a,training_seed=s,cell=f'n{n}-s{setting}',scene=i,**metrics)
                 for s in scheduler.FORMAL_SEEDS for a in ('full','same_info','arboids_cbf')
                 for n in (3,6) for setting in (0,1) for i in range(20)]
            with patch.object(figures,'ROOT',root), patch.object(figures,'chart',return_value='rendered chart'), \
                 patch.object(figures,'trajectories',return_value=[]), patch.object(figures,'learning',return_value=[]), \
                 patch.object(figures,'generalization',return_value=[]), patch.object(figures,'vrx_rows',return_value=vrx), \
                 patch.object(figures,'alternating',side_effect=AssertionError('Optional extension was read')):
                figures.finish(study,output,manuscript,include_extensions=False)
            result=json.loads((output/'tables.json').read_text())
            self.assertTrue(result['numerical']['complete'])
            self.assertFalse(result['extensions_complete'])
            self.assertEqual(result['common_opponents'],[])
            self.assertFalse((output/'table-common-opponents.csv').exists())
            self.assertFalse((study/'adversarial').exists())
            with (output/'table-formal-seed-effects.csv').open() as f:
                import csv
                seed_effects=list(csv.DictReader(f))
            self.assertEqual(len(seed_effects),30)
            self.assertEqual({int(r['seed']) for r in seed_effects},set(scheduler.FORMAL_SEEDS))
            self.assertTrue((output/'table-core-means.csv').exists())
            self.assertTrue((output/'table-core-seed-means.csv').exists())
            self.assertIn('capped-capture-time difference is -2.00 s',manuscript.read_text())

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
