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
from interaction_review import review_evidence, ARMS, CORE_ARMS, SEEDS, METRICS, CANDIDATE_RESPONSE_PROTOCOL


def candidate_response(seed=42):
    return dict(protocol=CANDIDATE_RESPONSE_PROTOCOL, resamples=4,
        states=[dict(state=i, scene_seed=350000000+SEEDS.index(seed)*1000000+i,
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
    def test_protocol_describes_the_actual_configured_rollout_and_batch_budget(self):
        config = scheduler.yaml.safe_load((scheduler.ROOT/'train/configs/interaction-aware-sac.yaml').read_text())
        config['interaction'].update(horizon_steps=300, pairs_per_batch=32, intervention_interval=500)
        config['rl']['batch_size'] = 2048
        with patch.object(scheduler.yaml, 'safe_load', return_value=config):
            protocol = scheduler.protocol_specification()
        self.assertEqual(protocol['long_horizon_steps'], 300)
        self.assertEqual(protocol['auxiliary_pairs_per_update'], 32)
        self.assertEqual(protocol['real_replay_batch'], 2048)
        self.assertEqual(protocol['auxiliary_to_real_batch_ratio'], 32/2048)
        self.assertIn('32 snapshots every 500 real steps', protocol['intervention_sampling'])

    def test_formal_vrx_pairs_scenes_and_retains_task_failures(self):
        import shlex
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            with patch.object(scheduler, 'verified_existing_pretrain', return_value=None):
                jobs, _ = scheduler.build_jobs(study, 'python', Path('/activate'))
            vrx = [j for j in jobs if j['name'].startswith('vrx-')]
            self.assertEqual(len(vrx), 1200)
            pairs = {}
            for job in vrx:
                _, seed, arm, defenders, setting, trial = job['name'].split('-')
                command = shlex.split(job['command'][2])
                scene = int(command[command.index('--seed') + 1])
                self.assertIn(int(seed), scheduler.FORMAL_SEEDS)
                self.assertGreaterEqual(scene, 540000000)
                pairs.setdefault((seed, defenders, setting, trial), {})[arm] = scene
                self.assertIn('review-formal-core', job['dependencies'])
            for arms in pairs.values():
                self.assertEqual(set(arms), set(scheduler.VRX_ARMS))
                self.assertEqual(len(set(arms.values())), 1)
            scheduler.write_json(vrx[0]['result'], dict(passed=True, outcome_code=1))
            self.assertTrue(scheduler.valid_completion(vrx[0], study))
            scheduler.write_json(vrx[0]['result'], dict(passed=False, outcome_code=1))
            self.assertFalse(scheduler.valid_completion(vrx[0], study))

    def test_runner_releases_lock_when_source_validation_fails(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            scheduler.write_json(study/'manifest.json', dict(root=str(scheduler.ROOT),
                code={'scripts/run_interaction_study.py': 'stale'}))
            handles = []
            original_open = Path.open
            def opened(path, *args, **kwargs):
                handle = original_open(path, *args, **kwargs)
                if path.name == 'runner.lock':
                    handles.append(handle)
                return handle
            fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
            with patch.dict(sys.modules, {'fcntl': fcntl}), patch.object(Path, 'open', opened):
                with self.assertRaisesRegex(RuntimeError, 'Frozen scientific code changed'):
                    scheduler.run(study)
            self.assertEqual(len(handles), 1)
            self.assertTrue(handles[0].closed)

    def test_performance_search_spends_its_budget_on_candidates_without_evidence_barriers(self):
        previous_jobs = [dict(name=f'develop-42-{arm}-10000', command=['python', 'train/train_interaction.py',
            '--output', f'/previous/training/seed-42/{arm}', '--arm', arm, '--stop-at-step', '10000'])
            for arm in ('full', 'short')]
        jobs, candidates = scheduler.performance_search_jobs('/search', 'python', '/previous', previous_jobs, '/source.pth')
        active = {'time_single_n3', 'time_single_n6', 'time_team_n3', 'time_team_n6'}
        self.assertEqual(set(candidates), {'long', 'short'} | active)
        self.assertEqual(len([j for j in jobs if j['kind'] == 'gpu']), 12)
        self.assertFalse(any(j['kind'] == 'review' or j['name'].startswith('train-') for j in jobs))
        graph = {j['name']: j for j in jobs}; done = set()
        while len(done) < len(graph):
            ready = {n for n, j in graph.items() if n not in done and set(j['dependencies']) <= done}
            self.assertTrue(ready); done.update(ready)
        for name in active:
            first = graph[f'develop-performance-{name}-10000']
            self.assertEqual(first['dependencies'], [])
            self.assertEqual(graph[f'develop-performance-{name}-50000']['dependencies'],
                             [f'develop-performance-{name}-25000'])
        for name in active:
            command = candidates[name]['command']
            self.assertEqual(command[command.index('--reward-objective') + 1], 'capped-time-v1')
            self.assertEqual(command[command.index('--gate-objective') + 1], 'paired-improvement-v1')
            self.assertEqual(command[command.index('--defenders') + 1], name[-1])
            self.assertEqual(command[command.index('--intervention-scope') + 1], name.split('_')[1])
        for name in ('long', 'short'):
            self.assertFalse(candidates[name]['train'])
            self.assertFalse(any(j['kind'] == 'gpu' and j.get('candidate') == name for j in jobs))
            for step in (5000, 10000): self.assertIn(f'diagnostics-performance-{name}-{step}', graph)

    def test_performance_summary_pairs_all_failures_without_filtering(self):
        baseline = [dict(defenders=n, agility=2.25, scene_seed=n * 100 + i, capture_time=t, capture=int(t < 60),
            success=int(t < 60), collision=0, breach=int(t == 60), timeout=0)
            for n in (3, 6) for i, t in enumerate((20., 60., 40.))]
        rows = [dict(row, capture_time=row['capture_time'] - (5. if row['capture'] else 0.)) for row in baseline]
        result = scheduler.performance_summary(rows, baseline)
        for cell in result.values():
            self.assertEqual(cell['episodes'], 3)
            self.assertAlmostEqual(cell['time_difference_from_source']['mean'], -10. / 3)
            self.assertAlmostEqual(cell['capture'], 2. / 3)
        with self.assertRaisesRegex(ValueError, 'Duplicate reference scenes'):
            scheduler.performance_summary(rows, baseline + baseline[:1])
        with self.assertRaisesRegex(ValueError, 'Duplicate candidate scenes'):
            scheduler.performance_summary(rows + rows[:1], baseline)
        with self.assertRaisesRegex(ValueError, 'same paired scenes'):
            scheduler.performance_summary(rows[:-1], baseline)
        with self.assertRaisesRegex(ValueError, 'same paired scenes'):
            scheduler.performance_summary([dict(r, agility=1.5) for r in rows], baseline)
        self.assertEqual(set(scheduler.performance_summary([dict(r, agility=1.5) for r in rows])),
                         {'n3-a1.5', 'n6-a1.5'})

    def test_paired_task_regression_holds_even_when_calibration_passes(self):
        for change in (0., 12.):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                study=Path(directory)
                job,manifest=automatic_pair_fixture(study)
                manifest['technical_review_automation']=scheduler.technical_review_policy(True,True)
                scheduler.write_json(study/'manifest.json',manifest)
                for arm,step in (('full',250000),('same_info',100000)):
                    episodes=[dict(scene_seed=452200000+i,defenders=3,capture_time=20.+i/10) for i in range(32)]
                    scheduler.write_json(study/f'reviews/screens/42-{arm}-5000/completed.json',dict(episodes=episodes))
                    current=study/f'reviews/screens/42-{arm}-{step}/completed.json'
                    data=json.loads(current.read_text())
                    data['episodes']=[dict(r,capture_time=r['capture_time']+change) for r in episodes]
                    scheduler.write_json(current,data)
                evidence=scheduler.build_signal_first_evidence(job,study,{j['name']:j for j in manifest['jobs']})
                scheduler.write_json(study/'reviews'/job['name']/'evidence.json',evidence)
                self.assertTrue(scheduler.automatic_technical_review(job,study,manifest))
                decision=json.loads((study/'reviews'/job['name']/'decision.json').read_text())
                self.assertEqual(decision['decision'],'continue' if change==0 else 'hold')
                measured=decision['automatic_technical_review']['measurements']['full']['task_change_from_5000']
                self.assertAlmostEqual(measured['mean'],change)

    def test_composed_revision_is_fresh_matched_and_retains_formal_evidence_gates(self):
        shared=dict(name='pretrain-101',kind='gpu',dependencies=['old-review'],
            command=['python','train/formal_study_training.py','--arm','pretrain','--output','/previous/pretrain/seed-101'])
        with patch.object(scheduler,'verified_existing_pretrain',return_value=None):
            jobs,_,common=scheduler.composed_revision_jobs(Path('/revision'),'python',Path('/activate'),shared)
        graph={j['name']:j for j in jobs}
        self.assertEqual(common,str(Path('/previous/pretrain/seed-101/actor.pth')))
        self.assertEqual(graph['pretrain-101']['command'],shared['command'])
        self.assertEqual(graph['pretrain-101']['dependencies'],[])
        for name,j in graph.items():
            if name.startswith(('develop-','train-')):
                c=j['command'];arm=c[c.index('--arm')+1]
                self.assertNotIn('--recondition-bootstrap-if-needed',c)
                self.assertNotIn('--isolate-bootstrap-if-needed',c)
                if arm!='arboids_cbf':
                    self.assertEqual(c[c.index('--critic-control-coordinates')+1],'nominal-thrust-v2')
                    self.assertEqual(c[c.index('--label-repetitions')+1],'8')
                    self.assertEqual(c[c.index('--critic-warmup-updates')+1],'5000')
                    self.assertEqual(c[c.index('--entropy-objective')+1],'proposal-mean-v3')
                    self.assertEqual(c[c.index('--critic-warmup-target')+1],'complete_real_state_return')
                    self.assertEqual(float(c[c.index('--policy-learning-rate')+1]),1e-5)
                    self.assertEqual(c[c.index('--long-horizon-steps')+1],'300')
                    self.assertEqual(c[c.index('--bootstrap-estimator')+1],'mean')
                if name.startswith('develop-101-'): self.assertEqual(c[c.index('--pretrain')+1],common)
            if name.startswith('train-'): self.assertIn('review-protocol-freeze',j['dependencies'])
        done=set()
        while len(done)<len(graph):
            ready={n for n,j in graph.items() if n not in done and set(j['dependencies'])<=done}
            self.assertTrue(ready,'Revision must be acyclic with every fixed endpoint reachable')
            done.update(ready)
        for arm in ('no_peer','short','model_value'):
            for step in (5000,10000,50000,250000): self.assertEqual(graph[f'develop-42-{arm}-{step}']['endpoint_step'],step)
        for arm in scheduler.DEVELOPMENT_ARMS:
            self.assertEqual(graph[f'develop-101-{arm}-5000']['dependencies'],['pretrain-101'])
            self.assertIn(f'develop-101-{arm}-5000',graph[f'develop-101-{arm}-250000']['dependencies'])
            self.assertIn('review-pair-10000',graph[f'develop-42-{arm}-50000']['dependencies'])
        self.assertIn('review-pair-10000',scheduler.technical_review_policy(True,True)['reviews'])
        self.assertTrue(graph['review-formal-core']['requires_claim_assessment'])

    def test_development_revisions_skip_the_reserved_formal_bank_family(self):
        self.assertEqual(scheduler.next_development_bank_offset(0), 20000000)
        self.assertEqual(scheduler.next_development_bank_offset(80000000), 100000000)
        self.assertEqual(scheduler.next_development_bank_offset(100000000), 410000000)
        self.assertEqual(scheduler.next_development_bank_offset(410000000), 430000000)
        with self.assertRaises(ValueError): scheduler.next_development_bank_offset(-1)
        with self.assertRaises(ValueError): scheduler.next_development_bank_offset(2**31)

    def test_parallel_development_preserves_budgets_commands_and_formal_barrier(self):
        with patch.object(scheduler,'verified_existing_pretrain',return_value=None):
            original,_=scheduler.build_jobs(Path('/study'),'python',Path('/activate'))
        updated=scheduler.parallel_development_jobs(original,Path('/study'),'python')
        graph={j['name']:j for j in updated}
        self.assertEqual(len(graph),len(updated))
        for job in original:
            self.assertEqual(graph[job['name']]['command'],job['command'])
        done=set()
        approvals={'review-signal','review-pair-connectivity','review-pair-50000','review-pair-100000'}
        while True:
            ready={n for n,j in graph.items() if n not in done and set(j['dependencies'])<=done
                and (j['kind']!='review' or n in approvals)}
            if not ready: break
            done.update(ready)
        for seed in scheduler.PILOT_SEEDS:
            for arm in scheduler.DEVELOPMENT_ARMS:
                self.assertIn(f'develop-{seed}-{arm}-250000',done)
                self.assertEqual(graph[f'develop-{seed}-{arm}-250000']['endpoint_step'],250000)
                self.assertIn(f'diagnostics-confirmation-{seed}-{arm}-250000',done)
            self.assertIn(f'diagnostics-development-mechanism-{seed}',done)
            for arm in ('full','same_info'):
                self.assertIn(f'diagnostics-precision-{seed}-{arm}-250000',done)
        self.assertFalse(any(n.startswith('train-') for n in done))
        self.assertFalse(any(f'pretrain-{s}' in done for s in scheduler.FORMAL_SEEDS))
        self.assertTrue({'review-pair-250000','review-development-replication','review-small-matrix'}<=
                        set(graph['review-protocol-freeze']['dependencies']))
        while len(done)<len(graph):
            ready={n for n,j in graph.items() if n not in done and set(j['dependencies'])<=done}
            self.assertTrue(ready,'Parallel development graph must be acyclic and complete')
            done.update(ready)
        self.assertEqual(updated,scheduler.parallel_development_jobs(updated,Path('/study'),'python'))

    def test_precision_preserves_noise_correction_and_rejects_mismatched_repetitions(self):
        from interaction_review import precision_summary
        rows=[dict(return_differences=[-1.,1.],critic_predictions=[0.,0.],predicted_difference=0.)]
        summary=precision_summary(rows)
        self.assertEqual(summary['E_env'],0.)
        self.assertEqual(summary['mse_excess_over_zero'],0.)
        self.assertEqual(summary['prediction_mse_noise_corrected'],-1.)
        self.assertEqual(summary['zero_mse_noise_corrected'],-1.)
        rows[0]['critic_predictions']=[1.,2.]
        rows[0]['predicted_difference']=1.
        summary=precision_summary(rows)
        self.assertEqual(summary['E_env'],1.)
        self.assertEqual(summary['prediction_mse_noise_corrected'],0.)
        self.assertEqual(summary['mse_excess_over_zero'],1.)
        with self.assertRaisesRegex(ValueError, 'minimum-head'):
            precision_summary([dict(return_differences=[-1., 1.], critic_predictions=[0., 0.])])
        with self.assertRaises(ValueError):
            precision_summary(rows+[dict(return_differences=[1.,2.,3.],critic_predictions=[0.,0.])])
        with self.assertRaises(ValueError):
            precision_summary([dict(return_differences=[float('nan'),1.],critic_predictions=[0.,0.])])

    def test_explicit_diagnostic_checkpoint_is_bound_without_arm_or_cli_step(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory);checkpoint=study/'training/seed-42/full/probe-250000.pth'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'fixed probe')
            path=study/'reviews/development-mechanism/seed-42/completed.json'
            scheduler.write_json(path,dict(complete=True,input=dict(seed=42,checkpoint=str(checkpoint),
                checkpoint_sha256=scheduler.digest(checkpoint))))
            parent=dict(name='diagnostic',kind='cpu',completion=str(path),command=['python'])
            job=dict(name='review',kind='review',dependencies=['diagnostic'],policy={})
            evidence=scheduler.build_signal_first_evidence(job,study,{'diagnostic':parent})
            self.assertIn(str(checkpoint.relative_to(study)),evidence['artifact_inputs'])
            checkpoint.write_bytes(b'changed probe')
            with self.assertRaises(ValueError):
                scheduler.build_signal_first_evidence(job,study,{'diagnostic':parent})

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

    def test_core_evidence_survives_adding_ablations_but_not_editing_cached_results(self):
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            data=study/'reviews/data-joint-42'
            artifacts={}
            for arm in (*CORE_ARMS,'initial_cbf'):
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
            for arm in (*CORE_ARMS,'initial_cbf'):
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
