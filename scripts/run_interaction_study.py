"""Fixed, resumable IA-CRRL job graph; waits for pre-existing server work."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
import yaml

SEEDS = (42, 101, 202, 303, 404)
ARMS = ('full', 'same_info', 'arboids_cbf', 'short', 'model_value')
CORE_ARMS = ('full', 'same_info', 'model_value')
PILOT_SEEDS = SEEDS[:2]
VRX_ARMS = ('full', 'same_info', 'arboids_cbf')
BLOCKING_PROJECTS = ('/root/arboids-formal-20261005', '/root/autodl-tmp/channel-evidence-20261005')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    temporary.replace(path)


def verified_existing_pretrain(seed):
    if seed != 42:
        return None
    directory=ROOT/'train/experiments/paper-parameters-seed42-20260921'
    try:
        manifest=json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
        expected=yaml.safe_load((ROOT/'train/configs/paper-parameters.yaml').read_text(encoding='utf-8'))
        config=yaml.safe_load((directory/'config.yaml').read_text(encoding='utf-8'))
        with (directory/'metrics1.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
        actor=directory/'adares1.pth'
        if (manifest['seed']==seed and manifest['protocol']=='paper-parameters-v1'
                and manifest.get('fresh_initialization') is True and manifest.get('config')==expected
                and manifest.get('remote_git_head')=='593db227ab07780430dee0d5c62d251c2633066f'
                and manifest.get('changed_source_sha256',{}).get('train/envs/TADgame.py')=='b6171d5f7a2e4a36e38299b4c00bee4e090c47876ec72ef2132ff6b442a30b34'
                and config==expected and int(rows[-1]['step'])==1000000
                and (directory/'exit_code').read_text().strip()=='0' and actor.exists()):
            if digest(actor)!='b8ec0679815ec28149ba8de1321e06f420a8df00bd74e22a1abfff0fb84fe03b':
                return None
            return dict(path=str(actor),sha256=digest(actor),seed=seed,steps=1000000,
                        provenance=str(directory/'run_manifest.json'))
    except (FileNotFoundError,KeyError,ValueError,IndexError):
        pass
    return None


def build_jobs(study, python, vrx_activate):
    study=Path(study)
    jobs, pretrains, inherited = [], {}, {}

    def add(name, kind, command, dependencies=(), **extra):
        jobs.append(dict(name=name,kind=kind,command=list(map(str,command)),dependencies=list(dependencies),**extra))

    for seed in SEEDS:
        reused=verified_existing_pretrain(seed)
        pre_id=f'pretrain-{seed}'
        if reused:
            pretrains[seed]=reused['path']; inherited[pre_id]=reused
        else:
            directory=study/'pretrain'/f'seed-{seed}'
            pretrains[seed]=str(directory/'actor.pth')
            add(pre_id,'gpu',[python,'-X','utf8','train/formal_study_training.py','--arm','pretrain',
                '--seed',seed,'--scene-seed',100000000+seed*10000,'--steps',1000000,
                '--output',directory,'--device','cuda:0','--exact-steps'])
        for arm in ARMS:
            train_id=f'train-{seed}-{arm}'
            directory=study/'training'/f'seed-{seed}'/arm
            add(train_id,'gpu',[python,'-X','utf8','train/train_interaction.py','--output',directory,
                '--arm',arm,'--seed',seed,'--pretrain',pretrains[seed],'--device','cuda:0'],[pre_id])
            add(f'eval-{seed}-{arm}','cpu',[python,'-X','utf8','train/interaction_evaluation.py','task',
                '--checkpoint',directory/'policy.pth','--output',study/'evaluation'/f'seed-{seed}'/arm,
                '--seed',seed,'--arm',arm],[train_id])
            if arm!='arboids_cbf':
                add(f'calibration-{seed}-{arm}','cpu',[python,'-X','utf8','train/interaction_evaluation.py','calibration',
                    '--checkpoint',directory/'resume.pth','--pretrain',pretrains[seed],
                    '--output',study/'calibration'/f'seed-{seed}'/arm,'--seed',seed],[train_id])
        for reference in ('boids','original','long_reference'):
            add(f'eval-{seed}-{reference}','cpu',[python,'-X','utf8','train/interaction_evaluation.py','task',
                '--checkpoint',pretrains[seed],'--output',study/'evaluation'/f'seed-{seed}'/reference,
                '--seed',seed,'--arm',reference],[pre_id])
    # GPU priorities finish the primary matrix before alternating-opponent extensions.
    for seed in SEEDS:
        for arm in VRX_ARMS:
            directory=study/'training'/f'seed-{seed}'/arm
            for stage in range(1,6):
                dependency=f'train-{seed}-{arm}' if stage==1 else f'adv-{seed}-{arm}-{stage-1}'
                add(f'adv-{seed}-{arm}-{stage}','gpu',[python,'-X','utf8','train/interaction_adversarial.py','phase',
                    '--study',study,'--arm',arm,'--seed',seed,'--stage',stage,'--device','cuda:0'],[dependency])
            add(f'adv-eval-{seed}-{arm}','cpu',[python,'-X','utf8','train/interaction_adversarial.py','cross-evaluate',
                '--study',study,'--arm',arm,'--seed',seed],[f'adv-{seed}-{arm}-5',f'adv-{seed}-arboids_cbf-5'])
            for n in (3,6):
                for setting in (0,1):
                    for trial in range(20):
                        name=f'vrx-{seed}-{arm}-n{n}-s{setting}-{trial:02d}'
                        scene=410000000+SEEDS.index(seed)*10000+n*100+setting*1000+trial
                        command=['python','-X','utf8','vrx/run_experiment.py','--checkpoint',directory/'policy.pth',
                            '--controller','IACRRL','--num-robots',n+1,'--setting',setting,'--seed',scene,
                            '--agility','2.25','--duration','60','--termination-rule','paper','--headless',
                            '--assets-dir',ROOT/'.vrx-assets','--output-dir',study/'vrx','--run-id',name]
                        if trial==0: command.append('--capture-frames')
                        shell='source '+shlex.quote(str(vrx_activate))+'\nexport ROS_DOMAIN_ID=121\nexec '+shlex.join(list(map(str,command)))
                        add(name,'vrx',['bash','-lc',shell],[f'train-{seed}-{arm}'],result=str(study/'vrx'/name/'result.json'))
    prerequisites=[j['name'] for j in jobs]
    add('runtime','exclusive',[python,'-X','utf8','train/interaction_evaluation.py','runtime','--study',
        study,'--output',study/'runtime'],prerequisites)
    add('analysis','cpu',[python,'-X','utf8','train/interaction_evaluation.py','summarize','--output',study],['runtime'])
    add('figures','cpu',[python,'-X','utf8','scripts/build_interaction_figures.py','--study',study],['analysis'])
    return staged_jobs(jobs, study, python),inherited


def staged_jobs(original, study, python):
    """Establish a bounded two-seed evidence chain before buying the full matrix."""
    jobs = json.loads(json.dumps(original))
    added = []

    def diagnostics(stage, seed, scope='full'):
        arms = CORE_ARMS if scope == 'core' else ARMS
        name = f'diagnostics-{scope}-{stage}-{seed}'
        completion = 'core-completed.json' if scope == 'core' else 'completed.json'
        added.append(dict(name=name, kind='cpu', phase=f'{scope}-{stage}',
            command=list(map(str, [python, '-X', 'utf8', 'train/interaction_review.py',
                '--study', study, '--stage', stage, '--seed', seed, '--scope', scope])),
            completion=str(Path(study)/f'reviews/data-{stage}-{seed}'/completion),
            dependencies=[f'{"gate" if stage == "gate" else "train"}-{seed}-{arm}' for arm in arms]))
        return name

    def review(name, stage, seeds, scope, dependencies, allowed, budget, criteria):
        added.append(dict(name=name, kind='review', command=[], stage=stage,
            seeds=list(seeds), scope=scope, dependencies=dependencies, phase=name,
            policy=dict(allowed_assessments=allowed, next_stage_budget=budget, criteria=criteria,
                on_hold='Stop expansion; diagnose with development data. No automatic retuning or final-test reuse.')))

    for job in jobs:
        name = job['name']
        if name.startswith('train-'):
            _, seed, arm = name.split('-')
            seed = int(seed)
            entry = (None if seed == SEEDS[0] and arm in CORE_ARMS else
                     'review-core-pilot' if seed == SEEDS[1] and arm in CORE_ARMS else
                     'review-core-replication' if seed in PILOT_SEEDS else 'review-deployment')
            gate = dict(job, name=f'gate-{seed}-{arm}',
                        command=job['command'] + ['--stop-after-stage', 'gate'],
                        dependencies=list(job['dependencies']), phase=entry or 'core-gate')
            if entry: gate['dependencies'].append(entry)
            added.append(gate)
            job['dependencies'] = [gate['name']]
            job['phase'] = entry or 'core-joint'
            if seed == SEEDS[0] and arm in CORE_ARMS:
                job['dependencies'].append('review-core-gate')
        elif name.startswith('pretrain-'):
            seed = int(name.split('-')[1])
            if seed != SEEDS[0]:
                job['dependencies'].append('review-core-pilot' if seed == SEEDS[1] else 'review-deployment')
        elif name.startswith(('eval-', 'calibration-', 'vrx-')):
            job['dependencies'].append('review-submission')
            job['phase'] = 'submission'
        elif name.startswith('adv-'):
            job['dependencies'].append('review-extensions')
            job.update(phase='optional-extension', optional=True)

    gate = diagnostics('gate', SEEDS[0], 'core')
    core = [diagnostics('joint', seed, 'core') for seed in PILOT_SEEDS]
    full = [diagnostics('joint', seed) for seed in SEEDS]
    # Reuse the immutable per-arm validation results when adding the two ablations.
    for job in added:
        if job['name'] in full[:2]: job['dependencies'].append('review-core-replication')
    review('review-core-gate', 'gate', SEEDS[:1], 'core', [gate], ['technical_ready'],
        dict(additional_training_steps=3000000),
        ['Finite parameters and unchanged frozen proposal branches; inspect gate saturation.',
         'A weak gate-only score does not reject joint learning; spend only this one-seed joint budget.'])
    review('review-core-pilot', 'joint', SEEDS[:1], 'core', core[:1], ['promising', 'inconclusive'],
        dict(pretraining_steps=1000000, additional_training_steps=3750000),
        ['Compare paired capture, capped time, safety, and E_delta against both matched controls.',
         'An inconclusive first seed permits only the fixed second seed; unsupported mechanisms require a hold.'])
    review('review-core-replication', 'joint', PILOT_SEEDS, 'core', core, ['promising'],
        dict(additional_training_steps=5000000),
        ['Inspect both seed-level contrasts; a pooled favorable mean cannot hide a reversed replication.',
         'Require task benefit and conditional-value evidence beyond ordinary value supervision, without a material safety loss.',
         'Inconclusive or unsupported evidence stops expansion; no remaining three seeds are released.'])
    review('review-evidence', 'joint', PILOT_SEEDS, 'full', full[:2], ['promising'],
        dict(vrx_validation_episodes=60, seconds_per_episode=60),
        ['Use short-horizon and ARBoids+CBF results to assess which central claims the two-seed evidence supports.',
         'Check consistency of mechanism and task outcomes across all five methods; retain every valid failure.'])

    # A small deployment validation bank is independent of all 1,200 final VRX trials.
    pilots = []
    for job in jobs:
        if not job['name'].startswith('vrx-'): continue
        _, seed, arm, n, setting, trial = job['name'].split('-')
        if int(seed) not in PILOT_SEEDS or (n, setting) not in (('n3', 's0'), ('n6', 's1')) or int(trial) >= 5:
            continue
        pilot = json.loads(json.dumps(job))
        pilot['name'] = job['name'].replace('vrx-', 'vrx-pilot-', 1)
        shell = pilot['command'][2]
        old_scene = 410000000 + SEEDS.index(int(seed))*10000 + int(n[1:])*100 + int(setting[1:])*1000 + int(trial)
        shell = shell.replace(f'--seed {old_scene}', f'--seed {old_scene + 20000000}')
        shell = shell.replace(str(Path(study)/'vrx'), str(Path(study)/'vrx-validation'))
        pilot['command'][2] = shell.replace(job['name'], pilot['name'])
        pilot.update(result=str(Path(study)/'vrx-validation'/pilot['name']/'result.json'),
                     dependencies=['review-evidence'], phase='deployment-validation',
                     validation=dict(seed=int(seed), arm=arm, defenders=int(n[1:]), setting=int(setting[1:]),
                                     trial=int(trial), scene_seed=old_scene + 20000000))
        added.append(pilot)
        pilots.append(pilot['name'])
    review('review-deployment', 'deployment', PILOT_SEEDS, 'full', pilots, ['promising'],
        dict(pretraining_steps=3000000, additional_training_steps=18750000),
        ['All 60 prescribed validation episodes must be valid; poor task outcomes remain in the evidence.',
         'Inspect paired deployment outcomes in both representative conditions before expanding to all five seeds.'])
    review('review-submission', 'joint', SEEDS, 'full', full + ['review-deployment'], ['promising'],
        dict(numerical_test_episodes=80000, calibration_states=1280, calibration_repetitions=4,
             vrx_test_episodes=1200),
        ['Inspect all five seeds and necessary ablations before opening the independent final tests.',
         'Final-test results are confirmatory evidence, not data for method selection or retuning.'])

    # Primary submission outputs must finish without any alternating-opponent training.
    primary = [j['name'] for j in jobs if j['name'].startswith(('eval-', 'calibration-', 'vrx-'))]
    runtime = next(j for j in jobs if j['name'] == 'runtime')
    runtime['dependencies'] = primary
    figures = next(j for j in jobs if j['name'] == 'figures')
    extension_figures = json.loads(json.dumps(figures))
    extension_figures.update(name='figures-extensions', optional=True, phase='optional-extension',
        dependencies=['figures'] + [j['name'] for j in jobs if j['name'].startswith('adv-')])
    figures['command'].append('--without-extensions')
    added.append(extension_figures)
    review('review-extensions', 'joint', SEEDS, 'full', ['figures'], ['necessary'],
        dict(additional_training_steps=37500000, common_opponent_test_episodes=13500),
        ['Primary submission evidence is already exported. Continue only for a specific necessary robustness claim.',
         'State the unresolved question and why existing evidence cannot answer it; this extension is not automatic.'])
    added[-1]['optional'] = True

    def priority(job):
        name = job['name']
        if job.get('optional'): return 20
        if name.startswith('review-'): return 0
        if name.startswith('diagnostics-'): return 1
        if name.startswith('gate-'): return 2
        if name.startswith('train-'): return 3
        if name.startswith('pretrain-'): return 4
        return 10
    return sorted(jobs + added, key=priority)


def source_hashes():
    paths=[]
    for directory in ('train','scripts','vrx'):
        paths.extend(p for p in (ROOT/directory).glob('*.py'))
    for directory in ('train/policy','train/envs','train/utils','train/configs'):
        paths.extend(p for p in (ROOT/directory).iterdir() if p.suffix in ('.py','.yaml'))
    paths.extend((ROOT/'third_party/arboids_release').rglob('*.py'))
    return {str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sorted(set(paths))}


def prepare(study, python, vrx_activate):
    study=Path(study)
    manifest=study/'manifest.json'
    if manifest.exists(): raise RuntimeError('The frozen study already exists; use run to resume it.')
    jobs, inherited=build_jobs(study,python,vrx_activate)
    write_json(manifest,dict(format='interaction-study-v3',root=str(ROOT),seeds=SEEDS,protocol='paper-parameters-v1',
        code=source_hashes(),jobs=jobs,inherited_pretraining=inherited,
        limits=dict(gpu=1,cpu=2,vrx=1,rollout_workers=4),
        cpu_affinity=sorted(os.sched_getaffinity(0))[:20] if hasattr(os,'sched_getaffinity') else [],
        wait_for_projects=BLOCKING_PROJECTS))
    write_json(study/'status.json',dict(state='prepared',jobs=len(jobs),completed=0,active=[]))


def blocking_processes():
    result=[]
    if not Path('/proc').exists():
        raise RuntimeError('The production scheduler runs on the authorized Linux server.')
    for path in Path('/proc').glob('[0-9]*'):
        try:
            cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=str((path/'cwd').resolve())
            if any(marker in cmd or cwd.startswith(marker) for marker in BLOCKING_PROJECTS):
                result.append(dict(pid=int(path.name),command=cmd[:500]))
        except (OSError,PermissionError): pass
    return result


def valid_completion(job, study):
    name=job['name']
    try:
        if job['kind']=='review': return review_passed(job, study)
        if name.startswith('diagnostics-'):
            result=json.loads(Path(job['completion']).read_text())
            return bool(result.get('complete'))
        if name.startswith('vrx-'):
            result=json.loads(Path(job['result']).read_text())
            return bool(result.get('passed')) and result.get('outcome_code') in (1,2,3,4)
        if name.startswith('adv-eval-'):
            _,_,seed,arm=name.split('-')
            with (study/f'adversarial/seed-{seed}/{arm}/common-opponents.csv').open() as f:
                return len(list(csv.DictReader(f)))==900
        if name.startswith('adv-'):
            _,seed,arm,phase=name.split('-')
            result=json.loads((study/f'adversarial/seed-{seed}/{arm}/phase-{phase}/progress.json').read_text())
            return bool(result.get('complete')) and result['step']==500000
        if name=='analysis': return json.loads((study/'analysis.json').read_text())['complete']
        if name in ('figures', 'figures-extensions'):
            directory=ROOT/'paper/figures/interaction'
            summary=json.loads((directory/'tables.json').read_text())
            return (summary.get('numerical',{}).get('complete') and
                    all((directory/f'fig{i}-{label}.pdf').exists() for i,label in
                        ((4,'vrx'),(5,'learning'),(6,'generalization'))) and
                    (name=='figures' or bool(summary.get('extensions_complete')) and
                     (directory/'fig7-alternating.pdf').exists()))
        output=Path(job['command'][job['command'].index('--output')+1])
        if name.startswith('gate-'):
            result=json.loads((output/'progress.json').read_text())
            return result['step']>=250000 and (output/'gate-endpoint.pth').exists()
        if name.startswith('train-'):
            result=json.loads((output/'progress.json').read_text())
            return bool(result.get('complete')) and result['step']==1250000
        result=json.loads((output/'completed.json').read_text())
        if name.startswith('pretrain-'): return result['completed'] and result['actual_phase_steps']==1000000
        if name.startswith('eval-'): return result['complete'] and result['episodes_per_cell']==200 and len(result['cells'])==10
        if name.startswith('calibration-'): return result['complete'] and result['states']==64 and result['repetitions']==4
        return bool(result.get('complete'))
    except (OSError,ValueError,KeyError,IndexError): return False


def evidence_inputs_match(job, study, evidence):
    if not evidence.get('technical_checks_passed'): return False
    if job.get('policy') != evidence.get('policy'): return False
    for key, expected in evidence['inputs'].items():
        relative=evidence.get('input_paths',{}).get(key,f'reviews/data-{job["stage"]}-{key}/completed.json')
        if digest(Path(study)/relative)!=expected: return False
    for relative, expected in evidence.get('artifact_inputs',{}).items():
        if digest(Path(study)/relative)!=expected: return False
    return True


def review_passed(job, study):
    directory=Path(study)/'reviews'/job['name']
    try:
        evidence=json.loads((directory/'evidence.json').read_text())
        decision=json.loads((directory/'decision.json').read_text())
        if decision['evidence_sha256']!=digest(directory/'evidence.json'): return False
        if not evidence_inputs_match(job,study,evidence): return False
        if job.get('policy') and decision.get('assessment') not in job['policy']['allowed_assessments']:
            return False
        return decision.get('decision')=='continue' and bool(decision.get('rationale','').strip())
    except (OSError,ValueError,KeyError): return False


def decide_review(study, name, decision, rationale, evidence_sha256, assessment=None):
    study=Path(study)
    manifest=json.loads((study/'manifest.json').read_text())
    job=next(j for j in manifest['jobs'] if j['name']==name and j['kind']=='review')
    directory=study/'reviews'/name
    path=directory/'evidence.json'
    if digest(path)!=evidence_sha256:
        raise ValueError('Review evidence changed since it was inspected.')
    evidence=json.loads(path.read_text())
    if decision=='continue':
        if not evidence_inputs_match(job,study,evidence):
            raise ValueError('Resolve failed or changed evidence before continuing.')
        if job.get('policy') and assessment not in job['policy']['allowed_assessments']:
            raise ValueError('This scientific assessment cannot release the next budget at this review.')
    if not rationale or len(rationale.strip())<40:
        raise ValueError('Record a substantive evidence-based reason, including limitations.')
    previous=[]
    if (directory/'decision.json').exists():
        old=json.loads((directory/'decision.json').read_text())
        previous=old.pop('history',[])+[old]
    write_json(directory/'decision.json',dict(decision=decision,assessment=assessment,rationale=rationale.strip(),
        evidence_sha256=evidence_sha256,time=time.time(),history=previous))


def migrate_stages(study):
    """Execution-only migration; retain outcomes, checkpoints, and previous hashes."""
    import fcntl
    study=Path(study)
    lock=(study/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=study/'manifest.json'
    manifest=json.loads(path.read_text())
    if manifest['format'] not in ('interaction-study-v1','interaction-study-v2'):
        raise ValueError('Only the original or five-arm staged study needs this migration.')
    records=[]
    for record_path in (study/'jobs').glob('*.json'):
        record=json.loads(record_path.read_text())
        if record.get('state')=='running' and Path(f'/proc/{record.get("pid",0)}').exists():
            raise RuntimeError('Stop the owned study jobs at recoverable checkpoints before migration.')
        if record['name'].startswith(('review-', 'diagnostics-')):
            raise RuntimeError('Existing stage evidence needs an explicit migration; it must not be silently replaced.')
        records.append((record_path,record))
    for progress in (study/'training').glob('seed-*/*/progress.json'):
        if json.loads(progress.read_text())['step']>250000:
            raise RuntimeError('Joint training already began; gate endpoints need an explicit recovery decision.')
    current=source_hashes()
    changed={p:dict(before=h,after=current.get(p)) for p,h in manifest['code'].items() if current.get(p)!=h}
    allowed={'scripts/run_interaction_study.py','scripts/build_interaction_figures.py','train/interaction_review.py'}
    if manifest['format']=='interaction-study-v1': allowed.add('train/train_interaction.py')
    if set(changed)-allowed or set(current)-set(manifest['code'])-{'train/interaction_review.py'}:
        raise RuntimeError('Unreviewed source changes would alter the frozen study.')
    for p in set(current)-set(manifest['code']): changed[p]=dict(before=None,after=current[p])
    first=next(j for j in manifest['jobs'] if j['name'].startswith('train-'))
    original=[]
    for job in manifest['jobs']:
        if job['name'].startswith(('gate-', 'diagnostics-', 'review-')): continue
        job=dict(job,dependencies=[d for d in job['dependencies'] if not d.startswith(('gate-', 'diagnostics-', 'review-'))])
        if job['name'].startswith('train-'):
            job['dependencies']=[f'pretrain-{job["name"].split("-")[1]}']
        original.append(job)
    manifest['jobs']=staged_jobs(original,study,first['command'][0])
    manifest.setdefault('execution_revisions',[]).append(dict(time=time.time(),prior_manifest_sha256=digest(path),changes=changed,
        reason='User authorized a minimum necessary evidence chain before submission expansion; methods and training budgets unchanged.',
        preserved_job_records=[r['name'] for _,r in records]))
    manifest.update(format='interaction-study-v3',code=current)
    write_json(path,manifest)
    for record_path, record in records:
        if record.get('state')=='running':
            record.update(state='interrupted',interrupted_at=time.time(),reason='checkpointed stage-scheduling migration')
            write_json(record_path,record)
    write_json(study/'status.json',dict(state='prepared',jobs=len(manifest['jobs']),
        completed=sum(r.get('state')=='completed' for _,r in records),active=[]))
    lock.close()


def run(study):
    import fcntl
    study=Path(study)
    manifest=json.loads((study/'manifest.json').read_text(encoding='utf-8'))
    if manifest['root']!=str(ROOT): raise RuntimeError('Run the manifest in its original project path.')
    lock=(study/'runner.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('This study already has a live scheduler.')
    for path,expected in manifest['code'].items():
        if digest(ROOT/path)!=expected: raise RuntimeError('Frozen scientific code changed: '+path)
    for item in manifest['inherited_pretraining'].values():
        if digest(item['path'])!=item['sha256']: raise RuntimeError('A common pretraining artifact changed.')
    jobs={j['name']:j for j in manifest['jobs']}
    done=set(manifest['inherited_pretraining'])
    (study/'jobs').mkdir(exist_ok=True)
    for path in (study/'jobs').glob('*.json'):
        item=json.loads(path.read_text())
        if item.get('state')=='completed':
            if jobs[item['name']]['kind']=='review' and not review_passed(jobs[item['name']],study):
                raise RuntimeError('A completed stage review no longer matches its evidence.')
            done.add(item['name'])
        elif item.get('state')=='running':
            pid=item.get('pid',0)
            if pid and Path(f'/proc/{pid}').exists():
                raise RuntimeError(f'Existing study child {pid} is alive; do not duplicate it.')
            if valid_completion(jobs[item['name']],study):
                done.add(item['name'])
                write_json(path,dict(item,state='completed',recovered=True))
    active={}
    logs=study/'logs'; logs.mkdir(exist_ok=True)
    external_clear=False
    failed=False
    pending_reviews=[]

    def status(state, blockers=None):
        write_json(study/'status.json',dict(state=state,runner_pid=os.getpid(),updated_at=time.time(),
            jobs=len(jobs),completed=len(done-set(manifest['inherited_pretraining'])),
            active=[dict(name=k,pid=v[0].pid,kind=jobs[k]['kind']) for k,v in active.items()],
            pending_reviews=pending_reviews,blockers=blockers or [],
            core_evidence_complete='review-evidence' in done,
            submission_evidence_complete='figures' in done,
            optional_jobs=sum(bool(j.get('optional')) for j in jobs.values())))

    try:
        while len(done-set(manifest['inherited_pretraining'])) < len(jobs):
            blockers=blocking_processes()
            if blockers and not active:
                status('waiting_existing_workload',blockers)
                time.sleep(30)
                continue
            if not external_clear:
                required=2 if any((study/'jobs').glob('*.json')) else 32
                if shutil.disk_usage(study).free < required*1024**3:
                    raise RuntimeError(f'At least {required} GiB free is required before scheduling.')
                external_clear=True
            for name,details in list(active.items()):
                process,log=details[:2]
                code=process.poll()
                if code is None: continue
                log.close(); del active[name]
                valid=code==0 and valid_completion(jobs[name],study)
                record=json.loads((study/'jobs'/f'{name}.json').read_text())
                record.update(returncode=code,state='completed' if valid else 'failed',finished_at=time.time())
                record['wall_seconds']=record['finished_at']-record['started_at']
                write_json(study/'jobs'/f'{name}.json',record)
                if valid: done.add(name)
                else: failed=True
            if failed:
                status('failed_waiting_for_running_jobs')
                if not active: raise RuntimeError('An experiment job failed; inspect its existing job log before resuming.')
                time.sleep(5); continue
            counts={kind:sum(jobs[name]['kind']==kind for name in active) for kind in ('gpu','cpu','vrx','exclusive')}
            pending_reviews=[]
            for name,job in jobs.items():
                if name in done or name in active or not set(job['dependencies']).issubset(done): continue
                kind=job['kind']
                if blockers or counts['exclusive']: continue
                if kind=='review':
                    directory=study/'reviews'/name
                    if not (directory/'evidence.json').exists():
                        from interaction_review import review_evidence, deployment_evidence
                        evidence=(deployment_evidence(study) if job['stage']=='deployment' else
                                  review_evidence(study,job['stage'],job['seeds'],job['scope']))
                        evidence['policy']=job['policy']
                        write_json(directory/'evidence.json',evidence)
                    if review_passed(job,study):
                        done.add(name)
                        write_json(study/'jobs'/f'{name}.json',dict(name=name,state='completed',finished_at=time.time(),
                            evidence_sha256=digest(directory/'evidence.json'),decision_sha256=digest(directory/'decision.json')))
                    else:
                        pending_reviews.append(name)
                        write_json(study/'jobs'/f'{name}.json',dict(name=name,state='waiting_review',
                            evidence_sha256=digest(directory/'evidence.json')))
                    continue
                if shutil.disk_usage(study).free < 2*1024**3:
                    raise RuntimeError('Less than 2 GiB free: preserve running work and resolve storage before scheduling more jobs.')
                if kind=='exclusive':
                    if active: continue
                elif counts[kind]>=manifest['limits'][kind]: continue
                log=(logs/f'{name}.log').open('a',encoding='utf-8')
                environment=os.environ.copy()
                environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',
                                   XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
                if kind!='gpu': environment['CUDA_VISIBLE_DEVICES']=''
                if kind=='vrx': environment['ROS_DOMAIN_ID']='121'
                cpus=manifest.get('cpu_affinity',[])
                if kind=='gpu': assigned=cpus[:6]
                elif kind=='cpu':
                    occupied={v[2] for v in active.values() if len(v)>2}
                    slot=next((i for i in (0,1) if i not in occupied),0)
                    assigned=cpus[6+3*slot:9+3*slot]
                elif kind=='vrx': assigned=cpus[12:20]
                else: assigned=cpus
                process=subprocess.Popen(job['command'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,preexec_fn=(lambda:os.sched_setaffinity(0,assigned)) if assigned else None)
                active[name]=(process,log,slot if kind=='cpu' else -1); counts[kind]+=1
                write_json(study/'jobs'/f'{name}.json',dict(name=name,pid=process.pid,state='running',started_at=time.time()))
                if kind=='exclusive': break
            status('running' if active else 'waiting_review' if pending_reviews else 'waiting_dependencies')
            if not active and not blockers and not any(name not in done and set(j['dependencies']).issubset(done) for name,j in jobs.items()):
                if len(done-set(manifest['inherited_pretraining'])) != len(jobs):
                    raise RuntimeError('Unsatisfied dependency graph.')
            time.sleep(5)
        status('complete')
    except BaseException as exc:
        # Do not terminate jobs implicitly. A subsequent run detects surviving children.
        status('interrupted' if isinstance(exc,KeyboardInterrupt) else 'error')
        write_json(study/'error.json',dict(error=str(exc),time=time.time()))
        raise
    finally:
        for value in active.values(): value[1].close()
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','run','status','migrate-stages','migrate-evidence','decide-review'])
    parser.add_argument('--study',type=Path,default=ROOT/'train/experiments/ia-crrl-20261007')
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--vrx-activate',type=Path,default=ROOT/'vrx_ws/activate.bash')
    parser.add_argument('--review')
    parser.add_argument('--decision',choices=['continue','hold'])
    parser.add_argument('--rationale')
    parser.add_argument('--evidence-sha256')
    parser.add_argument('--assessment',choices=['technical_ready','promising','inconclusive','unsupported','necessary'])
    args=parser.parse_args()
    if args.mode=='prepare': prepare(args.study.resolve(),args.python,args.vrx_activate.resolve())
    elif args.mode=='run': run(args.study.resolve())
    elif args.mode in ('migrate-stages','migrate-evidence'): migrate_stages(args.study.resolve())
    elif args.mode=='decide-review':
        decide_review(args.study.resolve(),args.review,args.decision,args.rationale,args.evidence_sha256,args.assessment)
    else: print((args.study/'status.json').read_text(encoding='utf-8'))


if __name__=='__main__': main()
