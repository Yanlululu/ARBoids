"""Fixed, resumable IA-CRRL job graph; waits for pre-existing server work."""
import argparse
import csv
import hashlib
import json
import math
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
PEER_ARMS = ('full', 'no_peer')
PILOT_SEEDS = SEEDS[:2]
VRX_ARMS = ('full', 'same_info', 'arboids_cbf')
FORMAL_SEEDS = (202, 303, 404, 505, 606)
DEVELOPMENT_ARMS = ('full', 'same_info', 'no_peer', 'short', 'model_value')
FORMAL_ARMS = (*ARMS, 'no_peer')
BLOCKING_PROJECTS = ('/root/arboids-formal-20261005', '/root/autodl-tmp/channel-evidence-20261005')


def technical_review_policy():
    """Operational bounds for completing the existing pair, never an efficacy test."""
    return dict(version='bounded-pair-technical-v1',
        reviews=['review-pair-50000', 'review-pair-100000'],
        maximum_endpoint_step=250000, maximum_environment_error_to_zero=1.25,
        maximum_collision_fraction=0.,
        interpretation='Finite, calibrated development only; no automatic promising or frozen assessment.')


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


def build_legacy_jobs(study, python, vrx_activate, *, raw=False, seeds=SEEDS, arms=ARMS):
    study=Path(study)
    jobs, pretrains, inherited = [], {}, {}

    def add(name, kind, command, dependencies=(), **extra):
        jobs.append(dict(name=name,kind=kind,command=list(map(str,command)),dependencies=list(dependencies),**extra))

    for seed in seeds:
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
        for arm in arms:
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
    for seed in seeds:
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
                        scene=(540000000 if raw else 410000000)+seeds.index(seed)*10000+n*100+setting*1000+trial
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
    return (jobs if raw else staged_jobs(jobs, study, python)), inherited


def protocol_specification():
    """Concrete candidate protocol; the freeze review binds this object and the source hashes."""
    config = yaml.safe_load((ROOT/'train/configs/interaction-aware-sac.yaml').read_text())
    from interaction_evaluation import formal_evidence_standard
    return dict(development_seeds=PILOT_SEEDS, formal_seeds=FORMAL_SEEDS,
        development_arms=DEVELOPMENT_ARMS, formal_arms=FORMAL_ARMS,
        development_environment_steps=250000, formal_environment_steps=1000000,
        formal_gate_steps=250000, formal_joint_steps=750000, formal_curriculum_interval=187500,
        common_pretraining_steps=1000000, configuration=config,
        pair_reference=[0., .5, 1.], long_horizon_steps=100, short_horizon_steps=10,
        intervention_sampling='64 snapshots every 1000 real steps; reuse between refreshes; no outcome filtering',
        auxiliary_pairs_per_update=64, real_replay_batch=4096, auxiliary_to_real_batch_ratio=64/4096,
        no_peer='Mask only teammate learned/Boids candidate fields; preserve teammate state, joint critic, labels and CBF.',
        same_info='Remove only model-based auxiliary supervision.',
        model_value='Same paired rollout calls as Full; regress ordinary returns instead of their difference.',
        bootstrap='A separate twin evaluator learns only from the same real replay; its Polyak target supplies both real TD and rollout tails. Auxiliary gradients never update this evaluator.',
        bootstrap_updates='One extra twin-evaluator optimizer step per SAC update for all five matched arms; charged separately.',
        endpoint_selection='Fixed endpoints only, never best test checkpoint.',
        development_banks=[390000000, 392000000, 393000000, 394000000, 395000000,
                           396000000,397000000,400000000,402000000], confirmation_banks=[398000000, 399000000],
        formal_banks=dict(task=510000000, calibration=520000000, mechanism=530000000,
                          vrx=540000000, primary=560000000),
        cost_counters=['environment_steps', 'model_steps', 'gradient_updates', 'wall_seconds'],
        formal_evidence=formal_evidence_standard(),
        decision_rule='Primary task effect is Full minus ARBoids+CBF capped capture time; report CR/SR/collision jointly. Assess A--D paired evidence and task relevance without a percentage or unanimous-seed threshold.')


def build_jobs(study, python, vrx_activate, resume_steps=None):
    """Signal -> minimum pair -> replicated five-arm development -> freeze -> formal evidence."""
    study, resume_steps = Path(study), resume_steps or {}
    formal, inherited = build_legacy_jobs(study, python, vrx_activate, raw=True,
                                         seeds=FORMAL_SEEDS, arms=FORMAL_ARMS)
    jobs, pretrains = [], {}
    def add(name, kind, command=(), dependencies=(), **extra):
        jobs.append(dict(name=name, kind=kind, command=list(map(str, command)),
                         dependencies=list(dependencies), **extra))
        return name
    def review(name, dependencies, budget, criteria, assessment='technical_ready', **extra):
        return add(name, 'review', dependencies=dependencies, stage='development-v6',
            policy=dict(allowed_assessments=[assessment], next_stage_budget=budget,
                criteria=criteria, on_hold='Pause expansion; diagnose only with development data.'), **extra)
    def screen(seed, arm, step, dependencies, fresh=False):
        name=f'diagnostics-{"confirmation" if fresh else "screen"}-{seed}-{arm}-{step}'
        command=[python,'-X','utf8','train/interaction_review.py','--mode','screen','--study',study,
                 '--seed',seed,'--arm',arm,'--checkpoint-step',step,'--workers',2]
        if fresh: command.append('--fresh')
        return add(name,'cpu',command,dependencies,
            completion=str(study/'reviews'/('confirmation' if fresh else 'screens')/f'{seed}-{arm}-{step}/completed.json'),
            phase='confirmation' if fresh else 'development', seed=seed, arm=arm, checkpoint_step=step)
    def train(seed, arm, boundary, dependencies):
        command=[python,'-X','utf8','train/train_interaction.py','--output',study/f'training/seed-{seed}/{arm}',
             '--arm',arm,'--seed',seed,'--pretrain',pretrains[seed],'--device','cuda:0','--stop-at-step',boundary]
        prior=study/f'training/seed-{seed}/{arm}/config.yaml'
        if prior.exists() and yaml.safe_load(prior.read_text())['interaction'].get('bootstrap_source','coupled')=='coupled':
            command.append('--recondition-bootstrap-if-needed')
        return add(f'develop-{seed}-{arm}-{boundary}', 'gpu',
            command,
            dependencies, phase='development', endpoint_step=boundary)
    for seed in PILOT_SEEDS:
        reused = verified_existing_pretrain(seed)
        if reused:
            inherited[f'pretrain-{seed}']=reused
            pretrains[seed]=reused['path']
        else:
            pretrains[seed]=study/f'pretrain/seed-{seed}/actor.pth'
            add(f'pretrain-{seed}','gpu',[python,'-X','utf8','train/formal_study_training.py',
                '--arm','pretrain','--seed',seed,'--scene-seed',100000000+seed*10000,'--steps',1000000,
                '--output',study/f'pretrain/seed-{seed}','--device','cuda:0','--exact-steps'],
                [] if seed==42 else ['review-pair-250000'], phase='development')
    signal_job=add('diagnostics-signal','exclusive',
        [python,'-X','utf8','train/interaction_review.py','--study',study,'--mode','signal',
         '--seed',42,'--workers',4], ['pretrain-42'], completion=str(study/'reviews/signal/completed.json'))
    previous=review('review-signal',[signal_job],dict(maximum_new_environment_steps=10000),
        ['Zero difference, swap sign, one-current-gate intervention, shared RNG and snapshot/history isolation must pass.',
         'Inspect legitimate gate consequences and executed thrust in both state strata; sensitivity is not learned utility.'])
    latest={}
    recovery=[]
    for arm in ('full','same_info'):
        if resume_steps.get(f'42/{arm}',0):
            step=resume_steps[f'42/{arm}']
            fixed=step if (study/f'training/seed-42/{arm}/probe-{step}.pth').exists() else 0
            latest[42,arm]=screen(42,arm,fixed,[previous])
        else:
            step=train(42,arm,5000,[previous,'pretrain-42'])
            latest[42,arm]=screen(42,arm,5000,[step])
        recovered=study/f'reviews/bootstrap-repair/42-{arm}/completed.json'
        if recovered.exists():
            recovery.append(add(f'diagnostics-bootstrap-repair-42-{arm}','gpu',
                [python,'-X','utf8','train/train_interaction.py','--output',study/f'training/seed-42/{arm}',
                 '--device','cuda:0','--repair-bootstrap'],[previous],completion=str(recovered),phase='development-recovery'))
    audits=[]
    if resume_steps.get('42/full',0)>=100000:
        for arm in ('full','same_info'):
            if (study/f'reviews/bootstrap-repair/42-{arm}/completed.json').exists():
                step=resume_steps[f'42/{arm}']
                audits.append(add(f'diagnostics-bootstrap-check-42-{arm}-{step}','cpu',
                    [python,'-X','utf8','train/interaction_review.py','--study',study,'--mode','bootstrap-check',
                     '--seed',42,'--arm',arm,'--checkpoint-step',step,'--workers',2],[latest[42,arm]],
                    completion=str(study/f'reviews/bootstrap-check/42-{arm}-{step}/completed.json')))
                continue
            suffix=''
            audits.append(add(f'diagnostics-target-audit-42-{arm}{suffix}','cpu',
                [python,'-X','utf8','train/interaction_review.py','--study',study,'--mode','target-audit',
                 '--seed',42,'--arm',arm,'--workers',2], [latest[42,arm]],
                completion=str(study/f'reviews/target-audit/42-{arm}{suffix}/completed.json')))
    previous=review('review-pair-connectivity-bootstrap' if recovery else 'review-pair-connectivity',
        [*latest.values(),*audits,*recovery],dict(maximum_new_environment_steps=100000),
        ['Inspect finite TD/model targets, auxiliary scale, critic and adapter gradients and frozen proposal parameters.',
         'Existing checkpoints beyond an early milestone remain development work; unequal budgets cannot establish a method contrast.'])
    recovered_pair_review=previous
    for boundary in (50000,100000,250000):
        for arm in ('full','same_info'):
            if resume_steps.get(f'42/{arm}',0)<boundary:
                # The recovered Full is already beyond the early checkpoints.
                # Its bounded gate-stage finish can run while Same-info catches up.
                prerequisite=(recovered_pair_review if recovery and arm=='full' and
                    resume_steps.get('42/full',0)>=100000 else previous)
                task=train(42,arm,boundary,[prerequisite,'pretrain-42'])
                latest[42,arm]=screen(42,arm,boundary,[task])
        dependencies=[previous,*latest.values()]
        if recovery and 100000<=resume_steps.get('42/full',0)<250000 and boundary==100000:
            # Full can finish while Same-info catches up; review its completed endpoint.
            dependencies=[('diagnostics-screen-42-full-250000' if
                name==latest[42,'full'] else name) for name in dependencies]
        previous=review(f'review-pair-{boundary}',dependencies,
            dict(maximum_new_environment_steps=300000 if boundary<250000 else 2000000),
            ['Assess heldout E_model separately from independent full-environment E_env.',
             'Use actual control consequences and task/safety trends; a 250k success-rate lead is not required.',
             'Model fit alone, or a lucky return lead without control effects, does not justify expanding.',
             'A promising mechanism with a small task gap permits the bounded next development stage; otherwise hold.'],
            assessment='promising' if boundary==250000 else 'technical_ready')
    for seed, arms in ((42,('no_peer',)),(101,('full','same_info','no_peer'))):
        for arm in arms:
            task=train(seed,arm,250000,['review-pair-250000',f'pretrain-{seed}'])
            latest[seed,arm]=screen(seed,arm,250000,[task])
    replicated=review('review-development-replication',list(latest.values()),
        dict(maximum_new_environment_steps=1000000),
        ['Compare Full/Same-info/no-peer at the same 250k budget in both development seeds.',
         'Inspect seed-level E_env, capture efficiency, safety and actual composition effects; retain valid failures.'],
        assessment='promising')
    for seed in PILOT_SEEDS:
        for arm in ('short','model_value'):
            task=train(seed,arm,250000,[replicated,f'pretrain-{seed}'])
            latest[seed,arm]=screen(seed,arm,250000,[task])
    candidate=review('review-small-matrix',list(latest.values()),dict(new_training_steps=0),
        ['Inspect all five candidate methods at 250k in both development seeds before any formal training.',
         'Short versus Full identifies the horizon contribution; ordinary return versus paired difference identifies supervision specificity.'],
        assessment='promising')
    confirmation=[screen(seed,arm,250000,[candidate],fresh=True)
                  for seed in PILOT_SEEDS for arm in DEVELOPMENT_ARMS]
    review('review-protocol-freeze',confirmation,
        dict(formal_environment_steps=5*len(FORMAL_ARMS)*1000000, common_pretraining_steps=5000000),
        ['Confirm the candidate on a new development validation bank.',
         'Freeze architecture, lambda, horizons, intervention frequency/reference sampling, curriculum, reward, observations and CBF.',
         'Freeze every control, real/model/update budget, independent formal seeds, final test banks and endpoint rule.',
         'No formal-test feedback may change this protocol; developer seeds 42/101 are excluded from confirmatory training.'],
        assessment='frozen', freezes_protocol=True)
    primary=[]
    for seed in FORMAL_SEEDS:
        for arm in FORMAL_ARMS:
            name=add(f'primary-{seed}-{arm}','cpu',[python,'-X','utf8','train/interaction_evaluation.py','task',
                '--checkpoint',study/f'training/seed-{seed}/{arm}/policy.pth',
                '--output',study/f'primary/seed-{seed}/{arm}','--seed',seed,'--arm',arm,'--primary-only'],
                [f'train-{seed}-{arm}'], phase='formal-primary')
            primary.append(name)
            if arm!='arboids_cbf': primary.append(f'calibration-{seed}-{arm}')
        name=add(f'diagnostics-mechanism-{seed}','cpu',[python,'-X','utf8','train/interaction_review.py',
            '--study',study,'--mode','mechanism','--seed',seed,'--workers',2],
            [f'train-{seed}-full'], completion=str(study/f'mechanism/seed-{seed}/completed.json'),phase='formal-mechanism')
        primary.append(name)
    evidence=add('diagnostics-formal-evidence','cpu',
        [python,'-X','utf8','train/interaction_evaluation.py','evidence','--output',study],primary,
        completion=str(study/'reviews/formal-evidence/completed.json'),phase='formal-evidence')
    review('review-formal-core',[evidence],dict(task_episodes=90000,vrx_episodes=1200),
        ['Complete all five independent seeds at the frozen one-million-step endpoint.',
         'Inspect final task, Delta-Q, mean/permutation and four-branch interventions before broad task extensions.'],
        assessment='supported',requires_claim_assessment=True)
    for job in formal:
        name=job['name']
        if name.startswith('train-'):
            job['command'].append('--formal')
            job['formal_steps']=1000000
        if name.startswith(('train-','pretrain-')):
            job['dependencies'].append('review-protocol-freeze')
            job['phase']='formal-training'
        elif name.startswith(('eval-','vrx-')):
            job['dependencies'].append('review-formal-core')
            job['phase']='task-extension'
        elif name.startswith('adv-'):
            job['dependencies'].append('review-extensions')
            job.update(optional=True,phase='optional-extension')
        elif name=='runtime':
            job['dependencies']=[n for n in job['dependencies'] if not n.startswith('adv-')]
        elif name=='figures':
            job['command'].append('--without-extensions')
        jobs.append(job)
    review('review-extensions',['figures'],dict(additional_training_steps=37500000),
        ['Proceed only for a specific unresolved learning-opponent robustness question after the main results.'],
        assessment='necessary', optional=True)
    figures=next(j for j in formal if j['name']=='figures')
    add('figures-extensions','cpu',figures['command'][:-1],
        ['figures']+[j['name'] for j in formal if j['name'].startswith('adv-')],optional=True)
    def priority(j):
        return (20 if j.get('optional') else 0 if j['kind']=='review' else
                1 if j['name'].startswith('diagnostics-') else 2 if j['phase']=='development' else 5)
    # Preserve construction order for equal priorities and avoid interpreting absent phases.
    for job in jobs: job.setdefault('phase','signal-first')
    return sorted(jobs,key=priority), inherited


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
    return sorted(advance_candidate_screen(strengthen_jobs(jobs + added, study, python), study, python), key=priority)


def strengthen_jobs(jobs, study, python):
    """A one-seed candidate ablation, with a bounded second seed only if inconclusive."""
    jobs = json.loads(json.dumps(jobs))
    graph = {j['name']: j for j in jobs}
    condition = dict(review='review-peer-pilot', assessment='inconclusive')
    for seed in PILOT_SEEDS:
        first = seed == PILOT_SEEDS[0]
        entry = 'review-core-replication' if first else 'review-peer-pilot'
        original = graph[f'train-{seed}-full']
        command = list(original['command'])
        command[command.index('--arm')+1] = 'no_peer'
        command[command.index('--output')+1] = str(Path(study)/f'training/seed-{seed}/no_peer')
        additions = [dict(name=f'gate-{seed}-no_peer', kind='gpu', phase='peer-control',
            command=command+['--stop-after-stage', 'gate'], dependencies=[f'pretrain-{seed}', entry]),
            dict(name=f'train-{seed}-no_peer', kind='gpu', phase='peer-control', command=command,
                 dependencies=[f'gate-{seed}-no_peer'])]
        for stage in ('gate', 'joint'):
            dependencies = [f'{"gate" if stage == "gate" else "train"}-{seed}-{a}' for a in PEER_ARMS]
            if stage == 'joint': dependencies.append(f'diagnostics-peer-gate-{seed}')
            additions.append(dict(name=f'diagnostics-peer-{stage}-{seed}', kind='cpu', phase='peer-control',
                command=list(map(str, [python, '-X', 'utf8', 'train/interaction_review.py', '--study', study,
                    '--stage', stage, '--seed', seed, '--scope', 'peer'])), dependencies=dependencies,
                completion=str(Path(study)/f'reviews/data-{stage}-{seed}/peer-completed.json')))
        if not first:
            for job in additions:
                job['condition'] = condition
                if entry not in job['dependencies']: job['dependencies'].append(entry)
        jobs.extend(additions)
    for name, seeds, allowed, dependencies in (
            ('review-peer-pilot', PILOT_SEEDS[:1], ['promising', 'inconclusive'], ['diagnostics-peer-joint-42']),
            ('review-peer-replication', PILOT_SEEDS, ['promising'], ['diagnostics-peer-joint-42', 'diagnostics-peer-joint-101'])):
        job = dict(name=name, kind='review', command=[], stage='joint', scope='peer', seeds=list(seeds),
            dependencies=dependencies, phase='peer-control',
            policy=dict(allowed_assessments=allowed,
                next_stage_budget=(dict(additional_training_steps_if_inconclusive=1250000) if len(seeds)==1 else {}),
                criteria=['Check paired Full/no-peer task and conditional-value evidence, plus controlled gate and thrust responses.',
                    'Response alone does not demonstrate benefit. A promising single seed is pilot evidence only.',
                    'An inconclusive first seed releases only seed 101; an inconclusive second seed or unsupported evidence holds expansion.'],
                on_hold='Stop expansion; retain the method, budgets and every valid task failure.'))
        if len(seeds)==2:
            job['condition'] = condition
            job['dependencies'].append('review-peer-pilot')
        jobs.append(job)
    graph['review-core-gate']['policy']['criteria'].append(
        'Inspect fixed-state peer-candidate gate and executed-thrust responses; flat response is diagnostic, not an automatic veto.')
    graph['review-core-replication']['policy']['next_stage_budget']['additional_training_steps'] += 1250000
    graph['review-evidence']['dependencies'].extend(['review-peer-pilot', 'review-peer-replication'])
    for name in ('review-evidence', 'review-submission', 'review-extensions'):
        graph[name]['peer_control'] = True
        graph[name]['policy']['criteria'].append(
            'Inspect the matched no-peer candidate control before expansion; state whether it has one or two training seeds.')
    return jobs


def advance_candidate_screen(jobs, study, python):
    """Put the candidate ablation in the first gate screen and evaluate endpoints as they arrive."""
    jobs = json.loads(json.dumps(jobs))
    graph = {j['name']: j for j in jobs}
    seed = SEEDS[0]
    graph[f'gate-{seed}-no_peer'].update(dependencies=[f'pretrain-{seed}'], phase='core-gate')
    graph[f'train-{seed}-no_peer'].update(dependencies=[f'gate-{seed}-no_peer', 'review-core-gate'], phase='core-joint')
    screen_arms = (*CORE_ARMS, 'no_peer')
    for arm in ('initial_cbf', *screen_arms):
        name = f'diagnostics-arm-gate-{seed}-{arm}'
        dependencies = ([f'pretrain-{seed}'] if arm == 'initial_cbf' else
                        [f'gate-{seed}-{arm}', f'diagnostics-arm-gate-{seed}-initial_cbf'])
        jobs.append(dict(name=name, kind='cpu', phase='early-screen', dependencies=dependencies,
            command=list(map(str, [python, '-X', 'utf8', 'train/interaction_review.py', '--study', study,
                '--stage', 'gate', '--seed', seed, '--scope', 'peer' if arm=='no_peer' else 'core', '--arm', arm])),
            completion=str(Path(study)/f'reviews/data-gate-{seed}/arm-{arm}-completed.json')))
    for scope, arms in (('core', CORE_ARMS), ('peer', PEER_ARMS)):
        graph[f'diagnostics-{scope}-gate-{seed}']['dependencies'].extend(
            f'diagnostics-arm-gate-{seed}-{arm}' for arm in (*arms, 'initial_cbf'))
    gate_review = graph['review-core-gate']
    gate_review['dependencies'].append(f'diagnostics-peer-gate-{seed}')
    gate_review['peer_gate_control'] = True
    gate_review['policy']['next_stage_budget']['additional_training_steps'] += 1000000
    gate_review['policy']['criteria'].append(
        'Review all four first-seed gate endpoints, including Full versus the no-peer candidate control, before joint training.')
    graph['review-core-replication']['policy']['next_stage_budget']['additional_training_steps'] -= 1250000
    return jobs


def process_identity(pid):
    """Linux process start time, excluding exited/zombie processes and PID reuse."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] in ('Z', 'X') else fields[19]
    except (OSError, IndexError):
        return None


def verify_running_job(job, record):
    pid = record['pid']
    identity = process_identity(pid)
    if identity is None:
        raise RuntimeError('The recorded study process has already exited.')
    command = Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    if ([os.fsdecode(x) for x in command] != job['command'] or
            Path(f'/proc/{pid}/cwd').resolve() != ROOT or os.getpgid(pid) != pid or
            record.get('process_identity', identity) != identity):
        raise RuntimeError('The live process does not match the owned study job.')
    return identity


class AdoptedProcess:
    """Observe an existing owned child without interrupting its learning state."""
    def __init__(self, pid, identity):
        self.pid, self.identity = pid, identity

    def poll(self):
        return None if process_identity(self.pid) == self.identity else 'unobserved'


def resource_assignment(manifest, kind, jobs, active):
    cpus = manifest.get('cpu_affinity', [])
    if kind == 'exclusive': return -1, cpus
    occupied = {v[2] for name, v in active.items() if jobs[name]['kind'] == kind}
    slot = next(i for i in range(manifest['limits'][kind]) if i not in occupied)
    gpu_width, cpu_width = manifest.get('cpu_per_gpu', 6), manifest.get('cpu_per_evaluation', 3)
    if kind == 'gpu':
        start, width = slot*gpu_width, gpu_width
    elif kind == 'cpu':
        start, width = manifest['limits']['gpu']*gpu_width + slot*cpu_width, cpu_width
    else:
        return slot, cpus[-8:]
    assigned = cpus[start:start+width]
    if cpus and len(assigned) != width:
        raise RuntimeError('The configured process slots exceed the assigned CPU budget.')
    return slot, assigned


def set_tree_affinity(pid, cpus):
    if not cpus: return
    parents = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    owned = {pid}
    while True:
        children = {child for child, parent in parents.items() if parent in owned}
        if children <= owned: break
        owned.update(children)
    for child in owned:
        try: os.sched_setaffinity(child, cpus)
        except ProcessLookupError: pass


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
    write_json(manifest,dict(format='interaction-study-v6',root=str(ROOT),seeds=FORMAL_SEEDS,protocol='paper-parameters-v1',
        development_seeds=PILOT_SEEDS, candidate_protocol=protocol_specification(),
        technical_review_automation=technical_review_policy(),
        code=source_hashes(),jobs=jobs,inherited_pretraining=inherited,
        limits=dict(gpu=2,cpu=2,vrx=1,rollout_workers=4), cpu_per_gpu=6,cpu_per_evaluation=4,
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
        if name.startswith('develop-'):
            result=json.loads((output/'progress.json').read_text())
            return result['step']>=job['endpoint_step'] and (output/f'probe-{job["endpoint_step"]}.pth').exists()
        if name.startswith('gate-'):
            result=json.loads((output/'progress.json').read_text())
            return result['step']>=250000 and (output/'gate-endpoint.pth').exists()
        if name.startswith('train-'):
            result=json.loads((output/'progress.json').read_text())
            return bool(result.get('complete')) and result['step']==job.get('formal_steps',1250000)
        result=json.loads((output/'completed.json').read_text())
        if name.startswith('pretrain-'): return result['completed'] and result['actual_phase_steps']==1000000
        if name.startswith('primary-'):
            return result['complete'] and result['episodes_per_cell']==200 and result['cells']==[[3,2.25],[6,2.25]]
        if name.startswith('eval-'): return result['complete'] and result['episodes_per_cell']==200 and len(result['cells'])==10
        if name.startswith('calibration-'): return result['complete'] and result['states']==64 and result['repetitions']==4
        return bool(result.get('complete'))
    except (OSError,ValueError,KeyError,IndexError): return False


def evidence_inputs_match(job, study, evidence):
    if not evidence.get('technical_checks_passed'): return False
    if job.get('policy') != evidence.get('policy'): return False
    if job.get('requires_claim_assessment') and not evidence.get('formal_evidence'): return False
    if job.get('freezes_protocol'):
        manifest=json.loads((Path(study)/'manifest.json').read_text())
        if (evidence.get('frozen_protocol')!=manifest['candidate_protocol'] or
                evidence.get('frozen_source')!=manifest['code']): return False
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


def conditional_selection(job, study, jobs):
    """Bind an optional replication to the exact reviewed first-seed decision."""
    condition = job.get('condition')
    if not condition:
        return True, None
    parent = jobs[condition['review']]
    if not review_passed(parent, study):
        raise RuntimeError('The conditional replication requires a current reviewed pilot decision.')
    path = Path(study)/'reviews'/condition['review']/'decision.json'
    decision = json.loads(path.read_text())
    return decision['assessment'] == condition['assessment'], digest(path)


def build_review_evidence(job, study, jobs):
    if job['stage']=='development-v6':
        return build_signal_first_evidence(job, study, jobs)
    from interaction_review import review_evidence, deployment_evidence
    study = Path(study)
    evidence = (deployment_evidence(study) if job['stage'] == 'deployment' else
                review_evidence(study, job['stage'], job['seeds'], job['scope']))
    def include(label, extra):
        evidence[label] = extra
        evidence['artifact_inputs'].update(extra['artifact_inputs'])
        for key, fingerprint in extra['inputs'].items():
            evidence['artifact_inputs'][extra['input_paths'][key]] = fingerprint
    if job.get('scope') == 'peer' and job['stage'] == 'joint':
        include('gate_stage', review_evidence(study, 'gate', job['seeds'], 'peer'))
    if job.get('peer_gate_control'):
        include('peer_candidate_control', review_evidence(study, 'gate', job['seeds'], 'peer'))
    if job.get('peer_control'):
        selected, _ = conditional_selection(dict(condition=dict(review='review-peer-pilot', assessment='inconclusive')),
                                             study, jobs)
        reviews = ['review-peer-pilot'] + (['review-peer-replication'] if selected else [])
        for name in reviews:
            if not review_passed(jobs[name], study):
                raise RuntimeError('The candidate ablation has not passed its required review.')
            for filename in ('decision.json', 'evidence.json'):
                path = study/'reviews'/name/filename
                evidence['artifact_inputs'][str(path.relative_to(study))] = digest(path)
        seeds = PILOT_SEEDS if selected else PILOT_SEEDS[:1]
        include('peer_candidate_control', review_evidence(study, 'joint', seeds, 'peer'))
        include('peer_candidate_gate_stage', review_evidence(study, 'gate', seeds, 'peer'))
    evidence['policy'] = job['policy']
    return evidence


def build_signal_first_evidence(job, study, jobs):
    study = Path(study)
    inputs, input_paths, artifacts, summaries = {}, {}, {}, {}
    for name in job['dependencies']:
        parent=jobs.get(name)
        if parent is None: continue
        if parent['kind']=='review':
            if not review_passed(parent, study): raise ValueError('An earlier development decision is not valid.')
            paths=[study/'reviews'/name/f for f in ('evidence.json','decision.json')]
        elif parent.get('completion'):
            paths=[Path(parent['completion'])]
        else:
            output=Path(parent['command'][parent['command'].index('--output')+1])
            paths=[output/'completed.json']
            paths.extend(p for p in (output/'episodes.csv',output/'calibration.csv') if p.exists())
        for path in paths:
            relative=str(path.relative_to(study))
            artifacts[relative]=digest(path)
        primary=paths[0]
        inputs[name], input_paths[name]=digest(primary), str(primary.relative_to(study))
        data=json.loads(primary.read_text())
        if parent['kind']!='review':
            if not data.get('complete'):
                raise ValueError('An incomplete diagnostic cannot release a development stage.')
            summaries[name]={key:data[key] for key in ('step','summary','learning_checks','frozen_proposals',
                'identifiability','E_delta','states','repetitions','interpretation','task_by_defenders')
                if key in data and not (key=='states' and isinstance(data[key],list))}
            if data.get('candidate_response'):
                summaries[name]['candidate_response']=data['candidate_response']['summary']
            spec=data.get('input',{})
            if spec.get('checkpoint_sha256'):
                command=parent['command']
                step=int(command[command.index('--checkpoint-step')+1]) if '--checkpoint-step' in command else 0
                checkpoint=study/f'training/seed-{spec["seed"]}/{spec["arm"]}'/('resume.pth' if step==0 else f'probe-{step}.pth')
                if digest(checkpoint)!=spec['checkpoint_sha256']:
                    raise ValueError('A diagnostic checkpoint changed after calibration.')
                artifacts[str(checkpoint.relative_to(study))]=spec['checkpoint_sha256']
            if data.get('repaired_probe_sha256'):
                checkpoint=study/f'training/seed-{spec["seed"]}/{spec["arm"]}/probe-{data["step"]}.pth'
                if digest(checkpoint)!=data['repaired_probe_sha256']:
                    raise ValueError('The reconditioned bootstrap probe changed.')
                artifacts[str(checkpoint.relative_to(study))]=data['repaired_probe_sha256']
                summaries[name]['bootstrap_repair']=dict(passed=data['passed'],bootstrap_updates=data['bootstrap_updates'],
                    actor_unchanged=data['actor_unchanged'],calibration=data['calibration']['summary'])
            if 'states_sha256' in data:
                state_path=primary.with_name('states.json')
                if digest(state_path)!=data['states_sha256']: raise ValueError('Signal states changed.')
                artifacts[str(state_path.relative_to(study))]=data['states_sha256']
            if name=='diagnostics-formal-evidence':
                for relative,expected in data['artifact_inputs'].items():
                    if digest(study/relative)!=expected: raise ValueError('A formal evidence input changed.')
                    artifacts[relative]=expected
            if name.startswith('primary-'):
                with primary.with_name('episodes.csv').open() as f: rows=list(csv.DictReader(f))
                if len(rows)!=400: raise ValueError('Incomplete formal primary and strong-interaction banks.')
                summaries[name]['summary']={m:sum(float(r[m]) for r in rows)/len(rows)
                    for m in ('success','capture','collision','breach','timeout','capture_time')}
    result=dict(stage=job['name'], technical_checks_passed=True, inputs=inputs, input_paths=input_paths,
        artifact_inputs=artifacts, summaries=summaries, policy=job['policy'],
        interpretation='Review mechanism, environment calibration, actual control and task/safety jointly. Unequal-step checkpoints are connectivity evidence only.')
    if job.get('requires_claim_assessment'):
        result['formal_evidence']=json.loads((study/'reviews/formal-evidence/completed.json').read_text())
    if job.get('freezes_protocol'):
        manifest=json.loads((study/'manifest.json').read_text())
        result.update(frozen_protocol=manifest['candidate_protocol'], frozen_source=manifest['code'])
        if manifest['candidate_protocol'].get('formal_evidence'):
            required={f'diagnostics-confirmation-{s}-{a}-250000' for s in PILOT_SEEDS for a in DEVELOPMENT_ARMS}
            if set(summaries)!=required: raise ValueError('Freeze requires all ten fresh development confirmations.')
            trends={}
            for seed in PILOT_SEEDS:
                arms={a:summaries[f'diagnostics-confirmation-{seed}-{a}-250000'] for a in DEVELOPMENT_ARMS}
                if any(x.get('step')!=250000 or not x.get('frozen_proposals',{}).get('finite') or
                       not x.get('frozen_proposals',{}).get('frozen_proposals_equal') for x in arms.values()):
                    raise ValueError('Development endpoints must be finite and correctly frozen.')
                trends[str(seed)]=dict(E_env_full_minus_same_info=arms['full']['summary']['E_env']-arms['same_info']['summary']['E_env'],
                    strong_capped_time_full_minus_no_peer=arms['full']['task_by_defenders']['6']['capture_time']-
                        arms['no_peer']['task_by_defenders']['6']['capture_time'])
            result['formal_entry_evidence']=dict(trends=trends,all_controls_complete=True,
                implementation_review='review-signal',recovery_history=manifest.get('bootstrap_revision'),
                requirement='Assess repeated conditional-value improvement, interpretable strong-interaction control and recovery history before freezing.')
            signal=study/'reviews/review-signal'
            signal_job=jobs['review-signal']
            if not review_passed(signal_job,study): raise ValueError('Implementation invariants have not passed.')
            for filename in ('evidence.json','decision.json'):
                result['artifact_inputs'][str((signal/filename).relative_to(study))]=digest(signal/filename)
    return result


def decide_review(study, name, decision, rationale, evidence_sha256, assessment=None, *, automatic=None, findings=None):
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
        if job.get('requires_claim_assessment'):
            if not findings or set(findings)!={'A','B','C','D'}:
                raise ValueError('Record an explicit A--D assessment with practical relevance before continuing.')
            if not all(evidence['formal_evidence']['statistical_checks'].values()):
                raise ValueError('The prespecified formal evidence checks do not support every A--D claim.')
            for claim,entry in findings.items():
                if entry.get('status')!='supported' or len(entry.get('rationale','').strip())<40:
                    raise ValueError('Every claim requires a supported, substantive evidence assessment.')
            if len(findings['A'].get('practical_relevance','').strip())<40:
                raise ValueError('Interpret absolute seconds and joint task/safety outcomes; significance alone is insufficient.')
        if job.get('freezes_protocol') and evidence.get('formal_entry_evidence'):
            required={'implementation','conditional_value','interaction','small_controls','fresh_confirmation','protocol'}
            if not findings or set(findings)!=required or any(v.get('status')!='supported' or
                    len(v.get('rationale','').strip())<40 for v in findings.values()):
                raise ValueError('Formal entry requires six explicit development and protocol assessments.')
            trends=evidence['formal_entry_evidence']['trends'].values()
            if not all(v['E_env_full_minus_same_info']<0 for v in trends):
                raise ValueError('The two development seeds do not yet show repeated conditional-value improvement.')
    if not rationale or len(rationale.strip())<40:
        raise ValueError('Record a substantive evidence-based reason, including limitations.')
    previous=[]
    if (directory/'decision.json').exists():
        old=json.loads((directory/'decision.json').read_text())
        previous=old.pop('history',[])+[old]
    record=dict(decision=decision,assessment=assessment,rationale=rationale.strip(),
        evidence_sha256=evidence_sha256,time=time.time(),history=previous)
    if automatic is not None: record['automatic_technical_review']=automatic
    if findings is not None: record['findings']=findings
    write_json(directory/'decision.json',record)


def automatic_technical_review(job, study, manifest):
    """Resolve only explicitly enabled intermediate pair checks; retain every hold."""
    policy=manifest.get('technical_review_automation')
    if (policy!=technical_review_policy() or job['name'] not in policy['reviews'] or
            job.get('freezes_protocol') or job.get('stage')!='development-v6' or
            job.get('policy',{}).get('allowed_assessments')!=['technical_ready']):
        return False
    directory=Path(study)/'reviews'/job['name']
    if (directory/'decision.json').exists(): return False
    path=directory/'evidence.json'
    fingerprint=digest(path)
    evidence=json.loads(path.read_text())
    failures, measurements=[], {}
    try:
        if not evidence_inputs_match(job,study,evidence):
            failures.append('Evidence, fixed probes or review policy no longer match.')
        screens={}
        for name in job['dependencies']:
            if not name.startswith('diagnostics-screen-42-'): continue
            relative=evidence['input_paths'][name]
            data=json.loads((Path(study)/relative).read_text())
            arm=data['input']['arm']
            if arm not in ('full','same_info') or arm in screens:
                raise ValueError('Expected exactly one Full and one Same-info screen.')
            screens[arm]=data
        if set(screens)!={'full','same_info'}:
            raise ValueError('Both fixed pair screens are required.')
        boundary=int(job['name'].rsplit('-',1)[1])
        for arm,data in screens.items():
            summary,learning,frozen=data['summary'],data['learning_checks'],data['frozen_proposals']
            if (data.get('complete') is not True or data.get('technical_checks_passed') is not True or
                    data['input']['seed']!=42 or not boundary<=data['step']<=policy['maximum_endpoint_step']):
                failures.append(f'{arm}: incomplete or out-of-budget diagnostic.')
            if frozen.get('finite') is not True or frozen.get('frozen_proposals_equal') is not True:
                failures.append(f'{arm}: nonfinite or changed frozen proposals.')
            required=('td_loss','bootstrap_td_loss','bootstrap_disagreement','td_target_abs_max',
                      'critic_gradient_norm','actor_gradient_norm','adapter_gradient_norm')
            for field in required: float(learning[field])
            if not all(math.isfinite(float(v)) for v in learning.values()):
                failures.append(f'{arm}: nonfinite learning telemetry.')
            if not all(math.isfinite(float(v)) for v in summary.values()):
                failures.append(f'{arm}: nonfinite environment diagnostic.')
            error,zero=float(summary['E_env']),float(summary['environment_zero_predictor'])
            if error<0 or zero<0 or error>policy['maximum_environment_error_to_zero']*zero+1e-12:
                failures.append(f'{arm}: environment calibration exceeds the numerical continuation bound.')
            if not 0<=summary['collision']<=policy['maximum_collision_fraction']:
                failures.append(f'{arm}: a collision requires diagnosis before continuation.')
            if not 0<summary['executed_nonzero_fraction']<=1:
                failures.append(f'{arm}: no verified executed intervention effect.')
            # A flat candidate-response curve or a task lead is not an efficacy decision.
            measurements[arm]=dict(step=data['step'],E_env=error,environment_zero_predictor=zero,
                collision=summary['collision'],success=summary['success'],capture=summary['capture'],
                capture_time=summary['capture_time'],bootstrap_td_loss=learning['bootstrap_td_loss'])
    except (OSError,KeyError,TypeError,ValueError,OverflowError) as exc:
        failures.append('Missing or invalid technical evidence: '+str(exc))
    rationale=('Automatic bounded pair check: '+('; '.join(failures) if failures else
        'Both fixed screens passed finite telemetry, frozen-parameter, environment calibration and executed-control checks, with no collisions.')+
        ' This only completes the already scheduled pair through 250k; unequal budgets do not establish efficacy, '
        'and replication, protocol freeze and formal training retain their separate evidence reviews.')
    decide_review(study,job['name'],'hold' if failures else 'continue',rationale,fingerprint,
        'unsupported' if failures else 'technical_ready',
        automatic=dict(policy=policy,measurements=measurements,failures=failures))
    return True


def migrate_technical_reviews(study):
    """Revise only scheduler decisions while preserving owned live training jobs."""
    import fcntl
    study=Path(study)
    with (study/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=study/'manifest.json'
        manifest=json.loads(path.read_text())
        if manifest['format']!='interaction-study-v6' or manifest['root']!=str(ROOT):
            raise ValueError('Technical review automation requires the original v6 study path.')
        if (study/'reviews/review-protocol-freeze/evidence.json').exists():
            raise RuntimeError('A frozen protocol cannot enter a development execution revision.')
        current=source_hashes()
        changed={p:dict(before=manifest['code'].get(p),after=current.get(p))
                 for p in set(current)|set(manifest['code']) if manifest['code'].get(p)!=current.get(p)}
        if set(changed)-{'scripts/run_interaction_study.py'} or set(current)!=set(manifest['code']):
            raise RuntimeError('Only scheduler source may change during live training.')
        jobs={j['name']:j for j in manifest['jobs']}
        running=[]
        for record_path in (study/'jobs').glob('*.json'):
            record=json.loads(record_path.read_text())
            if record.get('state')=='running' and process_identity(record.get('pid',0)) is not None:
                identity=verify_running_job(jobs[record['name']],record)
                running.append(dict(name=record['name'],pid=record['pid'],process_identity=identity))
        job=jobs['review-pair-100000']
        endpoint='diagnostics-screen-42-full-250000'
        # Bind the already completed Full endpoint instead of retaining its old recovery screen.
        if endpoint in jobs and valid_completion(jobs[endpoint],study):
            dependencies=[endpoint if n.startswith('diagnostics-screen-42-full-') else n
                          for n in job['dependencies']]
            if dependencies!=job['dependencies']:
                if (study/'reviews'/job['name']/'evidence.json').exists():
                    raise RuntimeError('An existing 100k review cannot be silently rebound.')
                job['dependencies']=dependencies
        manifest.setdefault('execution_revisions',[]).append(dict(time=time.time(),
            prior_manifest_sha256=digest(path),changes=changed,preserved_running=running,
            reason='User requested resolution of repeated training pauses. Automatically decide only the '
                   '50k/100k technical checks within the existing minimum-pair 250k budget; keep scientific stage reviews.'))
        manifest.update(code=current,technical_review_automation=technical_review_policy())
        write_json(path,manifest)


def migrate_evidence_standard(study):
    """Register A--D before formal data, preserving all existing learner jobs and artifacts."""
    import fcntl
    from interaction_evaluation import formal_evidence_standard
    study=Path(study)
    with (study/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=study/'manifest.json';manifest=json.loads(path.read_text())
        if manifest['format']!='interaction-study-v6' or manifest['root']!=str(ROOT):
            raise ValueError('Evidence registration requires the original v6 study path.')
        if (study/'reviews/review-protocol-freeze/evidence.json').exists():
            raise RuntimeError('Register evidence standards before protocol freeze and formal outcomes.')
        if any((study/f'training/seed-{seed}').exists() for seed in FORMAL_SEEDS):
            raise RuntimeError('Formal training already exists; its standard cannot be changed retrospectively.')
        current=source_hashes()
        changed={p:dict(before=manifest['code'].get(p),after=current.get(p))
                 for p in set(current)|set(manifest['code']) if manifest['code'].get(p)!=current.get(p)}
        allowed={'scripts/run_interaction_study.py','scripts/build_interaction_figures.py',
                 'train/interaction_evaluation.py','train/interaction_review.py'}
        if set(changed)-allowed or set(current)!=set(manifest['code']):
            raise RuntimeError('Evidence registration cannot change the learner or its training protocol.')
        jobs={j['name']:j for j in manifest['jobs']};running=[]
        for record_path in (study/'jobs').glob('*.json'):
            record=json.loads(record_path.read_text())
            if record.get('state')=='running' and process_identity(record.get('pid',0)) is not None:
                if jobs[record['name']]['kind']!='gpu' or not record['name'].startswith('develop-'):
                    raise RuntimeError('Finish active evaluation jobs before revising their evidence code.')
                running.append(dict(name=record['name'],pid=record['pid'],
                    process_identity=verify_running_job(jobs[record['name']],record)))
        core=jobs['review-formal-core'];name='diagnostics-formal-evidence'
        if name not in jobs:
            python=next(j['command'][0] for j in manifest['jobs'] if j['name'].startswith('develop-'))
            manifest['jobs'].append(dict(name=name,kind='cpu',phase='formal-evidence',
                command=[python,'-X','utf8','train/interaction_evaluation.py','evidence','--output',str(study)],
                dependencies=list(core['dependencies']),completion=str(study/'reviews/formal-evidence/completed.json')))
            core['dependencies']=[name]
        core.update(requires_claim_assessment=True)
        core['policy']['allowed_assessments']=['supported']
        core['policy']['criteria']=['Assess A--D using paired five-seed effects and 95% intervals.',
            'Primary outcome is capped capture time against ARBoids+CBF; interpret absolute seconds and CR/SR/collision.',
            'Require independent E_env attribution, the prespecified six-defender contrast, and separate matching/interaction/behavior evidence.']
        manifest.setdefault('execution_revisions',[]).append(dict(time=time.time(),prior_manifest_sha256=digest(path),
            changes=changed,preserved_running=running,
            reason='User registered A--D formal evidence, capped capture time, paired five-seed intervals and fixed strong-interaction attribution before any formal results.'))
        manifest['candidate_protocol']['formal_evidence']=formal_evidence_standard()
        manifest['candidate_protocol']['decision_rule']=protocol_specification()['decision_rule']
        manifest.update(code=current,evidence_standard_revision='AD-capped-capture-v1')
        write_json(path,manifest)


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
    manifest.update(format='interaction-study-v5',code=current)
    write_json(path,manifest)
    for record_path, record in records:
        if record.get('state')=='running':
            record.update(state='interrupted',interrupted_at=time.time(),reason='checkpointed stage-scheduling migration')
            write_json(record_path,record)
    write_json(study/'status.json',dict(state='prepared',jobs=len(manifest['jobs']),
        completed=sum(r.get('state')=='completed' for _,r in records),active=[]))
    lock.close()


def migrate_candidate_control(study):
    """Preserve the active v3 study while adding the reviewed candidate-control scope."""
    import fcntl
    study = Path(study)
    with (study/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = study/'manifest.json'
        manifest = json.loads(path.read_text())
        if manifest['format'] != 'interaction-study-v3' or manifest['root'] != str(ROOT):
            raise ValueError('Candidate-control migration requires the original v3 study path.')
        if any((study/'reviews').rglob('*.json')):
            raise RuntimeError('Existing diagnostic evidence needs an explicit revision; it cannot be replaced.')
        records = []
        for record_path in (study/'jobs').glob('*.json'):
            record = json.loads(record_path.read_text())
            if record.get('state') == 'running' and Path(f'/proc/{record.get("pid",0)}').exists():
                raise RuntimeError('Stop the owned jobs at recoverable checkpoints before migration.')
            records.append((record_path, record))
        current = source_hashes()
        changed = {p: dict(before=manifest['code'].get(p), after=current.get(p))
                   for p in set(current)|set(manifest['code']) if manifest['code'].get(p) != current.get(p)}
        allowed = {'scripts/run_interaction_study.py', 'train/interaction_review.py',
                   'train/policy/interaction_sac.py', 'train/interaction_rollout.py', 'train/train_interaction.py',
                   'scripts/build_interaction_figures.py'}
        if set(changed)-allowed or set(current) != set(manifest['code']):
            raise RuntimeError('Unreviewed source changes would alter the frozen study.')
        checkpoints = {str(p.relative_to(study)): digest(p) for p in (study/'training').glob('seed-*/*/*.pth')}
        for item in manifest['inherited_pretraining'].values():
            if digest(item['path']) != item['sha256']:
                raise RuntimeError('The inherited common pretraining changed.')
        first = next(j for j in manifest['jobs'] if j['name'].startswith('train-'))
        manifest['jobs'] = strengthen_jobs(manifest['jobs'], study, first['command'][0])
        manifest.setdefault('execution_revisions', []).append(dict(time=time.time(), prior_manifest_sha256=digest(path),
            changes=changed, preserved_checkpoints=checkpoints, preserved_job_records=[r['name'] for _, r in records],
            reason='User authorized fixed-state candidate diagnostics and a matched no-peer pilot before five-seed expansion. '
                   'Original arm settings, learned parameters and training budgets are retained.'))
        manifest.update(format='interaction-study-v4', code=current)
        write_json(path, manifest)
        for record_path, record in records:
            if record.get('state') == 'running':
                record.update(state='interrupted', interrupted_at=time.time(), reason='candidate-control migration at saved checkpoint')
                write_json(record_path, record)
        write_json(study/'status.json', dict(state='prepared', jobs=len(manifest['jobs']),
            completed=sum(r.get('state')=='completed' for _,r in records), active=[]))


def migrate_early_screen(study, gpu_slots=4):
    """Advance the first candidate screen and retain live compatible training processes."""
    import fcntl
    study = Path(study)
    with (study/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = study/'manifest.json'
        manifest = json.loads(path.read_text())
        if manifest['format'] != 'interaction-study-v4' or manifest['root'] != str(ROOT):
            raise ValueError('Early-screen migration requires the original v4 study path.')
        if (study/'reviews/review-core-gate/evidence.json').exists():
            raise RuntimeError('An existing first-stage decision needs an explicit evidence revision.')
        if gpu_slots not in (1, 2, 3, 4):
            raise ValueError('The first screen supports one to four shared-GPU training slots.')
        cpus = manifest['cpu_affinity']
        required_cpus = 4*gpu_slots + 2*manifest['limits']['cpu']
        quota_path = Path('/sys/fs/cgroup/cpu.max')
        quota = len(cpus)
        if quota_path.exists():
            amount, period = quota_path.read_text().split()
            if amount != 'max': quota = min(quota, int(amount)//int(period))
        if required_cpus > quota:
            raise ValueError('Parallel slots exceed the existing CPU affinity or container quota.')
        current = source_hashes()
        changed = {p: dict(before=manifest['code'].get(p), after=current.get(p))
                   for p in set(current)|set(manifest['code']) if manifest['code'].get(p) != current.get(p)}
        allowed = {'scripts/run_interaction_study.py', 'scripts/build_interaction_figures.py', 'train/interaction_review.py'}
        if set(changed)-allowed or set(current) != set(manifest['code']):
            raise RuntimeError('Unreviewed source changes would alter the frozen learner.')
        jobs = {j['name']: j for j in manifest['jobs']}
        running = []
        for record_path in (study/'jobs').glob('*.json'):
            record = json.loads(record_path.read_text())
            if record.get('state') == 'running' and process_identity(record.get('pid', 0)) is not None:
                identity = verify_running_job(jobs[record['name']], record)
                running.append(dict(name=record['name'], pid=record['pid'], process_identity=identity))
        first = next(j for j in manifest['jobs'] if j['name'].startswith('train-'))
        manifest['jobs'] = advance_candidate_screen(manifest['jobs'], study, first['command'][0])
        manifest.setdefault('execution_revisions', []).append(dict(time=time.time(), prior_manifest_sha256=digest(path),
            changes=changed, preserved_running=running, previous_limits=manifest['limits'].copy(),
            reason='User authorized moving the no-peer control into the first 250k screen and using idle compute. '
                   'Existing training processes continue; learner settings, datasets and per-arm step budgets are unchanged.'))
        manifest['limits']['gpu'] = gpu_slots
        manifest.update(format='interaction-study-v5', code=current, cpu_per_gpu=4, cpu_per_evaluation=2)
        write_json(path, manifest)


def migrate_signal_first(study):
    """Retain durable development checkpoints; replace the premature formal graph."""
    import fcntl
    import study_runtime
    import torch
    study=Path(study)
    with (study/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=study/'manifest.json'
        manifest=json.loads(path.read_text())
        if manifest['format']!='interaction-study-v5' or manifest['root']!=str(ROOT):
            raise ValueError('Signal-first migration requires the owned v5 development study.')
        for record_path in (study/'jobs').glob('*.json'):
            record=json.loads(record_path.read_text())
            if record.get('state') in ('running','paused_protocol_revision') and process_identity(record.get('pid',0)):
                raise RuntimeError('Stop owned jobs at durable checkpoints before changing the development protocol.')
        if list((study/'reviews').glob('review-*/decision.json')):
            raise RuntimeError('Previously reviewed scientific evidence needs an explicit protocol revision.')
        before=digest(path)
        current=source_hashes()
        changes={p:dict(before=h,after=current.get(p)) for p,h in manifest['code'].items() if current.get(p)!=h}
        allowed={'scripts/run_interaction_study.py','scripts/build_interaction_figures.py','train/interaction_review.py',
                 'train/train_interaction.py','train/policy/interaction_sac.py','train/interaction_evaluation.py'}
        if set(changes)-allowed or set(current)-set(manifest['code']):
            raise RuntimeError('The signal-first revision contains unreviewed source changes.')
        resume_steps, preserved={},{}
        for checkpoint in (study/'training').glob('seed-*/*/resume.pth'):
            seed=int(checkpoint.parents[1].name.split('-')[1])
            arm=checkpoint.parent.name
            saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
            if seed not in PILOT_SEEDS or saved['step']>250000 or saved['agent']['stage']!='gate':
                raise ValueError('A formal or joint checkpoint cannot be relabeled as a small development run.')
            key=f'{seed}/{arm}'
            resume_steps[key]=saved['step']
            preserved[key]=dict(step=saved['step'],sha256=digest(checkpoint),
                simulated_steps=saved['simulated_steps'],elapsed=saved['elapsed'])
            prior=json.loads((checkpoint.parent/'progress.json').read_text())
            preserved[key]['last_reported_step']=prior['step']
            preserved[key]['uncheckpointed_steps']=max(0,prior['step']-saved['step'])
            # Match the public progress to the last complete atomic resume, without rewriting its contents.
            write_json(checkpoint.parent/'progress.json',dict(step=saved['step'],stage='gate',complete=False,
                total=saved['config']['training']['gate_steps']+saved['config']['training']['joint_steps'],
                simulated_steps=saved['simulated_steps'],elapsed=saved['elapsed'],
                updates=max(0,saved['step']-saved['config']['training']['warm_steps']+1)))
            del saved
        first=next(j for j in manifest['jobs'] if j['name'].startswith('train-'))
        jobs,inherited=build_jobs(study,first['command'][0],ROOT/'vrx_ws/activate.bash',resume_steps)
        retired=set(j['name'] for j in manifest['jobs'])-set(j['name'] for j in jobs)
        manifest.setdefault('execution_revisions',[]).append(dict(time=time.time(),prior_manifest_sha256=before,
            reason='User adopted signal-first, pair-first development, two-seed small ablations and a separate formal freeze.',
            changes=changes,preserved_checkpoints=preserved,retired_jobs=sorted(retired)))
        manifest.update(format='interaction-study-v6',code=current,jobs=jobs,inherited_pretraining=inherited,
            seeds=FORMAL_SEEDS,development_seeds=PILOT_SEEDS,development_resume_steps=resume_steps,
            candidate_protocol=protocol_specification(),retired_jobs=sorted(retired),
            limits=dict(gpu=2,cpu=2,vrx=1,rollout_workers=4),cpu_per_gpu=6,cpu_per_evaluation=4)
        write_json(path,manifest)
        write_json(study/'status.json',dict(state='prepared_signal_first',active=[],jobs=len(jobs),
            preserved_checkpoints=preserved,limits=manifest['limits']))


def migrate_bootstrap(study):
    """Bind the verified development repair; retain the staged comparison graph."""
    import fcntl
    import study_runtime
    import torch
    study=Path(study)
    with (study/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=study/'manifest.json'
        manifest=json.loads(path.read_text())
        if manifest['format']!='interaction-study-v6' or manifest['root']!=str(ROOT):
            raise ValueError('Bootstrap migration requires the owned signal-first development study.')
        freeze=study/'reviews/review-protocol-freeze/decision.json'
        if freeze.exists() and json.loads(freeze.read_text()).get('decision')=='continue':
            raise RuntimeError('A frozen formal protocol cannot be changed by development recovery.')
        for record_path in (study/'jobs').glob('*.json'):
            record=json.loads(record_path.read_text())
            if record.get('state') in ('running','paused_protocol_revision') and process_identity(record.get('pid',0)):
                raise RuntimeError('Stop owned jobs before binding the bootstrap revision.')
        status=json.loads((study/'status.json').read_text()) if (study/'status.json').exists() else {}
        if status.get('active'):
            raise RuntimeError('Complete the active validation jobs before migrating.')
        before=digest(path)
        current=source_hashes()
        changes={p:dict(before=h,after=current.get(p)) for p,h in manifest['code'].items() if current.get(p)!=h}
        allowed={'scripts/run_interaction_study.py','scripts/build_interaction_figures.py','train/interaction_review.py',
            'train/train_interaction.py','train/policy/interaction_sac.py','train/interaction_rollout.py',
            'train/configs/interaction-aware-sac.yaml'}
        if set(changes)-allowed or set(current)-set(manifest['code']):
            raise RuntimeError('The bootstrap revision contains unrelated source changes.')
        steps,preserved={},{}
        for checkpoint in (study/'training').glob('seed-*/*/resume.pth'):
            seed=int(checkpoint.parents[1].name.split('-')[1]);arm=checkpoint.parent.name
            saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
            if seed not in PILOT_SEEDS or saved['step']>250000 or saved['agent']['stage']!='gate':
                raise ValueError('Only unfrozen gate-stage development may enter bootstrap recovery.')
            key=f'{seed}/{arm}'
            steps[key]=saved['step']
            preserved[key]=dict(step=saved['step'],sha256=digest(checkpoint),
                bootstrap_source=saved['config']['interaction'].get('bootstrap_source','coupled'))
            if seed==42 and arm in ('full','same_info'):
                record=json.loads((study/f'reviews/bootstrap-repair/{seed}-{arm}/completed.json').read_text())
                probe=checkpoint.parent/f'probe-{record["step"]}.pth'
                if (preserved[key]['bootstrap_source']!='real_td' or not record.get('complete') or
                    not record['passed'] or not record['actor_unchanged'] or digest(probe)!=record.get('repaired_probe_sha256')):
                    raise ValueError('The minimum pair requires completed, verified bootstrap recovery.')
            del saved
        if not {'42/full','42/same_info'}.issubset(steps):
            raise ValueError('Both minimum-pair development checkpoints are required.')
        first=next(j for j in manifest['jobs'] if j['name'].startswith('train-'))
        jobs,inherited=build_jobs(study,first['command'][0],ROOT/'vrx_ws/activate.bash',steps)
        retired=set(manifest.get('retired_jobs',[])) | (set(j['name'] for j in manifest['jobs'])-set(j['name'] for j in jobs))
        manifest.setdefault('execution_revisions',[]).append(dict(time=time.time(),prior_manifest_sha256=before,
            reason='User requested resolution of Full bootstrap miscalibration. Auxiliary supervision is disconnected '
                   'from the real-TD bootstrap teacher; recovery and fresh post-update calibration are development costs.',
            changes=changes,preserved_checkpoints=preserved,retired_jobs=sorted(retired)))
        manifest.update(code=current,jobs=jobs,inherited_pretraining=inherited,development_resume_steps=steps,
            candidate_protocol=protocol_specification(),bootstrap_revision='real-transition-only-bootstrap-v1',
            retired_jobs=sorted(retired))
        write_json(path,manifest)
        for job in jobs:
            if job['name'].startswith('diagnostics-') and job.get('completion') and valid_completion(job,study):
                result=json.loads(Path(job['completion']).read_text())
                wall=(result['learning_wall_seconds']+result['calibration']['wall_seconds']
                      if job['name'].startswith('diagnostics-bootstrap-repair-') else result['wall_seconds'])
                write_json(study/'jobs'/f'{job["name"]}.json',dict(name=job['name'],state='completed',returncode=0,
                    completed_before_migration=True,finished_at=time.time(),wall_seconds=wall))
        write_json(study/'status.json',dict(state='bootstrap_verified',active=[],jobs=len(jobs),
            preserved_checkpoints=preserved,limits=manifest['limits'],pending_reviews=['review-pair-connectivity-bootstrap']))


def run(study, adopt_running=False):
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
    active={}
    logs=study/'logs'; logs.mkdir(exist_ok=True)
    for path in (study/'jobs').glob('*.json'):
        item=json.loads(path.read_text())
        if item['name'] not in jobs:
            if item['name'] not in manifest.get('retired_jobs',[]):
                raise RuntimeError('Unrecognized historical job record: '+item['name'])
            continue
        if item.get('state') in ('completed', 'skipped'):
            job = jobs[item['name']]
            if job.get('condition'):
                enabled, fingerprint = conditional_selection(job, study, jobs)
                if (enabled == (item['state'] == 'skipped') or
                        item.get('condition_decision_sha256') != fingerprint):
                    raise RuntimeError('A conditional job no longer matches its reviewed pilot decision.')
            elif item['state'] == 'skipped':
                raise RuntimeError('A required job cannot be skipped.')
            if item['state'] == 'skipped':
                done.add(item['name'])
                continue
            if jobs[item['name']]['kind']=='review' and not review_passed(jobs[item['name']],study):
                raise RuntimeError('A completed stage review no longer matches its evidence.')
            done.add(item['name'])
        elif item.get('state')=='running':
            pid=item.get('pid',0)
            if pid and process_identity(pid) is not None:
                if not adopt_running:
                    raise RuntimeError(f'Existing study child {pid} is alive; do not duplicate it.')
                job = jobs[item['name']]
                identity = verify_running_job(job, item)
                slot, assigned = resource_assignment(manifest, job['kind'], jobs, active)
                set_tree_affinity(pid, assigned)
                log = (logs/f'{item["name"]}.log').open('a', encoding='utf-8')
                active[item['name']] = (AdoptedProcess(pid, identity), log, slot)
                item.update(adopted_at=time.time(), process_identity=identity, slot=slot, assigned_cpus=assigned)
                write_json(path, item)
                continue
            if valid_completion(jobs[item['name']],study):
                done.add(item['name'])
                write_json(path,dict(item,state='completed',recovered=True))
    external_clear=False
    failed=False
    pending_reviews=[]

    def status(state, blockers=None):
        write_json(study/'status.json',dict(state=state,runner_pid=os.getpid(),updated_at=time.time(),
            jobs=len(jobs),completed=len(done-set(manifest['inherited_pretraining'])),
            active=[dict(name=k,pid=v[0].pid,kind=jobs[k]['kind'],slot=v[2]) for k,v in active.items()],
            limits=manifest['limits'],
            pending_reviews=pending_reviews,blockers=blockers or [],
            core_evidence_complete=('review-formal-core' if manifest.get('format')=='interaction-study-v6' else 'review-evidence') in done,
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
                valid=code in (0, 'unobserved') and valid_completion(jobs[name],study)
                record=json.loads((study/'jobs'/f'{name}.json').read_text())
                record.update(returncode=None if code=='unobserved' else code,
                    state='completed' if valid else 'failed',finished_at=time.time())
                if code=='unobserved': record['completion_basis']='validated endpoint after adopting an existing process'
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
                enabled, selection_hash = conditional_selection(job, study, jobs)
                if not enabled:
                    done.add(name)
                    write_json(study/'jobs'/f'{name}.json',dict(name=name,state='skipped',finished_at=time.time(),
                        reason='The reviewed first candidate-ablation seed did not request a second seed.',
                        condition_decision_sha256=selection_hash))
                    continue
                kind=job['kind']
                if blockers or counts['exclusive']: continue
                if kind=='review':
                    directory=study/'reviews'/name
                    if not (directory/'evidence.json').exists():
                        evidence=build_review_evidence(job,study,jobs)
                        write_json(directory/'evidence.json',evidence)
                    automatic_technical_review(job,study,manifest)
                    if review_passed(job,study):
                        done.add(name)
                        write_json(study/'jobs'/f'{name}.json',dict(name=name,state='completed',finished_at=time.time(),
                            condition_decision_sha256=selection_hash,
                            evidence_sha256=digest(directory/'evidence.json'),decision_sha256=digest(directory/'decision.json')))
                    else:
                        pending_reviews.append(name)
                        write_json(study/'jobs'/f'{name}.json',dict(name=name,state='waiting_review',
                            condition_decision_sha256=selection_hash,
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
                slot, assigned=resource_assignment(manifest,kind,jobs,active)
                process=subprocess.Popen(job['command'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,preexec_fn=(lambda:os.sched_setaffinity(0,assigned)) if assigned else None)
                active[name]=(process,log,slot); counts[kind]+=1
                write_json(study/'jobs'/f'{name}.json',dict(name=name,pid=process.pid,state='running',started_at=time.time(),
                    process_identity=process_identity(process.pid),slot=slot,assigned_cpus=assigned,
                    condition_decision_sha256=selection_hash))
                if kind=='exclusive': break
            status('running' if active else 'waiting_review' if pending_reviews else 'waiting_dependencies')
            if not active and pending_reviews:
                decisions=[study/'reviews'/name/'decision.json' for name in pending_reviews]
                if all(p.exists() and json.loads(p.read_text()).get('decision')=='hold' for p in decisions):
                    status('held')
                    return
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
    parser.add_argument('mode',choices=['prepare','run','status','migrate-stages','migrate-evidence','migrate-candidate-control',
                                       'migrate-early-screen','migrate-signal-first','migrate-bootstrap',
                                       'migrate-technical-reviews','migrate-evidence-standard','decide-review'])
    parser.add_argument('--study',type=Path,default=ROOT/'train/experiments/ia-crrl-20261007')
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--vrx-activate',type=Path,default=ROOT/'vrx_ws/activate.bash')
    parser.add_argument('--review')
    parser.add_argument('--decision',choices=['continue','hold'])
    parser.add_argument('--rationale')
    parser.add_argument('--evidence-sha256')
    parser.add_argument('--assessment',choices=['technical_ready','promising','inconclusive','unsupported','necessary','frozen','supported'])
    parser.add_argument('--findings-file',type=Path)
    parser.add_argument('--gpu-slots', type=int, choices=(1,2,3,4), default=4)
    parser.add_argument('--adopt-running', action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare': prepare(args.study.resolve(),args.python,args.vrx_activate.resolve())
    elif args.mode=='run': run(args.study.resolve(),args.adopt_running)
    elif args.mode in ('migrate-stages','migrate-evidence'): migrate_stages(args.study.resolve())
    elif args.mode=='migrate-candidate-control': migrate_candidate_control(args.study.resolve())
    elif args.mode=='migrate-early-screen': migrate_early_screen(args.study.resolve(),args.gpu_slots)
    elif args.mode=='migrate-signal-first': migrate_signal_first(args.study.resolve())
    elif args.mode=='migrate-bootstrap': migrate_bootstrap(args.study.resolve())
    elif args.mode=='migrate-technical-reviews': migrate_technical_reviews(args.study.resolve())
    elif args.mode=='migrate-evidence-standard': migrate_evidence_standard(args.study.resolve())
    elif args.mode=='decide-review':
        decide_review(args.study.resolve(),args.review,args.decision,args.rationale,args.evidence_sha256,args.assessment,
            findings=json.loads(args.findings_file.read_text(encoding='utf-8')) if args.findings_file else None)
    else: print((args.study/'status.json').read_text(encoding='utf-8'))


if __name__=='__main__': main()
