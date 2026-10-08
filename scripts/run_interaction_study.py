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

ARMS = ('full', 'same_info', 'arboids_cbf', 'short', 'model_value')
PILOT_SEEDS = (42, 101)
VRX_ARMS = ('full', 'same_info', 'arboids_cbf')
FORMAL_SEEDS = (202, 303, 404, 505, 606)
DEVELOPMENT_ARMS = ('full', 'same_info', 'no_peer', 'short', 'model_value')
FORMAL_ARMS = (*ARMS, 'no_peer')
BLOCKING_PROJECTS = ('/root/arboids-formal-20261005', '/root/autodl-tmp/channel-evidence-20261005')


def technical_review_policy(connectivity=False, regression=False):
    """Operational bounds for completing the existing pair, never an efficacy test."""
    policy = dict(version='bounded-pair-connectivity-v2' if connectivity else 'bounded-pair-technical-v1',
        reviews=(['review-pair-connectivity'] if connectivity else [])+['review-pair-50000', 'review-pair-100000'],
        maximum_endpoint_step=250000, maximum_environment_error_to_zero=1.25,
        maximum_collision_fraction=0.,
        interpretation='Finite, calibrated development only; no automatic promising or frozen assessment.')
    if regression:
        policy.update(version='bounded-pair-regression-v3', task_regression_reference_step=5000,
                      task_regression_bootstrap_repetitions=10000)
        policy['reviews'].insert(1, 'review-pair-10000')
    return policy


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


def build_formal_jobs(study, python, vrx_activate):
    study=Path(study)
    seeds, arms = FORMAL_SEEDS, FORMAL_ARMS
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
                        scene=540000000+seeds.index(seed)*10000+n*100+setting*1000+trial
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
    return jobs, inherited


def protocol_specification():
    """Concrete candidate protocol; the freeze review binds this object and the source hashes."""
    config = yaml.safe_load((ROOT/'train/configs/interaction-aware-sac.yaml').read_text())
    interaction, batch = config['interaction'], config['rl']['batch_size']
    from interaction_evaluation import formal_evidence_standard
    return dict(development_seeds=PILOT_SEEDS, formal_seeds=FORMAL_SEEDS,
        development_arms=DEVELOPMENT_ARMS, formal_arms=FORMAL_ARMS,
        development_environment_steps=250000, formal_environment_steps=1000000,
        formal_gate_steps=250000, formal_joint_steps=750000, formal_curriculum_interval=187500,
        common_pretraining_steps=1000000, configuration=config,
        pair_reference=[0., .5, 1.], long_horizon_steps=interaction['horizon_steps'], short_horizon_steps=10,
        intervention_sampling=f"{interaction['pairs_per_batch']} snapshots every {interaction['intervention_interval']} real steps; reuse between refreshes; no outcome filtering",
        auxiliary_pairs_per_update=interaction['pairs_per_batch'], real_replay_batch=batch,
        auxiliary_to_real_batch_ratio=interaction['pairs_per_batch']/batch,
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
    formal, inherited = build_formal_jobs(study, python, vrx_activate)
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


def performance_search_jobs(study, python, previous, previous_jobs, pretrain):
    """Spend the exploration budget on four candidate methods and paired tasks."""
    study, previous = Path(study), Path(previous)
    graph = {j['name']: j for j in previous_jobs}
    candidates = {}
    for name, arm in (('long', 'full'), ('short', 'short')):
        command = list(graph[f'develop-42-{arm}-10000']['command'])
        candidates[name] = dict(output=command[command.index('--output') + 1], train=False,
            command=command, inherited_from=str(previous), training_defenders=3,
            description=f'Continue v8 candidate interaction with H={300 if arm == "full" else 10}.')
    for scope, defenders in (('single', 3), ('single', 6), ('team', 3), ('team', 6)):
        name = f'time_{scope}_n{defenders}'
        output = study/'training'/name
        command = [python, '-X', 'utf8', 'train/train_interaction.py', '--output', str(output),
            '--arm', 'full', '--seed', '42', '--pretrain', str(pretrain), '--device', 'cuda:0',
            '--stop-at-step', '10000', '--critic-control-coordinates', 'nominal-thrust-v2',
            '--warm-steps', '1000',
            '--bootstrap-estimator', 'mean', '--policy-learning-rate', '0.00003',
            '--long-horizon-steps', '300', '--label-repetitions', '4',
            '--reward-objective', 'capped-time-v1', '--gate-objective', 'paired-improvement-v1',
            '--defenders', str(defenders), '--intervention-scope', scope]
        candidates[name] = dict(output=str(output), command=command, train=True, training_defenders=defenders,
            intervention_scope=scope,
            description='Direct paired gate improvement from complete deployment-policy capped-time returns.')
    jobs = []
    def screen(name, step, dependencies):
        jobs.append(dict(name=f'diagnostics-performance-{name}-{step}', kind='cpu',
            command=[python, '-X', 'utf8', 'scripts/run_interaction_study.py', 'performance-screen',
                '--study', str(study), '--candidate', name, '--checkpoint-step', str(step)],
            dependencies=dependencies, completion=str(study/'screens'/f'{name}-{step}'/'completed.json')))
    screen('source', 0, [])
    for name in ('long', 'short'):
        screen(name, 5000, ['diagnostics-performance-source-0'])
        screen(name, 10000, ['diagnostics-performance-source-0'])
    latest = {}
    for step in (10000, 25000, 50000):
        for name, candidate in candidates.items():
            if not candidate['train']:
                continue
            command = list(candidate['command'])
            command[command.index('--stop-at-step') + 1] = str(step)
            job = f'develop-performance-{name}-{step}'
            jobs.append(dict(name=job, kind='gpu', command=command,
                dependencies=[] if name not in latest else [latest[name]], endpoint_step=step,
                phase='performance-exploration', candidate=name))
            screen(name, step, [job, 'diagnostics-performance-source-0'])
            latest[name] = job
    return jobs, candidates


def prepare_performance_search(study, previous, python):
    import fcntl
    from interaction_evaluation import formal_evidence_standard
    study, previous = Path(study).resolve(), Path(previous).resolve()
    if (study/'manifest.json').exists():
        raise RuntimeError('Performance search already exists; resume its runner.')
    with (previous/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads((previous/'manifest.json').read_text())
        if old['root'] != str(ROOT) or old.get('superseded_by'):
            raise ValueError('Expected the current owned development study.')
        if (previous/'reviews/review-protocol-freeze/decision.json').exists():
            raise RuntimeError('An assessed formal freeze cannot be changed by exploration.')
        for path in (previous/'jobs').glob('*.json'):
            record = json.loads(path.read_text())
            if record.get('pid') and process_identity(record['pid']):
                raise RuntimeError('Finish or checkpoint owned jobs before changing the source manifest.')
        source = verified_existing_pretrain(42)
        if source is None:
            raise RuntimeError('Verify the shared source actor before launching candidates.')
        jobs, candidates = performance_search_jobs(study, python, previous, old['jobs'], source['path'])
        inherited = {}
        for name in ('long', 'short'):
            directory = Path(candidates[name]['output'])
            config = yaml.safe_load((directory/'config.yaml').read_text())
            if (config['interaction'].get('entropy_objective') != 'proposal-mean-v3' or
                    config['interaction'].get('bootstrap_estimator') != 'mean' or
                    not (directory/'probe-5000.pth').exists() or not (directory/'probe-10000.pth').exists()):
                raise ValueError('Only the unchanged v8 candidate checkpoints may be continued.')
            inherited[name] = dict(path=str(directory/'resume.pth'), sha256=digest(directory/'resume.pth'),
                step=json.loads((directory/'progress.json').read_text())['step'],
                previous_source_hashes=old['code'])
        manifest = dict(format='interaction-performance-v1', root=str(ROOT), code=source_hashes(),
            jobs=jobs, candidates=candidates, inherited_pretraining={'pretrain-42': source},
            inherited_development=inherited, previous_study=str(previous),
            limits=dict(gpu=4, cpu=2, vrx=1, rollout_workers=4), cpu_per_gpu=4, cpu_per_evaluation=2,
            cpu_affinity=old['cpu_affinity'], wait_for_projects=old.get('wait_for_projects', []),
            exploration=dict(priority='Find excellent candidate-interaction task performance, strengthen it, then complete paper evidence.',
                maximum_endpoint_step=50000, development_seed=42, episodes_per_condition=32,
                conditions=[dict(defenders=3, agility=2.25, bank=860000000),
                            dict(defenders=6, agility=2.25, bank=861000000)],
                confirmation_banks=[864000000, 865000000],
                selection='Compare capped capture time, capture, success and physical collision jointly. Keep every fixed endpoint and scenario.',
                reference='Shared one-million-step pretrained ARBoids+CBF; screening reference only, not the final matched-budget comparison.',
                retired_routes=dict(long='100-episode validation time 20.384 to 33.598 s, capture .98 to .62 at 10000.',
                                    short='100-episode validation time 20.448 to 34.472 s, capture .98 to .60 at 10000.'),
                deferred=['complete ablations', 'cross-seed replication', 'high-precision value calibration', 'mechanism proof']),
            formal_evidence=formal_evidence_standard(), formal_method_frozen=False, formal_jobs_released=False)
        write_json(study/'manifest.json', manifest)
        for job in jobs:
            if job['name'].startswith('develop-') and valid_completion(job, study):
                write_json(study/'jobs'/f'{job["name"]}.json', dict(name=job['name'], state='completed',
                    returncode=0, inherited_from=str(previous), completion_basis='Existing fixed endpoint and progress verified'))
        old['superseded_by'] = str(study)
        old['superseded_reason'] = manifest['exploration']['priority']
        write_json(previous/'manifest.json', old)
        write_json(study/'status.json', dict(state='prepared', active=[], jobs=len(jobs),
            core_evidence_complete=False, submission_evidence_complete=False))


_PERFORMANCE_POLICIES = {}


def _performance_episode(task):
    import study_runtime
    import torch
    from interaction_evaluation import OriginalPolicy
    from interaction_rollout import DeploymentPolicy
    from train_interaction import episode
    checkpoint, original, defenders, agility, seed = task
    torch.set_num_threads(1)
    key = checkpoint, original
    if key not in _PERFORMANCE_POLICIES:
        _PERFORMANCE_POLICIES[key] = OriginalPolicy(checkpoint) if original else DeploymentPolicy(checkpoint)
    row, _ = episode(_PERFORMANCE_POLICIES[key], seed, defenders, agility, safety=True)
    return dict(row, scene_seed=seed, defenders=defenders, agility=agility)


def performance_summary(rows, baseline=None):
    import numpy as np
    if not rows:
        raise ValueError('Performance summaries require task outcomes.')
    key = lambda r: (r['defenders'], r['agility'], r['scene_seed'])
    scenes = {key(r) for r in rows}
    if len(scenes) != len(rows):
        raise ValueError('Duplicate candidate scenes.')
    if baseline is not None:
        reference = {key(r): r for r in baseline}
        if len(reference) != len(baseline):
            raise ValueError('Duplicate reference scenes.')
        if scenes != set(reference):
            raise ValueError('Candidate and reference must contain the same paired scenes and conditions.')
    results = {}
    for defenders, agility in sorted({(r['defenders'], r['agility']) for r in rows}):
        cell = [r for r in rows if (r['defenders'], r['agility']) == (defenders, agility)]
        result = {key: float(np.mean([r[key] for r in cell]))
                  for key in ('capture_time', 'capture', 'success', 'collision', 'breach', 'timeout')}
        result['episodes'] = len(cell)
        if baseline is not None:
            differences = np.asarray([r['capture_time'] - reference[key(r)]['capture_time'] for r in cell])
            rng = np.random.default_rng(862000000 + defenders)
            means = differences[rng.integers(len(cell), size=(2000, len(cell)))].mean(-1)
            result['time_difference_from_source'] = dict(mean=float(differences.mean()),
                interval_95=list(map(float, np.quantile(means, [.025, .975]))),
                interpretation='Exploratory paired scenarios for one development seed; not a formal five-seed claim.')
        results[f'n{defenders}-a{agility:g}'] = result
    return results


def performance_screen(study, candidate, step):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    study = Path(study).resolve()
    manifest = json.loads((study/'manifest.json').read_text())
    if manifest['format'] != 'interaction-performance-v1':
        raise ValueError('Expected a performance-search manifest.')
    original = candidate == 'source'
    checkpoint = (Path(manifest['inherited_pretraining']['pretrain-42']['path']) if original else
                  Path(manifest['candidates'][candidate]['output'])/f'probe-{step}.pth')
    fingerprint = digest(checkpoint)
    tasks = [(str(checkpoint), original, c['defenders'], c['agility'], c['bank'] + i)
             for c in manifest['exploration']['conditions']
             for i in range(manifest['exploration']['episodes_per_condition'])]
    started = time.perf_counter()
    with ProcessPoolExecutor(2, mp_context=mp.get_context('spawn')) as executor:
        rows = list(executor.map(_performance_episode, tasks, chunksize=4))
    if digest(checkpoint) != fingerprint:
        raise RuntimeError('Evaluation checkpoint changed during its fixed screen.')
    baseline = None if original else json.loads((study/'screens/source-0/completed.json').read_text())
    if baseline is not None:
        if baseline['checkpoint_sha256'] != manifest['inherited_pretraining']['pretrain-42']['sha256']:
            raise RuntimeError('Screen reference no longer matches the shared source.')
    output = dict(complete=True, candidate=candidate, step=step, checkpoint=str(checkpoint),
        checkpoint_sha256=fingerprint, wall_seconds=time.perf_counter()-started,
        summary=performance_summary(rows, None if original else baseline['episodes']), episodes=rows,
        formal_evidence=False)
    write_json(study/'screens'/f'{candidate}-{step}'/'completed.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'episodes'}), flush=True)


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


def composed_revision_jobs(study,python,vrx_activate,shared_pretrain_job):
    """Fresh, matched physical-coordinate revision with bounded early control work."""
    import copy
    study=Path(study)
    jobs,inherited=build_jobs(study,python,vrx_activate)
    jobs=parallel_development_jobs(jobs,study,python)
    graph={j['name']:j for j in jobs}
    shared=copy.deepcopy(shared_pretrain_job);shared['dependencies']=[]
    graph['pretrain-101'].update(shared)
    common=Path(shared['command'][shared['command'].index('--output')+1])/'actor.pth'
    graph['diagnostics-signal']['kind']='cpu'
    signal=graph['diagnostics-signal']['command'];signal[signal.index('--workers')+1]='2'
    flags=['--critic-control-coordinates','nominal-thrust-v2','--label-repetitions','8','--critic-warmup-updates','5000',
           '--entropy-objective','proposal-mean-v3','--critic-warmup-target','complete_real_state_return',
           '--policy-learning-rate','0.00001','--long-horizon-steps','300','--bootstrap-estimator','mean']
    for job in jobs:
        command=job['command']
        if '--pretrain' in command and command[command.index('--pretrain')+1]==str(study/'pretrain/seed-101/actor.pth'):
            command[command.index('--pretrain')+1]=str(common)
        if job['name'].startswith(('develop-','train-')) and command[command.index('--arm')+1]!='arboids_cbf':
            command.extend(flags)
    early_screens=[]
    for arm in ('full','same_info'):
        task=copy.deepcopy(graph[f'develop-42-{arm}-50000'])
        task.update(name=f'develop-42-{arm}-10000',endpoint_step=10000)
        task['command'][task['command'].index('--stop-at-step')+1]='10000'
        task['dependencies']=['review-pair-connectivity',f'develop-42-{arm}-5000','pretrain-42']
        diagnostic=copy.deepcopy(graph[f'diagnostics-screen-42-{arm}-50000'])
        diagnostic.update(name=f'diagnostics-screen-42-{arm}-10000',checkpoint_step=10000,
            dependencies=[task['name']],completion=str(study/f'reviews/screens/42-{arm}-10000/completed.json'))
        diagnostic['command'][diagnostic['command'].index('--checkpoint-step')+1]='10000'
        jobs.extend([task,diagnostic]);early_screens.append(diagnostic['name'])
        graph[f'develop-42-{arm}-50000']['dependencies']=['review-pair-10000',task['name'],'pretrain-42']
    early_review=copy.deepcopy(graph['review-pair-50000'])
    early_review.update(name='review-pair-10000',dependencies=['review-pair-connectivity',*early_screens])
    jobs.append(early_review)
    for arm in ('no_peer','short','model_value'):
        endpoint=graph[f'develop-42-{arm}-250000']
        for step,review,prior in ((5000,'review-signal',None),(10000,'review-pair-connectivity',5000),
                                  (50000,'review-pair-10000',10000)):
            job=copy.deepcopy(endpoint);job['name']=f'develop-42-{arm}-{step}';job['endpoint_step']=step
            command=job['command'];command[command.index('--stop-at-step')+1]=str(step)
            job['dependencies']=[review,'pretrain-42']
            if prior: job['dependencies'].append(f'develop-42-{arm}-{prior}')
            jobs.append(job)
        endpoint['dependencies']=['review-pair-100000',f'develop-42-{arm}-50000','pretrain-42']
    # The checked implementation may collect its bounded 5k probes concurrently
    # with the independent signal audit. Expansion still needs that audit.
    for job in jobs:
        if job['name'].startswith('develop-42-') and job.get('endpoint_step')==5000:
            job['dependencies']=['pretrain-42']
    graph['review-pair-connectivity']['dependencies'].append('review-signal')
    # The second common actor is already available. Its bounded matched probes
    # can test seed sensitivity while the first seed's longer jobs are running.
    for arm in DEVELOPMENT_ARMS:
        endpoint=graph[f'develop-101-{arm}-250000']
        task=copy.deepcopy(endpoint)
        task.update(name=f'develop-101-{arm}-5000',endpoint_step=5000,dependencies=['pretrain-101'])
        task['command'][task['command'].index('--stop-at-step')+1]='5000'
        screen=copy.deepcopy(graph[f'diagnostics-screen-101-{arm}-250000'])
        screen.update(name=f'diagnostics-screen-101-{arm}-5000',checkpoint_step=5000,dependencies=[task['name']],
            completion=str(study/f'reviews/screens/101-{arm}-5000/completed.json'))
        screen['command'][screen['command'].index('--checkpoint-step')+1]='5000'
        jobs.extend([task,screen]);endpoint['dependencies'].append(task['name'])
    return jobs,inherited,str(common)


def next_development_bank_offset(previous):
    offset = int(previous) + 20000000
    # Retain every earlier development bank and reserve the entire formal
    # family. The next unused development family begins at 800M.
    if 407000000 + offset >= 510000000 and 390000000 + offset < 800000000:
        offset = 410000000
    if previous < 0 or 408000000 + offset >= 2**31:
        raise ValueError('Development bank allocation is outside the reserved seed space.')
    return offset


def prepare_composed_revision(study,previous,python,vrx_activate):
    """Retain old evidence; transfer the unchanged common pretrain and start fresh matched actors."""
    import fcntl
    study,previous=Path(study).resolve(),Path(previous).resolve()
    if (study/'manifest.json').exists(): raise RuntimeError('The revision already exists; resume it.')
    with (previous/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old_path=previous/'manifest.json';old=json.loads(old_path.read_text())
        if old['root']!=str(ROOT) or old['format']!='interaction-study-v6':
            raise ValueError('The previous owned development study is required.')
        if old.get('superseded_by') or (previous/'reviews/review-protocol-freeze/evidence.json').exists() or any(
                (previous/f'training/seed-{seed}').exists() for seed in FORMAL_SEEDS):
            raise RuntimeError('This revision must precede formal protocol freeze and formal outcomes.')
        old_jobs={j['name']:j for j in old['jobs']};live=[]
        for path in (previous/'jobs').glob('*.json'):
            record=json.loads(path.read_text())
            if record.get('state')=='running' and process_identity(record.get('pid',0)) is not None:
                if record['name']!='pretrain-101': raise RuntimeError('Checkpoint the owned old learners before changing protocols.')
                verify_running_job(old_jobs[record['name']],record);live.append(record)
        preserved={}
        for path in (previous/'training').glob('seed-*/*/resume.pth'):
            progress=json.loads(path.with_name('progress.json').read_text())
            preserved[str(path.relative_to(previous))]=dict(sha256=digest(path),step=progress['step'])
        jobs,inherited,common=composed_revision_jobs(study,python,vrx_activate,old_jobs['pretrain-101'])
        bank_offset=next_development_bank_offset(old.get('development_bank_offset',0))
        protocol=protocol_specification()
        protocol['configuration']['interaction'].update(critic_control_coordinates='nominal-thrust-v2',label_repetitions=8,
                                                       entropy_objective='proposal-mean-v3',horizon_steps=300,
                                                       bootstrap_estimator='mean')
        protocol['configuration']['training'].update(critic_warmup_updates=5000,critic_warmup_target='complete_real_state_return')
        protocol['configuration']['rl'].update(actor_learning_rate=1e-5,temperature_learning_rate=1e-5)
        protocol.update(critic_coordinates='Value depends on normalized nominal composed thrust, not its proposal/gate factorization.',
            bootstrap_estimator='Arithmetic mean of the two independent real-data evaluator heads for TD and truncated tails; retain the main-critic minimum for policy improvement and the registered value-difference diagnostics. Identical across all five matched arms.',
            entropy_objective='No gate-density reward in either stage. Gate-only continuation optimizes task return; joint learning includes only mean proposal log density per active defender. Proposal temperature starts at 0.2 and adapts in the joint stage to -2 per defender. TD, actor, labels and calibration share this objective; task reward and all A--D thresholds remain fixed.',
            intervention_sampling='64 fixed root interventions every 1000 real steps after critic warmup; eight independent paired futures per root.',
            long_horizon_steps=300,
            critic_warmup='5000 state-value evaluator updates against complete returns from the initial fixed-policy real episodes; exclude root entropy and unfinished episodes. Control-input columns start at zero and remain zero during this initialization. Copy the evaluator into both main and lagged critics, then release all control columns for ordinary learning. Identical across five matched arms, charged separately.',
            learning_timescales='Critics use 1e-4; actor and temperature use 1e-5 in all five matched arms.',
            long_labels='Full, No-peer and Model-return branches run for at most 300 steps, reaching actual 60-second task termination with no learned tail. Short retains 10-step labels and a lagged real-data value tail.',
            optimizer_isolation='Main and bootstrap optimizers clone all state tensors, including CPU Adam counters; Same-info equality is checked after updates.',
            development_banks=[x+bank_offset for x in protocol['development_banks']]+[x+bank_offset for x in (404000000,405000000,406000000,407000000)],
            confirmation_banks=[398000000+bank_offset,399000000+bank_offset],
            development_precision=dict(states=64,repetitions=16,full_environment=True,banks=[404000000+bank_offset,405000000+bank_offset]))
        manifest=dict(format='interaction-study-v6',root=str(ROOT),seeds=list(FORMAL_SEEDS),protocol='paper-parameters-v1',
            development_seeds=list(PILOT_SEEDS),candidate_protocol=protocol,code=source_hashes(),jobs=jobs,
            inherited_pretraining=inherited,pretraining_outputs={'101':common},development_bank_offset=bank_offset,
            technical_review_automation=technical_review_policy(True,True),
            limits=dict(gpu=4,cpu=2,vrx=1,rollout_workers=4),cpu_per_gpu=4,cpu_per_evaluation=2,
            cpu_affinity=old['cpu_affinity'],wait_for_projects=old.get('wait_for_projects',list(BLOCKING_PROJECTS)),
            development_execution=dict(version='composed-control-v8',seeds=list(PILOT_SEEDS),arms=list(DEVELOPMENT_ARMS),
                endpoint_steps=250000,precision_states=64,precision_repetitions=16,mechanism_states=32,mechanism_repetitions=4),
            previous_development=dict(study=str(previous),manifest_sha256=digest(old_path),preserved_checkpoints=preserved,
                reason='User requested continued changes until all A--D criteria hold. Mean bootstrapping reduced evaluator underestimation but v7 Full/Same-info 10k validation capped times were 26.822/29.888 seconds versus about 20.35 at 5k. Same-info median pre-transform gate standard deviation increased from 0.100 to 0.875; its initial log-standard-deviation bias gradient was -0.19874 from entropy versus -0.00023 from value. Alternative deterministic expectation and gate sampling did not restore task performance. Remove gate-density regularization in all five matched arms, retaining proposal entropy only during joint learning, common initialization and budgets, and all historical results and costs.'))
        write_json(study/'manifest.json',manifest)
        shared_path=previous/'jobs/pretrain-101.json'
        shared_record=json.loads(shared_path.read_text()) if shared_path.exists() else None
        if shared_record and (live or valid_completion(old_jobs['pretrain-101'],previous)):
            write_json(study/'jobs/pretrain-101.json',dict(shared_record,transferred_from=str(previous)))
        old.update(superseded_by=str(study),superseded_at=time.time())
        write_json(old_path,old)
        write_json(previous/'status.json',dict(state='superseded_development_revision',active=[],revision=str(study),
            preserved_checkpoints=preserved,transferred_pretraining=[r['pid'] for r in live]))
        write_json(study/'status.json',dict(state='prepared',jobs=len(jobs),active=live,previous_study=str(previous)))


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
                checkpoint=(Path(spec['checkpoint']) if spec.get('checkpoint') else
                    study/f'training/seed-{spec["seed"]}/{spec["arm"]}'/('resume.pth' if step==0 else f'probe-{step}.pth'))
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
            if {n for n in summaries if n.startswith('diagnostics-confirmation-')}!=required:
                raise ValueError('Freeze requires all ten fresh development confirmations.')
            trends={}
            for seed in PILOT_SEEDS:
                arms={a:summaries[f'diagnostics-confirmation-{seed}-{a}-250000'] for a in DEVELOPMENT_ARMS}
                if any(x.get('step')!=250000 or not x.get('frozen_proposals',{}).get('finite') or
                       not x.get('frozen_proposals',{}).get('frozen_proposals_equal') for x in arms.values()):
                    raise ValueError('Development endpoints must be finite and correctly frozen.')
                trends[str(seed)]=dict(E_env_full_minus_same_info=arms['full']['summary']['E_env']-arms['same_info']['summary']['E_env'],
                    strong_capped_time_full_minus_no_peer=arms['full']['task_by_defenders']['6']['capture_time']-
                        arms['no_peer']['task_by_defenders']['6']['capture_time'])
                if manifest.get('development_execution'):
                    precision={a:summaries[f'diagnostics-precision-{seed}-{a}-250000']['summary']
                               for a in ('full','same_info')}
                    if any(p['states']!=64 or p['repetitions']!=16 for p in precision.values()):
                        raise ValueError('The registered high-precision development bank is incomplete.')
                    trends[str(seed)].update(
                        confirmation_E_env_full_minus_same_info=trends[str(seed)]['E_env_full_minus_same_info'],
                        E_env_full_minus_same_info=precision['full']['E_env']-precision['same_info']['E_env'],
                        precision=precision)
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
    if (policy not in (technical_review_policy(),technical_review_policy(True),technical_review_policy(True,True)) or job['name'] not in policy['reviews'] or
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
        boundary=5000 if job['name']=='review-pair-connectivity' else int(job['name'].rsplit('-',1)[1])
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
            if policy.get('task_regression_reference_step') and boundary>policy['task_regression_reference_step']:
                import numpy as np
                initial=json.loads((Path(study)/f'reviews/screens/42-{arm}-5000/completed.json').read_text())
                key=lambda row:(row['scene_seed'],row['defenders'])
                baseline={key(row):row['capture_time'] for row in initial['episodes']}
                current={key(row):row['capture_time'] for row in data['episodes']}
                if len(current)!=32 or set(current)!=set(baseline):
                    raise ValueError('The regression check requires the same 32 paired task scenes.')
                delta=np.array([current[k]-baseline[k] for k in sorted(current)])
                rng=np.random.default_rng(790042)
                means=delta[rng.integers(len(delta),size=(policy['task_regression_bootstrap_repetitions'],len(delta)))].mean(1)
                interval=np.quantile(means,[.025,.975])
                measurements[arm]['task_change_from_5000']=dict(mean=float(delta.mean()),ci95=interval.tolist())
                if interval[0]>0.:
                    failures.append(f'{arm}: paired capture time regressed from the fixed 5k screen; diagnose before expansion.')
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


def parallel_development_jobs(jobs, study, python):
    """Collect the bounded development matrix concurrently; keep scientific formal-entry reviews."""
    import copy
    jobs=copy.deepcopy(jobs); study=Path(study)
    graph={j['name']:j for j in jobs}
    anchor='review-pair-100000'
    if anchor not in graph: raise ValueError('The pair technical checkpoint is required.')
    for seed in PILOT_SEEDS:
        if seed!=42: graph[f'pretrain-{seed}']['dependencies']=[anchor]
        for arm in DEVELOPMENT_ARMS:
            name=f'develop-{seed}-{arm}-250000'
            if name in graph and (seed,arm) not in ((42,'full'),(42,'same_info')):
                graph[name]['dependencies']=[anchor,f'pretrain-{seed}']
            graph[f'diagnostics-confirmation-{seed}-{arm}-250000']['dependencies']=[
                f'diagnostics-screen-{seed}-{arm}-250000']
    diagnostics=[]
    for seed in PILOT_SEEDS:
        for arm in ('full','same_info'):
            name=f'diagnostics-precision-{seed}-{arm}-250000'
            diagnostics.append(dict(name=name,kind='cpu',phase='development-precision',
                command=[str(python),'-X','utf8','train/interaction_review.py','--mode','precision',
                    '--study',str(study),'--seed',str(seed),'--arm',arm,'--workers','2'],
                dependencies=[f'diagnostics-screen-{seed}-{arm}-250000'],
                completion=str(study/f'reviews/precision/{seed}-{arm}-250000/completed.json')))
        diagnostics.append(dict(name=f'diagnostics-development-mechanism-{seed}',kind='cpu',phase='development-mechanism',
            command=[str(python),'-X','utf8','train/interaction_review.py','--mode','development-mechanism',
                '--study',str(study),'--seed',str(seed),'--workers','2'],
            dependencies=[f'diagnostics-screen-{seed}-full-250000'],
            completion=str(study/f'reviews/development-mechanism/seed-{seed}/completed.json')))
    freeze=graph['review-protocol-freeze']
    freeze['dependencies']=list(dict.fromkeys([*freeze['dependencies'],'review-pair-250000',
        'review-development-replication','review-small-matrix',*[j['name'] for j in diagnostics]]))
    # Independent value/mechanism work gets CPU priority without delaying any GPU training.
    return diagnostics+[j for j in jobs if j['name'] not in {d['name'] for d in diagnostics}]


def migrate_parallel_development(study):
    """User-authorized use of idle resources, without changing learner settings or formal standards."""
    import fcntl
    study=Path(study)
    with (study/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=study/'manifest.json'; manifest=json.loads(path.read_text())
        if manifest['format']!='interaction-study-v6' or manifest['root']!=str(ROOT):
            raise ValueError('Parallel development requires the original v6 study path.')
        if (study/'reviews/review-protocol-freeze/evidence.json').exists() or any(
                (study/f'training/seed-{seed}').exists() for seed in FORMAL_SEEDS):
            raise RuntimeError('Development execution must be registered before formal protocol freeze.')
        jobs={j['name']:j for j in manifest['jobs']}
        if not review_passed(jobs['review-pair-100000'],study):
            raise RuntimeError('The bounded pair technical check has not passed.')
        quota=len(manifest['cpu_affinity'])
        quota_path=Path('/sys/fs/cgroup/cpu.max')
        if quota_path.exists():
            amount,period=quota_path.read_text().split()
            if amount!='max': quota=min(quota,int(amount)//int(period))
        if quota<20: raise RuntimeError('Four training and two evaluation slots require 20 assigned CPUs.')
        current=source_hashes()
        changed={p:dict(before=manifest['code'].get(p),after=current.get(p))
            for p in set(current)|set(manifest['code']) if manifest['code'].get(p)!=current.get(p)}
        allowed={'scripts/run_interaction_study.py','train/interaction_review.py','scripts/build_interaction_figures.py'}
        if set(changed)-allowed or set(current)!=set(manifest['code']):
            raise RuntimeError('Parallel execution cannot modify the learner or training protocol.')
        running=[]
        for record_path in (study/'jobs').glob('*.json'):
            record=json.loads(record_path.read_text())
            if record.get('state')=='running' and process_identity(record.get('pid',0)) is not None:
                if jobs[record['name']]['kind']!='gpu' or not record['name'].startswith(('develop-','pretrain-')):
                    raise RuntimeError('Finish active evaluation jobs before revising diagnostic code.')
                running.append(dict(name=record['name'],pid=record['pid'],
                    process_identity=verify_running_job(jobs[record['name']],record)))
        python=next(j['command'][0] for j in manifest['jobs'] if j['name'].startswith('develop-'))
        manifest['jobs']=parallel_development_jobs(manifest['jobs'],study,python)
        manifest.setdefault('execution_revisions',[]).append(dict(time=time.time(),prior_manifest_sha256=digest(path),
            changes=changed,preserved_running=running,previous_limits=manifest['limits'].copy(),
            reason='User requested using idle compute to complete all evidence requirements. Collect the existing two-seed five-arm 250k development matrix concurrently, with independent precision and mechanism diagnostics. Preserve all scientific reviews before formal training.'))
        manifest['limits'].update(gpu=4,cpu=2)
        manifest.update(code=current,cpu_per_gpu=4,cpu_per_evaluation=2,
            development_execution=dict(version='parallel-bounded-v1',seeds=list(PILOT_SEEDS),
                arms=list(DEVELOPMENT_ARMS),endpoint_steps=250000,precision_states=64,
                precision_repetitions=16,mechanism_states=32,mechanism_repetitions=4))
        banks=manifest['candidate_protocol']['development_banks']
        manifest['candidate_protocol']['development_banks']=sorted(set(banks)|{404000000,405000000,406000000,407000000})
        manifest['candidate_protocol']['development_precision']=dict(states=64,repetitions=16,
            full_environment=True,bank='404M/405M',purpose='Independent Full/Same-info conditional-value trend before formal freeze.')
        write_json(path,manifest)


def migrate_control_isolation(study):
    """Retry only the two failed legacy controls through an exact evaluator split."""
    import fcntl
    study=Path(study)
    with (study/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=study/'manifest.json';manifest=json.loads(path.read_text())
        if manifest['format']!='interaction-study-v6' or manifest['root']!=str(ROOT) or not manifest.get('development_execution'):
            raise ValueError('Control isolation requires the registered parallel development study.')
        if (study/'reviews/review-protocol-freeze/evidence.json').exists():
            raise RuntimeError('Legacy control recovery is prohibited after protocol freeze.')
        current=source_hashes()
        changed={p:dict(before=manifest['code'].get(p),after=current.get(p))
            for p in set(current)|set(manifest['code']) if manifest['code'].get(p)!=current.get(p)}
        allowed={'train/train_interaction.py','scripts/run_interaction_study.py','scripts/build_interaction_figures.py'}
        if set(changed)-allowed or set(current)!=set(manifest['code']):
            raise RuntimeError('Only exact control isolation and execution accounting may change.')
        jobs={j['name']:j for j in manifest['jobs']};running=[];attempts={}
        for record_path in (study/'jobs').glob('*.json'):
            record=json.loads(record_path.read_text())
            if record.get('state')=='running' and process_identity(record.get('pid',0)) is not None:
                if record['name'] in ('develop-42-no_peer-250000','develop-42-model_value-250000'):
                    raise RuntimeError('The failed control must have exited before retrying it.')
                running.append(dict(name=record['name'],pid=record['pid'],
                    process_identity=verify_running_job(jobs[record['name']],record)))
        for arm in ('no_peer','model_value'):
            name=f'develop-42-{arm}-250000';checkpoint=study/f'training/seed-42/{arm}/resume.pth'
            record=json.loads((study/f'reviews/bootstrap-repair/42-{arm}/completed.json').read_text())
            if record.get('complete') or record.get('passed') or digest(checkpoint)!=record['previous_checkpoint_sha256']:
                raise RuntimeError('Recovery must preserve the unchanged checkpoint of the failed rebuild.')
            if record['step']!=20000 or not record['actor_unchanged']:
                raise RuntimeError('Only the two retained 20k development controls are covered.')
            attempts[arm]=dict(checkpoint_sha256=digest(checkpoint),
                failed_job=json.loads((study/f'jobs/{name}.json').read_text()))
            command=jobs[name]['command']
            if '--recondition-bootstrap-if-needed' not in command: raise RuntimeError('Unexpected prior recovery command.')
            command[command.index('--recondition-bootstrap-if-needed')]='--isolate-bootstrap-if-needed'
        manifest.setdefault('execution_revisions',[]).append(dict(time=time.time(),prior_manifest_sha256=digest(path),
            changes=changed,preserved_running=running,failed_reconditioning=attempts,
            reason='The two 20k legacy controls failed the reconstruction-specific error-halving test. Preserve their exact actor/critic/target and all training state, verify identical calibration, and isolate subsequent real-TD bootstrap updates. Retain failed reconstruction costs; no efficacy assessment is implied.'))
        manifest['code']=current
        write_json(path,manifest)


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
    if manifest.get('superseded_by'):
        raise RuntimeError('This development protocol was superseded; resume '+manifest['superseded_by'])
    if manifest['root']!=str(ROOT): raise RuntimeError('Run the manifest in its original project path.')
    with (study/'runner.lock').open('a') as lock:
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
                    required=(2 if any((study/'jobs').glob('*.json')) else
                              8 if manifest['format'] == 'interaction-performance-v1' else 32)
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','prepare-composed-revision','prepare-performance-search','performance-screen','run','status','migrate-bootstrap',
                                       'migrate-technical-reviews','migrate-evidence-standard','migrate-parallel-development',
                                       'migrate-control-isolation','decide-review'])
    parser.add_argument('--study',type=Path,default=ROOT/'train/experiments/ia-crrl-20261007')
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--previous-study',type=Path)
    parser.add_argument('--candidate')
    parser.add_argument('--checkpoint-step', type=int, default=0)
    parser.add_argument('--vrx-activate',type=Path,default=ROOT/'vrx_ws/activate.bash')
    parser.add_argument('--review')
    parser.add_argument('--decision',choices=['continue','hold'])
    parser.add_argument('--rationale')
    parser.add_argument('--evidence-sha256')
    parser.add_argument('--assessment',choices=['technical_ready','promising','inconclusive','unsupported','necessary','frozen','supported'])
    parser.add_argument('--findings-file',type=Path)
    parser.add_argument('--adopt-running', action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare': prepare(args.study.resolve(),args.python,args.vrx_activate.resolve())
    elif args.mode=='prepare-composed-revision':
        if args.previous_study is None: parser.error('--previous-study is required for a scientific revision.')
        prepare_composed_revision(args.study,args.previous_study,args.python,args.vrx_activate.resolve())
    elif args.mode=='prepare-performance-search':
        if args.previous_study is None: parser.error('--previous-study is required.')
        prepare_performance_search(args.study, args.previous_study, args.python)
    elif args.mode=='performance-screen':
        if args.candidate is None: parser.error('--candidate is required.')
        performance_screen(args.study, args.candidate, args.checkpoint_step)
    elif args.mode=='run': run(args.study.resolve(),args.adopt_running)
    elif args.mode=='migrate-bootstrap': migrate_bootstrap(args.study.resolve())
    elif args.mode=='migrate-technical-reviews': migrate_technical_reviews(args.study.resolve())
    elif args.mode=='migrate-evidence-standard': migrate_evidence_standard(args.study.resolve())
    elif args.mode=='migrate-parallel-development': migrate_parallel_development(args.study.resolve())
    elif args.mode=='migrate-control-isolation': migrate_control_isolation(args.study.resolve())
    elif args.mode=='decide-review':
        decide_review(args.study.resolve(),args.review,args.decision,args.rationale,args.evidence_sha256,args.assessment,
            findings=json.loads(args.findings_file.read_text(encoding='utf-8')) if args.findings_file else None)
    else: print((args.study/'status.json').read_text(encoding='utf-8'))


if __name__=='__main__': main()
