"""Sequential local evidence suite. Frozen manifests, real completion checks, resume."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'train/experiments/formal-evidence-20261004'
FROZEN = ROOT/'train/experiments/predictive-mappo-performance-stable-best-42/candidate.pth'
BASELINE = Path('D:/ARBoids/train/experiments/paper-parameters-seed42-20260921/adares1.pth')
TRAIN_SEEDS = [101, 202, 303, 404, 505]
CONTINUATION = 7538623


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda: file.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def interval(k,n):
    z=1.959963984540054
    center=(k/n+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half),min(1.,center+half)]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def linux(path):
    p = str(Path(path).resolve()).replace('\\', '/')
    return '/mnt/'+p[0].lower()+p[2:] if len(p)>1 and p[1]==':' else p


def process_alive(pid):
    if not pid:
        return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:
            return False
        code=wintypes.DWORD()
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        try:
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid),0)
        return True
    except ProcessLookupError:
        return False


def make_manifest(directory, python):
    jobs = []
    def add(name, command, result, category, **details):
        jobs.append(dict(name=name, command=[str(v) for v in command], result=str(result), category=category, **details))
    def evaluate(name, checkpoint, kind, result, first_seed, episodes, **conditions):
        command = [python, '-X', 'utf8', '-u', 'train/formal_study_evaluation.py', '--checkpoint', checkpoint,
                   '--kind', kind, '--first-seed', first_seed, '--episodes', episodes, '--output', result]
        for key, value in conditions.items():
            command.extend(['--'+key.replace('_','-'), value])
        add(name, command, result/'summary.json', 'evaluation', expected_checkpoint=str(checkpoint))
    vrx_rng = random.Random(20261004)
    # First obtain the requested deployment comparison; training cannot consume VRX resources concurrently.
    for i in range(100):
        arms = [('baseline', BASELINE, 'AdaRes'), ('candidate', FROZEN, 'MAPPO')]
        vrx_rng.shuffle(arms)
        for label, checkpoint, controller in arms:
            name = f'vrx-{i:03d}-{label}'
            result_dir = directory/'vrx'/name
            command = ['wsl', '-d', 'ARBoids-22.04', '-u', 'root', '--cd', linux(ROOT), '--exec', 'bash', '-c',
                       'set -e\nsource /opt/arboids-runtime/vrx_ws/activate.bash\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\nexec python -X utf8 -u "$@"',
                       'formal-vrx', 'vrx/run_experiment.py', '--checkpoint', linux(checkpoint),
                       '--controller', controller, '--setting', 1, '--agility', 2.25, '--seed', 87000000+i,
                       '--termination-rule', 'paper', '--headless', '--synchronize-feedback',
                       '--assets-dir', '/mnt/d/ARBoids/.vrx-assets', '--output-dir', linux(directory/'vrx'), '--run-id', name]
            add(name, command, result_dir/'result.json', 'vrx', arm=label, scene_seed=87000000+i,
                expected_sha256=digest(checkpoint))
    for i in range(100):
        impairments=[('delay-200ms','--message-delay-steps',1),('delay-400ms','--message-delay-steps',2),
                     ('loss-10pct','--message-drop-probability',.1),('loss-30pct','--message-drop-probability',.3)]
        vrx_rng.shuffle(impairments)
        for label, flag, value in impairments:
            name=f'vrx-stress-{i:03d}-{label}'
            reference=next(j for j in jobs if j['name']==f'vrx-{i:03d}-candidate')
            command=list(reference['command'])
            command[command.index('--run-id')+1]=name
            command += [flag, value]
            add(name,command,directory/'vrx'/name/'result.json','vrx',arm=label,scene_seed=87000000+i,
                expected_sha256=digest(FROZEN))
    # The old adaptive-development candidate is only a frozen transfer/robustness case study.
    conditions = [('agility-1.5', dict(agility=1.5)), ('agility-2.25', dict(agility=2.25)),
                  ('agility-2.5', dict(agility=2.5)), ('agility-3.0', dict(agility=3.)),
                  ('team-2', dict(defenders=2)), ('team-4', dict(defenders=4)),
                  ('direct-attacker', dict(attacker='direct')), ('wide-apf-attacker', dict(attacker='wide-apf'))]
    for i, (label, condition) in enumerate(conditions):
        for arm, checkpoint, kind in [('baseline', BASELINE, 'sac'), ('candidate', FROZEN, 'mappo')]:
            out = directory/'frozen-generalization'/label/arm
            evaluate(f'frozen-{label}-{arm}', checkpoint, kind, out, 88000000+i*10000, 500, **condition)
    impairments = [('delay-200ms', dict(delay_steps=1)), ('delay-400ms', dict(delay_steps=2)),
                   ('loss-10pct', dict(drop_probability=.1)), ('loss-30pct', dict(drop_probability=.3))]
    evaluate('frozen-communication-reference', FROZEN, 'mappo', directory/'communication'/'reliable', 89000000, 500)
    for label, condition in impairments:
        evaluate('frozen-'+label, FROZEN, 'mappo', directory/'communication'/label, 89000000, 500, **condition)
    # Five genuinely independent SAC initializations; variants are blocked by that warm start.
    for seed_index, seed in enumerate(TRAIN_SEEDS):
        run = directory/f'seed-{seed}'
        def training(arm, source, steps, scene_seed):
            out = run/arm
            command = [python, '-X', 'utf8', '-u', 'train/formal_study_training.py', '--arm', arm, '--seed', seed,
                       '--scene-seed', scene_seed, '--steps', steps, '--output', out, '--device', 'cuda:0', '--workers', 2]
            if source is not None:
                command.extend(['--source', source])
            add(f'train-{seed}-{arm}', command, out/'completed.json', 'training', arm=arm, training_seed=seed,
                nominal_steps=steps, expected_phase='pretrain' if arm=='pretrain' else 'continuation')
        warm_scenes = 100000000+seed_index*40000000
        fine_scenes = 110000000+seed_index*40000000
        training('pretrain', None, 1000000, warm_scenes)
        evaluate(f'primary-{seed}-pretrain', run/'pretrain/actor.pth', 'sac', run/'primary/pretrain',
                 80000000+seed_index*10000, 2000)
        # A seeded order for trainable MAPPO ablations; all use identical reset streams/budgets.
        arms = ['no_prediction', 'no_joint', 'fixed_gain', 'full', 'arboids']
        random.Random(seed).shuffle(arms)
        for arm in arms:
            training(arm, run/('pretrain/resume.pth' if arm=='arboids' else 'pretrain/actor.pth'), CONTINUATION, fine_scenes)
            ckpt = run/arm/('actor.pth' if arm=='arboids' else 'policy.pth')
            evaluate(f'primary-{seed}-{arm}', ckpt, 'sac' if arm=='arboids' else 'mappo',
                     run/'primary'/arm, 80000000+seed_index*10000, 2000)
            if arm=='arboids':
                training('fixed_rule', ckpt, CONTINUATION, fine_scenes)
                evaluate(f'primary-{seed}-fixed_rule', run/'fixed_rule/policy.pth', 'mappo',
                         run/'primary/fixed_rule', 80000000+seed_index*10000, 2000)
        for i, (label, condition) in enumerate(conditions):
            for arm in ('arboids', 'full'):
                ckpt = run/arm/('actor.pth' if arm=='arboids' else 'policy.pth')
                evaluate(f'generalization-{seed}-{label}-{arm}', ckpt, 'sac' if arm=='arboids' else 'mappo',
                         run/'generalization'/label/arm, 81000000+seed_index*100000+i*10000, 500, **condition)
    # Record unsupported scales explicitly, without disguising infrastructure errors as task failures.
    command = [python, '-X', 'utf8', '-u', 'train/formal_study_evaluation.py', '--checkpoint', FROZEN,
               '--kind', 'mappo', '--first-seed', 90000000, '--episodes', 1, '--defenders', 5,
               '--output', directory/'scaling/team-5']
    add('scale-limit-team-5', command, directory/'scaling/team-5/unsupported.json', 'unsupported')
    sources = sorted(set([*ROOT.joinpath('train').rglob('*.py'), *ROOT.joinpath('vrx').glob('*.py'),
                          ROOT/'train/configs/formal-mappo.yaml', ROOT/'train/configs/paper-parameters.yaml', Path(__file__)]))
    sources = [p for p in sources if not any(part in ('experiments','__pycache__') for part in p.parts)]
    history_files=0
    # New evaluation ranges are entirely outside prior development/audit scenes.
    for root in (ROOT/'train/experiments',Path('D:/ARBoids/train/experiments')):
        for path in root.rglob('*.csv'):
            if directory in path.parents:
                continue
            with path.open(encoding='utf-8-sig',newline='') as file:
                reader=csv.DictReader(file)
                if 'seed' not in (reader.fieldnames or []):
                    continue
                history_files += 1
                for row in reader:
                    value=row.get('seed','')
                    if value and 80000000 <= int(float(value)) <= 90000000:
                        raise ValueError(f'Proposed evaluation range overlaps prior data: {path}')
    return dict(version=1, study='formal-evidence-20261004', created_local_date='2026-10-04',
                training_seeds=TRAIN_SEEDS, pretrain_steps=1000000, continuation_steps=CONTINUATION,
                budget_unit='team environment transitions; completed final episode overshoot <=299 per phase',
                evaluation_selection='final checkpoint only; no tuning or early stopping from evaluation',
                fixed_rule='same joint predictor, gain=1, no learned correction, applied to equal-budget continued SAC',
                communication='per directed-link packet delay/loss; current local observation; reliable startup snapshot',
                limitations=['Five independent training seeds, not five evaluations of seed 42.',
                             'Fresh frozen recipe; historical adaptive tuning is not replayed.',
                             'APF, direct approach and wider APF repulsion; no trained adversary evidence yet.',
                             'Prediction uses synchronous messages in primary tests; link faults are emulated.',
                             'VRX is frozen candidate vs original parameter rerun, not a five-training-seed VRX study.'],
                source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
                historical_seed_csv_files_checked=history_files,
                frozen_inputs={str(FROZEN):digest(FROZEN), str(BASELINE):digest(BASELINE)}, jobs=jobs)


def valid_result(job):
    path = Path(job['result'])
    if not path.exists():
        return False
    result = json.loads(path.read_text(encoding='utf-8'))
    if job['category']=='unsupported':
        return result.get('measured_task_failure') is False
    if job['category']=='vrx':
        if not result.get('passed') or result.get('checkpoint_sha256') != job['expected_sha256']:
            return False
        return result.get('seed') == job['scene_seed']
    if not result.get('completed'):
        return False
    if job['category']=='training':
        checkpoint = Path(result['checkpoint'])
        if not checkpoint.exists() or digest(checkpoint) != result['checkpoint_sha256']:
            return False
        if job['arm'] != 'fixed_rule' and not 0 <= result['actual_phase_steps']-job['nominal_steps'] <= 299:
            return False
        return result['seed']==job['training_seed']
    return digest(job['expected_checkpoint']) == result['checkpoint_sha256']


def report(directory, manifest, status):
    lines = ['# 正式证据实验进度', '', f"状态：{status['state']}；完成 {status['completed_jobs']}/{len(manifest['jobs'])} 个任务。", '',
             '以下只汇总已实际完成的结果；排队中、运行中和基础设施失败均不算完成。', '',
             '## 冻结模型 VRX 对照', '', '| 模型 | 已完成回合 | 成功 | 碰撞 | 突破 |', '|---|---:|---:|---:|---:|']
    vrx_results={}
    for arm in ('baseline','candidate','delay-200ms','delay-400ms','loss-10pct','loss-30pct'):
        rows=[]
        for job in manifest['jobs']:
            if job['category']=='vrx' and job['arm']==arm and valid_result(job):
                rows.append(json.loads(Path(job['result']).read_text(encoding='utf-8')))
        if rows:
            vrx_results[arm]=dict(episodes=len(rows), success_count=sum(r['success'] for r in rows),
                collision_count=sum(r['defender_collision'] for r in rows),
                breach_count=sum(r['outcome_code']==1 for r in rows),
                mean_episode_inference_seconds=statistics.mean(r['inference_seconds']['mean'] for r in rows),
                maximum_inference_seconds=max(r['inference_seconds']['maximum'] for r in rows),
                control_deadline_misses=sum(r['control_deadline_misses'] for r in rows))
            for metric in ('success','collision','breach'):
                vrx_results[arm][metric+'_wilson95']=interval(vrx_results[arm][metric+'_count'],len(rows))
            lines.append(f"| {arm} | {len(rows)}/100 | {sum(r['success'] for r in rows)/len(rows):.3%} | "
                         f"{sum(r['defender_collision'] for r in rows)/len(rows):.3%} | "
                         f"{sum(r['outcome_code']==1 for r in rows)/len(rows):.3%} |")
        else:
            lines.append(f'| {arm} | 0/100 | 待完成 | 待完成 | 待完成 |')
    paired=[]
    for i in range(100):
        paths=[directory/'vrx'/f'vrx-{i:03d}-{arm}'/'result.json' for arm in ('baseline','candidate')]
        if all(path.exists() for path in paths):
            a,b=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
            if a.get('passed') and b.get('passed'):
                if a['initial_poses'] != b['initial_poses']:
                    raise RuntimeError('Paired VRX initial poses differ.')
                paired.append(dict(seed=a['seed'], baseline_success=int(a['success']),candidate_success=int(b['success']),
                    baseline_collision=int(a['defender_collision']),candidate_collision=int(b['defender_collision']),
                    baseline_breach=int(a['outcome_code']==1),candidate_breach=int(b['outcome_code']==1)))
    write_json(directory/'vrx-paired-summary.json',dict(complete=len(vrx_results)==6 and all(r['episodes']==100 for r in vrx_results.values()),
                                                      primary_complete=len(paired)==100,matched_pairs=len(paired),
                                                      arms=vrx_results,paired=paired))
    lines.extend(['', '## 五种子主要对照', '', '统计单位为独立训练种子；完整五种子结果才能形成算法稳定性结论。', '',
                  '| 方法 | 已完成种子 | 成功率均值 | 碰撞率均值 | 突破率均值 |', '|---|---:|---:|---:|---:|'])
    summaries = {}
    seed_statistics={}
    for arm in ('pretrain','arboids','no_prediction','fixed_rule','no_joint','fixed_gain','full'):
        rows = []
        for seed in TRAIN_SEEDS:
            path=directory/f'seed-{seed}/primary/{arm}/summary.json'
            if path.exists():
                rows.append(json.loads(path.read_text(encoding='utf-8')))
        summaries[arm]=rows
        if rows:
            seed_statistics[arm]=dict(completed_training_seeds=len(rows),complete=len(rows)==5)
            for metric in ('success','collision','breach'):
                rates=[r[metric+'_rate'] for r in rows]
                seed_statistics[arm][metric]=dict(mean=statistics.mean(rates),sample_sd=statistics.stdev(rates) if len(rates)>1 else None,
                                                 minimum=min(rates),maximum=max(rates),rates=rates)
            means=[statistics.mean(r[k+'_rate'] for r in rows) for k in ('success','collision','breach')]
            lines.append(f"| {arm} | {len(rows)}/5 | "+' | '.join(f'{x:.3%}' for x in means)+' |')
        else:
            lines.append(f'| {arm} | 0/5 | 待完成 | 待完成 | 待完成 |')
    write_json(directory/'training-seed-statistics.json',seed_statistics)
    comparisons={}
    if len(summaries['full'])==5:
        for arm, rows in summaries.items():
            if arm=='full' or len(rows)!=5:
                continue
            comparisons[arm]={}
            for metric in ('success','collision','breach'):
                delta=[a[metric+'_rate']-b[metric+'_rate'] for a,b in zip(summaries['full'], rows)]
                mean, sd=statistics.mean(delta),statistics.stdev(delta)
                half=2.7764451051977987*sd/math.sqrt(5)
                comparisons[arm][metric]=dict(per_training_seed_differences=delta, mean=mean,
                    sample_sd=sd, t95=[mean-half,mean+half], training_seeds=5,
                    interval_scope='approximate paired t interval across training seeds; not multiplicity adjusted')
        write_json(directory/'seed-level-comparisons.json', comparisons)
    lines.extend(['', '## 泛化与通信', '', '逐场景结果、耗时和条件见 frozen-generalization、communication 及各 seed 的 generalization 目录。',
                  '2/4 艇测试直接使用 3 艇训练的共享 Actor，不重训、不使用或调整 Critic。5 艇超过当前联合枚举上限，单独报告能力限制。',
                  '每个训练阶段保持完整预算；最终完整回合至多超出 299 步，实际步数保存在 completed.json。', '',
                  f"当前任务：{status.get('current_job') or '无'}", f"最近错误：{status.get('error') or '无'}", ''])
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args, manifest):
    directory = args.output
    lock = directory/'runner.lock'
    # OS-held advisory lock is released on crash; stale lock files do not imply a live run.
    lock_file = lock.open('a+b')
    if os.name=='nt':
        import msvcrt
        lock_file.seek(0)
        if lock_file.read(1)==b'':
            lock_file.write(b'0'); lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((directory/'status.json').read_text(encoding='utf-8'))
    if process_alive(previous.get('child_pid')):
        lock_file.close()
        raise RuntimeError('A previously launched child is still alive; do not duplicate its experiment.')
    status=dict(state='running', runner_pid=os.getpid(), completed_jobs=0, current_job=None)
    try:
        for job in manifest['jobs']:
            if valid_result(job):
                status['completed_jobs'] += 1
                continue
            for relative, expected in manifest['source_sha256'].items():
                if digest(ROOT/relative) != expected:
                    raise RuntimeError(f'Frozen source changed: {relative}; do not mix experimental recipes.')
            for path, expected in manifest['frozen_inputs'].items():
                if digest(path) != expected:
                    raise RuntimeError('Frozen input checkpoint changed.')
            if job['category']=='vrx' and Path(job['result']).parent.exists():
                raise RuntimeError(f"Incomplete VRX directory for {job['name']}; retain failure and prepare an explicit retry.")
            status.update(current_job=job['name'], job_started_unix=time.time(), child_pid=None, error=None)
            write_json(directory/'status.json', status)
            report(directory,manifest,status)
            log_path = directory/'logs'/f"{job['name']}.log"
            log_path.parent.mkdir(exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
            with log_path.open('a',encoding='utf-8') as log:
                command = job['command']
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                status['child_pid']=process.pid
                write_json(directory/'status.json',status)
                returncode=process.wait()
            if returncode or not valid_result(job):
                raise RuntimeError(f"{job['name']} failed its completion check (exit={returncode}); see {log_path.name}")
            status['completed_jobs'] += 1
            status['child_pid']=None
            status['last_completed_job']=job['name']
            write_json(directory/'status.json',status)
            report(directory,manifest,status)
        status.update(state='complete',current_job=None,child_pid=None,finished_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=f'{type(error).__name__}: {error}',finished_unix=time.time())
        raise
    finally:
        write_json(directory/'status.json',status)
        report(directory,manifest,status)
        lock_file.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','status'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--python',default='D:/ARBoids/.venv/Scripts/python.exe')
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'manifest.json'
    if args.action=='prepare':
        if path.exists():
            raise ValueError('A frozen manifest already exists; do not overwrite it.')
        manifest=make_manifest(args.output,args.python)
        write_json(path,manifest)
        status=dict(state='prepared',completed_jobs=0,current_job=None)
        write_json(args.output/'status.json',status)
        report(args.output,manifest,status)
        print(json.dumps(dict(jobs=len(manifest['jobs']),manifest_sha256=digest(path))),flush=True)
    elif args.action=='run':
        run(args,json.loads(path.read_text(encoding='utf-8')))
    else:
        print((args.output/'status.json').read_text(encoding='utf-8'))


if __name__=='__main__':
    main()
