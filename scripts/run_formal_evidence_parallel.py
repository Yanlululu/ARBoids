"""Server execution adapter for the unchanged, frozen formal evidence study."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'train/experiments/formal-evidence-20261004'
TRAIN_PYTHON = '/root/autodl-tmp/ARBoids/.venv/bin/python'
VRX_ACTIVATE = '/root/autodl-tmp/ARBoids/vrx_ws/activate.bash'
ASSETS = '/root/autodl-tmp/ARBoids/.vrx-assets'
ORIGINAL_ROOT = 'C:/Users/yanluyao/.codex/worktrees/7042/ARBoids'
ORIGINAL_BASELINE = 'D:/ARBoids/train/experiments/paper-parameters-seed42-20260921/adares1.pth'
spec = importlib.util.spec_from_file_location('frozen_runner', ROOT/'scripts/run_formal_evidence.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def translate(value):
    value = str(value).replace('\\', '/')
    prefixes = [(ORIGINAL_BASELINE, str(ROOT/'inputs/arboids-seed42.pth')),
                ('/mnt/d/'+ORIGINAL_BASELINE[3:], str(ROOT/'inputs/arboids-seed42.pth')),
                (ORIGINAL_ROOT, str(ROOT)),
                ('/mnt/c/'+ORIGINAL_ROOT[3:], str(ROOT))]
    for source, target in prefixes:
        if value == source or value.startswith(source+'/'):
            return target+value[len(source):]
    return value


def option(command, name):
    return command[command.index(name)+1]


def set_option(command, name, value):
    command[command.index(name)+1] = str(value)


def build_manifest(original, migration):
    result = copy.deepcopy(original)
    result['source_sha256'] = {key.replace('\\','/'):value for key,value in original['source_sha256'].items()}
    result['frozen_inputs'] = {translate(key):value for key,value in original['frozen_inputs'].items()}
    result['migration'] = migration
    result['server_adapter_sha256'] = frozen.digest(__file__)
    producers = {}
    for original_order, job in enumerate(result['jobs']):
        job['original_order'] = original_order
        for field in ('result','expected_checkpoint'):
            if field in job:
                job[field] = translate(job[field])
        command = [translate(item) for item in job['command']]
        if job['category']=='vrx':
            command = command[command.index('vrx/run_experiment.py'):]
            set_option(command,'--assets-dir',ASSETS)
        else:
            command[0] = TRAIN_PYTHON
        job['command'] = command
        job['execution_group'] = 'vrx' if job['category']=='vrx' else ('training' if job['category']=='training' else 'evaluation')
        if job['category']=='training':
            directory = str(Path(job['result']).parent).replace('\\','/')
            for file in ('actor.pth','resume.pth','policy.pth'):
                producers[directory+'/'+file] = job['name']
    for job in result['jobs']:
        command=job['command']
        inputs = [option(command,flag).replace('\\','/') for flag in ('--source','--checkpoint') if flag in command]
        job['dependencies'] = sorted({producers[path] for path in inputs if path in producers})
    return result


def verify(manifest):
    for relative, expected in manifest['source_sha256'].items():
        if frozen.digest(ROOT/relative)!=expected:
            raise RuntimeError('Frozen source changed: '+relative)
    for path, expected in manifest['frozen_inputs'].items():
        if frozen.digest(path)!=expected:
            raise RuntimeError('Frozen input changed: '+path)
    if frozen.digest(__file__)!=manifest['server_adapter_sha256']:
        raise RuntimeError('Server execution adapter changed.')


def make_slots():
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)<20:
        raise RuntimeError('The fixed server plan requires 20 available logical CPUs.')
    slots=[]
    for index in range(3):
        slots.append(dict(name=f'train-{index}',group='training',cpus=cpus[index*4:index*4+4]))
    for index in range(2):
        slots.append(dict(name=f'vrx-{index}',group='vrx',cpus=cpus[12+index*3:15+index*3],domain=100+index))
    for index in range(2):
        slots.append(dict(name=f'eval-{index}',group='evaluation',cpus=[cpus[18+index]]))
    return slots


def priority(job):
    if job['category']=='training':
        return (0 if job['arm']=='pretrain' else 1,job['original_order'])
    return (0 if job['name'].startswith('primary-') else 1,job['original_order'])


def command_for(job, slot, attempt):
    command=list(job['command'])
    result=Path(job['result'])
    if job['category']=='vrx':
        if attempt>1:
            set_option(command,'--output-dir',result.parent)
            set_option(command,'--run-id',f'infrastructure-attempt-{attempt:02d}')
            result=result.parent/f'infrastructure-attempt-{attempt:02d}/result.json'
        shell=(f'set -e\nsource {shlex.quote(VRX_ACTIVATE)}\n'
               f'export ROS_DOMAIN_ID={slot["domain"]}\n'
               'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg\n'
               'exec python -X utf8 -u "$@"')
        command=['bash','-c',shell,'formal-vrx',*command]
    return ['taskset','-c',','.join(map(str,slot['cpus'])),*command],result


def startup_retry_allowed(result, log):
    return (result.get('passed') is False and result.get('control_steps')==0
            and 'outcome_code' not in result
            and result.get('error')=='TimeoutError: Waiting for all vessel poses and thrust bridge subscribers'
            and 'Network is unreachable' in log)


def save_progress(manifest, state, completed, active, failed, attempts, refresh_report=False):
    status=dict(state=state,runner_pid=os.getpid(),execution_host='remote-rtx4090',
                completed_jobs=len(completed),total_jobs=len(manifest['jobs']),
                current_job=', '.join(v['job']['name'] for v in active.values()) or None,
                active_jobs=[dict(name=v['job']['name'],pid=v['process'].pid,slot=v['slot'],attempt=v['attempt']) for v in active.values()],
                failed_jobs=failed,attempts=attempts,error=failed or None,updated_unix=time.time(),
                local_completed_jobs=len(manifest['migration']['local_completed_jobs']))
    frozen.write_json(STUDY/'status.json',status)
    if refresh_report:
        frozen.report(STUDY,manifest,status)
        with (STUDY/'report.md').open('a',encoding='utf-8') as file:
            file.write('\n## 服务器并发执行\n\n3 个训练槽位、2 个 VRX 槽位、2 个评估槽位；每个任务保持原完整预算。\n'
                       '本机已完成结果保留；远程 CPU/GPU 与并发负载不同，推理耗时按平台分层，不能混合声称单机性能。\n'
                       'VRX 每个槽位使用独立 ROS_DOMAIN_ID；实例另有独立 Gazebo 分区。\n')
        timing={}
        local=set(manifest['migration']['local_completed_jobs'])
        for job in manifest['jobs']:
            if job['category']!='vrx' or job['name'] not in completed:
                continue
            data=json.loads(Path(job['result']).read_text(encoding='utf-8'))
            key=('local' if job['name'] in local else 'remote-concurrent')+'/'+job['arm']
            timing.setdefault(key,[]).append(dict(seed=data['seed'],inference_seconds=data['inference_seconds'],
                control_seconds=data['control_seconds'],control_steps=data['control_steps'],
                control_deadline_misses=data['control_deadline_misses']))
        frozen.write_json(STUDY/'vrx-timing-by-platform.json',timing)


def run(manifest):
    import fcntl
    lock=(STUDY/'server-runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify(manifest)
    previous=json.loads((STUDY/'status.json').read_text(encoding='utf-8'))
    if previous.get('execution_host')=='remote-rtx4090':
        for active_job in previous.get('active_jobs',[]):
            if frozen.process_alive(active_job['pid']):
                raise RuntimeError('A previous experiment is still alive; refuse to duplicate it.')
    completed={job['name'] for job in manifest['jobs'] if frozen.valid_result(job)}
    attempts=previous.get('attempts',{})
    pending={job['name']:job for job in manifest['jobs'] if job['name'] not in completed}
    active={}
    failed={}
    slots=make_slots()
    history=STUDY/'server-execution.jsonl'
    last_report=0
    try:
        while pending or active:
            changed=False
            for slot_name, info in list(active.items()):
                code=info['process'].poll()
                if code is None:
                    continue
                info['log'].close()
                job=info['job']
                accepted=dict(job,result=str(info['result']))
                event=dict(job=job['name'],attempt=info['attempt'],returncode=code,finished_unix=time.time(),slot=info['slot'])
                if code==0 and frozen.valid_result(accepted):
                    if info['result']!=Path(job['result']):
                        old=json.loads(Path(job['result']).read_text(encoding='utf-8'))
                        recovered=json.loads(info['result'].read_text(encoding='utf-8'))
                        for field in ('initial_poses','seed','checkpoint_sha256','agility','setting'):
                            if recovered.get(field)!=old.get(field):
                                raise RuntimeError('Infrastructure retry changed '+field)
                        recovered['infrastructure_recovery']=dict(prior_result=old,accepted_attempt_result=str(info['result']))
                        frozen.write_json(Path(job['result']),recovered)
                    completed.add(job['name'])
                    event['valid']=True
                else:
                    result=json.loads(info['result'].read_text(encoding='utf-8')) if info['result'].exists() else {}
                    gazebo=info['result'].parent/'gazebo.log'
                    log=gazebo.read_text(encoding='utf-8',errors='replace') if gazebo.exists() else ''
                    if job['category']=='vrx' and info['attempt']<3 and startup_retry_allowed(result,log):
                        pending[job['name']]=job
                        event['retry']='zero-control-step network startup failure'
                    else:
                        failed[job['name']]=dict(returncode=code,result=str(info['result']),log=str(info['log_path']),error=result.get('error'))
                        event['valid']=False
                with history.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(event)+'\n')
                del active[slot_name]
                changed=True
            for slot in slots:
                if slot['name'] in active:
                    continue
                eligible=[job for job in pending.values() if job['execution_group']==slot['group'] and set(job['dependencies'])<=completed]
                if not eligible:
                    continue
                job=min(eligible,key=priority)
                verify(manifest)
                attempt=attempts.get(job['name'],0)+1
                if job['category']=='vrx' and attempt>1:
                    _, prior_path=command_for(job,slot,attempt-1)
                    prior=json.loads(prior_path.read_text(encoding='utf-8')) if prior_path.exists() else {}
                    prior_log=prior_path.parent/'gazebo.log'
                    diagnostics=prior_log.read_text(encoding='utf-8',errors='replace') if prior_log.exists() else ''
                    if attempt>3 or not startup_retry_allowed(prior,diagnostics):
                        failed[job['name']]=dict(error='Prior VRX attempt needs explicit review; no automatic replacement',result=str(prior_path))
                        del pending[job['name']]
                        changed=True
                        continue
                command,result=command_for(job,slot,attempt)
                if job['category']=='vrx' and result.parent.exists():
                    # Existing partial attempts are not overwritten or silently discarded.
                    failed[job['name']]=dict(error='Existing incomplete VRX attempt needs explicit review',result=str(result))
                    del pending[job['name']]
                    changed=True
                    continue
                log_path=STUDY/'logs'/f"{job['name']}.server-attempt-{attempt:02d}.log"
                log_path.parent.mkdir(parents=True,exist_ok=True)
                log=log_path.open('a',encoding='utf-8')
                environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
                process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[slot['name']]=dict(job=job,slot=slot,process=process,attempt=attempt,result=result,log=log,log_path=log_path)
                attempts[job['name']]=attempt
                del pending[job['name']]
                with history.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(job=job['name'],attempt=attempt,pid=process.pid,slot=slot,
                        command=command,started_unix=time.time(),concurrent_jobs=[v['job']['name'] for v in active.values()]))+'\n')
                changed=True
            if not active and pending:
                for name,job in pending.items():
                    failed[name]=dict(error='Required predecessor failed',dependencies=job['dependencies'])
                pending.clear()
                changed=True
            if changed or time.monotonic()-last_report>15:
                save_progress(manifest,'running',completed,active,failed,attempts,refresh_report=changed)
                last_report=time.monotonic()
            if active:
                time.sleep(2)
        save_progress(manifest,'failed' if failed else 'complete',completed,active,failed,attempts,True)
    except BaseException as error:
        failed['scheduler']=dict(error=f'{type(error).__name__}: {error}')
        # Keep child identifiers for explicit recovery instead of duplicating live work.
        save_progress(manifest,'failed',completed,active,failed,attempts,True)
        raise
    finally:
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','status'))
    args=parser.parse_args()
    path=STUDY/'manifest.remote.json'
    if args.action=='prepare':
        if path.exists():
            raise RuntimeError('Remote manifest already frozen.')
        original=json.loads((STUDY/'manifest.json').read_text(encoding='utf-8'))
        migration=json.loads((STUDY/'migration.json').read_text(encoding='utf-8'))
        manifest=build_manifest(original,migration)
        verify(manifest)
        frozen.write_json(path,manifest)
        print(json.dumps(dict(jobs=len(manifest['jobs']),local_completed=len(migration['local_completed_jobs']),slots=make_slots())))
    elif args.action=='run':
        run(json.loads(path.read_text(encoding='utf-8')))
    else:
        print((STUDY/'status.json').read_text(encoding='utf-8'))


if __name__=='__main__':
    main()
