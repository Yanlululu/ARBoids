"""Serial, resource-aware execution of a concrete experiment dependency matrix.

Each job declares argv, dependencies and actual JSON/file completion checks.
No training success is inferred from launch, a zero exit, or a file name alone.
"""
import argparse,ctypes,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(value,path):
    path=Path(path);tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def memory():
    if os.name=='nt':
        class Status(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in
                ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended']]
        s=Status();s.length=ctypes.sizeof(s)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):raise OSError('Memory status unavailable')
        ram=s.avail_phys/2**30;commit=s.avail_page/2**30
    else:
        info={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        ram=info['MemAvailable']/2**30;commit=ram+info['SwapFree']/2**30
    return {'available_ram_gib':ram,'commit_headroom_gib':commit}


def process_memory(process):
    if os.name=='nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(x,ctypes.c_size_t) for x in
              ['peak_rss','rss','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
        c=Counters();c.cb=ctypes.sizeof(c)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(process._handle)),ctypes.byref(c),c.cb):
            raise OSError('Child memory status unavailable')
        return c.peak_rss/2**30
    status=Path(f'/proc/{process.pid}/status')
    if not status.exists():return 0.
    return max([int(line.split()[1])/2**20 for line in status.read_text().splitlines() if line.startswith(('VmRSS:','VmHWM:'))] or [0.])


def check_outputs(job):
    hashes={}
    for check in job['outputs']:
        path=Path(check['path'])
        if not path.is_file():return None
        if 'equals' in check:
            value=json.loads(path.read_text(encoding='utf-8'))
            for key,expected in check['equals'].items():
                actual=value
                for part in key.split('.'):actual=actual[part]
                if actual!=expected:return None
        hashes[str(path)]=digest(path)
    return hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--wait',action='store_true');p.add_argument('--start-ram-gib',type=float,default=3.75)
    p.add_argument('--stop-ram-gib',type=float,default=3.0);p.add_argument('--commit-gib',type=float,default=4.)
    a=p.parse_args();path=Path(a.manifest).resolve();spec=json.loads(path.read_text(encoding='utf-8'))
    for filename,expected in spec.get('code_sha256',{}).items():
        if digest(filename)!=expected:raise ValueError('Experiment implementation changed: '+filename)
    jobs=spec['jobs'];ids=[j['job_id'] for j in jobs]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate job IDs')
    before=set()
    for job in jobs:
        if not set(job.get('depends_on',[]))<=before:raise ValueError('Jobs must be in dependency order')
        before.add(job['job_id'])
    if a.dry_run:
        for job in jobs:print(json.dumps({'job_id':job['job_id'],'argv':job['argv'],'depends_on':job.get('depends_on',[]),'output_checks':job['outputs']},ensure_ascii=False))
        return
    state_path=path.with_name('matrix_status.json');identity=digest(path)
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'manifest_sha256':identity,'jobs':{}}
    if state['manifest_sha256']!=identity:raise ValueError('Experiment matrix changed; create a new development identity')
    state['manager_pid']=os.getpid();state['completed']=False
    for job in jobs:
        jid=job['job_id'];record=state['jobs'].get(jid,{})
        if record.get('status')=='completed':
            if check_outputs(job)!=record['output_sha256']:raise ValueError('Completed output changed: '+jid)
            continue
        while True:
            resource=memory();state['resources']=resource
            if resource['available_ram_gib']>=a.start_ram_gib and resource['commit_headroom_gib']>=a.commit_gib:break
            state['status']='waiting_for_memory';state['next_job']=jid;atomic(state,state_path)
            print(json.dumps({'status':state['status'],'next_job':jid,**resource}),flush=True)
            if not a.wait:return
            time.sleep(20)
        for dependency in job.get('depends_on',[]):
            previous=state['jobs'].get(dependency)
            if not previous or previous['status']!='completed':raise ValueError('Dependency not complete '+dependency)
        for filename,expected in spec.get('code_sha256',{}).items():
            if digest(filename)!=expected:raise ValueError('Implementation changed during experiment: '+filename)
        attempt=len(record.get('attempts',[]))+1;stop=path.with_name(jid+'.stop')
        if stop.exists():stop.unlink()
        argv=[sys.executable,'-X','utf8',*job['argv']]
        if job.get('cooperative_stop'):argv+=['--stop-file',str(stop)]
        if job.get('resume_marker') and Path(job['resume_marker']).exists():argv+=['--resume']
        log=path.with_name(f'{jid}.attempt{attempt}.log')
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        state['status']='running';state['next_job']=jid
        entry={'status':'running','started_at':time.time(),'argv':argv,'log':str(log),'resources_start':resource}
        record.setdefault('attempts',[]).append(entry);state['jobs'][jid]=record
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(argv,cwd=spec['cwd'],env=env,stdout=stream,stderr=subprocess.STDOUT)
            entry['pid']=process.pid;atomic(state,state_path);low=0;requested=None;peak_rss=0
            while process.poll() is None:
                time.sleep(2);resource=memory();state['resources']=resource
                peak_rss=max(peak_rss,process_memory(process))
                low=low+1 if resource['available_ram_gib']<a.stop_ram_gib else 0
                if low>=3 or resource['available_ram_gib']<1.5 or resource['commit_headroom_gib']<2.:
                    if requested is None:
                        stop.write_text('memory pressure',encoding='utf-8');requested=time.time()
                    if not job.get('cooperative_stop') or time.time()-requested>60:process.terminate()
                atomic(state,state_path)
        entry.update(exit_code=process.returncode,finished_at=time.time(),peak_process_ram_gib=peak_rss)
        outputs=check_outputs(job)
        if process.returncode==0 and outputs is not None:
            record.update(status='completed',output_sha256=outputs);entry['status']='completed'
        else:
            record['status']=entry['status']='resource_stopped' if requested is not None else 'failed'
            state['status']=record['status'];atomic(state,state_path)
            print(json.dumps({'job_id':jid,'status':record['status'],'log':str(log)}),flush=True)
            return
        atomic(state,state_path);print(json.dumps({'job_id':jid,'status':'completed','peak_process_ram_gib':peak_rss}),flush=True)
    state['completed']=True;state['status']='completed';state['next_job']=None;atomic(state,state_path)


if __name__=='__main__':main()
