"""Frozen single-world evaluation with full controller identity and per-step traces."""
import argparse,json,time,gzip,inspect
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .bridge import Source,World,Controller,ControlOptions
from .collect import load_behavior
from .opponents import load_opponent
from .contracts import Protocol,verify_freeze,file_hash,runtime_identity,LayoutRegistry,canonical_hash,validate_task_banks


def atomic_json(value,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def committed_rows(out,tasks):
    rows={}
    for path in sorted((out/'task_records').glob('*.json')):
        row=json.loads(path.read_text(encoding='utf-8'));index=int(path.stem)
        if index>=len(tasks) or row['task']!=tasks[index]:raise ValueError('Committed task identity changed')
        trace=out/row['trace']
        if trace.parent!=out or not trace.is_file() or file_hash(trace)!=row['trace_sha256']:
            raise ValueError('Committed task trace missing or changed')
        rows[index]=row
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--source',required=True)
    ap.add_argument('--tasks',required=True);ap.add_argument('--registry',required=True);ap.add_argument('--checkpoint')
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--phase',choices=['development','final'],default='development');ap.add_argument('--freeze')
    ap.add_argument('--budget',type=int,default=96);ap.add_argument('--bank',type=int,default=12)
    ap.add_argument('--rounds',type=int,default=1);ap.add_argument('--risk',choices=['expected','mean_distance'],default='expected')
    ap.add_argument('--risk-weight',type=float,help='Online lambda_c only; frozen evaluation still locks the effective value')
    ap.add_argument('--opponent-spec',help='Frozen APF/TorchScript opponent specification')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--max-tasks',type=int,default=0,help='Bound this call; retain the full task bank identity')
    ap.add_argument('--selection',choices=['joint','independent'],default='joint')
    ap.add_argument('--stop-file',help='Commit the current task, then stop at its boundary')
    ap.add_argument('--prepare-runtime');a=ap.parse_args();out=Path(a.out)
    if a.max_tasks<0:raise ValueError('Invalid task limit')
    if out.exists() and not (a.prepare_runtime or a.resume):raise ValueError('Refuse result overwrite; use --resume')
    tasks=json.loads(Path(a.tasks).read_text());p=Protocol(**tasks.get('protocol',{}));torch.set_num_threads(1)
    validate_task_banks((tasks['role'],tasks))
    if a.risk_weight is not None:p=replace(p,risk_weight=a.risk_weight)
    opponent=load_opponent(a.opponent_spec,a.device)
    control={'bank':a.bank,'budget':a.budget,'rounds':a.rounds,'risk':a.risk,'nominal_beam_width':128,
             'search':'whole_configuration_single_coordinate_neighbors','initializer':'source_center_reserved_inside_bank-v1','history_cache':True}
    if a.selection=='independent':
        control.update(search='simultaneous_common_source_center')
    if a.selection=='independent' and not a.checkpoint:raise ValueError('CITF-independent requires the matched learned predictor')
    options=ControlOptions(a.bank,a.budget,a.rounds,a.risk,a.selection)
    checked_source=Source(a.project,a.source,a.device)
    checked_world=World(a.project,p,tasks['tasks'][0]['defenders'])
    checked_model=load_behavior(a.checkpoint,checked_source,checked_world,a.device)
    checked_controller=options.build(Controller,checked_source,checked_model,a.device,world=checked_world)
    if checked_model is not None and hasattr(checked_model,'protocol_transfer'):
        control['initialization_protocol_transfer']=checked_model.protocol_transfer
    control['supported_selections']=sorted(getattr(checked_controller,'supported_selections',('joint',)))
    method='CITF-independent' if a.selection=='independent' else ('CITF' if a.checkpoint else 'CITF-Phys')
    code_entries=[path for path in (inspect.getsourcefile(Controller),inspect.getsourcefile(type(checked_controller))) if path]
    identity=runtime_identity(a.project,a.source,a.checkpoint,p,control,a.device,method,
        opponent=opponent.identity,code_entries=code_entries)
    if a.prepare_runtime:
        # The actual controller, including its supported modes, was checked above.
        Path(a.prepare_runtime).parent.mkdir(parents=True,exist_ok=True)
        Path(a.prepare_runtime).write_text(json.dumps(identity,ensure_ascii=False,indent=2));return
    if a.phase=='final':
        if not a.freeze:raise ValueError('Final evaluation requires the complete freeze manifest')
        frozen=verify_freeze(a.freeze,a.checkpoint,a.tasks,identity)
        if frozen['registry_sha256']!=file_hash(a.registry):raise ValueError('Layout allocation changed after freezing')
    elif tasks['role']=='final_test':raise ValueError('Final bank cannot be evaluated in development mode')
    transaction={'runtime':identity,'tasks_sha256':file_hash(a.tasks),'registry_sha256':file_hash(a.registry),'phase':a.phase}
    if a.resume:
        recorded=json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
        if canonical_hash(recorded)!=canonical_hash(transaction):raise ValueError('Evaluation resume identity changed')
    else:
        out.mkdir(parents=True);(out/'task_records').mkdir()
        atomic_json(transaction,out/'evaluation.json');atomic_json(identity,out/'runtime.json')
    source=checked_source;registry=LayoutRegistry(a.registry)
    del checked_controller,checked_model,checked_world
    committed=committed_rows(out,tasks['tasks']);executed=0
    seen=set()
    for index,t in enumerate(tasks['tasks']):
        tid=t.get('task_id',f'task_{index:05d}')
        if tid in seen:raise ValueError('Duplicate task ID')
        seen.add(tid)
        if index in committed:continue
        if a.stop_file and Path(a.stop_file).exists():continue
        if a.max_tasks and executed>=a.max_tasks:continue
        world=World(a.project,p,t['defenders']);world.reset(t['scene_seed'],t['future_seed'],t['agility'])
        family=world.initial_identity['fingerprint']
        if t.get('geometry_family_id')!=family:raise ValueError('Declared task geometry differs from reset geometry')
        registered=registry.data['families'].get(family)
        if not registered or registered['split']!=tasks['role']:raise ValueError('Task not allocated to declared split')
        model=load_behavior(a.checkpoint,source,world,a.device)
        controller=options.build(Controller,source,model,a.device,world=world)
        lat=[];calls=[];step=0;trace_path=out/f'{index:05d}_steps.jsonl.gz'
        pending=trace_path.with_suffix(trace_path.suffix+'.pending')
        with gzip.open(pending,'wt',encoding='utf-8') as trace:
            while not world.done:
                before=world.state().copy();physical_time=float(world.env.Current_T)
                if a.device.startswith('cuda'):torch.cuda.synchronize()
                start=time.perf_counter();force,decision=controller.act(world)
                if a.device.startswith('cuda'):torch.cuda.synchronize()
                duration=time.perf_counter()-start;lat.append(duration);calls.append(decision['evaluated_unique'])
                attacker_action=opponent.act(world)
                result=world.step(force,attacker_action=attacker_action)
                trace.write(json.dumps({'task_id':tid,'step':step,'sensor_sim_time':physical_time,'next_sim_time':world.env.Current_T,
                  'state_before':before.tolist(),'state_after':world.state().tolist(),'decision':decision,
                  'opponent_identity':opponent.identity,'executed_attacker_action':np.asarray(world.env.att_action).tolist(),
                  'compute_seconds':duration,'deadline_missed':duration>p.control_dt,'events':result},ensure_ascii=False)+'\n');step+=1
        pending.replace(trace_path)
        row=dict(task=t,**result,steps=len(lat),latency_mean_ms=1000*float(np.mean(lat)),latency_p95_ms=1000*float(np.percentile(lat,95)),
          latency_p99_ms=1000*float(np.percentile(lat,99)),deadline_fraction=float(np.mean(np.asarray(lat)>p.control_dt)),
          mean_unique_candidates=float(np.mean(calls)),capped_capture_time=result['duration'] if result['capture'] else p.time_limit,
          trace=trace_path.name,trace_sha256=file_hash(trace_path))
        atomic_json(row,out/'task_records'/f'{index:05d}.json')
        committed[index]=row;executed+=1
        print(json.dumps({'task':index,'task_id':tid,'outcome':row['outcome'],'committed':len(committed),'total':len(tasks['tasks'])}),flush=True)
    rows=[committed[i] for i in sorted(committed)]
    if not rows:raise ValueError('No evaluation tasks')
    partial=out/'tasks.jsonl.pending';partial.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');partial.replace(out/'tasks.jsonl')
    report=dict(completed=len(rows)==len(tasks['tasks']),phase=a.phase,N=len(rows),expected_N=len(tasks['tasks']),runtime=identity,tasks_sha256=file_hash(a.tasks),registry_sha256=file_hash(a.registry),
      timing='single-world offline synchronous simulation; latency is measured but physics does not advance during compute',
      success=sum(r['success'] for r in rows),capture=sum(r['capture'] for r in rows),collision=sum(r['collision'] for r in rows),
      physical_collision=sum(r['physical_collision'] for r in rows),physical_breach=sum(r['physical_breach'] for r in rows),
      radial_only_failure=sum(r['radial_only_failure'] for r in rows),timeout=sum(r['timeout'] for r in rows),
      ever_radial_disadvantage=sum(r.get('ever_radial_disadvantage',0) for r in rows),
      radial_recovery=sum(r.get('radial_recovery',0) for r in rows),
      capped_capture_time=float(np.mean([r['capped_capture_time'] for r in rows])))
    atomic_json(report,out/'summary.json');print(json.dumps({k:report[k] for k in ('completed','N','expected_N','success','capture','collision')},ensure_ascii=False))
if __name__=='__main__':main()
