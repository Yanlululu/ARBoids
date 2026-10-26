"""Deployment-window search for the 17:00 mixed-scene experiment."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import two_hour_development as t

e = t.e
OUT = e.BASE / 'mixed-conditional-development-20261026'


def initialize():
    state = t.initialize()
    OUT.mkdir(exist_ok=True)
    return state


def tasks_for(count=800, repeats=0):
    tasks = e.read(OUT/'mixed-design.pth')['tasks']
    if count != 800:
        indices = np.linspace(0,len(tasks)-1,count, dtype=int)
        tasks = [tasks[int(i)] for i in indices]
    if repeats:
        tasks = [(scene, e.seed32(9261031,j,int(scene[0])))
                 for scene,_ in tasks for j in range(repeats)]
    return tasks


def load_actor(path,d):
    actor=t.load_actor(path,d)
    support=e.read(path).get('candidate_support')
    if support is None:return actor
    import local_intervention_value as l
    decision=actor.successor_task_value
    decision.register_buffer('support_features',torch.as_tensor(support['features'],dtype=torch.float32))
    decision.register_buffer('support_options',torch.as_tensor(support['observed_options'],dtype=torch.bool))
    native_catalog=decision.catalog_estimates
    def supported_catalog(context,controls):
        ranking,values=native_catalog(context,controls)
        features=l.state_features(context,controls)
        distance=(features[:,None]-decision.support_features[None]).square().mean(-1)
        nearest=distance.topk(min(support['neighbors'],distance.shape[1]),largest=False).indices
        allowed=decision.support_options[nearest].any(1)
        allowed[:,0]=True
        return ranking.masked_fill(~allowed,-torch.inf),values
    decision.catalog_estimates=supported_catalog
    return actor


def evaluate(path, tasks, r, d, name, batch=80, device='cuda:0', fixed_width=0):
    path = Path(path)
    result_path = OUT/(name+'-evaluation.pth')
    if result_path.exists():
        result = e.read(result_path)
        if (result['tasks'] != tasks or result['policy_sha256'] != e.digest(path)
                or result.get('fixed_inference_width', 0) != fixed_width):
            raise ValueError('An evaluation is immutable for its policy and task list.')
        if result['completed']: return result
    started = time.monotonic()
    rows = r.batched_credit_rows(path,tasks,batch_size=batch,device=device,
        actor_override=load_actor(path,d),fixed_inference_width=fixed_width)
    result = dict(policy=str(path),policy_sha256=e.digest(path),tasks=tasks,rows=rows,
        completed=True,success=sum(row['success'] for row in rows),
        elapsed_seconds=time.monotonic()-started,role='mixed-scene development',
        inference_batch=batch,inference_device=device,fixed_inference_width=fixed_width)
    e.save(result,result_path)
    print(json.dumps(dict(name=name,success=result['success'],count=len(rows),
        elapsed_seconds=result['elapsed_seconds'])),flush=True)
    return result


def restrict_windows(model,l,span,margin,confidence):
    recipe = model['successor_task_value']
    if recipe['kind']=='local_difference_joint':
        result=l.limit_decision_windows(model,maximum_delay=0,maximum_span=span,decision_period=5)
        result['successor_task_value'].update(intervention_budget=1,margin=margin,confidence=confidence)
    else:
        result=copy.deepcopy(model)
        result['successor_task_value']['options']=[option for option in recipe['options']
            if option[1]==0 and option[2]<=span]
        result['successor_task_value'].update(intervention_budget=1,gain_margin=margin)
    result['evidence_role']='Development selection of immediate interventions on mixed scenes.'
    return result


def sweep(args):
    manifest,d,l,m,r=initialize()
    tasks=e.read(OUT/'mixed-design.pth')['tasks']
    tasks=[tasks[i] for i in range(0,800,10)]
    if args.prefix:
        paths=[OUT/(args.prefix+'-conditional.pth'),OUT/(args.prefix+'-ordinary.pth')]
        prefix=args.prefix
    else:
        paths=[t.OUT/'learned-conditional-events-conservative-n80.pth',t.OUT/'learned-ordinary-n80.pth']
        prefix='prior-strata'
    for span,margin,confidence in [(5,.02,.5),(20,.05,1.)]:
        for arm,path in zip(('conditional','ordinary'),paths):
            model=restrict_windows(e.read(path),l,span,margin,confidence)
            name=f'{prefix}-{arm}-immediate{span}'
            target=OUT/(name+'.pth')
            if not target.exists():e.save(model,target)
            evaluate(target,tasks,r,d,name+'-n80',args.batch,args.device)


def train(args):
    manifest,d,l,m,r=initialize()
    while True:
        data=e.read(OUT/'mixed-paired.pth')
        if data['completed']:break
        if not args.wait_data:raise ValueError('Use completed measured acquisition.')
        time.sleep(15)
    bank=e.read(OUT/'mixed-factual.pth')
    if not data['completed'] or not bank['completed']:
        raise ValueError('Use completed measured acquisition.')
    data,bank=l.retain_vessel_identity(data,bank)
    prefix=args.prefix or 'mixed-'+('mlp-' if args.architecture=='mlp' else '')+args.objective
    paths=[OUT/(prefix+'-conditional.pth'),OUT/(prefix+'-ordinary.pth')]
    if not all(path.exists() for path in paths):
        objective=args.objective
        if objective=='decision':
            source=manifest['runtime_sources']['local_intervention_value.py']
            node=next(node for node in ast.parse(source).body
                      if isinstance(node,ast.FunctionDef) and node.name=='fit_success_intervention_pair')
            function=ast.get_source_segment(source,node)
            marker='rank_weight=0., decision_regret_weight=0.'
            if function.count(marker)!=1:raise RuntimeError('Expected the common value-fit recipe.')
            function=function.replace(marker,'rank_weight=2., decision_regret_weight=4.')
            exec(compile(function,'<conditional-decision-regret>', 'exec'),l.__dict__)
            objective='marginal_events'
        options=(dict(identity_conditioned=True) if args.architecture=='mlp' else
                 dict(identity_conditioned=True,finite_motion_basis=True))
        l.fit_success_intervention_pair(data,*paths,factual_bank=bank,updates=args.updates,
            ensemble_size=3,width=64,seed=args.seed,device=args.device,architecture=args.architecture,
            conditional_objective=objective,
            network_options=options)
    if args.no_screen:return
    tasks=e.read(OUT/'mixed-design.pth')['tasks']
    tasks=[tasks[i] for i in range(0,800,10)]
    for path in paths:evaluate(path,tasks,r,d,path.stem+'-n80',args.batch,args.device)


def support_search(args):
    manifest,d,l,m,r=initialize()
    data=e.read(OUT/'mixed-paired.pth')
    bank=e.read(OUT/'mixed-factual.pth')
    if not data['completed']:raise ValueError('Candidate support requires completed measured roots.')
    data,bank=l.retain_vessel_identity(data,bank)
    contexts=torch.as_tensor(np.asarray([root['context'] for root in data['roots']]),dtype=torch.float32)
    controls=torch.as_tensor(np.asarray([root['effect_controls'] for root in data['roots']]),dtype=torch.float32)
    features=l.state_features(contexts,controls).numpy()
    tasks=e.read(OUT/'mixed-design.pth')['tasks']
    tasks=[tasks[i] for i in range(0,800,10)]
    prefix=args.prefix or 'mixed-mlp-joint_likelihood'
    for neighbors in (1,3,6):
        for arm in ('conditional','ordinary'):
            model=e.read(OUT/(prefix+'-'+arm+'.pth'))
            options=model['successor_task_value']['options']
            lookup={tuple(option):index for index,option in enumerate(options)}
            observed=np.zeros((len(data['roots']),len(options)),dtype=bool)
            for index,root in enumerate(data['roots']):
                for option,_ in root['outcomes']:
                    observed[index,lookup[tuple(root['options'][option])]]=True
            model['candidate_support']=dict(features=features,observed_options=observed,
                neighbors=neighbors,source_sha256=e.digest(OUT/'mixed-paired.pth'),
                rule='Rank only alternatives measured at nearest public-state roots; factual action is always admissible.',
                outcome_labels_used_for_support=False)
            name=prefix+f'-supportk{neighbors}-'+arm
            target=OUT/(name+'.pth')
            if not target.exists():e.save(model,target)
            evaluate(target,tasks,r,d,name+'-n80',args.batch,args.device,args.fixed_width)


def fresh_design(prefix=None):
    label='joint-final-new200' if prefix is None else 'joint-'+prefix+'-final-new200'
    target=OUT/(label+'-design.pth')
    if target.exists():return e.read(target)
    if prefix is not None:
        base=fresh_design()
        stem='mixed-mlp-joint_likelihood-'+prefix
        paths=dict(conditional=OUT/(stem+'-conditional.pth'),ordinary=OUT/(stem+'-ordinary.pth'),
                   ci=t.OUT/'best-m1-400.pth')
        design=dict(tasks=base['tasks'],policies={name:dict(path=str(path),sha256=e.digest(path))
            for name,path in paths.items()},frozen_unix=time.time(),role=base['role'],
            selection='Development-only candidate support screen; new-geometry outcomes not used.')
        e.save(design,target)
        return design
    used={int(scene[0]) for scene,_ in e.read(OUT/'mixed-design.pth')['tasks']}
    tasks=[]
    index=0
    while len(tasks)<200:
        seed=e.seed32(1104261719,index)
        index+=1
        if seed in used:continue
        used.add(seed)
        tasks.append(((seed,6,2. if len(tasks)<100 else 2.5),None))
    paths=dict(conditional=OUT/'mixed-mlp-joint_likelihood-conditional.pth',
        ordinary=OUT/'mixed-mlp-joint_likelihood-ordinary.pth',ci=t.OUT/'best-m1-400.pth')
    design=dict(tasks=tasks,policies={name:dict(path=str(path),sha256=e.digest(path))
        for name,path in paths.items()},frozen_unix=time.time(),
        role='New paired geometry after freezing the two trained networks; excluded from subsequent tuning.')
    e.save(design,target)
    return design


def fresh(args):
    manifest,d,l,m,r=initialize()
    design=fresh_design(args.prefix)
    label='joint-final-new200' if args.prefix is None else 'joint-'+args.prefix+'-final-new200'
    policy=design['policies'][args.arm]
    if e.digest(Path(policy['path']))!=policy['sha256']:
        raise ValueError('A frozen evaluation policy changed.')
    result=evaluate(policy['path'],design['tasks'],r,d,label+'-'+args.arm,
        args.batch,args.device,args.fixed_width)
    result.update(role=design['role'],design_sha256=e.digest(OUT/(label+'-design.pth')))
    e.save(result,OUT/(label+'-'+args.arm+'-evaluation.pth'))


def repeat_seed(args):
    settings=copy.copy(args)
    settings.prefix=f'mixed-mlp-joint_likelihood-seed{args.seed}'
    settings.architecture='mlp'
    settings.objective='joint_likelihood'
    settings.no_screen=True
    train(settings)
    manifest,d,l,m,r=initialize()
    base_design=fresh_design('supportk3')
    policies={}
    for arm in ('conditional','ordinary'):
        reference=e.read(OUT/f'mixed-mlp-joint_likelihood-supportk3-{arm}.pth')
        model=e.read(OUT/(settings.prefix+'-'+arm+'.pth'))
        if model['successor_task_fit']['training_observation_sha256']!=reference['successor_task_fit']['training_observation_sha256']:
            raise ValueError('Training-seed replication must retain the exact observations.')
        model['candidate_support']=copy.deepcopy(reference['candidate_support'])
        model['seed_replication']=dict(seed=args.seed,reference_policy=base_design['policies'][arm],
            rule='Same frozen algorithm, observations, updates and support; no selection on replication outcomes.')
        path=OUT/(settings.prefix+'-supportk3-'+arm+'.pth')
        if not path.exists():e.save(model,path)
        policies[arm]=dict(path=str(path),sha256=e.digest(path))
    label=f'joint-supportk3-seed{args.seed}-new200'
    design=dict(tasks=base_design['tasks'],policies=policies,frozen_unix=time.time(),
        role='Training-seed replication on the same new-geometry cohort; no hyperparameter reselection.')
    e.save(design,OUT/(label+'-design.pth'))
    for arm,policy in policies.items():
        result=evaluate(policy['path'],design['tasks'],r,d,label+'-'+arm,
            args.batch,args.device,args.fixed_width)
        result.update(role=design['role'],design_sha256=e.digest(OUT/(label+'-design.pth')))
        e.save(result,OUT/(label+'-'+arm+'-evaluation.pth'))


def freeze_memory():
    initialize()
    tasks=e.read(OUT/'mixed-design.pth')['tasks']
    sources={}
    indexed={}
    for name in ('mixed-intervention-search.pth','new-generated-intervention-search.pth',
                 'new-scene-intervention-search.pth'):
        path=OUT/name
        if not path.exists():continue
        data=e.read(path)
        sources[name]=e.digest(path)
        for record in data['records']:
            index=int(record['development_task_index'])
            if int(record['measured_row']['seed'])!=int(tasks[index][0][0]):
                raise ValueError('A measured intervention must match the saved scene.')
            if index in indexed:
                if not np.array_equal(indexed[index]['features'],record['features']):
                    raise ValueError('Inconsistent initial states for a development index.')
                continue
            indexed[index]=copy.deepcopy(record)
    records=[indexed[index] for index in sorted(indexed)]
    fingerprint=hashlib.sha256(json.dumps([
        dict(index=record['development_task_index'],delay=record['delay'],span=record['span'],
             preferences=record['preferences'].tolist(),features=record['features'].tolist())
        for record in records],sort_keys=True).encode()).hexdigest()
    model=e.read(t.OUT/'best-m1-400.pth')
    model.update(empirical_interventions=records,development_subset_interventions=True,
        evidence_role='Mixed-scene development memory; all reused scenes are development.',
        empirical_training_sources=sources,empirical_records_sha256=fingerprint)
    target=OUT/f'mixed-memory-merged-r{len(records)}-{fingerprint[:10]}.pth'
    if not target.exists():e.save(model,target)
    print(json.dumps(dict(policy=str(target),records=len(records))),flush=True)
    return target


def rescue_new(args):
    manifest,d,l,m,r=initialize()
    tasks=e.read(OUT/'mixed-design.pth')['tasks']
    bank=e.read(OUT/'mixed-factual.pth')
    if not bank['completed']:raise ValueError('The new layouts need completed factual observations.')
    policy=t.OUT/'best-m1-400.pth'
    rows=[dict(success=float(record['values'][0]),physical_steps=record['physical_steps'])
          for record in bank['records']]
    base=dict(policy=str(policy),rows=rows)
    indices=[i for i in range(600,800) if not rows[i]['success']]
    source=Path(t.__file__).read_text(encoding='utf8')
    node=next(node for node in ast.parse(source).body
              if isinstance(node,ast.FunctionDef) and node.name=='capture_roots')
    function=ast.get_source_segment(source,node).replace("'failure-roots.pth'","'new-scene-roots.pth'")
    namespace=dict(t.__dict__,OUT=OUT)
    exec(compile(function,'<new-scene-state-collection>','exec'),namespace)
    roots=namespace['capture_roots'](base,dict(manifest,heldout_tasks=tasks),d,r,indices,args.batch,args.device)
    path=OUT/'new-scene-intervention-search.pth'
    data=e.read(path) if path.exists() else dict(records=[],measured=[],rescued=[],paired_groups=[],completed=False)
    actor=t.load_actor(policy,d)
    coefficients=[(0.,0.),(1.,1.),(0.,1.),(1.,0.),(.5,.5),(0.,.5),(.5,0.),(1.,.5),(.5,1.)]
    for mode,span in [('late60',30),('half',1),('late120',30),('initial',1),
                      ('half',5),('initial',20),('late60',60),('quarter',1),('initial',60)]:
        for index in indices:
            if time.time()>args.deadline:
                e.save(data,path)
                return
            seed=tasks[index][0][0]
            if seed in data['rescued']:continue
            terminal=rows[index]['physical_steps']
            step=(max(0,terminal-int(mode[4:])) if mode.startswith('late') else
                  5*(terminal//10) if mode=='half' else 5*(terminal//20) if mode=='quarter' else 0)
            identity=(seed,step,span)
            if identity in data['measured'] or (seed,step) not in roots:continue
            snapshot=roots[(seed,step)]
            packet={k:v.to(args.device) for k,v in d.mechanism_tensor(
                d.joint_packet(snapshot['environment'],snapshot['observation'])).items()}
            actor.to(args.device)
            with torch.no_grad():control=actor(packet,deterministic=True,gate_only=True)[0][0].cpu().numpy()[:,:3]
            one=np.ones((len(control),1),np.float32)
            jobs=[];preferences=[]
            for boat in range(len(control)):
                for pair in coefficients:
                    preference=np.full((len(control),2),-1.,np.float32);preference[boat]=pair
                    action=np.concatenate((control,-one,span*one,-2*one,np.zeros((len(control),2),np.float32),
                        np.zeros_like(one),preference,preference),-1)
                    jobs.append(dict(environment=snapshot['environment'],observation=snapshot['observation'],
                        random_state=snapshot['random_state'],action=action,isolate_factual_start=True))
                    preferences.append(preference)
            measured=r.batched_credit_rows(policy,[snapshot['task']]*len(jobs),batch_size=len(jobs),
                device=args.device,initial_jobs=jobs,actor_override=actor)
            good=[i for i,row in enumerate(measured) if row['success']]
            if good:
                chosen=min(good,key=lambda i:measured[i]['duration'])
                data['records'].append(dict(features=roots[(seed,0)]['features'],root_features=snapshot['features'],
                    preferences=preferences[chosen],delay=step,span=span,development_task_index=index,
                    root_steps=step,measured_row=measured[chosen],selection_role='Successful measured development continuation.'))
                data['rescued'].append(seed)
            data['measured'].append(identity)
            data['paired_groups'].append(dict(task=snapshot['task'],root_steps=step,span=span,
                preferences=preferences,rows=measured,isolate_factual_start=True))
            e.save(data,path)
            print(json.dumps(dict(phase='new_scene_rescue',task=index,step=step,span=span,
                successful_options=len(good),rescued=len(data['rescued']))),flush=True)
    data['completed']=True;e.save(data,path)


def calibrate(args):
    manifest,d,l,m,r=initialize()
    prefix=args.prefix or 'retained-marginal_events'
    if prefix.startswith('retained'):
        data=e.read(e.BASE/'local-difference-joint-development/F11-r136-six-trained-paired-data-seed42.pth')
        bank=e.read(e.BASE/'local-difference-joint-development/F11-r136-six-trained-factual-data-seed42.pth')
    else:
        data=e.read(OUT/'mixed-paired.pth')
        bank=e.read(OUT/'mixed-factual.pth')
    data,bank=l.retain_vessel_identity(data,bank)
    original=e.read(OUT/(prefix+'-conditional.pth'))
    calibrated=l.refresh_local_calibration(original,data,device=args.device,
        effective_future_confidence=True,empirical_difference_correction=True,
        joint_factual_calibration=True,factual_probability_calibration=True,
        temporal_value_bounds=True,project_task_value=True,confidence=.5,margin=.02)
    target=OUT/(prefix+'-calibrated.pth')
    if not target.exists():e.save(calibrated,target)
    tasks=e.read(OUT/'mixed-design.pth')['tasks']
    evaluate(target,[tasks[i] for i in range(0,800,10)],r,d,target.stem+'-n80',args.batch,args.device)


def stability(args):
    manifest,d,l,m,r=initialize()
    policy=Path(args.policy) if args.policy else freeze_memory()
    model=e.read(policy)
    records=model['empirical_interventions']
    all_tasks=e.read(OUT/'mixed-design.pth')['tasks']
    tasks=[]
    for record in records:
        index=int(record['development_task_index'])
        scene=all_tasks[index][0]
        for repetition in range(args.repeats or 4):
            tasks.append((scene,e.seed32(9261032,args.probe_round,index,repetition)))
    prefix=policy.stem+f'-stability{args.probe_round}'
    baseline=evaluate(t.OUT/'best-m1-400.pth',tasks,r,d,prefix+'-baseline',args.batch,args.device,args.fixed_width)
    proposed=evaluate(policy,tasks,r,d,prefix+'-proposed',args.batch,args.device,args.fixed_width)
    repetitions=args.repeats or 4
    details=[]
    for index,record in enumerate(records):
        group=slice(index*repetitions,(index+1)*repetitions)
        before=baseline['rows'][group];after=proposed['rows'][group]
        rescued=sum(not a['success'] and b['success'] for a,b in zip(before,after))
        harmed=sum(a['success'] and not b['success'] for a,b in zip(before,after))
        details.append(dict(development_task_index=record['development_task_index'],
            rescued=rescued,harmed=harmed,baseline=sum(a['success'] for a in before),
            proposed=sum(a['success'] for a in after)))
    result=dict(policy=str(policy),policy_sha256=e.digest(policy),details=details,
        repeats=repetitions,probe_round=args.probe_round,baseline=baseline['success'],
        proposed=proposed['success'],tasks=tasks,
        role='New stochastic continuations of selected development layouts; not unseen geometry.')
    e.save(result,OUT/(prefix+'-comparison.pth'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('details','tasks')}),flush=True)
    if args.probe_round==0 or args.select_stable:
        selected=[record for record,detail in zip(records,details)
                  if detail['rescued']>0 and detail['harmed']==0]
        robust=copy.deepcopy(model)
        robust.update(empirical_interventions=selected,stability_selection=result)
        target=OUT/(policy.stem+f'-robust{len(selected)}.pth')
        if not target.exists():e.save(robust,target)
        print(json.dumps(dict(robust_policy=str(target),retained=len(selected))),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('train','sweep','evaluate','freeze','calibrate','stability','rescue-new','support','fresh','repeat-seed'))
    parser.add_argument('--prefix')
    parser.add_argument('--arm',choices=('conditional','ordinary','ci'),default='conditional')
    parser.add_argument('--policy')
    parser.add_argument('--name')
    parser.add_argument('--count',type=int,default=800)
    parser.add_argument('--repeats',type=int,default=0)
    parser.add_argument('--probe-round',type=int,default=0)
    parser.add_argument('--deadline',type=float,default=1793009714.)
    parser.add_argument('--objective',default='marginal_events',choices=('marginal_events','factual_strata','decision','joint_likelihood'))
    parser.add_argument('--architecture',default='interaction',choices=('interaction','mlp'))
    parser.add_argument('--updates',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--no-screen',action='store_true')
    parser.add_argument('--batch',type=int,default=80)
    parser.add_argument('--fixed-width',type=int,default=0)
    parser.add_argument('--wait-data',action='store_true')
    parser.add_argument('--select-stable',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.operation=='train':train(args)
    elif args.operation=='sweep':sweep(args)
    elif args.operation=='freeze':freeze_memory()
    elif args.operation=='calibrate':calibrate(args)
    elif args.operation=='stability':stability(args)
    elif args.operation=='support':support_search(args)
    elif args.operation=='fresh':fresh(args)
    elif args.operation=='repeat-seed':repeat_seed(args)
    elif args.operation=='rescue-new':rescue_new(args)
    else:
        manifest,d,l,m,r=initialize()
        path=Path(args.policy)
        evaluate(path,tasks_for(args.count,args.repeats),r,d,
            args.name or path.stem+f'-n{args.count}-repeat{args.repeats}',args.batch,args.device,args.fixed_width)


if __name__=='__main__':main()
