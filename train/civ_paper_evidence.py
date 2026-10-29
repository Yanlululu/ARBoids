"""Matched evidence for the frozen unified CIV controller, without legacy relabeling."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np
import torch
import civ_rl as c
import civ_generalization as g
from civ_rl_experiment import _atomic_save, digest, make_tasks, run_streamed_episodes

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'train/experiments/author-source-fusion-20261011'
DEVELOPMENT=BASE/'unified-civ-development-20261027'
SELECTED=DEVELOPMENT/'finalplan-optimization-1045'
OUTPUT=DEVELOPMENT/'paper-evidence-20261027'
DESIGN=OUTPUT/'design.pth'

def read(path):return torch.load(path,map_location='cpu',weights_only=False)

def design():
    if DESIGN.exists():return read(DESIGN)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    models={name:SELECTED/f'unified-selected-calibrated-eta0075-{kind}.pth'
        for name,kind in [('CIV','conditional'),('ordinary','ordinary')]}
    selected=read(SELECTED/'selected-calibrated-comparison.pth')
    cells={'selected-development':selected['tasks'],'confirmation':make_tasks(98000000,200,6,(2.,2.5))}
    for i,agility in enumerate((1.,1.5,2.,2.5,3.,3.5)):
        cells[f'agility-{agility:g}']=make_tasks(99000000+10000*i,24,6,(agility,))
    for i,n in enumerate((3,4,5,7,8)):
        cells[f'fleet-{n}']=make_tasks(99200000+10000*i,24,n,(2.,2.5))
    analysis=read(BASE/'r20-independent-evidence/analysis-manifest.pth')
    vrx=analysis['vrx_tasks'][:6]+analysis['vrx_tasks'][50:56]
    value=read(models['CIV'])
    observed={tuple(s) for s in value['fit']['training_scenes']+value['fit']['validation_scenes']}
    if any(tuple(t['scene']) in observed for k,tasks in cells.items() if k!='selected-development' for t in tasks):
        raise ValueError('A new evidence task overlaps value fitting.')
    record=dict(completed=True,algorithm='CIV-RL-unified-v1',inference_device='cpu',
        models={name:dict(path=str(path),sha256=digest(path)) for name,path in models.items()},
        cells=cells,vrx_tasks=vrx,source_sha256=value['source_sha256'],runtime=c.runtime_identity(),
        selection_scope='Best completed development configuration; selected-development is explicitly post-selection.',
        confirmation_scope='Fresh scene and future seeds frozen before new evaluations; no endpoint-based selection.',
        baseline_seeds=[42],cv_repetitions='Existing repeated value fits share their frozen CI source.',
        sensitivity=dict(eta=[0.,.025,.075,.15],lambda_multiplier=[0.,.5,1.,2.]))
    _atomic_save(record,DESIGN)
    print('DESIGN_FROZEN',len(cells),sum(map(len,cells.values())),flush=True)
    return record

def controller(record,method,margin=None,multiplier=None,model=None):
    item=record['models']['ordinary' if method=='ordinary' else 'CIV']
    path=Path(model) if model else Path(item['path'])
    if not model and digest(path)!=item['sha256']:raise ValueError('Frozen paper policy changed.')
    policy=read(path)
    result=g.load(path,'cpu') if policy.get('value_architecture')==g.ARCHITECTURE else c.load_controller(path,'cpu')
    if method in ('CI','independent','local'):
        result=c.CIVController(result.source,replace(result.config,generator='joint' if method=='CI' else method))
    if margin is not None:result.config=replace(result.config,acceptance_margin=margin)
    if multiplier is not None:
        settings=dict(result.source.actuator_fusion_refinement)
        settings['risk_weight']=settings.get('risk_weight',32.)*multiplier
        result.source.actuator_fusion_refinement=settings
    return result,path

def baseline(method,tasks):
    import civ_baseline_evaluation as b
    b.initialize_baselines()
    identity=location=training=None
    if method in ('arboids','rp','sac'):
        directory=BASE/'independent-training'/f'source-six-{method}-seed42-20261021'
        record,_=b.m.validate_author_source_repeat(directory)
        identity=record['selected_checkpoint_sha256'];location=str(directory);training=record['training_steps']
    elif method=='mappo':
        directory=BASE/'r20-six-baseline-comparison/mappo-seed42'
        record=read(directory/'training-result.pth')
        if not record['completed'] or record['steps']!=record['budget']:raise ValueError('Incomplete MAPPO baseline.')
        location=str(directory/'best.pth');identity=digest(location);training=record['steps']
        if identity!=record['best_sha256']:raise ValueError('MAPPO identity mismatch.')
    rows=[b.episode((method,location,identity,task)) for task in tasks]
    return rows,dict(policy=location,policy_sha256=identity,training_steps=training)

def evaluate(args):
    import fcntl
    import tempfile
    identity=hashlib.sha256(repr((str(OUTPUT),args.cell,args.method,args.margin,args.multiplier)).encode()).hexdigest()
    lock=Path(tempfile.gettempdir())/'civ-paper-evidence-locks';lock.mkdir(exist_ok=True)
    with (lock/identity).open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        return _evaluate(args)

def _evaluate(args):
    record=design();tasks=record['cells'][args.cell]
    suffix=('' if args.margin is None else f'-eta{args.margin:g}')+('' if args.multiplier is None else f'-lambda{args.multiplier:g}')
    path=OUTPUT/'evaluations'/args.cell/f'{args.method}{suffix}.pth'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=read(path)
        if not old['completed'] or old['design_sha256']!=digest(DESIGN):raise ValueError('Inconsistent resumed evaluation.')
        return old
    started=time.monotonic()
    if args.method in ('arboids','rp','sac','boids','mappo'):
        rows,metadata=baseline(args.method,tasks)
    else:
        policy,policy_path=controller(record,args.method,args.margin,args.multiplier,args.model)
        rows=run_streamed_episodes(policy,tasks,batch_size=64,progress=True)
        metadata=dict(policy=str(policy_path),policy_sha256=digest(policy_path),config=asdict(policy.config),
            source_sha256=record['source_sha256'],runtime=c.runtime_identity())
    if len(rows)!=len(tasks):raise ValueError('Incomplete endpoint evidence.')
    result=dict(completed=True,method=args.method,tasks=tasks,rows=rows,**metadata,
        design_sha256=digest(DESIGN),cell=args.cell,elapsed_seconds=time.monotonic()-started,
        success=sum(r['success'] for r in rows),collision=sum(r['collision'] for r in rows),
        independent_of_development_selection=args.cell!='selected-development',
        eta=args.margin,lambda_multiplier=args.multiplier)
    _atomic_save(result,path)
    print('EVIDENCE_COMPLETE',args.cell,args.method,result['success'],len(rows),flush=True)
    return result

def vrx(args):
    record=design();adapter=ROOT/'vrx/run_r20_evidence.py'
    folder=OUTPUT/('vrx-cuda' if args.vrx_device.startswith('cuda') else 'vrx')
    for index,task in enumerate(record['vrx_tasks']):
        if index%args.shards!=args.shard:continue
        for method in ('CIV','CI','arboids'):
            run_id=f'case{index:03d}-{method}'
            result_path=folder/run_id/'result.json'
            if result_path.exists():
                old=json.loads(result_path.read_text())
                if old.get('passed'):continue
                if old.get('success') is not None or not old.get('error'):
                    raise ValueError('Do not retry a valid mission outcome: '+run_id)
                run_id+='-attempt2';result_path=folder/run_id/'result.json'
                if result_path.exists():
                    old=json.loads(result_path.read_text())
                    if old.get('passed'):continue
                    raise ValueError('Repeated transport failure needs inspection: '+run_id)
            command=['python','-X','utf8','-u',str(adapter),'--variant',method,'--inference-device',args.vrx_device,
                '--num-robots','7','--agility',str(task['agility']),'--setting',str(task['setting']),
                '--seed',str(task['seed']),'--duration','80','--action-period','.2','--headless',
                '--startup-timeout','180','--wall-timeout','600','--assets-dir',str(ROOT/'.vrx-assets'),
                '--output-dir',str(folder),'--run-id',run_id]
            if method in ('CIV','CI'):command.extend(('--civ-checkpoint',record['models']['CIV']['path']))
            if index==0:command.append('--capture-frames')
            shell='source vrx_ws/activate.bash >/dev/null && export ROS_DOMAIN_ID='+str(120+args.shard)+' ARBOIDS_RENDER_THREADS=2 && exec '+shlex.join(command)
            completed=subprocess.run(['xvfb-run','-a','bash','-lc',shell],cwd=ROOT)
            if not result_path.exists():raise RuntimeError('Missing native Gazebo result: '+run_id)
            result=json.loads(result_path.read_text());result.update(paper_design_sha256=digest(DESIGN),
                paired_scene_index=index,command_returncode=completed.returncode)
            result_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
            print('VRX_COMPLETE',run_id,result['passed'],result.get('success'),flush=True)
            if not result['passed']:raise RuntimeError('VRX transport failed: '+run_id)

def adaptation_design():
    path=OUTPUT/'adaptation-design.pth'
    if path.exists():return read(path)
    paper=design();old=read(BASE/'r20-independent-evidence/analysis-manifest.pth')['learned_opponent']
    training=Path(old['path']).parent/'attacker-training.pth'
    record=dict(completed=True,defender=paper['models']['CIV'],paper_design_sha256=digest(DESIGN),
        initial_attacker=dict(path=old['path'],sha256=digest(old['path'])),
        initial_training=dict(path=str(training),sha256=digest(training)),
        evaluation_tasks=make_tasks(99600000,64,6,(2.,2.5)),
        paired_training_tasks=make_tasks(99700000,192,6,(2.,2.5)),
        additional_attacker_steps=32000,attacker_stages=[8000,16000,32000],
        vector_environments=32,batch_size=256,updates_per_vector_step=8,warm_steps=2048,
        defender_updates=3000,paired_futures=2,inference_device='cuda:0',
        attacker_reward='Unchanged source attacker reward.',
        defender_update='Resume conditional value networks using freshly paired terminal outcomes against the continued attacker; CI remains frozen.')
    _atomic_save(record,path);return record

def opponent_helpers():
    from types import SimpleNamespace
    value=read(BASE/'r20-independent-evidence/analysis-manifest.pth')
    source=value['opponent_helpers']
    if hashlib.sha256(source.encode()).hexdigest()!=value['opponent_helpers_sha256']:
        raise ValueError('Frozen source attacker helper identity changed.')
    namespace=dict(np=np,torch=torch,sys=sys,Path=Path,hashlib=hashlib,d=c.d,
        _SOURCE_ATTACKER=None,_SOURCE_ATTACKER_SHA256=None)
    exec(compile(source,'<verified-source-opponent>','exec'),namespace)
    return SimpleNamespace(**namespace)

def opponent_binding(checkpoint):
    """Bind to the environment actually used by the unified executor."""
    r=opponent_helpers()
    from envs.TADgame import TADEnv
    original=getattr(TADEnv,'_civ_original_step',TADEnv.step)
    TADEnv._civ_original_step=original
    calls=dict(count=0)
    if checkpoint is None:TADEnv.step=original;return calls
    opponent=r.FrozenSourceAttacker(read(checkpoint))
    def step(environment,rl_action,controller='Boids',att_action=None,**kwargs):
        if att_action is None:
            att_action=opponent(r.public_attacker_observation(environment));calls['count']+=1
        return original(environment,rl_action,controller,att_action,**kwargs)
    TADEnv.step=step
    return calls

def adaptation_evaluation(policy_path,attacker_path,stage):
    path=OUTPUT/'adaptation'/f'evaluation-{stage}.pth'
    legacy=OUTPUT/'adaptation'/f'{stage}.pth'
    if legacy.exists() and 'rows' in read(legacy):path=legacy
    if path.exists():
        old=read(path)
        if (old['policy_sha256']!=digest(policy_path) or
                old['attacker_sha256']!=(digest(attacker_path) if attacker_path else None)):
            raise ValueError('A resumed adaptation evaluation changed a frozen policy.')
        return old
    specification=adaptation_design()
    numpy_state=np.random.get_state();torch_state=torch.get_rng_state();cuda_state=torch.cuda.get_rng_state_all()
    try:
        policy=g.load(policy_path,'cuda:0');calls=opponent_binding(attacker_path)
        rows=run_streamed_episodes(policy,specification['evaluation_tasks'],batch_size=64,progress=True)
        if attacker_path and calls['count']<=0:raise ValueError('The learned attacker was not executed.')
    finally:
        opponent_binding(None);np.random.set_state(numpy_state);torch.set_rng_state(torch_state);torch.cuda.set_rng_state_all(cuda_state)
    value=dict(completed=True,stage=stage,tasks=specification['evaluation_tasks'],rows=rows,
        policy=str(policy_path),policy_sha256=digest(policy_path),attacker=str(attacker_path) if attacker_path else None,
        attacker_sha256=digest(attacker_path) if attacker_path else None,opponent_action_calls=calls['count'],success=sum(x['success'] for x in rows),
        collision=sum(x['collision'] for x in rows),inference_device='cuda:0',
        design_sha256=digest(OUTPUT/'adaptation-design.pth'),runtime=c.runtime_identity(),driver_sha256=digest(__file__))
    path.parent.mkdir(parents=True,exist_ok=True);_atomic_save(value,path)
    print('ADAPTATION_EVALUATION',stage,value['success'],len(rows),flush=True)
    return value

def train_attacker(args):
    r=opponent_helpers()
    from envs.TADgame import TADEnv
    from utils.config import _dict_to_namespace
    specification=adaptation_design();folder=OUTPUT/'adaptation';folder.mkdir(parents=True,exist_ok=True)
    final=folder/'attacker-final.pth'
    if final.exists():return read(final)
    model=specification['defender']['path'];policy=g.load(model,'cuda:0');executor=c.CIVExecutor(policy)
    initial=read(specification['initial_training']['path'])
    if not initial['completed']:raise ValueError('Attacker continuation needs a completed original learner.')
    configuration=copy.deepcopy(initial['specification']['config']);configuration['rl']['batch_size']=specification['batch_size']
    source=c.d.numerical_source();agent=source.SAC(_dict_to_namespace(configuration),2,6,2,attacker=True,device=torch.device('cuda:0'))
    for key in ('actor','critic','critic_target','actor_optimizer','critic_optimizer','alpha_optimizer'):
        getattr(agent,key).load_state_dict(initial['agent'][key])
    with torch.no_grad():agent.log_alpha.copy_(initial['agent']['log_alpha'])
    agent.alpha=agent.log_alpha.exp().detach().to(agent.device)
    original_step=initial['step'];original_episodes=initial['episodes'];del initial
    replay=sys.modules[source.SAC.__module__].ReplayBuffer(8,2)
    np.random.seed(2026101507);torch.manual_seed(2026101507)
    step=0;episodes=0;updates=0;serial=0;stage_index=0;started=time.monotonic()
    def initialize():
        nonlocal serial
        seed=99500000+serial;serial+=1;np.random.seed(seed)
        env=TADEnv(6,protocol=policy.source.environment_protocol)
        obs,_=env.reset(agility=2.+.5*(serial%2),noisy_agility=False)
        return dict(environment=env,observation=obs,state=c.ExecutionState(),random=np.random.get_state())
    jobs=[initialize() for _ in range(specification['vector_environments'])]
    def save_stage(name):
        path=folder/f'attacker-{name}.pth'
        actor={k:v.detach().cpu().clone() for k,v in agent.actor.state_dict().items()}
        if path.exists():
            old=read(path)
            if old['additional_steps']!=step or any(not torch.equal(actor[k],old['actor'][k]) for k in actor):
                raise ValueError('A completed attacker stage must retain its original weights.')
            return path
        record=dict(completed=True,actor=actor,specification=dict(configuration=configuration,config=configuration,
            source_commit=c.d.SOURCE_COMMIT,defender_sha256=digest(model),inference_device='cuda:0',
            additional_steps=step,initial_training_sha256=specification['initial_training']['sha256'],
            continuation='Actor, critics, target critic, optimizers and entropy resumed; replay recollected against frozen unified CIV.'),
            step=original_step+step,additional_steps=step,episodes=original_episodes+episodes,gradient_updates=updates,
            design_sha256=digest(OUTPUT/'adaptation-design.pth'),driver_sha256=digest(__file__))
        _atomic_save(record,path)
        print('ATTACKER_STAGE',name,step,episodes,updates,time.monotonic()-started,flush=True)
        return path
    zero=save_stage('0')
    adaptation_evaluation(model,None,'original-apf-before')
    adaptation_evaluation(model,zero,'attacker-0')
    while jobs:
        packet=c.d.mechanism_stack([c.d.joint_packet(j['environment'],j['observation']) for j in jobs],'cuda:0')
        commands,_=executor.actions(packet,[j['state'] for j in jobs])
        before=np.stack([r.public_attacker_observation(j['environment']) for j in jobs])
        with torch.no_grad():attacks=agent.actor(torch.as_tensor(before,dtype=torch.float32,device='cuda:0'),False,False)[0].cpu().numpy()
        following=[]
        for i,job in enumerate(jobs):
            np.random.set_state(job['random']);env=job['environment']
            obs,_,done,_=c.d.joint_task_step(env,commands[i],policy.source.training_gamma,policy.source.training_step_cost,attacker_action=attacks[i])
            side=env.LearningSide
            try:env.LearningSide='Att';reward=float(env._get_rewards(done))
            finally:env.LearningSide=side
            replay.store(before[i],attacks[i],reward,r.public_attacker_observation(env),bool(done))
            job['observation']=obs;job['state'].step+=1;job['random']=np.random.get_state();step+=1
            if done:
                episodes+=1
                if step<specification['additional_attacker_steps']:following.append(initialize())
            else:following.append(job)
        jobs=following
        if replay.size>=specification['warm_steps']:
            with torch.enable_grad():
                for _ in range(specification['updates_per_vector_step']):agent.learn(replay);updates+=1
        while stage_index<len(specification['attacker_stages'])-1 and step>=specification['attacker_stages'][stage_index]:
            threshold=specification['attacker_stages'][stage_index];stage_index+=1
            path=save_stage(str(threshold));adaptation_evaluation(model,path,f'attacker-{threshold}')
    final=save_stage('final')
    learner={key:getattr(agent,key).state_dict() for key in ('actor','critic','critic_target','actor_optimizer','critic_optimizer','alpha_optimizer')}
    learner['log_alpha']=agent.log_alpha.detach().cpu()
    _atomic_save(dict(completed=True,agent=learner,additional_steps=step,gradient_updates=updates,
        original_steps=original_step,episodes=episodes,design_sha256=digest(OUTPUT/'adaptation-design.pth')),
        folder/'attacker-training.pth')
    adaptation_evaluation(model,final,'attacker-final')

def adapt_defender(args):
    from civ_rl_experiment import collect_shard
    specification=adaptation_design();folder=OUTPUT/'adaptation';attacker=folder/'attacker-final.pth'
    if not attacker.exists():raise ValueError('Complete actual attacker continuation first.')
    dataset=folder/'paired-data.pth';model=specification['defender']['path'];previous=read(model)
    if not dataset.exists():
        calls=opponent_binding(attacker)
        try:
            result=collect_shard((previous['source_path'],previous['config'],specification['paired_training_tasks'],
                specification['paired_futures'],'cuda:0',False))
            if calls['count']<=0:raise ValueError('Paired outcomes did not execute the continued attacker.')
        finally:opponent_binding(None)
        for root in result['roots']:root['split']='validation' if root['scene'][0]%5==0 else 'train'
        record=dict(completed=True,algorithm='CIV-RL-unified-v1',source_path=previous['source_path'],source_sha256=previous['source_sha256'],
            config=previous['config'],execution_identity=previous['execution_identity'],tasks=specification['paired_training_tasks'],
            **result,attacker_sha256=digest(attacker),opponent_action_calls=calls['count'],execution_backend='cuda',design_sha256=digest(OUTPUT/'adaptation-design.pth'),
            paired_source='Fresh complete factual and alternative branches under the continued attacker; original source and current CI predecessor.')
        _atomic_save(record,dataset)
    updated=folder/'defender-updated.pth'
    if not updated.exists() or not read(updated)['fit'].get('search_completed'):
        command=[sys.executable,'-u','-B','-X','utf8','train/civ_generalization.py','fit','--data',str(dataset),
            '--kind','conditional','--resume',str(model),'--updates',str(specification['defender_updates']),
            '--members','3','--width','64','--dropout','.04','--weight-decay','.0001','--learning-rate','.0001',
            '--batch-size','48','--gain-weight','2','--rank-weight','0','--normalize','--hash-split','--zero-uncertainty',
            '--loss-mode','paired_calibrated','--seed','202610271508','--output',str(updated)]
        subprocess.run(command,cwd=ROOT,check=True)
    adaptation_evaluation(updated,attacker,'defender-updated')
    adaptation_evaluation(updated,None,'original-apf-after')

def learning(args):
    """Reproduce actual CV states and evaluate each full closed-loop policy."""
    folder=OUTPUT/'learning';folder.mkdir(parents=True,exist_ok=True)
    reference=read(design()['models']['CIV']['path']);reproduced=folder/'conditional.pth'
    if not reproduced.exists() or not read(reproduced)['fit'].get('search_completed'):
        command=[sys.executable,'-u','-B','-X','utf8','train/civ_generalization.py','fit','--data',
            reference['fit']['data_sources'][0]['path'],'--kind','conditional','--updates','4500',
            '--members','3','--width','64','--dropout','.04','--weight-decay','.0001','--learning-rate','.0003',
            '--batch-size','48','--gain-weight','2','--rank-weight','0','--normalize','--hash-split','--zero-uncertainty',
            '--loss-mode','paired_calibrated','--seed','2026103111','--checkpoint-every','1500','--save-stages','--output',str(reproduced)]
        subprocess.run(command,cwd=ROOT,check=True)
    record=read(reproduced)
    differences=[float((a[k]-b[k]).abs().max()) for a,b in zip(record['networks'],reference['networks']) for k in a]
    match=dict(completed=True,max_parameter_difference=max(differences),
        selected_update_match=record['fit']['updates']==reference['fit']['updates'],
        source_policy_sha256=design()['models']['CIV']['sha256'],reproduced_sha256=digest(reproduced))
    _atomic_save(match,folder/'reproduction.pth')
    if match['max_parameter_difference']!=0. or not match['selected_update_match']:
        raise ValueError('Reproduced learning states do not match the selected final fit.')
    for update in (1500,3000,4500):
        stage=folder/f'conditional-update{update}.pth'
        values=argparse.Namespace(cell='confirmation',method=f'learning-{update}',model=str(stage),margin=.075,multiplier=None)
        evaluate(values)
    print('LEARNING_CURVE_COMPLETE',flush=True)


def retain_defender(args):
    """A second, explicit continuation with original-task replay for retention."""
    folder=OUTPUT/'adaptation';first=folder/'defender-updated.pth';target=folder/'paired-data.pth'
    specification=adaptation_design();original=read(specification['defender']['path'])
    old=Path(original['fit']['data_sources'][0]['path']);design_path=folder/'retention-design.pth'
    protocol=dict(completed=True,initial=dict(path=str(first),sha256=digest(first)),
        current_attacker_data=dict(path=str(target),sha256=digest(target)),
        original_apf_replay=dict(path=str(old),sha256=digest(old)),
        original_task_prefix=192,updates_per_member=3000,learning_rate=.00003,seed=202610271607,
        purpose='Continue the first selected defender update with both original-APF retention labels and current-attacker labels; do not relabel original outcomes as new-attacker evidence.',
        selection='Same scene-hash validation and paired-gain criterion across the two explicitly identified task sources; evaluation outcomes are not model-selection inputs.',
        motivation='The first continuation did not improve new-attacker success and reduced original-task retention; retain and report that first result.')
    if design_path.exists():
        if read(design_path)!=protocol:raise ValueError('Retention continuation design changed.')
    else:_atomic_save(protocol,design_path)
    updated=folder/'defender-retained.pth'
    if not updated.exists() or not read(updated)['fit'].get('search_completed'):
        command=[sys.executable,'-u','-B','-X','utf8','train/civ_generalization.py','fit',
            '--data',str(old),str(target),'--old-scene-limit','192','--kind','conditional',
            '--resume',str(first),'--updates','3000','--members','3','--width','64','--dropout','.04',
            '--weight-decay','.0001','--learning-rate','.00003','--batch-size','48','--gain-weight','2',
            '--rank-weight','0','--normalize','--hash-split','--zero-uncertainty','--loss-mode','paired_calibrated',
            '--seed','202610271607','--output',str(updated)]
        subprocess.run(command,cwd=ROOT,check=True)
    adaptation_evaluation(updated,folder/'attacker-final.pth','defender-retained')
    adaptation_evaluation(updated,None,'original-apf-retained')
    print('RETENTION_CONTINUATION_COMPLETE',flush=True)

def illustrations(args):
    path=OUTPUT/'illustrations.pth'
    if path.exists():return read(path)
    from envs.TADgame import TADEnv
    selected=read(SELECTED/'selected-calibrated-comparison.pth')
    index=next(i for i,(a,b) in enumerate(zip(selected['rows']['CIV'],selected['rows']['CI'])) if a['success'] and not b['success'])
    task=selected['tasks'][index];policy,policy_path=controller(design(),'CIV')
    prefix=c.run_episodes(policy,[task],collect_roots=True,stop_at_root=True)[0]
    root=prefix['root'];environment=root['environment']
    packet=c.d.mechanism_tensor(c.d.joint_packet(environment,root['observation']),'cpu')
    source,catalog,features,decision=c.fixed_inference(policy.decide,packet,policy.config.inference_width)
    choice=int(decision['choice'][0]);expected=selected['rows']['CIV'][index]['selected']
    if choice!=expected:raise ValueError('The illustrated decision differs from its completed endpoint.')
    seed,n,agility=task['scene'];np.random.seed(seed);initial=TADEnv(n,protocol=policy.source.environment_protocol);initial.reset(agility=agility,noisy_agility=False)
    def boats(env):
        return dict(defenders=np.asarray([np.r_[b.pos,b.theta,b.velocity_r] for b in env.defender_list]),
            attacker=np.r_[env.attacker.pos,env.attacker.theta,env.attacker.velocity_r])
    record=dict(completed=True,task=task,selected_task_index=index,choice=choice,
        case_rule='First rescue in the frozen order of the selected development cohort; geometry illustration only.',
        initial=boats(initial),decision_state=boats(environment),decision_time=environment.Current_T,
        catalog={k:v[0].detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in catalog.items()},
        features={k:v[0].detach().cpu().numpy() for k,v in features.items()},
        decision={k:v[0].detach().cpu().numpy() for k,v in decision.items()},
        endpoints={k:selected['rows'][k][index] for k in ('CIV','CI','ordinary')},
        policy_sha256=digest(policy_path),selected_comparison_sha256=digest(SELECTED/'selected-calibrated-comparison.pth'))
    _atomic_save(record,path);print('ILLUSTRATION_GEOMETRY_COMPLETE',index,choice,flush=True)
    return record

def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    p=argparse.ArgumentParser();p.add_argument('command',choices=('design','evaluate','vrx','train-attacker','adapt-defender','retain-defender','learning','illustrations'))
    p.add_argument('--method',default='CIV');p.add_argument('--cell',default='confirmation')
    p.add_argument('--model');p.add_argument('--margin',type=float);p.add_argument('--multiplier',type=float)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--vrx-device',default='cuda:0')
    args=p.parse_args()
    {'design':lambda _:design(),'evaluate':evaluate,'vrx':vrx,'train-attacker':train_attacker,'adapt-defender':adapt_defender,'retain-defender':retain_defender,'learning':learning,'illustrations':illustrations}[args.command](args)

if __name__=='__main__':main()
