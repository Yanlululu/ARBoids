"""Fresh paired-data pilot for the unified CIV-RL deployment interface."""
import argparse
import copy
from dataclasses import asdict,replace
import hashlib
import json
import os
import time
import itertools
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import civ_rl as c


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_tasks(seed,count,n=3,agilities=(2.,2.5)):
    return [dict(scene=(seed+i,n,float(agilities[i%len(agilities)])),future=seed+1000000+i) for i in range(count)]


def collect_shard(arguments):
    source_path,settings,tasks,future_count,device,include_factual_future=arguments
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    config=c.CIVConfig(**settings);controller=c.CIVController(c.load_source(source_path,device),config)
    rows=c.run_episodes(controller,tasks,collect_roots=True)
    roots=[]
    for row in rows:
        if row['root'] is None:continue
        root=row.pop('root');root['factual']=row
        packet=c.d.mechanism_tensor(c.d.joint_packet(root['environment'],root['observation']),device)
        _,_,features,_=c.fixed_inference(controller.decide,packet,config.inference_width)
        root['features']={key:value[0].cpu().numpy() for key,value in features.items()}
        roots.append(root)
    branch_tasks=[];initial_jobs=[];mapping=[]
    for r,root in enumerate(roots):
        feature=root['features'];valid=np.flatnonzero(feature['valid'])
        root['targets']=np.full((future_count,len(feature['valid'])),-1.,dtype=np.float32)
        root['terminal_outcomes']=np.full_like(root['targets'],-1,dtype=np.int64)
        root['futures']=[int(root['scene'][0]+2000000+future*10000) for future in range(future_count)]
        if include_factual_future:root['futures'][0]=None
        for future,seed in enumerate(root['futures']):
            for option in valid:
                branch_tasks.append(dict(scene=root['scene'],future=seed,program=dict(choice=int(option),
                    gates=feature['gates'][option],duration=int(feature['durations'][option]))))
                initial_jobs.append(root);mapping.append((r,future,int(option)))
    results=c.run_episodes(controller,branch_tasks,initial_jobs=initial_jobs,progress=True)
    for (r,future,option),row in zip(mapping,results):
        roots[r]['targets'][future,option]=row['success']
        roots[r]['terminal_outcomes'][future,option]=row['outcome']
    return dict(roots=roots,factual_rows=rows,simulation_steps=sum(row['simulation_steps'] for row in rows+results),
        branch_count=len(results),runtime=c.runtime_identity())


def collect(args):
    path=Path(args.output)
    if path.exists():raise ValueError('A completed or interrupted dataset must not be silently replaced.')
    source=Path(args.source).resolve();settings=asdict(c.CIVConfig(generator=args.generator,decision_step=args.decision_step))
    tasks=make_tasks(args.seed,args.roots,args.vessels,args.agilities)
    chunks=[tasks[i::args.workers] for i in range(args.workers)]
    work=[(str(source),settings,chunk,args.futures,args.device,args.include_factual_future) for chunk in chunks if chunk]
    if args.workers==1:parts=[collect_shard(work[0])]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:parts=list(pool.map(collect_shard,work))
    if any(part['runtime']!=parts[0]['runtime'] for part in parts):raise RuntimeError('Collector runtime mismatch.')
    roots=sorted([root for part in parts for root in part['roots']],key=lambda root:tuple(root['scene']))
    validation_scenes={tuple(task['scene']) for task in tasks[-args.validation:]}
    for root in roots:root['split']='validation' if tuple(root['scene']) in validation_scenes else 'train'
    record=dict(algorithm='CIV-RL-unified-v1',completed=True,execution_identity=c.CIVConfig(**settings).identity(),
        source_path=str(source),source_sha256=digest(source),config=settings,runtime=parts[0]['runtime'],tasks=tasks,
        roots=roots,factual_rows=[row for part in parts for row in part['factual_rows']],
        include_factual_future=args.include_factual_future,driver_sha256=digest(__file__),
        simulation_steps=sum(part['simulation_steps'] for part in parts),branch_count=sum(part['branch_count'] for part in parts))
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(record,path)
    for split in ('train','validation'):
        selected=[root for root in roots if root['split']==split]
        targets=np.concatenate([root['targets'] for root in selected]);valid=np.concatenate(
            [np.broadcast_to(root['features']['valid'],root['targets'].shape) for root in selected])
        baseline=targets[:,:1]
        print(json.dumps(dict(split=split,roots=len(selected),factual_success=float(baseline.mean()),
            rescue_pairs=int(((baseline==0)&(targets==1)&valid).sum()),
            harm_pairs=int(((baseline==1)&(targets==0)&valid).sum()))),flush=True)
    print(json.dumps(dict(completed=str(path),branches=record['branch_count'],simulation_steps=record['simulation_steps'])),flush=True)


def tensor_features(roots,device):
    return {key:torch.as_tensor(np.stack([root['features'][key] for root in roots]),device=device)
        for key in ('node','edge','global_features','valid')}


def fit_objective(logits,targets,valid,kind,gain_weight=0.):
    if kind=='ordinary':
        if gain_weight:raise ValueError('The ordinary control does not use paired-gain supervision.')
        return c.ordinary_loss(logits,targets,valid)
    loss=c.conditional_loss(logits,targets,valid)
    if gain_weight:
        p0=logits[:,0:1,0].sigmoid()
        gain=(1.-p0)*logits[...,1].sigmoid()-p0*logits[...,2].sigmoid()
        mask=valid.clone();mask[:,0]=False
        paired=((gain-(targets-targets[:,:1])).square()*mask).sum()/mask.sum().clamp_min(1)
        loss=loss+gain_weight*paired
    return loss


@torch.no_grad()
def validation_choices(networks,features,roots,kind,margin,penalty):
    gain,uncertainty=c.ensemble_values(networks,features,kind)
    selected,_=c.choose_options(gain,uncertainty,features['valid'],margin,penalty)
    values=[];rescues=harms=changes=0
    for root,choice in zip(roots,selected.cpu().tolist()):
        y0=root['targets'][:,0];yz=root['targets'][:,choice]
        values.extend(yz-y0);rescues+=int(((y0==0)&(yz==1)).sum());harms+=int(((y0==1)&(yz==0)).sum())
        changes+=int(choice>0)
    return dict(gain=float(np.mean(values)),rescues=rescues,harms=harms,changed_roots=changes,
        margin=margin,choices=selected.cpu().tolist())


def fit(args):
    data=torch.load(args.data,map_location='cpu',weights_only=False)
    if not data['completed'] or data['algorithm']!='CIV-RL-unified-v1':raise ValueError('Fresh completed unified data required.')
    config=c.CIVConfig(**data['config'])
    if config.identity()!=data['execution_identity']:raise ValueError('Dataset execution semantics mismatch.')
    if digest(data['source_path'])!=data['source_sha256']:raise ValueError('Dataset source changed.')
    training=[root for root in data['roots'] if root['split']=='train']
    validation=[root for root in data['roots'] if root['split']=='validation']
    if set(tuple(root['scene']) for root in training)&set(tuple(root['scene']) for root in validation):
        raise ValueError('Whole initial scenes must remain on one side of validation.')
    for root in data['roots']:
        valid=root['features']['valid']
        if not np.isin(root['targets'][:,valid],(0.,1.)).all():raise ValueError('Incomplete closed-loop labels.')
    torch.set_num_threads(1);torch.set_grad_enabled(True);torch.backends.cuda.matmul.allow_tf32=False
    train_features=tensor_features(training,args.device);val_features=tensor_features(validation,args.device)
    targets=torch.as_tensor(np.stack([root['targets'] for root in training]),device=args.device)
    networks=[];optimizers=[];randoms=[]
    for member in range(args.members):
        torch.manual_seed(args.seed+member)
        net=c.CIVValueNetwork().to(args.device);networks.append(net)
        optimizers.append(torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4))
        randoms.append(np.random.default_rng(args.seed+member))
    if args.gain_weight<0:raise ValueError('Paired-gain supervision needs a nonnegative weight.')
    history=[];best=None;best_state=None
    checkpoints=sorted(set((max(1,args.updates//4),max(1,args.updates//2),args.updates)))
    for update in range(1,args.updates+1):
        losses=[]
        for net,opt,rng in zip(networks,optimizers,randoms):
            index=torch.as_tensor(rng.integers(0,len(training),min(32,len(training))),device=args.device)
            future=torch.as_tensor(rng.integers(0,targets.shape[1],len(index)),device=args.device)
            features={key:value[index] for key,value in train_features.items()}
            target=targets[index,future].clamp_min(0.)
            loss=fit_objective(net(features),target,features['valid'],args.kind,args.gain_weight)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step()
            losses.append(float(loss.detach()))
        if update in checkpoints:
            for margin in (0.,.05,.1,.2):
                score=validation_choices(networks,val_features,validation,args.kind,margin,config.uncertainty_penalty)
                score.update(update=update,loss=float(np.mean(losses)))
                history.append(score)
                # Identical selection rule for conditional and ordinary value.
                rank=(score['gain'],-score['harms'],-score['changed_roots'],-update,margin)
                if best is None or rank>best:
                    best=rank;best_score=score.copy();best_state=[copy.deepcopy(net.state_dict()) for net in networks]
            print(json.dumps(dict(update=update,loss=float(np.mean(losses)),best=best_score)),flush=True)
    for net,state in zip(networks,best_state):net.load_state_dict(state);net.eval()
    config=replace(config,acceptance_margin=best_score['margin'])
    fit_record=dict(completed=True,training_roots=len(training),validation_roots=len(validation),
        training_fleets=sorted({int(root['scene'][1]) for root in training}),
        member_seeds=[args.seed+i for i in range(args.members)],updates=args.updates,selected=best_score,
        parameter_count=sum(p.numel() for p in networks[0].parameters()),data_path=str(Path(args.data).resolve()),
        data_sha256=digest(args.data),runtime=c.runtime_identity(),validation_history=history,
        paired_gain_weight=args.gain_weight,learning_objective=(
            'BCE(p0,Y0)+(1-Y0)BCE(r,Yz)+Y0*BCE(h,1-Yz)+lambda*MSE(Delta,Yz-Y0)'
            if args.kind=='conditional' else 'Mean of three BCE(Q,Yz) heads'))
    c.save_checkpoint(args.output,data['source_path'],config,networks,args.kind,fit_record)
    print(json.dumps(dict(completed=args.output,fit=best_score)),flush=True)


def evaluate(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    if args.model:controller=c.load_controller(args.model,args.device)
    else:controller=c.CIVController(c.load_source(args.source,args.device),c.CIVConfig(generator=args.generator))
    tasks=make_tasks(args.seed,args.tasks,args.vessels,args.agilities)
    rows=c.run_episodes(controller,tasks,progress=True)
    record=dict(completed=True,algorithm='CIV-RL-unified-v1',model=args.model,
        policy_sha256=digest(args.model or args.source),execution_identity=controller.config.identity(),
        config=asdict(controller.config),runtime=c.runtime_identity(),tasks=tasks,rows=rows,
        success=sum(row['success'] for row in rows),collision=sum(row['collision'] for row in rows),
        selected=sum(row['selected']>0 for row in rows),simulation_steps=sum(row['simulation_steps'] for row in rows))
    Path(args.output).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({key:record[key] for key in ('completed','success','collision','selected','simulation_steps')}),flush=True)


def _atomic_save(record,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.partial-{os.getpid()}')
    try:
        torch.save(record,temporary);os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def hard_design(args):
    """Freeze independent initial scenes before any terminal outcomes are observed."""
    from envs.TADgame import TADEnv
    destination=Path(args.output)
    if destination.exists():raise ValueError('The hard-scene design is already frozen.')
    policy=torch.load(args.model,map_location='cpu',weights_only=False)
    source=c.load_source(policy['source_path'],'cpu');environment_protocol=source.environment_protocol;del source
    strata=[(ag,bearing) for ag in (3.,3.5,4.) for bearing in range(3)]
    def scenes(start,count,split):
        quotas={s:count//len(strata)+(i<count%len(strata)) for i,s in enumerate(strata)}
        used={s:0 for s in strata};tasks=[];seed=start
        while len(tasks)<count:
            ag=(3.,3.5,4.)[(seed-start)%3]
            np.random.seed(seed);env=TADEnv(6,protocol=environment_protocol)
            env.reset(agility=ag,noisy_agility=False)
            headings=np.array([d.theta for d in env.defender_list])
            direction=np.arctan2(np.sin(headings).mean(),np.cos(headings).mean())
            bearing=np.arctan2(env.attacker.pos[1],env.attacker.pos[0])
            angle=abs(np.arctan2(np.sin(direction-bearing),np.cos(direction-bearing)))
            sector=min(2,int(angle/(np.pi/3)));stratum=(ag,sector)
            if used[stratum]<quotas[stratum]:
                initial=np.concatenate((env.attacker.pos,[env.attacker.theta],
                    np.concatenate([np.r_[d.pos,d.theta] for d in env.defender_list])))
                tasks.append(dict(scene=(seed,6,ag),future=seed+1000000,split=split,
                    stratum=stratum,initial_signature=hashlib.sha256(initial.tobytes()).hexdigest()))
                used[stratum]+=1
            seed+=1
        return tasks
    train=scenes(args.seed,4000,'train');validation=scenes(args.seed+10000000,800,'validation')
    test=scenes(args.seed+30000000,2000,'test');all_tasks=train+validation+test
    assert len({t['scene'][0] for t in all_tasks})==len(all_tasks)
    assert len({t['initial_signature'] for t in all_tasks})==len(all_tasks)
    record=dict(completed=True,source_path=policy['source_path'],source_sha256=policy['source_sha256'],
        config=policy['config'],execution_identity=policy['execution_identity'],execution_backend='cuda',
        environment_protocol=environment_protocol,
        strata=strata,training_roots=3000,validation_roots=600,test_scenes=2000,futures_per_root=3,
        prefix_tasks=train+validation,test_tasks=test,source_policy=str(Path(args.model).resolve()),
        source_policy_sha256=digest(args.model),runtime=c.runtime_identity(),driver_sha256=digest(__file__),
        protocol='Six vessels; attacker agility 3.0/3.5/4.0; equal initial toward/cross/away heading strata; unchanged source environment.',
        reserve_prefixes_only=True,outcome_based_scene_selection=False)
    _atomic_save(record,destination)
    print(json.dumps(dict(frozen=str(destination),training=3000,validation=600,test=2000,prefix_pool=len(train+validation))),flush=True)


def hard_prefix(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    design=torch.load(args.design,map_location='cpu',weights_only=False)
    assert digest(design['source_path'])==design['source_sha256']
    controller=c.CIVController(c.load_source(design['source_path'],args.device),c.CIVConfig(**design['config']))
    tasks=design['prefix_tasks'][args.shard::args.workers]
    rows=c.run_episodes(controller,tasks,batch_size=args.batch_size,collect_roots=True,stop_at_root=True,progress=True)
    for task,row in zip(tasks,rows):
        if row['root'] is not None:
            row['root'].update(split=task['split'],stratum=task['stratum'],initial_signature=task['initial_signature'])
    _atomic_save(dict(completed=True,rows=rows,tasks=tasks,design_sha256=digest(args.design),runtime=c.runtime_identity()),args.output)
    print('PREFIX_COMPLETE',args.shard,len(rows),sum(r['root'] is not None for r in rows),flush=True)


def hard_roots(args):
    design=torch.load(args.design,map_location='cpu',weights_only=False);folder=Path(args.output).parent
    parts=[torch.load(folder/f'prefix-{i:02d}.pth',weights_only=False) for i in range(args.workers)]
    assert all(p['completed'] and p['design_sha256']==digest(args.design) and p['runtime']==design['runtime'] for p in parts)
    pool=[r['root'] for p in parts for r in p['rows'] if r['root'] is not None];roots=[]
    rng=np.random.default_rng(args.seed)
    for split,total in [('train',3000),('validation',600)]:
        for i,stratum in enumerate(design['strata']):
            eligible=sorted([r for r in pool if r['split']==split and tuple(r['stratum'])==tuple(stratum)],key=lambda r:tuple(r['scene']))
            needed=total//len(design['strata'])+(i<total%len(design['strata']))
            if len(eligible)<needed:raise ValueError(f'Insufficient actionable prefixes in {split}/{stratum}: {len(eligible)} < {needed}')
            chosen=[eligible[j] for j in rng.permutation(len(eligible))[:needed]]
            roots.extend(chosen)
    # Freeze nested 1000/2000/3000-scene training budgets independently of labels.
    for split in ('train','validation'):
        selected=[r for r in roots if r['split']==split]
        for order,index in enumerate(rng.permutation(len(selected))):selected[index]['split_order']=int(order)
    roots.sort(key=lambda r:(0 if r['split']=='validation' else 1,r['split_order']))
    record=dict(design,roots=roots,prefix_acquisition_complete=True,terminal_labels_complete=False,
        excluded_early_terminal=sum(r['root'] is None for p in parts for r in p['rows']),
        prefix_sources=[dict(path=str(folder/f'prefix-{i:02d}.pth'),sha256=digest(folder/f'prefix-{i:02d}.pth')) for i in range(args.workers)])
    _atomic_save(record,args.output);print('ROOT_BANK_COMPLETE',len(roots),record['excluded_early_terminal'],flush=True)


class RelativeFeedbackState(c.ExecutionState):
    """Hold a candidate's change while retaining the live CI reference."""
    def __init__(self,*args,reference_gates=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.reference_gates=reference_gates;self.offset=None

    def commit(self,choice,gates,duration,delay):
        if choice and self.reference_gates is None:raise ValueError('A relative plan needs its root CI reference.')
        super().commit(choice,gates,duration,delay)
        if choice:self.offset=self.gates-np.asarray(self.reference_gates,dtype=np.float32)

    def coefficients(self,reference):
        if self.decided and self.choice and self.start<=self.step<self.end:
            return np.clip(reference+self.offset,0.,1.)
        return reference


class RelativeFeedbackExecutor(c.CIVExecutor):
    @torch.no_grad()
    def actions(self,packet,states,forced=None):
        due=[i for i,s in enumerate(states) if isinstance(s,RelativeFeedbackState)
             and not s.decided and s.step==self.controller.config.decision_step]
        if due:
            part={key:value[due] for key,value in packet.items()}
            _,catalog=c.fixed_inference(lambda p:self.controller.proposals(p,reference_only=True),
                part,self.controller.config.inference_width)
            for index,position in enumerate(due):states[position].reference_gates=catalog['gates'][index,0].cpu().numpy()
        return super().actions(packet,states,forced)


def run_streamed_episodes(controller,tasks,*,batch_size=128,initial_jobs=None,progress=False,intervention_mode='absolute'):
    """Refill completed slots while preserving each episode's state and RNG."""
    from envs.TADgame import TADEnv
    if not tasks:return []
    if len({task['scene'][1] for task in tasks})!=1:raise ValueError('Streaming requires a common fleet size.')
    if intervention_mode not in ('absolute','relative_feedback'):raise ValueError('Unknown intervention semantics.')
    device=next(controller.source.parameters()).device
    executor=(RelativeFeedbackExecutor if intervention_mode=='relative_feedback' else c.CIVExecutor)(controller)
    results=[None]*len(tasks);saved_random=np.random.get_state();next_index=0;completed=0;reported=0
    def initialize(index):
        task=tasks[index];scene=tuple(task['scene']);seed,n,agility=scene
        mode=task.get('program',{}).get('intervention_mode',intervention_mode)
        if mode not in ('absolute','relative_feedback'):raise ValueError('Unknown branch semantics.')
        if mode=='relative_feedback' and intervention_mode!='relative_feedback':
            raise ValueError('Relative branches require the shared relative executor.')
        state_type=RelativeFeedbackState if mode=='relative_feedback' else c.ExecutionState
        if initial_jobs is None:
            np.random.seed(seed);env=TADEnv(n,protocol=controller.source.environment_protocol)
            observation,_=env.reset(agility=agility,noisy_agility=False);state=state_type()
        else:
            root=initial_jobs[index];env=copy.deepcopy(root['environment']);observation=root['observation'].copy()
            np.random.set_state(root['random_state']);state=state_type(step=root['step'])
            if mode=='relative_feedback':state.reference_gates=root['features']['gates'][0]
            if 'program' in task:
                program=task['program'];state.commit(program['choice'],program['gates'],program['duration'],controller.config.delay)
        if task.get('future') is not None:np.random.seed(task['future'])
        return dict(index=index,env=env,obs=observation,rng=np.random.get_state(),state=state,decision=None,
            simulation_steps=0,discounted_return=0.)
    try:
        active=[initialize(i) for i in range(min(batch_size,len(tasks)))];next_index=len(active)
        while active:
            packet=c.d.mechanism_stack([c.d.joint_packet(job['env'],job['obs']) for job in active],str(device))
            commands,decisions=executor.actions(packet,[job['state'] for job in active]);following=[]
            for index,job in enumerate(active):
                np.random.set_state(job['rng'])
                observation,reward,done,_=c.d.joint_task_step(job['env'],commands[index],controller.source.training_gamma,controller.source.training_step_cost)
                job['discounted_return']+=controller.source.training_gamma**job['state'].step*reward
                job['rng']=np.random.get_state();job['obs']=observation;job['state'].step+=1;job['simulation_steps']+=1
                if index in decisions:job['decision']=decisions[index]
                if done:
                    task=tasks[job['index']]
                    results[job['index']]=dict(scene=task['scene'],future=task.get('future'),**c.d.outcome(job['env'],done),
                        simulation_steps=job['simulation_steps'],decision=job['decision'],selected=job['state'].choice,
                        discounted_return=job['discounted_return'])
                    completed+=1
                    if next_index<len(tasks):following.append(initialize(next_index));next_index+=1
                else:
                    if job['state'].step>int(round(job['env'].Total_T/job['env'].Action_T))+2:raise RuntimeError('Missing terminal event.')
                    following.append(job)
            active=following
            if progress and (completed-reported>=batch_size or completed==len(tasks)):
                print(json.dumps(dict(completed=completed,total=len(tasks),successes=sum(r['success'] for r in results if r is not None))),flush=True)
                reported=completed
        assert all(r is not None for r in results)
        return results
    finally:np.random.set_state(saved_random)


def hard_collect(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    bank=torch.load(args.design,map_location='cpu',weights_only=False);folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    assert bank['prefix_acquisition_complete'] and digest(bank['source_path'])==bank['source_sha256']
    controller=c.CIVController(c.load_source(bank['source_path'],args.device),c.CIVConfig(**bank['config']))
    size=32
    for chunk in range(args.shard,(len(bank['roots'])+size-1)//size,args.workers):
        path=folder/f'branches-{chunk:03d}.pth'
        if path.exists():
            existing=torch.load(path,map_location='cpu',weights_only=False)
            if not existing['completed'] or existing['root_bank_sha256']!=digest(args.design):raise ValueError('Incompatible branch shard.')
            continue
        roots=copy.deepcopy(bank['roots'][chunk*size:(chunk+1)*size]);tasks=[];initial=[];mapping=[]
        for index,root in enumerate(roots):
            packet=c.d.mechanism_tensor(c.d.joint_packet(root['environment'],root['observation']),args.device)
            _,_,features,_=c.fixed_inference(controller.decide,packet,controller.config.inference_width)
            feature={key:value[0].cpu().numpy() for key,value in features.items()};root['features']=feature
            root['futures']=[1000000000+int(root['scene'][0])*8+j for j in range(3)]
            root['targets']=np.full((3,len(feature['valid'])),-1.,dtype=np.float32)
            root['terminal_outcomes']=np.full_like(root['targets'],-1,dtype=np.int64)
            for future,seed in enumerate(root['futures']):
                for option in np.flatnonzero(feature['valid']):
                    tasks.append(dict(scene=root['scene'],future=seed,program=dict(choice=int(option),gates=feature['gates'][option],duration=int(feature['durations'][option]))))
                    initial.append(root);mapping.append((index,future,int(option)))
        rows=run_streamed_episodes(controller,tasks,batch_size=args.batch_size,initial_jobs=initial,progress=True)
        for (index,future,option),row in zip(mapping,rows):
            roots[index]['targets'][future,option]=row['success'];roots[index]['terminal_outcomes'][future,option]=row['outcome']
        assert all(np.isin(r['targets'][:,r['features']['valid']],(0.,1.)).all() for r in roots)
        _atomic_save(dict(completed=True,chunk=chunk,roots=roots,branch_count=len(rows),root_bank_sha256=digest(args.design),
            runtime=c.runtime_identity(),execution_backend='cuda',driver_sha256=digest(__file__),streamed_slots=True),path)
        print('BRANCH_CHUNK_COMPLETE',chunk,len(roots),len(rows),flush=True)


def hard_merge(args):
    bank=torch.load(args.design,map_location='cpu',weights_only=False);folder=Path(args.output).parent
    parts=[torch.load(folder/f'branches-{i:03d}.pth',weights_only=False) for i in range((len(bank['roots'])+31)//32)]
    assert all(p['completed'] and p['root_bank_sha256']==digest(args.design) and p['runtime']==bank['runtime'] for p in parts)
    roots=[r for p in parts for r in p['roots']]
    assert len(roots)==3600 and sum(r['split']=='train' for r in roots)==3000 and len({tuple(r['scene']) for r in roots})==3600
    bank.update(roots=roots,terminal_labels_complete=True,completed=True,branch_count=sum(p['branch_count'] for p in parts),
        branch_sources=[dict(path=str(folder/f'branches-{i:03d}.pth'),sha256=digest(folder/f'branches-{i:03d}.pth')) for i in range(len(parts))])
    _atomic_save(bank,args.output);print('HARD_BANK_COMPLETE',len(roots),bank['branch_count'],flush=True)


def space_design(args):
    """Freeze an outcome-independent development sample and two future panels."""
    import civ_generalization as g
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    output=Path(args.output)
    if output.exists():raise ValueError('Do not replace a frozen candidate-space design.')
    policy=torch.load(args.model,map_location='cpu',weights_only=False)
    controller=g.load(args.model,args.device)
    tasks=make_tasks(args.seed,args.roots,6,(2.,2.5))
    observed={tuple(s) for s in policy['fit']['training_scenes']+policy['fit']['validation_scenes']}
    if any(tuple(t['scene']) in observed for t in tasks):raise ValueError('Development scene overlap.')
    reference=c.CIVController(controller.source,controller.config)
    rows=c.run_episodes(reference,tasks,collect_roots=True,stop_at_root=True,progress=True)
    roots=[]
    for index,row in enumerate(rows):
        if row['root'] is None:continue
        root=row.pop('root');root['sample_index']=index
        packet=c.d.mechanism_tensor(c.d.joint_packet(root['environment'],root['observation']),args.device)
        _,_,features,decision=c.fixed_inference(controller.decide,packet,controller.config.inference_width)
        root['features']={key:value[0].cpu().numpy() for key,value in features.items()}
        root['frozen_cv_choice']=int(decision['choice'][0])
        root['futures']=[args.seed+10000000+index*1000+j for j in range(2*args.futures)]
        roots.append(root)
    record=dict(completed=True,purpose='candidate-space development diagnosis; not formal test evidence',
        model=str(Path(args.model).resolve()),model_sha256=digest(args.model),
        source_path=policy['source_path'],source_sha256=policy['source_sha256'],
        config=asdict(controller.config),execution_identity=controller.config.identity(),
        runtime=c.runtime_identity(),execution_backend=torch.device(args.device).type,
        tasks=tasks,roots=roots,prefix_rows=rows,selection_futures=args.futures,validation_futures=args.futures,
        primary='first future panel selects; second independently evaluates every frozen choice',
        selection_budgets=[v for v in (2,4,8,16,32) if v<=args.futures],
        tie_rule='retain CI when tied; otherwise lowest candidate index',
        seed=args.seed,driver_sha256=digest(__file__))
    output.parent.mkdir(parents=True,exist_ok=True);_atomic_save(record,output)
    print('SPACE_DESIGN_COMPLETE',len(tasks),len(roots),args.futures,flush=True)


def space_collect(args):
    """Evaluate the unchanged candidate space, plus its actual deployed CV choice."""
    import civ_generalization as g
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    bank=torch.load(args.design,map_location='cpu',weights_only=False);folder=Path(args.output)
    if digest(bank['model'])!=bank['model_sha256'] or digest(bank['source_path'])!=bank['source_sha256']:
        raise ValueError('Frozen policy changed.')
    if c.runtime_identity()!=bank['runtime']:raise ValueError('Frozen execution runtime changed.')
    controller=g.load(bank['model'],args.device);count=bank['selection_futures'];size=8
    folder.mkdir(parents=True,exist_ok=True)
    for chunk in range(args.shard,(len(bank['roots'])+size-1)//size,args.workers):
        path=folder/f'space-branches-{chunk:03d}.pth'
        if path.exists():
            old=torch.load(path,map_location='cpu',weights_only=False)
            if not old['completed'] or old['design_sha256']!=digest(args.design):raise ValueError('Incompatible shard.')
            continue
        roots=copy.deepcopy(bank['roots'][chunk*size:(chunk+1)*size]);tasks=[];initial=[];mapping=[]
        for index,root in enumerate(roots):
            feature=root['features'];root['targets']=np.full((2*count,len(feature['valid'])),-1.,np.float32)
            root['deployed_cv_targets']=np.full(count,-1.,np.float32);root['deployed_cv_choices']=np.full(count,-1,np.int64)
            for future,seed in enumerate(root['futures'][:count]):
                for option in np.flatnonzero(feature['valid']):
                    tasks.append(dict(scene=root['scene'],future=seed,program=dict(choice=int(option),
                        gates=feature['gates'][option],duration=int(feature['durations'][option]))))
                    initial.append(root);mapping.append((index,future,int(option)))
        rows=run_streamed_episodes(controller,tasks,batch_size=args.batch_size,initial_jobs=initial,progress=True)
        for (index,future,option),row in zip(mapping,rows):roots[index]['targets'][future,option]=row['success']
        branch_count=len(rows);tasks=[];initial=[];mapping=[]
        for index,root in enumerate(roots):
            feature=root['features'];options={0,root['frozen_cv_choice']}
            for budget in bank['selection_budgets']:
                labels=root['targets'][:budget];delta=labels-labels[:,:1]
                mean=delta.mean(0);se=delta.std(0,ddof=1)/np.sqrt(budget)
                options.add(int(np.where(feature['valid'],mean,-np.inf).argmax()))
                options.add(int(np.where(feature['valid'],mean-se,-np.inf).argmax()))
            # Only selected actions need held-out execution. Their selection
            # is complete before any held-out future is simulated.
            root['heldout_options']=sorted(options)
            for future in range(count,2*count):
                seed=root['futures'][future]
                for option in root['heldout_options']:
                    tasks.append(dict(scene=root['scene'],future=seed,program=dict(choice=int(option),
                        gates=feature['gates'][option],duration=int(feature['durations'][option]))))
                    initial.append(root);mapping.append((index,future,int(option)))
                tasks.append(dict(scene=root['scene'],future=seed));initial.append(root);mapping.append((index,future,-1))
        rows=run_streamed_episodes(controller,tasks,batch_size=args.batch_size,initial_jobs=initial,progress=True)
        for (index,future,option),row in zip(mapping,rows):
            if option>=0:roots[index]['targets'][future,option]=row['success']
            else:
                roots[index]['deployed_cv_targets'][future-count]=row['success']
                roots[index]['deployed_cv_choices'][future-count]=row['selected']
        if not all(np.isin(r['targets'][:count,r['features']['valid']],(0.,1.)).all() and
                   np.isin(r['targets'][count:,r['heldout_options']],(0.,1.)).all() for r in roots):
            raise ValueError('Incomplete terminal labels.')
        branch_count+=len(rows)
        _atomic_save(dict(completed=True,roots=roots,branch_count=branch_count,design_sha256=digest(args.design),
            runtime=c.runtime_identity(),execution_backend=torch.device(args.device).type),path)
        print('SPACE_CHUNK_COMPLETE',chunk,len(roots),branch_count,flush=True)


def space_analyze(args):
    """Report future-separated gains with whole-scene bootstrap uncertainty."""
    design=torch.load(args.design,map_location='cpu',weights_only=False);folder=Path(args.output)
    paths=[folder/f'space-branches-{i:03d}.pth' for i in range((len(design['roots'])+7)//8)]
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    if any(not p['completed'] or p['design_sha256']!=digest(args.design) for p in parts):
        raise ValueError('All frozen development roots must finish.')
    roots=sorted([r for p in parts for r in p['roots']],key=lambda r:r['sample_index'])
    count=design['selection_futures'];rng=np.random.default_rng(design['seed']+19)
    bootstrap=rng.integers(len(design['tasks']),size=(10000,len(design['tasks'])))
    early={i:r for i,r in enumerate(design['prefix_rows']) if r.get('root',None) is None and 'success' in r}
    methods={};choices={};mismatches=[]
    def assess(name,picks):
        baseline=[];outcome=[];root_gains=np.zeros(len(design['tasks']),np.float64)
        for root,choice in zip(roots,picks):
            y0=root['targets'][count:,0];yz=root['targets'][count:,choice]
            baseline.extend(y0);outcome.extend(yz);root_gains[root['sample_index']]=float((yz-y0).mean())
        for row in early.values():baseline.extend([row['success']]*count);outcome.extend([row['success']]*count)
        y0=np.asarray(baseline);yz=np.asarray(outcome);boot=root_gains[bootstrap].mean(1)
        methods[name]=dict(success=int(yz.sum()),factual_success=int(y0.sum()),total=len(y0),
            rescues=int(((y0==0)&(yz==1)).sum()),harms=int(((y0==1)&(yz==0)).sum()),
            rescue_fraction_of_CI_failures=float(((y0==0)&(yz==1)).sum()/max(1,(y0==0).sum())),
            harm_fraction_of_CI_successes=float(((y0==1)&(yz==0)).sum()/max(1,(y0==1).sum())),
            gain=float(root_gains.mean()),scene_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
            changed_roots=int(np.count_nonzero(picks)),root_gains=root_gains.tolist())
        choices[name]=list(map(int,picks))
    assess('CI',[0]*len(roots));assess('frozen_CV',[r['frozen_cv_choice'] for r in roots])
    for budget in design['selection_budgets']:
        selected=[];cautious=[]
        for root in roots:
            labels=root['targets'][:budget];delta=labels-labels[:,:1];valid=root['features']['valid']
            mean=delta.mean(0);se=delta.std(0,ddof=1)/np.sqrt(budget)
            selected.append(int(np.where(valid,mean,-np.inf).argmax()))
            cautious.append(int(np.where(valid,mean-se,-np.inf).argmax()))
        assess(f'selection_{budget}_futures',selected);assess(f'selection_{budget}_one_se',cautious)
    for value in methods.values():
        difference=np.asarray(value['root_gains'])-np.asarray(methods['frozen_CV']['root_gains'])
        value['gain_over_frozen_CV']=float(difference.mean())
        value['gain_over_frozen_CV_ci95']=np.quantile(difference[bootstrap].mean(1),[.025,.975]).tolist()
    for root in roots:
        choice=root['frozen_cv_choice']
        if (not np.array_equal(root['deployed_cv_choices'],np.full(count,choice)) or
                not np.array_equal(root['deployed_cv_targets'],root['targets'][count:,choice])):
            mismatches.append(root['sample_index'])
    result=dict(completed=True,purpose=design['purpose'],design_sha256=digest(args.design),
        roots=len(roots),early_terminal=len(early),methods=methods,choices=choices,
        deployment_mismatch_scene_indices=mismatches,
        branch_sources=[dict(path=str(path),sha256=digest(path)) for path in paths],
        interpretation='A finite-future selection reference, not a mathematical upper bound or deployable oracle policy.')
    _atomic_save(result,folder/'space-result.pth')
    print(json.dumps({k:v for k,v in result.items() if k not in ('choices','branch_sources','methods')}),flush=True)
    for name,value in methods.items():print(name,json.dumps({k:v for k,v in value.items() if k!='root_gains'}),flush=True)


def space_feedback_collect(args):
    """Single-factor comparison: absolute hold versus live-reference offset."""
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    bank=torch.load(args.design,map_location='cpu',weights_only=False)
    old=torch.load(args.comparison,map_location='cpu',weights_only=False)
    if old['design_sha256']!=digest(args.design):raise ValueError('The comparison needs this exact root panel.')
    if c.runtime_identity()!=bank['runtime'] or digest(bank['source_path'])!=bank['source_sha256']:
        raise ValueError('The fixed CI source or its runtime changed.')
    controller=c.CIVController(c.load_source(bank['source_path'],args.device),c.CIVConfig(**bank['config']))
    count=bank['selection_futures'];folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True);size=8
    for chunk in range(args.shard,(len(bank['roots'])+size-1)//size,args.workers):
        path=folder/f'feedback-branches-{chunk:03d}.pth'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if not saved['completed'] or saved['comparison_sha256']!=digest(args.comparison):raise ValueError('Incompatible feedback shard.')
            continue
        roots=copy.deepcopy(bank['roots'][chunk*size:(chunk+1)*size]);tasks=[];initial=[];mapping=[]
        for index,root in enumerate(roots):
            feature=root['features'];root['relative_targets']=np.full((count,len(feature['valid'])),-1.,np.float32)
            for future,seed in enumerate(root['futures'][:count]):
                for option in np.flatnonzero(feature['valid']):
                    tasks.append(dict(scene=root['scene'],future=seed,program=dict(choice=int(option),
                        gates=feature['gates'][option],duration=int(feature['durations'][option]))))
                    initial.append(root);mapping.append((index,future,int(option)))
        rows=run_streamed_episodes(controller,tasks,batch_size=args.batch_size,initial_jobs=initial,
            progress=True,intervention_mode='relative_feedback')
        for (index,future,option),row in zip(mapping,rows):roots[index]['relative_targets'][future,option]=row['success']
        branch_count=len(rows);tasks=[];initial=[];mapping=[]
        for index,root in enumerate(roots):
            feature=root['features'];labels=root['relative_targets'];delta=labels-labels[:,:1]
            mean=delta.mean(0);se=delta.std(0,ddof=1)/np.sqrt(count)
            global_index=chunk*size+index
            selections=dict(CI=('absolute',0),frozen_CV=('absolute',root['frozen_cv_choice']),
                absolute_reference=('absolute',old['choices'][f'selection_{count}_futures'][global_index]),
                absolute_conservative=('absolute',old['choices'][f'selection_{count}_one_se'][global_index]),
                feedback_reference=('relative_feedback',int(np.where(feature['valid'],mean,-np.inf).argmax())),
                feedback_conservative=('relative_feedback',int(np.where(feature['valid'],mean-se,-np.inf).argmax())))
            root['selections']=selections;root['holdout_values']={name:np.full(count,-1.,np.float32) for name in selections}
            root['heldout_futures']=[root['futures'][0]+2*count+j for j in range(count)]
            programs={}
            for name,(mode,choice) in selections.items():programs.setdefault((mode if choice else 'absolute',choice),[]).append(name)
            for future,seed in enumerate(root['heldout_futures']):
                for (mode,option),names in programs.items():
                    tasks.append(dict(scene=root['scene'],future=seed,program=dict(choice=int(option),
                        gates=feature['gates'][option],duration=int(feature['durations'][option]),intervention_mode=mode)))
                    initial.append(root);mapping.append((index,future,names))
        rows=run_streamed_episodes(controller,tasks,batch_size=args.batch_size,initial_jobs=initial,
            progress=True,intervention_mode='relative_feedback')
        for (index,future,names),row in zip(mapping,rows):
            for name in names:roots[index]['holdout_values'][name][future]=row['success']
        if not all(np.isin(r['relative_targets'][:,r['features']['valid']],(0.,1.)).all() and
                   all(np.isin(v,(0.,1.)).all() for v in r['holdout_values'].values()) for r in roots):
            raise ValueError('Incomplete matched feedback outcomes.')
        branch_count+=len(rows)
        _atomic_save(dict(completed=True,roots=roots,branch_count=branch_count,design_sha256=digest(args.design),
            comparison_sha256=digest(args.comparison),runtime=c.runtime_identity(),driver_sha256=digest(__file__),
            execution_backend=torch.device(args.device).type,
            changed_factor='candidate gates held absolutely or as a clipped offset from the live CI reference'),path)
        print('FEEDBACK_CHUNK_COMPLETE',chunk,len(roots),branch_count,flush=True)


def space_feedback_analyze(args):
    design=torch.load(args.design,map_location='cpu',weights_only=False);folder=Path(args.output)
    paths=[folder/f'feedback-branches-{i:03d}.pth' for i in range((len(design['roots'])+7)//8)]
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    if any(not p['completed'] or p['design_sha256']!=digest(args.design) or
           p['comparison_sha256']!=digest(args.comparison) for p in parts):raise ValueError('Incomplete matched feedback panel.')
    roots=sorted([r for p in parts for r in p['roots']],key=lambda r:r['sample_index'])
    count=design['selection_futures'];rng=np.random.default_rng(design['seed']+19)
    bootstrap=rng.integers(len(design['tasks']),size=(10000,len(design['tasks'])))
    early=[r for r in design['prefix_rows'] if 'success' in r];methods={}
    for name in roots[0]['holdout_values']:
        baseline=[];outcome=[];gains=np.zeros(len(design['tasks']),np.float64)
        for root in roots:
            y0=root['holdout_values']['CI'];yz=root['holdout_values'][name]
            baseline.extend(y0);outcome.extend(yz);gains[root['sample_index']]=float((yz-y0).mean())
        for row in early:baseline.extend([row['success']]*count);outcome.extend([row['success']]*count)
        y0=np.asarray(baseline);yz=np.asarray(outcome)
        methods[name]=dict(success=int(yz.sum()),factual_success=int(y0.sum()),total=len(y0),
            rescues=int(((y0==0)&(yz==1)).sum()),harms=int(((y0==1)&(yz==0)).sum()),
            gain=float(gains.mean()),scene_bootstrap_ci95=np.quantile(gains[bootstrap].mean(1),[.025,.975]).tolist(),
            changed_roots=sum(r['selections'][name][1]>0 for r in roots),root_gains=gains.tolist())
    for name,value in methods.items():
        control='absolute_conservative' if name=='feedback_conservative' else 'absolute_reference'
        difference=np.asarray(value['root_gains'])-np.asarray(methods[control]['root_gains'])
        value['gain_over_absolute_reference']=float(difference.mean())
        value['gain_over_absolute_reference_ci95']=np.quantile(difference[bootstrap].mean(1),[.025,.975]).tolist()
    result=dict(completed=True,purpose='single-factor candidate-execution development comparison; no value training',
        changed_factor=parts[0]['changed_factor'],roots=len(roots),early_terminal=len(early),methods=methods,
        selection_futures=count,heldout_future_indices=list(range(2*count,3*count)),
        design_sha256=digest(args.design),comparison_sha256=digest(args.comparison),
        branch_sources=[dict(path=str(p),sha256=digest(p)) for p in paths])
    _atomic_save(result,folder/'feedback-result.pth')
    for name,value in methods.items():print(name,json.dumps({k:v for k,v in value.items() if k!='root_gains'}),flush=True)
    print('FEEDBACK_SPACE_COMPLETE',flush=True)


@torch.no_grad()
def intervention_catalogs(actor,packet,config):
    """One frozen CI forecast, shared proposal work, three equal-size shortlists."""
    source=actor(packet,deterministic=True,gate_only=True)[0][...,:3]
    settings=dict(actor.actuator_fusion_refinement);settings['selection']='joint'
    info={};c._refine_actuator_interactions(actor,packet,source,source,candidate_output=info,**settings)
    grid,unary,risk,reference=(info[k] for k in ('candidates','unary','pair_risk','reference'))
    b,n,k=unary.shape;weight=float(settings.get('risk_weight',32.));budget=config.candidate_budget
    single=reference[:,None,None,:].expand(-1,n,k,-1).clone()
    for i in range(n):single[:,i,:,i]=torch.arange(k,device=source.device)
    legacy=torch.cat((reference[:,None],info.get('beam_plans',reference[:,None]),
                      info['refined_plans'],single.flatten(1,2)),1)
    pair=[]
    for i,j in itertools.combinations(range(n),2):
        plans=reference[:,None,:].expand(-1,k*k,-1).clone();code=torch.arange(k*k,device=source.device)
        plans[:,:,i]=code//k;plans[:,:,j]=code%k;pair.append(plans)
    pair=torch.cat(pair,1)
    # Alternative joint CI optima reuse exactly the same unary and pair forecasts.
    joint=[]
    for multiplier in (0.,.25,.5,2.,4.):
        scores=source.new_zeros((b,1));plans=torch.empty(b,1,0,dtype=torch.long,device=source.device)
        for i in range(n):
            values=scores[:,:,None]+unary[:,i,None]
            for j in range(i):values=values+weight*multiplier*risk[:,j,i].gather(1,plans[:,:,j,None].expand(-1,-1,k))
            scores,index=values.flatten(1).topk(min(128,values[0].numel()),largest=False)
            plans=torch.cat((plans.gather(1,(index//k)[...,None].expand(-1,-1,i)),(index%k)[...,None]),-1)
        joint.append(plans)
    inverse=info['order'].argsort(-1)
    def materialize(plans,valid):
        score,own,interaction=c.plan_costs(unary,risk,plans,weight)
        gates=grid[:,None].expand(-1,plans.shape[1],-1,-1,-1).gather(3,plans[...,None,None].expand(-1,-1,-1,1,2)).squeeze(3)
        gates=gates.gather(2,inverse[:,None,:,None].expand(-1,plans.shape[1],-1,2))
        own=own.gather(2,inverse[:,None].expand(-1,plans.shape[1],-1))
        interaction=interaction.gather(2,inverse[:,None,:,None].expand_as(interaction))
        interaction=interaction.gather(3,inverse[:,None,None,:].expand_as(interaction))
        physical=plans.gather(2,inverse[:,None].expand(-1,plans.shape[1],-1))
        endpoints=info['prediction_paths'][...,-1,:][:,None].expand(-1,plans.shape[1],-1,-1,-1)
        endpoints=endpoints.gather(3,physical[...,None,None].expand(-1,-1,-1,1,2)).squeeze(3)
        return dict(gates=gates,valid=valid,costs=score,unary=own,pair_risk=interaction,endpoints=endpoints,has_ci=1.)
    outputs={}
    for family,pool in [('legacy',legacy),('pair',torch.cat((reference[:,None],pair),1)),
                        ('joint',torch.cat((reference[:,None],*joint,info['refined_plans'],pair),1))]:
        costs,_,_=c.plan_costs(unary,risk,pool,weight)
        order=costs.argsort(dim=-1,stable=True).cpu().tolist();pool_cpu=pool.cpu().numpy();cost_cpu=costs.cpu().numpy()
        grid_cpu=grid.cpu().numpy();chosen=[];masks=[]
        for row in range(b):
            ref=pool_cpu[row,0];ref_gates=grid_cpu[row,np.arange(n),ref]
            seen={tuple(ref)};indices=[0];used_pairs=set()
            ranking=order[row]
            if family=='joint':
                # Five predeclared risk tradeoffs get first access to the five slots.
                ranking=[1+q*128 for q in range(5)]+ranking
            for index in ranking:
                plan=pool_cpu[row,index];key=tuple(plan)
                if key in seen or not np.isfinite(cost_cpu[row,index]):continue
                moved=tuple(np.flatnonzero(np.max(np.abs(grid_cpu[row,np.arange(n),plan]-ref_gates),axis=-1)>1e-6))
                if family=='legacy' and cost_cpu[row,index]>cost_cpu[row,0]+n*config.cost_regret_per_vessel+1e-6:continue
                if family=='pair' and (len(moved)!=2 or moved in used_pairs):continue
                if family=='joint' and len(moved)<2:continue
                seen.add(key);indices.append(index);used_pairs.add(moved)
                if len(indices)==budget:break
            count=len(indices);indices+=([0]*(budget-count))
            chosen.append(indices);masks.append([True]*count+[False]*(budget-count))
        index=torch.as_tensor(chosen,device=source.device)
        plans=pool.gather(1,index[...,None].expand(-1,-1,n))
        outputs[family]=materialize(plans,torch.as_tensor(masks,device=source.device))
    return source,outputs


def intervention_trigger(packet,step):
    """Observable close encounter or a three-second collision prediction; no outcomes."""
    if step<5:return False
    _,_,target,velocity,_=c.d.PublicTargetMotion.encode(packet,True)
    motion=packet['motion'];position=motion[0,:,:2];speed=motion[0,:,3:5]
    separation=position[:,None]-position[None];relative=speed[:,None]-speed[None]
    closing=(separation*relative).sum(-1)
    ttc=(-closing/relative.square().sum(-1).clamp_min(1e-6)).clamp(0.,3.)
    closest=(separation+ttc[...,None]*relative).norm(dim=-1)
    mask=~torch.eye(len(position),dtype=torch.bool,device=position.device)
    conflict=bool(((closest<=8.)&(closing<0.)&(ttc>0.)&mask).any())
    interception=bool(((position-target[0]).norm(dim=-1)<=15.).any())
    radial=-(target[0]*velocity[0]).sum()/target[0].norm().clamp_min(1e-6)
    breach=bool((radial>0.)&((target[0].norm()-15.)/radial.clamp_min(1e-6)<=6.))
    return conflict or interception or breach


def intervention_design(args):
    """Freeze scenes, factors, budgets and the continuation gate before outcomes."""
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'design.pth'
    if destination.exists():raise ValueError('The intervention design is already frozen.')
    policy=torch.load(args.model,map_location='cpu',weights_only=False)
    tasks=make_tasks(args.seed,args.roots,6,(2.,2.5))
    observed={tuple(s) for s in policy['fit']['training_scenes']+policy['fit']['validation_scenes']}
    prior=folder.parent
    excluded=[]
    for path,key in [(prior/'hard3000-20261027-1410/hard-design.pth','test_tasks'),
                     (prior/'root-cause-repair-20261027/current-space/space-design.pth','tasks')]:
        record=torch.load(path,map_location='cpu',weights_only=False);observed.update(tuple(t['scene']) for t in record[key]);excluded.append(dict(path=str(path),sha256=digest(path)))
    paper=torch.load(prior/'paper-evidence-20261027/design.pth',map_location='cpu',weights_only=False)
    observed.update(tuple(t['scene']) for cell in paper['cells'].values() for t in cell)
    if any(tuple(t['scene']) in observed for t in tasks):raise ValueError('New development scenes overlap prior evaluation or fitting.')
    cells=[dict(name=f'{family}_fixed_absolute',family=family,schedule='fixed',mode='absolute',durations=(5,20),stage='structure') for family in ('legacy','pair','joint')]
    cells += [dict(name=f'joint_{schedule}_feedback_{window}',family='joint',schedule=schedule,mode='relative_feedback',durations=duration,stage='timing')
        for schedule in ('fixed','trigger') for window,duration in [('short',(5,20)),('long',(20,60))]]
    record=dict(completed=True,purpose='New intervention-space development; no formal model selection on old tests',
        source_path=policy['source_path'],source_sha256=policy['source_sha256'],model=str(Path(args.model).resolve()),model_sha256=digest(args.model),
        config=policy['config'],runtime=c.runtime_identity(),driver_sha256=digest(__file__),tasks=tasks,cells=cells,seed=args.seed,
        selection_futures=args.futures,validation_futures=args.futures,candidate_slots=11,forecast_calls_per_root=1,
        proposal_budget='All arms compute the same 5x128 joint beam bank and complete pair pool from one CI forecast.',
        trigger=dict(earliest_step=5,collision_horizon_seconds=3.,collision_clearance=8.,interception_distance=15.,breach_seconds=6.),
        gate=dict(minimum_gain=.005,confidence='one-sided lower bound positive after Holm correction across seven primary mean selectors',
            confirmation='Repeat a qualifying mechanism on a second frozen new scene cohort before any value fitting.'),
        scenario_sha256=hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest(),excluded=excluded)
    _atomic_save(record,destination);print('INTERVENTION_DESIGN_COMPLETE',len(tasks),len(cells),flush=True)


def intervention_prefix(args):
    from envs.TADgame import TADEnv
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    design=torch.load(args.design,map_location='cpu',weights_only=False);folder=Path(args.output)
    assert digest(design['source_path'])==design['source_sha256'] and c.runtime_identity()==design['runtime']
    controller=c.CIVController(c.load_source(design['source_path'],args.device),c.CIVConfig(**design['config']))
    executor=c.CIVExecutor(controller);jobs=[];rng_saved=np.random.get_state();started=time.monotonic()
    for index in range(args.shard,len(design['tasks']),args.workers):
        task=design['tasks'][index];seed,n,agility=task['scene'];np.random.seed(seed)
        env=TADEnv(n,protocol=controller.source.environment_protocol);obs,_=env.reset(agility=agility,noisy_agility=False)
        np.random.seed(task['future']);state=c.ExecutionState(decided=True)
        jobs.append(dict(index=index,scene=task['scene'],env=env,obs=obs,rng=np.random.get_state(),state=state,roots={},terminal=None))
    active=list(range(len(jobs)))
    try:
        while active:
            packets=[c.d.joint_packet(jobs[i]['env'],jobs[i]['obs']) for i in active]
            packet=c.d.mechanism_stack(packets,args.device)
            for offset,index in enumerate(active):
                job=jobs[index];step=job['state'].step;part={k:v[offset:offset+1] for k,v in packet.items()}
                for schedule,due in [('fixed',step==20),('trigger',intervention_trigger(part,step))]:
                    if due and schedule not in job['roots']:
                        job['roots'][schedule]=dict(environment=copy.deepcopy(job['env']),observation=job['obs'].copy(),
                            random_state=job['rng'],step=step,scene=job['scene'],sample_index=job['index'])
            keep=[j for j,i in enumerate(active) if len(jobs[i]['roots'])<2]
            active=[active[j] for j in keep]
            if not active:break
            packet={k:v[keep] for k,v in packet.items()};commands,_=executor.actions(packet,[jobs[i]['state'] for i in active]);following=[]
            for offset,index in enumerate(active):
                job=jobs[index];np.random.set_state(job['rng'])
                obs,_,done,_=c.d.joint_task_step(job['env'],commands[offset],controller.source.training_gamma,controller.source.training_step_cost)
                job['rng']=np.random.get_state();job['obs']=obs;job['state'].step+=1
                if done:job['terminal']=dict(**c.d.outcome(job['env'],done),step=job['state'].step)
                else:following.append(index)
            active=following
        for job in jobs:
            for root in job['roots'].values():
                packet=c.d.mechanism_tensor(c.d.joint_packet(root['environment'],root['observation']),args.device)
                source,catalogs=c.fixed_inference(lambda p:intervention_catalogs(controller.source,p,controller.config),packet,64)
                root['catalogs']={family:{k:v[0].cpu().numpy() if torch.is_tensor(v) else v for k,v in catalog.items()} for family,catalog in catalogs.items()}
                root['features']={'gates':root['catalogs']['legacy']['gates']}
            for key in ('env','obs','rng','state'):job.pop(key)
        _atomic_save(dict(completed=True,jobs=jobs,design_sha256=digest(args.design),runtime=c.runtime_identity(),driver_sha256=digest(__file__),seconds=time.monotonic()-started),folder/f'prefix-{args.shard:02d}.pth')
        print('INTERVENTION_PREFIX_COMPLETE',args.shard,len(jobs),flush=True)
    finally:np.random.set_state(rng_saved)


def intervention_merge(args):
    design=torch.load(args.design,map_location='cpu',weights_only=False);folder=Path(args.output)
    paths=[folder/f'prefix-{i:02d}.pth' for i in range(args.workers)];parts=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    assert all(p['completed'] and p['design_sha256']==digest(args.design) and p['runtime']==design['runtime'] for p in parts)
    jobs=sorted([j for p in parts for j in p['jobs']],key=lambda j:j['index'])
    assert [j['index'] for j in jobs]==list(range(len(design['tasks'])))
    _atomic_save(dict(completed=True,jobs=jobs,design_sha256=digest(args.design),sources=[dict(path=str(p),sha256=digest(p)) for p in paths]),folder/'roots.pth')
    print('INTERVENTION_ROOTS_COMPLETE',len(jobs),{s:sum(s in j['roots'] for j in jobs) for s in ('fixed','trigger')},flush=True)


def intervention_collect(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    design=torch.load(args.design,map_location='cpu',weights_only=False);folder=Path(args.output)
    bank=torch.load(folder/'roots.pth',map_location='cpu',weights_only=False)
    assert bank['design_sha256']==digest(args.design) and c.runtime_identity()==design['runtime'] and digest(design['source_path'])==design['source_sha256']
    controller=c.CIVController(c.load_source(design['source_path'],args.device),c.CIVConfig(**design['config']))
    count=design['selection_futures'];cells=[v for v in design['cells'] if v['stage']==args.stage];size=4
    for chunk in range(args.shard,(len(bank['jobs'])+size-1)//size,args.workers):
        destination=folder/f'{args.stage}-branches-{chunk:03d}.pth'
        if destination.exists():
            prior=torch.load(destination,map_location='cpu',weights_only=False)
            assert prior['completed'] and prior['design_sha256']==digest(args.design);continue
        started=time.monotonic();samples=[];tasks=[];initial=[];mapping=[]
        for job in bank['jobs'][chunk*size:(chunk+1)*size]:
            for cell in cells:
                sample=dict(index=job['index'],scene=job['scene'],cell=cell,root_step=None,terminal=job['terminal'])
                if cell['schedule'] not in job['roots']:
                    sample['early_or_untriggered']=True;samples.append(sample);continue
                root=job['roots'][cell['schedule']];catalog=root['catalogs'][cell['family']]
                sample.update(early_or_untriggered=False,root_step=root['step'],selection=np.full((count,11),-1,np.int8),
                    valid=np.r_[True,np.repeat(catalog['valid'][1:],2)],plans=[(0,0)]+[(i,int(d)) for i in range(1,6) for d in cell['durations']],
                    gates=catalog['gates'],changed_vessels=(np.max(np.abs(catalog['gates']-catalog['gates'][:1]),axis=-1)>1e-6).sum(-1).tolist())
                sample_index=len(samples);samples.append(sample)
                for future in range(count):
                    seed=design['seed']+10000000+job['index']*1000+future
                    for option,(plan,duration) in enumerate(sample['plans']):
                        tasks.append(dict(scene=job['scene'],future=seed,program=dict(choice=option,gates=catalog['gates'][plan],duration=duration,intervention_mode=cell['mode'])))
                        initial.append(root);mapping.append((sample_index,future,option))
        rows=run_streamed_episodes(controller,tasks,batch_size=args.batch_size,initial_jobs=initial,progress=True,intervention_mode='relative_feedback')
        trajectory=[]
        for (sample_index,future,option),row in zip(mapping,rows):
            sample=samples[sample_index];sample['selection'][future,option]=row['success']
            trajectory.append(dict(index=sample['index'],cell=sample['cell']['name'],phase='selection',future=row['future'],option=option,**{k:row[k] for k in ('success','outcome','collision','simulation_steps','discounted_return')}))
        tasks=[];initial=[];mapping=[]
        for sample_index,sample in enumerate(samples):
            if sample['early_or_untriggered']:continue
            delta=sample['selection']-sample['selection'][:,:1];mean=delta.mean(0);se=delta.std(0,ddof=1)/np.sqrt(count)
            selected=int(np.where(sample['valid'],mean,-np.inf).argmax());conservative=int(np.where(sample['valid'],mean-se,-np.inf).argmax())
            sample['choices']=dict(reference=selected,conservative=conservative)
            options=sorted({0,selected,conservative});sample['validation']={option:np.full(count,-1,np.int8) for option in options}
            job=bank['jobs'][sample['index']];cell=sample['cell'];root=job['roots'][cell['schedule']]
            for future in range(count):
                seed=design['seed']+10000000+sample['index']*1000+count+future
                for option in options:
                    plan,duration=sample['plans'][option]
                    tasks.append(dict(scene=sample['scene'],future=seed,program=dict(choice=option,gates=sample['gates'][plan],duration=duration,intervention_mode=cell['mode'])))
                    initial.append(root);mapping.append((sample_index,future,option))
        rows=run_streamed_episodes(controller,tasks,batch_size=args.batch_size,initial_jobs=initial,progress=True,intervention_mode='relative_feedback')
        for (sample_index,future,option),row in zip(mapping,rows):
            sample=samples[sample_index];sample['validation'][option][future]=row['success']
            trajectory.append(dict(index=sample['index'],cell=sample['cell']['name'],phase='validation',future=row['future'],option=option,**{k:row[k] for k in ('success','outcome','collision','simulation_steps','discounted_return')}))
        assert all(s['early_or_untriggered'] or (np.isin(s['selection'],[0,1]).all() and all(np.isin(v,[0,1]).all() for v in s['validation'].values())) for s in samples)
        _atomic_save(dict(completed=True,samples=samples,trajectories=trajectory,design_sha256=digest(args.design),root_bank_sha256=digest(folder/'roots.pth'),
            runtime=c.runtime_identity(),driver_sha256=digest(__file__),seconds=time.monotonic()-started,simulation_steps=sum(r['simulation_steps'] for r in trajectory)),destination)
        print('INTERVENTION_CHUNK_COMPLETE',args.stage,chunk,len(samples),len(trajectory),flush=True)


def intervention_analyze(args):
    """Aggregate all prespecified cells, retaining every scene and both selectors."""
    folder=Path(args.output);design=torch.load(args.design,map_location='cpu',weights_only=False)
    chunks=(len(design['tasks'])+3)//4
    paths=[folder/f'{stage}-branches-{chunk:03d}.pth' for stage in ('structure','timing') for chunk in range(chunks)]
    parts=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    if any(not p['completed'] or p['design_sha256']!=digest(args.design) or p['driver_sha256']!=design['driver_sha256'] for p in parts):
        raise ValueError('Every frozen-source experimental cell must complete before analysis.')
    samples=[s for part in parts for s in part['samples']];n=len(design['tasks']);count=design['validation_futures']
    rng=np.random.default_rng(design['seed']+19);bootstrap=rng.integers(n,size=(20000,n));signs=rng.choice([-1.,1.],size=(20000,n));methods={}
    for cell in design['cells']:
        rows=sorted([s for s in samples if s['cell']['name']==cell['name']],key=lambda s:s['index'])
        if [r['index'] for r in rows]!=list(range(n)):raise ValueError('Missing or repeated initial scenes.')
        for selector in ('reference','conservative'):
            gains=np.zeros(n);baseline=success=rescues=harms=active=positive=negative=untriggered=0
            for row in rows:
                if row['early_or_untriggered']:
                    baseline+=count*row['terminal']['success'];success+=count*row['terminal']['success'];untriggered+=1;continue
                choice=row['choices'][selector];y0=row['validation'][0];yz=row['validation'][choice];delta=yz-y0
                gains[row['index']]=delta.mean();baseline+=int(y0.sum());success+=int(yz.sum())
                rescues+=int(((y0==0)&(yz==1)).sum());harms+=int(((y0==1)&(yz==0)).sum())
                active+=choice>0;positive+=choice>0 and delta.mean()>0;negative+=choice>0 and delta.mean()<0
            mean=float(gains.mean());pvalue=float((1+((signs*gains).mean(1)>=mean-1e-12).sum())/(len(signs)+1)) if mean>0 else 1.
            methods[cell['name']+'/'+selector]=dict(success=success,ci_success=baseline,total=n*count,rescues=rescues,harms=harms,
                gain=mean,root_bootstrap_ci95=np.quantile(gains[bootstrap].mean(1),[.025,.975]).tolist(),root_paired_randomization_p_one_sided=pvalue,
                root_gains=gains.tolist(),active_roots=int(active),positive_selected_roots=int(positive),negative_selected_roots=int(negative),
                selected_candidate_effective_rate=float(positive/active) if active else 0.,nondecision_roots=untriggered,
                evaluated_candidate_slots=11,mean_distinct_valid_slots=float(np.mean([r['valid'].sum() for r in rows if not r['early_or_untriggered']])),
                mean_changed_vessels_per_nonnull_plan=float(np.mean([v for r in rows if not r['early_or_untriggered'] for v in r['changed_vessels'][1:]])))
    primary=[(name,value) for name,value in methods.items() if name.endswith('/reference')]
    cumulative=0.
    for index,(name,value) in enumerate(sorted(primary,key=lambda pair:pair[1]['root_paired_randomization_p_one_sided'])):
        cumulative=max(cumulative,min(1.,(len(primary)-index)*value['root_paired_randomization_p_one_sided']))
        value['holm_p']=cumulative
        value['qualifies_for_new_scene_confirmation']=bool(value['gain']>=design['gate']['minimum_gain'] and value['root_bootstrap_ci95'][0]>0 and cumulative<.05)
    comparisons={}
    for alternative,control in [('pair_fixed_absolute','legacy_fixed_absolute'),('joint_fixed_absolute','legacy_fixed_absolute'),
            ('joint_fixed_feedback_short','joint_fixed_absolute'),('joint_trigger_feedback_short','joint_fixed_feedback_short'),
            ('joint_fixed_feedback_long','joint_fixed_feedback_short'),('joint_trigger_feedback_long','joint_trigger_feedback_short')]:
        a=methods[alternative+'/reference'];b=methods[control+'/reference'];delta=np.asarray(a['root_gains'])-np.asarray(b['root_gains'])
        comparisons[alternative+' vs '+control]=dict(gain_difference=float(delta.mean()),root_bootstrap_ci95=np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist(),
            estimand='Difference in intervention gain relative to each schedule-specific CI continuation; roots paired by initial scene.')
    csv_path=folder/'episodes.csv'
    fields=['sample_index','scene_seed','fleet','agility','cell','phase','future_seed','option','success','outcome','collision','simulation_steps','discounted_return','weight','prefix_terminal']
    with csv_path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for part in parts:
            for row in part['trajectories']:
                scene=design['tasks'][row['index']]['scene']
                writer.writerow(dict(sample_index=row['index'],scene_seed=scene[0],fleet=scene[1],agility=scene[2],cell=row['cell'],phase=row['phase'],future_seed=row['future'],
                    option=row['option'],**{k:row[k] for k in ('success','outcome','collision','simulation_steps','discounted_return')},weight=1,prefix_terminal=0))
            for sample in part['samples']:
                if sample['early_or_untriggered']:
                    scene=sample['scene'];terminal=sample['terminal']
                    writer.writerow(dict(sample_index=sample['index'],scene_seed=scene[0],fleet=scene[1],agility=scene[2],cell=sample['cell']['name'],phase='validation',future_seed='',option=0,
                        success=terminal['success'],outcome=terminal['outcome'],collision=terminal['collision'],simulation_steps=terminal['step'],discounted_return='',weight=count,prefix_terminal=1))
    choices={cell['name']:{str(s['index']):s.get('choices',{'reference':0,'conservative':0}) for s in samples if s['cell']['name']==cell['name']} for cell in design['cells']}
    (folder/'choices.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
    qualified=[name for name,value in primary if value['qualifies_for_new_scene_confirmation']]
    result=dict(completed=True,development_only=True,design_sha256=digest(args.design),methods=methods,controlled_comparisons=comparisons,
        statistical_unit=f'{n} independent initial scenes; repeated futures clustered within scene',
        gate=design['gate'],qualified_for_new_scene_confirmation=qualified,value_training_allowed=False,
        decision='confirm on a second new scene cohort before value training' if qualified else 'stop value training on these tested spaces; no qualifying reproducible opportunity yet',
        simulation_steps=sum(p['simulation_steps'] for p in parts),full_branch_rollouts=sum(len(p['trajectories']) for p in parts),
        worker_seconds=sum(p['seconds'] for p in parts),raw_episodes_sha256=digest(csv_path),choices_sha256=digest(folder/'choices.json'),
        branch_sources=[dict(path=str(p),sha256=digest(p)) for p in paths])
    (folder/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,value in primary:print(name,json.dumps({k:v for k,v in value.items() if k!='root_gains'}),flush=True)
    print('INTERVENTION_ANALYSIS_COMPLETE',qualified,flush=True)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    collect_parser=sub.add_parser('collect');collect_parser.set_defaults(function=collect)
    collect_parser.add_argument('--source',required=True);collect_parser.add_argument('--generator',choices=('joint','local'),required=True)
    collect_parser.add_argument('--roots',type=int,default=96);collect_parser.add_argument('--validation',type=int,default=32)
    collect_parser.add_argument('--futures',type=int,default=2);collect_parser.add_argument('--workers',type=int,default=4)
    collect_parser.add_argument('--include-factual-future',action='store_true')
    collect_parser.add_argument('--decision-step',type=int,default=20)
    collect_parser.add_argument('--vessels',type=int,default=6)
    fit_parser=sub.add_parser('fit');fit_parser.set_defaults(function=fit)
    fit_parser.add_argument('--data',required=True);fit_parser.add_argument('--kind',choices=('conditional','ordinary'),required=True)
    fit_parser.add_argument('--updates',type=int,default=1200);fit_parser.add_argument('--members',type=int,default=3)
    fit_parser.add_argument('--gain-weight',type=float,default=0.)
    evaluation=sub.add_parser('evaluate');evaluation.set_defaults(function=evaluate)
    evaluation.add_argument('--source');evaluation.add_argument('--model');evaluation.add_argument('--generator',default='joint',choices=('joint','independent','local'))
    evaluation.add_argument('--tasks',type=int,default=160);evaluation.add_argument('--vessels',type=int,default=6)
    for command in (collect_parser,fit_parser,evaluation):
        command.add_argument('--agilities',type=float,nargs='+',default=[2.,2.5])
        command.add_argument('--seed',type=int,required=True);command.add_argument('--output',required=True)
        command.add_argument('--device',default='cuda:0')
    for name,function in [('hard-design',hard_design),('hard-prefix',hard_prefix),('hard-roots',hard_roots),('hard-collect',hard_collect),('hard-merge',hard_merge)]:
        command=sub.add_parser(name);command.set_defaults(function=function)
        command.add_argument('--model');command.add_argument('--design');command.add_argument('--output',required=True)
        command.add_argument('--seed',type=int,default=60000000);command.add_argument('--device',default='cuda:0')
        command.add_argument('--workers',type=int,default=12);command.add_argument('--shard',type=int,default=0)
        command.add_argument('--batch-size',type=int,default=128)
    for name,function in [('space-design',space_design),('space-collect',space_collect),('space-analyze',space_analyze),
                          ('space-feedback-collect',space_feedback_collect),('space-feedback-analyze',space_feedback_analyze)]:
        command=sub.add_parser(name);command.set_defaults(function=function)
        command.add_argument('--model');command.add_argument('--design');command.add_argument('--output',required=True)
        command.add_argument('--comparison')
        command.add_argument('--seed',type=int,default=146000000);command.add_argument('--roots',type=int,default=128)
        command.add_argument('--futures',type=int,default=16);command.add_argument('--workers',type=int,default=8)
        command.add_argument('--shard',type=int,default=0);command.add_argument('--batch-size',type=int,default=128)
        command.add_argument('--device',default='cuda:0')
    for name,function in [('intervention-design',intervention_design),('intervention-prefix',intervention_prefix),
                          ('intervention-merge',intervention_merge),('intervention-collect',intervention_collect),('intervention-analyze',intervention_analyze)]:
        command=sub.add_parser(name);command.set_defaults(function=function)
        command.add_argument('--model');command.add_argument('--design');command.add_argument('--output',required=True)
        command.add_argument('--seed',type=int,default=180000000);command.add_argument('--roots',type=int,default=128)
        command.add_argument('--futures',type=int,default=16);command.add_argument('--workers',type=int,default=16)
        command.add_argument('--shard',type=int,default=0);command.add_argument('--batch-size',type=int,default=128)
        command.add_argument('--stage',choices=('structure','timing'),default='structure');command.add_argument('--device',default='cuda:0')
    args=parser.parse_args();args.function(args)


if __name__=='__main__':main()
