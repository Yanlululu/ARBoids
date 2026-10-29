"""Public target-motion prediction for mechanism-one candidate forecasts."""
import study_runtime  # Configure numerical runtimes before NumPy and PyTorch.
import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import joint_fusion as d
from joint_fusion import atomic_save


def fixed_batch_inference(function, packet, *arguments, width=0):
    """Keep neural arithmetic shapes fixed when paired trajectories terminate.

    Duplicate padding is discarded before execution. It fixes arithmetic
    shapes while retaining each real trajectory and the controller's math.
    """
    if not isinstance(width, int) or isinstance(width, bool) or width < 0:
        raise ValueError('The fixed inference width must be a nonnegative integer.')
    if not width:
        return function(packet, *arguments)
    count = len(next(iter(packet.values())))
    if not count or any(not torch.is_tensor(x) or len(x) != count
                        for x in (*packet.values(), *arguments)):
        raise ValueError('Fixed inference requires aligned nonempty tensor batches.')
    pieces = []
    tuple_output = None
    for begin in range(0, count, width):
        stop = min(begin+width, count)
        def padded(value):
            part = value[begin:stop]
            return (part if len(part) == width else
                    torch.cat((part, part[-1:].expand(width-len(part), *part.shape[1:])), 0))
        result = function({key:padded(value) for key,value in packet.items()},
                          *(padded(value) for value in arguments))
        is_tuple = isinstance(result, tuple)
        outputs = result if is_tuple else (result,)
        if (tuple_output is not None and tuple_output != is_tuple or
                any(not torch.is_tensor(value) or len(value) != width for value in outputs)):
            raise ValueError('Fixed inference must return one aligned tensor or a tensor tuple.')
        tuple_output = is_tuple
        pieces.append(tuple(value[:stop-begin] for value in outputs))
    joined = tuple(torch.cat(values, 0) for values in zip(*pieces))
    return joined if tuple_output else joined[0]


def collect_target_feedback_data(predecessor,factual_bank,output,*,seed=42,steps=12,
                                 branches=3,batch_size=64,device='cuda:0',record_sequences=False,
                                 include_initial_roots=False,fixed_inference_width=0):
    """Observe short candidate continuations for public target-motion fitting."""
    d.numerical_source()
    predecessor=Path(predecessor);factual_bank=Path(factual_bank);output=Path(output)
    if steps<1 or branches<1 or batch_size<1:
        raise ValueError('Target-response collection needs positive continuation and batch sizes.')
    digest=hashlib.sha256(predecessor.read_bytes()).hexdigest()
    actor=d.joint_load(str(predecessor),predecessor.stat().st_mtime_ns).to(device)
    bank=torch.load(factual_bank,map_location='cpu',weights_only=False)
    if not bank.get('completed') or bank['predecessor_sha256']!=digest:
        raise ValueError('Use completed factual states from this exact predecessor.')
    records=list(bank['records'])
    if include_initial_roots:
        for record in bank['records']:
            scene=record['scene'];scene_seed,n,agility=scene
            np.random.seed(scene_seed)
            if actor.environment_protocol == 'source':
                environment=d.numerical_source().TADEnv(n)
            else:
                from envs.TADgame import TADEnv
                environment=TADEnv(n,protocol=actor.environment_protocol)
            observation,_=environment.reset(agility=agility,noisy_agility=False)
            records.append(dict(scene=scene,root_snapshot=dict(environment=environment,
                observation=observation.copy())))
    specification=dict(predecessor=str(predecessor),predecessor_sha256=digest,
        factual_bank=str(factual_bank),factual_bank_sha256=hashlib.sha256(factual_bank.read_bytes()).hexdigest(),
        seed=seed,steps=steps,branches=branches,batch_size=batch_size,inference_device=device,
        control_seconds=.2,absolute_frame=True,
        inputs='public defender poses, velocities and Boids commands; public target position and velocity',
        excluded_inputs='attacker agility, hidden water velocity, attacker controller and realized future',
        continuation='fixed actuator fusion coefficients with refreshed unchanged SAC and Boids commands',
        root_scenes=[tuple(x['scene']) for x in records])
    if record_sequences:specification['record_sequences']=True
    if include_initial_roots:
        specification.update(include_initial_roots=True,root_scenes=[tuple(x['scene']) for x in records])
    if fixed_inference_width:specification['fixed_inference_width']=fixed_inference_width
    result=(torch.load(output,map_location='cpu',weights_only=False) if output.exists() else
        dict(specification=specification,next_task=0,nodes=[],global_state=[],targets=[],root_ids=[],
            physical_steps=0,completed=False))
    if record_sequences and not result['next_task']:
        result.update({key:[] for key in ('sequence_ids','sequence_ticks','public_motion','public_boids',
            'public_target','public_velocity','next_public_target','next_public_velocity')})
    if result['specification']!=specification:
        raise ValueError('Cannot resume public dynamics acquisition with changed states or sampling.')
    jobs=[(root,branch) for root in range(len(records)) for branch in range(branches)]
    if result['completed']:
        if result['next_task']!=len(jobs):raise ValueError('Completed dynamics data have an incomplete task prefix.')
        actor.cpu();return result
    start=int(result['next_task'])
    if start%batch_size or not 0<=start<len(jobs):
        raise ValueError('Target-response acquisition must resume at a complete native batch.')
    for begin in range(start,len(jobs),batch_size):
        current=jobs[begin:begin+batch_size];environments=[];observations=[];random_states=[];coefficients=[]
        for root,branch in current:
            snapshot=records[root]['root_snapshot']
            environments.append(copy.deepcopy(snapshot['environment']))
            observations.append(snapshot['observation'].copy())
            future=int(np.random.SeedSequence([seed,root,branch,1776]).generate_state(1)[0])
            random_states.append(np.random.RandomState(future).get_state())
            n=len(snapshot['observation'])
            codes=(np.full(n,4) if branch==0 else
                np.random.default_rng(np.random.SeedSequence([seed,root,branch,1777])).integers(9,size=n))
            coefficients.append(np.stack((codes//3,codes%3),-1).astype(np.float32)*.5)
        active=list(range(len(current)))
        for tick in range(steps):
            if not active:break
            packets=[d.joint_packet(environments[i],observations[i]) for i in active]
            packet={key:torch.as_tensor(np.stack([item[key] for item in packets]),device=device)
                for key in packets[0]}
            with torch.no_grad():
                nodes,global_state,target,velocity,axis=d.PublicTargetMotion.encode(packet,True)
                source=fixed_batch_inference(lambda current:
                    actor(current,deterministic=True,gate_only=True)[0][...,:2],
                    packet,width=fixed_inference_width)
                gates=torch.as_tensor(np.stack([coefficients[i] for i in active]),device=device)
                action=torch.cat((gates*source+(1.-gates)*packet['node'][...,11:13],
                    torch.ones_like(source[...,:1])),-1).cpu().numpy()
            following=[];after=[]
            for local,i in enumerate(active):
                np.random.set_state(random_states[i])
                obs,_,done,_=d.joint_task_step(environments[i],action[local],actor.training_gamma,actor.training_step_cost)
                random_states[i]=np.random.get_state();observations[i]=obs
                after.append(d.joint_packet(environments[i],obs))
                if not done:following.append(i)
            with torch.no_grad():
                next_packet={key:torch.as_tensor(np.stack([item[key] for item in after]),device=device)
                    for key in ('node','motion')}
                next_target,next_velocity=actor.public_target_motion(next_packet)
                def rotate(vector):
                    return torch.stack(((vector*axis).sum(-1),axis[:,0]*vector[:,1]-axis[:,1]*vector[:,0]),-1)
                labels=torch.cat((rotate(next_target-target-.2*velocity),rotate(next_velocity-velocity)),-1)
            result['nodes'].append(nodes.cpu().numpy());result['global_state'].append(global_state.cpu().numpy())
            result['targets'].append(labels.cpu().numpy())
            result['root_ids'].append(np.asarray([current[i][0] for i in active],np.int64))
            if record_sequences:
                result['sequence_ids'].append(np.asarray([begin+i for i in active],np.int64))
                result['sequence_ticks'].append(np.full(len(active),tick,np.int64))
                for key,value in (('public_motion',packet['motion']),('public_boids',packet['node'][...,11:13]),
                                  ('public_target',target),('public_velocity',velocity),
                                  ('next_public_target',next_target),('next_public_velocity',next_velocity)):
                    result[key].append(value.cpu().numpy())
            result['physical_steps']+=len(active);active=following
        result['next_task']=begin+len(current);result['completed']=result['next_task']==len(jobs)
        atomic_save(result,output)
        print(dict(phase='public_target_feedback_collection',seed=seed,tasks=result['next_task'],
            total=len(jobs),physical_steps=result['physical_steps']),flush=True)
    actor.cpu();return result


def target_response_validation_roots(data,roots,seed,requested=None):
    """Keep every initial and later snapshot of one layout on one side."""
    unique=np.unique(roots)
    scenes=data['specification'].get('root_scenes')
    if requested is None and scenes is not None:
        families=list(dict.fromkeys(tuple(scenes[int(root)]) for root in unique))
        chosen=np.random.default_rng(seed).permutation(len(families))[:max(1,len(families)//5)]
        selected={families[int(index)] for index in chosen}
        held=np.asarray([root for root in unique if tuple(scenes[int(root)]) in selected])
    else:
        held=(np.random.default_rng(seed).permutation(unique)[:max(1,len(unique)//5)]
            if requested is None else np.asarray(requested,dtype=np.int64))
    if not len(held) or len(np.unique(held))!=len(held) or not set(held)<set(unique):
        raise ValueError('Target-response validation needs distinct observed roots and a separate training group.')
    if scenes is not None:
        validation={tuple(scenes[int(root)]) for root in held}
        training={tuple(scenes[int(root)]) for root in unique if root not in set(held)}
        if validation & training:
            raise ValueError('Initial and later states from the same layout cannot cross prediction validation.')
    return held


def fit_target_feedback_model(data,output,*,updates=6000,ensemble_size=3,width=48,seed=42,device='cuda:0',
                              representation='pooled_residual',validation_root_ids=None):
    """Fit measured one-step public responses, validating on separate roots."""
    if not data.get('completed') or ensemble_size<1 or updates<ensemble_size or updates%ensemble_size:
        raise ValueError('Target-response learning needs complete data and equal positive member budgets.')
    specification=data['specification'];predecessor=Path(specification['predecessor'])
    if hashlib.sha256(predecessor.read_bytes()).hexdigest()!=specification['predecessor_sha256']:
        raise ValueError('The target-response predecessor changed after observation collection.')
    roots=np.concatenate(data['root_ids']);unique=np.unique(roots)
    if len(unique)<5:raise ValueError('Public response validation requires at least five distinct factual roots.')
    held_roots=target_response_validation_roots(data,roots,seed,validation_root_ids)
    held=np.isin(roots,held_roots);train_ids=np.flatnonzero(~held);held_ids=np.flatnonzero(held)
    nodes=torch.as_tensor(np.concatenate(data['nodes']),device=device)
    global_state=torch.as_tensor(np.concatenate(data['global_state']),device=device)
    target=torch.as_tensor(np.concatenate(data['targets']),device=device)
    if nodes.dtype!=torch.float32 or global_state.dtype!=torch.float32 or target.dtype!=torch.float32:
        raise ValueError('Public target-response observations must retain their native float32 representation.')
    def root_weights(indices):
        _,inverse,counts=np.unique(roots[indices],return_inverse=True,return_counts=True)
        weights=1./counts[inverse];return torch.as_tensor(weights/weights.sum(),device=device,dtype=torch.float32)
    probability=root_weights(train_ids);validation_weights=root_weights(held_ids)
    train_ids=torch.as_tensor(train_ids,device=device);held_ids=torch.as_tensor(held_ids,device=device)
    scales=target.new_tensor([.1,.1,.5,.5]);times=target.new_full((len(target),),.2)
    baseline=((target[held_ids]/scales).square().mean(-1)*validation_weights).sum().item()
    states=[];records=[];per_member=updates//ensemble_size
    for member in range(ensemble_size):
        torch.manual_seed(seed+1009*member)
        network=d.target_feedback_network(width,True,representation).to(device)
        optimizer=torch.optim.Adam(network.parameters(),lr=1e-3)
        best_loss=baseline;best_step=0
        best_state={key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
        with torch.enable_grad():
            for step in range(per_member):
                selected=train_ids[torch.multinomial(probability,min(256,len(train_ids)),replacement=True)]
                prediction=torch.cat(network(nodes[selected],global_state[selected],times[selected]),-1)
                loss=((prediction-target[selected])/scales).square().mean()
                rate=1e-4+9e-4*.5*(1.+np.cos(np.pi*step/max(1,per_member-1)))
                for group in optimizer.param_groups:group['lr']=rate
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
                optimizer.step()
                if (step+1)%100==0 or step+1==per_member:
                    with torch.no_grad():
                        estimate=torch.cat(network(nodes[held_ids],global_state[held_ids],times[held_ids]),-1)
                        validation=(((estimate-target[held_ids])/scales).square().mean(-1)*validation_weights).sum().item()
                    if validation<best_loss:
                        best_loss=validation;best_step=step+1
                        best_state={key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
        states.append(best_state);records.append(dict(member=member,updates=per_member,
            selected_update=best_step,validation_normalized_mse=best_loss))
        print(dict(phase='public_target_feedback_fit',seed=seed,**records[-1],
            constant_velocity_normalized_mse=baseline),flush=True)
        del network,optimizer
    predictor=d.PublicTargetFeedback(width,states,True,representation=representation).to(device)
    with torch.no_grad():
        predictions=[torch.cat(network(nodes[held_ids],global_state[held_ids],times[held_ids]),-1)
            for network in predictor.models]
        predicted=torch.stack(predictions).mean(0)
        normalized=(((predicted-target[held_ids])/scales).square().mean(-1)*validation_weights).sum().item()
        coordinate_mse=((predicted-target[held_ids]).square()*validation_weights[:,None]).sum(0).cpu().tolist()
        baseline_mse=(target[held_ids].square()*validation_weights[:,None]).sum(0).cpu().tolist()
    model=copy.deepcopy(torch.load(predecessor,map_location='cpu',weights_only=False))
    model['target_feedback']=dict(width=width,states=states,absolute_frame=True,control_step=.2)
    if representation!='pooled_residual':model['target_feedback']['representation']=representation
    model['target_feedback_fit']=dict(completed=True,seed=seed,updates=updates,ensemble_size=ensemble_size,
        width=width,parameters_per_member=sum(value.numel() for value in predictor.models[0].parameters()),
        observations=len(target),training_roots=int(len(unique)-len(held_roots)),validation_roots=held_roots.tolist(),
        acquisition=specification,physical_steps=data['physical_steps'],members=records,
        validation_normalized_mse=normalized,constant_velocity_normalized_mse=baseline,
        validation_coordinate_mse=coordinate_mse,constant_velocity_coordinate_mse=baseline_mse,
        validation_scope='development prediction validation split by actual factual root; no final evaluation tasks',
        candidate_conditioning='repeated public state updates inside each complete candidate fleet path')
    atomic_save(model,output)
    del predictor
    return model


def fit_target_feedback_rollout_model(data,output,*,initial_model=None,updates=6000,
                                     ensemble_size=3,width=48,seed=42,device='cuda:0',mixture=False,
                                     representation='pooled_residual',validation_root_ids=None):
    """Fit recursive public target dynamics on measured candidate continuations.

    Each rollout uses measured public fleet states as exogenous inputs. The
    target position and velocity are the model's own preceding predictions;
    later target observations enter the loss only. Deployment supplies the
    corresponding complete fleet forecasts instead of measured fleet states.
    """
    if (not data.get('completed') or not data['specification'].get('record_sequences')
            or ensemble_size<1 or updates<ensemble_size or updates%ensemble_size
            or (mixture and ensemble_size<2)):
        raise ValueError('Recursive response fitting needs complete sequences and equal positive member budgets.')
    specification=data['specification'];predecessor=Path(specification['predecessor'])
    if hashlib.sha256(predecessor.read_bytes()).hexdigest()!=specification['predecessor_sha256']:
        raise ValueError('The target-response predecessor changed after sequence acquisition.')
    sequence=np.concatenate(data['sequence_ids']);ticks=np.concatenate(data['sequence_ticks'])
    root_ids=np.concatenate(data['root_ids']);unique_sequences=np.unique(sequence)
    horizon=int(specification['steps']);count=len(unique_sequences)
    if not np.array_equal(unique_sequences,np.arange(data['next_task'])):
        raise ValueError('Every acquired continuation must retain its public sequence identity.')
    if len(np.unique(np.stack((sequence,ticks),-1),axis=0))!=len(sequence):
        raise ValueError('Public continuation sequences contain duplicate physical transitions.')
    arrays={}
    for key in ('public_motion','public_boids','public_target','public_velocity',
                'next_public_target','next_public_velocity'):
        observations=np.concatenate(data[key])
        if observations.dtype!=np.float32 or not np.isfinite(observations).all():
            raise ValueError('Public response sequences must retain finite native float32 measurements.')
        padded=np.zeros((count,horizon,*observations.shape[1:]),np.float32)
        padded[sequence,ticks]=observations
        arrays[key]=torch.as_tensor(padded,device=device)
    mask=np.zeros((count,horizon),bool);mask[sequence,ticks]=True
    if not np.array_equal(mask,np.arange(horizon)[None]<mask.sum(1)[:,None]):
        raise ValueError('Measured continuations must contain an uninterrupted physical prefix.')
    sequence_roots=np.asarray([np.unique(root_ids[sequence==i]).item() for i in unique_sequences])
    roots=np.unique(sequence_roots)
    if len(roots)<5:raise ValueError('Recursive prediction validation needs at least five factual roots.')
    held_roots=target_response_validation_roots(data,roots,seed,validation_root_ids)
    held=np.isin(sequence_roots,held_roots);training_ids=np.flatnonzero(~held);validation_ids=np.flatnonzero(held)
    def root_weights(ids):
        _,inverse,counts=np.unique(sequence_roots[ids],return_inverse=True,return_counts=True)
        weights=1./counts[inverse]
        return torch.as_tensor(weights/weights.sum(),device=device,dtype=torch.float32)
    probability=root_weights(training_ids);validation_weights=root_weights(validation_ids)
    training_ids=torch.as_tensor(training_ids,device=device);validation_ids=torch.as_tensor(validation_ids,device=device)
    mask=torch.as_tensor(mask,device=device)
    initial=None;initial_sha256=None
    if initial_model is not None:
        initial_model=Path(initial_model);initial=torch.load(initial_model,map_location='cpu',weights_only=False)
        initial_sha256=hashlib.sha256(initial_model.read_bytes()).hexdigest()
        fit=initial['target_feedback_fit'];recipe=initial['target_feedback']
        if (fit['acquisition']['predecessor_sha256']!=specification['predecessor_sha256']
                or set(fit['validation_roots'])!=set(held_roots.tolist())
                or recipe['width']!=width or len(recipe['states'])!=ensemble_size
                or recipe.get('representation','pooled_residual')!=representation):
            raise ValueError('Warm target dynamics must preserve the predecessor, capacity and held-out root split.')
    def errors(predictor,indices):
        target=arrays['public_target'][indices,0];velocity=arrays['public_velocity'][indices,0]
        values=[]
        for tick in range(horizon):
            motion=arrays['public_motion'][indices,tick];boids=arrays['public_boids'][indices,tick]
            if predictor is None:
                following=target+.2*velocity;next_velocity=velocity
            elif isinstance(predictor,d.PublicTargetFeedback):
                following,next_velocity=predictor.predict_state(motion,boids,target,velocity)
            else:
                nodes,global_state,_,_,axis=d.PublicTargetMotion.encode_state(motion,boids,target,velocity,True)
                displacement,change=predictor(nodes,global_state,target.new_full((len(indices),),.2))
                def world(vector):
                    return torch.stack((axis[:,0]*vector[:,0]-axis[:,1]*vector[:,1],
                        axis[:,1]*vector[:,0]+axis[:,0]*vector[:,1]),-1)
                following=target+.2*velocity+world(displacement);next_velocity=velocity+world(change)
            values.append(torch.cat((following-arrays['next_public_target'][indices,tick],
                next_velocity-arrays['next_public_velocity'][indices,tick]),-1))
            active=mask[indices,tick,None]
            target=torch.where(active,following,target);velocity=torch.where(active,next_velocity,velocity)
        return torch.stack(values,1)
    scales=torch.ones((horizon,4),device=device,dtype=torch.float32)*.5
    scales[:,:2]=(.1+.05*torch.arange(1,horizon+1,device=device))[:,None]
    def costs(error,indices):
        return ((error/scales).square().mean(-1)*mask[indices]).sum(-1)/mask[indices].sum(-1)
    def validate(predictor,details=False):
        with torch.no_grad():
            residual=torch.cat([errors(predictor,ids) for ids in validation_ids.split(64)])
            loss=(costs(residual,validation_ids)*validation_weights).sum().item()
            mass=validation_weights[:,None]*mask[validation_ids]
            mse=(residual.square()*mass[:,:,None]).sum(0)/mass.sum(0).clamp_min(1e-8)[:,None]
        return (loss,mse.cpu().tolist()) if details else loss
    baseline,baseline_mse=validate(None,True)
    states=[];records=[];per_member=updates//ensemble_size
    mixture_record=None
    if mixture:
        # A single latent component persists throughout a continuation. Each
        # component sees the same public fleet inputs and recursively advances
        # its own target state, rather than averaging incompatible maneuvers.
        networks=torch.nn.ModuleList()
        for member in range(ensemble_size):
            torch.manual_seed(seed+1009*member)
            network=d.target_feedback_network(width,True,representation).to(device)
            if initial is not None:network.load_state_dict(initial['target_feedback']['states'][member])
            networks.append(network)
        initial_losses=[validate(network) for network in networks]
        def mixture_terms(indices):
            residual=torch.stack([errors(network,indices) for network in networks])
            squared=(residual/scales).square().sum(-1)*mask[indices][None]
            log_density=-.5*squared.sum(-1)
            nll=-(torch.logsumexp(log_density,0)-np.log(ensemble_size))/(4*mask[indices].sum(-1))
            responsibility=log_density.softmax(0)
            return nll,responsibility
        def validate_mixture():
            with torch.no_grad():
                values=[mixture_terms(ids) for ids in validation_ids.split(32)]
                loss=(torch.cat([value[0] for value in values])*validation_weights).sum().item()
                mass=(torch.cat([value[1] for value in values],1)*validation_weights[None]).sum(-1)
            return loss,mass.cpu().tolist()
        initial_likelihood,_=validate_mixture();best_loss=initial_likelihood;best_step=0
        best_state=copy.deepcopy(networks.state_dict())
        optimizer=torch.optim.Adam(networks.parameters(),lr=3e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,per_member,eta_min=3e-5)
        with torch.enable_grad():
            for step in range(per_member):
                selected=training_ids[torch.multinomial(probability,32,replacement=True)]
                likelihood,responsibility=mixture_terms(selected)
                mass=responsibility.mean(-1).clamp_min(1e-8)
                loss=likelihood.mean()+.01*(mass*(mass*ensemble_size).log()).sum()
                if not torch.isfinite(loss):raise FloatingPointError('Target trajectory mixture produced a nonfinite loss.')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(networks.parameters(),5.);optimizer.step();schedule.step()
                if (step+1)%100==0 or step+1==per_member:
                    validation,_=validate_mixture()
                    if validation<best_loss:
                        best_loss=validation;best_step=step+1;best_state=copy.deepcopy(networks.state_dict())
                if (step+1)%500==0 or step+1==per_member:
                    print(dict(phase='public_target_mixture_fit',seed=seed,update=step+1,
                        member_updates=(step+1)*ensemble_size,loss=float(loss.detach()),
                        best_validation=best_loss,initial_validation=initial_likelihood),flush=True)
        networks.load_state_dict(best_state)
        _,component_mass=validate_mixture()
        mixture_record=dict(validation_normalized_nll=best_loss,initial_normalized_nll=initial_likelihood,
            selected_update=best_step,validation_component_responsibility=component_mass,
            likelihood='equal-prior mixture of complete recursive public target trajectories with fixed diagonal scales',
            position_scale_by_tick=scales[:,0].cpu().tolist(),velocity_scale=.5,
            component_balance_weight=.01,optimizer_steps=per_member,total_member_updates=updates)
        for member,network in enumerate(networks):
            states.append({key:value.detach().cpu().clone() for key,value in network.state_dict().items()})
            records.append(dict(member=member,updates=per_member,selected_update=best_step,
                initial_validation_normalized_mse=initial_losses[member],validation_normalized_mse=validate(network)))
        del networks,network,optimizer,schedule,best_state
    for member in range(0 if mixture else ensemble_size):
        torch.manual_seed(seed+1009*member)
        network=d.target_feedback_network(width,True,representation).to(device)
        zero_state={key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
        if initial is not None:network.load_state_dict(initial['target_feedback']['states'][member])
        initial_loss=validate(network);best_loss=min(baseline,initial_loss);best_step=0
        best_state=({key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
                    if initial_loss<=baseline else zero_state)
        optimizer=torch.optim.Adam(network.parameters(),lr=3e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,per_member,eta_min=3e-5)
        with torch.enable_grad():
            for step in range(per_member):
                selected=training_ids[torch.multinomial(probability,32,replacement=True)]
                loss=costs(errors(network,selected),selected).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Recursive target dynamics produced a nonfinite loss.')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(),5.);optimizer.step();schedule.step()
                if (step+1)%100==0 or step+1==per_member:
                    validation=validate(network)
                    if validation<best_loss:
                        best_loss=validation;best_step=step+1
                        best_state={key:value.detach().cpu().clone() for key,value in network.state_dict().items()}
                if (step+1)%500==0 or step+1==per_member:
                    print(dict(phase='public_target_rollout_fit',seed=seed,member=member,update=step+1,
                        loss=float(loss.detach()),best_validation=best_loss,constant_velocity=baseline),flush=True)
        states.append(best_state);records.append(dict(member=member,updates=per_member,
            selected_update=best_step,initial_validation_normalized_mse=initial_loss,
            validation_normalized_mse=best_loss))
        del network,optimizer,schedule
    predictor=d.PublicTargetFeedback(width,states,True,scenario_mode='members' if mixture else 'mean',
        representation=representation).to(device)
    normalized,coordinate_mse=validate(predictor,True)
    initial_error=None
    if initial is not None:
        before=d.PublicTargetFeedback(**initial['target_feedback']).to(device)
        initial_error=validate(before);del before
    model=copy.deepcopy(torch.load(predecessor,map_location='cpu',weights_only=False))
    model['target_feedback']=dict(width=width,states=states,absolute_frame=True,control_step=.2)
    if representation!='pooled_residual':model['target_feedback']['representation']=representation
    if mixture:model['target_feedback']['scenario_mode']='members'
    model['target_feedback_fit']=dict(completed=True,seed=seed,updates=updates,ensemble_size=ensemble_size,
        width=width,parameters_per_member=sum(value.numel() for value in predictor.models[0].parameters()),
        observations=len(sequence),sequences=count,sequence_steps=horizon,training_roots=int(len(roots)-len(held_roots)),
        validation_roots=held_roots.tolist(),acquisition=specification,physical_steps=data['physical_steps'],members=records,
        initial_model=None if initial_model is None else str(initial_model),initial_model_sha256=initial_sha256,
        initial_training_updates=0 if initial is None else initial['target_feedback_fit']['updates'],
        validation_normalized_mse=normalized,constant_velocity_normalized_mse=baseline,
        initial_validation_normalized_mse=initial_error,validation_horizon_coordinate_mse=coordinate_mse,
        constant_velocity_horizon_coordinate_mse=baseline_mse,
        objective='recursive public target position and velocity errors across complete measured continuations',
        validation_scope='development prediction validation split by actual factual root; no final evaluation tasks',
        candidate_conditioning='recursive target state and public fleet response inside each complete candidate path')
    if mixture:
        model['target_feedback_fit'].update(mixture=mixture_record,
            objective='complete-trajectory mixture likelihood of recursive candidate-conditioned public target responses',
            forecast_scenarios=ensemble_size,
            initial_total_training_updates=(0 if initial is None else initial['target_feedback_fit']['updates']+
                initial['target_feedback_fit'].get('initial_total_training_updates',
                    initial['target_feedback_fit'].get('initial_training_updates',0))))
    atomic_save(model,output)
    del predictor
    return model
