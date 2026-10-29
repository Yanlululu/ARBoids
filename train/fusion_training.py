"""Train mechanism-one fleet selection on complete factual episodes."""
import study_runtime
import copy
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

import joint_fusion as d
from fusion_prediction import fixed_batch_inference
from joint_fusion import atomic_save, initialize


def fleet_rollouts(policy, tasks, batch_size=1, device='cpu', *, keep_steps=False,
                   actor_override=None, collect_fleet=False, root_sample_seed=None,
                   fixed_inference_width=0):
    """Keep each layout/future's dynamics stream separate during batched inference.

    Optional reservoir sampling records one factual state per complete episode.
    Its private random stream never changes the policy or the environment.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('Episode batch size must be a positive integer.')
    if (not isinstance(fixed_inference_width, int) or isinstance(fixed_inference_width, bool)
            or fixed_inference_width < 0):
        raise ValueError('Fixed inference width must be a nonnegative integer.')
    if collect_fleet and fixed_inference_width and batch_size > fixed_inference_width:
        raise ValueError('Fleet collection requires one inference chunk per episode batch.')
    if collect_fleet and (keep_steps or root_sample_seed is not None):
        raise ValueError('Collect factual trajectories separately from sampled fleet training.')
    tasks = list(tasks)
    if not tasks:
        raise ValueError('At least one complete episode is required.')
    initialize()
    actor = (d.joint_load(policy) if actor_override is None else actor_override).to(device)
    fleet = actor.feedback_fleet_policy
    if collect_fleet and (fleet is None or not fleet.sample or not fleet.collect):
        raise ValueError('Fleet training requires a sampling policy and decision collection.')
    results = []
    for begin in range(0, len(tasks), batch_size):
        current = tasks[begin:begin+batch_size]
        environments, observations, random_states = [], [], []
        for scene, future in current:
            seed, n, agility = scene
            if n < 2 or not math.isfinite(agility) or agility <= 0:
                raise ValueError('Episodes require at least two defenders and positive agility.')
            np.random.seed(int(seed))
            if actor.environment_protocol == 'source':
                env = d.numerical_source().TADEnv(n)
            else:
                from envs.TADgame import TADEnv
                env = TADEnv(n, protocol=actor.environment_protocol)
            obs, _ = env.reset(agility=agility, noisy_agility=False)
            if future is not None:
                np.random.seed(int(future))
            environments.append(env)
            observations.append(obs)
            random_states.append(np.random.get_state())
        count = len(current)
        rows, roots = [None]*count, [None]*count
        root_counts = np.zeros(count, dtype=int)
        root_rngs = ([np.random.default_rng(np.random.SeedSequence(
            [int(root_sample_seed), int(scene[0]), 0 if future is None else int(future), 39797]))
            for scene, future in current] if root_sample_seed is not None else None)
        trajectories = [[] for _ in current]
        fleet_steps = [[] for _ in current]
        source_returns, task_returns = np.zeros(count), np.zeros(count)
        active = list(range(count))
        maximum_steps = max(math.ceil(env.Total_T/.2)+1 for env in environments)
        for step in range(maximum_steps):
            if not active:
                break
            if root_rngs is not None and step % 5 == 0:
                for i in active:
                    root_counts[i] += 1
                    if root_rngs[i].integers(root_counts[i]) == 0:
                        roots[i] = dict(environment=copy.deepcopy(environments[i]),
                            observation=observations[i].copy(), random_state=copy.deepcopy(random_states[i]),
                            root_steps=step)
            controls = {}
            for n in sorted({current[i][0][1] for i in active}):
                indices = [i for i in active if current[i][0][1] == n]
                packets = [d.joint_packet(environments[i], observations[i]) for i in indices]
                packet = {key:torch.as_tensor(np.stack([p[key] for p in packets]), device=device)
                          for key in packets[0]}
                with torch.no_grad():
                    physical = fixed_batch_inference(actor.control, packet,
                        width=fixed_inference_width).cpu().numpy()
                for j, i in enumerate(indices):
                    controls[i] = physical[j]
                    if collect_fleet:
                        fleet_steps[i].append({key:value[j].clone()
                            for key,value in fleet.last_decision.items()})
            following = []
            for i in active:
                env = environments[i]
                np.random.set_state(random_states[i])
                before = d.joint_packet(env, observations[i]) if keep_steps else None
                obs, reward, done, source_reward = d.joint_task_step(
                    env, controls[i], actor.training_gamma, actor.training_step_cost)
                if collect_fleet:
                    fleet_steps[i][-1].update(reward=reward, done=bool(done), discount=actor.training_gamma)
                if keep_steps:
                    trajectories[i].append(dict(packet=before, action=controls[i].copy(),
                        done=done, physical_steps=1, reward=reward, source_reward=source_reward,
                        discount=actor.training_gamma))
                random_states[i] = np.random.get_state()
                observations[i] = obs
                source_returns[i] += .99**step*source_reward
                task_returns[i] += actor.training_gamma**step*reward
                if done:
                    scene, future = current[i]
                    seed, n, agility = scene
                    rows[i] = dict(seed=int(seed), defenders=n, agility=agility, **d.outcome(env,done),
                        terminal=int(done), physical_steps=step+1, completed=True,
                        discounted_return=float(source_returns[i]), task_return=float(task_returns[i]),
                        task_discount=actor.training_gamma**(step+1), environment_protocol=actor.environment_protocol)
                    if future is not None:
                        rows[i]['future_seed'] = int(future)
                    if fixed_inference_width:
                        rows[i]['fixed_inference_width'] = fixed_inference_width
                else:
                    following.append(i)
            active = following
        if active:
            raise RuntimeError('A source episode did not reach its physical terminal.')
        for i, row in enumerate(rows):
            if collect_fleet:
                result = dict(row=row, fleet_steps=fleet_steps[i])
            elif keep_steps:
                continuation = 0.
                for record in reversed(trajectories[i]):
                    continuation = record['source_reward']+.99*continuation
                    record['source_return'] = continuation
                result = dict(row=row, steps=trajectories[i], simulation_steps=len(trajectories[i]))
            else:
                result = row
            if root_rngs is not None:
                roots[i]['eligible_states'] = int(root_counts[i])
                if not keep_steps:
                    result = dict(row=result)
                result['root_snapshot'] = roots[i]
            results.append(result)
    actor.cpu()
    return results


def collect_factual_data(predecessor, tasks, output, *, seed=42, batch_size=1,
                         device='cpu', keep_steps=True, fixed_inference_width=0):
    """Create M1 prediction inputs without depending on a value-learning bank."""
    predecessor = Path(predecessor)
    if predecessor.resolve() == Path(output).resolve():
        raise ValueError('Keep factual trajectories separate from their predecessor checkpoint.')
    identity = hashlib.sha256(predecessor.read_bytes()).hexdigest()
    tasks = list(tasks)
    rollouts = fleet_rollouts(predecessor, tasks, batch_size, device,
        keep_steps=keep_steps, root_sample_seed=seed, fixed_inference_width=fixed_inference_width)
    if hashlib.sha256(predecessor.read_bytes()).hexdigest() != identity:
        raise ValueError('The factual predecessor changed during collection.')
    records = [dict(scene=tuple(scene), future_seed=future, root_snapshot=row['root_snapshot'])
               for (scene,future),row in zip(tasks,rollouts)]
    data = dict(completed=True, predecessor=str(predecessor.resolve()), predecessor_sha256=identity,
        tasks=tasks, records=records, seed=seed, fixed_inference_width=fixed_inference_width,
        physical_steps=sum(row['row']['physical_steps'] for row in rollouts))
    if keep_steps:
        data['rollouts'] = rollouts
    atomic_save(data, output)
    return data


def training_bank(start, per_cell, cells):
    return [(start+c*10000+i,n,a) for c,(n,a) in enumerate(cells) for i in range(per_cell)]


def fleet_anchor_kl(anchor_distribution, distribution):
    """Evaluate the trust penalty in log space, including very rare plans."""
    independent=isinstance(distribution,torch.distributions.Independent)
    reference=anchor_distribution.base_dist if independent else anchor_distribution
    current=distribution.base_dist if independent else distribution
    value=(reference.probs*(reference.logits-current.logits)).sum(-1)
    return value.sum(-1) if independent else value


def fleet_return_targets(rollouts, credit_target='discounted_task'):
    """Assign observed episode outcomes without changing the environment reward.

    Terminal-success credit keeps the probability of eventual success as the
    state baseline. It neither discounts a late capture nor centers advantages
    across decisions from trajectories of different lengths.
    The paired variant subtracts the observed frozen-predecessor outcome on
    the same exogenous random stream. That outcome is independent of sampled
    learner actions; shared successes and failures cancel without changing
    the task objective. Its critic predicts remaining terminal improvement.
    """
    if credit_target not in ('discounted_task', 'terminal_success', 'paired_terminal_success'):
        raise ValueError('Unknown fleet learning target.')
    records=[];advantages=[];targets=[]
    for rollout in rollouts:
        steps=rollout['fleet_steps']
        if credit_target in ('terminal_success','paired_terminal_success'):
            target=float(rollout['row']['success'])
            if target not in (0.,1.):
                raise ValueError('Fleet success credit requires actual terminal outcomes.')
            if credit_target=='paired_terminal_success':
                factual=rollout.get('factual_success')
                if factual not in (0.,1.):
                    raise ValueError('Paired fleet credit requires a measured fixed-predecessor outcome.')
                target-=float(factual)
            for record in steps:
                baseline=(float(record['value']) if credit_target=='paired_terminal_success' else
                    1./(1.+np.exp(-np.clip(float(record['value']),-60.,60.))))
                records.append(record);advantages.append(target-baseline);targets.append(target)
            continue
        following_value=0.;advantage=0.;local_advantages=[];local_targets=[]
        for record in reversed(steps):
            value=float(record['value']);discount=record['discount']*(not record['done'])
            delta=record['reward']+discount*following_value-value
            advantage=delta+discount*.95*advantage
            local_advantages.append(advantage);local_targets.append(advantage+value)
            following_value=value
        records.extend(steps);advantages.extend(reversed(local_advantages));targets.extend(reversed(local_targets))
    return records,advantages,targets


def fit_feedback_fleet_policy(predecessor,output,*,seed=42,iterations=24,episodes_per_iteration=32,
                              batch_size=1,width=48,epochs=4,minibatch=64,learning_rate=3e-4,
                              analytic_temperature=2.,entropy_weight=.01,device='cpu',
                              candidate_contrast=False,sampling_temperature=1.,
                              training_agilities=(2.,2.5),training_task_base=4166500000,
                              representation='candidate_graph',credit_target='discounted_task',
                              anchor_kl_weight=0.,fixed_inference_width=0,scenario_evaluation=False,
                              scenario_regret_weight=0.,training_defenders=6):
    """Optimize candidate selection on actual complete source episodes.

    SAC, Boids, dynamics and the task reward stay frozen. Each action is one
    complete feedback forecast from the shared fleet candidate library, or
    independent per-vessel reference choices for the matched learned control.
    Both use the same graph and receive real on-policy team-return credit.
    """
    from actuator_fusion import FeedbackFleetPolicy
    initialize();predecessor=Path(predecessor);output=Path(output)
    if any(not isinstance(value,int) or isinstance(value,bool) or value<1 for value in
           (iterations,episodes_per_iteration,batch_size,width,epochs,minibatch,training_defenders)) or training_defenders<2:
        raise ValueError('Fleet learning requires positive integer budgets and at least two defenders.')
    if (not np.isfinite(learning_rate) or learning_rate<=0. or not np.isfinite(entropy_weight) or entropy_weight<0.):
        raise ValueError('Fleet learning requires a positive finite learning rate and nonnegative entropy weight.')
    if output.resolve()==predecessor.resolve():
        raise ValueError('Keep the frozen predecessor separate from the trained policy.')
    if credit_target not in ('discounted_task','terminal_success','paired_terminal_success'):
        raise ValueError('Unknown fleet learning target.')
    if not np.isfinite(anchor_kl_weight) or anchor_kl_weight<0.:
        raise ValueError('The fixed-predecessor trust penalty must be nonnegative.')
    if (not isinstance(fixed_inference_width,int) or isinstance(fixed_inference_width,bool)
            or fixed_inference_width<0 or fixed_inference_width and batch_size>fixed_inference_width):
        raise ValueError('Fixed fleet learning requires one inference chunk per episode batch.')
    training_agilities=tuple(map(float,training_agilities))
    if (not training_agilities or len(set(training_agilities))!=len(training_agilities)
            or not np.isfinite(training_agilities).all() or min(training_agilities)<=0.
            or not isinstance(training_task_base,int) or training_task_base<0):
        raise ValueError('Fleet training requires distinct positive agilities and a nonnegative task base.')
    source_hash=hashlib.sha256(predecessor.read_bytes()).hexdigest()
    source=torch.load(predecessor,map_location='cpu',weights_only=False)
    config=source.get('actuator_fusion_refinement') or {}
    selection=config.get('selection')
    if not config.get('task_feedback_prediction') or selection not in ('joint','single_reference'):
        raise ValueError('Fleet learning requires a complete-feedback joint or independent predecessor.')
    specification=dict(seed=seed,episodes_per_iteration=episodes_per_iteration,batch_size=batch_size,
        width=width,epochs=epochs,minibatch=minibatch,learning_rate=learning_rate,
        analytic_temperature=analytic_temperature,entropy_weight=entropy_weight,
        predecessor=str(predecessor),predecessor_sha256=source_hash,
        objective='on-policy clipped joint-fleet task-return policy gradient with a learned state baseline',
        candidate_budget='same 81 balanced fleet plans plus every single-vessel reference change',
        source_policies_frozen=True,training_task_base=training_task_base,training_defenders=training_defenders)
    if fixed_inference_width:
        specification.update(fixed_inference_width=fixed_inference_width,collected_feature_dtype='float32')
    if scenario_evaluation:
        specification['scenario_evaluation']=True
    if scenario_regret_weight:
        specification['scenario_regret_weight']=scenario_regret_weight
    if training_agilities!=(2.,2.5):
        specification['training_agilities']=training_agilities
    if representation!='candidate_graph':
        specification['representation']=representation
    if candidate_contrast:
        specification['candidate_contrast']=True
        specification['objective']='within-state candidate-contrast clipped joint-fleet task-return policy gradient'
    if selection=='single_reference':
        specification['selection']=selection
        specification['objective']='factorized single-vessel reference policy gradient on the same complete-feedback queries and team returns'
    if sampling_temperature!=1.:
        specification['sampling_temperature']=sampling_temperature
    if config.get('independent_feedback_proposal'):
        specification['candidate_budget']+=' and its complete independent recombination'
    if config.get('nominal_feedback_proposals'):
        specification['candidate_budget']+=' plus jointly and independently proposed nominal fleet plans'
    if config.get('feedback_refinement_rounds'):
        specification['feedback_refinement_rounds']=config['feedback_refinement_rounds']
        specification['candidate_budget']+=' with common complete-feedback single-vessel neighborhood expansions'
    if credit_target!='discounted_task':
        specification.update(credit_target=credit_target,
            objective='undiscounted observed terminal-success policy gradient with a probability baseline')
    if credit_target=='paired_terminal_success':
        specification.update(objective='terminal-success policy gradient with a measured common-task predecessor control variate',
            factual_control_variate='one frozen deterministic predecessor episode on each identical task and random stream',
            factual_policy_sha256=source_hash,critic_initialization='zero terminal-gain output')
    if anchor_kl_weight:
        specification.update(anchor_kl_weight=anchor_kl_weight,
            anchor_policy_sha256=source_hash,anchor_policy_frozen=True)
    model=torch.load(output,map_location='cpu',weights_only=False) if output.exists() else copy.deepcopy(source)
    history=model.get('fleet_learning',{})
    if history and history['specification']!=specification:
        raise ValueError('Cannot resume fleet learning with changed data, budget or source parameters.')
    recipe=model.get('feedback_fleet_policy') or {}
    if recipe and recipe.get('representation','candidate_graph')!=representation:
        raise ValueError('A changed fleet representation requires fresh parameters.')
    torch.manual_seed(seed);np.random.seed(seed)
    actor=copy.deepcopy(d.joint_load(str(predecessor),predecessor.stat().st_mtime_ns)).to(device)
    learner=FeedbackFleetPolicy(width,analytic_temperature,
        (model.get('feedback_fleet_policy') or {}).get('state'),candidate_contrast,
        sampling_temperature,representation,scenario_evaluation,scenario_regret_weight).to(device)
    if fixed_inference_width:learner.collection_dtype=torch.float32
    reference_actor=None
    if credit_target=='paired_terminal_success':
        reference_actor=copy.deepcopy(d.joint_load(str(predecessor),predecessor.stat().st_mtime_ns)).to(device)
        reference_actor.eval()
        if reference_actor.feedback_fleet_policy is not None:
            reference_actor.feedback_fleet_policy.sample=False
            reference_actor.feedback_fleet_policy.collect=False
        if not history:
            torch.nn.init.zeros_(learner.critic[-1].weight)
            torch.nn.init.zeros_(learner.critic[-1].bias)
    anchor=None
    if anchor_kl_weight:
        anchor_recipe=source.get('feedback_fleet_policy') or {}
        if not anchor_recipe.get('state'):
            raise ValueError('The trust penalty needs a previously learned candidate policy.')
        if (anchor_recipe.get('scenario_evaluation',False)!=scenario_evaluation
                or anchor_recipe.get('scenario_regret_weight',0.)!=scenario_regret_weight):
            raise ValueError('The trust reference must use the same complete-scenario representation.')
        anchor=FeedbackFleetPolicy(**anchor_recipe).to(device).eval().requires_grad_(False)
    actor.feedback_fleet_policy=learner
    optimizer=torch.optim.Adam(learner.parameters(),lr=learning_rate)
    if history:
        optimizer.load_state_dict(history['optimizer'])
        torch.set_rng_state(history['torch_rng'])
        if device.startswith('cuda'):torch.cuda.set_rng_state(history['cuda_rng'],device)
        np.random.set_state(history['numpy_rng'])
    completed=int(history.get('completed_iterations',0));learning=list(history.get('iterations',[]))
    for iteration in range(completed,iterations):
        started=time.monotonic();learner.sample=True;learner.collect=True;learner.eval()
        tasks=[(scene,None) for scene in training_bank(training_task_base+seed*100000+iteration*episodes_per_iteration,
            max(1,(episodes_per_iteration+len(training_agilities)-1)//len(training_agilities)),
            tuple((training_defenders,agility) for agility in training_agilities))][:episodes_per_iteration]
        with torch.no_grad():
            reference_rows=(fleet_rollouts(predecessor,tasks,batch_size=batch_size,device=device,
                actor_override=reference_actor,fixed_inference_width=fixed_inference_width)
                if reference_actor is not None else None)
            rollouts=fleet_rollouts(predecessor,tasks,batch_size=batch_size,device=device,
                actor_override=actor,collect_fleet=True,fixed_inference_width=fixed_inference_width)
        if reference_rows is not None:
            if len(reference_rows)!=len(rollouts):
                raise ValueError('Every sampled fleet trajectory needs its actual matched predecessor episode.')
            for rollout,reference_row in zip(rollouts,reference_rows):
                rollout['factual_success']=float(reference_row['success'])
        learner.to(device);learner.sample=False;learner.collect=False;learner.train()
        records,advantages,targets=fleet_return_targets(rollouts,credit_target)
        if not records:
            raise RuntimeError('Fleet training collected no physical decisions.')
        advantages=torch.tensor(advantages,dtype=torch.float32,device=device)
        if credit_target in ('terminal_success','paired_terminal_success'):
            advantages=advantages/advantages.std(unbiased=False).clamp_min(.1)
        else:
            advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
        targets=torch.tensor(targets,dtype=torch.float32,device=device)
        losses=[];divergences=[];stop=False;updates=0
        with torch.enable_grad():
            for epoch in range(epochs):
                for begin in range(0,len(records),minibatch):
                    if begin==0:permutation=torch.randperm(len(records)).tolist()
                    indices=permutation[begin:begin+minibatch]
                    selected=[records[index] for index in indices]
                    features=torch.stack([record['features'] for record in selected]).to(device,torch.float32)
                    costs=torch.stack([record['costs'] for record in selected]).to(device)
                    choices=torch.stack([record['index'] for record in selected]).to(device)
                    previous=torch.stack([record['logp'] for record in selected]).to(device)
                    logits,values=learner(features,costs)
                    distribution=learner.policy_distribution(logits,selection,features.shape[-2],
                        temperature=learner.sampling_temperature)
                    log_ratio=distribution.log_prob(choices)-previous
                    ratio=log_ratio.exp();advantage=advantages[indices]
                    policy_loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                    value_loss=(torch.nn.functional.binary_cross_entropy_with_logits(values,targets[indices])
                        if credit_target=='terminal_success' else
                        torch.nn.functional.smooth_l1_loss(values,targets[indices]))
                    loss=policy_loss+.5*value_loss-entropy_weight*distribution.entropy().mean()
                    if anchor is not None:
                        with torch.no_grad():
                            anchor_logits,_=anchor(features,costs)
                            anchor_distribution=anchor.policy_distribution(anchor_logits,selection,
                                features.shape[-2],temperature=learner.sampling_temperature)
                        loss=loss+anchor_kl_weight*fleet_anchor_kl(anchor_distribution,distribution).mean()
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Fleet learning produced a nonfinite objective before an update.')
                    divergence=((ratio-1)-log_ratio).mean().detach()
                    if epoch>0 and float(divergence)>.03:
                        stop=True;break
                    optimizer.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(learner.parameters(),.5);optimizer.step();updates+=1
                    losses.append(float(loss.detach()));divergences.append(float(divergence))
                if stop:break
        record=dict(iteration=iteration+1,episodes=len(rollouts),physical_steps=len(records),updates=updates,
            success=sum(rollout['row']['success'] for rollout in rollouts),
            task_return=float(np.mean([rollout['row']['task_return'] for rollout in rollouts])),
            loss=float(np.mean(losses)),mean_kl=float(np.mean(divergences)),
            elapsed_seconds=time.monotonic()-started,tasks=tasks)
        if reference_rows is not None:
            record.update(factual_success=sum(row['success'] for row in reference_rows),
                factual_outcomes=[bool(row['success']) for row in reference_rows],
                factual_physical_steps=sum(row['physical_steps'] for row in reference_rows),
                rescued=sum(rollout['row']['success'] and not reference['success']
                    for rollout,reference in zip(rollouts,reference_rows)),
                harmed=sum(reference['success'] and not rollout['row']['success']
                    for rollout,reference in zip(rollouts,reference_rows)))
        learning.append(record)
        model['feedback_fleet_policy']=dict(width=width,analytic_temperature=analytic_temperature,
            state={key:value.detach().cpu().clone() for key,value in learner.state_dict().items()})
        if candidate_contrast:model['feedback_fleet_policy']['candidate_contrast']=True
        if scenario_evaluation:model['feedback_fleet_policy']['scenario_evaluation']=True
        if scenario_regret_weight:model['feedback_fleet_policy']['scenario_regret_weight']=scenario_regret_weight
        if representation!='candidate_graph':
            model['feedback_fleet_policy']['representation']=representation
        if sampling_temperature!=1.:
            model['feedback_fleet_policy']['sampling_temperature']=sampling_temperature
        model['fleet_learning']=dict(specification=specification,completed_iterations=iteration+1,
            iterations=learning,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
            numpy_rng=np.random.get_state(),total_training_episodes=sum(row['episodes'] for row in learning),
            total_physical_steps=sum(row['physical_steps'] for row in learning))
        if reference_rows is not None:
            model['fleet_learning'].update(total_factual_episodes=sum(row['episodes'] for row in learning),
                total_factual_physical_steps=sum(row['factual_physical_steps'] for row in learning),
                total_environment_episodes=2*sum(row['episodes'] for row in learning))
        atomic_save(model,output)
        print(dict(phase='feedback_fleet_learning',policy=str(output),seed=seed,
            **{key:value for key,value in record.items() if key not in ('tasks','factual_outcomes')}),flush=True)
        del records,rollouts,features,costs,logits,values,loss
        actor.to(device)
    actor.cpu();learner.eval();learner.sample=False;learner.collect=False
    return model
