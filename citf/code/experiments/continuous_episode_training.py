"""Whole-task PPO with persistent exploration, optional replay and teacher warmstart."""
import argparse
import concurrent.futures
from dataclasses import asdict
import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.bridge import Source, World
from citf.contracts import Protocol, immutable_run_contract, require_same_contract, file_hash, validate_task_banks
from citf.train import atomic_save, read_journal
from experiments.contact_event_policy import retained_teacher
from experiments.continuous_actor_critic import observed_history, ReferenceController
from experiments.continuous_episode_policy import EpisodeConfig, EpisodeActorCritic, load_policy
from experiments.interaction_trajectories import atomic_json
from experiments.synchronized_configuration_policy import joint


WORKER = {}


def initialize_worker(arguments, protocol, configuration):
    torch.set_num_threads(1)
    source = Source(arguments['project'], arguments['source'])
    cfg = EpisodeConfig(**configuration)
    teacher = ReferenceController(source, cfg, bank=12, budget=96, rounds=1)
    coach = (joint.JointPolicyController(source, retained_teacher(cfg.teacher_checkpoint,cfg.teacher_sha256),
        bank=12, budget=96, rounds=1) if arguments['warmstart_episodes'] else None)
    WORKER.update(arguments=arguments, protocol=Protocol(**protocol), teacher=teacher,
        model=None, version=None, coach=coach)


@torch.no_grad()
def collect_episode(job):
    args = WORKER['arguments']
    out = Path(args['out'])
    if WORKER['version'] != job['update']:
        WORKER['model'] = load_policy(out / 'rollout_policy.pt')
        WORKER['version'] = job['update']
    model = WORKER['model']
    torch.manual_seed(job['policy_seed'])
    task = job['task']
    world = World(args['project'], WORKER['protocol'], task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    previous = np.zeros((task['defenders'], 2), np.float32)
    states = [world.state().astype(np.float32)]
    data = {key: [] for key in ['reference', 'previous', 'latent', 'logp', 'value', 'force']}
    started = time.time()
    while not world.done:
        reference, _ = WORKER['teacher'].act(world)
        reference = np.asarray(reference, np.float32)
        distribution, value = model.distribution(
            torch.from_numpy(observed_history(world,model.cfg.history))[None],
            torch.tensor([world.protocol.time_limit-world.env.Current_T], dtype=torch.float32),
            torch.from_numpy(reference)[None], torch.from_numpy(previous)[None])
        expert_force = WORKER['coach'].act(world)[0] if job.get('teacher_rollout') else None
        if expert_force is None:
            latent_tensor = distribution.sample()
        else:
            difference = torch.as_tensor(expert_force-reference,dtype=torch.float32) / model.cfg.residual_scale
            latent_tensor = difference.clamp(-1+1e-6,1-1e-6).atanh()[None]
        latent = latent_tensor[0].numpy()
        force = (model.force(torch.from_numpy(reference)[None], latent_tensor)[0].numpy()
                 if expert_force is None else np.asarray(expert_force,np.float32))
        data['reference'].append(reference)
        data['previous'].append(previous)
        data['latent'].append(latent)
        data['logp'].append(float(distribution.log_prob(latent_tensor).sum()))
        data['value'].append(float(value[0]))
        data['force'].append(force)
        result = world.step(force)
        states.append(world.state().astype(np.float32))
        previous = latent
    steps = len(data['latent'])
    terminal = 1. if result['success'] else -1.
    returns = terminal - args['time_cost'] * np.arange(steps, 0, -1, dtype=np.float32)
    filename = ('warm_' if job.get('teacher_rollout') else '') + f"{job['ordinal']:07d}.npz"
    path=out/'episodes'/filename
    with path.with_suffix('.pending').open('wb') as stream:
        np.savez_compressed(stream, state=np.stack(states), returns=returns,
            **{key: np.asarray(value, np.float32) for key, value in data.items()})
    path.with_suffix('.pending').replace(path)
    row = dict(update=job['update'], ordinal=job['ordinal'], task=task,
        sampling=job['sampling'], replay_source=job.get('replay_source'), policy_seed=job['policy_seed'],
        episode=filename, steps=steps, episode_return=float(returns[0]),
        absolute_physical_residual=float(np.abs(np.asarray(data['force'])-np.asarray(data['reference'])).mean()),
        teacher_rollout=bool(job.get('teacher_rollout')), elapsed_seconds=time.time()-started, **result)
    return row


def complete_batch(rows, out, time_limit, control_dt, device, history=8):
    result = {key: [] for key in ['history', 'remaining', 'reference', 'previous', 'latent', 'logp', 'value', 'returns', 'force']}
    for row in rows:
        with np.load(out / 'episodes' / row['episode']) as arrays:
            steps = row['steps']
            indices = np.maximum(0, np.arange(steps)[:, None] + np.arange(1-history, 1)[None])
            result['history'].append(arrays['state'][indices])
            result['remaining'].append(time_limit-control_dt*np.arange(steps, dtype=np.float32))
            for key in ['reference', 'previous', 'latent', 'logp', 'value', 'returns', 'force']:
                result[key].append(arrays[key].copy())
    data = {key: torch.as_tensor(np.concatenate(value), dtype=torch.float32, device=device)
        for key, value in result.items()}
    advantage = data['returns']-data['value']
    data['advantage'] = (advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
    return data


def update_policy(model, optimizer, data, args, rng):
    model.train()
    records = []
    early = False
    for epoch in range(args.epochs):
        order = rng.permutation(len(data['returns']))
        for start in range(0, len(order), args.minibatch):
            index = order[start:start+args.minibatch]
            distribution, value = model.distribution(data['history'][index], data['remaining'][index],
                data['reference'][index], data['previous'][index])
            logp = distribution.log_prob(data['latent'][index]).sum((-1, -2))
            delta = logp-data['logp'][index]
            ratio = delta.clamp(-20, 20).exp()
            kl = float(((ratio-1)-delta).mean().detach())
            if kl > args.target_kl and records:
                early = True
                break
            advantage = data['advantage'][index]
            policy = -torch.minimum(ratio*advantage,
                ratio.clamp(1-args.clip, 1+args.clip)*advantage).mean()
            clipped = data['value'][index]+(value-data['value'][index]).clamp(-.2, .2)
            value_loss = .5*torch.maximum((value-data['returns'][index]).square(),
                (clipped-data['returns'][index]).square()).mean()
            loss = policy+.5*value_loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite complete-episode learning objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            with torch.no_grad():
                model.log_std.clamp_(-5., -1.)
            records.append(dict(policy=float(policy.detach()), value_loss=float(value_loss.detach()),
                kl=kl, gradient=float(gradient), clip_fraction=float((abs(ratio-1)>args.clip).float().mean().detach())))
        if early:
            break
    target, predicted = data['returns'], data['value']
    detail = {key: float(np.mean([row[key] for row in records])) for key in records[0]}
    detail.update(minibatch_updates=len(records), kl_stopped=early,
        monte_carlo_value_explained_variance=float((1-(target-predicted).var(unbiased=False)/target.var(unbiased=False).clamp_min(1e-8)).detach()),
        stationary_latent_std=model.log_std.exp().detach().cpu().tolist())
    return detail


def main():
    parser = argparse.ArgumentParser()
    for name in ['project', 'source', 'tasks', 'out']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--teacher')
    parser.add_argument('--initial')
    parser.add_argument('--encoder')
    parser.add_argument('--failures')
    parser.add_argument('--reference-mode',choices=['teacher','source','direct'])
    parser.add_argument('--warmstart-episodes',type=int,default=0)
    parser.add_argument('--warmstart-updates',type=int,default=200)
    parser.add_argument('--seed', type=int, default=9501)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--updates', type=int, default=24)
    parser.add_argument('--episodes-per-update', type=int, default=64)
    parser.add_argument('--stages', nargs='+', type=int, default=[8, 16, 24])
    parser.add_argument('--failure-fraction', type=float)
    parser.add_argument('--correlation', type=float, default=.8)
    parser.add_argument('--minibatch', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--value-lr', type=float, default=3e-4)
    parser.add_argument('--clip', type=float, default=.2)
    parser.add_argument('--target-kl', type=float, default=.025)
    parser.add_argument('--time-cost', type=float, default=.00005)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.failure_fraction is None:
        args.failure_fraction=.5 if args.failures else 0.
    if not 0<=args.failure_fraction<=1 or not 0<=args.correlation<1:
        parser.error('Replay fraction must be in [0,1] and correlation in [0,1).')
    if min(args.workers,args.updates,args.episodes_per_update,args.minibatch,args.epochs,args.warmstart_updates)<1 or args.warmstart_episodes<0:
        parser.error('Training sizes must be positive; warmstart episodes may be zero.')
    if min(args.lr,args.encoder_lr,args.value_lr,args.clip,args.target_kl)<=0 or args.time_cost<0:
        parser.error('Learning rates, clipping and KL limit must be positive; time cost must be nonnegative.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    (out / 'episodes').mkdir(exist_ok=True)
    bank = json.loads(Path(args.tasks).read_text())
    if bank['role'] != 'train':
        raise ValueError('Episode policy learning requires training layouts')
    validate_task_banks(('train',bank))
    families = {row['geometry_family_id'] for row in bank['tasks']}
    physical = Protocol(**bank['protocol'])
    old = [json.loads(line) for line in Path(args.failures).read_text().splitlines()] if args.failures else []
    failures = [row for row in old if not row['success']]
    if (args.failure_fraction>0 and not failures) or any(row['task']['geometry_family_id'] not in families for row in failures):
        raise ValueError('Failure replay must come from the declared training layouts')
    allocated={(x['geometry_family_id'],x['scene_seed'],x['defenders'],x['agility']) for x in bank['tasks']}
    if any((x['task']['geometry_family_id'],x['task']['scene_seed'],x['task']['defenders'],x['task']['agility']) not in allocated for x in failures):
        raise ValueError('Replay geometry, team size and agility must match an allocated training task')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed+41)
    initial = torch.load(args.initial, map_location='cpu', weights_only=False) if args.initial else None
    configuration = dict(initial['config']) if initial else dict(history=physical.history_steps)
    configuration.update(revision=EpisodeConfig.revision,correlation=args.correlation)
    if args.teacher:
        configuration['teacher_checkpoint']=str(Path(args.teacher).resolve())
    if args.encoder:
        configuration['encoder_checkpoint']=str(Path(args.encoder).resolve())
    configuration['reference_mode']=args.reference_mode or configuration.get('reference_mode',
        'teacher' if configuration.get('teacher_checkpoint') else 'direct')
    if configuration['reference_mode']=='teacher' or args.warmstart_episodes:
        if not configuration.get('teacher_checkpoint'):raise ValueError('Teacher use requires --teacher')
        configuration['teacher_sha256']=file_hash(configuration['teacher_checkpoint'])
    cfg = EpisodeConfig(**configuration)
    if cfg.history!=physical.history_steps:
        raise ValueError('Actor history must match the declared task history window')
    if (cfg.reference_mode=='teacher' or args.warmstart_episodes) and not cfg.teacher_checkpoint:
        raise ValueError('Teacher control or teacher warmstart requires --teacher')
    model = EpisodeActorCritic(cfg).to(args.device)
    state = model.state_dict()
    if initial:
        state.update(initial['state'])
    model.load_state_dict(state)
    if args.encoder and not initial:
        model.initialize_encoder(args.encoder)
    actor_params = [*model.command.parameters(), *model.action_context.parameters(),
        *model.actor.parameters(), *model.previous.parameters(), model.log_std]
    optimizer = torch.optim.Adam([dict(params=model.encoder.parameters(), lr=args.encoder_lr),
        dict(params=actor_params, lr=args.lr), dict(params=model.critic.parameters(), lr=args.value_lr)], eps=1e-5)
    uses_teacher=cfg.reference_mode=='teacher' or args.warmstart_episodes>0
    teacher_hash=(file_hash(cfg.teacher_checkpoint) if uses_teacher else
                  (initial or {}).get('teacher_sha256'))
    encoder_hash=file_hash(args.encoder) if args.encoder else (initial or {}).get('encoder_initialization_sha256')
    contract_args={k:v for k,v in vars(args).items() if k not in ('resume','workers','device','out','updates','stages')}
    contract=immutable_run_contract(dict(arguments=contract_args,config=asdict(cfg),physical_protocol=asdict(physical)),
        files=[path for path in (args.tasks,args.failures,args.initial,
            cfg.teacher_checkpoint if uses_teacher else None,args.encoder,args.source) if path]+[
            Path(args.project)/'train'/name for name in ('envs/TADgame.py','envs/modules.py','policy/networks.py')],
        code=[__file__])
    update = steps = episodes = warmstart_steps = 0
    warmstart_completed=args.warmstart_episodes==0
    if args.resume:
        payload = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        require_same_contract(payload.get('contract'),contract,'Complete-episode training resume')
        model.load_state_dict(payload['state'])
        optimizer.load_state_dict(payload['optimizer'])
        rng.bit_generator.state = payload['numpy_rng']
        torch.set_rng_state(payload['torch_rng'])
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        update, steps, episodes = payload['update'], payload['environment_steps'], payload['episodes']
        if args.updates<update:raise ValueError('Training endpoint precedes the restored checkpoint')
        read_journal(out/'learning.jsonl',committed_update=update)
        warmstart_completed=payload.get('warmstart_completed',False)
        warmstart_steps=payload.get('warmstart_environment_steps',0)
    protocol = dict(arguments=vars(args), config=asdict(cfg), physical_protocol=asdict(physical), contract=contract, train_layouts=len(families),
        fixed_replay_episodes=len(failures), fixed_replay_layouts=len({x['task']['geometry_family_id'] for x in failures}),
        curriculum=dict(failure_fraction=args.failure_fraction,fresh_future_fraction=1-args.failure_fraction),
        learning='all collected episodes complete; exact terminal MonteCarlo returns,gamma1,PPO joint12-channel conditional Gaussian',
        execution='every0.2s; previous actual latent is actor and critic state; persistent noise and greedy evaluation share conditional mean',
        initialization=dict(checkpoint=args.initial,teacher_warmstart_episodes=args.warmstart_episodes,
                            teacher_used_online=cfg.reference_mode=='teacher',reference_mode=cfg.reference_mode),
        selection=dict(stages=args.stages,criterion='complete common-task success,collision,cappedtime,earlier'),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if not args.resume:
        atomic_json(protocol, out / 'protocol.json')
    started = time.time()
    retained_rows = {}
    if args.resume and (out / 'episodes.jsonl').exists():
        for row in read_journal(out/'episodes.jsonl'):
            if row['update'] > update:
                retained_rows[row['ordinal']] = row

    def checkpoint(stage):
        payload = dict(config=asdict(cfg), state=model.state_dict(), optimizer=optimizer.state_dict(),
            update=stage, environment_steps=steps, episodes=episodes, numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
            protocol=protocol,contract=contract,warmstart_completed=warmstart_completed,
            warmstart_environment_steps=warmstart_steps,
            teacher_sha256=teacher_hash, encoder_initialization_sha256=encoder_hash)
        atomic_save(payload, out / 'last.pt')
        if (stage == 0 and warmstart_completed) or stage in args.stages:
            atomic_save(payload, out / f'stage_{stage:06d}.pt')

    if not args.resume:
        checkpoint(0)
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context,
            initializer=initialize_worker, initargs=(vars(args), bank['protocol'], asdict(cfg))) as pool, \
            (out / 'learning.jsonl').open('a' if args.resume else 'w') as log, \
            (out / 'episodes.jsonl').open('a' if args.resume else 'w') as episode_log:
        if not warmstart_completed:
            model.eval()
            atomic_save(dict(config=asdict(cfg),state=model.state_dict()),out/'rollout_policy.pt')
            jobs=[]
            for ordinal in range(args.warmstart_episodes):
                base=bank['tasks'][int(rng.integers(len(bank['tasks'])))]
                task=dict(base,allocated_task_id=base['task_id'],future_seed=int(rng.integers(100000000,2000000000)),
                          task_id=f'warm{args.seed}-{ordinal}')
                jobs.append(dict(update=0,ordinal=ordinal,task=task,teacher_rollout=True,
                    sampling='teacher_initialization',policy_seed=int(rng.integers(1,2000000000))))
            warm_path=out/'warmstart_episodes.jsonl'
            saved={row['ordinal']:row for row in read_journal(warm_path)}
            warm_rows=[]
            with warm_path.open('a',encoding='utf-8') as stream:
                pending=[]
                for job in jobs:
                    row=saved.get(job['ordinal'])
                    if row is None:
                        pending.append(pool.submit(collect_episode,job))
                    elif row['task']==job['task'] and row['policy_seed']==job['policy_seed']:
                        warm_rows.append(row)
                    else:
                        raise ValueError('Teacher warmstart allocation changed during resume')
                for future in concurrent.futures.as_completed(pending):
                    row=future.result();warm_rows.append(row)
                    stream.write(json.dumps(row)+'\n');stream.flush()
            warm_rows.sort(key=lambda row:row['ordinal'])
            warm=complete_batch(warm_rows,out,physical.time_limit,physical.control_dt,args.device,cfg.history)
            warm_optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
            model.train()
            for _ in range(args.warmstart_updates):
                index=rng.choice(len(warm['returns']),size=min(args.minibatch,len(warm['returns'])),replace=False)
                output=model(warm['history'][index],warm['remaining'][index],warm['reference'][index],warm['previous'][index])
                force=model.force(warm['reference'][index],output['mean'])
                loss=((force-warm['force'][index])/750).square().mean()+.5*(output['value']-warm['returns'][index]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite teacher warmstart objective')
                warm_optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),.5);warm_optimizer.step()
            warmstart_steps=sum(row['steps'] for row in warm_rows)
            steps+=warmstart_steps;episodes+=len(warm_rows);warmstart_completed=True
            checkpoint(0)
            log.write(json.dumps(dict(phase='teacher_initialization',episodes=len(warm_rows),
                environment_steps=warmstart_steps,updates=args.warmstart_updates,
                teacher_used_online=cfg.reference_mode=='teacher',loss=float(loss.detach())))+'\n');log.flush()
            del warm,warm_optimizer
        for update in range(update+1, args.updates+1):
            model.eval()
            atomic_save(dict(config=asdict(cfg), state=model.state_dict()), out / 'rollout_policy.pt')
            jobs = []
            for index in range(args.episodes_per_update):
                replay = index < round(args.episodes_per_update*args.failure_fraction)
                if replay:
                    old_row = failures[int(rng.integers(len(failures)))]
                    base = old_row['task']
                    future = base['future_seed']
                    origin = dict(task_id=base['task_id'], update=old_row['update'])
                else:
                    base = bank['tasks'][int(rng.integers(len(bank['tasks'])))]
                    future = int(rng.integers(100000000, 2000000000))
                    origin = None
                ordinal = episodes+index
                task = dict(base, allocated_task_id=base['task_id'], future_seed=future,
                    task_id=f'episode{args.seed}-{ordinal}')
                jobs.append(dict(update=update, ordinal=ordinal, task=task, replay_source=origin,
                    sampling='training_failure_replay' if replay else 'training_pool_fresh_future',
                    policy_seed=int(rng.integers(1, 2000000000))))
            collection_started = time.time()
            rows = []
            pending = []
            for job in jobs:
                row = retained_rows.get(job['ordinal'])
                if row is None:
                    pending.append(pool.submit(collect_episode, job))
                elif row['task'] == job['task'] and row['policy_seed'] == job['policy_seed'] and row['update'] == update:
                    rows.append(row)
                else:
                    raise ValueError('Retained partial episode differs from the resumed allocation')
            for future in concurrent.futures.as_completed(pending):
                row = future.result()
                rows.append(row)
                episode_log.write(json.dumps(row)+'\n')
                episode_log.flush()
            rows.sort(key=lambda row: row['ordinal'])
            collection_seconds = time.time()-collection_started
            batch_steps = sum(row['steps'] for row in rows)
            steps += batch_steps
            episodes += len(rows)
            data = complete_batch(rows, out, physical.time_limit, physical.control_dt, args.device,cfg.history)
            detail = update_policy(model, optimizer, data, args, rng)
            sampling = {kind: dict(episodes=sum(x['sampling']==kind for x in rows),
                success=sum(x['sampling']==kind and x['success'] for x in rows))
                for kind in ['training_failure_replay', 'training_pool_fresh_future']}
            row = dict(update=update, total_updates=args.updates, environment_steps=steps,
                completed_episodes=episodes, batch_success=sum(x['success'] for x in rows),
                batch_collision=sum(x['collision'] for x in rows), batch_radial=sum(x['radial_only_failure'] for x in rows),
                batch_mean_return=float(np.mean([x['episode_return'] for x in rows])), sampling=sampling,
                absolute_physical_residual=float(np.mean([x['absolute_physical_residual'] for x in rows])),
                collection_steps_per_second=batch_steps/collection_seconds, elapsed_seconds=time.time()-started, **detail)
            log.write(json.dumps(row)+'\n')
            log.flush()
            checkpoint(update)
            atomic_json(dict(completed=False, **row), out / 'status.json')
            print(json.dumps(row), flush=True)
        summary = dict(completed=True, updates=args.updates, environment_steps=steps,
            completed_episodes=episodes, training_layouts=len(families), elapsed_seconds=time.time()-started,
            warmstart_environment_steps=warmstart_steps,reference_mode=cfg.reference_mode,
            teacher_used_online=cfg.reference_mode=='teacher',
            closed_loop_evaluation='declared stages require complete independent256 evaluation')
        atomic_json(summary, out / 'summary.json')
        atomic_json(summary, out / 'status.json')


if __name__ == '__main__':
    main()
