"""Train the deployed joint feedback policy on fresh complete trajectories.

Danger replay stores reachable simulator states, not stale actions or returns.
Each replay draws a new future and runs the CURRENT policy at every control tick.
The failure constraint is estimated only from complete, ordinary task starts;
danger continuations are a separately recorded recovery curriculum.
"""
import argparse
from collections import deque
import concurrent.futures
from dataclasses import asdict
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from citf.bridge import World, source_modules
from citf.contracts import Protocol, immutable_run_contract, require_same_contract, validate_task_banks
from citf.opponents import load_opponent
from citf.train import atomic_save
from experiments.joint_replanning_policy import (JointConfig, JointReplanningTransformer,
    batch_observations, decide, load_policy)


WORKER = {}


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def initialize_worker(arguments, physical):
    torch.set_num_threads(1)
    source_modules(str(Path(arguments['project']).resolve()))
    WORKER.update(arguments=arguments, physical=Protocol(**physical), model=None, version=None,
                  opponent=load_opponent(arguments.get('opponent_spec')))


def margins(world):
    """Observed separation and short constant-velocity encounter diagnostics."""
    state = world.state()
    n = len(state)-1
    i, j = np.triu_indices(n, 1)
    delta = state[i, :2]-state[j, :2]
    dv = state[i, 3:5]-state[j, 3:5]
    t = np.clip(-(delta*dv).sum(-1)/np.maximum((dv*dv).sum(-1), .01), 0, 2)
    current = np.linalg.norm(delta, axis=-1).min()-world.protocol.collision_radius
    predicted = np.linalg.norm(delta+t[:, None]*dv, axis=-1).min()-world.protocol.collision_radius
    enemy = state[-1]
    breach = np.linalg.norm(enemy[:2])-world.protocol.target_radius
    inward = min(0., np.dot(enemy[:2], enemy[3:5])/max(np.linalg.norm(enemy[:2]), 1e-6))
    return dict(pair=float(current), predicted_pair=float(predicted), breach=float(breach),
                predicted_breach=float(breach+2*inward))


def potential(world):
    state = world.state()
    # A telescoping potential supplies early learning signal without altering
    # the complete-task return. Terminal potential is always zero.
    return -float(np.linalg.norm(state[:-1, :2]-state[-1, :2], axis=-1).min())/40


def independent_pursuit(world):
    """Optional training-only motor initialization; no fleet/guard logic."""
    state = world.state()
    goal = state[-1, :2]+.6*state[-1, 3:5]
    return np.asarray([world.env.force_to_thrust(.1*(goal-boat[:2]), boat[2], robot='def')
                       for boat in state[:-1]], dtype=np.float32).clip(-500, 1000)


def remember_predecessors(recent, chosen, now, severity, reason):
    # Include time to change course, not just the last unavoidable collision tick.
    for seconds in (2., 4., 8.):
        eligible = [state for state in recent if state['elapsed'] <= now-seconds+1e-7]
        if not eligible:
            continue
        entry = eligible[-1]
        key = entry['elapsed']
        if key not in chosen or chosen[key]['severity'] < severity:
            chosen[key] = dict(entry, severity=severity, reason=reason)


@torch.no_grad()
def collect(job):
    args = WORKER['arguments']
    if WORKER['version'] != job['version']:
        WORKER['model'] = load_policy(Path(args['out'])/'rollout.pt')
        WORKER['version'] = job['version']
    model = WORKER['model']
    torch.manual_seed(job['policy_seed'])
    task = job['task']
    validation = job['kind'] == 'validation'
    warmstart = job['kind'] == 'warmstart'
    if 'snapshot' in job:
        snapshot = job['snapshot']
        world = snapshot['world'].clone()
        previous = snapshot['previous_force'].copy()
        # Keep the complete physical state/history/time; replace only the
        # unobserved future random stream. No past outcome becomes a label.
        world.rng = np.random.RandomState(job['future_seed'])
        if world.done or asdict(world.protocol) != asdict(WORKER['physical']):
            raise ValueError('Danger continuation has a terminal or incompatible world')
    else:
        world = World(args['project'], WORKER['physical'], task['defenders'])
        world.reset(task['scene_seed'], job['future_seed'], task['agility'])
        if world.initial_identity['fingerprint'] != task['geometry_family_id']:
            raise ValueError('Task layout differs from its recorded geometry')
        previous = np.zeros((task['defenders'], 2), np.float32)
    start_time = float(world.env.Current_T)
    stride = max(1, round(1/world.protocol.control_dt))
    recent = deque(maxlen=math.ceil(10/world.protocol.control_dt/stride)+1)
    chosen, decisions, rewards, costs, targets = {}, [], [], [], []
    step, compute_seconds, min_margin = 0, 0., float('inf')
    start = time.perf_counter()
    while not world.done:
        now = float(world.env.Current_T)
        if not validation:
            if step % stride == 0:
                recent.append(dict(world=world.clone(), previous_force=previous.copy(), task=task,
                                   elapsed=now, update=job['update'], origin=job['ordinal']))
            observed = margins(world)
            gap = min(observed.values())
            min_margin = min(min_margin, gap)
            if gap < 4.:
                remember_predecessors(recent, chosen, now, 1+max(0., 4-gap), 'near_encounter')
        before = potential(world)
        tick_start = time.perf_counter()
        force, decision = decide(model, world, previous, stochastic=not validation and not warmstart)
        compute_seconds += time.perf_counter()-tick_start
        if warmstart:
            force = independent_pursuit(world)
            targets.append(torch.from_numpy(force.copy()))
        result = world.step(force, attacker_action=WORKER['opponent'].act(world))
        after = 0. if world.done else potential(world)
        terminal = (4. if result['capture'] else 1. if result['timeout'] else 0.) if world.done else 0.
        reward = terminal-.2*world.protocol.control_dt/world.protocol.time_limit+after-before
        failure = bool(world.done and not result['success'])
        if not validation:
            decisions.append(decision)
            rewards.append(reward)
            costs.append(float(failure))
        previous = force.copy()
        step += 1
    if not validation and not result['success']:
        remember_predecessors(recent, chosen, float(world.env.Current_T), 20.,
                              'collision' if result['collision'] else 'breach')
    snapshots = sorted(chosen.values(), key=lambda x: (-x['severity'], x['elapsed']))[:6]
    for entry in snapshots:
        entry.update(key=f"{job['update']}:{job['ordinal']}:{entry['elapsed']:.6f}", visits=0)
    row = dict(update=job['update'], ordinal=job['ordinal'], kind=job['kind'], task=task,
               future_seed=job['future_seed'], policy_seed=job['policy_seed'], steps=step,
               start_time=start_time, snapshot_key=job.get('snapshot', {}).get('key'),
               capped_capture_time=result['duration'] if result['capture'] else world.protocol.time_limit,
               minimum_predicted_margin=None if validation else min_margin,
               mean_control_ms=1000*compute_seconds/step, seconds=time.perf_counter()-start, **result)
    return dict(decisions=decisions, rewards=rewards, costs=costs, targets=targets,
                snapshots=snapshots, row=row)


def advantages(rewards, values, lam):
    result = np.zeros(len(rewards), np.float32)
    carry, next_value = 0., 0.
    for i in range(len(rewards)-1, -1, -1):
        carry = rewards[i]+next_value-values[i]+lam*carry
        result[i] = carry
        next_value = values[i]
    return result


def training_batch(episodes, device, lam, multiplier):
    observed, latent, logp, values, value_targets, cost_targets, advantage = [], [], [], [], [], [], []
    for episode in episodes:
        rows = episode['decisions']
        value = np.asarray([x['value'] for x in rows], np.float32)
        cost_value = np.asarray([x['cost_value'] for x in rows], np.float32)
        reward_advantage = advantages(episode['rewards'], value, lam)
        cost_advantage = advantages(episode['costs'], cost_value, lam)
        observed.extend(x['observation'] for x in rows)
        latent.extend(x['latent'] for x in rows)
        logp.extend(x['logp'] for x in rows)
        values.extend(value)
        value_targets.extend(reward_advantage+value)
        cost_targets.extend(np.cumsum(np.asarray(episode['costs'])[::-1])[::-1].copy())
        advantage.extend(reward_advantage-multiplier*cost_advantage)
    data = batch_observations(observed, device)
    data.update(latent=torch.stack(latent).to(device),
                **{key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in
                   dict(old_logp=logp, old_value=values, value_target=value_targets,
                        cost_target=cost_targets, advantage=advantage).items()})
    data['advantage'] = (data['advantage']-data['advantage'].mean())/data['advantage'].std(unbiased=False).clamp_min(1e-6)
    return data


def optimize(model, optimizer, data, args, rng):
    model.train()
    records, stopped = [], False
    for _ in range(args.epochs):
        order = rng.permutation(len(data['latent']))
        for start in range(0, len(order), args.minibatch):
            index = torch.as_tensor(order[start:start+args.minibatch], device=args.device)
            batch = {key: value[index] for key, value in data.items()}
            output = model(batch)
            distribution = output['distribution']
            delta = distribution.log_prob(batch['latent'])-batch['old_logp']
            ratio = delta.clamp(-20, 20).exp()
            kl = float(((ratio-1)-delta).mean().detach())
            if kl > args.target_kl:
                stopped = True
                break
            policy = -torch.minimum(ratio*batch['advantage'],
                ratio.clamp(1-args.clip, 1+args.clip)*batch['advantage']).mean()
            clipped = batch['old_value']+(output['value']-batch['old_value']).clamp(-.2, .2)
            value_loss = .5*torch.maximum((output['value']-batch['value_target']).square(),
                                         (clipped-batch['value_target']).square()).mean()
            cost_loss = .5*(output['cost_value']-batch['cost_target']).square().mean()
            # Entropy of normalized tanh actions, including its stable Jacobian.
            sample = distribution.rsample()
            jacobian = (2*(math.log(2)-sample-F.softplus(-2*sample))).sum(-1)
            entropy = (-distribution.log_prob(sample)+jacobian).mean()
            loss = policy+.5*(value_loss+cost_loss)-args.entropy*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite joint on-policy objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            records.append(dict(policy=float(policy.detach()), value=float(value_loss.detach()),
                                cost_value=float(cost_loss.detach()), entropy=float(entropy.detach()),
                                kl=kl, gradient=float(norm)))
        if stopped:
            break
    return dict(minibatches=len(records), kl_stopped=stopped,
                **({key: float(np.mean([r[key] for r in records])) for key in records[0]} if records else {}))


def warmstart(model, optimizer, episodes, args, rng):
    # Supervised initialization is finished before any PPO collection. Its
    # actions/likelihoods never enter the on-policy policy-gradient batch.
    observed = [row['observation'] for ep in episodes for row in ep['decisions']]
    data = batch_observations(observed, args.device)
    targets = torch.stack([target for ep in episodes for target in ep['targets']]).to(args.device)
    model.train()
    for _ in range(args.warmstart_updates):
        index = torch.as_tensor(rng.integers(len(targets), size=min(args.minibatch, len(targets))), device=args.device)
        output = model({key: value[index] for key, value in data.items()})
        loss = F.smooth_l1_loss((model.force(output['mean'])-250)/750, (targets[index]-250)/750)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


def merge_pool(pool, episodes, capacity):
    by_key = {entry['key']: entry for entry in pool}
    for episode in episodes:
        for entry in episode['snapshots']:
            by_key[entry['key']] = entry
    ordered = sorted(by_key.values(), key=lambda x: (x['severity']/math.sqrt(1+x['visits']), x['update']), reverse=True)
    kept, per_layout = [], {}
    for entry in ordered:
        family = entry['task']['geometry_family_id']
        if per_layout.get(family, 0) >= max(6, capacity//8):
            continue
        kept.append(entry)
        per_layout[family] = per_layout.get(family, 0)+1
        if len(kept) == capacity:
            break
    return kept


def summarize(rows):
    count = len(rows)
    return dict(N=count, **{key: sum(row[key] for row in rows) for key in
                ('success', 'capture', 'collision', 'physical_collision', 'physical_breach', 'timeout')},
                capped_capture_time=float(np.mean([row['capped_capture_time'] for row in rows])))


def main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'tasks', 'validation-tasks', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--initial')
    parser.add_argument('--opponent-spec')
    parser.add_argument('--seed', type=int, default=10001)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--updates', type=int, default=64)
    parser.add_argument('--episodes-per-update', type=int, default=24)
    parser.add_argument('--validation-every', type=int, default=4)
    parser.add_argument('--danger-fraction', type=float, default=.35)
    parser.add_argument('--danger-capacity', type=int, default=256)
    parser.add_argument('--failure-budget', type=float, default=.01)
    parser.add_argument('--dual-lr', type=float, default=5.)
    parser.add_argument('--initial-dual', type=float, default=1.)
    parser.add_argument('--warmstart-episodes', type=int)
    parser.add_argument('--warmstart-updates', type=int, default=200)
    parser.add_argument('--without-joint-context', action='store_true')
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--exploration-rank', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--minibatch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gae-lambda', type=float, default=.97)
    parser.add_argument('--clip', type=float, default=.2)
    parser.add_argument('--target-kl', type=float, default=.02)
    parser.add_argument('--entropy', type=float, default=.001)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-file')
    args = parser.parse_args()
    if args.warmstart_episodes is None:
        args.warmstart_episodes = 0 if args.initial else 16
    if min(args.workers, args.updates, args.episodes_per_update, args.validation_every,
           args.danger_capacity, args.epochs, args.minibatch) < 1:
        parser.error('Counts must be positive')
    if (not 0 <= args.danger_fraction < 1 or not 0 <= args.failure_budget < 1
            or not 0 <= args.gae_lambda <= 1 or not 0 < args.clip < 1
            or min(args.lr, args.target_kl) <= 0 or min(args.dual_lr, args.initial_dual, args.entropy) < 0
            or min(args.warmstart_episodes, args.warmstart_updates) < 0):
        parser.error('Invalid optimization or curriculum setting')
    out = Path(args.out).resolve()
    args.out = str(out)
    args.project = str(Path(args.project).resolve())
    train = json.loads(Path(args.tasks).read_text(encoding='utf-8'))
    validation = json.loads(Path(args.validation_tasks).read_text(encoding='utf-8'))
    validate_task_banks(('train', train), ('validation', validation))
    sizes = {task['defenders'] for bank in (train, validation) for task in bank['tasks']}
    if len(sizes) != 1 or min(sizes) < 2:
        raise ValueError('One fleet size per training run is required; the actor supports arbitrary fleet sizes')
    physical = asdict(Protocol(**train['protocol']))
    cfg = JointConfig(history=physical['history_steps'], width=args.width, heads=args.heads,
                      joint_context=not args.without_joint_context, exploration_rank=args.exploration_rank)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # Register the archived environment classes before restoring curriculum worlds.
    source_modules(args.project)
    model = JointReplanningTransformer(cfg).to(args.device)
    model.physical_protocol = physical
    opponent = load_opponent(args.opponent_spec)
    files = [args.tasks, args.validation_tasks]
    if args.initial:
        files.append(args.initial)
    if args.opponent_spec:
        files.append(args.opponent_spec)
    parameters = {key: value for key, value in vars(args).items()
                  if key not in ('out', 'resume', 'workers', 'device', 'stop_file')}
    contract = immutable_run_contract(dict(arguments=parameters, config=asdict(cfg), physical_protocol=physical,
        opponent=opponent.identity), files=files, code=[__file__, Path(__file__).with_name('joint_replanning_policy.py')])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    update, steps, warm_steps, pool, history, best = 0, 0, 0, [], [], None
    multiplier = args.initial_dual
    if args.resume:
        saved = torch.load(out/'latest.pt', map_location='cpu', weights_only=False)
        require_same_contract(saved['contract'], contract, 'Joint continuous training resume')
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != 'step':
                    state[key] = value.to(args.device)
        update, steps, warm_steps = saved['update'], saved['environment_steps'], saved['warmstart_environment_steps']
        pool, history, best, multiplier = saved['danger_pool'], saved['history'], saved['best'], saved['multiplier']
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if args.device.startswith('cuda') and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else:
        out.mkdir(parents=True, exist_ok=False)
        if args.initial:
            initial = load_policy(args.initial)
            initial.validate_runtime(None, World(args.project, Protocol(**physical), min(sizes)))
            if asdict(initial.cfg) != asdict(cfg):
                raise ValueError('Initial joint policy architecture/configuration differs')
            model.load_state_dict(initial.state_dict())
        atomic_json(dict(contract=contract, rollout='current policy redecides every physical tick',
            danger='reachable training states; new future and fresh current-policy actions/returns',
            warmstart='independent motor tracking only; never an online reference',
            evaluation='complete unchanged validation bank from original starts'), out/'protocol.json')

    def policy_payload():
        return dict(config=asdict(cfg), state={key: value.detach().cpu() for key, value in model.state_dict().items()},
                    physical_protocol=physical, update=update, environment_steps=steps,
                    warmstart_environment_steps=warm_steps, opponent=opponent.identity)

    def save_checkpoint():
        payload = policy_payload()
        atomic_save(dict(payload, optimizer=optimizer.state_dict(), contract=contract,
            danger_pool=pool, multiplier=multiplier, history=history, best=best,
            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else None), out/'latest.pt')
        atomic_save(payload, out/'policy.pt')
        atomic_json(dict(completed=update >= args.updates, update=update, environment_steps=steps,
                         warmstart_environment_steps=warm_steps, danger_states=len(pool),
                         failure_multiplier=multiplier, best=best, latest=history[-1] if history else None), out/'status.json')

    def jobs_for(kind, count, version):
        jobs = []
        for ordinal in range(count):
            task = validation['tasks'][ordinal] if kind == 'validation' else train['tasks'][int(rng.integers(len(train['tasks'])))]
            job = dict(kind=kind, task=task, ordinal=ordinal, update=update, version=version,
                       policy_seed=int(rng.integers(2**31-1)),
                       future_seed=task['future_seed'] if kind == 'validation' else int(rng.integers(2**31-1)))
            if kind == 'on_policy' and pool and ordinal < min(count-1, int(count*args.danger_fraction)):
                weights = np.asarray([p['severity']/math.sqrt(1+p['visits']) for p in pool])
                selected = pool[int(rng.choice(len(pool), p=weights/weights.sum()))]
                selected['visits'] += 1
                job.update(kind='danger', snapshot=selected, task=selected['task'])
            jobs.append(job)
        return jobs

    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context,
            initializer=initialize_worker, initargs=(vars(args), physical)) as executor:
        if not args.resume:
            atomic_save(policy_payload(), out/'rollout.pt')
            if args.warmstart_episodes:
                episodes = list(executor.map(collect, jobs_for('warmstart', args.warmstart_episodes, 'warmstart')))
                warmstart(model, optimizer, episodes, args, rng)
                warm_steps = sum(ep['row']['steps'] for ep in episodes)
                pool = merge_pool(pool, episodes, args.danger_capacity)
                atomic_json(dict(episodes=[ep['row'] for ep in episodes], environment_steps=warm_steps,
                                 updates=args.warmstart_updates), out/'warmstart.json')
                # Motor imitation does not carry optimizer moments into policy learning.
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            save_checkpoint()
        while update < args.updates:
            if args.stop_file and Path(args.stop_file).exists():
                break
            started = time.perf_counter()
            atomic_save(policy_payload(), out/'rollout.pt')
            episodes = list(executor.map(collect, jobs_for('on_policy', args.episodes_per_update, f'train-{update}')))
            ordinary = [ep['row'] for ep in episodes if ep['row']['kind'] == 'on_policy']
            failure_rate = float(np.mean([1-row['success'] for row in ordinary]))
            multiplier = max(0., multiplier+args.dual_lr*(failure_rate-args.failure_budget))
            data = training_batch(episodes, args.device, args.gae_lambda, multiplier)
            metrics = optimize(model, optimizer, data, args, rng)
            steps += sum(ep['row']['steps'] for ep in episodes)
            pool = merge_pool(pool, episodes, args.danger_capacity)
            update += 1
            row = dict(update=update, ordinary=summarize(ordinary),
                danger=summarize([ep['row'] for ep in episodes if ep['row']['kind'] == 'danger'])
                       if any(ep['row']['kind'] == 'danger' for ep in episodes) else None,
                failure_multiplier=multiplier, optimization=metrics,
                episodes=[ep['row'] for ep in episodes])
            if update % args.validation_every == 0 or update == args.updates:
                atomic_save(policy_payload(), out/'rollout.pt')
                evaluated = list(executor.map(collect, jobs_for('validation', len(validation['tasks']), f'validation-{update}')))
                row['validation'] = summarize([ep['row'] for ep in evaluated])
                row['validation_episodes'] = [ep['row'] for ep in evaluated]
                report = row['validation']
                score = (report['success'], report['capture'], -report['capped_capture_time'])
                if best is None or score > tuple(best['score']):
                    best = dict(update=update, score=list(score), metrics=report)
                    atomic_save(policy_payload(), out/'best.pt')
            row['seconds'] = time.perf_counter()-started
            history.append(row)
            save_checkpoint()
            print(json.dumps({key: value for key, value in row.items()
                              if key not in ('episodes', 'validation_episodes')}, ensure_ascii=False), flush=True)
    save_checkpoint()


if __name__ == '__main__':
    main()
