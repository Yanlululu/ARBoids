"""Budget-matched, restartable experiments. No validation-driven model selection."""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from envs.TADgame import TADEnv
from policy.SAC import SAC, ReplayBuffer
from policy.mappo import PredictiveMAPPO
from parallel_mappo import ParallelRollouts
from utils.config import _dict_to_namespace
from utils.protocol import apply_adapter_exploration

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('arboids', 'no_prediction', 'fixed_rule', 'no_joint', 'fixed_gain', 'full')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state():
    n = np.random.get_state()
    state = dict(python=random.getstate(), numpy_kind=n[0], numpy_keys=torch.from_numpy(n[1].astype(np.int64)),
                 numpy_pos=n[2], numpy_gauss=n[3], numpy_cached=n[4], torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state((state['numpy_kind'], state['numpy_keys'].numpy().astype(np.uint32),
                         state['numpy_pos'], state['numpy_gauss'], state['numpy_cached']))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_sac(path, agent, replay, config, counters):
    """Full continuation state, saved only at complete episode boundaries."""
    if any(not torch.isfinite(p).all() for network in (agent.actor, agent.critic, agent.critic_target)
           for p in network.parameters()) or not torch.isfinite(agent.log_alpha).all():
        raise FloatingPointError('Non-finite SAC parameters; preserve the last valid checkpoint.')
    state = dict(format='arboids-full-v1', config=config, counters=counters, rng=rng_state(),
                 actor=agent.actor.state_dict(), critic=agent.critic.state_dict(),
                 critic_target=agent.critic_target.state_dict(),
                 actor_optimizer=agent.actor_optimizer.state_dict(),
                 critic_optimizer=agent.critic_optimizer.state_dict(),
                 alpha_optimizer=agent.alpha_optimizer.state_dict(), log_alpha=agent.log_alpha.detach(),
                 replay=dict(count=replay.count, size=replay.size, max_size=replay.max_size,
                             arrays={k: torch.from_numpy(getattr(replay, k)[:replay.size])
                                     for k in ('s', 'a', 'r', 's_', 'dw')}))
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_sac(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 'arboids-full-v1':
        raise ValueError('Actor-only weights cannot resume a SAC training experiment.')
    cfg = _dict_to_namespace(state['config'])
    env = TADEnv(cfg.agent.defender_num, True, True, **state['config']['environment'])
    agent = SAC(cfg, env.feature1_dim, env.feature2_dim, 3, adaptive=True, device=torch.device(device))
    for key in ('actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer'):
        getattr(agent, key).load_state_dict(state[key])
    with torch.no_grad():
        agent.log_alpha.copy_(state['log_alpha'])
    agent.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
    agent.alpha = agent.log_alpha.exp().to(agent.device)
    replay = ReplayBuffer(env.state_dim, 3)
    rs = state['replay']
    replay.count, replay.size, replay.max_size = rs['count'], rs['size'], rs['max_size']
    for key, values in rs['arrays'].items():
        target = np.zeros((replay.max_size, *values.shape[1:]), dtype=values.numpy().dtype)
        target[:replay.size] = values.numpy()
        setattr(replay, key, target)
    restore_rng(state['rng'])
    return agent, replay, state['config'], state['counters']


def append_row(path, row):
    with Path(path).open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def finish(directory, metadata, checkpoint):
    metadata.update(completed=True, checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=digest(checkpoint))
    atomic_json(directory / 'completed.json', metadata)


def sac_run(args):
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / 'resume.pth'
    if (directory / 'completed.json').exists():
        raise ValueError('Completed training must not be silently repeated.')
    initial = latest if latest.exists() else args.source
    if initial:
        agent, replay, config, counters = load_sac(initial, args.device)
        if counters['training_seed'] != args.seed:
            raise ValueError('Seed differs from continuation checkpoint.')
        if not latest.exists():
            counters = dict(counters, phase_start=counters['steps'], phase_episodes=0, phase='continuation')
    else:
        config = yaml.safe_load((ROOT/'train/configs/paper-parameters.yaml').read_text(encoding='utf-8'))
        seed_all(args.seed)
        cfg = _dict_to_namespace(config)
        env = TADEnv(3, True, True, **config['environment'])
        agent = SAC(cfg, env.feature1_dim, env.feature2_dim, 3, adaptive=True, device=torch.device(args.device))
        replay = ReplayBuffer(env.state_dim, 3)
        counters = dict(steps=0, phase_start=0, episodes=0, phase_episodes=0, phase='pretrain', training_seed=args.seed)
    cfg = _dict_to_namespace(config)
    atomic_json(directory/'config.json', dict(config=config, seed=args.seed, budget=args.steps,
                source_sha256=digest(args.source) if args.source else None, source=str(args.source)))
    env = TADEnv(3, True, True, **config['environment'])
    end = counters['phase_start'] + args.steps
    started, start_step = time.perf_counter(), counters['steps']
    prior_elapsed = counters.get('elapsed_seconds', 0.)
    next_save = counters['steps'] + args.save_interval
    next_log = counters['steps'] + 1000
    while counters['steps'] < end:
        phase_step = counters['steps'] - counters['phase_start']
        agility = 2. + min(3, int(phase_step // 250000)) * .25 if counters['phase'] == 'pretrain' else 2.
        # A shared reset stream across arms; policy/replay randomness remains independent.
        episode_seed = args.scene_seed + counters['phase_episodes']
        np.random.seed(episode_seed)
        observation, _ = env.reset(agility, noisy_agility=True)
        done = 0
        while not done:
            if counters['steps'] < config['training']['warm_steps']:
                action = np.random.uniform(-1., 1., (3, 3))
                action[:, 2] = .5 * action[:, 2] + .5
            else:
                action = agent.choose_action(observation, False).reshape(3, 3)
                action = apply_adapter_exploration(action, cfg.training)
            if not np.isfinite(action).all():
                raise FloatingPointError('Non-finite training action.')
            following, rewards, done, _ = env.step(action, 'AdaRes')
            for i in range(3):
                replay.store(observation[i], action[i], rewards[i], following[i], bool(done))
            observation = following
            counters['steps'] += 1
            if counters['steps'] >= config['training']['warm_steps']:
                agent.learn(replay)
            if counters['steps'] >= next_log:
                elapsed = time.perf_counter() - started
                append_row(directory/'progress.jsonl', dict(steps=counters['steps'], target=end,
                    elapsed_seconds=elapsed, steps_per_second=(counters['steps']-start_step)/max(elapsed, 1e-9),
                    phase=counters['phase'], training_seed=args.seed))
                next_log = counters['steps'] + 1000
            if getattr(args, 'exact_steps', False) and counters['steps'] >= end:
                break
        counters['episodes'] += 1
        counters['phase_episodes'] += 1
        if counters['steps'] >= next_save or counters['steps'] >= end:
            counters['elapsed_seconds'] = prior_elapsed + time.perf_counter() - started
            save_sac(latest, agent, replay, config, counters)
            next_save = counters['steps'] + args.save_interval
    actor_file = directory/'actor.pth'
    torch.save(agent.actor.state_dict(), actor_file)
    finish(directory, dict(**counters, nominal_phase_steps=args.steps,
                           actual_phase_steps=counters['steps']-counters['phase_start'],
                           overshoot=counters['steps']-end, seed=args.seed,
                           resume_sha256=digest(latest)), actor_file)


def mappo_config(arm):
    config = yaml.safe_load((ROOT/'train/configs/formal-mappo.yaml').read_text(encoding='utf-8'))
    s = config['mappo']
    if arm == 'no_prediction':
        s.update(prediction_disabled=True, compatibility_prior=False, compatibility_peer_intent=False,
                 compatibility_context_gain=False, compatibility_mixture_points=0,
                 compatibility_joint_mixture=False)
        config['prediction']['mixture_points'] = 0
    elif arm == 'no_joint':
        s['compatibility_joint_mixture'] = False
    elif arm in ('fixed_gain', 'fixed_rule'):
        s.update(compatibility_context_gain=False, fixed_compatibility_gain=True)
    elif arm != 'full':
        raise ValueError(f'Unknown MAPPO arm: {arm}')
    return config


def initialize_matched_agent(config, baseline, seed, device='cpu'):
    """Keep every shared initial tensor identical despite absent ablation modules."""
    seed_all(seed)
    canonical_config=copy.deepcopy(config)
    canonical_config['mappo'].update(prediction_disabled=False, compatibility_prior=True,
        compatibility_peer_intent=True, compatibility_context_gain=True, fixed_compatibility_gain=False,
        compatibility_mixture_points=9, compatibility_joint_mixture=True)
    canonical_config['prediction']['mixture_points']=9
    canonical=PredictiveMAPPO(canonical_config,device)
    canonical.actor.initialize_from_baseline(baseline,proposal_std=.05,gate_std=.1)
    agent=PredictiveMAPPO(config,device)
    shared=canonical.actor.state_dict()
    agent.actor.load_state_dict({key:shared[key] for key in agent.actor.state_dict()})
    agent.critic.load_state_dict(canonical.critic.state_dict())
    # Constructor draw counts must not change the optimizer's minibatch RNG stream.
    seed_all(seed+1000000)
    return agent


def mappo_run(args):
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory/'policy.pth'
    if (directory/'completed.json').exists():
        raise ValueError('Completed training must not be silently repeated.')
    config = mappo_config(args.arm)
    if latest.exists():
        agent = PredictiveMAPPO.from_checkpoint(latest, args.device, restore_rng=True)
        if agent.config != config or agent.training_state.get('formal_seed') != args.seed:
            raise ValueError('Formal recipe or seed changed during resume.')
    else:
        agent = initialize_matched_agent(config,torch.load(args.source,map_location='cpu',weights_only=True),
                                         args.seed,args.device)
        agent.training_state.update(formal_seed=args.seed, next_episode=0,
                                    baseline_sha256=digest(args.source), formal_arm=args.arm)
    atomic_json(directory/'config.json', dict(config=config, seed=args.seed, budget=args.steps,
                                            source_sha256=digest(args.source), arm=args.arm))
    if args.arm == 'fixed_rule':
        # The rule is applied to the equal-budget SAC actor, not to a more-trained policy.
        agent.save(latest)
        finish(directory, dict(seed=args.seed, arm=args.arm, optimizer_updates=0,
                               rule_training_steps=0, inherited_actor_sha256=digest(args.source)), latest)
        return
    started = time.perf_counter()
    start_step = agent.total_steps
    with ParallelRollouts(config, args.workers, environments_per_worker=4) as pool:
        while agent.total_steps < args.steps:
            # With at most 300 steps/episode, all batches except the final single episode
            # remain below the nominal budget. Final excess is at most 299 steps.
            remaining = args.steps - agent.total_steps
            count = min(config['training']['episodes_per_update'], max(1, remaining//300))
            first = agent.training_state['next_episode']
            rollout = pool.run(agent, range(args.scene_seed+first, args.scene_seed+first+count),
                               agility=2., noisy=True)
            metrics = agent.update(rollout)
            agent.training_state['next_episode'] = first + count
            elapsed = time.perf_counter() - started
            metrics.update(arm=args.arm, seed=args.seed, elapsed_seconds=elapsed,
                           steps_per_second=(agent.total_steps-start_step)/max(elapsed, 1e-9))
            append_row(directory/'progress.jsonl', metrics)
            temporary = latest.with_suffix('.tmp')
            agent.save(temporary)
            temporary.replace(latest)
    finish(directory, dict(seed=args.seed, arm=args.arm, nominal_phase_steps=args.steps,
                           actual_phase_steps=agent.total_steps, overshoot=agent.total_steps-args.steps,
                           updates=agent.updates), latest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('pretrain',)+ARMS, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scene-seed', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--save-interval', type=int, default=25000)
    parser.add_argument('--exact-steps', action='store_true', help='Stop at the exact requested final step.')
    args = parser.parse_args()
    if min(args.steps, args.workers, args.save_interval) <= 0:
        parser.error('Budgets, workers and checkpoint intervals must be positive.')
    if args.arm != 'pretrain' and not args.source:
        parser.error('Continuation and comparison arms require their paired source.')
    torch.set_num_threads(1)
    if args.arm in ('pretrain', 'arboids'):
        sac_run(args)
    else:
        mappo_run(args)


if __name__ == '__main__':
    main()
