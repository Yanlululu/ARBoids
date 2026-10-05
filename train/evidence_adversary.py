"""Train a shared attacker panel and evaluate frozen defenders without adaptation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from envs.TADgame import TADEnv
from evidence_eval import ENVIRONMENT, PhysicalPolicy, sha256
from evidence_matrix import EXCLUDED
from policy.SAC import SAC, ReplayBuffer
from policy.networks import ActorAtt
from RL.collector import CollisionStarts
from RL.guidance import preserve_rng
from train_mappo import minimum_spacing, summarize
from utils.config import load_config
from utils.manager import set_seed


ATTACKER_SEEDS = (1709, 2711, 3719)


def load_attacker(path):
    weights = torch.load(path, map_location='cpu', weights_only=True)
    actor = ActorAtt(2, 6, 2, weights['l1.weight'].shape[0]).eval()
    actor.load_state_dict(weights)
    return actor


@torch.no_grad()
def evaluate_duel(defender, attacker, seeds, parallel=8, duration=60.):
    """Identical initial-state and RNG conventions to the primary evaluator."""
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or parallel < 1:
        raise ValueError('Unique nonempty seeds and a positive batch size are required')
    pending, rows, cursor = [], [], 0
    start = time.perf_counter()
    with preserve_rng():
        while cursor < len(seeds) or pending:
            while cursor < len(seeds) and len(pending) < parallel:
                seed = int(seeds[cursor])
                np.random.seed(seed)
                env = TADEnv(3, LearningSide='Att', **dict(ENVIRONMENT, total_time=duration))
                obs, att_obs = env.reset(2., noisy_agility=False)
                initial = np.r_[env.attacker.pos, env.attacker.theta,
                                np.asarray([d.pos for d in env.defender_list]).ravel(),
                                [d.theta for d in env.defender_list]]
                pending.append(dict(env=env, obs=obs, att_obs=att_obs, rng=np.random.get_state(),
                                    seed=seed, reward=0., attacker_reward=0., steps=0,
                                    spacing=minimum_spacing(env), energy=0.,
                                    initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest()))
                cursor += 1
            frames = [p['env'].structured_frame(defender.motion_features) for p in pending]
            commands = defender.act(frames, [p['obs'] for p in pending])
            att_commands = (attacker(torch.as_tensor(np.asarray([p['att_obs'] for p in pending]),
                                       dtype=torch.float32), True, False)[0].numpy()
                            if attacker is not None else [None] * len(pending))
            remaining = []
            for i, item in enumerate(pending):
                env = item['env']
                np.random.set_state(item['rng'])
                obs, att_reward, outcome, info = env.step_thrust(commands[i], att_commands[i])
                if not np.isfinite(att_reward) or not np.isfinite(commands[i]).all():
                    raise FloatingPointError('Non-finite adversarial evaluation')
                item.update(obs=obs, att_obs=info['attacker_obs'], rng=np.random.get_state())
                item['steps'] += 1
                item['attacker_reward'] += float(att_reward)
                item['reward'] += float(env._get_paper_rewards().mean())
                item['spacing'] = min(item['spacing'], minimum_spacing(env))
                item['energy'] += float(np.mean(commands[i] ** 2)) * env.Action_T
                if item['steps'] > int(np.ceil(duration / env.Action_T)) + 1:
                    raise RuntimeError('Adversarial evaluation exceeded the episode horizon')
                if not outcome:
                    remaining.append(item)
                    continue
                rows.append(dict(seed=item['seed'], outcome_code=int(outcome), success=int(outcome in (3, 4)),
                                 steps=item['steps'], time_seconds=env.Current_T, reward=item['reward'],
                                 attacker_reward=item['attacker_reward'], min_spacing=item['spacing'],
                                 thrust_squared_integral=item['energy'], initial_sha256=item['initial_sha256']))
            pending = remaining
    rows.sort(key=lambda r: r['seed'])
    return dict(passed=True, rows=rows, summary=summarize(rows), defender=defender.metadata(),
                environment=dict(ENVIRONMENT, total_time=duration), wall_seconds=time.perf_counter() - start,
                attacker_return_mean=float(np.mean([r['attacker_reward'] for r in rows])),
                scope='Frozen defenders; common learned attacker, or APF when no attacker checkpoint is supplied')


def train_attacker(seed, root, device, steps=500000):
    directory = root / 'results/attackers' / f'seed{seed}'
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    torch.set_num_threads(1)
    set_seed(seed)
    cfg = load_config(Path(__file__).parent / 'configs/paper-parameters.yaml')
    attacker = SAC(cfg, 2, 6, 2, attacker=True, device=torch.device(device))
    buffer = ReplayBuffer(8, 2)
    policies = [PhysicalPolicy('channel', root / 'inputs/channel-frozen.pth'),
                PhysicalPolicy('arboids', root / 'inputs/arboids-reference.pth')]
    selection_rng = np.random.RandomState(seed + 771991)
    config = dict(training_seed=seed, total_steps=steps, warm_steps=cfg.training.warm_steps,
                  defender_mixture=[p.metadata() for p in policies], probabilities=[.5, .5],
                  environment=ENVIRONMENT, development_seeds=[630000, 630064], eval_interval=100000,
                  checkpoint_selection='Maximize mean breach rate, then attacker return, across both defenders',
                  update='One paper-preset SAC gradient update per environment step after warmup',
                  excluded_seed_ranges=EXCLUDED, source_sha256=sha256(__file__))
    (directory / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    attacker.save(directory / 'initial.pth')
    def reset():
        env = TADEnv(3, LearningSide='Att', **ENVIRONMENT)
        env.reset(2., noisy_agility=True)
        return env
    starts = CollisionStarts(reset, excluded_seed_ranges=EXCLUDED)
    env, _, _ = starts.reset()
    observation, att_observation = env._get_obs()
    index = int(selection_rng.choice(2))
    counts, episode_counts = [0, 0], [0, 0]
    metrics, best_score, selected_step = [], None, 0
    elapsed = time.perf_counter()
    for step in range(1, steps + 1):
        command = policies[index].act([env.structured_frame(policies[index].motion_features)], [observation])[0]
        att_action = (np.random.uniform(-1., 1., 2) if step <= cfg.training.warm_steps
                      else attacker.choose_action(att_observation, False))
        next_observation, reward, outcome, info = env.step_thrust(command, att_action)
        next_att = info['attacker_obs']
        if not np.isfinite(reward) or not np.isfinite(att_action).all():
            raise FloatingPointError('Non-finite attacker training sample')
        buffer.store(att_observation, att_action, reward, next_att, bool(outcome))
        observation, att_observation = next_observation, next_att
        counts[index] += 1
        if step >= cfg.training.warm_steps:
            attacker.learn(buffer)
        if outcome:
            episode_counts[index] += 1
            env, _, _ = starts.reset()
            observation, att_observation = env._get_obs()
            index = int(selection_rng.choice(2))
        if step % 10000 == 0:
            print(f'[ATTACKER] seed={seed} step={step}/{steps} wall={time.perf_counter()-elapsed:.1f}s', flush=True)
        if step % 100000 == 0 or step == steps:
            if not all(torch.isfinite(p).all().item() for p in attacker.actor.parameters()):
                raise FloatingPointError('Non-finite attacker checkpoint')
            attacker.save(directory / 'final.pth')
            with preserve_rng():
                model = load_attacker(directory / 'final.pth')
                result = [evaluate_duel(p, model, range(630000, 630064)) for p in policies]
            score = (float(np.mean([r['summary']['breaches'] / 64 for r in result])),
                     float(np.mean([r['attacker_return_mean'] for r in result])))
            if best_score is None or score > best_score:
                best_score, selected_step = score, step
                attacker.save(directory / 'best.pth')
            row = dict(step=step, selected_step=selected_step, breach_rate=score[0], attacker_return=score[1],
                       channel_breaches=result[0]['summary']['breaches'], reference_breaches=result[1]['summary']['breaches'],
                       elapsed_seconds=time.perf_counter()-elapsed, channel_steps=counts[0], reference_steps=counts[1])
            metrics.append(row)
            with (directory / 'metrics.csv').open('w', newline='', encoding='utf-8') as target:
                writer = csv.DictWriter(target, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(metrics)
            print(json.dumps(row), flush=True)
    (directory / 'complete.json').write_text(json.dumps(dict(passed=True, seed=seed, steps=steps,
        sha256=sha256(directory / 'best.pth'), selected_step=selected_step, wall_seconds=time.perf_counter()-elapsed,
        defender_steps=counts, defender_episodes=episode_counts, config=config), indent=2), encoding='utf-8')


def schedule(root, device):
    directory = root / 'results/attackers'
    directory.mkdir(parents=True, exist_ok=True)
    for seed in ATTACKER_SEEDS:
        run = directory / f'seed{seed}'
        if (run / 'complete.json').exists():
            continue
        if run.exists():
            raise RuntimeError(f'An interrupted attacker run requires diagnosis: {run}')
        with (directory / f'seed{seed}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-X', 'utf8', '-u', __file__, 'train', '--root', str(root),
                                     '--seed', str(seed), '--device', device], stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        if result.returncode:
            raise RuntimeError(f'Attacker seed {seed} failed: {result.returncode}')
        print(f'[ATTACKER COMPLETE] seed={seed}', flush=True)
    (directory / 'complete.json').write_text(json.dumps(dict(passed=True, seeds=ATTACKER_SEEDS)), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'schedule', 'evaluate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1709)
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--kind', choices=('channel', 'arboids', 'boids'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--attacker', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--episodes', type=int, default=2048)
    parser.add_argument('--eval-seed', type=int, default=9050000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'train':
        train_attacker(args.seed, args.root.resolve(), args.device, args.steps)
    elif args.command == 'schedule':
        schedule(args.root.resolve(), args.device)
    else:
        if not args.output or not args.kind:
            parser.error('evaluate requires --kind and --output')
        if args.output.exists():
            raise FileExistsError(args.output)
        model = load_attacker(args.attacker) if args.attacker else None
        result = evaluate_duel(PhysicalPolicy(args.kind, args.checkpoint), model,
                               range(args.eval_seed, args.eval_seed + args.episodes))
        result['attacker'] = dict(checkpoint=str(args.attacker), sha256=sha256(args.attacker)) if args.attacker else dict(kind='APF')
        result['evaluator_sha256'] = sha256(__file__)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
