"""Train the five fixed IA-CRRL study arms with full resumable state."""
import study_runtime
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from envs.TADgame import TADEnv
from envs.snapshot import RandomState, seed_random, preserved_random_state
from policy.interaction_sac import JointReplay, make_agent
from interaction_rollout import public_packet, execute, compact_snapshot, SnapshotPool, InterventionSampler


ARMS = ('arboids_cbf', 'same_info', 'model_value', 'short', 'full')


def arm_config(config, arm):
    if arm not in ARMS:
        raise ValueError('Unknown study arm: ' + arm)
    config = copy.deepcopy(config)
    config['interaction'].update(arm=arm, auxiliary=('none' if arm in ('arboids_cbf', 'same_info')
                                                    else 'value' if arm == 'model_value' else 'difference'),
                                  horizon_steps=10 if arm == 'short' else 100)
    return config


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def atomic_torch(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(data, temporary)
    temporary.replace(path)


def outcome_row(env, outcome):
    events = env.physical_events()
    captured = outcome == 3
    return dict(success=int(outcome in (3, 4)), capture=int(captured), collision=int(events['collision']),
                breach=int(events['breach']), timeout=int(outcome == 4), outcome=outcome,
                capture_time=float(env.Current_T if captured else env.Total_T),
                duration=float(env.Current_T))


def episode(policy, seed, defenders=3, agility=2.25, *, safety=True, trajectory=False, attacker=None):
    with preserved_random_state():
        seed_random(int(seed))
        env = TADEnv(defenders, protocol='paper-parameters-v1', LearningSide='Att' if attacker else 'Def')
        obs, _ = env.reset(agility, noisy_agility=False)
        np.random.seed(int(seed) + 1_000_000)
        packet, done, history = public_packet(env, obs), 0, []
        while not done:
            action, _ = policy.choose_action(packet, deterministic=True)
            attacker_action = None
            if attacker is not None:
                _, att_obs = env._get_obs()
                attacker_action = attacker(att_obs)
            before = packet
            packet, _, done, thrust, _, _ = execute(env, packet, action, safety=safety, attacker_action=attacker_action)
            if trajectory:
                history.append(dict(time=float(env.Current_T), motion=packet['motion'].copy(),
                                    attacker=np.asarray(env.attacker.pos).copy(), gates=action[:, 2].copy(),
                                    action=action.copy(), thrust=thrust.copy(), before=before['motion'].copy()))
        return outcome_row(env, done), history


def evaluate(policy, episodes, seed_base, defenders=3, agility=2.25, safety=True):
    rows = [episode(policy, seed_base + i, defenders, agility, safety=safety)[0] for i in range(episodes)]
    return {key: float(np.mean([r[key] for r in rows]))
            for key in ('success', 'capture', 'collision', 'breach', 'timeout', 'capture_time')}


def export_policy(agent, output, step):
    atomic_torch(Path(output) / 'policy.pth', dict(format='interaction-sac-deployment-v1',
        config=agent.config, actor={k: v.detach().cpu() for k, v in agent.actor.state_dict().items()},
        step=int(step), stage=agent.stage))


def validate(config):
    if config['environment'] != dict(protocol='paper-parameters-v1', total_time=60., agility_noise_half_width=.5):
        raise ValueError('This study requires the fixed 60-second paper protocol.')
    t, i = config['training'], config['interaction']
    positive = ('joint_steps', 'warm_steps', 'replay_capacity', 'eval_interval',
                'eval_episodes', 'log_interval', 'checkpoint_interval')
    if t['gate_steps'] < 0 or any(t[k] <= 0 for k in positive) or t['warm_steps'] >= t['gate_steps'] + t['joint_steps']:
        raise ValueError('Invalid training intervals or step budget.')
    if any(i[k] <= 0 for k in ('intervention_interval', 'pairs_per_batch', 'snapshot_capacity', 'snapshot_interval', 'horizon_steps')):
        raise ValueError('Invalid intervention settings.')
    if i['auxiliary'] not in ('none', 'value', 'difference') or i['auxiliary_weight'] < 0:
        raise ValueError('Invalid auxiliary objective.')
    if not i['safety']:
        raise ValueError('Training study arms must share the CBF filter.')


def run_training(config, exp=None, device='cpu', *, output=None, max_steps=None, attacker=None):
    config = copy.deepcopy(config)
    validate(config)
    torch.set_num_threads(1)
    t, interaction = config['training'], config['interaction']
    output = Path(output if output is not None else exp.exp_dir)
    output.mkdir(parents=True, exist_ok=True)
    resume = Path(t['resume']) if t.get('resume') else output / 'resume.pth'
    if resume.exists():
        # Full resumes contain our own simulator and RNG objects; deployment uses weights_only=True.
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved.get('format') != 'interaction-sac-resume-v1':
            raise ValueError('Unsupported resume format.')
        old, new = copy.deepcopy(saved['config']), copy.deepcopy(config)
        for c in (old, new):
            for key in ('resume', 'pretrain_checkpoint'):
                c['training'].pop(key, None)
        if old != new:
            raise ValueError('Resume configuration differs from the saved scientific settings.')
    else:
        saved = None
        if (output / 'metrics.csv').exists():
            raise RuntimeError('Existing metrics have no recoverable checkpoint.')
    seed_random(int(t['seed']))
    agent = make_agent(config, device)
    replay, pool = JointReplay(t['replay_capacity']), SnapshotPool(interaction['snapshot_capacity'])
    step, simulated, episodes, prior_elapsed, auxiliary, losses = 0, 0, 0, 0., None, {}
    env, packet, done = None, None, True
    if saved is not None:
        agent.load_state_dict(saved['agent'])
        replay.load_state_dict(saved['replay'])
        pool = saved['pool']
        step, simulated, episodes = saved['step'], saved['simulated_steps'], saved['episodes']
        prior_elapsed, auxiliary, done = saved['elapsed'], saved['auxiliary'], saved['done']
        if saved['snapshot'] is not None:
            env = copy.deepcopy(saved['snapshot'].environment)
            packet = public_packet(env)
        saved['random'].restore()
        del saved
    elif t.get('initial_state'):
        initial = torch.load(t['initial_state'], map_location='cpu', weights_only=False)
        if initial['config']['interaction']['arm'] != interaction['arm']:
            raise ValueError('Cannot change study arms between adversarial phases.')
        agent.load_state_dict(initial['agent'])
        agent.set_stage('joint' if t['gate_steps'] == 0 else 'gate')
        del initial
    elif t.get('pretrain_checkpoint'):
        weights = torch.load(t['pretrain_checkpoint'], map_location='cpu', weights_only=True)
        if agent.legacy:
            agent.actor.load_state_dict(weights, strict=True)
        else:
            agent.actor.initialize_source(weights)
    elif not t.get('allow_random_initialization', False):
        raise ValueError('A verified common pretraining checkpoint is required.')
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    started = time.perf_counter()
    total = t['gate_steps'] + t['joint_steps']
    stop_at = total if max_steps is None else min(total, step + int(max_steps))
    sampler = InterventionSampler(interaction['workers'])
    metrics_fields = ['step', 'stage', 'elapsed', 'simulated_steps', 'success', 'capture', 'collision',
                      'breach', 'timeout', 'capture_time']

    def checkpoint():
        elapsed = prior_elapsed + time.perf_counter() - started
        atomic_torch(output / 'resume.pth', dict(format='interaction-sac-resume-v1', config=config,
            agent=agent.state_dict(), replay=replay.state_dict(), pool=pool, step=step, simulated_steps=simulated,
            episodes=episodes, elapsed=elapsed, auxiliary=auxiliary, done=bool(done),
            snapshot=None if env is None else compact_snapshot(env), random=RandomState.capture()))
        export_policy(agent, output, step)
        atomic_json(output / 'progress.json', dict(step=step, total=total, stage=agent.stage,
                    simulated_steps=simulated, elapsed=elapsed, complete=step == total))

    try:
        while step < stop_at:
            stage = 'gate' if step < t['gate_steps'] else 'joint'
            if agent.stage != stage:
                agent.set_stage(stage)
                auxiliary = None
            if done:
                env = TADEnv(config['agent']['defender_num'], LearningSide='Att' if attacker else 'Def',
                             **config['environment'])
                agility = t.get('fixed_agility', 2.25 if stage == 'gate' else 2. + .25 * min(3, (step - t['gate_steps']) // 250000))
                obs, _ = env.reset(agility, noisy_agility=True)
                packet, done = public_packet(env, obs), False
                episodes += 1
            if step % interaction['snapshot_interval'] == 0 and interaction['auxiliary'] != 'none':
                pool.add(env)
            action, _ = agent.choose_action(packet)
            attacker_action = None
            if attacker is not None:
                _, att_obs = env._get_obs()
                attacker_action = attacker(att_obs)
            following, reward, outcome, thrust, _, _ = execute(env, packet, action, attacker_action=attacker_action)
            replay.store(packet, action, thrust, reward, following, bool(outcome))
            packet, done, step = following, bool(outcome), step + 1
            if step >= t['warm_steps']:
                if interaction['auxiliary'] != 'none' and step % interaction['intervention_interval'] == 0:
                    auxiliary, used = sampler.generate(agent, pool,
                        attacker=None if attacker is None else attacker.state_dict())
                    simulated += used
                losses = agent.learn(replay, auxiliary)
            if step % t['eval_interval'] == 0 or step == total:
                scores = evaluate(agent, t['eval_episodes'], 310000000 + int(t['seed']) * 10000,
                                  config['agent']['defender_num'])
                row = dict(step=step, stage=stage, elapsed=prior_elapsed + time.perf_counter()-started,
                           simulated_steps=simulated, **scores)
                exists = (output / 'metrics.csv').exists()
                with (output / 'metrics.csv').open('a', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=metrics_fields)
                    if not exists:
                        writer.writeheader()
                    writer.writerow(row)
                print('[IA-EVAL] ' + json.dumps(row), flush=True)
            if step % t['log_interval'] == 0:
                status = dict(step=step, total=total, stage=stage, simulated_steps=simulated,
                              elapsed=prior_elapsed + time.perf_counter()-started, complete=False, **losses)
                atomic_json(output / 'progress.json', status)
                print('[IA-TRAIN] ' + json.dumps(status), flush=True)
            if step % t['checkpoint_interval'] == 0 or step == stop_at:
                checkpoint()
        return dict(output=str(output), step=step, complete=step == total)
    except KeyboardInterrupt:
        checkpoint()
        raise
    finally:
        sampler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/interaction-aware-sac.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, default='full')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pretrain', type=Path)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = arm_config(yaml.safe_load(args.config.read_text(encoding='utf-8')), args.arm)
    config['training'].update(seed=args.seed, pretrain_checkpoint=None if args.pretrain is None else str(args.pretrain))
    if args.smoke:
        config['training'].update(gate_steps=20, joint_steps=20, warm_steps=8, replay_capacity=128,
            eval_interval=20, eval_episodes=2, checkpoint_interval=20, log_interval=10,
            allow_random_initialization=args.pretrain is None)
        config['rl']['batch_size'] = 16
        config['interaction'].update(workers=0, pairs_per_batch=2, intervention_interval=10,
                                     horizon_steps=2, snapshot_interval=2)
    print(json.dumps(run_training(config, device=args.device, output=args.output, max_steps=args.max_steps)), flush=True)


if __name__ == '__main__':
    main()
