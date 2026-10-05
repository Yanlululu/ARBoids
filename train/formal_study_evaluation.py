"""Fixed-scene evaluation without critic access or checkpoint selection."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from envs.TADgame import TADEnv
from policy.mappo import PredictiveMAPPO
from policy.networks import ActorAdap
from policy.message_stress import MessageStress, payload_bytes
from formal_study_training import atomic_json, digest


def wilson(k, n):
    z = 1.959963984540054
    denominator = 1 + z*z/n
    center = (k/n + z*z/(2*n))/denominator
    half = z*math.sqrt(k/n*(1-k/n)/n + z*z/(4*n*n))/denominator
    return [max(0., center-half), min(1., center+half)]


def load_policy(path, kind, defenders):
    if kind == 'mappo':
        agent = PredictiveMAPPO.from_checkpoint(path, 'cpu')
        trained_defenders = agent.defender_num
        # Actor's mean/attention embeddings accept arbitrary team sizes. Critic is
        # deliberately never evaluated or resized in zero-shot team-size tests.
        agent.defender_num = defenders
        return agent, trained_defenders
    actor = ActorAdap(6, 8, 3, 512, 1.)
    actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    actor.eval()
    return actor, 3


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = digest(args.checkpoint)
    specification = dict(checkpoint_sha256=checkpoint_hash, kind=args.kind, first_seed=args.first_seed,
                         episodes=args.episodes, defenders=args.defenders, agility=args.agility,
                         attacker=args.attacker, delay_steps=args.delay_steps, drop_probability=args.drop_probability,
                         protocol='paper-parameters-v1', duration=60., action_period=.2)
    spec_path = args.output/'specification.json'
    if spec_path.exists() and json.loads(spec_path.read_text(encoding='utf-8')) != specification:
        raise ValueError('Existing evaluation has a different frozen specification.')
    atomic_json(spec_path, specification)
    actor, trained_n = load_policy(args.checkpoint, args.kind, args.defenders)
    if args.kind == 'mappo' and actor.settings.compatibility_joint_mixture:
        combinations = (actor.settings.compatibility_mixture_points+1)**args.defenders
        if combinations > 10000:
            atomic_json(args.output/'unsupported.json', dict(reason='joint enumeration limit', combinations=combinations,
                        maximum=10000, measured_task_failure=False, **specification))
            print('UNSUPPORTED: joint enumeration limit; no fabricated task outcome.', flush=True)
            return
    journal = args.output/'episodes.jsonl'
    rows = []
    if journal.exists():
        rows = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
        if [row['seed'] for row in rows] != list(range(args.first_seed, args.first_seed+len(rows))):
            raise ValueError('Incomplete or duplicated evaluation seed sequence.')
    if len(rows) > args.episodes:
        raise ValueError('Too many rows in evaluation journal.')
    all_inference = []
    for i in range(len(rows), args.episodes):
        seed = args.first_seed+i
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = TADEnv(args.defenders, True, True, protocol='paper-parameters-v1', total_time=60.)
        if args.attacker == 'wide-apf':
            env.Obs_R = 12.  # Same APF, enlarged repulsive obstacle; never used in training.
        observation, _ = env.reset(args.agility, noisy_agility=False)
        network = MessageStress(args.delay_steps, args.drop_probability, seed=seed+12345)
        collision, done, steps, inference = False, 0, 0, []
        while not done:
            started = time.perf_counter()
            with torch.no_grad():
                if args.kind == 'mappo':
                    action = network.act(actor, observation, env.prediction_snapshot(),
                                         env.thrust_to_action(env.boids_actions), env.Current_T)
                else:
                    action = actor(torch.as_tensor(observation, dtype=torch.float32), True, False)[0].numpy()
            inference.append(time.perf_counter()-started)
            if not np.isfinite(action).all():
                raise FloatingPointError('Non-finite evaluation action, not a task failure.')
            attacker_action = None
            if args.attacker == 'direct':
                thrust = env._APF_navi_step(env.attacker.pos, np.zeros(2), [], env.attacker.theta)
                attacker_action = env.thrust_to_action(thrust, env.attacker.agility)
            observation, _, done, _ = env.step(action, 'AdaRes', attacker_action)
            events = env.physical_events()
            collision |= events['collision']
            steps += 1
            if steps > 300:
                raise RuntimeError('Finite-horizon evaluation exceeded 300 decisions.')
        row = dict(seed=seed, success=int(done>2), capture=int(done==3), timeout=int(done==4),
                   collision=int(collision), breach=int(events['breach']), outcome_code=int(done), steps=steps,
                   inference_mean=float(np.mean(inference)), inference_p95=float(np.quantile(inference,.95)),
                   inference_max=float(max(inference)), inference_deadline_misses=sum(t>.2 for t in inference),
                   message_attempts=network.attempts, message_drops=network.drops,
                   maximum_message_age=network.maximum_age)
        with journal.open('a', encoding='utf-8') as file:
            file.write(json.dumps(row, allow_nan=False)+'\n')
            file.flush()
        rows.append(row)
        all_inference.extend(inference)
        if (i+1)%25 == 0:
            print(json.dumps(dict(completed=i+1, total=args.episodes, **{k:sum(r[k] for r in rows) for k in
                              ('success','collision','breach')})), flush=True)
    summary = dict(**specification, completed=True, trained_defenders=trained_n, task_episodes=len(rows))
    for k in ('success', 'capture', 'timeout', 'collision', 'breach'):
        count = sum(row[k] for row in rows)
        summary[k+'_count'] = count
        summary[k+'_rate'] = count/len(rows)
        summary[k+'_wilson95'] = wilson(count, len(rows))
    summary['mean_episode_inference_seconds'] = float(np.mean([row['inference_mean'] for row in rows]))
    summary['maximum_inference_seconds'] = max(row['inference_max'] for row in rows)
    summary['message_payload'] = payload_bytes(args.defenders, actor.settings.compatibility_peer_intent) if args.kind=='mappo' else None
    if digest(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed while evaluating.')
    with (args.output/'episodes.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(args.output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--kind', choices=('sac','mappo'), required=True)
    parser.add_argument('--first-seed', type=int, required=True)
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--defenders', type=int, default=3)
    parser.add_argument('--agility', type=float, default=2.)
    parser.add_argument('--attacker', choices=('apf','direct','wide-apf'), default='apf')
    parser.add_argument('--delay-steps', type=int, default=0)
    parser.add_argument('--drop-probability', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.defenders < 2 or args.agility <= 0:
        parser.error('Positive episode count/agility and at least two defenders required.')
    torch.set_num_threads(1)
    evaluate(args)


if __name__ == '__main__':
    main()
