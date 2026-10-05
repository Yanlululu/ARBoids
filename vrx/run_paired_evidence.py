"""Prospective paired VRX comparison of two frozen checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2100000)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    inputs = dict(channel=dict(path=str(args.channel.resolve()), sha256=checksum(args.channel)),
                  reference=dict(path=str(args.reference.resolve()), sha256=checksum(args.reference)),
                  episodes=args.episodes, seed=args.seed, setting=1, agility=2.25,
                  duration=60., termination_rule='paper', action_period=.2,
                  canonical_agent_order=True, ros_domain_id=os.environ.get('ROS_DOMAIN_ID'),
                  trial_source_sha256=checksum(Path(__file__).with_name('run_experiment.py')),
                  randomization_seed=314159)
    lock = root / 'protocol.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8')) != inputs:
        raise RuntimeError('Frozen VRX protocol or checkpoints changed')
    lock.write_text(json.dumps(inputs, indent=2), encoding='utf-8')
    order = random.Random(314159)
    rows = []
    started = time.monotonic()
    for episode in range(args.episodes):
        arms = ['channel', 'reference']
        order.shuffle(arms)
        pair = {}
        for arm in arms:
            name = f'{episode:03d}-{arm}'
            directory = root / name
            result_path = directory / 'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding='utf-8'))
                if not result.get('passed'):
                    raise RuntimeError(f'Infrastructure failure requires explicit diagnosis before retry: {name}')
            else:
                command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).with_name('run_experiment.py')),
                           '--checkpoint', inputs[arm]['path'], '--controller', 'ChannelMAPPO' if arm == 'channel' else 'AdaRes',
                           '--setting', '1', '--agility', '2.25', '--seed', str(args.seed + episode),
                           '--duration', '60', '--action-period', '0.2', '--termination-rule', 'paper',
                           '--headless', '--canonical-agent-order', '--wall-timeout', '600',
                           '--output-dir', str(root), '--run-id', name]
                with (root / f'{name}.log').open('w', encoding='utf-8') as log:
                    done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=660)
                result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'passed': False}
                if done.returncode or not result.get('passed'):
                    raise RuntimeError(f'Infrastructure failure in {name}: {result.get("error")}')
            if result.get('checkpoint_sha256') != inputs[arm]['sha256']:
                raise RuntimeError('Trial checkpoint changed')
            pair[arm] = result
        if pair['channel']['initial_poses'] != pair['reference']['initial_poses']:
            raise RuntimeError('Paired VRX initial conditions differ')
        rows.append(dict(seed=args.seed + episode, **pair))
        summary = {}
        for arm in ('channel', 'reference'):
            values = [r[arm] for r in rows]
            summary[arm] = dict(episodes=len(values), successes=sum(v['success'] for v in values),
                               collisions=sum(v['outcome_code'] == 2 for v in values),
                               breaches=sum(v['outcome_code'] == 1 for v in values),
                               timeout_denials=sum(v['outcome_code'] == 4 for v in values))
        result = dict(passed=len(rows) == args.episodes, protocol=inputs, pairs=rows,
                      summary=summary, wall_seconds=time.monotonic() - started)
        temporary = root / 'paired-results.json.tmp'
        temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
        temporary.replace(root / 'paired-results.json')
        print(f'[PAIR] {episode + 1}/{args.episodes} {json.dumps(summary)}', flush=True)


if __name__ == '__main__':
    main()
