"""Evaluate repeatable VRX episodes, retaining infrastructure failures separately."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--controller', choices=['AdaRes', 'Res', 'RL', 'Boids', 'ChannelMAPPO'], default='AdaRes')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--setting', type=int, choices=[0, 1], default=1)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--agility', type=float, default=2.25)
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--termination-rule', choices=['source', 'paper'], default='source')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--capture-first', action='store_true', help='Save real camera frames for the first episode')
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('episodes must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for i in range(args.episodes):
        run_id = f'episode-{i:03d}'
        command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).with_name('run_experiment.py')),
                   '--checkpoint', str(args.checkpoint.resolve()), '--setting', str(args.setting),
                   '--controller', args.controller,
                   '--seed', str(args.seed+i), '--agility', str(args.agility),
                   '--duration', str(args.duration), '--termination-rule', args.termination_rule,
                   '--output-dir', str(args.output_dir.resolve()),
                   '--run-id', run_id]
        if not args.gui:
            command.append('--headless')
        if args.capture_first and i == 0:
            command.append('--capture-frames')
        env = dict(os.environ, GZ_PARTITION=f'arboids-{os.getpid()}-{i}')
        with (args.output_dir / f'{run_id}.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        result_file = args.output_dir / run_id / 'result.json'
        result = json.loads(result_file.read_text()) if result_file.exists() else {'passed': False, 'error': 'Missing result.json'}
        result['returncode'] = completed.returncode
        results.append(result)
        (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'[BATCH] {i+1}/{args.episodes} passed={result["passed"]} outcome={result.get("outcome")}', flush=True)
        if completed.returncode or not result['passed']:
            raise RuntimeError(f'VRX infrastructure failure in {run_id}: {result.get("error")}')
        time.sleep(1)
    summary = dict(passed=True, episodes=len(results), successes=sum(r['success'] for r in results),
                   success_rate=sum(r['success'] for r in results)/len(results), setting=args.setting,
                   agility=args.agility, first_seed=args.seed, termination_rule=args.termination_rule,
                   checkpoint_sha256=results[0]['checkpoint_sha256'])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (args.output_dir / 'episodes.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ['seed', 'setting', 'agility', 'success', 'outcome', 'simulation_seconds', 'wall_seconds', 'control_steps']
        writer = csv.DictWriter(file, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
