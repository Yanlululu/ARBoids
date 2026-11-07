"""Evaluate repeatable VRX episodes, retaining infrastructure failures separately."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--controller', choices=['AdaRes', 'Res', 'RL', 'Boids', 'MAPPO', 'CITF', 'ARBoidsUnified'], default='AdaRes')
    parser.add_argument('--scene-seed', type=int, default=74002)
    parser.add_argument('--policy-seed', type=int, default=84002)
    parser.add_argument('--research-stage', choices=['development', 'formal'], default='formal')
    parser.add_argument('--registry', help='Common study registry required by formal unified batches.')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--setting', type=int, choices=[0, 1], default=1)
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--agility', type=float, default=2.25)
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--termination-rule', choices=['source', 'paper'], default='source')
    parser.add_argument('--apf-obstacle-rule', choices=['all_obstacles', 'legacy_ordered'], default='all_obstacles')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--capture-first', action='store_true', help='Save real camera frames for the first episode')
    parser.add_argument('--assets-dir', type=Path, help='Use the existing shared VRX world assets')
    args = parser.parse_args()
    unified = args.controller in ('CITF', 'ARBoidsUnified')
    if args.episodes <= 0:
        parser.error('episodes must be positive')
    if unified and args.research_stage == 'formal' and (args.episodes != 100
            or args.scene_seed != 74002 or args.policy_seed != 84002):
        parser.error('Formal unified VRX retains all 100 registered scenes with seeds 74002/84002.')
    identity, adapter = None, None
    if unified:
        sys.path.insert(0, os.environ.get('CITF_CODE_ROOT', str(Path(__file__).resolve().parents[1]/'citf'/'code')))
        from unified.protocol import (load_registry, read_endpoint, evaluation_identity, vrx_source_identity,
                                      vrx_rows, outcome_metrics, attach_identity, validate_result, digest)
        from unified.train import write_json
        adapter = vrx_source_identity(Path(__file__).parent)
        if args.research_stage == 'formal':
            if not args.registry:
                parser.error('Formal unified VRX requires --registry from the numeric study.')
            if (args.setting != 0 or args.termination_rule != 'paper' or args.duration != 60.
                    or args.agility != 2.25):
                parser.error('Formal unified VRX uses open-water setting 0, paper events, 60 s and agility 2.25.')
            registry = load_registry(args.registry)
            method = 'full' if args.controller == 'CITF' else 'arboids'
            endpoint = read_endpoint(args.checkpoint, registry, method, 2101)
            identity = evaluation_identity(registry, method, 2101, endpoint,
                dict(registry['task'], agility=2.25), args.scene_seed, args.policy_seed, 100, kind='vrx')
            protocol = dict(registry_id=registry['id'], adapter_source=adapter,
                            task=identity['task'], scene_seed=74002, policy_seed=84002, episodes=100)
            protocol['id'] = digest(protocol)
            protocol_path = args.output_dir.parent/'vrx_protocol.json'
            protocol_path.parent.mkdir(parents=True, exist_ok=True)
            if protocol_path.exists():
                if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
                    raise ValueError('Both VRX methods must retain the same frozen adapter source and task.')
            else:
                write_json(protocol_path, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for i in range(args.episodes):
        run_id = f'episode-{i:03d}'
        command = [sys.executable, '-X', 'utf8', '-u', str(Path(__file__).with_name('run_experiment.py')),
                   '--checkpoint', str(args.checkpoint.resolve()), '--setting', str(args.setting),
                   '--controller', args.controller,
                   '--seed', str(args.seed+i), '--agility', str(args.agility),
                   '--duration', str(args.duration), '--termination-rule', args.termination_rule,
                   '--apf-obstacle-rule', args.apf_obstacle_rule,
                   '--output-dir', str(args.output_dir.resolve()),
                   '--run-id', run_id]
        if not args.gui:
            command.append('--headless')
        if unified:
            command += ['--scene-seed', str(args.scene_seed), '--scene-index', str(i),
                        '--policy-seed', str(args.policy_seed), '--research-stage', args.research_stage]
            if args.registry:
                command += ['--registry', str(Path(args.registry).resolve())]
        if args.capture_first and i == 0:
            command.append('--capture-frames')
        if args.assets_dir:
            command += ['--assets-dir', str(args.assets_dir.resolve())]
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
                   collision_rate=sum(r.get('outcome_code') == 2 if unified else bool(r.get('defender_collision')) for r in results)/len(results),
                   capture_rate=sum(r.get('outcome_code') == 3 for r in results)/len(results),
                   timeout_rate=sum(r.get('outcome_code') == 4 for r in results)/len(results),
                   breach_rate=sum(r.get('outcome_code') == 1 for r in results)/len(results),
                   controller=args.controller,
                   agility=args.agility, first_seed=args.seed, termination_rule=args.termination_rule,
                   apf_obstacle_rule=results[0]['apf_obstacle_rule'], radial_early_failure=False)
    if unified:
        summary.update(scene_seed=args.scene_seed, policy_seed=args.policy_seed, research_stage=args.research_stage,
                       checkpoint=str(args.checkpoint.resolve()), training_seed=results[0]['training_seed'],
                       training_transitions=results[0]['training_transitions'],
                       implementation_version=results[0]['implementation_version'], task=results[0]['task'],
                       control_deadline_misses=sum(r['control_deadline_misses'] for r in results),
                       discarded_late_outputs=sum(r['discarded_late_outputs'] for r in results),
                       control_steps=sum(r['control_steps'] for r in results))
        if identity is not None:
            rows = vrx_rows(results, identity, adapter)
            summary.update(outcome_metrics(rows, args.episodes, args.scene_seed, identity['task']['duration']),
                           rows=rows, adapter_source=adapter, vrx_protocol_id=protocol['id'])
            attach_identity(summary, identity)
            validate_result(summary, identity)
        else:
            capture_seconds = [r['simulation_seconds'] for r in results if r['outcome_code'] == 3]
            summary.update(mean_termination_seconds=sum(r['simulation_seconds'] for r in results)/len(results),
                           capture_count=len(capture_seconds),
                           mean_capture_seconds=sum(capture_seconds)/len(capture_seconds) if capture_seconds else None)
        timing = {'inference_seconds': [], 'control_seconds': []}
        cancelled = 0
        for result in results:
            with np.load(result['trajectory']) as trajectory:
                cancelled += int(trajectory['TerminalCancellation'].sum())
                complete = trajectory['PolicyStarted'].astype(bool) & ~trajectory['TerminalCancellation'].astype(bool)
                for source, target in [('PolicyWallSeconds', 'inference_seconds'),
                                       ('ControlWallSeconds', 'control_seconds')]:
                    timing[target].extend(trajectory[source][complete].tolist())
        for name, samples in timing.items():
            summary[name] = (dict(count=len(samples), mean=float(np.mean(samples)),
                                  p95=float(np.quantile(samples, .95)), maximum=float(np.max(samples)))
                             if samples else dict(count=0, mean=None, p95=None, maximum=None))
        summary['terminal_cancelled_queries'] = cancelled
        summary['control_records'] = sum(r['control_records'] for r in results)
        summary['skipped_control_cycles'] = sum(r['skipped_control_cycles'] for r in results)
        summary['control_hold_cycles'] = sum(r['control_hold_cycles'] for r in results)
        summary['control_denominator'] = 'all scheduled simulation-time cycles, including unsubmitted cycles'
        summary['control_deadline_miss_rate'] = summary['control_deadline_misses']/summary['control_steps']
    else:
        summary['checkpoint_sha256'] = results[0]['checkpoint_sha256']
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (args.output_dir / 'episodes.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ['seed', 'scene_seed', 'scene_index', 'setting', 'agility', 'success', 'outcome', 'defender_collision', 'simulation_seconds', 'wall_seconds', 'control_steps']
        writer = csv.DictWriter(file, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
