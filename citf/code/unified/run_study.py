"""Finite main-study training queue and the two prescribed frozen-model sweeps.

Run only after choosing the development recipe. This is a one-shot experiment
campaign, not a scheduled service; its registry fixes all seeds before training.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .train import write_json
from .environment import TaskConfig
from .protocol import (BASELINE_RECIPE, FULL_KEYS, build_registry, load_registry, registered_arguments,
                       run_identity, read_endpoint, evaluation_identity, validate_result, validate_learning_curve)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True, help='Chosen development configuration.json')
    parser.add_argument('--baseline-recipe', help='Chosen public-baseline configuration.json; omitted uses author-width/batch and 1:1 updates.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--seeds', type=int, nargs=5, default=[2101, 2102, 2103, 2104, 2105])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.parallel < 1 or args.seeds != [2101, 2102, 2103, 2104, 2105]:
        raise ValueError('Use a positive job count and the five preregistered formal training seeds 2101 through 2105.')
    recipe = json.loads(Path(args.recipe).read_text(encoding='utf-8'))
    chosen = recipe['arguments']
    baseline_recipe = dict(BASELINE_RECIPE)
    if args.baseline_recipe:
        baseline_source = json.loads(Path(args.baseline_recipe).read_text(encoding='utf-8'))
        baseline_arguments = baseline_source['arguments']
        baseline_recipe = {key: baseline_arguments[key] for key in baseline_recipe}
    if chosen.get('no_action_interaction') or chosen.get('feedback', 0) <= 0:
        raise ValueError('Freeze a complete-method development recipe for the common formal study.')
    root = Path(args.out).resolve()
    code = Path(__file__).resolve().parents[1]
    methods = ['full', 'no_action', 'no_feedback', 'arboids', 'rp', 'sac']
    registry = build_registry(recipe, baseline_recipe)
    registry_path = root/'registry.json'
    if root.exists():
        if not args.resume or not registry_path.exists():
            raise FileExistsError('Use a new study output directory, or explicitly resume its registered campaign.')
        if load_registry(registry_path) != registry:
            raise ValueError('A resumed formal campaign retains its complete recipe and seed registry.')
    else:
        root.mkdir(parents=True)
        write_json(registry_path, registry)
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    common_keys = ['thrust_delta', 'collision_penalty']
    full_keys = FULL_KEYS

    def options(keys):
        result = []
        for key in keys:
            if key in chosen and chosen[key] is not None:
                result += ['--'+key.replace('_', '-'), str(chosen[key])]
        return result

    queue = []
    # Interleave training seeds so every completed wave has matched versions.
    for seed in args.seeds:
        for method in methods:
            out = root/method/str(seed)
            full = method in methods[:3]
            expected_arguments = registered_arguments(registry, method, seed)
            expected_run = run_identity(expected_arguments, registry['task'], registry)
            status_path = out/'status.json'
            launch_path = out/'launch.json'
            configuration_path = out/'configuration.json'
            if configuration_path.exists():
                configuration = json.loads(configuration_path.read_text())
                if configuration.get('run_identity') != expected_run:
                    raise ValueError(f'{out} does not match the registered method, seed, task and learning recipe.')
            if launch_path.exists():
                prior_pid = json.loads(launch_path.read_text())['pid']
                process_command = Path(f'/proc/{prior_pid}/cmdline')
                if process_command.exists() and str(out).encode() in process_command.read_bytes():
                    raise RuntimeError(f'{out} still has an active training process; do not start a duplicate campaign.')
            if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'completed':
                record = json.loads(status_path.read_text())
                endpoint = read_endpoint(out/'last.pt', registry, method, seed)
                if (not configuration_path.exists() or record.get('run_id') != expected_run['id']
                        or record['transitions'] != 1000000 or record.get('updates') != endpoint['updates']
                        or record.get('update_counts') != endpoint['update_counts']):
                    raise ValueError(f'{out} has a different completed training budget.')
                validate_learning_curve(out/'evaluations.jsonl', expected_run, endpoint=endpoint)
                continue
            command = [sys.executable, '-u', '-m', 'unified.train' if full else 'unified.train_baseline',
                       '--out', str(out), '--steps', '1000000', '--seed', str(seed), '--stage', 'formal',
                       '--eval-interval', '5000', '--eval-count', '100', '--eval-seed', '73001',
                       '--registry', str(registry_path)]+options(common_keys)
            if full:
                command += options([key for key in full_keys if key != 'feedback'])
                command += ['--feedback', '0' if method == 'no_feedback' else str(chosen['feedback'])]
                if method == 'no_action':
                    command += ['--no-action-interaction']
            else:
                command += ['--method', method, '--target-entropy', str(expected_arguments['target_entropy'])]
                for key, value in baseline_recipe.items():
                    command += ['--'+key.replace('_', '-'), str(value)]
            if (out/'last.pt').exists():
                command += ['--resume', str(out/'last.pt')]
            elif (out/'configuration.json').exists():
                raise RuntimeError(f'{out} failed before a resumable checkpoint; resolve that recorded run explicitly.')
            queue.append((method, seed, out, command))
    running = []
    while queue or running:
        while queue and len(running) < args.parallel:
            method, seed, out, command = queue.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = (out/'training.log').open('a')
            process = subprocess.Popen(command, cwd=code, env=environment, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            write_json(out/'launch.json', dict(pid=process.pid, command=command, cwd=str(code)))
            running.append((process, method, seed, out))
            print(json.dumps(dict(event='started', method=method, seed=seed, pid=process.pid)), flush=True)
        for entry in running[:]:
            process, method, seed, out = entry
            if process.poll() is None:
                continue
            status = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
            if process.returncode or status.get('status') != 'completed' or status.get('transitions') != 1000000:
                # Other already-running jobs may finish; no additional jobs are launched.
                write_json(root/'status.json', dict(status='stopped_on_training_failure', method=method,
                           seed=seed, running=[dict(method=m, seed=s, pid=p.pid) for p,m,s,_ in running if p.poll() is None]))
                raise RuntimeError(f'Incomplete scientific run: {out}. Remaining active jobs are recorded in study status.')
            running.remove(entry)
            print(json.dumps(dict(event='completed', method=method, seed=seed)), flush=True)
        if running:
            time.sleep(5)

    endpoints = {}
    for method in methods:
        for seed in args.seeds:
            directory = root/method/str(seed)
            endpoint = directory/'last.pt'
            endpoints[str(endpoint)] = read_endpoint(endpoint, registry, method, seed)
            run = run_identity(registered_arguments(registry, method, seed), registry['task'], registry)
            validate_learning_curve(directory/'evaluations.jsonl', run, endpoint=endpoints[str(endpoint)])
            status = json.loads((directory/'status.json').read_text())
            if (status.get('status') != 'completed' or status.get('run_id') != run['id']
                    or status.get('update_counts') != endpoints[str(endpoint)]['update_counts']):
                raise ValueError(f'{directory} has an inconsistent completion record.')

    def evaluate(checkpoint, output, scene_seed, policy_seed, defenders=3, agility=2.):
        method = 'boids' if checkpoint is None else checkpoint.parent.parent.name
        seed = None if checkpoint is None else int(checkpoint.parent.name)
        task = asdict(replace(TaskConfig(**registry['task']), defenders=defenders, agility=agility))
        identity = evaluation_identity(registry, method, seed, None if checkpoint is None else endpoints[str(checkpoint)],
                                       task, scene_seed, policy_seed, 100)

        def read_registered_result():
            return validate_result(json.loads(output.read_text()), identity)
        if output.exists():
            return read_registered_result()
        command = [sys.executable, '-u', '-m', 'unified.evaluate', '--out', str(output), '--count', '100',
                   '--stage', 'formal', '--registry', str(registry_path), '--scene-seed', str(scene_seed), '--policy-seed', str(policy_seed),
                   '--defenders', str(defenders), '--agility', str(agility),
                   '--collision-penalty', str(chosen.get('collision_penalty', 50.)),
                   '--thrust-delta', str(chosen['thrust_delta'])]
        if checkpoint is not None:
            command += ['--checkpoint', str(checkpoint)]
        with (root/'evaluation.log').open('a') as log:
            subprocess.run(command, cwd=code, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        return read_registered_result()

    table = []
    for method in methods:
        rates = []
        for index, seed in enumerate(args.seeds):
            run = root/method/str(seed)
            endpoint = run/'last.pt'
            result = evaluate(endpoint, run/'main_test.json', 74001, 84001+index)
            rates.append(result['success_rate'])
            if method in ('no_action', 'no_feedback'):
                continue
            for agility in (1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.):
                evaluate(endpoint, run/f'agility_{agility:.2f}.json', 74004, 84001+index, agility=agility)
            for defenders in range(2, 8):
                evaluate(endpoint, run/f'team_{defenders}.json', 74005, 84001+index, defenders=defenders)
        table.append(dict(method=method, seeds=args.seeds, success_rates=rates,
                          mean=statistics.mean(rates), std=statistics.stdev(rates)))
    boids = root/'boids'
    boids.mkdir(exist_ok=True)
    reference = evaluate(None, boids/'main_test.json', 74001, 84001)
    for agility in (1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.):
        evaluate(None, boids/f'agility_{agility:.2f}.json', 74004, 84001, agility=agility)
    for defenders in range(2, 8):
        evaluate(None, boids/f'team_{defenders}.json', 74005, 84001, defenders=defenders)
    table.append(dict(method='boids', success_rate=reference['success_rate'], trained=False))
    write_json(root/'main_results.json', dict(rows=table, registry_id=registry['id'], unit='independent training seed', scenarios_per_model=100))
    write_json(root/'status.json', dict(status='numeric_study_completed', registry_id=registry['id'], main_runs=30,
               system_experiments='alternating chain and VRX are separate prescribed tasks'))


if __name__ == '__main__':
    main()
