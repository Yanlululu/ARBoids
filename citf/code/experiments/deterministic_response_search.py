"""Optimize deployed greedy choices with paired parameter perturbations.

The short-response Transformer is fixed. Antithetic perturbations of its
64-dimensional action readout execute deterministic whole-configuration
policies on common training tasks. There is no value or outcome predictor,
per-control-tick categorical sampling, or future fusion-action sequence.
"""
import argparse
import concurrent.futures
from dataclasses import asdict, replace
import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.train import atomic_save
from citf.contracts import recorded_protocol,validate_task_banks
from experiments.candidate_response import CandidateResponseModel, ResponseConfig
from experiments.response_policy import initialize, rollout


def save_fixed_readout_average(directory, updates=(25, 30, 35, 40)):
    """Average fixed late readouts, equivalent to averaging their actor scores.

    All other parameters are frozen during this search. The stage range is
    fixed before evaluating the average; no success-based stage filtering is
    used, and the original stage and selected checkpoints remain unchanged.
    """
    directory = Path(directory)
    status = json.loads((directory / 'status.json').read_text())
    if not status.get('completed') or status['update'] < max(updates):
        raise ValueError('Readout averaging requires the complete registered search')
    destination = directory / f'average_{min(updates):06d}_{max(updates):06d}.pt'
    if destination.exists():
        return destination
    sources = [directory / f'stage_{update:06d}.pt' for update in updates]
    values = [torch.load(path, map_location='cpu', weights_only=False) for path in sources]
    value = values[-1]
    value['state'] = dict(value['state'])
    value['state']['actor.3.weight'] = torch.stack(
        [item['state']['actor.3.weight'] for item in values]).mean(0)
    value.pop('optimizer', None)
    value['selected'] = None
    value['readout_averaging'] = dict(updates=list(updates), sources=[str(path) for path in sources],
        parameter='actor.3.weight', weights=[1. / len(updates)] * len(updates),
        selection='fixed late stages, no validation or development result filtering',
        deployment='one current-configuration actor; no ensemble runtime cost',
        purpose='reduce task-sampling variation of greedy action scores')
    value['protocol'] = dict(value['protocol'], readout_averaging=value['readout_averaging'])
    atomic_save(value, destination)
    return destination


def main(model_factory=None, worker_initializer=initialize, worker_rollout=rollout):
    parser = argparse.ArgumentParser()
    for key in ('project', 'source', 'pretrained', 'train-bank', 'validation-bank', 'out', 'deadline'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--directions', type=int, default=8)
    parser.add_argument('--tasks-per-update', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--validation-every', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=.05)
    parser.add_argument('--lr', type=float, default=.01)
    args = parser.parse_args(); torch.set_num_threads(1); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    train = json.loads(Path(args.train_bank).read_text())
    validation = json.loads(Path(args.validation_bank).read_text())
    if train['role'] != 'train' or validation['role'] != 'validation':
        raise ValueError('Task search requires the registered train/validation roles')
    validate_task_banks(('train',train),('validation',validation))
    physical=recorded_protocol(train)
    if {x['geometry_family_id'] for x in train['tasks']} & {x['geometry_family_id'] for x in validation['tasks']}:
        raise ValueError('Training and validation physical layouts overlap')
    value = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    if model_factory is None:
        cfg = replace(ResponseConfig(**value['config']), deployment='policy',
            policy_prior_mode='reference_saturated', revision='CITF-deterministic-response-readout-search-v3')
        model = CandidateResponseModel(cfg).eval(); model.load_state_dict(value['state'])
        representation = dict(representation_kind='fixed-reference short-response Transformer',
                              initialization='checkpoint representation and actor weights')
    else:
        cfg, model, representation = model_factory(value)
    if hasattr(cfg,'radial_early_failure'):
        cfg=replace(cfg,radial_early_failure=physical.radial_early_failure);model.cfg=cfg
    model.requires_grad_(False)
    parameter = model.actor[-1].weight
    parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=args.lr)
    task_rng = np.random.default_rng(args.seed)
    noise_rng = np.random.default_rng(args.seed + 19000)
    order, cursor = [], 0
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    stop_file = str(out / 'STOP_DETERMINISTIC_SEARCH')
    protocol = dict(value['protocol'], arguments=vars(args),physical_protocol=asdict(physical),response_pretrained=args.pretrained,
        objective='ordinary complete-task return of the deployed deterministic current-configuration policy',
        parameter_estimator='antithetic Gaussian directions on a fixed response representation',
        frozen_response_encoder=True, optimized_parameters=parameter.numel(),
        per_tick_action_sampling=False, no_value_or_outcome_prediction=True,
        reward_duration='capped_capture: capture duration, otherwise80seconds; no benefit for failing earlier',
        selection='complete fixed validation tasks: success then fewer collisions then capped time',
        expected_training_episodes=args.iterations * 2 * args.directions * args.tasks_per_update)
    protocol.update(representation)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    update, episodes, steps, best = 0, 0, 0, None

    def state():
        return {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}

    def save(path):
        atomic_save(dict(config=asdict(cfg), state={k: v.detach() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), protocol=protocol, update=update, selected=best,
            seed=args.seed, training_episodes=episodes, training_steps=steps,
            task_rng=task_rng.bit_generator.state, noise_rng=noise_rng.bit_generator.state,
            task_order=order, task_cursor=cursor), path)

    def status(completed=False):
        (out / 'status.json').write_text(json.dumps(dict(completed=completed, update=update,
            total_updates=args.iterations, training_episodes=episodes, training_steps=steps,
            best=best, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))

    def make_job(task, parameters):
        return dict(task=task, state=parameters, training=False, record_transitions=False,
            policy_seed=args.seed, protocol=train.get('protocol', {}), deadline=deadline, stop_file=stop_file,
            reward_duration='capped_capture')

    status()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=worker_initializer,
            initargs=(args.project, args.source, asdict(cfg))) as pool:

        def validate():
            nonlocal best
            parameters = state()
            results = list(pool.map(worker_rollout, [make_job(task, parameters) for task in validation['tasks']]))
            complete = all(r['completed'] for r in results)
            if not complete:
                with (out / 'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(update=update, completed=False, results=results)) + '\n')
                return False
            rows = [r['record'] for r in results]
            success = sum(r['success'] for r in rows); collision = sum(r['collision'] for r in rows)
            capped = sum(r['duration'] if r['capture'] else 80 for r in rows) / len(rows)
            selection = [success, -collision, -capped]
            row = dict(update=update, completed=True, N=len(rows), success=success,
                collision=collision, capped_capture_time=capped, tasks=rows)
            with (out / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            if best is None or tuple(selection) > tuple(best['selection_key']):
                best = dict(update=update, selection_key=selection, success=success, collision=collision, N=len(rows))
                save(out / 'best.pt')
            save(out / f'stage_{update:06d}.pt'); save(out / 'last.pt'); status()
            print(json.dumps(dict(validation={k: v for k, v in row.items() if k != 'tasks'}, selected=best)), flush=True)
            return True

        proceed = validate()
        while proceed and update < args.iterations and time.time() < deadline and not Path(stop_file).exists():
            selected_tasks = []
            for _ in range(args.tasks_per_update):
                if cursor >= len(order):
                    order = task_rng.permutation(len(train['tasks'])).tolist(); cursor = 0
                task = dict(train['tasks'][order[cursor]]); cursor += 1
                task['future_seed'] = int(task_rng.integers(200000000, 2000000000))
                selected_tasks.append(task)
            noise = noise_rng.normal(size=(args.directions, *parameter.shape)).astype(np.float32)
            center = parameter.detach().numpy().copy()
            jobs, labels = [], []
            common = state()
            for direction in range(args.directions):
                for sign in (1, -1):
                    parameters = dict(common)
                    parameters['actor.3.weight'] = center + sign * args.sigma * noise[direction]
                    for index, task in enumerate(selected_tasks):
                        jobs.append(make_job(task, parameters)); labels.append((direction, sign, index))
            results = list(pool.map(worker_rollout, jobs))
            with (out / 'episodes.jsonl').open('a') as stream:
                for label, result in zip(labels, results):
                    stream.write(json.dumps(dict(update=update + 1, direction=label[0], sign=label[1],
                        task_in_update=label[2], **result.get('record', result))) + '\n')
            if not all(r['completed'] for r in results):
                break
            rewards = np.asarray([r['record']['reward'] for r in results]).reshape(args.directions, 2, args.tasks_per_update)
            difference = rewards[:, 0].mean(-1) - rewards[:, 1].mean(-1)
            gradient = (difference[:, None, None] * noise).mean(0) / (2 * args.sigma)
            scale = max(float(rewards.std()), .1)
            parameter.grad = -torch.from_numpy(gradient / scale).to(parameter)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            update += 1; episodes += len(results); steps += sum(r['record']['steps'] for r in results)
            row = dict(update=update, training_episodes=episodes, environment_steps=steps,
                N=len(results), success=sum(r['record']['success'] for r in results),
                collision=sum(r['record']['collision'] for r in results),
                reward_mean=float(rewards.mean()), paired_difference_abs=float(np.abs(difference).mean()),
                gradient_norm=float(np.linalg.norm(gradient)), readout_norm=float(parameter.detach().norm()))
            with (out / 'learning.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True); status()
            if update % args.validation_every == 0 or update == args.iterations:
                proceed = validate()
            else:
                save(out / 'last.pt')
        save(out / 'last.pt')
    status(update == args.iterations and proceed)


if __name__ == '__main__':
    main()
