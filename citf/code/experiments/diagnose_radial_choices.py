"""Observed short branches and independent-future action diagnostics.

These outcome-selected development cases diagnose decision mechanisms. They
do not train a value model or estimate the complete-bank success rate. Held
coefficient labels and one-action/learned-replanning continuations are kept
as different physical conditions, with disjoint future random seeds.
"""
import argparse
import concurrent.futures
from dataclasses import replace
import datetime
import gzip
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.bridge import Source, World, factorize, inputs
from citf.collect import sample_root
from citf.contracts import Protocol
from experiments import joint_configuration_policy as joint
from experiments.reference_bank_configuration_policy import (
    ReferenceBankConfig, ReferenceBankPolicy, ReferenceBankController)


worker = {}


def initialize(project, source, checkpoint, composed_checkpoint, out, deadline):
    torch.set_num_threads(1)
    value = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = joint.JointConfigurationPolicy(joint.JointPolicyConfig(**value['config']))
    model.load_state_dict(value['state']); model.eval()
    alternative = torch.load(composed_checkpoint, map_location='cpu', weights_only=False)
    composed = ReferenceBankPolicy(ReferenceBankConfig(**alternative['config']))
    composed.load_state_dict(alternative['state']); composed.eval()
    worker.update(project=project, source=Source(project, source), model=model,
                  composed=composed, out=Path(out), deadline=deadline)


@torch.no_grad()
def diagnose(job):
    if time.time() >= worker['deadline']:
        return dict(completed=False, job=job, reason='deadline before root')
    with gzip.open(job['trace'], 'rt') as stream:
        trace = [json.loads(line) for line in stream]
    task, index = job['task'], job['root_step']
    model, source = worker['model'], worker['source']
    protocol = replace(Protocol(**job['protocol']), risk_weight=model.cfg.objective_risk_weight)
    world = World(worker['project'], protocol, task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    for row in trace[:index]:
        world.step(row['decision']['physical_thrust'])
    observed = trace[index]
    reconstruction_error = float(np.max(np.abs(world.state() - np.asarray(observed['state_before']))))
    if world.done or reconstruction_error > 1e-6:
        return dict(completed=False, job=job, reason='recorded root not reconstructed',
                    state_max_abs_error=reconstruction_error)
    f = factorize(world, source)
    original_plans = np.asarray(observed['decision']['configurations'], dtype=np.int64)
    batch, _, _ = inputs(world, f, original_plans)
    prediction = model.encoder(batch)
    geometric_scores = joint.policy.geometry.costs(prediction, batch, model.cfg.clearance)[0]
    geometric_choice = original_plans[int(geometric_scores.argmin())].tolist()
    composed_controller = ReferenceBankController(source, worker['composed'], budget=160)
    _, composed_trace = composed_controller.act(world)
    roles = dict(learned=observed['decision']['selected_indices'],
                 physical=observed['decision']['physical_selected_indices'],
                 source_center=observed['decision']['source_center_indices'],
                 geometry=geometric_choice,
                 composed=composed_trace['selected_indices'])
    plans = np.asarray(list(dict.fromkeys(tuple(value) for value in roles.values())), dtype=np.int64)
    role_indices = {name: next(i for i, row in enumerate(plans) if row.tolist() == value)
                    for name, value in roles.items()}
    short_seeds = [91000000 + job['index'] * 100 + i for i in range(3)]
    evaluation_seeds = [92000000 + job['index'] * 100 + i for i in range(3)]
    item, metadata, branch_steps = sample_root(world, source, plans, short_seeds,
        dict(task=task, diagnostic_group=job['group'], root_step=index,
             root_id=job['name'], candidate_indices=plans.tolist(),
             candidate_roles=role_indices, selection_uses_future_labels=False,
             behavior_identity=dict(method='recorded joint7101 scale2 greedy trajectory')))
    observed_batch = {k: v[None] for k, v in item.items()}
    output = model.encoder(observed_batch)
    loss = joint.policy.geometry.task_objective(output, observed_batch,
                                               clearance=model.cfg.clearance)
    distance, radial, collision, pursuit = joint.policy.geometry.targets(observed_batch)
    actual_guard = (radial + model.cfg.guard_margin).clamp_min(0).square().mean(-1)[0]
    predicted = dict(team_mean=output['team_mu'][0].tolist(),
                     team_sigma=output['team_sigma'][0].tolist(),
                     team_risk=output['team_risk'][0].tolist(),
                     actual_team_mean=radial.mean(-1)[0].tolist(),
                     actual_team_risk=actual_guard.tolist(),
                     empirical_geometry_cost=loss['empirical_cost'][0].tolist(),
                     estimated_geometry_cost=loss['estimated_cost'][0].tolist())
    torch.save(dict(tensors=item, metadata=metadata), worker['out'] / (job['name'] + '.pt'))
    controller = joint.JointPolicyController(source, model)
    _, gates, _ = inputs(world, f, plans)
    outcomes = []
    for candidate, gate in enumerate(gates):
        for seed in evaluation_seeds:
            if time.time() >= worker['deadline']:
                return dict(completed=False, job=job, reason='deadline during independent continuation',
                            short_prediction=predicted, continuation_outcomes=outcomes)
            branch = world.clone(); branch.rng = np.random.RandomState(seed)
            force = (750 * (gate * f['learned'] + (1 - gate) * f['rules']) + 250).numpy()
            result = branch.step(force)
            steps = 1
            while not branch.done:
                if steps % 25 == 0 and time.time() >= worker['deadline']:
                    return dict(completed=False, job=job, reason='deadline in continuation',
                                short_prediction=predicted, continuation_outcomes=outcomes)
                force, _ = controller.act(branch)
                result = branch.step(force); steps += 1
            outcomes.append(dict(candidate=candidate, future_seed=seed,
                                 continuation_steps=steps, **result))
    return dict(completed=True, job=job, state_max_abs_error=reconstruction_error,
                candidate_roles=role_indices, candidates=plans.tolist(),
                candidate_in_original_bank=[row.tolist() in original_plans.tolist() for row in plans],
                short_condition='held fusion coefficients with source feedback until stop or horizon',
                continuation_condition='one current action, then unchanged greedy joint7101 policy until actual stop',
                short_future_seeds=short_seeds, continuation_future_seeds=evaluation_seeds,
                short_branch_transitions=branch_steps, short_prediction=predicted,
                continuation_outcomes=outcomes)


def main():
    parser = argparse.ArgumentParser()
    for key in ('project', 'source', 'checkpoint', 'composed-checkpoint', 'jobs', 'out', 'deadline'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    jobs = json.loads(Path(args.jobs).read_text())['jobs']
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    status = dict(completed=False, arguments=vars(args),
                  scope='outcome-selected development diagnostics, not complete-bank performance',
                  jobs=[dict(name=job['name'], status='pending') for job in jobs])
    (out / 'protocol.json').write_text(json.dumps(status, indent=2))
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, args.checkpoint, args.composed_checkpoint,
                      args.out, deadline)) as pool:
        futures = {pool.submit(diagnose, job): i for i, job in enumerate(jobs)}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(completed=False, job=jobs[index], error=repr(error))
            results.append(result)
            with (out / 'roots.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            status['jobs'][index]['status'] = 'completed' if result['completed'] else 'incomplete'
            (out / 'status.json').write_text(json.dumps(status, indent=2))
            print(json.dumps(dict(root=jobs[index]['name'], completed=result['completed'],
                                  roots_completed=sum(x['completed'] for x in results))), flush=True)
    status['completed'] = len(results) == len(jobs) and all(x['completed'] for x in results)
    status['completed_roots'] = sum(x['completed'] for x in results)
    status['total_roots'] = len(jobs)
    (out / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
