"""Prospective contact-window action interventions with common continuation.

The unchanged frozen teacher is the reference at every root. Alternatives
replace one tick or an explicitly declared block, then use the same teacher.
Full outcomes are diagnostic records only; the saved short physical paths
can be used separately on training layouts without an outcome/value head.
"""
import argparse
import concurrent.futures
from dataclasses import asdict
import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.bridge import World, factorize, inputs
from citf.contracts import Protocol, BRANCH_EVENT_FIELDS, COLLECTION_SCHEMA, recorded_protocol
from experiments import root_response_diagnostic as diagnostic
from experiments.candidate_response import CandidateResponseModel, ResponseConfig, ResponseController, add_reference


worker = {}
ROOT_CATEGORIES = ('contact12', 'contact8', 'capture6', 'missed_window', 'recovery2')


def initialize(project, source, teacher, student, out, deadline):
    diagnostic.initialize(project, source, teacher, out, deadline)
    value = torch.load(student, map_location='cpu', weights_only=False)
    model = CandidateResponseModel(ResponseConfig(**value['config'])).eval()
    model.load_state_dict(value['state'])
    if model.cfg.candidate_domain != 'original_physical_bank' or model.cfg.deployment != 'policy':
        raise ValueError('Use the declared original-domain student for this diagnosis')
    worker.update(student=model, teacher=teacher, student_checkpoint=student,
                  out=Path(out), deadline=deadline, project=project)


def geometry(state):
    relative = state[:-1, :2] - state[-1, :2]
    distance = np.linalg.norm(relative, axis=-1)
    nearest = int(np.argmin(distance))
    closing = -np.sum(relative * (state[:-1, 3:5] - state[-1, 3:5]), axis=-1) / np.maximum(distance, 1e-6)
    i, j = np.triu_indices(len(state) - 1, 1)
    separation = np.linalg.norm(state[i, :2] - state[j, :2], axis=-1).min()
    radial = np.linalg.norm(state[:-1, :2], axis=-1) - np.linalg.norm(state[-1, :2])
    return dict(distance=float(distance[nearest]), nearest=nearest,
                closing_speed=float(closing[nearest]), radial_gap=float(radial.min()),
                separation=float(separation), defender_distances=distance.tolist(),
                defender_closing_speeds=closing.tolist(), defender_radial_gaps=radial.tolist())


@torch.no_grad()
def bank_decisions(world):
    w = diagnostic.worker
    f, plans, physical_scores, reference, _ = diagnostic.physical_bank(world, w['source'], w['cfg'])
    original, _, _ = inputs(world, f, plans)
    teacher_logits = w['model'](original)['logits'][0].numpy()
    teacher_choice = min(range(len(plans)), key=lambda k: (-float(teacher_logits[k]), tuple(plans[k])))
    physical_choice = next(i for i, p in enumerate(plans) if np.array_equal(p, reference))
    order = [physical_choice] + [k for k in range(len(plans)) if k != physical_choice]
    candidate_keys = {'nominal', 'candidate', 'unary', 'candidate_valid'}
    batch = {k: v[:, order] if k in candidate_keys else v for k, v in original.items()}
    batch = add_reference(batch)
    student_logits = worker['student'](batch)['logits'][0].numpy()
    plans = plans[order]
    teacher_logits, physical_scores = teacher_logits[order], physical_scores[order]
    teacher_choice = order.index(teacher_choice)
    student_choice = min(range(len(plans)), key=lambda k: (-float(student_logits[k]),
        int(k != 0) if worker['student'].cfg.policy_prior_mode == 'reference_saturated' else 0,
        tuple(plans[k])))
    nominal = batch['nominal'][0].numpy()[:, None]
    valid = np.broadcast_to(batch['known_time_valid'][0].numpy(), (len(plans), 1, nominal.shape[-2]))
    contact = diagnostic.response_scores(nominal, valid, batch['unary'][0].numpy(),protocol=world.protocol)['contact'][:, 0]
    contact_choice = min(range(len(plans)), key=lambda k: (float(contact[k]), tuple(plans[k])))
    return f, plans, batch, dict(physical=0, teacher=teacher_choice, student=student_choice,
        nominal_contact=contact_choice), dict(teacher=teacher_logits, student=student_logits,
                                              physical=physical_scores, nominal_contact=contact)


@torch.no_grad()
def teacher_force(world):
    w = diagnostic.worker
    return diagnostic.synchronized.joint.JointPolicyController(w['source'], w['model']).act(world)[0]


@torch.no_grad()
def collect_layout(job):
    task, index, behavior = job['task'], job['index'], job['behavior']
    w = diagnostic.worker
    world = World(worker['project'], recorded_protocol(job), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    prefix, roots, reached = [], [], set()
    minimum, previous, receding_steps, missed_time = float('inf'), float('inf'), 0, None
    while not world.done:
        if time.time() >= worker['deadline']:
            return dict(completed=False, task=task, behavior=behavior, roots=roots, reason='deadline')
        state = world.state(); elapsed = float(world.env.Current_T); geo = geometry(state)
        distance = geo['distance']; minimum = min(minimum, distance)
        receding_steps = receding_steps + 1 if distance > previous + .01 else 0
        conditions = dict(contact12=distance <= 12, contact8=distance <= 8,
            capture6=distance <= 6,
            missed_window=minimum <= 8 and distance >= minimum + .15 and receding_steps >= 2,
            recovery2=missed_time is not None and elapsed >= missed_time + 2)
        triggers = [k for k in ROOT_CATEGORIES if conditions[k] and k not in reached]
        f, plans, batch, roles, scores = bank_decisions(world)
        if triggers:
            if 'missed_window' in triggers:
                missed_time = elapsed
            selected = []
            ranked_teacher = sorted(range(len(plans)), key=lambda k: (-float(scores['teacher'][k]), tuple(plans[k])))
            ranked_student = sorted(range(len(plans)), key=lambda k: (-float(scores['student'][k]), tuple(plans[k])))
            ranked_contact = sorted(range(len(plans)), key=lambda k: (float(scores['nominal_contact'][k]), tuple(plans[k])))
            alternatives = [*roles.values(), *ranked_contact[:3], *ranked_teacher[:3], *ranked_student[:3]]
            if job.get('fill_candidates', False):
                rng = np.random.default_rng(task['scene_seed'] + len(prefix) * 31)
                alternatives.extend(rng.permutation(len(plans)).tolist())
            for k in alternatives:
                if k not in selected:
                    selected.append(k)
                if len(selected) >= job['candidates']:
                    break
            name = f'layout{index:04d}_{behavior}_step{len(prefix):04d}'
            candidate_keys = {'nominal', 'candidate', 'unary', 'candidate_valid'}
            data = {k: (v[0, selected] if k in candidate_keys else v[0]).clone() for k, v in batch.items()}
            metadata = dict(root_id=name, task=task, split=job['role'], behavior=behavior,physical_protocol=asdict(world.protocol),
                teacher_checkpoint=worker['teacher'], student_checkpoint=worker['student_checkpoint'],
                root_categories=triggers, root_step=len(prefix), elapsed_seconds=elapsed,
                prefix_forces=prefix.copy(), candidate_indices=plans[selected].tolist(),
                candidate_roles={k: selected.index(v) for k, v in roles.items()},
                scores={k: v[selected].tolist() for k, v in scores.items()},
                observed=geo, current_state=state.tolist(), observed_minimum_distance=minimum,
                full_candidate_count=len(plans), geometry_family_id=task['geometry_family_id'],
                future_seeds=[(150000000 if job['role'] == 'train' else 140000000) + index * 10000 + (0 if behavior == 'teacher' else 5000) + len(roots) * 100 + j for j in range(job['futures'])],
                source_sha256=w['source'].sha256, environment_sha256=world.environment_sha256,
                selection='prospective first contact12/contact8/capture6/receding-after-near8/recovery2; no outcome selection')
            torch.save(dict(tensors=data, metadata=metadata), worker['out'] / (name + '.pt'))
            roots.append(dict(name=name, categories=triggers, elapsed=elapsed, observed=geo,
                              candidates=len(selected)))
            reached.update(triggers)
        force = diagnostic.forces(f, plans[roles[behavior]])
        result = world.step(force); prefix.append(force.tolist()); previous = distance
        if len(reached) == len(ROOT_CATEGORIES):
            break
    return dict(completed=True, task=task, behavior=behavior, roots=roots,
                unreached=sorted(set(ROOT_CATEGORIES) - reached), task_terminated=world.done)


@torch.no_grad()
def continue_root(job):
    value = torch.load(job['root_file'], map_location='cpu', weights_only=False)
    data, meta = value['tensors'], value['metadata']; task = meta['task']
    root = World(worker['project'], recorded_protocol(meta), task['defenders'])
    root.reset(task['scene_seed'], task['future_seed'], task['agility'])
    for force in meta['prefix_forces']:
        root.step(force)
    f = factorize(root, diagnostic.worker['source'])
    plans = np.asarray(meta['candidate_indices'], dtype=np.int64)
    _, gates, rotation = inputs(root, f, plans)
    interventions = ([(-2, 0), (-3, 0)] if job.get('feedback_controls_only') else
        [(-1, 0)] + [(i, h) for h in job['hold_steps'] for i in range(len(plans))])
    records, short_paths = [], []
    for candidate, hold in interventions:
        for future_seed in meta['future_seeds']:
            if time.time() >= worker['deadline']:
                return dict(completed=False, root_id=meta['root_id'], outcomes=records, reason='deadline')
            world = root.clone(); world.rng = np.random.RandomState(future_seed)
            step, result, path, observed = 0, None, [], []
            while not world.done:
                if step % 20 == 0 and time.time() >= worker['deadline']:
                    return dict(completed=False, root_id=meta['root_id'], outcomes=records, reason='deadline in branch')
                if candidate == -2:
                    force = ResponseController(diagnostic.worker['source'], worker['student']).act(world)[0]
                elif candidate == -3:
                    force = diagnostic.physical_action(world)
                elif step < hold:
                    a, b, _ = diagnostic.worker['source'].commands(world.env)
                    force = (750 * (gates[candidate] * a + (1 - gates[candidate]) * b) + 250).numpy()
                else:
                    force = teacher_force(world)
                result = world.step(force); step += 1
                if step <= job['short_steps']:
                    path.append((world.state()[:, :2] @ rotation).copy())
                    observed.append(geometry(world.state()))
            row = dict(candidate=candidate, hold_steps=hold, future_seed=future_seed,
                continuation_policy='student' if candidate == -2 else 'physical' if candidate == -3 else 'teacher',
                steps=step, short_steps=len(path), short_terminated=step <= job['short_steps'],
                short_min_distance=min(x['distance'] for x in observed),
                short_max_radial_gap=max(x['radial_gap'] for x in observed),
                short_min_separation=min(x['separation'] for x in observed), **result)
            records.append(row); short_paths.append(torch.tensor(np.stack(path, axis=1), dtype=torch.float32))
    output = dict(root_id=meta['root_id'], root_file=job['root_file'], metadata=meta,
                  outcomes=records, short_paths=short_paths, short_horizon=job['short_steps'])
    torch.save(output, worker['out'] / (meta['root_id'] + '.pt'))
    return dict(completed=True, root_id=meta['root_id'], root_file=job['root_file'],
                task=task, behavior=meta['behavior'], categories=meta['root_categories'],
                candidate_roles=meta['candidate_roles'], observed=meta['observed'], outcomes=records)


@torch.no_grad()
def short_root(job):
    value = torch.load(job['root_file'], map_location='cpu', weights_only=False)
    meta = value['metadata']; task = meta['task']
    world = World(worker['project'], recorded_protocol(meta), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    for force in meta['prefix_forces']:
        world.step(force)
    f = factorize(world, diagnostic.worker['source'])
    plans = np.asarray(meta['candidate_indices'], dtype=np.int64)
    batch, gates, rotation = inputs(world, f, plans)
    count, futures, entities = len(plans), len(meta['future_seeds']), task['defenders'] + 1
    horizon = world.protocol.prediction_steps
    paths = torch.zeros(count, futures, entities, horizon, 2)
    valid = torch.zeros(count, futures, horizon, dtype=torch.bool)
    attacker_actions = torch.zeros(count, futures, horizon, 2)
    stop = torch.zeros(count, futures, dtype=torch.long)
    outcomes = torch.zeros(count, futures, dtype=torch.long)
    terminated = torch.zeros(count, futures, dtype=torch.bool)
    stop_time = torch.zeros(count, futures)
    events = torch.zeros(count, futures, len(BRANCH_EVENT_FIELDS), dtype=torch.bool)
    transitions = 0
    for candidate in range(count):
        for future, seed in enumerate(meta['future_seeds']):
            branch = world.clone(); branch.rng = np.random.RandomState(seed)
            for tick in range(horizon):
                if time.time() >= worker['deadline']:
                    return dict(completed=False, root_id=meta['root_id'], reason='short collection deadline')
                if tick == 0:
                    a, b, _ = diagnostic.worker['source'].commands(branch.env)
                    force = (750 * (gates[candidate] * a + (1 - gates[candidate]) * b) + 250).numpy()
                else:
                    force = teacher_force(branch)
                result = branch.step(force); transitions += 1
                paths[candidate, future, :, tick] = torch.as_tensor(branch.state()[:, :2] @ rotation, dtype=torch.float32)
                attacker_actions[candidate, future, tick] = torch.as_tensor(branch.env.att_action)
                valid[candidate, future, tick] = True
                if branch.done:
                    break
            stop[candidate, future] = tick + 1; outcomes[candidate, future] = result['outcome']
            terminated[candidate, future] = branch.done; stop_time[candidate, future] = result['duration']
            events[candidate, future] = torch.tensor([bool(result[k]) for k in BRANCH_EVENT_FIELDS])
    tensors = {k: v[0].clone() for k, v in batch.items()}
    tensors.update(real_positions=paths, time_valid=valid, executed_attacker_actions=attacker_actions,
        branch_stop_step=stop, branch_outcome=outcomes, branch_terminated=terminated,
        branch_prediction_truncated=~terminated, branch_stop_time=stop_time, branch_events=events)
    metadata = dict(meta, collection_schema=COLLECTION_SCHEMA, branch_event_fields=BRANCH_EVENT_FIELDS,
        source_implementation_sha256=diagnostic.worker['source'].implementation_sha256,
        response_hold_steps=1, response_label_horizon=horizon,
        recourse_policy='frozen original7101 teacher, recomputed each control tick',
        prediction_condition='current candidate one tick, frozen teacher feedback until known horizon or first event',
        complete_task_outcomes_used_for_training=False,
        stopping_target='observed first event inclusive; terminal endpoint absorbing')
    torch.save(dict(tensors=tensors, metadata=metadata), worker['out'] / (meta['root_id'] + '.pt'))
    return dict(completed=True, root_id=meta['root_id'], task=task, split=meta['split'],
                categories=meta['root_categories'], branch_transitions=transitions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['collect', 'continue', 'short'])
    for key in ['project', 'source', 'teacher', 'student', 'jobs', 'out', 'deadline']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    supplied = json.loads(Path(args.jobs).read_text())
    if args.mode == 'collect':
        if supplied['role'] not in ('train', 'validation'):
            raise ValueError('Capture-window roots require development training or validation layouts')
        jobs = [dict(index=i, task=t, behavior=b, role=supplied['role'],physical_protocol=asdict(recorded_protocol(supplied)),
                     candidates=supplied.get('candidates', 6), futures=supplied.get('futures', 2),
                     fill_candidates=supplied.get('fill_candidates', False))
                for i, t in enumerate(supplied['tasks']) for b in ('teacher', 'student')]
    else:
        jobs = supplied['jobs']
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    status = dict(completed=False, arguments=vars(args), expected_jobs=len(jobs), finished_jobs=0,
                  jobs=[dict(index=i, status='pending') for i in range(len(jobs))])
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    function = {'collect': collect_layout, 'continue': continue_root, 'short': short_root}[args.mode]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, args.teacher, args.student, args.out, deadline)) as pool:
        pending = {pool.submit(function, job): i for i, job in enumerate(jobs)}
        for future in concurrent.futures.as_completed(pending):
            i = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(completed=False, job=jobs[i], error=repr(error))
            with (out / 'records.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            status['jobs'][i].update(status='completed' if result['completed'] else 'incomplete',
                roots=len(result.get('roots', [])), detail=result.get('error', result.get('reason')))
            status['finished_jobs'] += 1
            (out / 'status.json').write_text(json.dumps(status, indent=2))
            print(json.dumps(dict(job=i, completed=result['completed'], finished_jobs=status['finished_jobs'],
                                 expected_jobs=len(jobs), error=result.get('error'))), flush=True)
    status.update(completed=all(j['status'] == 'completed' for j in status['jobs']),
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
