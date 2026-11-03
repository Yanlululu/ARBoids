"""Fixed-reference response data and a candidate bank independent of scoring order.

Every sample preserves the strong physical choice at index zero. Candidates
act for the declared number of ticks, followed by common physical feedback.
Training uses short physical trajectories only, never complete-task returns.
"""
import argparse
import concurrent.futures
import datetime
import itertools
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.bridge import World, factorize, inputs
from citf.contracts import Protocol, BRANCH_EVENT_FIELDS, COLLECTION_SCHEMA
from experiments import root_response_diagnostic as diagnostic


def candidate_bank(world, source, cfg):
    f, original, values, reference, center = diagnostic.physical_bank(world, source, cfg)
    plans = [tuple(map(int, reference))]
    for row in original:
        key = tuple(map(int, row))
        if key not in plans:
            plans.append(key)
    original_count = len(plans)
    state = world.state()
    nearest = list(np.argsort(np.linalg.norm(state[:-1, :2] - state[-1, :2], axis=-1))[:4])
    inner = int(np.argmin(np.linalg.norm(state[:-1, :2], axis=-1)))
    if inner not in nearest:
        nearest[-1] = inner
    additions = []
    for i, j in itertools.combinations(nearest, 2):
        proposals = []
        for ci in f['valid'][i].nonzero()[:, 0].tolist():
            for cj in f['valid'][j].nonzero()[:, 0].tolist():
                if ci == reference[i] or cj == reference[j]:
                    continue
                row = reference.copy(); row[i], row[j] = ci, cj
                if tuple(row) not in plans:
                    proposals.append(row)
        if not proposals:
            continue
        b, _, _ = inputs(world, f, np.asarray(proposals))
        paths = b['nominal'][0].numpy()[:, None]
        valid = np.broadcast_to(b['known_time_valid'][0].numpy(),
                                (len(proposals), 1, paths.shape[-2]))
        score = diagnostic.response_scores(paths, valid, b['unary'][0].numpy())['contact'][:, 0]
        order = sorted(range(len(proposals)), key=lambda k: (float(score[k]), tuple(proposals[k])))
        key = tuple(map(int, proposals[order[0]]))
        if key not in plans:
            plans.append(key); additions.append(key)
    return f, np.asarray(plans, dtype=np.int64), dict(
        reference_indices=reference.tolist(), source_center_indices=center.tolist(),
        original_count=original_count, joint_additions=[list(x) for x in additions])


def initialize(project, source, checkpoint, out, deadline, role, futures, candidates, response_hold):
    diagnostic.initialize(project, source, checkpoint, out, deadline)
    diagnostic.worker.update(role=role, futures=futures, candidates=candidates, response_hold=response_hold)


def sample_response(world, plans, future_seeds, metadata):
    """Current configuration then the same known short physical recourse policy."""
    w = diagnostic.worker
    f = factorize(world, w['source']); b, gates, rotation = inputs(world, f, plans)
    count, defenders = gates.shape[:2]
    horizon = world.protocol.prediction_steps; futures = len(future_seeds)
    xy = torch.zeros(count, futures, defenders + 1, horizon, 2)
    valid = torch.zeros(count, futures, horizon, dtype=torch.bool)
    actions = torch.zeros(count, futures, horizon, 2)
    stop = torch.zeros(count, futures, dtype=torch.long)
    outcomes = torch.zeros_like(stop)
    terminated = torch.zeros_like(stop, dtype=torch.bool)
    stop_time = torch.zeros(count, futures, dtype=torch.float64)
    events = torch.zeros(count, futures, len(BRANCH_EVENT_FIELDS), dtype=torch.bool)
    cost = 0
    for c in range(count):
        for r, seed in enumerate(future_seeds):
            branch = world.clone(); branch.rng = np.random.RandomState(seed)
            for tick in range(horizon):
                if time.time() >= w['deadline']:
                    raise InterruptedError('Collection deadline before complete response root')
                if tick < w['response_hold']:
                    a, rule, _ = w['source'].commands(branch.env)
                    force = (750 * (gates[c] * a + (1 - gates[c]) * rule) + 250).numpy()
                else:
                    force = diagnostic.physical_action(branch)
                result = branch.step(force); cost += 1
                xy[c, r, :, tick] = torch.as_tensor(branch.state()[:, :2] @ rotation, dtype=torch.float32)
                actions[c, r, tick] = torch.as_tensor(branch.env.att_action)
                valid[c, r, tick] = True
                if branch.done:
                    break
            stop[c, r] = tick + 1; outcomes[c, r] = result['outcome']
            terminated[c, r] = branch.done; stop_time[c, r] = result['duration']
            events[c, r] = torch.tensor([bool(result[k]) for k in BRANCH_EVENT_FIELDS])
    data = {k: v[0].clone() for k, v in b.items()}
    data.update(real_positions=xy, time_valid=valid, executed_attacker_actions=actions,
                branch_stop_step=stop, branch_outcome=outcomes, branch_terminated=terminated,
                branch_prediction_truncated=~terminated, branch_stop_time=stop_time, branch_events=events)
    metadata.update(collection_schema=COLLECTION_SCHEMA, branch_event_fields=BRANCH_EVENT_FIELDS,
        future_seeds=future_seeds, source_sha256=w['source'].sha256,
        source_implementation_sha256=w['source'].implementation_sha256,
        environment_sha256=world.environment_sha256, geometry_identity=world.initial_identity,
        geometry_family_id=world.initial_identity['fingerprint'],
        elapsed_seconds=float(world.env.Current_T), response_hold_steps=w['response_hold'],
        recourse_policy='unchanged synchronized physical, recomputed every control tick',
        prediction_condition='current candidate for declared ticks, common physical recourse to 16 ticks or first task event',
        stopping_target='observed first event inclusive; terminal endpoint is an explicitly absorbing label')
    return data, metadata, cost


@torch.no_grad()
def collect_layout(job):
    w = diagnostic.worker
    task, index = job['task'], job['index']
    world = World(w['project'], Protocol(), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    controller = diagnostic.synchronized.joint.JointPolicyController(w['source'], w['model'])
    use_learned = bool(index % 2)
    reached, roots, prefix = set(), [], []
    requested = {'initial', 'normal4', 'normal10', 'contact12', 'contact8', 'radial6'}
    rng = np.random.default_rng(task['scene_seed'] + 451)
    while not world.done:
        if time.time() >= w['deadline']:
            return dict(completed=False, task=task, roots=roots, reason='deadline')
        state = world.state(); elapsed = float(world.env.Current_T)
        distance = float(np.linalg.norm(state[:-1, :2] - state[-1, :2], axis=-1).min())
        gap = float(np.linalg.norm(state[:-1, :2], axis=-1).min() - np.linalg.norm(state[-1, :2]))
        conditions = dict(initial=len(prefix) == 0, normal4=elapsed >= 4,
                          normal10=elapsed >= 10, contact12=distance <= 12,
                          contact8=distance <= 8, radial6=elapsed >= 4 and gap >= -6)
        triggers = [k for k, yes in conditions.items() if yes and k not in reached]
        if triggers:
            f, bank, detail = candidate_bank(world, w['source'], w['cfg'])
            b, _, _ = inputs(world, f, bank)
            scores = diagnostic.synchronized.physical_components(b, w['cfg'])[0][0].numpy()
            # Reference, the strongest nominal alternatives, all pair proposals,
            # then random physical-bank candidates cover response scale diversity.
            chosen = [0]
            for k in np.argsort(scores)[:4]:
                if int(k) not in chosen:
                    chosen.append(int(k))
            for k in range(detail['original_count'], len(bank)):
                if k not in chosen:
                    chosen.append(k)
            for k in rng.permutation(len(bank)):
                if len(chosen) >= w['candidates']:
                    break
                if int(k) not in chosen:
                    chosen.append(int(k))
            if len(chosen) < w['candidates']:
                raise ValueError('Response candidate count exceeds available unique configurations')
            chosen = chosen[:w['candidates']]
            plans = bank[chosen]
            name = f'layout{index:04d}_step{len(prefix):04d}'
            base = 110000000 if w['role'] == 'train' else 120000000
            futures = [base + index * 10000 + len(roots) * 100 + k for k in range(w['futures'])]
            metadata = dict(root_id=name, task=task, split=w['role'], scene_seed=task['scene_seed'],
                root_step=len(prefix), root_categories=triggers, prefix_forces=prefix.copy(),
                observed_nearest=distance, observed_radial_gap=gap,
                behavior_identity=dict(method='synchronized_joint7101' if use_learned else 'synchronized_physical'),
                prediction_condition='fixed coefficients with source feedback for 16 ticks, terminated at first task event',
                reference_candidate_index=0, candidate_indices=plans.tolist(),
                candidate_bank_indices=chosen, full_bank=bank.tolist(), **detail)
            tensors, metadata, count = sample_response(world, plans, futures, metadata)
            torch.save(dict(tensors=tensors, metadata=metadata), w['out'] / (name + '.pt'))
            roots.append(dict(name=name, categories=triggers, elapsed=elapsed, branch_transitions=count))
            reached.update(triggers)
        if reached == requested:
            break
        force = controller.act(world)[0] if use_learned else diagnostic.physical_action(world)
        world.step(force); prefix.append(force.tolist())
    return dict(completed=True, task=task, roots=roots, unreached=sorted(requested - reached),
                task_terminated=world.done, behavior='learned' if use_learned else 'physical')


def main():
    parser = argparse.ArgumentParser()
    for key in ['project', 'source', 'checkpoint', 'tasks', 'out', 'deadline']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--futures', type=int, default=4)
    parser.add_argument('--candidates', type=int, default=16)
    parser.add_argument('--response-hold', type=int, default=1)
    args = parser.parse_args()
    bank = json.loads(Path(args.tasks).read_text())
    if bank['role'] not in ('train', 'validation'):
        raise ValueError('Only declared development train and validation layouts may be sampled')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    jobs = [dict(index=i, task=t) for i, t in enumerate(bank['tasks'])]
    status = dict(completed=False, arguments=vars(args), jobs=[dict(index=i, status='pending') for i in range(len(jobs))])
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, args.checkpoint, args.out, deadline,
                      bank['role'], args.futures, args.candidates, args.response_hold)) as pool:
        pending = {pool.submit(collect_layout, job): i for i, job in enumerate(jobs)}
        for future in concurrent.futures.as_completed(pending):
            i = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(completed=False, task=jobs[i]['task'], error=repr(error))
            with (out / 'records.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            status['jobs'][i].update(status='completed' if result['completed'] else 'incomplete',
                                     roots=len(result.get('roots', [])), detail=result.get('error', result.get('reason')))
            (out / 'status.json').write_text(json.dumps(status, indent=2))
            print(json.dumps(dict(layout=i, **status['jobs'][i])), flush=True)
    status.update(completed=all(x['status'] == 'completed' for x in status['jobs']),
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
