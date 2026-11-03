"""Paired short-response experiments at prospectively declared defense states.

Future samples are labels only.  The first and second halves of the recorded
futures are disjoint selection and replication samples.  Complete task
continuations are diagnostics, never targets for a value or outcome network.
"""
import argparse
import concurrent.futures
from dataclasses import asdict, replace
import datetime
import itertools
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from citf.bridge import World, Source, factorize, inputs
from citf.collect import sample_root
from citf.contracts import Protocol
from citf.search import nominal_beam, initialize_bank, whole_configuration_search
from experiments import synchronized_configuration_policy as synchronized


worker = {}


def curves(paths):
    """[candidate,future,entity,time,xy] -> synchronized physical distances."""
    radius = np.linalg.norm(paths, axis=-1)
    radial = (radius[..., :-1, :] - radius[..., -1:, :]).min(-2)
    capture = np.linalg.norm(paths[..., :-1, :, :] - paths[..., -1:, :, :], axis=-1).min(-2)
    n = paths.shape[-3] - 1
    i, j = np.triu_indices(n, 1)
    separation = np.linalg.norm(paths[..., i, :, :] - paths[..., j, :, :], axis=-1).min(-2)
    return radial, capture, separation, radius[..., -1, :]


def response_scores(paths, valid, unary, steps=None):
    """Three declared geometric objects, all through the first physical event."""
    if steps is not None:
        paths, valid = paths[..., :steps, :], valid[..., :steps]
    radial, distance, separation, attack_radius = curves(paths)
    length = valid.shape[-1]
    tick = np.arange(length)
    loss = ((radial > 0) | (separation < 5) | (attack_radius < 15)) & valid
    captured = (distance < 5) & valid
    first_loss = np.where(loss, tick, length + 1).min(-1)
    first_capture = np.where(captured, tick, length + 1).min(-1)
    lose = (first_loss <= first_capture) & (first_loss <= length)
    win = (first_capture < first_loss) & (first_capture <= length)
    first = np.minimum(first_loss, first_capture)
    observed = valid & (tick <= first[..., None])
    worst_gap = np.where(observed, radial, -np.inf).max(-1)
    closest = np.where(observed, distance, np.inf).min(-1)
    n = paths.shape[-3] - 1
    i, j = np.triu_indices(n, 1)
    pair_distance = np.linalg.norm(paths[..., i, :, :] - paths[..., j, :, :], axis=-1)
    pair_min = np.where(observed[..., None, :], pair_distance, np.inf).min(-1)
    collision_cost = np.maximum(7 - pair_min, 0).square().sum(-1) if hasattr(pair_min, 'square') else (np.maximum(7 - pair_min, 0) ** 2).sum(-1)
    guard = unary[:, None] + 64 * collision_cost + 16 * np.maximum(worst_gap + 6, 0) ** 2
    event = 1000 * (lose.astype(float) - win.astype(float))
    # Continuous evidence only in the neighborhood of a real short event.
    # Capture is evaluated before first loss, retaining its temporal order.
    capture_potential = np.exp(-np.maximum(closest - 5, 0) / 2)
    coverage_barrier = np.exp(np.clip(worst_gap / 2, -40, 8))
    separation_barrier = np.exp(np.clip((5 - np.where(observed, separation, np.inf).min(-1)) / 1.5, -40, 8))
    ordered = event + .01 * np.tanh(guard / 100)
    contact = event - 5 * capture_potential + 5 * coverage_barrier + 5 * separation_barrier
    return dict(guard=guard, ordered=ordered, contact=contact,
                short_loss=lose, short_capture=win, worst_gap=worst_gap,
                closest_distance=closest, first_loss=first_loss, first_capture=first_capture)


def physical_bank(world, source, cfg, budget=96):
    f = factorize(world, source)
    ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), cfg.objective_risk_weight, 128)
    initial, center = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                      f['prior'].numpy(), 12, budget)
    def score(plans):
        b, _, _ = inputs(world, f, plans)
        return synchronized.physical_components(b, cfg)[0][0].numpy()
    choice, trace = whole_configuration_search(initial, score,
        [x.nonzero()[:, 0].numpy() for x in f['valid']], budget, 1)
    return f, np.asarray(trace['configurations'], dtype=np.int64), np.asarray(trace['scores']), np.asarray(choice), np.asarray(center)


def forces(f, indices):
    gate = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
    return (750 * (gate * f['learned'] + (1 - gate) * f['rules']) + 250).numpy()


def physical_action(world):
    f, _, _, choice, _ = physical_bank(world, worker['source'], worker['cfg'])
    return forces(f, choice)


@torch.no_grad()
def diagnostic_candidates(world):
    f, bank, values, reference, center = physical_bank(world, worker['source'], worker['cfg'])
    b, _, _ = inputs(world, f, bank)
    logits = worker['model'](b)['logits'][0].numpy()
    learned = bank[int(np.argmax(logits))]
    plans, roles = [], {}
    def add(plan, role=None):
        key = tuple(map(int, plan))
        if key not in plans:
            plans.append(key)
        if role:
            roles[role] = plans.index(key)
    add(reference, 'physical'); add(learned, 'learned'); add(center, 'source_center')
    for index in np.argsort(values):
        if len(plans) >= 6:
            break
        add(bank[index])
    state = world.state()
    nearest = list(np.argsort(np.linalg.norm(state[:-1, :2] - state[-1, :2], axis=-1))[:4])
    inner = int(np.argmin(np.linalg.norm(state[:-1, :2], axis=-1)))
    if inner not in nearest:
        nearest[-1] = inner
    original = set(map(tuple, bank.tolist()))
    alternatives = []
    for i, j in itertools.combinations(nearest, 2):
        proposals = []
        for ci in f['valid'][i].nonzero()[:, 0].tolist():
            for cj in f['valid'][j].nonzero()[:, 0].tolist():
                if ci == reference[i] or cj == reference[j]:
                    continue
                row = reference.copy(); row[i], row[j] = ci, cj
                if tuple(row) not in original:
                    proposals.append(row)
        if not proposals:
            continue
        pb, _, _ = inputs(world, f, np.asarray(proposals))
        nominal = pb['nominal'][0].numpy()[:, None]
        known = np.broadcast_to(pb['known_time_valid'][0].numpy(), (len(proposals), 1, nominal.shape[-2]))
        scores = response_scores(nominal, known, pb['unary'][0].numpy())
        order = np.argsort(scores['contact'][:, 0] + .02 * pb['unary'][0].numpy())
        alternatives.append((float(scores['contact'][order[0], 0]), proposals[int(order[0])]))
    for _, plan in sorted(alternatives, key=lambda x: x[0]):
        if len(plans) >= 12:
            break
        add(plan)
    rng = np.random.default_rng(int(world.env.Current_T * 5) + worker['scene_seed'] * 7)
    while len(plans) < 12:
        row = reference.copy()
        agents = rng.choice(len(row), 2, replace=False)
        for i in agents:
            row[i] = rng.choice(f['valid'][i].nonzero()[:, 0].numpy())
        add(row)
    plans = np.asarray(plans, dtype=np.int64)
    rb, _, _ = inputs(world, f, plans)
    metadata = dict(candidate_roles=roles, candidate_indices=plans.tolist(),
        candidate_in_original_bank=[tuple(row) in original for row in plans],
        full_bank_configurations=bank.tolist(), full_bank_scores=values.tolist(),
        physical_choice=reference.tolist(), learned_choice=learned.tolist(),
        nominal_score=synchronized.physical_components(rb, worker['cfg'])[0][0].tolist())
    return plans, metadata


def initialize(project, source, checkpoint, out, deadline):
    torch.set_num_threads(1)
    synchronized.install_semantics()
    value = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = synchronized.SynchronizedPolicyConfig(**value['config'])
    model = synchronized.joint.JointConfigurationPolicy(cfg)
    model.load_state_dict(value['state']); model.eval()
    worker.update(project=project, source=Source(project, source), cfg=cfg, model=model,
                  out=Path(out), deadline=deadline)


@torch.no_grad()
def collect_layout(job):
    task, index = job['task'], job['index']
    worker['scene_seed'] = task['scene_seed']
    world = World(worker['project'], Protocol(), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    use_learned = bool(index % 2)
    controller = synchronized.joint.JointPolicyController(worker['source'], worker['model'])
    prefix = []; reached = set(); roots = []
    while not world.done:
        if time.time() >= worker['deadline']:
            return dict(completed=False, task=task, roots=roots, reason='deadline')
        state = world.state(); elapsed = float(world.env.Current_T)
        closest = float(np.linalg.norm(state[:-1, :2] - state[-1, :2], axis=-1).min())
        gap = float(np.linalg.norm(state[:-1, :2], axis=-1).min() - np.linalg.norm(state[-1, :2]))
        triggers = []
        if elapsed >= 4 and 'normal4' not in reached:
            triggers.append('normal4')
        if closest <= 12 and 'contact12' not in reached:
            triggers.append('contact12')
        if elapsed >= 4 and (closest <= 8 or gap >= -6) and 'critical' not in reached:
            triggers.append('critical')
        if triggers:
            name = f"layout{index:04d}_step{len(prefix):04d}"
            plans, details = diagnostic_candidates(world)
            future = [97000000 + index * 10000 + len(roots) * 100 + i for i in range(8)]
            metadata = dict(root_id=name, task=task, split='validation',
                scene_seed=task['scene_seed'], root_step=len(prefix), root_categories=triggers,
                prefix_forces=prefix.copy(), observed_nearest=closest, observed_radial_gap=gap,
                behavior_identity=dict(method='synchronized_joint7101' if use_learned else 'synchronized_physical'),
                prediction_condition='held fusion coefficients with SAC/Boids feedback',
                selection_future_indices=list(range(4)), replication_future_indices=list(range(4, 8)), **details)
            tensors, metadata, count = sample_root(world, worker['source'], plans, future, metadata)
            torch.save(dict(tensors=tensors, metadata=metadata), worker['out'] / (name + '.pt'))
            roots.append(dict(name=name, categories=triggers, elapsed=elapsed,
                              branch_transitions=count, nearest=closest, radial_gap=gap))
            reached.update(triggers)
        if len(reached) == 3:
            break
        force = controller.act(world)[0] if use_learned else physical_action(world)
        result = world.step(force); prefix.append(force.tolist())
    return dict(completed=True, task=task, behavior='learned' if use_learned else 'physical',
                roots=roots, unreached=sorted({'normal4', 'contact12', 'critical'} - reached),
                terminated_before_all_roots=world.done)


@torch.no_grad()
def continue_root(job):
    value = torch.load(job['root_file'], map_location='cpu', weights_only=False)
    data, meta = value['tensors'], value['metadata']; task = meta['task']
    root = World(worker['project'], Protocol(), task['defenders'])
    root.reset(task['scene_seed'], task['future_seed'], task['agility'])
    for force in meta['prefix_forces']:
        root.step(force)
    f = factorize(root, worker['source'])
    plans = np.asarray(meta['candidate_indices'], dtype=np.int64)
    _, gates, _ = inputs(root, f, plans)
    results = []
    for candidate in job['candidates']:
        for seed in job['future_seeds']:
            if time.time() >= worker['deadline']:
                return dict(completed=False, job=job, outcomes=results, reason='deadline')
            world = root.clone(); world.rng = np.random.RandomState(seed)
            step = 0; result = None
            while not world.done:
                if step % 20 == 0 and time.time() >= worker['deadline']:
                    return dict(completed=False, job=job, outcomes=results, reason='deadline in continuation')
                if step < job['hold_steps']:
                    a, b, _ = worker['source'].commands(world.env)
                    force = (750 * (gates[candidate] * a + (1 - gates[candidate]) * b) + 250).numpy()
                else:
                    force = physical_action(world)
                result = world.step(force); step += 1
            results.append(dict(candidate=candidate, future_seed=seed, steps=step, **result))
    return dict(completed=True, job=job, outcomes=results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['collect', 'continue'])
    for key in ['project', 'source', 'checkpoint', 'jobs', 'out', 'deadline']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    supplied = json.loads(Path(args.jobs).read_text())
    jobs = [dict(index=i, task=t) for i, t in enumerate(supplied['tasks'])] if args.mode == 'collect' else supplied['jobs']
    state = dict(completed=False, arguments=vars(args), jobs=[dict(index=i, status='pending') for i in range(len(jobs))])
    (out / 'status.json').write_text(json.dumps(state, indent=2))
    fn = collect_layout if args.mode == 'collect' else continue_root
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, args.checkpoint, args.out, deadline)) as pool:
        pending = {pool.submit(fn, job): i for i, job in enumerate(jobs)}
        for future in concurrent.futures.as_completed(pending):
            i = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(completed=False, error=repr(error), job=jobs[i])
            with (out / 'records.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
            state['jobs'][i]['status'] = 'completed' if result['completed'] else 'incomplete'
            state['jobs'][i]['detail'] = result.get('error', result.get('reason'))
            (out / 'status.json').write_text(json.dumps(state, indent=2))
            print(json.dumps(dict(job=i, completed=result['completed'], roots=len(result.get('roots', [])), error=result.get('error'))), flush=True)
    state['completed'] = all(j['status'] == 'completed' for j in state['jobs'])
    state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'status.json').write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
