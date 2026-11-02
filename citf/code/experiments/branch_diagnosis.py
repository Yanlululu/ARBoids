"""Development-only comparison on one fixed pool of complete configurations.

The real branches are labels, never an online controller. The deterministic
feedback reference uses observable initial velocity, zero future current and
the registered opponent. It is distinct from the held-thrust CITF-Phys.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from citf.bridge import Controller, Source, World, factorize, inputs
from citf.collect import sample_root
from citf.contracts import (LayoutRegistry, Protocol, canonical_hash, file_hash,
                            tree_hash)
from citf.evaluate import atomic_json
from citf.opponents import load_opponent
from citf.risk import branch_minima, pair_indices
from citf.search import (initialize_bank, nominal_beam, nominal_score,
                         whole_configuration_search)
from citf.train import atomic_save


REVISION = 'citf-branch-diagnosis-v2'
GROUPS = {'time_seconds': [4., 12.], 'nominal_clearance_m': [5., 7.],
          'candidate_endpoint_spread_m': 1.}


def candidate_pool(f, protocol, count, rng, bank=12, budget=96, rounds=1):
    """Preserve the exact physics search set, then add observable-state probes."""
    unary, pair = f['unary'].numpy(), f['pair'].numpy()
    ranked, _ = nominal_beam(unary, pair, protocol.risk_weight, 128)
    initial, centre = initialize_bank(ranked, f['grid'].numpy(),
                                     f['valid'].numpy(), f['prior'].numpy(), bank, budget)
    choices = [x.nonzero()[:, 0].numpy() for x in f['valid']]
    selected, trace = whole_configuration_search(
        initial, lambda x: nominal_score(x, unary, pair, protocol.risk_weight),
        choices, budget, rounds)
    rows = [tuple(x) for x in trace['configurations']]
    if count < len(rows):
        raise ValueError('Diagnostic pool cannot truncate the actual physics search set')
    total = int(np.prod([len(x) for x in choices]))
    count = min(count, total)
    # Single-vessel changes around the common source centre support paired
    # third-vessel interventions, without looking at any future branch labels.
    for i, values in enumerate(choices):
        for value in values:
            row = centre.copy(); row[i] = value; key = tuple(row)
            if key not in rows and len(rows) < count:
                rows.append(key)
    attempts = 0
    while len(rows) < count and attempts < count * 1000:
        key = tuple(int(rng.choice(x)) for x in choices); attempts += 1
        if key not in rows:
            rows.append(key)
    if len(rows) != count:
        raise ValueError('Could not construct the declared unique diagnostic pool')
    return np.asarray(rows), {'physics_search': trace,
                             'physics_selected': selected.tolist(),
                             'source_center': centre.tolist()}


@torch.no_grad()
def feedback_distances(world, source, f, plans, opponent, stop_file=None):
    """Recompute source/Boids and opponent at each nominal branch state.

    Initial private current/relative velocity is discarded. The first command
    is exactly the observable root command also used by all real branches.
    Subsequent commands use the same source environment feedback equations.
    This offline reference additionally assumes the registered opponent model
    and agility; it is not relabelled as the deployable CITF-Phys baseline.
    """
    p = world.protocol; _, gates, _ = inputs(world, f, plans)
    i, j = pair_indices(len(gates[0])); distances = []; steps = 0
    states = world.state().copy()
    for gate in gates:
        if stop_file and Path(stop_file).exists():
            raise InterruptedError('Resource stop between nominal branches')
        branch = world.clone()
        branch.env.generate_random_current = lambda: np.zeros(3)
        branch.env._apf_force_noise = lambda: np.zeros(2)
        for boat, state in zip([*branch.env.defender_list, branch.env.attacker], states):
            boat.velocity_r = state[3:].copy()
            boat.current_velocity = np.zeros(3)
            boat.update_velocity(np.zeros(3))
        minimum = np.full(len(i), np.inf)
        for tick in range(p.prediction_steps):
            a, b = (f['learned'], f['rules']) if tick == 0 else source.commands(branch.env)[:2]
            force = (750 * (gate * a + (1 - gate) * b) + 250).numpy()
            branch.step(force, attacker_action=opponent.act(branch)); steps += 1
            xy = branch.state()[:, :2]
            minimum = np.minimum(minimum, np.linalg.norm(xy[i] - xy[j], axis=-1))
            if branch.done:
                break
        distances.append(minimum)
    return np.asarray(distances), steps


def choose(cost, plans):
    """Use the online search's lexical tie break for every scoring method."""
    return min(range(len(plans)), key=lambda k: (float(cost[k]), tuple(plans[k])))


def score_comparison(plans, unary, nominal_d, feedback_d, real_d, weight=32., clearance=7., search_plans=None):
    """real_d [C,R,P]; report sample oracle separately from cross-fit evidence."""
    risk = lambda d: np.maximum(clearance - d, 0.) ** 2
    actual = unary[:, None] + weight * risk(real_d).sum(-1)
    truth = actual.mean(-1)
    nominal = unary + weight * risk(nominal_d).sum(-1)
    feedback = unary + weight * risk(feedback_d).sum(-1)
    ix = {'nominal': choose(nominal, plans), 'feedback': choose(feedback, plans),
          'empirical_oracle': choose(truth, plans)}
    best = float(truth[ix['empirical_oracle']])
    result = {'selected_indices': ix, 'empirical_best_cost': best,
              'empirical_real_cost': truth.tolist(), 'nominal_cost': nominal.tolist(),
              'feedback_cost': feedback.tolist(), 'future_cost': actual.tolist()}
    for name in ('nominal', 'feedback'):
        k = ix[name]; predicted = nominal_d if name == 'nominal' else feedback_d
        near = real_d[k] < clearance
        missed = (predicted[k] >= clearance)[None] & near
        result[name] = {'real_cost': float(truth[k]), 'regret': float(truth[k] - best),
                        'distance_mae_m': float(np.abs(predicted[:, None] - real_d).mean()),
                        'risk_mse_m4': float(((risk(predicted) - risk(real_d).mean(1)) ** 2).mean()),
                        'selected_near_pairs': int(near.sum()), 'selected_missed_pairs': int(missed.sum()),
                        'selected_near_pair_miss_rate': float(missed.sum() / near.sum()) if near.any() else None}
    # Optimizing and evaluating the oracle on the same few futures exaggerates
    # its gain. Each held-out future is evaluated with a winner chosen from the
    # other futures; common random numbers still match all configurations.
    if actual.shape[1] > 1:
        gaps = []
        for r in range(actual.shape[1]):
            train_mean = (actual.sum(-1) - actual[:, r]) / (actual.shape[1] - 1)
            winner = choose(train_mean, plans)
            gaps.append(float(actual[ix['nominal'], r] - actual[winner, r]))
        result['oracle_leave_one_future_out_gain'] = float(np.mean(gaps))
    if search_plans is not None:
        allowed = {tuple(x) for x in search_plans}
        covered = [k for k, row in enumerate(plans) if tuple(row) in allowed]
        if not covered:
            raise ValueError('Physics search must be included in the common pool')
        result['physics_search_unique'] = len(covered)
        online = min(covered, key=lambda k: (float(nominal[k]), tuple(plans[k])))
        result['physics_online_selected_index'] = online
        result['physics_online_real_cost'] = float(truth[online])
        result['physics_online_regret'] = float(truth[online] - best)
        result['oracle_in_physics_search'] = tuple(plans[ix['empirical_oracle']]) in allowed
        result['physics_search_coverage_regret'] = float(truth[covered].min() - best)
    return result


def third_vessel_effect(plans, nominal_d, feedback_d, real_d):
    """Matched one-vessel changes, holding the evaluated pair's choices fixed."""
    i, j = np.triu_indices(plans.shape[1], 1)
    values = {'nominal': [], 'feedback': [], 'real': []}
    real_mean = real_d.mean(1)
    for left in range(len(plans)):
        for right in range(left):
            changed = np.flatnonzero(plans[left] != plans[right])
            if len(changed) != 1:
                continue
            mask = (i != changed[0]) & (j != changed[0])
            for key, data in [('nominal', nominal_d), ('feedback', feedback_d), ('real', real_mean)]:
                values[key].extend(np.abs(data[left, mask] - data[right, mask]).tolist())
    return {'matched_pair_interventions': len(values['real']),
            **{key + '_mean_abs_change_m': float(np.mean(value)) if value else None
               for key, value in values.items()}}


def summarize(rows):
    """All completed roots, with physical layout family as uncertainty unit."""
    def group(items):
        result = {'roots': len(items), 'families': len({r['geometry_family_id'] for r in items})}
        if not items:
            return result
        for name in ('nominal', 'feedback'):
            for metric in ('regret', 'distance_mae_m', 'risk_mse_m4'):
                result[name + '_' + metric] = float(np.mean([r['metrics'][name][metric] for r in items]))
            near = sum(r['metrics'][name]['selected_near_pairs'] for r in items)
            missed = sum(r['metrics'][name]['selected_missed_pairs'] for r in items)
            result[name + '_selected_near_pairs'] = near
            result[name + '_selected_missed_pairs'] = missed
            result[name + '_selected_near_pair_miss_rate'] = missed / near if near else None
        result['oracle_leave_one_future_out_gain'] = float(np.mean(
            [r['metrics']['oracle_leave_one_future_out_gain'] for r in items]))
        result['oracle_in_physics_search_fraction'] = float(np.mean(
            [r['metrics']['oracle_in_physics_search'] for r in items]))
        result['physics_search_coverage_regret'] = float(np.mean(
            [r['metrics']['physics_search_coverage_regret'] for r in items]))
        result['physics_online_regret'] = float(np.mean([r['metrics']['physics_online_regret'] for r in items]))
        for method in ('nominal', 'feedback', 'real'):
            effects = [r['metrics']['third_vessel_effect'][method + '_mean_abs_change_m'] for r in items]
            effects = [x for x in effects if x is not None]
            result[method + '_third_vessel_change_m'] = float(np.mean(effects)) if effects else None
        family_values = {}
        for r in items:
            family_values.setdefault(r['geometry_family_id'], []).append(
                r['metrics']['oracle_leave_one_future_out_gain'])
        values = np.asarray([np.mean(x) for x in family_values.values()])
        result['oracle_crossfit_family_mean'] = float(values.mean())
        result['oracle_crossfit_family_bootstrap95'] = None
        if len(values) >= 2:
            rng = np.random.default_rng(9471)
            means = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
            result['oracle_crossfit_family_bootstrap95'] = np.quantile(means, [.025, .975]).tolist()
        return result
    by = {}
    for field in ('time_group', 'interaction_group', 'disagreement_group', 'agility'):
        by[field] = {str(value): group([r for r in rows if r[field] == value])
                     for value in sorted({r[field] for r in rows})}
    return {'overall': group(rows), 'groups': by,
            'interpretation': 'Development short-branch geometry only; not closed-loop defense or formal performance.'}


def main():
    ap = argparse.ArgumentParser()
    for name in ('project', 'source', 'tasks', 'registry', 'out'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--opponent-spec')
    ap.add_argument('--root-times', nargs='+', type=float, default=[0., 2., 4., 8., 12., 20.])
    ap.add_argument('--pool', type=int, default=96); ap.add_argument('--futures', type=int, default=3)
    ap.add_argument('--bank', type=int, default=12); ap.add_argument('--budget', type=int, default=96)
    ap.add_argument('--rounds', type=int, default=1); ap.add_argument('--resume', action='store_true')
    ap.add_argument('--stop-file'); ap.add_argument('--max-roots-this-call', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1)
    bank = json.loads(Path(a.tasks).read_text(encoding='utf-8')); p = Protocol(**bank['protocol'])
    if bank['role'] not in ('train', 'validation'):
        raise ValueError('Diagnosis cannot use test or final_test layouts')
    if a.pool < 1 or a.futures < 2 or a.max_roots_this_call < 0:
        raise ValueError('Positive pool, at least two futures and nonnegative call limit required')
    if len(set(a.root_times)) != len(a.root_times) or any(
            t < 0 or t >= p.time_limit or abs(t / p.control_dt - round(t / p.control_dt)) > 1e-6
            for t in a.root_times):
        raise ValueError('Unique legal root ticks required')
    source = Source(a.project, a.source, a.device); opponent = load_opponent(a.opponent_spec, a.device)
    env = World(a.project, p, bank['tasks'][0]['defenders'])
    identity = {'revision': REVISION, 'protocol': p.__dict__, 'source_sha256': source.sha256,
                'source_implementation_sha256': source.implementation_sha256,
                'environment_sha256': env.environment_sha256, 'opponent': opponent.identity,
                'tasks_sha256': file_hash(a.tasks), 'registry_sha256': file_hash(a.registry),
                'citf_code_sha256': tree_hash(Path(__file__).parents[1] / 'citf'),
                'diagnosis_code_sha256': file_hash(__file__), 'groups': GROUPS,
                'arguments': {k: v for k, v in vars(a).items() if k not in
                              ('resume', 'stop_file', 'max_roots_this_call', 'out')},
                'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
                'feedback_reference': 'observable initial velocity, zero future current, registered opponent and agility; first command matched',
                'empirical_oracle': 'finite common-future diagnostic; cross-fit gain also reported; never used online'}
    out = Path(a.out)
    if a.resume:
        if canonical_hash(json.loads((out / 'protocol.json').read_text(encoding='utf-8'))) != canonical_hash(identity):
            raise ValueError('Diagnosis resume identity differs')
    else:
        if out.exists():
            raise ValueError('Refuse diagnosis overwrite; use identical-contract --resume')
        (out / 'roots').mkdir(parents=True); (out / 'tasks').mkdir()
        atomic_json(identity, out / 'protocol.json')
    registry = LayoutRegistry(a.registry); rows = []; completed_tasks = []; new_roots = 0
    stop = False
    for index, task in enumerate(bank['tasks']):
        task_record = out / 'tasks' / f'{index:05d}.json'
        if task_record.exists():
            record = json.loads(task_record.read_text(encoding='utf-8'))
            if record['task'] != task:
                raise ValueError('Committed task changed')
            for item in record['roots']:
                artifact = out / 'roots' / item['artifact']
                if file_hash(artifact) != item['artifact_sha256']:
                    raise ValueError('Committed root changed')
            rows.extend(record['roots']); completed_tasks.append(index); continue
        world = World(a.project, p, task['defenders'])
        world.reset(task['scene_seed'], task['future_seed'], task['agility'])
        family = world.initial_identity['fingerprint']
        if task.get('geometry_family_id') != family or registry.data['families'].get(family, {}).get('split') != bank['role']:
            raise ValueError('Unregistered or mismatched physical layout')
        policy = Controller(source, bank=a.bank, budget=a.budget, rounds=a.rounds)
        task_rows = []; tick = 0; requested = {round(t / p.control_dt) for t in a.root_times}
        while not world.done:
            if (a.stop_file and Path(a.stop_file).exists()) or (a.max_roots_this_call and new_roots >= a.max_roots_this_call):
                stop = True; break
            if tick in requested:
                root_key = f'{index:05d}_{tick:04d}'; root_record = out / 'roots' / (root_key + '.json')
                if root_record.exists():
                    row = json.loads(root_record.read_text(encoding='utf-8'))
                    if row['task_id'] != task['task_id'] or file_hash(out / 'roots' / row['artifact']) != row['artifact_sha256']:
                        raise ValueError('Root resume identity changed')
                else:
                    start = time.perf_counter(); f = factorize(world, source)
                    rng = np.random.default_rng(task['scene_seed'] * 1009 + tick + 48173)
                    plans, search = candidate_pool(f, p, a.pool, rng, a.bank, a.budget, a.rounds)
                    b, _, _ = inputs(world, f, plans); pi, pj = pair_indices(task['defenders'])
                    nd = (b['nominal'][0, :, pi] - b['nominal'][0, :, pj]).norm(dim=-1)
                    nd = nd.masked_fill(~b['known_time_valid'][0], torch.inf).amin(-1).numpy()
                    spread = float(b['nominal'][0, :, :-1, -1].std(0, unbiased=False).norm(dim=-1).mean())
                    strata = {'time_group': 'early' if world.env.Current_T < 4 else 'middle' if world.env.Current_T < 12 else 'late',
                               'interaction_group': 'collision_range' if nd.min() < 5 else 'near' if nd.min() < 7 else 'separated',
                               'disagreement_group': 'high' if spread >= 1 else 'low',
                               'nominal_min_distance_m': float(nd.min()), 'candidate_endpoint_spread_m': spread}
                    seeds = rng.integers(1, 2 ** 31 - 1, a.futures).tolist()
                    try:
                        fd, feedback_steps = feedback_distances(world, source, f, plans, opponent, a.stop_file)
                        tensors, meta, real_steps = sample_root(world, source, plans, seeds,
                            {'root_id': canonical_hash({'task': task, 'tick': tick, 'diagnosis': identity}),
                             'layout_id': task['layout_id'], 'split': bank['role'], 'scene_seed': task['scene_seed'],
                             'root_step': tick, 'candidate_indices': plans.tolist(),
                             'behavior_identity': {'method': 'CITF-Phys', 'bank': a.bank, 'budget': a.budget, 'rounds': a.rounds}},
                            opponent=opponent, stop_file=a.stop_file)
                    except InterruptedError:
                        stop = True; break
                    real_d, _ = branch_minima(tensors['real_positions'][None], tensors['time_valid'][None], tensors['entity_valid'][None])
                    metrics = score_comparison(plans, b['unary'][0].numpy(), nd, fd, real_d[0].numpy(),
                                               p.risk_weight, p.risk_clearance, search['physics_search']['configurations'])
                    metrics['third_vessel_effect'] = third_vessel_effect(plans, nd, fd, real_d[0].numpy())
                    artifact = out / 'roots' / (root_key + '.pt')
                    atomic_save({'tensors': tensors, 'metadata': meta, 'plans': plans,
                                 'nominal_minima': nd, 'feedback_minima': fd, 'search': search}, artifact)
                    row = {'task_id': task['task_id'], 'geometry_family_id': family, 'root_tick': tick,
                           'elapsed_seconds': float(world.env.Current_T), 'agility': task['agility'], **strata,
                           'candidates': len(plans), 'futures': a.futures, 'metrics': metrics,
                           'real_branch_transitions': real_steps, 'feedback_branch_transitions': feedback_steps,
                           'seconds': time.perf_counter() - start, 'artifact': artifact.name, 'artifact_sha256': file_hash(artifact)}
                    atomic_json(row, root_record); new_roots += 1
                    print(json.dumps({'task': index, 'tick': tick, 'roots_new': new_roots,
                                      'nominal_regret': metrics['nominal']['regret'],
                                      'oracle_crossfit_gain': metrics['oracle_leave_one_future_out_gain'] }), flush=True)
                task_rows.append(row)
            force, _ = policy.act(world)
            world.step(force, attacker_action=opponent.act(world)); tick += 1
            if tick > max(requested):
                break
        rows.extend(task_rows)
        if stop:
            break
        record = {'task': task, 'roots': task_rows, 'requested_ticks': sorted(requested),
                  'unreached_ticks': sorted(requested - {r['root_tick'] for r in task_rows}),
                  'behavior_stop_time': float(world.env.Current_T), 'behavior_terminated': world.done}
        atomic_json(record, task_record); completed_tasks.append(index)
    # A resource stop during replay must not hide previously committed roots.
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((out / 'roots').glob('*.json'))]
    for row in rows:
        if file_hash(out / 'roots' / row['artifact']) != row['artifact_sha256']:
            raise ValueError('Committed root artifact changed')
    report = summarize(rows)
    report.update(completed=len(completed_tasks) == len(bank['tasks']), completed_tasks=len(completed_tasks),
                  expected_tasks=len(bank['tasks']), real_branch_transitions=sum(r['real_branch_transitions'] for r in rows),
                  feedback_branch_transitions=sum(r['feedback_branch_transitions'] for r in rows),
                  status='resource_or_call_limit' if stop else 'completed', protocol_sha256=canonical_hash(identity))
    atomic_json(report, out / 'summary.json')
    print(json.dumps({k: report[k] for k in ('completed', 'completed_tasks', 'expected_tasks', 'overall')}), flush=True)


if __name__ == '__main__':
    main()
