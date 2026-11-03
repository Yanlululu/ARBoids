"""Spend first-action recourse labels near observable radial boundaries.

Training retains eight-second background roots and samples each 0.4 seconds
when the CURRENT innermost defender is within 16 metres radially of the
attacker. Validation remains a uniform two-second grid. Future outcomes do
not enter root or candidate selection. All layouts keep their prior split.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import citf.collect as collection
from citf.bridge import Source, World, factorize
from citf.contracts import (Protocol, SCHEMA, FEATURES, CONTEXT, COLLECTION_SCHEMA,
                            BRANCH_EVENT_FIELDS, canonical_hash, file_hash, tree_hash, validate_dataset)
from citf.opponents import APFOpponent
from citf.search import nominal_beam, initialize_bank
from experiments import collect_guard_local as local
from experiments import recourse_guard as recourse


SELECTION = dict(training='0.4 second grid when current minimum radial gap >= -16m; 8 second background grid',
                 validation='uniform 2 second grid', boundary_gap_m=-16.,
                 future_outcome_filter=False)


def adaptive_main():
    parser = argparse.ArgumentParser()
    for key in ('project', 'source', 'tasks', 'registry', 'out'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--layout-start', type=int, default=0)
    parser.add_argument('--layout-count', type=int, default=0)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--futures', type=int, default=3)
    parser.add_argument('--max-per-layout', type=int, required=True)
    parser.add_argument('--root-times', nargs='+', type=float, required=True)
    parser.add_argument('--stop-file')
    args = parser.parse_args()
    torch.set_num_threads(1)
    bank = json.loads(Path(args.tasks).read_text())
    if bank['role'] not in ('train', 'validation'):
        raise ValueError('Boundary acquisition is for declared development training and validation')
    protocol = Protocol(**bank.get('protocol', {}))
    requested = sorted({round(value / protocol.control_dt) for value in args.root_times})
    if requested != list(range(0, round(protocol.time_limit / protocol.control_dt), 2)):
        raise ValueError('This acquisition identity requires the full 0.4-second root grid')
    if args.max_per_layout != len(requested):
        raise ValueError('All eligible roots must be retained under the declared acquisition')
    out = Path(args.out)
    if out.exists():
        raise ValueError('Refuse dataset overwrite')
    source = Source(args.project, args.source, args.device)
    registry = collection.LayoutRegistry(args.registry)
    opponent = APFOpponent()
    items, roots, planned = [], [], []
    branch_steps = behavior_steps = 0
    started = time.perf_counter()
    selected = bank['tasks'][args.layout_start:args.layout_start + args.layout_count if args.layout_count else None]
    seen = set()
    for task in selected:
        world = World(args.project, protocol, task['defenders'])
        world.reset(task['scene_seed'], task['future_seed'], task['agility'])
        family = registry.register(world.initial_identity, bank['role'], task['layout_id'])
        if task.get('geometry_family_id') != family or family in seen:
            raise ValueError('Changed or repeated physical layout')
        seen.add(family)
        behavior = dict(method='StoppedGuard-Phys-development', predictor_sha256=None,
                        source_sha256=source.sha256, protocol_sha256=protocol.digest(),
                        bank=12, budget=96, rounds=1, risk_weight=64., guard_weight=16., guard_margin=6.,
                        opponent=opponent.identity)
        policy = local.PhysicalGuardBehavior(source, None, args.device, 12, 96, 1)
        entry = dict(layout_id=task['layout_id'], geometry_family_id=family,
                     requested_ticks=requested, collected_ticks=[], skipped_observable_ticks=[],
                     root_selection=SELECTION)
        planned.append(entry)
        tick = 0
        result = None
        while not world.done:
            if args.stop_file and Path(args.stop_file).exists():
                raise InterruptedError('Requested stop before the next actual branch')
            if tick in requested:
                radius = np.linalg.norm(world.state()[:, :2], axis=-1)
                current_gap = float(radius[:-1].min() - radius[-1])
                eligible = ((tick % 40 == 0 or current_gap >= SELECTION['boundary_gap_m'])
                            if bank['role'] == 'train' else tick % 10 == 0)
                if eligible:
                    f = factorize(world, source)
                    ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), protocol.risk_weight, 128)
                    top = max(1, args.candidates // 2)
                    initial, _ = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                                 f['prior'].numpy(), top, top)
                    choices = [row.nonzero()[:, 0].numpy() for row in f['valid']]
                    proposals = [tuple(row) for row in initial.tolist()]
                    rng = np.random.default_rng(task['scene_seed'] + 104729 * tick + 700000003)
                    attempts = 0
                    while len(proposals) < args.candidates and attempts < 10000:
                        candidate = tuple(int(rng.choice(row)) for row in choices)
                        attempts += 1
                        if candidate not in proposals:
                            proposals.append(candidate)
                    if len(proposals) != args.candidates:
                        raise ValueError('Insufficient distinct current candidates')
                    future = list(map(int, rng.integers(1, 2 ** 31 - 1, size=args.futures)))
                    root_id = canonical_hash(dict(geometry=family, step=tick, behavior=behavior,
                                                  future_seeds=future, agility=task['agility'], selection=SELECTION))
                    item, metadata, cost = collection.sample_root(world, source, np.asarray(proposals), future,
                        dict(root_id=root_id, layout_id=task['layout_id'], split=bank['role'],
                             scene_seed=task['scene_seed'], root_step=tick, candidate_indices=[list(row) for row in proposals],
                             behavior_identity=behavior, sampling_probability=1.,
                             root_selection=SELECTION, observed_current_radial_gap_m=current_gap),
                        opponent=opponent, stop_file=args.stop_file)
                    items.append(item)
                    roots.append(metadata)
                    branch_steps += cost
                    entry['collected_ticks'].append(tick)
                else:
                    entry['skipped_observable_ticks'].append(tick)
            force, _ = policy.act(world)
            result = world.step(force, attacker_action=opponent.act(world))
            behavior_steps += 1
            tick += 1
        entry.update(unreached_ticks=[value for value in requested if value >= tick],
                     reason='actual termination; earlier observable exclusions recorded separately',
                     task_terminated=True, actual_terminal_event=result,
                     behavior_stop_time=float(world.env.Current_T))
    if not items:
        raise ValueError('No eligible root reached')
    blob = dict(schema=SCHEMA, collection_schema=COLLECTION_SCHEMA, branch_event_fields=BRANCH_EVENT_FIELDS,
                history_features=FEATURES, context_features=CONTEXT,
                tensors={key: torch.stack([item[key] for item in items]) for key in items[0]},
                roots=roots, protocol=protocol.__dict__, provenance=dict(
                    tasks_sha256=file_hash(args.tasks), registry_sha256=file_hash(args.registry),
                    source_sha256=source.sha256, environment_sha256=world.environment_sha256,
                    branch_transitions=branch_steps, behavior_transitions=behavior_steps, planned_roots=planned,
                    opponent_identity=opponent.identity, collection_revision='CITF-first-action-guard-boundary-data-v1',
                    collection_code_sha256=canonical_hash({'boundary': file_hash(__file__),
                        'recourse': file_hash(recourse.__file__), 'local_candidates': file_hash(local.__file__),
                        'core': tree_hash(Path(collection.__file__).parent)}),
                    root_selection=SELECTION, seconds=time.perf_counter() - started, arguments=vars(args)))
    report = validate_dataset(blob)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, out)
    out.with_suffix('.json').write_text(json.dumps(report | blob['provenance'], indent=2))
    print(report, flush=True)


def main():
    collection.sample_root = recourse.sample_root
    collection.main = adaptive_main
    local.main()


if __name__ == '__main__':
    main()
