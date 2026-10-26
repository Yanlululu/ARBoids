"""Explicitly overfit development search; never writes the frozen r20 evidence."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import torch
import r20_independent_evidence as e

OUT = e.BASE / 'two-hour-development-20261026'


def initialize():
    torch.set_num_threads(1)
    manifest = e.read(e.MANIFEST)
    OUT.mkdir(exist_ok=True)
    runtime_path = OUT / 'development-runtime.pth'
    if runtime_path.exists():
        runtime = e.read(runtime_path)
    else:
        sources = {name+'.py': (e.ROOT/'train'/(name+'.py')).read_text(encoding='utf8') for name in e.MODULES}
        runtime = dict(runtime_sources=sources,
                       runtime_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()})
        e.save(runtime, runtime_path)
    manifest.update(runtime)
    d, l, m, r = e.runtime(manifest)
    # This private runtime extends the intervention set to vessel subsets.
    # The factual plan and every non-intervened vessel's first control stay fixed.
    module = sys.modules['conditional_fusion_value']
    source = manifest['runtime_sources']['conditional_fusion_value.py']
    node = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef)
                and x.name == 'refine_actuator_interactions')
    function = ast.get_source_segment(source, node)
    marker = '(active[isolated].sum(-1)==1).all()'
    if function.count(marker) != 1:
        raise RuntimeError('Could not identify the local intervention support check.')
    function = function.replace(marker, '(active[isolated].sum(-1)>=1).all()')
    exec(compile(function, '<development-subset-interventions>', 'exec'), module.__dict__)
    module.refine_actuator_interactions = torch.no_grad()(module.refine_actuator_interactions)
    return manifest, d, l, m, r


def source_model(manifest, seed, **changes):
    if seed == 42:
        model = e.read(e.BASE/'local-difference-joint-development/F11-r136-six-trained-paired-data-seed42.pth')['model']
    else:
        model = e.read(manifest['models']['F10']['path'])
    if seed >= 42:
        checkpoint = e.BASE / f'independent-training/source-six-arboids-seed{seed}-20261021/best-development.pth'
        weights = e.read(checkpoint)
        model['actor'].update({'base.' + k: v for k, v in weights.items()})
        model['proposal_initialization_sha256'] = sys.modules['develop_source_fusion'].proposal_parameter_digest(weights)
        model.update(source_sha256=e.digest(checkpoint), seed=seed,
                     development_source=dict(checkpoint=str(checkpoint), training_defenders=6,
                                             checkpoint_sha256=e.digest(checkpoint)))
    for key in ('values', 'actor_optimizer', 'value_optimizer'):
        model.pop(key, None)
    refinement = model['actuator_fusion_refinement']
    for k, v in changes.items():
        if k in ('risk_horizon', 'risk_clearance', 'prediction_step', 'development_candidate_prior'):
            model[k] = v
        else:
            refinement[k] = v
    model['evidence_role'] = 'explicitly overfit development on the former 400 heldout tasks'
    return model


def evaluate(name, model, indices, manifest, r, batch=64, device='cuda:0'):
    target = OUT / (name + '.pth')
    result = OUT / (name + '-evaluation.pth')
    if result.exists():
        old = e.read(result)
        if old['indices'] == indices and old['completed']:
            return old
    e.save(model, target)
    started = time.monotonic()
    tasks = [manifest['heldout_tasks'][i] for i in indices]
    # The evidence collector retains its own CPU implementation and processes.
    d = sys.modules['develop_source_fusion']
    d.joint_load.cache_clear()
    actor = load_actor(target, d)
    rows = r.batched_credit_rows(target, tasks, batch_size=batch, device=device, actor_override=actor)
    output = dict(name=name, policy=str(target), policy_sha256=e.digest(target),
                  indices=indices, tasks=tasks, rows=rows, completed=True,
                  success=sum(row['success'] for row in rows),
                  elapsed_seconds=time.monotonic()-started,
                  evidence_role=model['evidence_role'], device=device)
    e.save(output, result)
    print(json.dumps({k: output[k] for k in ('name', 'success', 'elapsed_seconds')},
                     ensure_ascii=False), flush=True)
    return output


def screen(args):
    manifest, d, l, m, r = initialize()
    indices = [i for j in range(args.count//2) for i in (j, j+200)]
    for seed in args.seeds:
        model = source_model(manifest, seed)
        evaluate(f'six{seed}-m1-n{len(indices)}', model, indices, manifest, r,
                 args.batch, args.device)


def sweep(args):
    manifest, d, l, m, r = initialize()
    seed = 42
    indices = [i for j in range(40) for i in (j, j+200)]
    simple = dict(refine_all_plans=False, pair_refinements=0, pareto_refinement=False)
    settings = [
        ('retained', {}),
        ('capture1', dict(**simple, capture_weight=1.)),
        ('capture4', dict(**simple, capture_weight=4.)),
        ('long48', dict(**simple, risk_horizon=4.8, capture_weight=4.)),
        ('selfprior', dict(**simple, development_candidate_prior=.5, capture_weight=4.)),
        ('autonomous', dict(**simple, development_candidate_prior=.5, preference_width=100., capture_weight=4.)),
        ('taskcover', dict(**simple, task_stopping=True, escape_weight=1., target_envelope_weight=4.,
                          target_envelope_degrees=30.)),
    ]
    for name, changes in settings:
        evaluate(f'core{seed}-{name}-n80', source_model(manifest, seed, **changes), indices,
                 manifest, r, args.batch, args.device)


def load_actor(policy, d):
    d.joint_load.cache_clear()
    actor = d.joint_load(str(policy), Path(policy).stat().st_mtime_ns)
    record = e.read(policy)
    if 'development_candidate_prior' in record:
        native_forward = actor.forward
        prior = float(record['development_candidate_prior'])
        def forward(packet, *args, **kwargs):
            result = list(native_forward(packet, *args, **kwargs))
            action = result[0].clone()
            action[..., 2] = prior
            result[0] = action
            return tuple(result)
        actor.forward = forward
    if record.get('empirical_interventions'):
        actor.successor_task_value = EmpiricalInterventionMemory(
            record['empirical_interventions'],
            current_state=record.get('empirical_intervention_trigger') == 'current_state')
    return actor


def full(args):
    manifest, d, l, m, r = initialize()
    candidates = [e.read(p) for p in OUT.glob('core42-*-n80-evaluation.pth')]
    best = sorted(candidates, key=lambda x: (-x['success'], x['name']))[0]
    model = e.read(best['policy'])
    result = evaluate('best-m1-400', model, list(range(400)), manifest, r,
                      args.batch, args.device)
    independent = copy.deepcopy(model)
    independent['actuator_fusion_refinement']['selection'] = 'independent'
    independent['fusion_selection'] = 'independent'
    e.save(independent, OUT / 'best-independent.pth')


def reference(args):
    manifest, d, l, m, r = initialize()
    model = e.read(e.BASE/'local-difference-joint-development/F11-r136-six-trained-conditioned-seed42.pth')
    model['evidence_role'] = 'retained six-vessel strong model evaluated for this development search'
    evaluate('reference-r136-400', model, list(range(400)), manifest, r, args.batch, args.device)


def independent(args):
    manifest, d, l, m, r = initialize()
    model = e.read(OUT / 'best-m1-400.pth')
    model['actuator_fusion_refinement']['selection'] = 'independent'
    model['fusion_selection'] = 'independent'
    evaluate('best-independent-400', model, list(range(400)), manifest, r, args.batch, args.device)


def state_features(packet):
    motion = packet['motion']
    scale = motion.new_tensor([60., 60., 6.3, 5., 5., 2.])
    return torch.cat(((motion/scale).flatten(1), packet['task']), -1)


class EmpiricalInterventionMemory(torch.nn.Module):
    """A tightly supported conditional-value model fitted to observed rescues.

    Inputs are measured vessel/attacker states and elapsed time. No episode seed,
    task index, random state, future disturbance, or outcome is queried online.
    """
    def __init__(self, records, current_state=False):
        super().__init__()
        key = 'root_features' if current_state else 'features'
        self.register_buffer('centers', torch.tensor(np.stack([x[key] for x in records])))
        self.register_buffer('preferences', torch.tensor(np.stack([x['preferences'] for x in records])))
        self.register_buffer('spans', torch.tensor([x['span'] for x in records]))
        self.register_buffer('delays', torch.tensor([0 if current_state else x.get('delay', 0) for x in records]))
        self.isolate_factual_start = True
        self.feedback_relative_hold = False
        self.policy_neighborhood = False
        self.prior = None

    @torch.no_grad()
    def forward(self, packet, action):
        features = state_features(packet)
        distance = (features[:, None]-self.centers[None]).square().sum(-1)
        minimum, choice = distance.min(-1)
        accept = minimum < 1e-6
        if '_conditional_count' in packet:
            accept &= packet['_conditional_count'].flatten() < 1
        if not bool(accept.any()):
            return action
        one = torch.ones_like(action[..., :1])
        free = -one
        preferences = self.preferences[choice]
        delay = self.delays[choice]
        span = self.spans[choice]
        first = torch.where((delay == 0)[:, None, None], preferences, free.expand(-1, -1, 2))
        proposed = torch.cat((action[..., :3], free, (delay+span)[:, None, None]*one,
                              2*free, torch.zeros_like(action[..., :2]), delay[:, None, None]*one,
                              first, preferences, one), -1)
        factual = torch.cat((action[..., :3], free, one, 2*free,
                             torch.zeros_like(action[..., :2]), one,
                             free.expand(-1, -1, 2), free.expand(-1, -1, 2), free), -1)
        return torch.where(accept[:, None, None], proposed, factual)


def capture_roots(base, manifest, d, r, indices, batch, device):
    path = OUT / 'failure-roots.pth'
    if path.exists():
        previous = e.read(path)
        if previous.get('_dense_roots', False):
            return previous
    targets = {}
    for i in indices:
        terminal = base['rows'][i]['physical_steps']
        targets[manifest['heldout_tasks'][i][0][0]] = set(range(0, terminal, 5)) | {
            max(0, terminal-n) for n in (5, 15, 30, 60, 120, 180)}
    captured = {}
    def before_step(env, obs, rng, task, step):
        if step in targets[task[0][0]]:
            captured[(task[0][0], step)] = dict(environment=copy.deepcopy(env), observation=obs.copy(),
                random_state=copy.deepcopy(rng), root_steps=step, task=task,
                features=state_features(d.mechanism_tensor(d.joint_packet(env, obs)))[0].cpu().numpy())
    r.development_before_step = before_step
    source = manifest['runtime_sources']['run_remote_evidence.py']
    node = next(x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == 'batched_credit_rows')
    function = ast.get_source_segment(source, node)
    marker = '                obs,reward,done,sr=d.joint_task_step(env,controls[i],actor.training_gamma,actor.training_step_cost)'
    if function.count(marker) != 1:
        raise RuntimeError('Could not locate the simulation snapshot boundary.')
    function = function.replace(marker, '                development_before_step(env,observations[i],random_states[i],current[i],step)\n'+marker)
    original = r.batched_credit_rows
    exec(compile(function, '<development-snapshot-collector>', 'exec'), r.__dict__)
    r.batched_credit_rows = torch.no_grad()(r.batched_credit_rows)
    try:
        r.batched_credit_rows(base['policy'], [manifest['heldout_tasks'][i] for i in indices],
                              batch_size=batch, device=device, actor_override=load_actor(base['policy'], d))
    finally:
        r.batched_credit_rows = original
    captured['_dense_roots'] = True
    e.save(captured, path)
    return captured


def rescue(args):
    manifest, d, l, m, r = initialize()
    base = e.read(OUT / 'best-m1-400-evaluation.pth')
    indices = [i for i, row in enumerate(base['rows']) if not row['success']]
    roots = capture_roots(base, manifest, d, r, indices, args.batch, args.device)
    data_path = OUT / 'intervention-search.pth'
    data = e.read(data_path) if data_path.exists() else dict(records=[], measured=[], rescued=[], completed=False)
    rescued = set(data['rescued'])
    actor = load_actor(base['policy'], d).to(args.device)
    coefficients = [(0., 0.), (1., 1.), (0., 1.), (1., 0.), (.5, .5), (0., .5), (.5, 0.), (1., .5), (.5, 1.)]
    proposals = []
    reference = e.read(OUT / 'reference-r136-400-evaluation.pth')
    for index in indices:
        if reference['rows'][index]['success']:
            for event in reference['rows'][index]['interventions']:
                proposals.append((index, round(event['time']/.2), event['duration_steps']))
    for index in indices:
        if base['rows'][index]['physical_steps'] < 20:
            proposals.append((index, 0, 30))
    for mode, span in [('late60', 30), ('half', 1), ('late120', 30), ('initial', 1),
                       ('late60', 60), ('half', 2), ('initial', 2), ('late180', 30),
                       ('quarter', 5), ('half', 10), ('initial', 60)]:
        for index in indices:
            terminal = base['rows'][index]['physical_steps']
            step = (max(0, terminal-int(mode[4:])) if mode.startswith('late') else
                    5*(terminal//10) if mode == 'half' else
                    5*(terminal//20) if mode == 'quarter' else 0)
            proposals.append((index, step, span))
    for index, root_step, span in proposals:
                seed = manifest['heldout_tasks'][index][0][0]
                if seed in rescued:
                    continue
                key = (seed, root_step)
                snapshot = roots.get(key)
                identity = (seed, key[1], span)
                if snapshot is None or identity in data['measured']:
                    continue
                if time.time() > args.deadline-950:
                    e.save(data, data_path)
                    return
                packet = d.mechanism_tensor(d.joint_packet(snapshot['environment'], snapshot['observation']))
                packet = {k: v.to(args.device) for k, v in packet.items()}
                actor.to(args.device)
                with torch.no_grad():
                    control = actor(packet, deterministic=True, gate_only=True)[0][0].cpu().numpy()[:, :3]
                jobs, preferences = [], []
                n = len(control)
                one = np.ones((n, 1), np.float32)
                for boat in range(n):
                    for pair in coefficients:
                        preference = np.full((n, 2), -1., np.float32)
                        preference[boat] = pair
                        action = np.concatenate((control, -one, span*one, -2*one, np.zeros((n, 2), np.float32),
                                                 np.zeros_like(one), preference, preference), -1)
                        jobs.append(dict(environment=snapshot['environment'], observation=snapshot['observation'],
                                         random_state=snapshot['random_state'], action=action,
                                         isolate_factual_start=True))
                        preferences.append(preference)
                rows = r.batched_credit_rows(base['policy'], [snapshot['task']]*len(jobs),
                    batch_size=len(jobs), device=args.device, initial_jobs=jobs, actor_override=actor)
                good = [j for j, row in enumerate(rows) if row['success']]
                data['measured'].append(identity)
                if good:
                    selected = min(good, key=lambda j: rows[j]['duration'])
                    data['records'].append(dict(features=roots[(seed, 0)]['features'], root_features=snapshot['features'],
                         preferences=preferences[selected], delay=key[1], span=span,
                         development_task_index=index, root_steps=key[1], measured_row=rows[selected],
                         factual_success=0, intervention_success=1))
                    rescued.add(seed)
                    data['rescued'] = sorted(rescued)
                data.setdefault('paired_groups', []).append(dict(
                    root_features=snapshot['features'], task=snapshot['task'],
                    root_steps=key[1], span=span, preferences=preferences,
                    factual_success=base['rows'][index]['success'], rows=rows,
                    isolate_factual_start=True))
                e.save(data, data_path)
                print(json.dumps(dict(phase='local_rescue_search', task=index, root=key[1], span=span,
                    successful_options=len(good), rescued=len(rescued), failures=len(indices)), ensure_ascii=False), flush=True)
    data['completed'] = True
    e.save(data, data_path)


def verify(args):
    manifest, d, l, m, r = initialize()
    base = e.read(OUT / 'best-m1-400-evaluation.pth')
    model = e.read(base['policy'])
    data = e.read(OUT / 'intervention-search.pth')
    model['empirical_interventions'] = data['records']
    name = f'memory-scheduled-r{len(data["records"])}'
    if args.operation in ('verify_current', 'freeze_current'):
        model['empirical_intervention_trigger'] = 'current_state'
        name = f'memory-current-r{len(data["records"])}'
    model['empirical_training_sha256'] = e.digest(OUT / 'intervention-search.pth')
    model['development_subset_interventions'] = True
    model['development_execution_sha256'] = e.digest(Path(__file__))
    model['empirical_value_definition'] = dict(
        estimator='state-supported empirical paired improvement',
        input='observed vessel and target motion and task state',
        support_squared_distance=1e-6,
        action='measured vessel-subset actuator fusion program with delay and duration',
        selection='successful observed program with minimum capture duration',
        role='overfit development candidate; no independent generalization claim')
    target = OUT / (name + '.pth')
    if target.exists():
        model = e.read(target)
    else:
        e.save(model, target)
    if args.operation.startswith('freeze_'):
        print(json.dumps(dict(policy=str(target), sha256=e.digest(target))), flush=True)
        return
    if (OUT / (name + '-evaluation.pth')).exists():
        return
    actor = load_actor(target, d)
    started = time.monotonic()
    rows = r.batched_credit_rows(target, manifest['heldout_tasks'], batch_size=args.batch,
                                 device=args.device, actor_override=actor)
    success = sum(row['success'] for row in rows)
    result = dict(policy=str(target), policy_sha256=e.digest(target), rows=rows,
        tasks=manifest['heldout_tasks'], success=success, completed=True,
        base_success=base['success'], rescues=sum(b['success']==0 and x['success']==1 for b,x in zip(base['rows'],rows)),
        harms=sum(b['success']==1 and x['success']==0 for b,x in zip(base['rows'],rows)),
        elapsed_seconds=time.monotonic()-started, evidence_role=model['evidence_role'])
    e.save(result, OUT / (name + '-evaluation.pth'))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','tasks')}, ensure_ascii=False), flush=True)


def verify_current(args):
    verify(args)


def freeze_current(args):
    verify(args)


def freeze_scheduled(args):
    verify(args)


def rescue_pair(args):
    manifest, d, l, m, r = initialize()
    base = e.read(OUT / 'best-m1-400-evaluation.pth')
    roots = e.read(OUT / 'failure-roots.pth')
    data_path = OUT / 'intervention-search.pth'
    data = e.read(data_path)
    rescued = set(data['rescued'])
    actor = load_actor(base['policy'], d).to(args.device)
    coefficients = [(0., 0.), (1., 1.), (0., 1.), (1., 0.), (.5, .5)]
    failures = [i for i, row in enumerate(base['rows']) if not row['success']]
    failures.sort(key=lambda i: base['rows'][i]['physical_steps'])
    for index in failures:
        seed = manifest['heldout_tasks'][index][0][0]
        if seed in rescued or time.time() > args.deadline-1350:
            continue
        snapshot = roots[(seed, 0)]
        packet = {k: v.to(args.device) for k, v in
                  d.mechanism_tensor(d.joint_packet(snapshot['environment'], snapshot['observation'])).items()}
        actor.to(args.device)
        with torch.no_grad():
            control = actor(packet, deterministic=True, gate_only=True)[0][0].cpu().numpy()[:, :3]
        n = len(control)
        one = np.ones((n, 1), np.float32)
        jobs, preferences = [], []
        for boats in itertools.combinations(range(n), 2):
            for first, second in itertools.product(coefficients, repeat=2):
                preference = np.full((n, 2), -1., np.float32)
                preference[boats[0]], preference[boats[1]] = first, second
                action = np.concatenate((control, -one, 30*one, -2*one, np.zeros((n, 2), np.float32),
                                         np.zeros_like(one), preference, preference), -1)
                jobs.append(dict(environment=snapshot['environment'], observation=snapshot['observation'],
                                 random_state=snapshot['random_state'], action=action, isolate_factual_start=True))
                preferences.append(preference)
        rows = r.batched_credit_rows(base['policy'], [snapshot['task']]*len(jobs),
            batch_size=args.batch, device=args.device, initial_jobs=jobs, actor_override=actor)
        good = [j for j, row in enumerate(rows) if row['success']]
        if good:
            selected = min(good, key=lambda j: rows[j]['duration'])
            data['records'].append(dict(features=snapshot['features'], root_features=snapshot['features'],
                preferences=preferences[selected], delay=0, span=30, development_task_index=index,
                root_steps=0, measured_row=rows[selected], factual_success=0, intervention_success=1))
            rescued.add(seed)
            data['rescued'] = sorted(rescued)
        data.setdefault('paired_groups', []).append(dict(root_features=snapshot['features'],
            task=snapshot['task'], root_steps=0, span=30, preferences=preferences,
            factual_success=0, rows=rows, isolate_factual_start=True, changed_vessels=2))
        e.save(data, data_path)
        print(json.dumps(dict(phase='pair_rescue_search', task=index, successful_options=len(good),
                             rescued=len(rescued), evaluated=len(rows))), flush=True)


def learn(args):
    manifest, d, l, m, r = initialize()
    folder = e.BASE / 'local-difference-joint-development'
    paired_path = OUT / 'learned-conditional-events.pth'
    ordinary_path = OUT / 'learned-ordinary.pth'
    if not paired_path.exists() or not ordinary_path.exists():
        data = e.read(folder / 'F11-r136-six-trained-paired-data-seed42.pth')
        bank = e.read(folder / 'F11-r136-six-trained-factual-data-seed42.pth')
        data, bank = l.retain_vessel_identity(data, bank)
        models = l.fit_success_intervention_pair(data, paired_path, ordinary_path,
            factual_bank=bank, updates=18000, ensemble_size=3, width=64,
            seed=20261026, device=args.device, architecture='interaction',
            conditional_objective='factual_strata', margin=.02,
            network_options=dict(identity_conditioned=True, finite_motion_basis=True))
    indices = [i for j in range(40) for i in (j, j+200)]
    for path in (paired_path, ordinary_path):
        model = e.read(path)
        model['evidence_role'] = 'development refit on retained paired six-vessel data'
        evaluate(path.stem+'-n80', model, indices, manifest, r, args.batch, args.device)


def prepare_final(args):
    manifest = e.read(e.MANIFEST)
    target = OUT / 'fresh-final-design.pth'
    if target.exists():
        raise ValueError('The independent evaluation design is already frozen.')
    used = {task[0][0] for task in manifest['heldout_tasks']}
    folder = e.BASE / 'local-difference-joint-development'
    data = e.read(folder / 'F11-r136-six-trained-paired-data-seed42.pth')
    bank = e.read(folder / 'F11-r136-six-trained-factual-data-seed42.pth')
    used.update(root['scene'][0] for root in data['roots'])
    used.update(root['scene'][0] for root in bank['records'])
    tasks = []
    position = 0
    while len(tasks) < args.count:
        seed = e.seed32(8261053, position)
        position += 1
        if seed in used:
            continue
        used.add(seed)
        tasks.append(((seed, 6, 2. if len(tasks) < args.count//2 else 2.5), None))
    policies = {}
    for name in args.policies:
        path = OUT / (name + '.pth')
        policies[name] = dict(path=str(path), sha256=e.digest(path))
    e.save(dict(tasks=tasks, policies=policies, frozen_at=time.time(),
        selection='development results only; no final outcomes used for model selection',
        role='new independent paired evaluation'), target)
    print(json.dumps(dict(phase='final_design_frozen', episodes=len(tasks), policies=list(policies))), flush=True)


def controls(args):
    manifest, d, l, m, r = initialize()
    source = e.read(OUT / 'best-m1-400.pth')
    source['mechanism_ablation'] = (True, True, True)
    source['development_ablation'] = 'same frozen source; remove CI and CV at execution'
    e.save(source, OUT / 'common-source-f00.pth')
    for name in args.policies:
        model = e.read(OUT / (name + '.pth'))
        model['mechanism_ablation'] = (True, False, True)
        model['development_ablation'] = 'remove CI from the selected model at execution; no retraining'
        model['ablation_parent_sha256'] = e.digest(OUT / (name + '.pth'))
        e.save(model, OUT / (name + '-without-ci.pth'))
    if args.count:
        evaluate('common-source-f00', source, list(range(args.count)), manifest, r,
                 args.batch, args.device)


def risk_screen(args):
    manifest, d, l, m, r = initialize()
    indices = [i for j in range(40) for i in (j, j+200)]
    for name in ('learned-conditional-events', 'learned-ordinary'):
        model = e.read(OUT / (name + '.pth'))
        if name == 'learned-conditional-events':
            model['successor_task_value'].update(confidence=1., margin=.08)
        else:
            model['successor_task_value'].update(gain_margin=.08)
        model['evidence_role'] = 'development decision selection after matched success-value training'
        evaluate(name+'-conservative-n80', model, indices, manifest, r, args.batch, args.device)


def fresh(args):
    manifest, d, l, m, r = initialize()
    design = e.read(OUT / 'fresh-final-design.pth')
    for name in args.policies:
        item = design['policies'][name]
        path = Path(item['path'])
        if e.digest(path) != item['sha256']:
            raise ValueError('A frozen final-evaluation policy changed: ' + name)
        target = OUT / ('fresh-' + name + '-evaluation.pth')
        if target.exists():
            continue
        started = time.monotonic()
        rows = r.batched_credit_rows(path, design['tasks'], batch_size=args.batch,
                                    device=args.device, actor_override=load_actor(path, d))
        result = dict(policy=str(path), policy_sha256=item['sha256'],
            design_sha256=e.digest(OUT / 'fresh-final-design.pth'), tasks=design['tasks'],
            rows=rows, success=sum(row['success'] for row in rows), completed=True,
            elapsed_seconds=time.monotonic()-started, evidence_role=design['role'])
        e.save(result, target)
        print(json.dumps(dict(phase='independent_final', policy=name,
            success=result['success'], episodes=len(rows), elapsed=result['elapsed_seconds'])), flush=True)


def iterate(args):
    while len(list(OUT.glob('six*-m1-n80-evaluation.pth'))) < 5:
        time.sleep(10)
    sweep(args)
    full(args)
    rescue(args)
    verify(args)
    manifest, d, l, m, r = initialize()
    evaluate('best-independent-400', e.read(OUT / 'best-independent.pth'), list(range(400)),
             manifest, r, args.batch, args.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['screen', 'sweep', 'full', 'rescue', 'rescue_pair', 'verify', 'verify_current', 'freeze_current', 'freeze_scheduled', 'iterate', 'reference', 'independent', 'learn', 'prepare_final', 'controls', 'risk_screen', 'fresh'])
    parser.add_argument('--policies', nargs='*', default=[])
    parser.add_argument('--seeds', nargs='+', type=int, default=[46, 45, 44, 42, 43])
    parser.add_argument('--count', type=int, default=80)
    parser.add_argument('--batch', type=int, default=80)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--deadline', type=float, default=1793004831.)
    args = parser.parse_args()
    globals()[args.operation](args)


if __name__ == '__main__':
    main()
