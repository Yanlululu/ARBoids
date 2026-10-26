"""Repeat the frozen r20 recipe and evaluate genuinely separate task populations."""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext

for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'train/experiments/author-source-fusion-20261011'
OUTPUT = BASE / 'r20-independent-evidence'
MANIFEST = OUTPUT / 'manifest.pth'
NODE = BASE / 'local-difference-joint-development/RESULT_NODE_20261020_043146_M2_R20.json'
LABELS = ('F00', 'F10', 'F01', 'F11', 'ordinary_no_M1', 'ordinary_with_M1')
SEEDS = tuple(range(42, 47))
MODULES = ('develop_source_fusion', 'conditional_fusion_value', 'local_intervention_value',
           'plain_return', 'mechanism_controls', 'run_remote_evidence')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def seed32(*parts):
    return int(np.random.SeedSequence([20261025, 835104, *map(int, parts)]).generate_state(1)[0])


def task_map(scene, future, seed):
    return ((seed32(10, seed, scene[0]), int(scene[1]), float(scene[2])),
            None if future is None else seed32(11, seed, future))


def initialize(sources):
    """Freeze designs and code before reading any new endpoint outcome."""
    if MANIFEST.exists():
        existing = read(MANIFEST)
        if existing['runtime_sha256'] != {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()}:
            raise ValueError('The existing frozen runtime differs.')
        return existing
    node = json.loads(NODE.read_text())
    if node['verification']['algorithm_identity_and_fairness_audit']['status'] != 'passed':
        raise ValueError('The indexed identity and fairness audit must pass first.')
    models = {}
    seen = set()
    for label in LABELS:
        item = node['artifacts']['model:' + label]
        path = ROOT / item['relative_path']
        if digest(path) != item['sha256']:
            raise ValueError('An indexed model changed: ' + label)
        models[label] = dict(path=str(path), sha256=item['sha256'])
    for key, item in node['artifacts'].items():
        if key.startswith('eval:'):
            evaluation = read(ROOT / item['relative_path'])
            seen.update(int(scene[0]) for scene, _ in evaluation['tasks'])
    designs = {}
    factual_tasks = {}
    for arm in ('F01', 'F11'):
        model = read(models[arm]['path'])
        source = Path(model['successor_task_fit']['training_data'])
        data = read(source)
        roots = []
        for root in data['roots']:
            seen.add(int(root['scene'][0]))
            # Retain an acquisition design, never the old observations or states.
            roots.append({k: copy.deepcopy(root[k]) for k in
                          ('scene', 'prefix_future', 'root_steps', 'lookback', 'options',
                           'futures', 'future_weights') if k in root})
            roots[-1]['outcomes'] = {key: dict(duration=row['duration'])
                                     for key, row in root['outcomes'].items()}
        tasks = []
        for name in model['successor_task_fit']['factual_bank_training_data']:
            bank = read(name)
            if 'records' not in bank:
                continue
            tasks.extend((tuple(record['scene']), record['future']) for record in bank['records'])
        tasks = list(dict.fromkeys(tasks))
        seen.update(int(scene[0]) for scene, _ in tasks)
        factual_tasks[arm] = tasks
        designs[arm] = dict(roots=roots, source=str(source), source_sha256=digest(source),
                            historical_simulation_steps=data['simulation_steps'],
                            observations=sum(len(root['outcomes']) for root in roots),
                            inherited_labels=False, inherited_states=False)
    heldout = [((seed32(90, cell, i), 6, agility), None)
               for cell, agility in enumerate((2., 2.5)) for i in range(200)]
    train_scenes = {task_map(root['scene'], root['prefix_future'], seed)[0][0]
                    for seed in SEEDS for design in designs.values() for root in design['roots']}
    train_scenes.update(task_map(scene, future, seed)[0][0] for seed in SEEDS
                        for tasks in factual_tasks.values() for scene, future in tasks)
    test_scenes = {scene[0] for scene, _ in heldout}
    if len(test_scenes) != len(heldout) or test_scenes & (seen | train_scenes):
        raise ValueError('The fixed test population is not disjoint.')
    if train_scenes & seen:
        raise ValueError('A newly sampled training layout overlaps the historical population.')
    model = read(models['F10']['path'])
    model['fusion_selection'] = 'independent'
    model['actuator_fusion_refinement']['selection'] = 'independent'
    model['comparison_scope'] = 'same r20 proposal and candidate library; independent rather than joint selection'
    independent = OUTPUT / 'fixed/M1-independent.pth'
    save(model, independent)
    models['M1_independent'] = dict(path=str(independent), sha256=digest(independent))
    manifest = dict(version=1, node_id=node['node_id'], indexed_node_sha256=digest(NODE),
        created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        seeds=SEEDS, models=models, designs=designs, factual_tasks=factual_tasks,
        heldout_tasks=heldout, runtime_sources=sources,
        runtime_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
        method='r20 population-normalized local ranking', independent_source_defenders=3,
        source_training_steps=1000000, updates_per_model=24000, ensemble_size=3, width=64,
        fitting_policy='fresh observations and fresh value initialization for every pipeline',
        endpoint_selection='fixed 24000 updates; no checkpoint or parameter choice using heldout outcomes',
        historical_r20_role='selected development reference; excluded from independent-pipeline mean',
        heldout_role='unseen endpoint tasks; never fitting, early stopping, or parameter selection')
    save(manifest, MANIFEST)
    print(dict(phase='r20_evidence_frozen', manifest_sha256=digest(MANIFEST),
               heldout_tasks=len(heldout), seeds=SEEDS,
               design_observations={k: v['observations'] for k, v in designs.items()}), flush=True)
    return manifest


def runtime(manifest):
    torch.set_num_threads(1)
    for name in MODULES:
        filename = name + '.py'
        source = manifest['runtime_sources'][filename]
        if hashlib.sha256(source.encode()).hexdigest() != manifest['runtime_sha256'][filename]:
            raise ValueError('The frozen runtime changed.')
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / 'train' / filename)
        sys.modules[name] = module
        exec(compile(source, module.__file__, 'exec'), module.__dict__)
    mp.set_start_method('fork', force=True)
    r = sys.modules['run_remote_evidence']
    r.initialize()
    return r.d, sys.modules['local_intervention_value'], sys.modules['mechanism_controls'], r


def resource_wait():
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                   '--format=csv,noheader,nounits'], text=True).splitlines()[0])
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        if free > 6500 and limit - current > 16 * 2**30:
            return
        print(dict(phase='resource_wait', gpu_free_mib=free, ram_free_gib=(limit-current)/2**30), flush=True)
        time.sleep(30)


def evaluate(policy, tasks, path, manifest, workers=2, role='heldout'):
    # Completed predecessor models can be tested while the next arm collects
    # data. Lock the existing directory so two evaluators never write one file.
    import fcntl
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return _evaluate_unlocked(policy, tasks, path, manifest, workers, role)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _evaluate_unlocked(policy, tasks, path, manifest, workers=2, role='heldout'):
    d = sys.modules['develop_source_fusion']
    r = sys.modules['run_remote_evidence']
    identity = digest(policy)
    previous = read(path) if Path(path).exists() else None
    if previous and (previous['tasks'] != tasks or previous['policy_sha256'] != identity
                     or previous['runtime_sha256'] != manifest['runtime_sha256']):
        raise ValueError('A resumed evaluation has a different identity.')
    if previous and previous['completed']:
        return previous
    rows = previous['rows'] if previous else [None] * len(tasks)
    revision = Path(policy).stat().st_mtime_ns
    start = time.monotonic()
    jobs = [(i, (scene, str(policy), revision, False, 0, 0, 0, 0, future))
            for i, (scene, future) in enumerate(tasks) if rows[i] is None]
    with (ProcessPoolExecutor(max_workers=workers, initializer=r.initialize_source_branches,
                              initargs=(str(d.SOURCE),)) if workers > 1 else nullcontext(None)) as pool:
        if pool is None:
            stream = ((i, d.joint_episode(task)) for i, task in jobs)
        else:
            pending = {pool.submit(d.joint_episode, task): i for i, task in jobs}
            stream = ((pending[f], f.result()) for f in as_completed(pending))
        for index, row in stream:
            rows[index] = row
            done = [row for row in rows if row is not None]
            if len(done) % 16 == 0 or len(done) == len(tasks):
                record = dict(policy=str(policy), policy_sha256=identity, tasks=tasks, rows=rows,
                    evaluated=len(done), completed=len(done) == len(tasks),
                    success=sum(row['success'] for row in done), runtime_sha256=manifest['runtime_sha256'],
                    evidence_role=role, manifest_sha256=digest(MANIFEST),
                    elapsed_seconds=time.monotonic()-start)
                save(record, path)
                print(dict(phase='r20_evaluation', policy=Path(policy).name, role=role,
                           evaluated=len(done), total=len(tasks), success=record['success']), flush=True)
    return record


def verify_reference(manifest, workers):
    """Replay success and failure cases before opening the new test population."""
    node = json.loads(NODE.read_text())
    result = {}
    for label in LABELS:
        item = node['artifacts']['eval:' + label + ':new4310-200']
        old = read(ROOT / item['relative_path'])
        indices = [next(i for i, row in enumerate(old['rows']) if bool(row['success']) == outcome)
                   for outcome in (True, False)]
        tasks = [old['tasks'][i] for i in indices]
        new = evaluate(manifest['models'][label]['path'], tasks,
                       OUTPUT / 'verification' / (label + '.pth'), manifest, workers, 'historical-replay')
        fields = ('success', 'outcome', 'duration')
        for expected, actual in zip((old['rows'][i] for i in indices), new['rows']):
            if any(expected[k] != actual[k] for k in fields):
                raise ValueError('Historical inference replay differs: ' + label)
        result[label] = dict(matched=2, policy_sha256=new['policy_sha256'])
    save(dict(completed=True, models=result, runtime_sha256=manifest['runtime_sha256']),
         OUTPUT / 'verification/result.pth')
    print(dict(phase='r20_reference_replay_passed', cases=12), flush=True)


def fixed_test(manifest, workers):
    verify_reference(manifest, workers)
    for label, item in manifest['models'].items():
        if digest(item['path']) != item['sha256']:
            raise ValueError('A frozen fixed model changed: ' + label)
        resource_wait()
        evaluate(item['path'], manifest['heldout_tasks'], OUTPUT / 'fixed' / (label + '-heldout.pth'),
                 manifest, workers)


def fit_parameters(template, seed, ordinary=False):
    fit, value = template['successor_task_fit'], template['successor_task_value']
    return dict(ordinary=ordinary, ordinary_architecture='interaction',
        updates=fit['updates'], ensemble_size=len(value['states']), width=fit['width'], seed=seed,
        edge_weight=fit['difference_weight'], anchor_weight=fit['factual_contrast_weight'],
        rank_weight=fit['local_rank_weight'], factual_weight=fit['factual_weight'],
        factual_bank_weight=fit['factual_bank_weight'], device='cuda:0',
        bandwidth=value['bandwidth'], margin=value['margin'], confidence=value['confidence'],
        global_penalty=value['global_penalty'], peer_conditioned=value['peer_conditioned'],
        context_time_scale=value['context_time_scale'], interpolation_power=value['interpolation_power'],
        conditional_strata=False, context_conditioned=True, local_contrast_head=False,
        projection_strength=value['projection_strength'], projection_steps=value['projection_steps'],
        motion_basis=True, joint_factual_calibration=not ordinary,
        balanced_local_sampling=not ordinary, stratified_local_calibration=False,
        intervention_locality=False, residual_potential=False)


def replay_fit(manifest):
    """Check the repeated learner against both original fitted networks."""
    l = sys.modules['local_intervention_value']
    verification = {}
    for arm in ('F01', 'F11'):
        resource_wait()
        original = read(manifest['models'][arm]['path'])
        data = read(original['successor_task_fit']['training_data'])
        bank = l.factual_bank_from_model(original, data)
        with tempfile.TemporaryDirectory(prefix='.r20-fit-replay-', dir=OUTPUT) as directory:
            fitted = l.fit_joint_potential(data, Path(directory)/'model.pth', factual_bank=bank,
                                          **fit_parameters(original, 42))
            differences = []
            for left, right in zip(original['successor_task_value']['states'],
                                   fitted['successor_task_value']['states']):
                if left.keys() != right.keys():
                    raise ValueError('Repeated value-network architecture differs.')
                differences.extend(float((left[k]-right[k]).abs().max()) for k in left)
            maximum = max(differences)
            if maximum != 0:
                raise ValueError(f'{arm} training replay differs, max absolute difference {maximum}')
            verification[arm] = dict(parameters_exact=True, max_absolute_difference=maximum,
                training_observation_sha256=fitted['successor_task_fit']['training_observation_sha256'])
        print(dict(phase='r20_fit_replay_passed', arm=arm, **verification[arm]), flush=True)
    save(dict(completed=True, arms=verification, runtime_sha256=manifest['runtime_sha256']),
         OUTPUT/'verification/training-replay.pth')


def source_models(manifest, seed):
    """Reuse only completed independent original-source training, never M2 weights."""
    from source_arboids import SourcePolicy, SOURCE_COMMIT
    directory = OUTPUT / f'seed{seed}'
    original = read(manifest['models']['F00']['path'])
    if seed == 42:
        source_directory = Path(sys.modules['develop_source_fusion'].SOURCE).parent
        source_record = dict(seed=42, source_commit=SOURCE_COMMIT, training_steps=1000000,
                             fresh_initialization=True, completed=True)
    else:
        source_directory = BASE/'independent-training'/f'source-seed{seed}'
        source_record = read(source_directory/'source-repeat-result.pth')
        if (source_record['seed'] != seed or not source_record['completed']
                or not source_record['fresh_initialization'] or source_record['source_commit'] != SOURCE_COMMIT
                or source_record['training_steps'] != 1000000):
            raise ValueError('The original-source training is not complete and independent.')
    checkpoint = source_directory/'adares1.pth'
    identity = digest(checkpoint)
    expected = original['source_sha256'] if seed == 42 else source_record['checkpoint_sha256']
    if identity != expected:
        raise ValueError('The independent source checkpoint changed.')
    with (source_directory/'metrics1.csv').open(newline='') as stream:
        metrics = list(csv.DictReader(stream))
    if int(metrics[-1]['step']) != 1000000:
        raise ValueError('The source did not complete the common one-million-step budget.')
    if seed != 42 and digest(source_directory/'config.yaml') != source_record['configuration_sha256']:
        raise ValueError('The recorded source training configuration changed.')
    # The frozen loader checks its proposal against the active source actor.
    # Each repeat must bind that check to its own independently trained source.
    sys.modules['develop_source_fusion'].initialize(str(checkpoint))
    weights = SourcePolicy(checkpoint).actor.state_dict()
    old = {k[5:]: v for k, v in original['actor'].items() if k.startswith('base.')}
    if old.keys() != weights.keys() or any(old[k].shape != weights[k].shape for k in old):
        raise ValueError('The independent source has a different architecture.')
    result = {}
    for arm in ('F00', 'F10', 'M1_independent'):
        target = directory/(arm+'.pth')
        if target.exists():
            model = read(target)
            if model['independent_source']['checkpoint_sha256'] != identity:
                raise ValueError('An independent predecessor changed.')
        else:
            model = read(manifest['models'][arm]['path'])
            if model.get('iteration', 0) or model.get('simulation_steps', 0) or model.get('successor_task_value'):
                raise ValueError('Independent initialization requires an untrained selector.')
            model['actor'].update({'base.'+k: v.detach().cpu().clone() for k, v in weights.items()})
            for key in ('values', 'actor_optimizer', 'value_optimizer'):
                model.pop(key, None)
            model.update(seed=seed, source_sha256=identity, history=[],
                independent_source=dict(seed=seed, checkpoint=str(checkpoint), checkpoint_sha256=identity,
                    configuration_sha256=digest(source_directory/'config.yaml'), training_steps=1000000,
                    training_defenders=3, completed=True, fresh_initialization=True, source_commit=SOURCE_COMMIT),
                r20_replication=dict(manifest_sha256=digest(MANIFEST), training_seed=seed,
                    inherited_value_weights=False, historical_fixed_node=False))
            save(model, target)
        result[arm] = target
    return result


def remapped_design(manifest, arm, seed):
    blueprint = copy.deepcopy(manifest['designs'][arm])
    for root in blueprint['roots']:
        root['scene'], root['prefix_future'] = task_map(root['scene'], root['prefix_future'], seed)
        root['futures'] = [None if f is None else seed32(11, seed, f) for f in root['futures']]
        if any('success' in row for row in root['outcomes'].values()):
            raise ValueError('A frozen acquisition design contains inherited labels.')
    return blueprint


def repeated_data(manifest, arm, seed, policy, workers, batch):
    l = sys.modules['local_intervention_value']
    directory = OUTPUT/f'seed{seed}'
    bank_path = directory/(arm+'-factual.pth')
    tasks = [task_map(scene, future, seed) for scene, future in manifest['factual_tasks'][arm]]
    reference = dict(policy_sha256=digest(policy), tasks=tasks,
                     rows=[dict(success=None) for _ in tasks])
    bank = l.collect_factual_trajectories(policy, [reference], bank_path, workers=workers,
        period=5, inference_batch=batch, inference_device='cpu')
    if not bank['completed'] or len(bank['records']) != len(tasks):
        raise ValueError('The independent factual bank is incomplete.')
    blueprint = remapped_design(manifest, arm, seed)
    observations = sum(len(root['outcomes']) for root in blueprint['roots'])
    budget = 400*(observations+len(blueprint['roots']))+800
    path = directory/(arm+'-paired-data.pth')
    data = l.recollect_local_design(policy, blueprint, path, simulation_budget=budget,
        workers=workers, inference_batch=batch, inference_device='cpu')
    if (not data['completed'] or data['next_design_root'] != len(blueprint['roots'])
            or data['measured_terminal_observations'] != observations):
        raise ValueError('The independent acquisition design was not completely remeasured.')
    for actual, design in zip(data['roots'], blueprint['roots']):
        observed = l.observed_programs(design)
        if set(actual['outcomes']) != set(observed['outcomes']):
            raise ValueError('The measured observations differ from the frozen design.')
    data.update(training_data=str(path), r20_replication=dict(training_seed=seed,
        manifest_sha256=digest(MANIFEST), inherited_labels=False, inherited_states=False,
        measurement_rule='remeasure every frozen design observation under the own predecessor',
        historical_observation_reuse='replaced by new physical measurement; counted in actual simulation steps'))
    save(data, path)
    continuation = l.paired_continuation_bank(data)
    combined = dict(bank, records=bank['records']+continuation['records'],
        state_labels=bank['state_labels']+continuation['state_labels'],
        paired_continuation_state_labels=continuation['state_labels'],
        training_data=[str(bank_path), str(path)])
    return data, combined


def ordinary_execution(model, template, data):
    """Keep the audited r20 absolute-return deployment path with fresh BCE weights."""
    original = template['successor_task_value']
    fresh = model['successor_task_value']
    options = list(fresh['options'])
    roots = data['roots']
    shape = (len(roots), len(options))
    recipe = {k: copy.deepcopy(v) for k, v in original.items() if k not in
              ('states', 'contexts', 'controls', 'options', 'corrections', 'member_corrections',
               'uncertainty', 'observed', 'factual_values', 'continuation_contexts',
               'continuation_controls', 'continuation_values', 'continuation_remaining')}
    recipe.update(states=fresh['states'], contexts=torch.as_tensor(np.asarray([r['context'] for r in roots])),
        controls=torch.as_tensor(np.asarray([r['effect_controls'] for r in roots])), options=options,
        corrections=np.zeros(shape, np.float32), member_corrections=np.zeros((3, *shape), np.float32),
        uncertainty=np.zeros(shape, np.float32), observed=np.zeros(shape, bool), factual_values=None,
        continuation_contexts=None, continuation_controls=None, continuation_values=None,
        continuation_remaining=None, ordinary=True, temporal_value_bounds=False,
        joint_factual_calibration=False, conditional_strata=False, inference_device='cuda:0')
    model['successor_task_value'] = recipe
    return model


def fit_repeat(manifest, arm, seed, data, bank):
    l = sys.modules['local_intervention_value']
    pair = (arm, 'ordinary_no_M1' if arm == 'F01' else 'ordinary_with_M1')
    paths = {}
    for label in pair:
        path = OUTPUT/f'seed{seed}'/(label+'.pth')
        if path.exists():
            saved = read(path)
            if (saved.get('r20_replication', {}).get('completed')
                    and saved['successor_task_fit']['predecessor_sha256'] == data['predecessor_sha256']):
                paths[label] = path
                continue
            raise ValueError('An existing repeated learner lacks completed matching provenance.')
        resource_wait()
        original = read(manifest['models'][label]['path'])
        ordinary = label.startswith('ordinary')
        history = []
        native_print = getattr(l, 'print', print)
        def fit_progress(value, **kwargs):
            if isinstance(value, dict) and value.get('phase') == 'joint_potential_fit':
                history.append(dict(value))
            return native_print(value, **kwargs)
        l.print = fit_progress
        try:
            temporary = path.with_name(path.stem+'-fitting.pth')
            model = l.fit_joint_potential(data, temporary, factual_bank=bank,
                                         **fit_parameters(original, seed, ordinary))
        finally:
            l.print = native_print
        if ordinary:
            model = ordinary_execution(model, original, data)
        else:
            model['successor_task_value']['inference_device'] = 'cuda:0'
        model['r20_replication'] = dict(completed=True, training_seed=seed,
            manifest_sha256=digest(MANIFEST), inherited_labels=False, inherited_value_weights=False,
            heldout_used_for_training=False, value_initialization='fresh', objective_history=history)
        save(model, path)
        temporary.unlink()
        paths[label] = path
    proposed, ordinary = (read(paths[label]) for label in pair)
    for field in ('training_observation_sha256', 'updates', 'member_updates', 'width',
                  'predecessor_sha256', 'simulation_steps'):
        if proposed['successor_task_fit'][field] != ordinary['successor_task_fit'][field]:
            raise ValueError('The paired learners are not matched: ' + field)
    return paths


def repeat(manifest, seeds, workers, batch):
    verification = OUTPUT/'verification/training-replay.pth'
    for seed in seeds:
        sources = source_models(manifest, seed)
        paths = dict(sources)
        for arm, predecessor in (('F01', 'F00'), ('F11', 'F10')):
            resource_wait()
            data, bank = repeated_data(manifest, arm, seed, sources[predecessor], workers, batch)
            while not verification.exists():
                print(dict(phase='waiting_for_frozen_training_replay', seed=seed), flush=True)
                time.sleep(30)
            checked = read(verification)
            if not checked['completed'] or checked['runtime_sha256'] != manifest['runtime_sha256']:
                raise ValueError('The r20 fitting replay is not verified.')
            paths.update(fit_repeat(manifest, arm, seed, data, bank))
        # The fitting process has initialized CUDA. Start evaluation in a fresh
        # interpreter so its CPU parent can safely fork independent GPU actors.
        subprocess.run([sys.executable, '-X', 'utf8', '-u', __file__, 'seed-test',
                        '--seeds', str(seed), '--workers', '2'], check=True)
        save(dict(completed=True, seed=seed, models={k: dict(path=str(v), sha256=digest(v))
                  for k, v in paths.items()}, manifest_sha256=digest(MANIFEST)),
             OUTPUT/f'seed{seed}'/'result.pth')
        print(dict(phase='r20_independent_pipeline_completed', seed=seed), flush=True)


def seed_test(manifest, seeds, workers):
    for seed in seeds:
        source_models(manifest, seed)
        for label in (*LABELS, 'M1_independent'):
            path = OUTPUT/f'seed{seed}'/(label+'.pth')
            if not path.exists():
                raise ValueError('The independent pipeline is missing a model: ' + str(path))
            resource_wait()
            evaluate(path, manifest['heldout_tasks'], OUTPUT/f'seed{seed}'/(label+'-heldout.pth'),
                     manifest, workers=workers, role='independent-pipeline-heldout')


def baseline_runtime():
    """Freeze already-audited original baselines independently of the new method."""
    path = OUTPUT/'baseline-runtime.pth'
    if not path.exists():
        source = (ROOT/'train/mechanism_controls.py').read_text(encoding='utf8')
        nodes = ast.parse(source).body
        names = {'AuthorSourceRepeatPolicy', 'validate_author_source_repeat', 'author_source_repeat_episode'}
        helpers = '\n\n'.join(ast.get_source_segment(source, node) for node in nodes
                              if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
        endpoint = (ROOT/'train/r20_baseline_comparison.py').read_text(encoding='utf8')
        save(dict(helpers=helpers, endpoint=endpoint,
                  helpers_sha256=hashlib.sha256(helpers.encode()).hexdigest(),
                  endpoint_sha256=hashlib.sha256(endpoint.encode()).hexdigest()), path)
    frozen = read(path)
    m = sys.modules['mechanism_controls']
    m._AUTHOR_REPEAT_POLICIES = {}
    exec(compile(frozen['helpers'], str(ROOT/'train/mechanism_controls.py'), 'exec'), m.__dict__)
    endpoint = types.ModuleType('r20_endpoint_runtime')
    endpoint.__file__ = str(ROOT/'train/r20_baseline_comparison.py')
    sys.modules[endpoint.__name__] = endpoint
    exec(compile(frozen['endpoint'], endpoint.__file__, 'exec'), endpoint.__dict__)
    return frozen


def baseline_episode(job):
    family, location, identity, scene, future = job
    if family in ('sac', 'rp', 'arboids'):
        return sys.modules['mechanism_controls'].author_source_repeat_episode(
            (scene, future, location, identity))
    return sys.modules['r20_endpoint_runtime'].evaluation_episode((family, location, scene, future))


def baseline_test(manifest, workers):
    frozen = baseline_runtime()
    pending = [('boids', None)] + [(family, seed) for family in ('sac', 'rp', 'arboids', 'mappo', 'maac')
                                  for seed in SEEDS]
    while pending:
        remaining = []
        for family, seed in pending:
            resource_wait()
            label = family if seed is None else f'{family}-seed{seed}'
            if family in ('sac', 'rp', 'arboids'):
                directory = BASE/'independent-training'/f'source-six-{family}-seed{seed}-20261021'
                training, _ = sys.modules['mechanism_controls'].validate_author_source_repeat(directory)
                location = str(directory)
                policy = str(directory/'best-development.pth')
                identity = training['selected_checkpoint_sha256']
            elif family == 'boids':
                location = policy = identity = None
                training = dict(completed=True, trainable=False)
            else:
                directory = BASE/'r20-six-baseline-comparison'/label
                if not (directory/'training-result.pth').exists():
                    remaining.append((family, seed))
                    continue
                training = read(directory/'training-result.pth')
                if not training['completed'] or training['steps'] != training['budget']:
                    remaining.append((family, seed))
                    continue
                location = policy = str(directory/'best.pth')
                identity = digest(policy)
                if identity != training['best_sha256']:
                    raise ValueError('An original baseline changed after checkpoint selection.')
            path = OUTPUT/'baselines'/label/'heldout.pth'
            old = read(path) if path.exists() else None
            tasks = manifest['heldout_tasks']
            compatible = {digest(OUTPUT/'baseline-runtime.pth')}
            if family == 'boids':
                compatible.update(frozen.get('compatible_boids_runtime_sha256', ()))
            if old and (old['tasks'] != tasks or old['policy_sha256'] != identity
                        or old['baseline_runtime_sha256'] not in compatible):
                raise ValueError('An original-baseline endpoint identity changed.')
            if old and old['completed']:
                continue
            rows = old['rows'] if old else [None]*len(tasks)
            jobs = [(i, (family, location, identity, scene, future))
                    for i, (scene, future) in enumerate(tasks) if rows[i] is None]
            with ProcessPoolExecutor(max_workers=workers, initializer=torch.set_num_threads,
                                     initargs=(1,)) as pool:
                futures = {pool.submit(baseline_episode, job): i for i, job in jobs}
                for future in as_completed(futures):
                    rows[futures[future]] = future.result()
                    finished = [row for row in rows if row is not None]
                    if len(finished) % 32 == 0 or len(finished) == len(tasks):
                        record = dict(algorithm=family, training_seed=seed, policy=policy,
                            policy_sha256=identity, source_training=training, tasks=tasks, rows=rows,
                            completed=len(finished) == len(tasks), evaluated=len(finished),
                            success=sum(row['success'] for row in finished),
                            baseline_runtime_sha256=digest(OUTPUT/'baseline-runtime.pth'),
                            manifest_sha256=digest(MANIFEST),
                            evidence_role='original baseline on unseen common endpoint tasks',
                            includes_mechanism_one=False, includes_mechanism_two=False)
                        save(record, path)
                        print(dict(phase='r20_original_baseline_heldout', method=label,
                                   evaluated=len(finished), success=record['success']), flush=True)
        pending = remaining
        if pending:
            print(dict(phase='waiting_for_original_baseline_training', pending=pending), flush=True)
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('verify', 'fixed-test', 'replay-fit', 'repeat',
                                             'seed-test', 'baseline-test'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    manifest = read(MANIFEST)
    runtime(manifest)
    resource_wait()
    if args.operation == 'verify':
        verify_reference(manifest, args.workers)
    elif args.operation == 'fixed-test':
        fixed_test(manifest, args.workers)
    elif args.operation == 'replay-fit':
        try:
            replay_fit(manifest)
        except Exception as error:
            save(dict(completed=False, error=repr(error), runtime_sha256=manifest['runtime_sha256']),
                 OUTPUT/'verification/training-replay.pth')
            raise
    elif args.operation == 'seed-test':
        seed_test(manifest, args.seeds, args.workers)
    elif args.operation == 'baseline-test':
        baseline_test(manifest, args.workers)
    else:
        repeat(manifest, args.seeds, args.workers, args.batch)


if __name__ == '__main__':
    main()
