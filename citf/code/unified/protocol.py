"""One identity and metric contract for training, evaluation and paper results."""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from . import IMPLEMENTATION_VERSION
from .environment import TaskConfig

SCHEMA = 1
METRICS_VERSION = 'exclusive-outcomes-capture-time-v1'
METHODS = ('full', 'no_action', 'no_feedback', 'arboids', 'rp', 'sac')
SEEDS = [2101, 2102, 2103, 2104, 2105]
BASELINE_RECIPE = dict(envs=64, batch=4096, width=512, capacity=1000000,
                       start_learning=50000, updates_per_joint_transition=1.,
                       lr=1e-4, gamma=.99, target_tau=.005)
FULL_KEYS = ('envs', 'updates_per_step', 'lr', 'batch', 'capacity', 'start_learning',
             'width', 'layers', 'neighbors', 'samples', 'search_steps', 'search_rate',
             'target_tau', 'temperature_start', 'temperature_end', 'temperature_steps',
             'endpoint_noise_start', 'endpoint_noise_end', 'exploration_steps', 'feedback',
             'feedback_start', 'feedback_ramp', 'policy_lr_end', 'policy_lr_decay_start',
             'policy_lr_decay_steps', 'value_terminal_fraction', 'value_importance_exponent')
RUNTIME_ARGUMENTS = {'out', 'resume', 'init_from', 'device', 'registry', 'previous'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def source_identity():
    directory = Path(__file__).resolve().parent
    files = {f'unified/{p.name}': file_digest(p) for p in sorted(directory.glob('*.py'))}
    from .baselines import author_networks
    files['author/networks.py'] = file_digest(author_networks().__file__)
    return dict(id=digest(files), files=files, implementation_version=IMPLEMENTATION_VERSION,
                author_commit='9eb6df464808af6b9729f39644b4d658e1695e2d')


def baseline_updates(transitions, start_learning, ratio=1.):
    # The author's first learn call follows transition start_learning itself.
    return math.floor(max(0, transitions-start_learning+1)*ratio)


def collection_updates(transitions, recipe):
    """CITF's vector-collection clock, including evaluation and 500k boundaries."""
    position, updates = 0, 0
    while position < transitions:
        boundary = min(transitions, (position//recipe['eval_interval']+1)*recipe['eval_interval'])
        if position < 500000:
            boundary = min(boundary, 500000)
        position = min(position+recipe['envs'], boundary)
        if position >= recipe['start_learning']:
            updates += recipe['updates_per_step']
    return updates


def learning_counts(method, transitions, recipe, updates):
    baseline = method in METHODS[3:]
    expected = (baseline_updates(transitions, recipe['start_learning'], recipe['updates_per_joint_transition'])
                if baseline else collection_updates(transitions, recipe))
    if updates != expected:
        raise ValueError(f'Incomplete learning budget: {updates} updates, expected {expected}.')
    if baseline:
        return dict(critic=updates, actor=updates, temperature=updates, target=updates)
    return dict(critic=updates, endpoint=updates//2, generator=updates//2, target=updates//2)


def method_of(arguments):
    if 'method' in arguments:
        return arguments['method']
    if arguments.get('no_action_interaction'):
        if arguments.get('feedback', 0) <= 0:
            raise ValueError('The registered ablations each remove exactly one mechanism.')
        return 'no_action'
    return 'no_feedback' if arguments.get('feedback') == 0 else 'full'


def canonical_arguments(arguments):
    return {k: v for k, v in arguments.items() if k not in RUNTIME_ARGUMENTS}


def model_definition(arguments):
    if 'method' in arguments:
        return dict(method=arguments['method'], width=arguments['width'], delta=arguments['thrust_delta'])
    from .networks import ModelConfig
    return asdict(ModelConfig(width=arguments['width'], layers=arguments['layers'],
                             samples=arguments['samples'], neighbors=arguments['neighbors'],
                             search_steps=arguments['search_steps'], search_rate=arguments['search_rate'],
                             thrust_delta=arguments['thrust_delta'], feedback=arguments['feedback'],
                             action_interaction=not arguments['no_action_interaction']))


def registered_arguments(registry, method, seed):
    if method not in METHODS or seed not in registry['seeds']:
        raise ValueError('Unregistered method or training seed.')
    args = dict(seed=seed, steps=1000000, stage='formal', eval_interval=5000,
                eval_count=100, eval_seed=73001, policy_eval_seed=83001,
                thrust_delta=registry['task']['thrust_delta'],
                collision_penalty=registry['task']['collision_penalty'])
    if method in METHODS[:3]:
        args.update(registry['citf_recipe'])
        args.update(no_action_interaction=method == 'no_action')
        if method == 'no_feedback':
            args['feedback'] = 0.
    else:
        args.update(registry['baseline_recipe'], method=method,
                    target_entropy=-3. if method == 'arboids' else -2.)
    return args


def build_registry(recipe, baseline=None):
    chosen = recipe['arguments']
    if chosen.get('no_action_interaction') or chosen.get('feedback', 0) <= 0:
        raise ValueError('Select the complete-method recipe before freezing the study.')
    missing = set(FULL_KEYS)-chosen.keys()
    if missing:
        raise ValueError(f'The selected recipe is incomplete: {sorted(missing)}')
    base = dict(BASELINE_RECIPE) if baseline is None else baseline
    if base != BASELINE_RECIPE:
        raise ValueError('Formal public baselines retain the independently registered author learning recipe.')
    registry = dict(schema=SCHEMA, implementation_version=IMPLEMENTATION_VERSION,
                    source=source_identity(), seeds=SEEDS, methods=list(METHODS),
                    team_transitions=1000000, validation_seed=73001, main_test_seed=74001,
                    agility_seed=74004, team_size_seed=74005, policy_seeds=list(range(84001, 84006)),
                    metrics_version=METRICS_VERSION, task=recipe['task'],
                    citf_recipe={k: chosen[k] for k in FULL_KEYS}, baseline_recipe=base,
                    baseline_entropy=dict(arboids=-3., rp=-2., sac=-2.),
                    recipe_origin_version=recipe['implementation_version'],
                    vrx_training_seed=2101, alternating_training_seed=2101)
    expected_task = asdict(TaskConfig(thrust_delta=chosen['thrust_delta'],
                                    collision_penalty=chosen['collision_penalty']))
    if registry['task'] != expected_task:
        raise ValueError('The selected main training task differs from the common task.')
    registry['id'] = digest(registry)
    return registry


def load_registry(path, current_source=True):
    registry = json.loads(Path(path).read_text(encoding='utf-8'))
    if registry.get('id') != digest({k: v for k, v in registry.items() if k != 'id'}):
        raise ValueError('The study registry is missing or has changed; use a newly registered study.')
    if registry.get('schema') != SCHEMA or registry.get('metrics_version') != METRICS_VERSION:
        raise ValueError('Incompatible study or metric schema.')
    if current_source and registry['source'] != source_identity():
        raise ValueError('The executing source differs from the frozen study, including uncommitted changes.')
    return registry


def run_identity(arguments, task, registry=None):
    canonical = canonical_arguments(arguments)
    method = method_of(arguments)
    if arguments['stage'] == 'formal':
        if registry is None or canonical != registered_arguments(registry, method, arguments['seed']):
            raise ValueError('Formal training must use the exact registered method, seed and complete recipe.')
        if task != registry['task']:
            raise ValueError('The formal training task must match the registry.')
        if arguments.get('init_from'):
            raise ValueError('Formal methods are trained independently from scratch.')
    result = dict(schema=SCHEMA, method=method, seed=arguments['seed'], arguments=canonical,
                  task=task, model=model_definition(arguments), source=source_identity(),
                  registry_id=None if registry is None else registry['id'],
                  initialization=None if not arguments.get('init_from') else file_digest(arguments['init_from']))
    result['id'] = digest(result)
    return result


def bind_run(args, task, resumed=None):
    registry = load_registry(args.registry) if args.registry else None
    run = run_identity(vars(args), asdict(task), registry)
    if resumed is not None and resumed.get('run_identity') != run:
        raise ValueError('Exact resume must retain the complete source, run identity and budget.')
    return run


def validate_checkpoint(payload, registry=None, method=None, seed=None, transitions=None):
    run = payload.get('run_identity')
    if not run or run.get('id') != digest({k: v for k, v in run.items() if k != 'id'}):
        raise ValueError('Checkpoint has no verified run identity; preserve it as historical development evidence.')
    if run['source'] != source_identity() or payload.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise ValueError('Use the frozen training and inference source for this checkpoint.')
    if run['arguments'] != canonical_arguments(payload['arguments']) or run['task'] != payload['task']:
        raise ValueError('Checkpoint metadata differs from its registered run.')
    if run['model'] != model_definition(payload['arguments']):
        raise ValueError('Checkpoint model configuration differs from its run.')
    actual_model = (dict(method=payload['method'], width=payload['width'], delta=payload['task']['thrust_delta'])
                    if 'method' in payload else payload['config'])
    if actual_model != run['model']:
        raise ValueError('Checkpoint architecture or ablation has changed.')
    if registry is not None:
        expected = run_identity(registered_arguments(registry, method, seed), registry['task'], registry)
        if run != expected:
            raise ValueError('Wrong method, seed, source or recipe for this registered checkpoint.')
    if transitions is not None and payload['transitions'] != transitions:
        raise ValueError('Use the registered fixed-budget checkpoint.')
    counts = learning_counts(run['method'], payload['transitions'], run['arguments'], payload['updates'])
    if payload.get('update_counts') != counts:
        raise ValueError('Checkpoint optimizer update counts are incomplete.')
    return run


def checkpoint_identity(path, payload):
    return dict(sha256=file_digest(path), run_id=payload['run_identity']['id'],
                transitions=payload['transitions'], updates=payload['updates'],
                update_counts=payload['update_counts'], training_source=payload['run_identity']['source'])


def read_endpoint(path, registry, method, seed, transitions=1000000):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    validate_checkpoint(payload, registry, method, seed, transitions)
    return checkpoint_identity(path, payload)


def validate_learning_curve(path, run, until=1000000):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    recipe = run['arguments']
    grid = list(range(recipe['eval_interval'], until+1, recipe['eval_interval']))
    if [row['transitions'] for row in rows] != grid:
        raise ValueError('The learning curve must retain every registered node once, in order.')
    for row in rows:
        if (row.get('run_id') != run['id'] or row.get('role') != 'validation'
                or row['task'] != run['task'] or row['scene_seed'] != recipe['eval_seed']
                or row['policy_seed'] != recipe['policy_eval_seed'] or row['tasks'] != recipe['eval_count']):
            raise ValueError('Learning-curve node belongs to a different run, task or validation bank.')
        if row.get('update_counts') != learning_counts(run['method'], row['transitions'], recipe, row['updates']):
            raise ValueError('Learning-curve node has inconsistent optimizer counts.')
        validate_metrics(row)
    return rows


def outcome_metrics(rows, count, scene_seed, horizon):
    if len(rows) != count or sorted(row['scene_index'] for row in rows) != list(range(count)):
        raise ValueError('Scenario indices must cover the complete bank exactly once.')
    names = ('breach', 'collision', 'capture', 'timeout')
    for row in rows:
        if row['scene_seed'] != scene_seed or row['outcome'] not in (1, 2, 3, 4):
            raise ValueError('Wrong scenario bank or unfinished episode.')
        expected = {name: int(row['outcome'] == index+1) for index, name in enumerate(names)}
        expected['success'] = expected['capture']+expected['timeout']
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError('Every episode needs exactly one terminal outcome and the common success definition.')
        if not math.isfinite(row['seconds']) or not 0 < row['seconds'] <= horizon+1e-3:
            raise ValueError('Invalid episode termination time.')
    totals = {key: sum(row[key] for row in rows) for key in (*names, 'success')}
    captures = [r['seconds'] for r in rows if r['capture']]
    mean = sum(r['seconds'] for r in rows)/count
    return dict(tasks=count, **totals, success_rate=totals['success']/count,
                capture_rate=len(captures)/count, mean_seconds=mean, mean_termination_seconds=mean,
                capture_count=len(captures), mean_capture_seconds=sum(captures)/len(captures) if captures else None,
                metrics_version=METRICS_VERSION)


def validate_metrics(result):
    metrics = outcome_metrics(result['rows'], result['tasks'], result['scene_seed'], result['task']['duration'])
    for key, value in metrics.items():
        recorded = result.get(key)
        if isinstance(value, float):
            matches = isinstance(recorded, (int, float)) and math.isclose(recorded, value, rel_tol=1e-9, abs_tol=1e-9)
        else:
            matches = key in result and recorded == value
        if not matches:
            raise ValueError(f'Invalid result metric {key}: {recorded!r}, expected {value!r}.')


def evaluation_identity(registry, method, seed, checkpoint, task, scene_seed, policy_seed, count, kind='numeric'):
    if count != 100:
        raise ValueError('Formal evaluations retain all 100 registered scenes.')
    base = TaskConfig(**registry['task'])
    if kind == 'vrx':
        valid = (method in ('full', 'arboids') and seed == 2101 and scene_seed == 74002
                 and policy_seed == 84002 and task == asdict(replace(base, agility=2.25)))
    else:
        main = scene_seed == 74001 and task == asdict(base)
        agility = (scene_seed == 74004 and task == asdict(replace(base, agility=task['agility']))
                   and task['agility'] in (1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.))
        teams = (scene_seed == 74005 and task == asdict(replace(base, defenders=task['defenders']))
                 and task['defenders'] in range(2, 8))
        valid = (method in (*METHODS, 'boids') and (seed in SEEDS or method == 'boids' and seed is None)
                 and policy_seed == (84001 if seed is None else 84001+SEEDS.index(seed))
                 and (main or (agility or teams) and method not in ('no_action', 'no_feedback')))
    if not valid:
        raise ValueError('Evaluation differs from the registered task, method, scene bank or policy seed.')
    if (method == 'boids') != (checkpoint is None):
        raise ValueError('Learned evaluations require a verified checkpoint.')
    if checkpoint is not None:
        expected_run = run_identity(registered_arguments(registry, method, seed), registry['task'], registry)
        if checkpoint['run_id'] != expected_run['id'] or checkpoint['transitions'] != 1000000:
            raise ValueError('Wrong endpoint checkpoint for this method and training seed.')
    return dict(schema=SCHEMA, registry_id=registry['id'], kind=kind, method=method, training_seed=seed,
                checkpoint=checkpoint, inference_source=registry['source'], task=task,
                scene_seed=scene_seed, scenario_indices=list(range(count)), policy_seed=policy_seed,
                metrics_version=METRICS_VERSION)


def attach_identity(result, identity):
    result.update(identity=identity, evaluation_id=digest(identity), research_stage='formal')
    return result


def validate_result(result, expected):
    if (result.get('identity') != expected or result.get('evaluation_id') != digest(expected)
            or result.get('research_stage') != 'formal' or result.get('task') != expected['task']
            or result.get('scene_seed') != expected['scene_seed'] or result.get('policy_seed') != expected['policy_seed']):
        raise ValueError('Cached result belongs to a different checkpoint, method, task, source or scenario bank.')
    validate_metrics(result)
    return result
