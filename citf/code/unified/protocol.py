"""One identity and metric contract for training, evaluation and paper results."""
from dataclasses import asdict, replace
import ast
import hashlib
import json
import math
from pathlib import Path

from . import IMPLEMENTATION_VERSION
from .environment import TaskConfig

SCHEMA = 1
METRICS_VERSION = 'exclusive-outcomes-capture-time-v1'
VRX_ADAPTER_CONTRACT = 'first-terminal-request-isolation-all-cycles-v1'
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
             'policy_lr_decay_steps', 'value_terminal_fraction', 'value_importance_exponent',
             'value_huber_delta', 'value_huber_start')
RUNTIME_ARGUMENTS = {'out', 'resume', 'init_from', 'fork_from', 'stop_at', 'device', 'registry', 'previous'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def policy_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        result.update(digest([name, str(tensor.dtype), list(tensor.shape)]).encode())
        result.update(tensor.numpy().tobytes())
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


def collection_updates(transitions, recipe, start=0):
    """CITF's vector-collection clock, including evaluation and 500k boundaries."""
    position, updates = start, 0
    while position < transitions:
        boundary = min(transitions, (position//recipe['eval_interval']+1)*recipe['eval_interval'])
        if position < 500000:
            boundary = min(boundary, 500000)
        position = min(position+recipe['envs'], boundary)
        if position >= recipe['start_learning']:
            updates += recipe['updates_per_step']
    return updates


def learning_counts(method, transitions, recipe, updates, prefix=None):
    baseline = method in METHODS[3:]
    if prefix is not None:
        if baseline or not 0 <= prefix['transitions'] <= transitions:
            raise ValueError('A development fork must retain a valid CITF training prefix.')
        prior = prefix['update_counts']
        expected = prior['critic']+collection_updates(transitions, recipe, start=prefix['transitions'])
        if updates != expected:
            raise ValueError(f'Incomplete fork learning budget: {updates} updates, expected {expected}.')
        policy_updates = updates//2-prior['critic']//2
        return dict(critic=updates, **{key: prior[key]+policy_updates for key in ('endpoint', 'generator', 'target')})
    expected = (baseline_updates(transitions, recipe['start_learning'], recipe['updates_per_joint_transition'])
                if baseline else collection_updates(transitions, recipe))
    if updates != expected:
        raise ValueError(f'Incomplete learning budget: {updates} updates, expected {expected}.')
    if baseline:
        return dict(critic=updates, actor=updates, temperature=updates, target=updates)
    return dict(critic=updates, endpoint=updates//2, generator=updates//2, target=updates//2)


def development_prefix(run):
    initialization = run.get('initialization')
    return (initialization if isinstance(initialization, dict)
            and initialization.get('kind') == 'development_fork' else None)


def validate_optimizer_counts(payload, keys, counts):
    optimizers = payload.get('optimizers', [])
    if len(optimizers) != len(keys):
        raise ValueError('Checkpoint is missing an optimizer state.')
    for key, optimizer in zip(keys, optimizers):
        states = optimizer['state']
        parameters = [p for group in optimizer['param_groups'] for p in group['params']]
        expected = counts[key]
        if (not parameters or len(set(parameters)) != len(parameters)
                or set(states) != (set(parameters) if expected else set())
                or any(float(state.get('step', -1)) != expected for state in states.values())):
            raise ValueError(f'The actual {key} optimizer states differ from its recorded update count.')


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
    from .networks import ModelConfig, TeamCritic
    import torch
    cfg = ModelConfig(width=chosen['width'], layers=chosen['layers'], neighbors=chosen['neighbors'])
    # Count the executing architecture without allocating weights or advancing
    # the training RNG; the control's effective path is checked separately.
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        full_count = sum(p.numel() for p in TeamCritic(cfg).parameters())
        local_count = sum(p.numel() for p in TeamCritic(replace(cfg, action_interaction=False)).parameters())
    difference = abs(local_count-full_count)/full_count
    if difference > .05:
        raise ValueError('The registered no-action critic must remain within 5% of the full critic capacity.')
    registry = dict(schema=SCHEMA, implementation_version=IMPLEMENTATION_VERSION,
                    source=source_identity(), seeds=SEEDS, methods=list(METHODS),
                    team_transitions=1000000, validation_seed=73001, main_test_seed=74001,
                    agility_seed=74004, team_size_seed=74005, policy_seeds=list(range(84001, 84006)),
                    metrics_version=METRICS_VERSION, task=recipe['task'],
                    citf_recipe={k: chosen[k] for k in FULL_KEYS}, baseline_recipe=base,
                    baseline_entropy=dict(arboids=-3., rp=-2., sac=-2.),
                    critic_capacity=dict(full=full_count, no_action=local_count,
                                         relative_difference=difference, maximum_relative_difference=.05),
                    recipe_origin_version=recipe['implementation_version'],
                    vrx_training_seed=2101, alternating_training_seed=2101,
                    vrx_adapter_contract=VRX_ADAPTER_CONTRACT)
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
    if registry.get('vrx_adapter_contract') != VRX_ADAPTER_CONTRACT:
        raise ValueError('The formal study must register the reviewed VRX event and control contract.')
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
        if arguments.get('init_from') or arguments.get('fork_from'):
            raise ValueError('Formal methods are trained independently from scratch.')
    result = dict(schema=SCHEMA, method=method, seed=arguments['seed'], arguments=canonical,
                  task=task, model=model_definition(arguments), source=source_identity(),
                  registry_id=None if registry is None else registry['id'],
                  initialization=None if not arguments.get('init_from') else file_digest(arguments['init_from']))
    result['id'] = digest(result)
    return result


def bind_run(args, task, resumed=None, forked=None):
    if args.registry is None and resumed is not None:
        args.registry = resumed['arguments'].get('registry')
    registry = load_registry(args.registry) if args.registry else None
    run = run_identity(vars(args), asdict(task), registry)
    if forked is not None:
        if resumed is not None or args.stage != 'development':
            raise ValueError('Full-state branching is only available for a new development run.')
        run['initialization'] = dict(kind='development_fork',
            parent_run_id=forked['run_identity']['id'], parent_source=forked['run_identity']['source'],
            transitions=forked['transitions'], update_counts=forked['update_counts'],
            source_checkpoint=str(Path(args.fork_from).resolve()))
        run['id'] = digest({k: v for k, v in run.items() if k != 'id'})
    if resumed is not None:
        run['initialization'] = resumed.get('run_identity', {}).get('initialization')
        run['id'] = digest({k: v for k, v in run.items() if k != 'id'})
    if resumed is not None and resumed.get('run_identity') != run:
        raise ValueError('Exact resume must retain the complete source, run identity and budget.')
    return run


def require_standard_controller(payload):
    variant = payload.get('controller_variant', payload.get('continuation', {}).get('variant', 'ordinary'))
    if variant != 'ordinary' or payload.get('representation', 'ordinary_refit') != 'ordinary_refit':
        raise ValueError('This checkpoint requires its candidate-specific controller; loading it as ordinary CITF would change its decisions.')


def validate_checkpoint(payload, registry=None, method=None, seed=None, transitions=None, current_source=True):
    require_standard_controller(payload)
    run = payload.get('run_identity')
    if not run or run.get('id') != digest({k: v for k, v in run.items() if k != 'id'}):
        raise ValueError('Checkpoint has no verified run identity; preserve it as historical development evidence.')
    if current_source and (run['source'] != source_identity() or payload.get('implementation_version') != IMPLEMENTATION_VERSION):
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
    prefix = development_prefix(run)
    if prefix is not None and run['arguments']['stage'] != 'development':
        raise ValueError('A development fork cannot be reused as an independent formal training run.')
    counts = learning_counts(run['method'], payload['transitions'], run['arguments'], payload['updates'], prefix)
    if payload.get('update_counts') != counts:
        raise ValueError('Checkpoint optimizer update counts are incomplete.')
    keys = (('actor', 'critic', 'temperature') if run['method'] in METHODS[3:]
            else ('critic', 'endpoint', 'generator'))
    validate_optimizer_counts(payload, keys, counts)
    recipe = run['arguments']
    expected_learner = (dict(gamma=recipe['gamma'], tau=recipe['target_tau'], target_entropy=recipe['target_entropy'])
                        if run['method'] in METHODS[3:] else dict(gamma=.99, tau=recipe['target_tau'], policy_delay=2,
                                                                value_huber_delta=recipe['value_huber_delta']))
    if payload.get('learner_recipe') != expected_learner:
        raise ValueError('Checkpoint learning rule differs from its registered recipe.')
    evaluation = payload.get('evaluation')
    if not evaluation or evaluation.get('transitions') != payload['transitions']:
        raise ValueError('Checkpoint is missing its own validation record.')
    validate_learning_node(evaluation, run)
    if (evaluation['updates'] != payload['updates']
            or evaluation['policy_id'] != policy_digest(payload['model'])):
        raise ValueError('Checkpoint weights or update count differ from its recorded validation.')
    return run


def checkpoint_identity(path, payload):
    return dict(sha256=file_digest(path), policy_id=policy_digest(payload['model']), run_id=payload['run_identity']['id'],
                transitions=payload['transitions'], updates=payload['updates'],
                update_counts=payload['update_counts'], training_source=payload['run_identity']['source'],
                validation_id=digest(payload['evaluation']))


def read_endpoint(path, registry, method, seed, transitions=1000000):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    validate_checkpoint(payload, registry, method, seed, transitions)
    return checkpoint_identity(path, payload)


def validate_learning_node(row, run):
    recipe = run['arguments']
    if (row.get('run_id') != run['id'] or row.get('role') != 'validation'
            or row.get('research_stage') != recipe['stage']
            or row['task'] != run['task'] or row['scene_seed'] != recipe['eval_seed']
            or row['policy_seed'] != recipe['policy_eval_seed'] or row['tasks'] != recipe['eval_count']):
        raise ValueError('Learning-curve node belongs to a different run, task or validation bank.')
    policy_id = row.get('policy_id', '')
    if len(policy_id) != 64 or any(ch not in '0123456789abcdef' for ch in policy_id):
        raise ValueError('Learning-curve node has no model-content identity.')
    if row.get('update_counts') != learning_counts(run['method'], row['transitions'], recipe, row['updates'], development_prefix(run)):
        raise ValueError('Learning-curve node has inconsistent optimizer counts.')
    validate_metrics(row)


def validate_learning_curve(path, run, until=1000000, endpoint=None):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    recipe = run['arguments']
    prefix = development_prefix(run)
    start = 0 if prefix is None else prefix['transitions']
    grid = list(range((start//recipe['eval_interval']+1)*recipe['eval_interval'], until+1, recipe['eval_interval']))
    grid = sorted(set(grid) | ({500000} if start < 500000 <= until else set())
                  | ({until} if until == recipe['steps'] and until > start else set()))
    if [row['transitions'] for row in rows] != grid:
        raise ValueError('The learning curve must retain every registered node once, in order.')
    for row in rows:
        validate_learning_node(row, run)
    if endpoint is not None and (not rows or endpoint.get('run_id') != run['id']
            or endpoint.get('transitions') != until or endpoint.get('policy_id') != rows[-1]['policy_id']
            or endpoint.get('validation_id') != digest(rows[-1])):
        raise ValueError('The learning-curve endpoint differs from the final checkpoint and its validation.')
    return rows


def outcome_metrics(rows, count, scene_seed, horizon):
    if (count < 1 or len(rows) != count or any(type(row['scene_index']) is not int for row in rows)
            or sorted(row['scene_index'] for row in rows) != list(range(count))):
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
        if row['timeout'] and row['seconds'] < horizon-1e-3:
            raise ValueError('An unfinished episode cannot be counted as a timeout success.')
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
    identity = dict(schema=SCHEMA, registry_id=registry['id'], kind=kind, method=method, training_seed=seed,
                checkpoint=checkpoint, inference_source=registry['source'], task=task,
                scene_seed=scene_seed, scenario_indices=list(range(count)), policy_seed=policy_seed,
                metrics_version=METRICS_VERSION)
    if kind == 'vrx':
        identity['adapter_contract'] = registry['vrx_adapter_contract']
    return identity


def attach_identity(result, identity):
    result.update(identity=identity, evaluation_id=digest(identity), research_stage='formal')
    return result


def validate_result(result, expected):
    if (result.get('identity') != expected or result.get('evaluation_id') != digest(expected)
            or result.get('research_stage') != 'formal' or result.get('task') != expected['task']
            or result.get('tasks') != len(expected['scenario_indices'])
            or result.get('scene_seed') != expected['scene_seed'] or result.get('policy_seed') != expected['policy_seed']):
        raise ValueError('Cached result belongs to a different checkpoint, method, task, source or scenario bank.')
    if expected['kind'] == 'numeric' and result.get('method') != expected['method']:
        raise ValueError('Result method label differs from the verified model.')
    if expected['kind'] == 'vrx' and result.get('adapter_source', {}).get('contract') != expected['adapter_contract']:
        raise ValueError('VRX results require the reviewed event-history and asynchronous-control adapter.')
    validate_metrics(result)
    return result


ATTACKER_RECIPE = dict(envs=64, batch=256, updates_per_step=4, capacity=1000000,
                       start_learning=5000, lr=1e-4)


def phase_identity(arguments, task, root_run, previous, attacker_recipe, starting_counts, starting_updates, registry):
    phase = arguments['phase']
    active = 'defender' if phase in (3, 5) else 'attacker'
    recipe = root_run['arguments']
    if arguments['seed'] != root_run['seed'] or root_run['method'] != 'full':
        raise ValueError('The alternating chain uses the complete method and its original seed.')
    inherited = recipe if active == 'defender' else attacker_recipe
    if any(arguments[key] != inherited[key] for key in ATTACKER_RECIPE):
        raise ValueError('An active side must retain its complete registered training recipe.')
    if arguments['stage'] == 'formal':
        expected_root = run_identity(registered_arguments(registry, 'full', 2101), registry['task'], registry)
        if root_run != expected_root or attacker_recipe != ATTACKER_RECIPE:
            raise ValueError('The formal chain must retain the registered full method and attacker recipe.')
        required = dict(steps=500000, seed=2101, eval_count=100, eval_seed=73003,
                        test_seed=74003, policy_eval_seed=83003, eval_interval=50000)
        if any(arguments[key] != value for key, value in required.items()) or task != registry['task']:
            raise ValueError('The formal alternating phase differs from the registered protocol.')
    result = dict(schema=SCHEMA, phase=phase, active_side=active, root_run=root_run,
                  previous_checkpoint=previous, arguments=canonical_arguments(arguments), task=task,
                  attacker_recipe=attacker_recipe, starting_counts=starting_counts.copy(),
                  starting_updates=starting_updates.copy(), source=source_identity(),
                  registry_id=None if registry is None else registry['id'])
    result['id'] = digest(result)
    return result


def phase_counts(identity, transitions, defender_updates, attacker_updates):
    active = identity['active_side']
    counts = identity['starting_counts'].copy()
    counts[active] += transitions
    expected = identity['starting_updates'].copy()
    expected[active] += collection_updates(transitions, identity['arguments'])
    if expected != dict(defender=defender_updates, attacker=attacker_updates):
        raise ValueError('The alternating learner updates differ from its inherited active-side recipe.')
    updates = dict(defender=dict(critic=defender_updates, endpoint=defender_updates//2,
                                generator=defender_updates//2, target=defender_updates//2),
                   attacker=dict(critic=attacker_updates, actor=attacker_updates,
                                 temperature=attacker_updates, target=attacker_updates))
    return counts, updates


def validate_phase_checkpoint(payload, registry=None, phase=None, completed=True):
    identity = payload.get('phase_identity')
    if not identity or identity.get('id') != digest({k: v for k, v in identity.items() if k != 'id'}):
        raise ValueError('Missing or changed alternating-phase identity.')
    if (identity['source'] != source_identity() or identity['arguments'] != canonical_arguments(payload['arguments'])
            or payload['phase'] != identity['phase'] or payload['task'] != identity['task']
            or payload['defender_recipe'] != identity['root_run']['arguments']
            or payload['attacker_recipe'] != identity['attacker_recipe']
            or payload['defender']['config'] != identity['root_run']['model']):
        raise ValueError('The alternating phase differs from its source, task or inherited recipe.')
    if phase is not None and payload['phase'] != phase or completed and payload['transitions'] != 500000:
        raise ValueError('Use the preceding completed 500k phase.')
    if registry is not None:
        expected = phase_identity(payload['arguments'], payload['task'], identity['root_run'],
                                  identity['previous_checkpoint'], payload['attacker_recipe'],
                                  identity['starting_counts'], identity['starting_updates'], registry)
        if identity != expected:
            raise ValueError('Wrong registered alternating chain.')
    counts, updates = phase_counts(identity, payload['transitions'], payload['defender']['updates'], payload['attacker']['updates'])
    if payload['side_transitions'] != counts or payload.get('update_counts') != updates:
        raise ValueError('Alternating phase has inconsistent cumulative transitions or optimizer counts.')
    validate_optimizer_counts(payload['defender'], ('critic', 'endpoint', 'generator'), updates['defender'])
    validate_optimizer_counts(payload['attacker'], ('actor', 'critic', 'temperature'), updates['attacker'])
    return identity


def phase_checkpoint_identity(path, payload):
    return dict(sha256=file_digest(path), phase_id=payload['phase_identity']['id'], phase=payload['phase'],
                root_run_id=payload['phase_identity']['root_run']['id'], transitions=payload['transitions'],
                side_transitions=payload['side_transitions'], update_counts=payload['update_counts'],
                previous_checkpoint=payload['phase_identity']['previous_checkpoint'])


def chain_evaluation_identity(registry, phase, checkpoint, task, role):
    root = run_identity(registered_arguments(registry, 'full', 2101), registry['task'], registry)
    if task != registry['task'] or phase not in range(1, 6) or checkpoint['transitions'] != 500000:
        raise ValueError('Invalid fixed-stage alternating evaluation.')
    if phase == 1:
        valid = checkpoint.get('run_id') == root['id'] and role == 'fixed-stage-endpoint'
    else:
        valid = (checkpoint.get('root_run_id') == root['id'] and checkpoint.get('phase') == phase
                 and (role == 'fixed-stage-endpoint' or phase == 5 and role == 'retention'))
    if not valid:
        raise ValueError('Wrong alternating checkpoint or endpoint role.')
    return dict(schema=SCHEMA, registry_id=registry['id'], kind='alternating', phase=phase,
                role=role, training_seed=2101, checkpoint=checkpoint, inference_source=registry['source'],
                task=task, scene_seed=74003, scenario_indices=list(range(100)), policy_seed=83003,
                opponent='APF' if phase == 1 or role == 'retention' else 'learned', metrics_version=METRICS_VERSION)


def vrx_source_identity(directory):
    directory = Path(directory)
    required = {'unified_vrx.py', 'run_experiment.py', 'run_batch.py', 'mappo_controller.py',
                'tad_vrx_experiment.py', 'models.py'}
    paths = {p.name: p for p in directory.glob('*.py')}
    if not required.issubset(paths):
        raise ValueError('Freeze and deploy the complete reviewed VRX adapter before running formal episodes.')
    tree = ast.parse(paths['unified_vrx.py'].read_text(encoding='utf-8-sig'))
    contracts = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'ADAPTER_CONTRACT' for target in node.targets)]
    if contracts != [VRX_ADAPTER_CONTRACT]:
        raise ValueError('Legacy VRX adapters lack the reviewed first-event, request-isolation and cycle-count contract.')
    files = {name: file_digest(path) for name, path in sorted(paths.items())}
    return dict(id=digest(files), files=files, contract=VRX_ADAPTER_CONTRACT)


def vrx_rows(episodes, identity, adapter_source):
    rows = []
    if adapter_source.get('contract') != identity.get('adapter_contract') or identity.get('adapter_contract') != VRX_ADAPTER_CONTRACT:
        raise ValueError('VRX episodes must retain the adapter contract registered before formal training.')
    if len(episodes) != len(identity['scenario_indices']):
        raise ValueError('VRX must retain every registered episode directory.')
    for expected_index, episode in zip(identity['scenario_indices'], episodes):
        index = episode['scene_index']
        if (type(index) is not int or index != expected_index
                or not episode.get('passed') or episode.get('identity') != identity
                or episode.get('adapter_source') != adapter_source
                or episode.get('adapter_contract') != VRX_ADAPTER_CONTRACT
                or episode.get('policy_seed') != identity['policy_seed']+index
                or episode.get('scene_seed') != identity['scene_seed'] or episode.get('task') != identity['task']):
            raise ValueError('VRX episode has a different source, checkpoint, task or policy random stream.')
        code = episode['outcome_code']
        row = dict(scene_seed=episode['scene_seed'], scene_index=index, outcome=code,
                   seconds=episode['simulation_seconds'], success=int(episode['success']),
                   breach=int(code == 1), collision=int(code == 2), capture=int(code == 3), timeout=int(code == 4))
        rows.append(row)
    return rows
