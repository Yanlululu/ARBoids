"""Prospective mechanism, operating-range and real-Gazebo evidence for r20."""
from __future__ import annotations

import argparse
import ast
import copy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import torch
import r20_independent_evidence as e

ANALYSIS = e.OUTPUT/'analysis-manifest.pth'
MODELS = ('F00', 'F10', 'F01', 'F11')
_ANALYSIS = None


def initialize(opponent_helpers=None):
    if ANALYSIS.exists():
        return e.read(ANALYSIS)
    manifest = e.read(e.MANIFEST)
    models = {label: dict(kind='joint', **record) for label, record in manifest['models'].items()}
    for family in ('arboids', 'mappo'):
        if family == 'arboids':
            directory = e.BASE/'independent-training/source-six-arboids-seed42-20261021'
            record = e.read(directory/'source-repeat-result.pth')
            path = directory/'best-development.pth'
            identity = record['selected_checkpoint_sha256']
        else:
            directory = e.BASE/'r20-six-baseline-comparison/mappo-seed42'
            record = e.read(directory/'training-result.pth')
            path = directory/'best.pth'
            identity = record['best_sha256']
        if not record['completed'] or e.digest(path) != identity:
            raise ValueError('Freeze completed original baselines before scope and VRX evaluation.')
        models[family] = dict(kind='baseline', family=family, training_seed=42,
            directory=str(directory), path=str(path), sha256=identity)
    groups = {}
    for index, agility in enumerate((1., 1.5, 3., 3.5)):
        groups[f'agility-{agility:g}'] = [((e.seed32(91, index, i), 6, agility), None) for i in range(100)]
    for n in (3, 4, 5, 7, 8):
        groups[f'fleet-{n}'] = [((e.seed32(92, n, i), n, 2.25), None) for i in range(100)]
    groups['common-learned-opponent'] = [((e.seed32(93, cell, i), 6, agility), None)
        for cell, agility in enumerate((2., 2.5)) for i in range(100)]
    mechanism = [((e.seed32(94, cell, i), 6, agility), None)
        for cell, agility in enumerate((2., 2.5)) for i in range(20)]
    vrx = [dict(seed=e.seed32(95, i), defenders=6,
                agility=2. if i < 50 else 2.5, setting=1, duration=80., action_period=.2)
           for i in range(100)]
    old = {scene[0] for scene, _ in manifest['heldout_tasks']}
    old.update(root['scene'][0] for design in manifest['designs'].values() for root in design['roots'])
    new = [scene[0] for tasks in (*groups.values(), mechanism) for scene, _ in tasks]
    new.extend(row['seed'] for row in vrx)
    if len(set(new)) != len(new) or old.intersection(new):
        raise ValueError('The prospective diagnostic populations overlap.')
    opponent = e.BASE/'six-source-opponent-adaptation-r148/best-development-attacker.pth'
    attack = e.read(opponent)
    if not attack.get('completed'):
        raise ValueError('The common learned opponent must be frozen and fully trained.')
    source = (e.ROOT/'train/run_remote_evidence.py').read_text(encoding='utf8')
    names = {'public_attacker_observation', 'FrozenSourceAttacker',
             '_source_opponent_step', 'configure_source_attacker'}
    helpers = '\n\n'.join(ast.get_source_segment(source, node) for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names)
    if opponent_helpers is not None:
        helpers = opponent_helpers
    if sum(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in ast.parse(helpers).body) != 4:
        raise ValueError('The common-opponent executor is incomplete.')
    adapter = e.ROOT/'vrx/run_r20_evidence.py'
    analysis = dict(version=1, r20_manifest_sha256=e.digest(e.MANIFEST), models=models,
        scope_tasks=groups, mechanism_tasks=mechanism, mechanism_futures=8,
        mechanism_root_rule='one uniformly reservoir-sampled eligible predecessor state per task',
        mechanism_plan_rule='union of factual and the three highest-scored plans of each frozen estimator',
        mechanism_continuation='one finite intervention then the same frozen predecessor; shared futures',
        vrx_tasks=vrx, vrx_methods=(*MODELS, 'arboids', 'mappo'),
        vrx_adapter_sha256=e.digest(adapter), vrx_adapter_source=adapter.read_text(encoding='utf8'),
        baseline_seed_selection='seed 42 fixed before scope or VRX outcomes; no endpoint-based selection',
        learned_opponent=dict(path=str(opponent), sha256=e.digest(opponent),
            training_steps=attack['step'], training_specification=attack['specification'],
            role='one common frozen learned adversary, trained on another development defender',
            adaptation_performed=False), opponent_helpers=helpers,
        opponent_helpers_sha256=hashlib.sha256(helpers.encode()).hexdigest(),
        frozen_model_role='prospective scope and mechanism evaluation of the indexed development model',
        cross_fleet_policy='record unsupported fixed-size M2/MAPPO execution explicitly; do not pad or disable M2',
        diagnostic_source=Path(__file__).read_text(encoding='utf8'))
    e.save(analysis, ANALYSIS)
    print(dict(phase='r20_diagnostic_design_frozen', sha256=e.digest(ANALYSIS),
               scope_tasks=sum(map(len, groups.values())), mechanism_tasks=len(mechanism),
               vrx_paired_scenes=len(vrx), vrx_methods=analysis['vrx_methods']), flush=True)
    return analysis


def initialize_worker(opponent=False):
    global _ANALYSIS
    _ANALYSIS = e.read(ANALYSIS)
    e.runtime(e.read(e.MANIFEST))
    e.baseline_runtime()
    if opponent:
        r = sys.modules['run_remote_evidence']
        r._SOURCE_ATTACKER = r._SOURCE_ATTACKER_SHA256 = None
        exec(compile(_ANALYSIS['opponent_helpers'], '<frozen-common-opponent>', 'exec'), r.__dict__)
        record = _ANALYSIS['learned_opponent']
        if e.digest(record['path']) != record['sha256']:
            raise ValueError('The common learned opponent changed.')
        r.configure_source_attacker(record['path'])


def scope_episode(job):
    label, scene, future = job
    model = _ANALYSIS['models'][label]
    path = Path(model['path'])
    if model['kind'] == 'joint':
        return sys.modules['develop_source_fusion'].joint_episode((scene, str(path), path.stat().st_mtime_ns,
                                                                 False, 0, 0, 0, 0, future))
    location = model['directory'] if label == 'arboids' else str(path)
    return e.baseline_episode((label, location, model['sha256'], scene, future))


def scope(analysis, workers):
    for group, tasks in analysis['scope_tasks'].items():
        for label in (*MODELS, 'arboids', 'mappo'):
            model = analysis['models'][label]
            path = e.OUTPUT/'scope'/group/(label+'.pth')
            old = e.read(path) if path.exists() else None
            if old and (old['tasks'] != tasks or old['policy_sha256'] != model['sha256']
                        or old['analysis_sha256'] != e.digest(ANALYSIS)):
                raise ValueError('A resumed scope evaluation changed its design or model.')
            if old and old['completed']:
                continue
            if e.digest(model['path']) != model['sha256']:
                raise ValueError('A frozen scope model changed.')
            if tasks[0][0][1] != 6 and label in ('F01', 'F11', 'mappo'):
                e.save(dict(completed=True, evaluated=0, supported=False, rows=[], tasks=tasks,
                    policy_sha256=model['sha256'], analysis_sha256=e.digest(ANALYSIS),
                    reason='The frozen deployed value/candidate representation requires six defenders.',
                    evidence_role='measured implementation boundary, not a failed task outcome'), path)
                continue
            e.resource_wait()
            rows = old['rows'] if old else [None]*len(tasks)
            jobs = [(i, (label, scene, future)) for i, (scene, future) in enumerate(tasks) if rows[i] is None]
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                    initargs=(group == 'common-learned-opponent',)) as pool:
                pending = {pool.submit(scope_episode, job): index for index, job in jobs}
                for result in as_completed(pending):
                    rows[pending[result]] = result.result()
                    done = [row for row in rows if row is not None]
                    if len(done) % 16 == 0 or len(done) == len(tasks):
                        record = dict(completed=len(done) == len(tasks), supported=True, tasks=tasks, rows=rows,
                            evaluated=len(done), success=sum(row['success'] for row in done),
                            policy_sha256=model['sha256'], analysis_sha256=e.digest(ANALYSIS),
                            common_opponent=analysis['learned_opponent'] if group == 'common-learned-opponent' else 'APF',
                            evidence_role='prospective scope of the frozen indexed r20 model')
                        e.save(record, path)
                        print(dict(phase='r20_scope', group=group, method=label,
                                   evaluated=len(done), total=len(tasks), success=record['success']), flush=True)


def reservoir_root(policy, scene, future, index):
    d = sys.modules['develop_source_fusion']
    actor = d.joint_load(str(policy), Path(policy).stat().st_mtime_ns)
    np.random.seed(scene[0]); torch.manual_seed(scene[0])
    env = d.numerical_source().TADEnv(scene[1])
    observations, _ = env.reset(agility=scene[2], noisy_agility=False)
    if future is not None:
        np.random.seed(future)
    generator = np.random.default_rng(e.seed32(96, index))
    done = elapsed = eligible = 0
    chosen = None
    while not done:
        packet = d.mechanism_tensor(d.joint_packet(env, observations))
        if elapsed % 5 == 0:
            eligible += 1
            if int(generator.integers(eligible)) == 0:
                chosen = dict(environment=copy.deepcopy(env), observation=observations.copy(),
                    random_state=np.random.get_state(), elapsed=elapsed)
        with torch.no_grad():
            action = actor(packet, deterministic=True)[0][0].numpy()
        observations, _, done, _, steps = d.joint_advance(env, observations, actor, action)
        elapsed += steps
    if chosen is None:
        raise ValueError('The predecessor trajectory has no eligible state.')
    chosen.update(eligible_states=eligible, factual_task_outcome=d.outcome(env, done))
    return chosen


def estimator_predictions(label, packet):
    d = sys.modules['develop_source_fusion']
    path = Path(_ANALYSIS['models'][label]['path'])
    actor = d.joint_load(str(path), path.stat().st_mtime_ns)
    value = actor.successor_task_value
    record = {}

    def capture(frame, event, result):
        if (event == 'return' and frame.f_code is inspect.unwrap(value.forward).__code__
                and frame.f_locals.get('self') is value and 'score' in frame.f_locals):
            local = frame.f_locals
            record['scores'] = local['score'][0].detach().cpu().numpy().copy()
            record['estimates'] = local['estimates'][:, 0].detach().cpu().numpy().copy()
            record['choice'] = int(local['choice'][0])
            record['accepted'] = bool(local['accept'][0])
            record['options'] = value.options.detach().cpu().numpy().copy()

    previous = sys.getprofile()
    sys.setprofile(capture)
    try:
        with torch.no_grad():
            action = actor(d.mechanism_tensor(packet), deterministic=True)[0][0].numpy()
    finally:
        sys.setprofile(previous)
    if not record:
        raise ValueError('The sampled state did not expose the frozen estimator decision.')
    record['deployed_action'] = action.copy()
    return record


def mechanism_one_cost(actor, packet, control):
    c = sys.modules['conditional_fusion_value']
    function = c.refine_actuator_interactions
    values = []
    previous = sys.getprofile()

    def capture(frame, event, result):
        if event == 'return' and frame.f_code is inspect.unwrap(function).__code__ and 'choices' in frame.f_locals:
            local = frame.f_locals
            choice = local['choices'][0]
            unary, risk = local['unary'][0], local['risk'][0]
            n = len(choice)
            unary_cost = sum(float(unary[i, choice[i]]) for i in range(n))
            pair_cost = sum(float(risk[i, j, choice[i], choice[j]]) for i in range(n) for j in range(i+1, n))
            values.append(dict(selection=local['selection'], unary_cost=unary_cost,
                pair_risk=pair_cost, total_cost=unary_cost+local['risk_weight']*pair_cost,
                choices=choice.detach().cpu().numpy().copy()))

    sys.setprofile(capture)
    try:
        with torch.no_grad():
            for selection in ('joint', 'independent'):
                options = dict(actor.actuator_fusion_refinement, selection=selection)
                function(actor, packet, control, control, **options)
    finally:
        sys.setprofile(previous)
    if len(values) != 2:
        raise ValueError('The diagnostic did not observe both matched M1 selectors.')
    return values


def mechanism_episode(job):
    arm, index, scene, future = job
    d = sys.modules['develop_source_fusion']
    c = sys.modules['conditional_fusion_value']
    l = sys.modules['local_intervention_value']
    predecessor = 'F00' if arm == 'F01' else 'F10'
    ordinary = 'ordinary_no_M1' if arm == 'F01' else 'ordinary_with_M1'
    policy = Path(_ANALYSIS['models'][predecessor]['path'])
    root = reservoir_root(policy, scene, future, index)
    packet = d.joint_packet(root['environment'], root['observation'])
    predictions = {label: estimator_predictions(label, packet) for label in (arm, ordinary)}
    if not np.array_equal(predictions[arm]['options'], predictions[ordinary]['options']):
        raise ValueError('The matched estimators do not share an intervention catalog.')
    chosen = {0}
    for value in predictions.values():
        chosen.update(map(int, np.argsort(-value['scores'], kind='stable')[:3]))
        chosen.add(value['choice'] if value['accepted'] else 0)
    chosen = sorted(chosen)
    actor = d.joint_load(str(policy), policy.stat().st_mtime_ns)
    tensor = d.mechanism_tensor(packet)
    with torch.no_grad():
        control = actor(tensor, deterministic=True, gate_only=True)[0][0].numpy()[:, :3]
        context, order = c.intervention_task_context(tensor, True, True)
    inverse = order[0].argsort().numpy()
    coefficients = np.asarray([[-1., -1.]]+[[a, b] for a in (0., .5, 1.) for b in (0., .5, 1.)])
    one = np.ones((scene[1], 1), np.float32)
    jobs = []
    keys = []
    futures = [e.seed32(97, 0 if arm == 'F01' else 1, index, i)
               for i in range(_ANALYSIS['mechanism_futures'])]
    for q in chosen:
        code, delay, span = map(int, predictions[arm]['options'][q])
        if code == 0:
            action = control.copy()
        else:
            preference = coefficients[(code//(10**np.arange(scene[1]))) % 10][inverse].astype(np.float32)
            first = preference if delay == 0 else np.full_like(preference, -1.)
            action = np.concatenate((control, -one, (delay+span)*one, -2*one,
                np.zeros((scene[1], 2), np.float32), delay*one, first, preference, np.zeros_like(one)), -1)
        for j, branch_future in enumerate(futures):
            keys.append((q, j))
            jobs.append((str(policy), dict(environment=root['environment'], observation=root['observation'],
                random_state=root['random_state'], action=action), scene, branch_future, max(1, delay+span)))
    measured = l.source_successor_batches(jobs, batch_size=32, device='cpu')
    outcomes = {key: {k: row[k] for k in ('success', 'outcome', 'duration', 'physical_steps')}
                for key, row in zip(keys, measured)}
    selected_predictions = {label: dict(choice=v['choice'], accepted=v['accepted'],
        plans={q: dict(option=tuple(map(int, v['options'][q])), score=float(v['scores'][q]),
                       members=v['estimates'][:, q].tolist()) for q in chosen})
        for label, v in predictions.items()}
    m1 = None
    if arm == 'F11':
        m1 = mechanism_one_cost(actor, tensor, torch.as_tensor(control[None]))
    return dict(arm=arm, index=index, scene=scene, future=future, completed=True,
        root_elapsed=root['elapsed'], eligible_states=root['eligible_states'],
        factual_task_outcome=root['factual_task_outcome'], context=context[0].numpy(),
        predictions=selected_predictions, outcomes=outcomes, futures=futures,
        plans=chosen, m1_candidate_objective=m1,
        simulation_steps=sum(row['physical_steps'] for row in measured),
        analysis_sha256=e.digest(ANALYSIS), diagnostic_source_sha256=e.digest(__file__),
        diagnostic_source=Path(__file__).read_text(encoding='utf8'),
        evidence_role='independent states and common-future finite intervention measurements; no fitting')


def mechanism(analysis, workers):
    tasks = analysis['mechanism_tasks']
    for arm in ('F01', 'F11'):
        jobs = []
        for i, (scene, future) in enumerate(tasks):
            path = e.OUTPUT/'mechanism'/arm/f'root-{i:03d}.pth'
            if path.exists():
                old = e.read(path)
                if old['analysis_sha256'] != e.digest(ANALYSIS) or old['scene'] != scene:
                    raise ValueError('A mechanism root changed identity.')
                if old['completed']:
                    continue
            jobs.append((i, path, (arm, i, scene, future)))
        e.resource_wait()
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker) as pool:
            pending = {pool.submit(mechanism_episode, job): (index, path) for index, path, job in jobs}
            for result in as_completed(pending):
                index, path = pending[result]
                row = result.result()
                e.save(row, path)
                print(dict(phase='r20_mechanism', arm=arm, root=index, plans=len(row['plans']),
                           futures=len(row['futures']), steps=row['simulation_steps']), flush=True)


def vrx_shard(analysis, shard, shards):
    adapter = e.ROOT/'vrx/run_r20_evidence.py'
    if e.digest(adapter) != analysis['vrx_adapter_sha256']:
        raise ValueError('The frozen VRX adapter changed.')
    for index, task in enumerate(analysis['vrx_tasks']):
        if index % shards != shard:
            continue
        order = list(analysis['vrx_methods'])
        np.random.default_rng(e.seed32(98, index)).shuffle(order)
        for label in order:
            model = analysis['models'][label]
            if e.digest(model['path']) != model['sha256']:
                raise ValueError('A frozen VRX model changed.')
            existing = []
            for attempt in (1, 2):
                run_id = f'case{index:03d}-{label}-attempt{attempt}'
                output = e.OUTPUT/'vrx/paired'/run_id
                result_path = output/'result.json'
                if result_path.exists():
                    record = json.loads(result_path.read_text())
                    if (record.get('policy_sha256') != model['sha256']
                            or record['seed'] != task['seed']
                            or record.get('deployment_adapter_sha256') != analysis['vrx_adapter_sha256']):
                        raise ValueError('A resumed VRX trial changed model or transport identity.')
                    existing.append(record)
                    if record['passed']:
                        break
                    continue
                if output.exists():
                    raise ValueError('An interrupted VRX attempt needs inspection before resuming: '+str(output))
                e.resource_wait()
                args = ['python', '-X', 'utf8', '-u', str(adapter), '--variant', label,
                    '--num-robots', str(task['defenders']+1), '--agility', str(task['agility']),
                    '--setting', str(task['setting']), '--seed', str(task['seed']),
                    '--duration', str(task['duration']), '--action-period', str(task['action_period']),
                    '--headless', '--startup-timeout', '180', '--wall-timeout', '600',
                    '--assets-dir', str(e.ROOT/'.vrx-assets'), '--output-dir', str(e.OUTPUT/'vrx/paired'),
                    '--run-id', run_id]
                if index == 0 or (label == 'F11' and index % 20 == 0):
                    args.append('--capture-frames')
                shell = ('source vrx_ws/activate.bash >/dev/null && export ROS_DOMAIN_ID='
                         +str(100+shard)+' ARBOIDS_RENDER_THREADS=2 && exec '+shlex.join(args))
                process = subprocess.run(['xvfb-run', '-a', 'bash', '-lc', shell], cwd=e.ROOT)
                if not result_path.exists():
                    raise RuntimeError('VRX failed before producing a native trial result: '+run_id)
                record = json.loads(result_path.read_text())
                record.update(analysis_sha256=e.digest(ANALYSIS), paired_scene_index=index,
                              attempt=attempt, command_returncode=process.returncode)
                result_path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
                existing.append(record)
                print(dict(phase='r20_vrx', case=index, method=label, attempt=attempt,
                    transport_passed=record['passed'], success=record.get('success'),
                    outcome=record.get('outcome'), deadlines_missed=record.get('control_deadline_misses')),
                    flush=True)
                if record['passed']:
                    break
            if not existing or not existing[-1]['passed']:
                print(dict(phase='r20_vrx_unresolved_transport', case=index, method=label), flush=True)


def vrx(analysis, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(vrx_shard, analysis, shard, workers) for shard in range(workers)]
        for future in as_completed(futures):
            future.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('initialize', 'scope', 'mechanism', 'vrx'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    analysis = initialize()
    if args.operation != 'initialize':
        {'scope': scope, 'mechanism': mechanism, 'vrx': vrx}[args.operation](analysis, args.workers)


if __name__ == '__main__':
    main()
