"""Validation-only stage evidence; never reads the independent final test sets."""
import argparse
import hashlib
import json
from pathlib import Path
import time

SEEDS = (42, 101, 202, 303, 404)
ARMS = ('full', 'same_info', 'arboids_cbf', 'short', 'model_value')
CORE_ARMS = ('full', 'same_info', 'model_value')
METRICS = ('success', 'capture', 'collision', 'breach', 'timeout', 'capture_time')
PROPOSALS = ('f1', 'f2', 't1', 'l1', 'l2', 'mean_layer', 'log_std_layer')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validation_calibration(payload, source, seed, states=32, repetitions=2):
    import numpy as np
    import torch
    from envs.TADgame import TADEnv
    from envs.snapshot import seed_random, preserved_random_state
    from interaction_rollout import public_packet, execute, compact_snapshot, FrozenPolicy, branch_return
    from policy.interaction_sac import tensor_packet
    rows = []
    with preserved_random_state():
        policy = FrozenPolicy(payload)
        for index in range(states):
            # Separate from training, curve validation (310M), and final calibration (370M).
            scene = 330000000 + SEEDS.index(seed) * 1000000 + index
            seed_random(scene)
            env = TADEnv(3, protocol='paper-parameters-v1')
            obs, _ = env.reset(2.25)
            packet = public_packet(env, obs)
            for _ in range(10 + index % 40):
                snapshot = compact_snapshot(env)
                action, _ = source.choose_action(packet)
                packet, _, done, _, _, _ = execute(env, packet, action)
                if done:
                    break
            packet = public_packet(snapshot.environment)
            action, _ = policy.action(packet, torch.Generator().manual_seed(scene + 100))
            alternative = action.copy()
            boat, reference = index % 3, (0., .5, 1.)[(index // 3) % 3]
            alternative[boat, 2] = reference
            p = {k: v.unsqueeze(0) for k, v in tensor_packet(packet).items()}
            with torch.no_grad():
                a = policy.critic(p, torch.as_tensor(action).unsqueeze(0))
                b = policy.critic(p, torch.as_tensor(alternative).unsqueeze(0))
                predicted = float((torch.minimum(*a) - torch.minimum(*b)).item())
            differences, used = [], 0
            for repeat in range(repetitions):
                future, noise = scene + repeat * 10000, scene + repeat * 10000 + 1
                ga, na, _ = branch_return(policy, snapshot, action, future, noise, 300)
                gb, nb, _ = branch_return(policy, snapshot, alternative, future, noise, 300)
                differences.append(ga - gb)
                used += na + nb
            actual = float(np.mean(differences))
            rows.append(dict(state=index, scene_seed=scene, boat=boat, reference=reference,
                predicted_difference=predicted, environment_difference=actual,
                absolute_error=abs(predicted - actual), zero_prediction_error=abs(actual),
                monte_carlo_standard_error=float(np.std(differences, ddof=1) / np.sqrt(repetitions)),
                simulated_steps=used))
    return rows


def check_endpoint(saved, source_weights, stage, expected_step):
    import torch
    if saved['stage'] != stage or saved['step'] != expected_step:
        raise ValueError('The review requires the fixed stage endpoint, not an intermediate/best checkpoint.')
    if saved['config']['environment']['protocol'] != 'paper-parameters-v1':
        raise ValueError('Review protocol mismatch.')
    for component in ('actor', 'critic'):
        for name, tensor in saved.get(component, {}).items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f'Non-finite endpoint {component}.{name}')
    state = {k.removeprefix('base.'): v for k, v in saved['actor'].items()
             if k.removeprefix('base.').split('.')[0] in PROPOSALS}
    expected = {k: v for k, v in source_weights.items() if k.split('.')[0] in PROPOSALS}
    if state.keys() != expected.keys():
        raise ValueError('Source proposal architecture changed.')
    frozen_equal = all(torch.equal(state[k], expected[k]) for k in state)
    if stage == 'gate' and not frozen_equal:
        raise ValueError('The frozen proposal branch changed during gate training.')
    return dict(finite=True, frozen_proposals_equal=frozen_equal,
                actor_parameters=sum(v.numel() for v in saved['actor'].values()))


def collect(study, stage, seed, scope='full'):
    import study_runtime
    import numpy as np
    import torch
    import yaml
    from interaction_evaluation import OriginalPolicy
    from interaction_rollout import DeploymentPolicy
    from train_interaction import episode, arm_config
    torch.set_num_threads(1)
    study = Path(study)
    output = study / 'reviews' / f'data-{stage}-{seed}'
    arms = CORE_ARMS if scope == 'core' else ARMS
    completion = 'core-completed.json' if scope == 'core' else 'completed.json'
    manifest = json.loads((study / 'manifest.json').read_text())
    inherited = manifest['inherited_pretraining'].get(f'pretrain-{seed}')
    source_path = Path(inherited['path']) if inherited else study / f'pretrain/seed-{seed}/actor.pth'
    paths = {a: study / f'training/seed-{seed}/{a}/{stage}-endpoint.pth' for a in arms}
    inputs = dict(stage=stage, seed=seed, endpoints={a: digest(p) for a, p in paths.items()},
                  source=digest(source_path), episodes=100, states=32, repetitions=2)
    input_path = output / ('inputs-core.json' if scope == 'core' else 'inputs.json')
    if input_path.exists() and json.loads(input_path.read_text()) != inputs:
        raise ValueError('Review inputs changed; preserve the existing version and create a new study revision.')
    write_json(input_path, inputs)
    source_weights = torch.load(source_path, weights_only=True, map_location='cpu')
    source = OriginalPolicy(source_path)
    base_config = yaml.safe_load((Path(__file__).parent / 'configs/interaction-aware-sac.yaml').read_text())
    expected_step = 250000 if stage == 'gate' else 1250000
    for arm in (*arms, 'initial_cbf'):
        result_path = output / f'{arm}.json'
        arm_input = dict(stage=stage, seed=seed, source=inputs['source'],
                         endpoint=inputs['endpoints'].get(arm), episodes=100, states=32, repetitions=2)
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if existing.get('input') != arm_input:
                raise ValueError('Cached validation inputs changed; do not reuse a different endpoint.')
            continue
        started = time.perf_counter()
        if arm == 'initial_cbf':
            policy, checks, calibration = source, {}, []
        else:
            saved = torch.load(paths[arm], weights_only=True, map_location='cpu')
            checks = check_endpoint(saved, source_weights, stage, expected_step)
            if stage == 'joint':
                gate = torch.load(paths[arm].with_name('gate-endpoint.pth'), weights_only=True, map_location='cpu')
                checks['gate_endpoint'] = check_endpoint(gate, source_weights, 'gate', 250000)
            expected = arm_config(base_config, arm)
            expected['training'].update(seed=seed, pretrain_checkpoint=str(source_path))
            actual = saved['config']
            for key in ('resume', 'pretrain_checkpoint'):
                expected['training'].pop(key, None)
                actual['training'].pop(key, None)
            if actual != expected:
                raise ValueError('A review endpoint differs from the fixed scientific configuration.')
            policy = DeploymentPolicy(paths[arm])
            calibration = validation_calibration(saved, source, seed) if arm != 'arboids_cbf' else []
        rows, steps, gates = [], 0, []
        for index in range(100):
            scene = 310000000 + seed * 10000 + index
            row, history = episode(policy, scene, safety=True, trajectory=True)
            steps += len(history)
            gates.extend(float(g) for item in history for g in item['gates'])
            rows.append(dict(scene_seed=scene, **row))
        summary = {key: float(np.mean([r[key] for r in rows])) for key in METRICS}
        summary.update(gate_quantiles=np.quantile(gates, [0., .1, .5, .9, 1.]).tolist(),
                       saturated_gate_fraction=float(np.mean((np.asarray(gates) < .01) | (np.asarray(gates) > .99))))
        if calibration:
            summary.update(E_delta=float(np.mean([r['absolute_error'] for r in calibration])),
                           zero_prediction_error=float(np.mean([r['zero_prediction_error'] for r in calibration])))
        write_json(result_path, dict(arm=arm, stage=stage, seed=seed, input=arm_input, checks=checks, summary=summary,
            episodes=rows, calibration=calibration, validation_steps=steps,
            calibration_steps=sum(r['simulated_steps'] for r in calibration),
            wall_seconds=time.perf_counter()-started))
    results = {a: json.loads((output / f'{a}.json').read_text()) for a in (*arms, 'initial_cbf')}
    matched = [results[a]['checks']['actor_parameters'] for a in arms if a != 'arboids_cbf']
    if len(set(matched)) != 1:
        raise ValueError('The same-information actor capacities are not matched.')
    write_json(output / completion, dict(complete=True, scope=scope, inputs=inputs,
        artifacts={a: digest(output / f'{a}.json') for a in results},
        validation_steps=sum(r['validation_steps'] for r in results.values()),
        calibration_steps=sum(r['calibration_steps'] for r in results.values()),
        wall_seconds=sum(r['wall_seconds'] for r in results.values())))


def review_evidence(study, stage, seeds, scope='full'):
    """Small reports only: the scheduler does not load models or use a GPU."""
    import numpy as np
    study = Path(study)
    arms = CORE_ARMS if scope == 'core' else ARMS
    completion = 'core-completed.json' if scope == 'core' else 'completed.json'
    rng = np.random.default_rng(197704)
    raw, inputs, input_paths, artifact_inputs = {}, {}, {}, {}
    for seed in seeds:
        directory = study / 'reviews' / f'data-{stage}-{seed}'
        completed = json.loads((directory / completion).read_text())
        if not completed.get('complete'):
            raise ValueError('Incomplete validation evidence.')
        if set(completed['artifacts']) != set((*arms, 'initial_cbf')):
            raise ValueError('The prescribed comparison matrix is incomplete.')
        inputs[str(seed)] = digest(directory / completion)
        input_paths[str(seed)] = str((directory / completion).relative_to(study))
        raw[seed] = {}
        for arm, expected in completed['artifacts'].items():
            path = directory / f'{arm}.json'
            if digest(path) != expected:
                raise ValueError('Validation artifact changed after completion.')
            artifact_inputs[str(path.relative_to(study))] = expected
            raw[seed][arm] = json.loads(path.read_text())
            rows = raw[seed][arm]['episodes']
            if len(rows) != 100 or [r['scene_seed'] for r in rows] != [310000000 + seed*10000+i for i in range(100)]:
                raise ValueError('Validation scenes are incomplete or unpaired.')
            if arm in ('full','same_info','short','model_value'):
                calibration=raw[seed][arm]['calibration']
                if len(calibration)!=32 or [r['state'] for r in calibration]!=list(range(32)):
                    raise ValueError('Validation intervention states are incomplete or unpaired.')
    def interval(values):
        values = np.asarray(values, dtype=float)
        groups, count = values.shape
        picked = rng.integers(groups, size=(10000, groups))
        scenes = rng.integers(count, size=(10000, groups, count))
        means = values[picked[..., None], scenes].mean(axis=(1, 2))
        return dict(difference=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist())
    comparisons = []
    for reference in (*[a for a in arms if a != 'full'], 'initial_cbf'):
        for metric in METRICS:
            differences = [[a[metric] - b[metric] for a, b in zip(raw[s]['full']['episodes'], raw[s][reference]['episodes'])]
                           for s in seeds]
            comparisons.append(dict(reference=reference, metric=metric, **interval(differences)))
        if reference in ('same_info', 'model_value', 'short'):
            differences = [[a['absolute_error'] - b['absolute_error'] for a, b in
                            zip(raw[s]['full']['calibration'], raw[s][reference]['calibration'])] for s in seeds]
            comparisons.append(dict(reference=reference, metric='E_delta', **interval(differences)))
    seed_contrasts = {str(s): {a: {m: float(np.mean([x[m]-y[m] for x,y in
        zip(raw[s]['full']['episodes'], raw[s][a]['episodes'])])) for m in METRICS}
        for a in arms if a != 'full'} for s in seeds}
    for seed in seeds:
        for reference in (a for a in arms if a not in ('full', 'arboids_cbf')):
            seed_contrasts[str(seed)][reference]['E_delta'] = float(np.mean([
                a['absolute_error']-b['absolute_error'] for a,b in
                zip(raw[seed]['full']['calibration'], raw[seed][reference]['calibration'])]))
    return dict(stage=stage, seeds=list(seeds), inputs=inputs, input_paths=input_paths, artifact_inputs=artifact_inputs,
        technical_checks_passed=True, seed_contrasts=seed_contrasts,
        scope='development validation only; intervals are descriptive, not final confirmatory tests',
        summaries={str(s): {a: r['summary'] for a, r in raw[s].items()} for s in seeds},
        paired_comparisons=comparisons,
        interpretation=('One training seed: no conclusion about across-seed robustness.' if len(seeds)==1 else
                        f'{len(seeds)} training seeds; inspect seed-level consistency and the entire matched-control matrix.'),
        review_required=['Check finite training and frozen proposals; retain all valid task failures.',
            'Assess E_delta together with capture and capped time against all matched controls.',
            'Saturated gates or an early loss of reward trigger diagnosis, not automatic retuning.',
            'A weak gate-only result alone does not rule out joint learning; record the rationale.',
            'Do not change the frozen method or claim superiority from this validation evidence.'])


def deployment_evidence(study):
    """Read only the prescribed 60-episode deployment validation bank."""
    import numpy as np
    study = Path(study)
    manifest = json.loads((study/'manifest.json').read_text())
    jobs = [j for j in manifest['jobs'] if j.get('phase') == 'deployment-validation']
    expected = {(s,a,n,c,i) for s in SEEDS[:2] for a in ('full','same_info','arboids_cbf')
                for n,c in ((3,0),(6,1)) for i in range(5)}
    actual = {(j['validation']['seed'],j['validation']['arm'],j['validation']['defenders'],
               j['validation']['setting'],j['validation']['trial']) for j in jobs}
    if len(jobs) != 60 or actual != expected:
        raise ValueError('The fixed deployment validation matrix is incomplete.')
    rows, inputs, input_paths = [], {}, {}
    for job in jobs:
        spec = job['validation']
        path = Path(job['result'])
        result = json.loads(path.read_text())
        if (not result.get('passed') or result.get('outcome_code') not in (1,2,3,4) or
                result.get('seed') != spec['scene_seed'] or result.get('setting') != spec['setting'] or
                result.get('num_robots') != spec['defenders']+1 or result.get('controller') != 'IACRRL'):
            raise ValueError('Deployment validation has an invalid or mismatched episode.')
        inputs[job['name']] = digest(path)
        input_paths[job['name']] = str(path.relative_to(study))
        code, duration = result['outcome_code'], float(result['simulation_seconds'])
        if not np.isfinite(duration): raise ValueError('Non-finite simulation duration.')
        row = dict(spec, success=int(code in (3,4)), capture=int(code==3),
            collision=int(result['defender_collision']),
            breach=int(np.linalg.norm(result['terminal_positions'][0]) <= result['target_radius']),
            timeout=int(code==4), capture_time=min(60.,duration) if code==3 else 60.)
        rows.append(row)
    lookup = {(r['seed'],r['arm'],r['defenders'],r['setting'],r['trial']):r for r in rows}
    contrasts = []
    for seed in SEEDS[:2]:
        for n,c in ((3,0),(6,1)):
            for arm in ('same_info','arboids_cbf'):
                contrasts.append(dict(seed=seed, reference=arm, defenders=n, setting=c,
                    **{m: float(np.mean([lookup[seed,'full',n,c,i][m]-lookup[seed,arm,n,c,i][m]
                        for i in range(5)])) for m in METRICS}))
    return dict(stage='deployment', seeds=list(SEEDS[:2]), inputs=inputs, input_paths=input_paths,
        technical_checks_passed=True, episodes=rows, seed_contrasts=contrasts,
        scope='Independent deployment validation only; five paired scenes per seed and condition, not final VRX evidence',
        interpretation='Inspect task failures and consistency across both seeds; this small bank screens transfer before replication.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--stage', choices=['gate', 'joint'], required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--scope', choices=['core', 'full'], default='full')
    args = parser.parse_args()
    collect(args.study, args.stage, args.seed, args.scope)


if __name__ == '__main__':
    main()
