"""Validation-only stage evidence; never reads the independent final test sets."""
import argparse
import hashlib
import json
from pathlib import Path
import time

SEEDS = (42, 101, 202, 303, 404)
ARMS = ('full', 'same_info', 'arboids_cbf', 'short', 'model_value')
CORE_ARMS = ('full', 'same_info', 'model_value')
PEER_ARMS = ('full', 'no_peer')
SCOPE_ARMS = dict(core=CORE_ARMS, full=ARMS, peer=PEER_ARMS)
CANDIDATE_RESPONSE_PROTOCOL = 'fixed-state-peer-resampling-v1'
METRICS = ('success', 'capture', 'collision', 'breach', 'timeout', 'capture_time')
PROPOSALS = ('f1', 'f2', 't1', 'l1', 'l2', 'mean_layer', 'log_std_layer')
DEVELOPMENT_SEEDS = (42, 101)
DEVELOPMENT_ARMS = ('full', 'same_info', 'no_peer', 'short', 'model_value')
SIGNAL_PROTOCOL = 'shared-state-signal-v1'


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


def controlled_candidate_response(actor, packet, boids, noise, peer_proposal, boat, peer):
    """Change one communicated learned candidate; hold state, own inputs and gate noise fixed."""
    import numpy as np
    import torch
    from interaction_rollout import safety_controller
    from policy.interaction_sac import tensor_packet
    if boat == peer:
        raise ValueError('The controlled candidate must belong to another boat.')
    p = {k: v.unsqueeze(0) for k, v in tensor_packet(packet).items()}
    with torch.no_grad():
        action, _ = actor(p['obs'], p['motion'], noise=noise)
        proposals = action[..., :2].clone()
        proposals[0, peer] = torch.as_tensor(peer_proposal, dtype=proposals.dtype)
        mask = torch.ones(action.shape[:2], dtype=torch.bool)
        gates, _ = actor.gates(actor.encode(p['obs'], mask), p['motion'], proposals,
                              p['obs'][..., 12:14], mask, noise=noise[..., 2:3])
    baseline = action[0].numpy()
    gate_only = baseline.copy()
    gate_only[boat, 2] = float(gates[0, boat, 0])
    peer_only = baseline.copy()
    peer_only[peer, :2] = proposals[0, peer].numpy()
    both = peer_only.copy()
    both[boat, 2] = gate_only[boat, 2]
    controller = safety_controller(len(baseline))
    def thrust(a):
        return controller.control(np.asarray(packet['motion'], dtype=float), a, boids)[1][boat]
    base_u, gate_u, peer_u, both_u = map(thrust, (baseline, gate_only, peer_only, both))
    nominal_delta = (gate_only[boat, 2] - baseline[boat, 2]) * (750.*baseline[boat, :2] + 250. - boids[boat])
    row = dict(boat=boat, peer=peer, gate_noise=float(noise[0, boat, 2]),
        peer_candidate_before=baseline[peer, :2].tolist(), peer_candidate_after=peer_only[peer, :2].tolist(),
        own_candidate=baseline[boat, :2].tolist(), gate_before=float(baseline[boat, 2]),
        gate_after=float(gate_only[boat, 2]), gate_change=float(gate_only[boat, 2]-baseline[boat, 2]),
        peer_candidate_change=float(np.linalg.norm(peer_only[peer, :2]-baseline[peer, :2])),
        nominal_thrust_change_N=float(np.linalg.norm(nominal_delta)),
        gate_only_executed_thrust_change_N=float(np.linalg.norm(gate_u-base_u)),
        direct_peer_cbf_thrust_change_N=float(np.linalg.norm(peer_u-base_u)),
        gate_effect_given_changed_peer_N=float(np.linalg.norm(both_u-peer_u)),
        total_executed_thrust_change_N=float(np.linalg.norm(both_u-base_u)))
    if not all(np.isfinite(v).all() for v in row.values()):
        raise FloatingPointError('Non-finite controlled candidate response.')
    return row


def validation_candidate_response(payload, source, seed, states=32, resamples=4, scene_base=None):
    import numpy as np
    import torch
    from envs.TADgame import TADEnv
    from envs.snapshot import seed_random, preserved_random_state
    from interaction_rollout import public_packet, execute, compact_snapshot, FrozenPolicy
    from policy.interaction_sac import tensor_packet
    rows, used = [], 0
    with preserved_random_state(), torch.no_grad():
        actor = FrozenPolicy(payload).actor
        for index in range(states):
            # Independent of curve validation, return calibration and final tests.
            scene = (350000000 + SEEDS.index(seed)*1000000 if scene_base is None else scene_base) + index
            seed_random(scene)
            env = TADEnv(3, protocol='paper-parameters-v1')
            obs, _ = env.reset(2.25)
            packet = public_packet(env, obs)
            for _ in range(10 + index % 40):
                snapshot = compact_snapshot(env)
                action, _ = source.choose_action(packet, deterministic=True)
                packet, _, done, _, _, _ = execute(env, packet, action)
                used += 1
                if done:
                    break
            env = snapshot.environment
            packet = public_packet(env)
            p = {k: v.unsqueeze(0) for k, v in tensor_packet(packet).items()}
            generator = torch.Generator().manual_seed(scene + 100)
            noise = torch.randn((1, 3, 3), generator=generator)
            perturbations = []
            for repeat in range(resamples):
                boat, peer = index % 3, (index % 3 + 1 + repeat % 2) % 3
                # Draw from this checkpoint's proposal distribution at the unchanged state.
                sampled, _ = actor(p['obs'], p['motion'], noise=torch.randn((1, 3, 3), generator=generator))
                perturbations.append(controlled_candidate_response(actor, packet, env.boids_actions,
                    noise, sampled[0, peer, :2], boat, peer))
            rows.append(dict(state=index, scene_seed=scene, perturbations=perturbations))
    flat = [r for state in rows for r in state['perturbations']]
    metrics = ('peer_candidate_change', 'nominal_thrust_change_N', 'gate_only_executed_thrust_change_N',
               'direct_peer_cbf_thrust_change_N', 'gate_effect_given_changed_peer_N', 'total_executed_thrust_change_N')
    summary = {f'mean_{m}': float(np.mean([r[m] for r in flat])) for m in metrics}
    changes = np.abs([r['gate_change'] for r in flat])
    summary.update(mean_abs_gate_change=float(np.mean(changes)), gate_change_q90=float(np.quantile(changes, .9)))
    return dict(protocol=CANDIDATE_RESPONSE_PROTOCOL, states=rows, resamples=resamples, summary=summary,
        simulated_steps=used, cbf_evaluations=4*len(flat),
        interpretation='Response diagnoses candidate use, not task benefit. Gate-only keeps physical peer commands fixed; '
                       'direct-peer CBF response keeps all gates fixed; the combined response includes both paths.')


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


def collect(study, stage, seed, scope='full', only_arm=None):
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
    arms = SCOPE_ARMS[scope]
    if only_arm is not None:
        if only_arm not in (*arms, 'initial_cbf'):
            raise ValueError('The single-arm diagnostic is outside this comparison scope.')
        arms = () if only_arm == 'initial_cbf' else (only_arm,)
    completion = 'completed.json' if scope == 'full' else f'{scope}-completed.json'
    if only_arm is not None:
        completion = f'arm-{only_arm}-completed.json'
    manifest = json.loads((study / 'manifest.json').read_text())
    inherited = manifest['inherited_pretraining'].get(f'pretrain-{seed}')
    source_path = Path(inherited['path']) if inherited else study / f'pretrain/seed-{seed}/actor.pth'
    paths = {a: study / f'training/seed-{seed}/{a}/{stage}-endpoint.pth' for a in arms}
    inputs = dict(stage=stage, seed=seed, endpoints={a: digest(p) for a, p in paths.items()},
                  source=digest(source_path), episodes=100, states=32, repetitions=2,
                  candidate_response=CANDIDATE_RESPONSE_PROTOCOL, candidate_resamples=4)
    input_path = output / ('inputs.json' if scope == 'full' else f'inputs-{scope}.json')
    if only_arm is not None:
        input_path = output / f'inputs-arm-{only_arm}.json'
    if input_path.exists() and json.loads(input_path.read_text()) != inputs:
        raise ValueError('Review inputs changed; preserve the existing version and create a new study revision.')
    write_json(input_path, inputs)
    source_weights = torch.load(source_path, weights_only=True, map_location='cpu')
    source = OriginalPolicy(source_path)
    base_config = yaml.safe_load((Path(__file__).parent / 'configs/interaction-aware-sac.yaml').read_text())
    expected_step = 250000 if stage == 'gate' else 1250000
    evaluated = (*arms, 'initial_cbf') if only_arm is None else (only_arm,)
    for arm in evaluated:
        result_path = output / f'{arm}.json'
        arm_input = dict(stage=stage, seed=seed, source=inputs['source'],
                         endpoint=inputs['endpoints'].get(arm), episodes=100, states=32, repetitions=2,
                         candidate_response=CANDIDATE_RESPONSE_PROTOCOL, candidate_resamples=4)
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            if existing.get('input') != arm_input:
                raise ValueError('Cached validation inputs changed; do not reuse a different endpoint.')
            continue
        started = time.perf_counter()
        response = None
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
            if arm != 'arboids_cbf':
                response = validation_candidate_response(saved, source, seed)
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
        if response:
            summary['candidate_response'] = response['summary']
        write_json(result_path, dict(arm=arm, stage=stage, seed=seed, input=arm_input, checks=checks, summary=summary,
            episodes=rows, calibration=calibration, candidate_response=response, validation_steps=steps,
            candidate_response_steps=0 if response is None else response['simulated_steps'],
            calibration_steps=sum(r['simulated_steps'] for r in calibration),
            wall_seconds=time.perf_counter()-started))
    results = {a: json.loads((output / f'{a}.json').read_text()) for a in evaluated}
    matched = [results[a]['checks']['actor_parameters'] for a in arms if a != 'arboids_cbf']
    if matched and len(set(matched)) != 1:
        raise ValueError('The same-information actor capacities are not matched.')
    write_json(output / completion, dict(complete=True, scope=scope, inputs=inputs,
        artifacts={a: digest(output / f'{a}.json') for a in results},
        validation_steps=sum(r['validation_steps'] for r in results.values()),
        calibration_steps=sum(r['calibration_steps'] for r in results.values()),
        candidate_response_steps=sum(r['candidate_response_steps'] for r in results.values()),
        wall_seconds=sum(r['wall_seconds'] for r in results.values())))


def review_evidence(study, stage, seeds, scope='full'):
    """Small reports only: the scheduler does not load models or use a GPU."""
    import numpy as np
    study = Path(study)
    arms = SCOPE_ARMS[scope]
    completion = 'completed.json' if scope == 'full' else f'{scope}-completed.json'
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
            if arm in ('full','same_info','short','model_value','no_peer'):
                calibration=raw[seed][arm]['calibration']
                if len(calibration)!=32 or [r['state'] for r in calibration]!=list(range(32)):
                    raise ValueError('Validation intervention states are incomplete or unpaired.')
                response = raw[seed][arm].get('candidate_response', {})
                states = response.get('states', [])
                if (response.get('protocol') != CANDIDATE_RESPONSE_PROTOCOL or response.get('resamples') != 4 or
                        len(states) != 32 or [r['state'] for r in states] != list(range(32)) or
                        [r['scene_seed'] for r in states] != [350000000+SEEDS.index(seed)*1000000+i for i in range(32)] or
                        any(len(r['perturbations']) != 4 for r in states)):
                    raise ValueError('Controlled candidate response evidence is incomplete or unpaired.')
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
        if reference in ('same_info', 'model_value', 'short', 'no_peer'):
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
            'Inspect fixed-state peer-candidate responses and separate gate effects from direct CBF coupling.',
            'Gate response alone establishes sensitivity, not useful cooperation; assess the matched no-peer control.',
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
    rows, inputs, input_paths, artifact_inputs = [], {}, {}, {}
    for job in jobs:
        spec = job['validation']
        path = Path(job['result'])
        result = json.loads(path.read_text())
        checkpoint = study/f'training/seed-{spec["seed"]}/{spec["arm"]}/policy.pth'
        fingerprint = digest(checkpoint)
        if (not result.get('passed') or result.get('outcome_code') not in (1,2,3,4) or
                result.get('seed') != spec['scene_seed'] or result.get('setting') != spec['setting'] or
                result.get('num_robots') != spec['defenders']+1 or result.get('controller') != 'IACRRL' or
                result.get('duration_limit') != 60 or result.get('termination_rule') != 'paper' or
                result.get('agility') != 2.25 or result.get('checkpoint_sha256') != fingerprint):
            raise ValueError('Deployment validation has an invalid or mismatched episode.')
        artifact_inputs[str(checkpoint.relative_to(study))] = fingerprint
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
    return dict(stage='deployment', seeds=list(SEEDS[:2]), inputs=inputs, input_paths=input_paths, artifact_inputs=artifact_inputs,
        technical_checks_passed=True, episodes=rows, seed_contrasts=contrasts,
        scope='Independent deployment validation only; five paired scenes per seed and condition, not final VRX evidence',
        interpretation='Inspect task failures and consistency across both seeds; this small bank screens transfer before replication.')


def source_for_seed(study, seed):
    manifest = json.loads((Path(study)/'manifest.json').read_text())
    inherited = manifest['inherited_pretraining'].get(f'pretrain-{seed}')
    return Path(inherited['path']) if inherited else Path(study)/f'pretrain/seed-{seed}/actor.pth'


def diagnostic_state(source, scene, index):
    """One state per episode; fixed ordinary/crowded strata, never reward-based selection."""
    import numpy as np
    from envs.TADgame import TADEnv
    from envs.snapshot import seed_random
    from interaction_rollout import compact_snapshot, public_packet, execute
    seed_random(scene)
    defenders = 3 if index % 2 == 0 else 6
    env = TADEnv(defenders, protocol='paper-parameters-v1')
    obs, _ = env.reset(2.25, noisy_agility=False)
    packet, used = public_packet(env, obs), 0
    for _ in range(10 + (index // 2) % 50):
        snapshot = compact_snapshot(env)
        action, _ = source.choose_action(packet, deterministic=True)
        packet, _, done, _, _, _ = execute(env, packet, action)
        used += 1
        if done:
            break
    packet = public_packet(snapshot.environment)
    positions = packet['motion'][:, :2]
    distances = np.linalg.norm(positions[:, None]-positions[None, :], axis=-1)
    distances[np.diag_indices(defenders)] = np.inf
    return snapshot, packet, dict(state=index, scene_seed=scene, defenders=defenders,
        stratum='ordinary' if defenders == 3 else 'crowded',
        minimum_pair_distance_m=float(distances.min()), prefix_steps=used)


def paired_consequences(policy, snapshot, left, right, future, noise, horizon, trace=False):
    from envs.snapshot import preserved_random_state
    from interaction_rollout import branch_return
    with preserved_random_state():
        a, na, ta = branch_return(policy, snapshot, left, future, noise, horizon, trace=trace)
        b, nb, tb = branch_return(policy, snapshot, right, future, noise, horizon, trace=trace)
    return a-b, na+nb, (ta, tb), (a, b)


def signal_state_checks(policy, snapshot, packet, scene, index, horizon=100):
    import pickle
    import numpy as np
    import torch
    from envs.snapshot import RandomState
    before = pickle.dumps(snapshot)
    random_before = RandomState.capture()
    noise, future = scene+100000, scene+200000
    action, _ = policy.action(packet, torch.Generator().manual_seed(noise))
    boat = index % len(action)
    left, right = action.copy(), action.copy()
    left[boat, 2], right[boat, 2] = 0., 1.
    delta, used, trace, returns = paired_consequences(policy, snapshot, left, right, future, noise, horizon, True)
    zero, count, zero_trace, _ = paired_consequences(policy, snapshot, left, left, future, noise, horizon, True)
    used += count
    swapped, count, reverse_trace, _ = paired_consequences(policy, snapshot, right, left, future, noise, horizon, True)
    used += count
    def traces_equal(a, b):
        return len(a) == len(b) and all(all(np.array_equal(x[k], y[k]) for k in x) for x, y in zip(a, b))
    identical = traces_equal(zero_trace[0], zero_trace[1])
    order_independent = traces_equal(trace[0], reverse_trace[1]) and traces_equal(trace[1], reverse_trace[0])
    root_difference = np.zeros_like(left)
    root_difference[boat, 2] = -1.
    root_correct = np.array_equal(trace[0][0]['action']-trace[1][0]['action'], root_difference)
    random_after = RandomState.capture()
    rng_equal = (np.array_equal(random_before.numpy[1], random_after.numpy[1]) and
        random_before.numpy[2:] == random_after.numpy[2:] and random_before.python == random_after.python and
        torch.equal(random_before.torch_cpu, random_after.torch_cpu))
    clean = before == pickle.dumps(snapshot) and rng_equal
    if not (abs(zero) < 1e-8 and abs(delta+swapped) < 1e-8 and identical and order_independent and root_correct and clean):
        raise AssertionError('Paired intervention, branch ordering, snapshot history or RNG isolation failed.')
    nominal = 750.*action[boat, :2]+250.-snapshot.environment.boids_actions[boat]
    executed = trace[1][0]['thrust'][boat]-trace[0][0]['thrust'][boat]
    return dict(boat=boat, left_gate=0., right_gate=1., model_difference=delta,
        branch_returns=list(returns), zero_difference=zero, swap_residual=delta+swapped,
        current_candidates_fixed=True, future_feedback_same_policy=True, snapshot_and_rng_unchanged=clean,
        nominal_thrust_difference_N=float(np.linalg.norm(nominal)),
        executed_thrust_difference_N=float(np.linalg.norm(executed)), simulated_steps=used,
        action=action.tolist(), obs=packet['obs'].tolist(), motion=packet['motion'].tolist(),
        central=packet['central'].tolist())


def _diagnostic_initializer(payload, source_path):
    import study_runtime
    import torch
    from interaction_rollout import FrozenPolicy
    from interaction_evaluation import OriginalPolicy
    torch.set_num_threads(1)
    global _DIAGNOSTIC_POLICY, _DIAGNOSTIC_SOURCE
    _DIAGNOSTIC_POLICY, _DIAGNOSTIC_SOURCE = FrozenPolicy(payload), OriginalPolicy(source_path)


def _signal_worker(job):
    scene, index = job
    snapshot, packet, row = diagnostic_state(_DIAGNOSTIC_SOURCE, scene, index)
    row.update(signal_state_checks(_DIAGNOSTIC_POLICY, snapshot, packet, scene, index))
    row['split'] = 'debug' if index < 64 else 'heldout'
    return row


def collect_signal(study, seed=42, workers=4):
    import copy
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import numpy as np
    import torch
    import yaml
    from envs.snapshot import preserved_random_state, seed_random
    from policy.interaction_sac import make_agent
    from interaction_rollout import frozen_payload
    from train_interaction import arm_config
    study, started = Path(study), time.perf_counter()
    source = source_for_seed(study, seed)
    output = study/'reviews/signal'
    inputs = dict(protocol=SIGNAL_PROTOCOL, source=digest(source), seed=seed, states=128,
        split='64 debug + 64 heldout, one independently seeded episode per state; equal 3/6-defender strata',
        scene_base=390000000+DEVELOPMENT_SEEDS.index(seed)*1000000,
        horizon=100, reward='discounted team reward plus future joint-policy entropy; no first-action entropy')
    if (output/'completed.json').exists():
        existing = json.loads((output/'completed.json').read_text())
        if existing['input'] != inputs or digest(output/'states.json') != existing['states_sha256']:
            raise ValueError('The completed signal bank does not match this development protocol.')
        return existing
    with preserved_random_state():
        seed_random(seed)
        config = arm_config(yaml.safe_load((Path(__file__).parent/'configs/interaction-aware-sac.yaml').read_text()), 'full')
        agent = make_agent(config)
        agent.actor.initialize_source(torch.load(source, map_location='cpu', weights_only=True))
        payload = frozen_payload(agent)
        del agent
    jobs = [(inputs['scene_base']+i, i) for i in range(128)]
    if workers:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                initializer=_diagnostic_initializer, initargs=(payload, str(source))) as executor:
            rows = list(executor.map(_signal_worker, jobs))
    else:
        _diagnostic_initializer(payload, str(source))
        rows = [_signal_worker(j) for j in jobs]
    write_json(output/'states.json', dict(input=inputs, states=rows))
    identifiable = {s: dict(states=sum(r['stratum']==s for r in rows),
        executed_nonzero=sum(r['stratum']==s and r['executed_thrust_difference_N']>1e-6 for r in rows),
        consequence_nonzero=sum(r['stratum']==s and abs(r['model_difference'])>1e-6 for r in rows))
        for s in ('ordinary', 'crowded')}
    result = dict(complete=True, technical_checks_passed=True, input=inputs, identifiability=identifiable,
        states_sha256=digest(output/'states.json'), simulated_steps=sum(r['simulated_steps'] for r in rows),
        state_collection_steps=sum(r['prefix_steps'] for r in rows), wall_seconds=time.perf_counter()-started,
        interpretation='Legal gate interventions and truncated model labels are identifiable only where reported; this does not establish learned utility.')
    write_json(output/'completed.json', result)
    return result


def _calibration_worker(job):
    import numpy as np
    import torch
    from policy.interaction_sac import tensor_packet
    scene, index, horizon, repetitions = job
    policy = _DIAGNOSTIC_POLICY
    snapshot, packet, row = diagnostic_state(_DIAGNOSTIC_SOURCE, scene, index)
    action, _ = policy.action(packet, torch.Generator().manual_seed(scene+100000))
    alternative = action.copy()
    boat, reference = index % len(action), (0., .5, 1.)[(index//3)%3]
    alternative[boat, 2] = reference
    p = {k: v.unsqueeze(0) for k, v in tensor_packet(packet).items()}
    with torch.no_grad():
        qa = policy.critic(p, torch.as_tensor(action).unsqueeze(0))
        qb = policy.critic(p, torch.as_tensor(alternative).unsqueeze(0))
        predictions = [float((a-b).item()) for a,b in zip(qa, qb)]
    differences, used, execution_delta = [], 0, None
    # Evaluation critic predicts delta Q; model bootstrap uses the exact lagged training critic.
    original_critic = policy.critic
    if horizon < 300 and hasattr(policy, 'label_critic'):
        policy.critic = policy.label_critic
    try:
        for repeat in range(repetitions):
            delta, count, traces, _ = paired_consequences(policy, snapshot, action, alternative,
                scene+200000+repeat*10000, scene+300000+repeat*10000, horizon, True)
            differences.append(delta)
            used += count
            if execution_delta is None:
                execution_delta = float(np.linalg.norm(traces[0][0]['thrust']-traces[1][0]['thrust']))
    finally:
        policy.critic = original_critic
    actual = float(np.mean(differences))
    row.update(boat=boat, reference=reference, critic_predictions=predictions, target_difference=actual,
        absolute_error=float(np.mean(np.abs(np.asarray(predictions)-actual))),
        zero_prediction_error=abs(actual), executed_thrust_difference_N=execution_delta,
        monte_carlo_standard_error=float(np.std(differences, ddof=1)/np.sqrt(repetitions)) if repetitions>1 else None,
        simulated_steps=used)
    return row


def _screen_initializer(payload, source_path):
    from policy.interaction_sac import TwinTeamCritic
    _diagnostic_initializer(payload, source_path)
    config = payload['config']
    target = TwinTeamCritic(config['rl']['hidden_dim'], config['interaction']['relation_dim']).eval().requires_grad_(False)
    target.load_state_dict(payload['target'])
    _DIAGNOSTIC_POLICY.label_critic = target


def collect_screen(study, seed, arm, checkpoint_step, workers=2, fresh=False):
    import copy
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import numpy as np
    import torch
    from envs.snapshot import preserved_random_state
    from policy.interaction_sac import make_agent, JointReplay
    from interaction_rollout import frozen_payload, DeploymentPolicy
    from interaction_evaluation import OriginalPolicy
    from train_interaction import episode
    torch.set_num_threads(1)
    study, started = Path(study), time.perf_counter()
    directory = study/f'training/seed-{seed}/{arm}'
    checkpoint = directory/('resume.pth' if checkpoint_step == 0 else f'probe-{checkpoint_step}.pth')
    source = source_for_seed(study, seed)
    output = study/'reviews'/('confirmation' if fresh else 'screens')/f'{seed}-{arm}-{checkpoint_step}'
    inputs = dict(protocol='fresh-two-density-v2' if fresh else 'heldout-model-and-environment-v1', checkpoint_sha256=digest(checkpoint),
                  source_sha256=digest(source), seed=seed, arm=arm, fresh=fresh)
    if (output/'completed.json').exists():
        existing = json.loads((output/'completed.json').read_text())
        if existing['input'] != inputs:
            raise ValueError('A reviewed screen checkpoint changed.')
        if 'candidate_response' not in existing:
            saved=torch.load(checkpoint,weights_only=checkpoint_step!=0,map_location='cpu')
            payload=(dict(config=saved['config'],actor=saved['agent']['actor'],critic=saved['agent']['critic'],
                alpha=float(saved['agent']['log_alpha'].exp()),gamma=saved['config']['rl']['GAMMA'])
                if checkpoint_step==0 else saved)
            base=(398000000 if fresh else 392000000)+DEVELOPMENT_SEEDS.index(seed)*1000000
            response=validation_candidate_response(payload,OriginalPolicy(source),seed,states=16,resamples=2,
                                                   scene_base=base+300000)
            existing.update(candidate_response=response,candidate_response_cbf_evaluations=response['cbf_evaluations'])
            existing['simulated_steps']+=response['simulated_steps']
            existing['wall_seconds']+=time.perf_counter()-started
            write_json(output/'completed.json',existing)
        return existing
    with preserved_random_state():
        saved = torch.load(checkpoint, weights_only=checkpoint_step != 0, map_location='cpu')
        if checkpoint_step == 0:
            agent = make_agent(saved['config'])
            agent.load_state_dict(saved['agent'])
            payload = frozen_payload(agent, online_critic=True)
            payload.update(step=saved['step'], stage=agent.stage, target=saved['agent']['target'])
            replay = JointReplay(1)
            replay.load_state_dict(saved['replay'])
            # One isolated update checks actual replay/labels/gradients; it never modifies the saved learner.
            saved['random'].torch_cuda=None  # The isolated probe runs on CPU.
            saved['random'].restore()
            checks = agent.learn(replay, saved['auxiliary'], diagnostics=True)
            del agent, replay, saved
        else:
            payload, checks = saved, saved.get('learning_checks', {})
        source_weights = torch.load(source, weights_only=True, map_location='cpu')
        endpoint = check_endpoint(payload, source_weights, payload['stage'], payload['step'])
    if not checks or any(not np.isfinite(v) for v in checks.values()):
        raise FloatingPointError('Missing or non-finite learning connectivity evidence.')
    # Training uses the 100M scene family. Model holdout and environment checks use disjoint episodes.
    base = (398000000 if fresh else 392000000)+DEVELOPMENT_SEEDS.index(seed)*1000000
    horizon = payload['config']['interaction']['horizon_steps']
    jobs = [(base+i, i, horizon, 1) for i in range(32)]
    jobs += [(base+100000+i, i, 300, 2) for i in range(16)]
    if workers:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                initializer=_screen_initializer, initargs=(payload, str(source))) as executor:
            rows = list(executor.map(_calibration_worker, jobs))
    else:
        _screen_initializer(payload, str(source))
        rows = [_calibration_worker(j) for j in jobs]
    model, environment = rows[:32], rows[32:]
    response=validation_candidate_response(payload,OriginalPolicy(source),seed,states=16,resamples=2,
                                           scene_base=base+300000)
    # Match actual task episodes across arms at each checkpoint; no best-checkpoint selection.
    class Policy:
        def choose_action(self, packet, deterministic=True):
            from policy.interaction_sac import tensor_packet
            p = {k: v.unsqueeze(0) for k,v in tensor_packet(packet).items()}
            with torch.no_grad():
                a,_ = _DIAGNOSTIC_POLICY.actor(p['obs'], p['motion'], deterministic=True)
            return a[0].numpy(), 0.
    _screen_initializer(payload, str(source))
    episodes = [dict(scene_seed=base+200000+i, defenders=3, **episode(Policy(), base+200000+i)[0]) for i in range(32)]
    if fresh:
        episodes.extend(dict(scene_seed=base+210000+i,defenders=6,
            **episode(Policy(),base+210000+i,defenders=6)[0]) for i in range(32))
    task_by_defenders={str(n):{m:float(np.mean([r[m] for r in episodes if r['defenders']==n])) for m in METRICS}
                      for n in sorted({r['defenders'] for r in episodes})}
    summary = {m: float(np.mean([r[m] for r in episodes])) for m in METRICS}
    summary.update(E_model=float(np.mean([r['absolute_error'] for r in model])),
                   E_env=float(np.mean([r['absolute_error'] for r in environment])),
                   environment_zero_predictor=float(np.mean([r['zero_prediction_error'] for r in environment])),
                   executed_nonzero_fraction=float(np.mean([r['executed_thrust_difference_N']>1e-6 for r in environment])))
    result = dict(complete=True, technical_checks_passed=True, input=inputs, step=payload['step'],
        frozen_proposals=endpoint, learning_checks=checks, model=model, environment=environment,
        episodes=episodes, summary=summary,task_by_defenders=task_by_defenders,
        candidate_response=response,candidate_response_cbf_evaluations=response['cbf_evaluations'],
        simulated_steps=sum(r['simulated_steps']+r['prefix_steps'] for r in rows)+response['simulated_steps'],
        task_steps=sum(round(r['duration']/.2) for r in episodes), wall_seconds=time.perf_counter()-started,
        scope='Development only; E_model uses heldout truncated lagged-critic labels, E_env uses disjoint full closed-loop episodes with the same reward, discount, entropy and terminal rule.')
    if digest(checkpoint) != inputs['checkpoint_sha256']:
        raise ValueError('The checkpoint changed while the screen was running.')
    write_json(output/'completed.json', result)
    return result


def _mechanism_worker(job):
    import numpy as np
    import torch
    from envs.snapshot import preserved_random_state
    from interaction_rollout import branch_return
    scene,index=job
    policy=_DIAGNOSTIC_POLICY
    snapshot,packet,row=diagnostic_state(_DIAGNOSTIC_SOURCE,scene,index)
    action,_=policy.action(packet,torch.Generator().manual_seed(scene+100000))
    boat,peer=index % len(action),(index+1) % len(action)
    reference=(0.,.5,1.)[(index//3)%3]
    configurations={name:action.copy() for name in ('00','10','01','11','mean','permuted')}
    configurations['10'][boat,2]=configurations['11'][boat,2]=reference
    configurations['01'][peer,2]=configurations['11'][peer,2]=reference
    configurations['mean'][:,2]=action[:,2].mean()
    permutation=np.random.default_rng(scene).permutation(len(action))
    configurations['permuted'][:,2]=action[permutation,2]
    returns={name:[] for name in configurations}
    first_thrust,examples,used={},{},0
    with preserved_random_state():
        for repeat in range(4):
            for name,a in configurations.items():
                value,count,trace=branch_return(policy,snapshot,a,scene+200000+repeat*10000,
                    scene+300000+repeat*10000,300,trace=True)
                returns[name].append(value)
                used+=count
                if repeat==0:
                    first_thrust[name]=trace[0]['thrust'].tolist()
                    if index<2:
                        examples[name]=[{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in t.items()} for t in trace]
    r={k:np.asarray(v) for k,v in returns.items()}
    contrasts=dict(mean=r['00']-r['mean'],permuted=r['00']-r['permuted'],
                   conditional_interaction=r['11']-r['10']-r['01']+r['00'])
    row.update(boat=boat,peer=peer,reference=reference,permutation=permutation.tolist(),
        configurations={k:v.tolist() for k,v in configurations.items()},executed_first_thrust=first_thrust,
        returns=returns,contrasts={k:dict(mean=float(v.mean()),standard_error=float(v.std(ddof=1)/2)) for k,v in contrasts.items()},
        simulated_steps=used,trajectory_examples=examples)
    return row


def _target_audit_worker(job):
    import numpy as np
    import torch
    from envs.snapshot import preserved_random_state
    from interaction_rollout import branch_return
    scene,index=job
    policy=_DIAGNOSTIC_POLICY
    snapshot,packet,row=diagnostic_state(_DIAGNOSTIC_SOURCE,scene,index)
    action,_=policy.action(packet,torch.Generator().manual_seed(scene+100000))
    alternative=action.copy()
    alternative[index % len(action),2]=(0.,.5,1.)[(index//3)%3]
    horizons={str(h):[] for h in (10,100,300)}
    used=0
    with preserved_random_state():
        for repeat in range(2):
            for horizon in (10,100,300):
                values,tails,terminated=[],[],[]
                online=policy.critic
                policy.critic=policy.label_critic
                try:
                    for a in (action,alternative):
                        value,count,trace=branch_return(policy,snapshot,a,scene+200000+repeat*10000,
                            scene+300000+repeat*10000,horizon,trace=True)
                        observed=sum(policy.gamma**k*(t['reward']-policy.alpha*t['logp']) for k,t in enumerate(trace))
                        values.append(value);tails.append(value-observed);terminated.append(bool(trace[-1]['outcome']))
                        used+=count
                finally:
                    policy.critic=online
                horizons[str(horizon)].append(dict(difference=values[0]-values[1],
                    bootstrap_difference=tails[0]-tails[1],bootstrap_values=tails,terminated=terminated))
    row.update(horizons=horizons,simulated_steps=used)
    return row


def collect_target_audit(study,seed,arm,workers=4):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import numpy as np
    import torch
    study,started=Path(study),time.perf_counter()
    checkpoint=study/f'training/seed-{seed}/{arm}/resume.pth'
    source=source_for_seed(study,seed)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    inputs=dict(checkpoint_sha256=digest(checkpoint),source_sha256=digest(source),seed=seed,arm=arm,
                states=16,repetitions=2,scene_base=394000000+DEVELOPMENT_SEEDS.index(seed)*1000000)
    suffix=f'-{saved["step"]}-real_td' if saved['config']['interaction'].get('bootstrap_source')=='real_td' else ''
    output=study/f'reviews/target-audit/{seed}-{arm}{suffix}'
    if (output/'completed.json').exists():
        previous=json.loads((output/'completed.json').read_text())
        if previous['input']!=inputs: raise ValueError('Target-audit checkpoint changed.')
        return previous
    payload=dict(config=saved['config'],actor=saved['agent']['actor'],critic=saved['agent']['critic'],
        target=saved['agent']['target'],alpha=float(saved['agent']['log_alpha'].exp()),
        gamma=saved['config']['rl']['GAMMA'])
    step=saved['step']
    del saved
    jobs=[(inputs['scene_base']+i,i) for i in range(16)]
    if workers:
        with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),
                initializer=_screen_initializer,initargs=(payload,str(source))) as executor:
            rows=list(executor.map(_target_audit_worker,jobs))
    else:
        _screen_initializer(payload,str(source))
        rows=[_target_audit_worker(j) for j in jobs]
    summary={}
    for h in ('10','100'):
        errors=[abs(m['difference']-e['difference']) for r in rows
                for m,e in zip(r['horizons'][h],r['horizons']['300'])]
        summary[h]=dict(label_to_environment_mae=float(np.mean(errors)),
            mean_absolute_bootstrap_difference=float(np.mean([abs(m['bootstrap_difference']) for r in rows for m in r['horizons'][h]])),
            one_branch_terminal=sum(sum(m['terminated'])==1 for r in rows for m in r['horizons'][h]))
    result=dict(complete=True,input=inputs,step=step,summary=summary,states=rows,
        simulated_steps=sum(r['simulated_steps']+r['prefix_steps'] for r in rows),wall_seconds=time.perf_counter()-started,
        interpretation='Paired short/long/bootstrap audit on the same heldout states and noise. This diagnoses target bias; it does not select a horizon from final tests.')
    if digest(checkpoint)!=inputs['checkpoint_sha256']: raise ValueError('Target-audit checkpoint changed while running.')
    write_json(output/'completed.json',result)
    return result


def collect_mechanism(study, seed, workers=2):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    import torch
    from interaction_evaluation import SEEDS as formal_seeds
    study,started=Path(study),time.perf_counter()
    checkpoint=study/f'training/seed-{seed}/full/joint-endpoint.pth'
    source=source_for_seed(study,seed)
    payload=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if payload['step']!=1000000 or payload['stage']!='joint':
        raise ValueError('Mechanism evidence requires the frozen formal one-million-step endpoint.')
    inputs=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),source=str(source),
                source_sha256=digest(source),states=32,repetitions=4,
                scene_base=530000000+formal_seeds.index(seed)*1000000)
    output=study/f'mechanism/seed-{seed}'
    if (output/'completed.json').exists():
        previous=json.loads((output/'completed.json').read_text())
        if previous['input']!=inputs or digest(output/'states.json')!=previous['states_sha256']:
            raise ValueError('Frozen mechanism inputs changed.')
        return previous
    jobs=[(inputs['scene_base']+i,i) for i in range(32)]
    if workers:
        with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),
                initializer=_diagnostic_initializer,initargs=(payload,str(source))) as executor:
            rows=list(executor.map(_mechanism_worker,jobs))
    else:
        _diagnostic_initializer(payload,str(source))
        rows=[_mechanism_worker(j) for j in jobs]
    write_json(output/'states.json',dict(input=inputs,states=rows))
    result=dict(complete=True,input=inputs,states_sha256=digest(output/'states.json'),
        simulated_steps=sum(r['simulated_steps'] for r in rows),
        state_collection_steps=sum(r['prefix_steps'] for r in rows),wall_seconds=time.perf_counter()-started,
        interpretation='Mean/permutation tests configuration matching; four branches measure conditional nonadditivity. Root candidates and future noise are shared; subsequent actions use the same frozen feedback policy.')
    write_json(output/'completed.json',result)
    return result


def _bootstrap_check_initializer(payloads, source):
    import torch
    from interaction_rollout import FrozenPolicy
    from interaction_evaluation import OriginalPolicy
    from policy.interaction_sac import TwinTeamCritic
    torch.set_num_threads(1)
    global _BOOTSTRAP_POLICIES, _BOOTSTRAP_SOURCE
    _BOOTSTRAP_SOURCE = OriginalPolicy(source)
    _BOOTSTRAP_POLICIES = {}
    for name, payload in payloads.items():
        policy = FrozenPolicy(payload)
        cfg = payload['config']
        target = TwinTeamCritic(cfg['rl']['hidden_dim'], cfg['interaction']['relation_dim']).eval().requires_grad_(False)
        target.load_state_dict(payload['target'])
        policy.label_critic = target
        _BOOTSTRAP_POLICIES[name] = policy


def _bootstrap_check_worker(job):
    """One full path supplies every truncation and its exact same-noise tail."""
    import numpy as np
    import torch
    from envs.snapshot import preserved_random_state
    from interaction_rollout import branch_return
    from policy.interaction_sac import tensor_packet
    scene, index, repetitions = job
    policies = _BOOTSTRAP_POLICIES
    policy = next(iter(policies.values()))
    snapshot, packet, row = diagnostic_state(_BOOTSTRAP_SOURCE, scene, index)
    action, _ = policy.action(packet, torch.Generator().manual_seed(scene+100000))
    alternative = action.copy()
    alternative[index % len(action), 2] = (0., .5, 1.)[(index//3) % 3]
    actions = (action, alternative)
    p = {k: v.unsqueeze(0) for k, v in tensor_packet(packet).items()}
    predictions = {}
    with torch.no_grad():
        for name, candidate in policies.items():
            q = [candidate.critic(p, torch.as_tensor(a).unsqueeze(0)) for a in actions]
            predictions[name] = [float((a-b).item()) for a,b in zip(*q)]
    values, errors, tail_errors, used = [], {k:{10:[],100:[]} for k in policies}, {k:[] for k in policies}, 0
    for repeat in range(repetitions):
        branches = []
        for a in actions:
            with preserved_random_state():
                value, count, trace = branch_return(policy, snapshot, a, scene+200000+10000*repeat,
                    scene+300000+10000*repeat, 300, trace=True, record_states=True)
            if not trace[-1]['outcome']:
                raise AssertionError('Calibration requires terminal environment returns.')
            used += count
            branches.append((value, trace))
        actual = branches[0][0]-branches[1][0]
        values.append(actual)
        for name, candidate in policies.items():
            for horizon in (10,100):
                labels=[]
                for value, trace in branches:
                    if len(trace) <= horizon:
                        labels.append(value)
                        continue
                    prefix = sum(policy.gamma**k*(t['reward']-policy.alpha*t['logp']) for k,t in enumerate(trace[:horizon]))
                    boundary = trace[horizon]
                    bp = {k:v.unsqueeze(0) for k,v in tensor_packet(boundary['packet']).items()}
                    with torch.no_grad():
                        q = candidate.label_critic(bp, torch.as_tensor(boundary['action']).unsqueeze(0))
                    predicted = float(torch.minimum(*q).item())-policy.alpha*boundary['logp']
                    labels.append(prefix+policy.gamma**horizon*predicted)
                    if horizon == 100:
                        tail_errors[name].append(abs(predicted-(value-prefix)/policy.gamma**horizon))
                errors[name][horizon].append(abs(labels[0]-labels[1]-actual))
    actual = float(np.mean(values))
    row.update(environment_difference=actual, zero_prediction_error=abs(actual),
        monte_carlo_standard_error=float(np.std(values,ddof=1)/np.sqrt(repetitions)) if repetitions>1 else None,
        variants={name:dict(critic_predictions=predictions[name],
            environment_error=float(np.mean(np.abs(np.asarray(predictions[name])-actual))),
            label_errors={str(h):errors[name][h] for h in (10,100)}, tail_errors=tail_errors[name]) for name in policies},
        simulated_steps=used)
    return row


def bootstrap_calibration(payloads, source, *, scene_base, states=64, repetitions=2, workers=4):
    """Compare teachers at a fixed policy on disjoint, preselected episodes."""
    import numpy as np
    import torch
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from envs.snapshot import preserved_random_state
    first = next(iter(payloads.values()))
    for payload in payloads.values():
        if payload['alpha'] != first['alpha'] or payload['gamma'] != first['gamma']:
            raise ValueError('Bootstrap comparisons must use the same return definition.')
        if any(not torch.equal(v,payload['actor'][k]) for k,v in first['actor'].items()):
            raise ValueError('Bootstrap comparisons require the identical frozen actor.')
    started = time.perf_counter()
    jobs=[(scene_base+i,i,repetitions) for i in range(states)]
    if workers:
        with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn'),
            initializer=_bootstrap_check_initializer,initargs=(payloads,str(source))) as executor:
            rows=list(executor.map(_bootstrap_check_worker,jobs))
    else:
        with preserved_random_state():
            _bootstrap_check_initializer(payloads,str(source))
            rows=[_bootstrap_check_worker(j) for j in jobs]
    summary={}
    for name in payloads:
        summary[name]={}
        for stratum in ('all','ordinary','crowded'):
            selected=[r for r in rows if stratum=='all' or r['stratum']==stratum]
            if not selected: continue
            tails=[e for r in selected for e in r['variants'][name]['tail_errors']]
            summary[name][stratum]=dict(environment_mae=float(np.mean([r['variants'][name]['environment_error'] for r in selected])),
                zero_prediction_mae=float(np.mean([r['zero_prediction_error'] for r in selected])),
                horizon_10_label_mae=float(np.mean([e for r in selected for e in r['variants'][name]['label_errors']['10']])),
                horizon_100_label_mae=float(np.mean([e for r in selected for e in r['variants'][name]['label_errors']['100']])),
                tail_value_mae=float(np.mean(tails)) if tails else None)
    return dict(protocol='fixed-policy-complete-return-bootstrap-calibration-v1',scene_base=scene_base,
        state_count=states,repetitions=repetitions,summary=summary,states=rows,
        simulated_steps=sum(r['simulated_steps']+r['prefix_steps'] for r in rows),wall_seconds=time.perf_counter()-started)


def collect_bootstrap_check(study,seed,arm,step,workers=4):
    import torch
    study=Path(study)
    checkpoint=study/f'training/seed-{seed}/{arm}/probe-{step}.pth'
    source=source_for_seed(study,seed)
    inputs=dict(seed=seed,arm=arm,checkpoint_sha256=digest(checkpoint),source_sha256=digest(source),
                scene_base=402000000+DEVELOPMENT_SEEDS.index(seed)*1000000,states=64,repetitions=2)
    output=study/f'reviews/bootstrap-check/{seed}-{arm}-{step}/completed.json'
    if output.exists():
        previous=json.loads(output.read_text())
        if previous['input']!=inputs: raise ValueError('The fixed bootstrap check changed.')
        return previous
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if payload['config']['interaction'].get('bootstrap_source')!='real_td':
        raise ValueError('Post-repair verification requires the independent bootstrap protocol.')
    result=bootstrap_calibration(dict(current=payload),source,scene_base=inputs['scene_base'],workers=workers)
    result.update(complete=True,input=inputs,step=step,
        interpretation='Fresh development episodes after resumed updates. Root calibration and same-noise tail errors are separate; these unequal-budget checkpoints do not establish a method ranking.')
    if digest(checkpoint)!=inputs['checkpoint_sha256']: raise ValueError('The bootstrap probe changed during verification.')
    write_json(output,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--mode', choices=['legacy', 'signal', 'screen', 'mechanism', 'target-audit','bootstrap-check'], default='legacy')
    parser.add_argument('--stage', choices=['gate', 'joint'])
    parser.add_argument('--seed', type=int, choices=(*SEEDS,505,606), required=True)
    parser.add_argument('--scope', choices=list(SCOPE_ARMS), default='full')
    parser.add_argument('--arm', choices=[*ARMS, 'no_peer', 'initial_cbf'])
    parser.add_argument('--checkpoint-step', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    if args.mode == 'signal':
        collect_signal(args.study, args.seed, args.workers)
    elif args.mode == 'screen':
        collect_screen(args.study, args.seed, args.arm, args.checkpoint_step, args.workers, args.fresh)
    elif args.mode == 'mechanism':
        collect_mechanism(args.study,args.seed,args.workers)
    elif args.mode == 'target-audit':
        collect_target_audit(args.study,args.seed,args.arm,args.workers)
    elif args.mode == 'bootstrap-check':
        collect_bootstrap_check(args.study,args.seed,args.arm,args.checkpoint_step,args.workers)
    elif args.stage is None:
        parser.error('--stage is required for legacy endpoint collection')
    else:
        collect(args.study, args.stage, args.seed, args.scope, args.arm)


if __name__ == '__main__':
    main()
