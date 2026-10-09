"""Paired contribution study using only the untouched upstream ARBoids baseline."""
import study_runtime  # Must precede NumPy/SciPy/JAX imports.
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import binomtest

from source_arboids import SourcePolicy, numerical_source, verify_source, snapshot, sha256, source_mixture, verify_source_config
from gate_study_control import GateController, METHODS


_POLICY = None
_CONTROLLERS = {}


def initialize(checkpoint):
    global _POLICY
    torch.set_num_threads(1)
    _POLICY = SourcePolicy(checkpoint)
    verify_source()


def controller(method, defenders):
    key = method, defenders
    if key not in _CONTROLLERS:
        if method == 'cbf':
            from cbf_source_baseline import CBFController
            _CONTROLLERS[key] = CBFController(defenders)
        else:
            _CONTROLLERS[key] = GateController(method)
    return _CONTROLLERS[key]


def setup_scene(scene):
    np.random.seed(scene['scene_seed'])
    env = numerical_source().TADEnv(defender_num=scene['defenders'])
    observation, _ = env.reset(agility=scene['agility'], noisy_agility=False)
    np.random.seed(scene['noise_seed'])
    return env, observation


def minimum_distance(env):
    p = np.asarray([b.pos for b in env.defender_list])
    a, b = np.triu_indices(len(p), 1)
    return float(np.linalg.norm(p[a]-p[b], axis=-1).min())


def digest_transition(digest, action, env, observation, reward, done):
    for values in (action, snapshot(env), env.attacker.pos, env.attacker.velocity,
                   observation, reward, [done, env.Current_T]):
        digest.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())


def direct_original(scene, policy):
    """Independent plain loop: original Actor + original TADEnv.step(AdaRes)."""
    env, observation = setup_scene(scene)
    digest = hashlib.sha256()
    done = 0
    while not done:
        action = policy(observation)
        observation, reward, done, _ = env.step(action, 'AdaRes')
        digest_transition(digest, action, env, observation, reward, done)
    return digest.hexdigest()


def rollout(scene, method, policy=None, trace=False):
    policy = _POLICY if policy is None else policy
    env, observation = setup_scene(scene)
    rule = controller(method, env.defender_num)
    # Compile the upstream JAX solver before timings and restore RNG so even
    # a third-party implementation cannot change the paired environment noise.
    cold_seconds = 0.
    if method == 'cbf':
        rng = np.random.get_state()
        started = time.perf_counter()
        rule.control(snapshot(env), policy(observation), env.boids_actions)
        cold_seconds = time.perf_counter()-started
        np.random.set_state(rng)
    digest = hashlib.sha256()
    done = steps = warnings = changed = 0
    task_return = 0.
    nearest = minimum_distance(env)
    collision = breach = False
    times = []
    rows = []
    max_violation = max_saturation = 0.
    while not done:
        started = time.perf_counter()
        action = policy(observation)
        original = action.copy()
        nominal = source_mixture(action, env.boids_actions)
        # The baseline calls the original step directly, without a gate or
        # physical override. Every added method starts from the same raw actor.
        if method == 'original':
            info = {'warning': False}
            physical = nominal
            passed_action, mode = action, 'AdaRes'
        else:
            action, physical, info = rule.control(snapshot(env), action, env.boids_actions)
            if method == 'cbf':
                passed_action = env.thrust_to_action(physical)
                mode = 'RL'  # Original public API, no modified environment.
                max_violation = max(max_violation, info['cbf_constraint_violation'])
                max_saturation = max(max_saturation, info['cbf_control_saturation'])
            else:
                passed_action, mode = action, 'AdaRes'
        times.append(time.perf_counter()-started)
        altered = (np.max(np.abs(physical.astype(float)-nominal.astype(float))) > 1e-3
                   if method == 'cbf' else not np.array_equal(action[:, 2], original[:, 2]))
        changed += int(altered)
        warnings += int(info['warning'])
        observation, reward, done, _ = env.step(passed_action, mode)
        if not np.isfinite(observation).all() or not np.isfinite(reward).all():
            raise FloatingPointError(f'Non-finite original-source transition: {scene}, {method}')
        distance = minimum_distance(env)
        nearest = min(nearest, distance)
        collision |= distance < env.Collision_R
        breach |= np.linalg.norm(env.attacker.pos) < env.Target_R
        task_return += float(np.mean(reward))
        steps += 1
        digest_transition(digest, passed_action, env, observation, reward, done)
        if trace:
            rows.append(dict(time=float(env.Current_T), positions=snapshot(env).tolist(),
                             attacker=env.attacker.pos.tolist(), theta=action[:, 2].tolist(),
                             minimum_distance=distance, altered=bool(altered)))
        if steps > 401:
            raise RuntimeError('Original-source episode exceeded its 80-second horizon.')
    result = dict(**scene, method=method, success=int(done>2), collision=int(collision),
                  target_breach=int(breach), source_loss=int(done==1), capture=int(done==3),
                  timeout=int(done==4), outcome=int(done), steps=steps,
                  duration=float(env.Current_T), team_return=task_return,
                  minimum_distance=nearest, altered_steps=changed, warning_steps=warnings,
                  policy_seconds_mean=float(np.mean(times)),
                  policy_seconds_p95=float(np.quantile(times, .95)),
                  policy_seconds_max=float(np.max(times)), cold_seconds=cold_seconds,
                  max_cbf_constraint_violation=max_violation,
                  max_cbf_control_saturation=max_saturation, trajectory_sha256=digest.hexdigest())
    return (result, rows) if trace else result


def run_scene(scene, methods):
    # Randomize method order within each paired scene. Environment noise has a
    # separate explicit seed and is reset for every method.
    order = np.random.default_rng(scene['scene_seed']+517).permutation(len(methods))
    result = []
    for position in order:
        method = methods[position]
        row = rollout(scene, method)
        if method == 'original':
            expected = direct_original(scene, _POLICY)
            row['original_equivalent'] = expected == row['trajectory_sha256']
            if not row['original_equivalent']:
                raise AssertionError('Baseline differs from the untouched-source direct rollout.')
        else:
            row['original_equivalent'] = ''
        result.append(row)
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_contrasts(rows, methods):
    index = {(r['cell'], r['scene_seed'], r['method']): r for r in rows}
    scenes = sorted({(r['cell'], r['scene_seed']) for r in rows})
    pairs = [('predictive_joint', m) for m in methods if m != 'predictive_joint']
    results = []
    for i, (a, b) in enumerate(pairs):
        if a not in methods:
            continue
        xa = [index[(*s, a)] for s in scenes]
        xb = [index[(*s, b)] for s in scenes]
        gains = sum(x['success']>y['success'] for x, y in zip(xa, xb))
        losses = sum(x['success']<y['success'] for x, y in zip(xa, xb))
        p = binomtest(gains, gains+losses, .5).pvalue if gains+losses else 1.
        delta = np.asarray([x['success']-y['success'] for x, y in zip(xa, xb)])
        ret = np.asarray([x['team_return']-y['team_return'] for x, y in zip(xa, xb)])
        rng = np.random.default_rng(68127+i)
        take = rng.integers(0, len(scenes), size=(10000, len(scenes)))
        interval = np.quantile(delta[take].mean(-1), [.025, .975])
        ret_interval = np.quantile(ret[take].mean(-1), [.025, .975])
        results.append(dict(method=a, comparison=b, scenes=len(scenes),
            success_gain=float(delta.mean()), success_gain_low=float(interval[0]),
            success_gain_high=float(interval[1]), paired_wins=gains, paired_losses=losses,
            p_raw=float(p), return_gain=float(ret.mean()), return_gain_low=float(ret_interval[0]),
            return_gain_high=float(ret_interval[1])))
    order = sorted(range(len(results)), key=lambda i: results[i]['p_raw'])
    adjusted = 0.
    for rank, i in enumerate(order):
        adjusted = max(adjusted, min(1., (len(results)-rank)*results[i]['p_raw']))
        results[i]['p_holm'] = adjusted
    return results


def summarize(rows):
    result = []
    for cell in sorted({r['cell'] for r in rows}) + ['all']:
        for method in METHODS:
            selected = [r for r in rows if r['method']==method and (cell=='all' or r['cell']==cell)]
            if not selected:
                continue
            result.append(dict(cell=cell, method=method, episodes=len(selected),
                successes=sum(r['success'] for r in selected), collisions=sum(r['collision'] for r in selected),
                source_losses=sum(r['source_loss'] for r in selected),
                actual_breaches=sum(r['target_breach'] for r in selected),
                success_rate=float(np.mean([r['success'] for r in selected])),
                collision_rate=float(np.mean([r['collision'] for r in selected])),
                mean_return=float(np.mean([r['team_return'] for r in selected])),
                mean_altered_steps=float(np.mean([r['altered_steps'] for r in selected])),
                mean_policy_ms=1000.*float(np.mean([r['policy_seconds_mean'] for r in selected])),
                mean_episode_p95_ms=1000.*float(np.mean([r['policy_seconds_p95'] for r in selected]))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--suite', choices=['mechanism','generalization','smoke'], default='mechanism')
    parser.add_argument('--episodes', type=int, default=256, help='Episodes per configuration cell')
    parser.add_argument('--seed', type=int, default=129000000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    if min(args.episodes, args.workers) < 1:
        parser.error('Episodes and workers must be positive.')
    if 'original' not in args.methods or len(set(args.methods)) != len(args.methods):
        parser.error('Each suite must include the untouched original baseline, with no duplicate methods.')
    source = verify_source()
    config = args.checkpoint.with_name('config.yaml')
    verify_source_config(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    combinations = ([(n, a) for n in (2,3,4,5) for a in (1.5,2.,2.5,3.)]
                    if args.suite=='generalization' else [(3,2.25)])
    scenes = []
    for index, (n, agility) in enumerate(combinations):
        for repetition in range(args.episodes):
            seed = args.seed+index*10000+repetition
            scenes.append(dict(cell=f'n{n}-a{agility:g}', defenders=n, agility=agility,
                               scene_seed=seed, noise_seed=seed+1000000))
    from cbf_source_baseline import dependency_provenance
    files = [Path(__file__), Path(__file__).with_name('source_arboids.py'),
             Path(__file__).with_name('gate_study_control.py'), Path(__file__).with_name('cbf_source_baseline.py')]
    code = {p.name: sha256(p) for p in files}
    checkpoint_hash = sha256(args.checkpoint)
    specification = dict(suite=args.suite, source=source, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash, config_sha256=sha256(config), code=code,
        external_baseline=dependency_provenance(), methods=args.methods, scene_count=len(scenes),
        episodes_per_cell=args.episodes, horizon=2., warning_distance=7., change_penalty=.02,
        reactive_gain=1., control_period=.2, original_horizon=80., seed=args.seed,
        candidates='Original theta plus 0, 0.25, 0.5, 0.75, 1 independently for each vessel',
        independent_rule='Simultaneous unilateral minimizers with all peers fixed to their original theta',
        physical_events='Source termination preserved; actual target breaches additionally counted separately',
        stats_unit='Independent scene, methods paired on initial state and future random stream')
    (args.output_dir/'specification.json').write_text(json.dumps(specification, indent=2)+'\n', encoding='utf-8')
    write_csv(args.output_dir/'scenes.csv', scenes)
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(str(args.checkpoint),)) as pool:
        pending = {pool.submit(run_scene, scene, args.methods): scene for scene in scenes}
        for finished, future in enumerate(as_completed(pending), 1):
            rows.extend(future.result())
            if finished==1 or finished%8==0 or finished==len(scenes):
                write_csv(args.output_dir/'episodes.csv', rows)
                print(f'[STUDY] {finished}/{len(scenes)} paired scenes; {len(rows)} method episodes; '
                      f'{time.perf_counter()-started:.1f}s', flush=True)
    rows.sort(key=lambda r:(r['cell'],r['scene_seed'],r['method']))
    write_csv(args.output_dir/'episodes.csv', rows)
    table = summarize(rows)
    contrasts = paired_contrasts(rows, args.methods)
    write_csv(args.output_dir/'main_table.csv', table)
    write_csv(args.output_dir/'paired_contrasts.csv', contrasts)
    unchanged = (code=={p.name:sha256(p) for p in files} and checkpoint_hash==sha256(args.checkpoint)
                 and verify_source()==source and specification['external_baseline']==dependency_provenance())
    equivalent = all(r['original_equivalent'] for r in rows if r['method']=='original')
    summary = dict(passed=bool(unchanged and equivalent), source_unchanged=unchanged,
                   all_original_trajectories_equivalent=equivalent, scene_count=len(scenes),
                   method_episodes=len(rows), elapsed_seconds=time.perf_counter()-started,
                   main_table=[r for r in table if r['cell']=='all'], contrasts=contrasts)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
