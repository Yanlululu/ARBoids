"""Collect frozen ARBoids states and run paired, one-cycle gate interventions.

No network is trained. Exhaustive closed-loop search is an offline diagnostic;
its selected gates are evaluated again with separate continuation seeds.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
from itertools import product
import math
from pathlib import Path

import numpy as np
import torch

from envs.TADgame import TADEnv
from envs.snapshot import SimulationSnapshot, preserved_random_state, seed_random
from policy.interaction_prediction import InteractionPredictor, PredictionConfig
from policy.networks import ActorAdap


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class FrozenARBoids:
    """Deterministic CPU policy; both proposal and original Adapter are frozen."""
    def __init__(self, checkpoint):
        self.path = Path(checkpoint).resolve()
        self.sha256 = digest(self.path)
        state = torch.load(self.path, map_location='cpu', weights_only=True)
        self.actor = ActorAdap(6, 8, 3, int(state['l1.weight'].shape[0]), 1.)
        self.actor.load_state_dict(state, strict=True)
        self.actor.eval().requires_grad_(False)
        if not all(torch.isfinite(p).all() for p in self.actor.parameters()):
            raise FloatingPointError('Non-finite baseline checkpoint.')

    @torch.no_grad()
    def __call__(self, observation):
        action, _ = self.actor(torch.as_tensor(observation, dtype=torch.float32), True, False)
        return action.cpu().numpy().copy()


def validate_action(action, n):
    action = np.asarray(action)
    if action.shape != (n, 3) or not np.isfinite(action).all():
        raise ValueError('Expected finite [defenders, 3] baseline actions.')
    if np.any(np.abs(action[:, :2]) > 1) or np.any((action[:, 2] < 0) | (action[:, 2] > 1)):
        raise ValueError('Baseline proposal/gate outside its bounds.')
    return action.copy()


def corrected_gates(theta, delta):
    """The proposed bounded correction; zero leaves the original gate exact."""
    theta, delta = np.asarray(theta), np.asarray(delta)
    if (theta.shape != delta.shape or not np.isfinite(theta).all() or
            not np.isfinite(delta).all() or np.any((theta < 0) | (theta > 1)) or
            np.any(np.abs(delta) > 1)):
        raise ValueError('Gate and correction must have matching, bounded finite values.')
    return np.where(delta >= 0, theta + (1 - theta) * delta, theta * (1 + delta))


def minimum_distance(env):
    positions = np.array([boat.pos for boat in env.defender_list])
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    return float(distances[np.triu_indices(len(positions), 1)].min())


def trajectory_digest(env):
    result = hashlib.sha256()
    for name in ('Pos_Def', 'Phi_Def', 'Pos_Att', 'Phi_Att', 'Rewards'):
        value = np.ascontiguousarray(getattr(env, name))
        result.update(name.encode())
        result.update(str((value.shape, value.dtype.str)).encode())
        result.update(value.tobytes())
    return result.hexdigest()


@dataclass
class GateCase:
    source_seed: int
    source_step: int
    kind: str
    snapshot: SimulationSnapshot
    observation: np.ndarray
    action: np.ndarray
    lookback_seconds: float | None = None
    source_trajectory_sha256: str | None = None

    @property
    def case_id(self):
        return f'{self.source_seed}:{self.source_step}:{self.kind}'

    @property
    def theta(self):
        return self.action[:, 2].copy()


def collect_cases(policy, seeds, *, env_factory, lookbacks=(1., 2., 3.),
                  collision_episodes=5, normal_episodes=5, sampling_seed=0,
                  agility=2., noisy_agility=False, progress=None):
    """Reservoir-sample source episodes, retaining all requested collision leads.

    A normal case is a uniformly sampled decision from a successful episode.
    Source rollouts are always completed; quotas never stop scene generation.
    Sampling uses its own generator and cannot perturb the baseline trajectory.
    """
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use a nonempty set of distinct source seeds.')
    if min(collision_episodes, normal_episodes) < 0 or collision_episodes + normal_episodes == 0:
        raise ValueError('Request a positive, nonnegative source-case quota.')
    if (not lookbacks or len(set(lookbacks)) != len(lookbacks) or
            not np.isfinite(lookbacks).all() or min(lookbacks) <= 0):
        raise ValueError('Use distinct positive collision lookbacks.')
    if not np.isfinite(agility) or agility <= 0:
        raise ValueError('Use finite positive attacker agility.')
    selector = np.random.default_rng(sampling_seed)
    reservoirs = {'collision': [], 'normal': []}
    seen = {'collision': 0, 'normal': 0}
    rows = []
    with preserved_random_state():
        for seed in seeds:
            seed_random(seed)
            env = env_factory()
            if env.protocol != 'paper-parameters-v1' or env.defender_num < 2 or env.LearningSide != 'Def':
                raise ValueError('Use the paper-parameters-v1 multi-defender environment.')
            observation, _ = env.reset(agility, noisy_agility)
            leads = [int(round(seconds / env.Action_T)) for seconds in lookbacks]
            if any(step < 1 or not np.isclose(step * env.Action_T, seconds)
                   for step, seconds in zip(leads, lookbacks)):
                raise ValueError('Lookbacks must be positive multiples of the control period.')
            recent = deque(maxlen=max(leads))
            normal = None
            done = steps = 0
            collision = breach = False
            total_reward = 0.
            nearest = minimum_distance(env)
            while not done:
                action = validate_action(policy(observation), env.defender_num)
                case = GateCase(int(seed), steps, 'normal', SimulationSnapshot.capture(env),
                                observation.copy(), action)
                recent.append(case)
                if selector.integers(steps + 1) == 0:
                    normal = case
                observation, rewards, done, _ = env.step(action, 'AdaRes')
                if not np.isfinite(observation).all() or not np.isfinite(rewards).all():
                    raise FloatingPointError('Non-finite source transition.')
                events = env.physical_events()
                collision |= events['collision']
                breach |= events['breach']
                total_reward += float(np.mean(rewards))
                nearest = min(nearest, minimum_distance(env))
                steps += 1
                if steps > math.ceil(env.Total_T / env.Action_T) + 1:
                    raise RuntimeError('Source episode exceeded the task deadline.')
            row = dict(source_seed=int(seed), steps=steps, outcome_code=int(done),
                       success=int(done > 2), collision=int(collision), breach=int(breach),
                       team_return=total_reward, minimum_distance=nearest,
                       trajectory_sha256=trajectory_digest(env))
            rows.append(row)
            kind = 'collision' if collision else ('normal' if done > 2 else None)
            if kind is not None:
                quota = collision_episodes if kind == 'collision' else normal_episodes
                group = [normal]
                if kind == 'collision':
                    group = []
                    starts = {case.source_step: case for case in recent}
                    for lead in leads:
                        if steps - lead in starts:
                            case = starts[steps - lead]
                            case.kind = kind
                            case.lookback_seconds = lead * env.Action_T
                            group.append(case)
                if group:
                    for case in group:
                        case.source_trajectory_sha256 = row['trajectory_sha256']
                    seen[kind] += 1
                    slot = int(selector.integers(seen[kind]))
                    if len(reservoirs[kind]) < quota:
                        reservoirs[kind].append(group)
                    elif slot < quota:
                        reservoirs[kind][slot] = group
            if progress is not None:
                progress(len(rows), len(seeds))
    cases = [case for groups in reservoirs.values() for group in groups for case in group]
    return sorted(cases, key=lambda c: (c.source_seed, c.source_step)), rows


def gate_options(theta, grid=(0., .25, .5, .75, 1.), max_combinations=4096):
    theta, grid = np.asarray(theta), np.asarray(grid, dtype=float)
    if not np.issubdtype(theta.dtype, np.floating):
        theta = theta.astype(float)
    if theta.ndim != 1 or len(theta) < 2 or not np.isfinite(theta).all() or np.any((theta < 0) | (theta > 1)):
        raise ValueError('Expected one bounded baseline gate per defender.')
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or
            np.any((grid < 0) | (grid > 1)) or len(np.unique(grid)) != len(grid)):
        raise ValueError('Grid values must be finite, unique and in [0, 1].')
    if len(grid)**len(theta) + 1 > max_combinations:
        raise ValueError('Grid exceeds the explicit combination limit.')
    # Preserve the baseline action dtype: converting float32 proposals to
    # float64 before affine thrust conversion changes the original controller.
    rows = [theta.copy()]
    seen = {tuple(theta.tolist())}
    for values in product(grid, repeat=len(theta)):
        values = np.asarray(values, dtype=theta.dtype)
        if tuple(values.tolist()) not in seen:
            rows.append(values)
            seen.add(tuple(values.tolist()))
    return np.stack(rows)


def align_environment_noise(env, control_steps):
    """Advance the exogenous stream to an absolute control-step index.

    TADEnv consumes one APF draw then one current per attacker/defender. This
    lets different lookback conditions share disturbances at the same task
    time. It advances only randomness, never physics or the attacker's path.
    """
    for _ in range(control_steps):
        env._apf_force_noise()
        for _ in range(env.defender_num + 1):
            env.generate_random_current()


def run_branch(case, policy, theta, *, future_seed=None, short_horizon=2., align_future_step=False):
    """Override only the first gate; then resume deterministic closed-loop ARBoids."""
    if not np.isfinite(short_horizon) or short_horizon <= 0:
        raise ValueError('Use a finite positive short horizon.')
    theta = np.asarray(theta)
    if theta.shape != case.theta.shape or not np.isfinite(theta).all() or np.any((theta < 0) | (theta > 1)):
        raise ValueError('Expected finite bounded branch gates.')
    with preserved_random_state():
        env = case.snapshot.restore(future_seed=future_seed)
        if align_future_step and future_seed is not None:
            align_environment_noise(env, case.source_step)
        if env._isTerminate():
            raise ValueError('Interventions must start before termination.')
        observation = case.observation.copy()
        action = case.action.copy()
        action[:, 2] = theta
        start = env.Current_T
        done = steps = 0
        collision = breach = False
        total_reward = 0.
        nearest = short_nearest = minimum_distance(env)
        while not done:
            if steps:
                action = validate_action(policy(observation), env.defender_num)
            # Omitting attacker_action recomputes APF from this branch's scene.
            observation, rewards, done, _ = env.step(action, 'AdaRes')
            if not np.isfinite(observation).all() or not np.isfinite(rewards).all():
                raise FloatingPointError('Non-finite continuation transition.')
            events = env.physical_events()
            collision |= events['collision']
            breach |= events['breach']
            total_reward += float(np.mean(rewards))
            distance = minimum_distance(env)
            nearest = min(nearest, distance)
            if env.Current_T - start <= short_horizon + 1e-8:
                short_nearest = min(short_nearest, distance)
            steps += 1
            if steps > math.ceil((env.Total_T - start) / env.Action_T) + 1:
                raise RuntimeError('Continuation exceeded the task deadline.')
        return dict(success=int(done > 2), collision=int(collision), breach=int(breach),
                    capture=int(done == 3), timeout=int(done == 4), outcome_code=int(done),
                    steps=steps, end_time=float(env.Current_T), team_return=total_reward,
                    short_min_distance=short_nearest, episode_min_distance=nearest,
                    trajectory_sha256=trajectory_digest(env))


def predict_options(case, options, *, horizon=2., safe_distance=7.):
    """Hold each fixed candidate mixture over the entire nominal horizon.

    Use the same thrust conversion and assignment precision as TADEnv.step.
    This open-loop forecast is distinct from the one-cycle closed-loop branch.
    """
    env = case.snapshot.environment
    if not np.isfinite(safe_distance) or safe_distance <= env.Collision_R:
        raise ValueError('Prediction safety distance must exceed the collision radius.')
    predictor = InteractionPredictor(PredictionConfig(horizon=horizon))
    states = env.prediction_snapshot()
    thrusts = []
    for theta in options:
        action = env.action_to_thrust(case.action[:, :2].copy())
        for i in range(env.defender_num):
            action[i] = theta[i] * action[i] + (1 - theta[i]) * env.boids_actions[i]
        thrusts.append(action)
    paths = predictor.trajectories(np.tile(states, (len(options), 1)),
                                   np.stack(thrusts).reshape(-1, 2))
    paths = paths.reshape(len(options), env.defender_num, -1, 2)
    # The proposed risk is over k=1..H. Report the present distance separately
    # so a shared t=0 minimum does not conceal a useful separating action.
    a, b = np.triu_indices(env.defender_num, 1)
    distances = np.linalg.norm(paths[:, a, 1:] - paths[:, b, 1:], axis=-1)
    nearest = distances.min(axis=(1, 2))
    risks = np.maximum(0., (safe_distance - nearest) / safe_distance)**2
    return [dict(predicted_min_distance=float(d), predicted_risk=float(r),
                 current_min_distance=minimum_distance(env)) for d, r in zip(nearest, risks)]


METRICS = ('success', 'collision', 'breach', 'team_return', 'short_min_distance')


def mean_metrics(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def select_candidate(averages, options):
    """Screen observed success/breach, then prefer fewer observed collisions.

    This is a fixed development-sample rule, not a noninferiority test. The
    baseline is always eligible; a separate future-seed set tests the selection.
    """
    reference = averages[0]
    eligible = [i for i, row in enumerate(averages)
                if row['success'] >= reference['success'] and row['breach'] <= reference['breach']]
    def score(i):
        row = averages[i]
        return (-row['collision'], row['success'], -row['breach'], row['team_return'],
                row['short_min_distance'], -float(np.linalg.norm(options[i] - options[0])), -i)
    return max(eligible, key=score)


def diagnose_case(case, policy, search_seeds, validation_seeds, *, grid=(0., .25, .5, .75, 1.),
                  horizon=2., safe_distance=7., max_combinations=4096, progress=None,
                  align_future_step=False):
    """Exhaustive search followed by paired independent-future validation."""
    search_seeds, validation_seeds = list(search_seeds), list(validation_seeds)
    combined = search_seeds + validation_seeds
    if (not search_seeds or not validation_seeds or len(set(combined)) != len(combined) or
            any(not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32 for seed in combined)):
        raise ValueError('Search and validation need nonempty, distinct, disjoint valid seeds.')
    if not np.array_equal(policy(case.observation), case.action):
        raise ValueError('Frozen checkpoint does not reproduce the saved candidates/gate.')
    original = run_branch(case, policy, case.theta, short_horizon=horizon)
    if (case.source_trajectory_sha256 is not None and
            original['trajectory_sha256'] != case.source_trajectory_sha256):
        raise ValueError('Snapshot continuation did not reproduce the source trajectory.')
    options = gate_options(case.theta, grid, max_combinations)
    predictions = predict_options(case, options, horizon=horizon, safe_distance=safe_distance)
    records = []
    by_candidate = [[] for _ in options]

    def record(phase, seed, index, role, result, reference):
        row = dict(case_id=case.case_id, source_seed=case.source_seed, source_step=case.source_step,
                   kind=case.kind, phase=phase, continuation_seed=int(seed), candidate_id=index,
                   role=role, **predictions[index], **result)
        row.update({f'theta_{i}': float(theta) for i, theta in enumerate(options[index])})
        row.update({key + '_delta': result[key] - reference[key] for key in METRICS})
        records.append(row)

    for seed in search_seeds:
        reference = run_branch(case, policy, options[0], future_seed=seed, short_horizon=horizon,
                               align_future_step=align_future_step)
        for index, theta in enumerate(options):
            result = reference if index == 0 else run_branch(
                case, policy, theta, future_seed=seed, short_horizon=horizon, align_future_step=align_future_step)
            by_candidate[index].append(result)
            record('search', seed, index, 'reference' if index == 0 else 'candidate', result, reference)
            if progress is not None:
                progress(len(records), len(options) * len(search_seeds) + 2 * len(validation_seeds))
    averages = [mean_metrics(rows) for rows in by_candidate]
    selected = select_candidate(averages, options)
    references, selections = [], []
    for seed in validation_seeds:
        reference = run_branch(case, policy, options[0], future_seed=seed, short_horizon=horizon,
                               align_future_step=align_future_step)
        result = reference if selected == 0 else run_branch(
            case, policy, options[selected], future_seed=seed, short_horizon=horizon,
            align_future_step=align_future_step)
        references.append(reference)
        selections.append(result)
        record('validation', seed, 0, 'reference', reference, reference)
        record('validation', seed, selected, 'selected', result, reference)
        if progress is not None:
            progress(len(records), len(options) * len(search_seeds) + 2 * len(validation_seeds))
    validation_reference, validation_selected = mean_metrics(references), mean_metrics(selections)
    summary = dict(case_id=case.case_id, source_seed=case.source_seed, source_step=case.source_step,
                   kind=case.kind, lookback_seconds=case.lookback_seconds,
                   start_time=float(case.snapshot.environment.Current_T),
                   options=len(options), selected_candidate_id=selected,
                   source_replay_verified=case.source_trajectory_sha256 is not None,
                   search_repeats=len(search_seeds), validation_repeats=len(validation_seeds),
                   prediction_risk_reduction=predictions[0]['predicted_risk'] - predictions[selected]['predicted_risk'],
                   action_span=float(np.linalg.norm(case.snapshot.environment.action_to_thrust(
                       case.action[:, :2]) - case.snapshot.environment.boids_actions)))
    for phase, reference, selected_metrics in (
            ('search', averages[0], averages[selected]),
            ('validation', validation_reference, validation_selected)):
        for key in METRICS:
            summary[f'{phase}_reference_{key}'] = reference[key]
            summary[f'{phase}_selected_{key}'] = selected_metrics[key]
            summary[f'{phase}_{key}_delta'] = selected_metrics[key] - reference[key]
        summary[f'{phase}_observed_improvement'] = bool(
            selected_metrics['collision'] < reference['collision'] and
            selected_metrics['success'] >= reference['success'] and
            selected_metrics['breach'] <= reference['breach'])
    summary.update({f'selected_theta_{i}': float(theta) for i, theta in enumerate(options[selected])})
    return summary, records


def summarize_cases(cases):
    """Keep cases nested within source episodes; do not claim independent states."""
    strata = {}
    for kind in sorted({row['kind'] for row in cases}):
        selected = [row for row in cases if row['kind'] == kind]
        sources = defaultdict(list)
        for row in selected:
            sources[row['source_seed']].append(row)
        strata[kind] = dict(
            cases=len(selected), source_episodes=len(sources),
            search_observed_improvement_cases=sum(row['search_observed_improvement'] for row in selected),
            validation_observed_improvement_cases=sum(row['validation_observed_improvement'] for row in selected),
            source_mean_validation_deltas={key: float(np.mean([
                np.mean([row[f'validation_{key}_delta'] for row in rows]) for rows in sources.values()
            ])) for key in METRICS})
    return strata
