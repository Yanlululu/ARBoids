"""Development diagnostics; original source, controllers and weights stay frozen."""
import study_runtime  # Select the same isolated numerical runtime first.
from copy import deepcopy
import hashlib
import itertools
import json

import numpy as np
from scipy.stats import spearmanr

from source_arboids import numerical_source, snapshot, actor_thrust, source_mixture
from gate_study_control import GateController, NominalModel, joint_indices
from evaluate_gate_contribution import (setup_scene, controller, minimum_distance,
                                       digest_transition)


HORIZON_STEPS = (1, 3, 5, 10)
RECOVERY_METHODS = ('original', 'reactive_joint', 'predictive_joint', 'fine_joint',
                    'cbf', 'projected_cbf', 'actual_shortlist')


def stable_number(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:15], 16)


def freeze(env, observation):
    state = dict(env={k: v for k, v in env.__dict__.items()
                      if k not in ('defender_list', 'attacker')},
                 defenders=[b.__dict__ for b in env.defender_list],
                 attacker=env.attacker.__dict__, observation=observation,
                 rng=np.random.get_state())
    return deepcopy(state)


def restore(state):
    state = deepcopy(state)
    env = numerical_source().TADEnv(defender_num=len(state['defenders']))
    env.__dict__.update(state['env'])
    for boat, values in zip(env.defender_list, state['defenders']):
        boat.__dict__.update(values)
    env.attacker.__dict__.update(state['attacker'])
    np.random.set_state(state['rng'])
    return env, state['observation']


def geometry(states):
    i, j = np.triu_indices(len(states), 1)
    p = states[i, :2]-states[j, :2]
    v = states[i, 3:5]-states[j, 3:5]
    pv, vv = np.sum(p*v, axis=-1), np.sum(v*v, axis=-1)
    t = -pv/np.maximum(vv, 1e-15)
    d = np.linalg.norm(p, axis=-1)
    cpa = np.linalg.norm(p+t[:, None]*v, axis=-1)
    approaching = (pv < 0.) & (t > 0.) & (t <= 2.) & (cpa < 7.)
    return dict(minimum_distance=float(d.min()), close_pairs=int((d < 9.).sum()),
                crossing_pairs=int(approaching.sum()),
                density='dense' if (d < 9.).sum() >= 2 else 'sparse')


def capture_scene(job, policy):
    scene, method = job['scene'], job['method']
    env, observation = setup_scene(scene)
    digest, saved, found = hashlib.sha256(), [], set()
    target = {}
    if job['kind'] == 'failure':
        for lead in (1., 2., 3.):
            at = max(0, int(round((float(job['expected']['duration'])-lead)/.2)))
            target.setdefault(at, []).append(lead)
    steps = done = 0
    while not done:
        geo = geometry(snapshot(env))
        categories = []
        if job['kind'] == 'failure' and steps in target:
            categories = ['failure']
        if job['kind'] == 'natural':
            if steps >= 5 and geo['minimum_distance'] >= 9. and not geo['crossing_pairs']:
                categories.append('ordinary')
            if geo['minimum_distance'] > 5. and geo['crossing_pairs']:
                categories.append('approaching')
        for category in categories:
            if category in found and job['kind'] == 'natural':
                continue
            found.add(category)
            meta = dict(**scene, suite=job['suite'], category=category, parent_method=method,
                        control_step=steps, time=float(env.Current_T),
                        source_outcome=int(job['expected']['outcome']), **geo)
            meta['case_id'] = f"{job['suite']}-{scene['scene_seed']}-{category}-s{steps}"
            if category == 'failure':
                meta['requested_leads'] = target[steps]
                meta['actual_lead'] = float(job['expected']['duration'])-env.Current_T
            saved.append(dict(meta=meta, state=freeze(env, observation)))
        action = policy(observation)
        if method != 'original':
            action, _, _ = controller(method, env.defender_num).control(
                snapshot(env), action, env.boids_actions)
        observation, reward, done, _ = env.step(action, 'AdaRes')
        digest_transition(digest, action, env, observation, reward, done)
        steps += 1
    actual = digest.hexdigest()
    if actual != job['expected']['trajectory_sha256']:
        raise AssertionError(f"Original recorded trajectory changed: {scene}, {method}")
    audit = dict(**scene, suite=job['suite'], method=method, kind=job['kind'],
                 trajectory_sha256=actual, original_equivalent=True,
                 saved_states=len(saved), missing_categories=';'.join(
                     sorted({'ordinary', 'approaching'}-found)) if job['kind']=='natural' else '')
    return saved, audit


class FineGate(GateController):
    def __init__(self):
        super().__init__('predictive_joint')

    def candidates(self, action, boids):
        theta = np.column_stack((action[:, 2], np.tile(np.linspace(0., 1., 9), (len(action), 1))))
        theta = theta.astype(action.dtype)
        learned = actor_thrust(action)
        thrusts = np.empty((len(action), 10, 2), dtype=learned.dtype)
        for i in range(len(action)):
            for k in range(10):
                thrusts[i, k] = theta[i, k]*learned[i]+(1-theta[i, k])*boids[i]
        return theta, thrusts


_FINE = None


def project_to_segments(physical, action, boids):
    base = np.asarray(boids, dtype=float)
    direction = actor_thrust(action).astype(float)-base
    denom = np.sum(direction*direction, axis=-1)
    theta = np.divide(np.sum((physical-base)*direction, axis=-1), denom,
                      out=np.zeros(len(action)), where=denom > 1e-20)
    theta = np.clip(theta, 0., 1.)
    projected = base+theta[:, None]*direction
    residual = np.linalg.norm(physical-projected, axis=-1)
    return theta, projected, residual


def control_step(env, observation, policy, method, first_theta=None):
    global _FINE
    action = policy(observation)
    nominal = source_mixture(action, env.boids_actions).astype(float)
    extra = dict(projection_rms=0., projection_max=0., cbf_constraint_violation=0.)
    if first_theta is not None:
        action = action.copy()
        action[:, 2] = first_theta
        physical, mode, passed = source_mixture(action, env.boids_actions), 'AdaRes', action
    elif method == 'original':
        physical, mode, passed = nominal, 'AdaRes', action
    elif method in ('cbf', 'projected_cbf'):
        _, physical, info = controller('cbf', env.defender_num).control(snapshot(env), action, env.boids_actions)
        theta, _, residual = project_to_segments(physical, action, env.boids_actions)
        extra.update(projection_rms=float(np.sqrt(np.mean(residual**2))),
                     projection_max=float(residual.max()),
                     cbf_constraint_violation=info['cbf_constraint_violation'])
        if method == 'projected_cbf':
            action[:, 2] = theta
            physical, mode, passed = source_mixture(action, env.boids_actions), 'AdaRes', action
        else:
            mode, passed = 'RL', env.thrust_to_action(physical)
    else:
        if method == 'fine_joint':
            if _FINE is None:
                _FINE = FineGate()
            rule = _FINE
        else:
            rule = controller('predictive_joint' if method == 'actual_shortlist' else method,
                              env.defender_num)
        action, physical, _ = rule.control(snapshot(env), action, env.boids_actions)
        mode, passed = 'AdaRes', action
    # Costs reflect saturated forces sent to the original plant.
    applied = np.clip(physical.astype(float), env.min_thrust, env.max_thrust)
    observation, reward, done, _ = env.step(passed, mode)
    if not np.isfinite(observation).all() or not np.isfinite(reward).all():
        raise FloatingPointError('Non-finite source transition in recovery.')
    return observation, reward, done, applied, nominal, extra


def branch(case, policy, method='predictive_joint', first_theta=None, stop_steps=None, noise_seed=None):
    env, observation = restore(case['state'])
    if noise_seed is not None:
        np.random.seed(noise_seed)
        # Align alternative futures by absolute control time across lead windows.
        for _ in range(case['meta']['control_step']):
            np.random.normal(0., .3, 2)
            for _ in range(env.defender_num+1):
                env.generate_random_current()
    done = steps = 0
    distance = minimum_distance(env)
    min_distance, task_return = distance, 0.
    force_cost = variation = deviation = projection_sum = projection_max = 0.
    previous = None
    trace = []
    while not done and (stop_steps is None or steps < stop_steps):
        observation, reward, done, physical, nominal, info = control_step(
            env, observation, policy, method, first_theta if steps == 0 else None)
        distance = minimum_distance(env)
        min_distance = min(min_distance, distance)
        task_return += float(np.mean(reward))
        force_cost += .2*float(np.mean((physical/1000.)**2))
        deviation += .2*float(np.mean(((physical-nominal)/1500.)**2))
        if previous is not None:
            variation += float(np.mean(np.abs(physical-previous)/1500.))
        previous = physical
        projection_sum += info['projection_rms']**2
        projection_max = max(projection_max, info['projection_max'])
        trace.append(dict(distance=distance, positions=snapshot(env), done=int(done),
                          team_return=task_return))
        steps += 1
        if steps > 401:
            raise RuntimeError('Source task exceeded 80 seconds.')
    return dict(method=method, outcome=int(done), success=int(done>2), collision=int(done==2),
                source_loss=int(done==1), capture=int(done==3), steps=steps,
                duration=steps*.2, minimum_distance=min_distance, team_return=task_return,
                force_square_integral=force_cost, force_square_time_mean=force_cost/(steps*.2),
                thrust_variation=variation, thrust_variation_per_second=variation/(steps*.2),
                deviation_integral=deviation, projection_rms=float(np.sqrt(projection_sum/steps)),
                projection_max=projection_max), trace


def state_paths(states, thrusts, count=10):
    model = NominalModel()
    state = np.asarray(states, dtype=float).copy()
    thrusts = np.clip(thrusts, model.model.min_thrust, model.model.max_thrust)
    result = np.empty((len(state), count+1, 6))
    result[:, 0] = state
    for step in range(count):
        for _ in range(4):
            state[:, :2] += state[:, 3:5]*.05
            state[:, 2] = (state[:, 2]+state[:, 5]*.05) % (2*np.pi)
            state[:, 3:] += model.acceleration(state, thrusts)*.05
        result[:, step+1] = state
    return result


def held_paths(case, thrusts, nominal=False):
    """Independent original WAMV calls, common source-generated current stream."""
    env, _ = restore(case['state'])
    n, k, _ = thrusts.shape
    initial = snapshot(env)
    boats = [[deepcopy(env.defender_list[i]) for _ in range(k)] for i in range(n)]
    if nominal:
        for i in range(n):
            for boat in boats[i]:
                boat.velocity_r = initial[i, 3:].copy()
                boat.update_velocity(np.zeros(3))
    output = np.empty((n, k, 11, 6))
    output[:, :, 0] = initial[:, None, :]
    for step in range(10):
        # Original step always draws APF noise, then attacker and defender currents.
        np.random.normal(0., .3, 2)
        currents = [env.generate_random_current() for _ in range(n+1)]
        for i in range(n):
            for c, boat in enumerate(boats[i]):
                boat.step(thrusts[i, c], np.zeros(3) if nominal else currents[i+1])
                output[i, c, step+1] = [boat.x, boat.y, boat.theta, *boat.velocity]
    return output


def option_distances(paths, indices, steps, include_initial=False):
    n, k, _, _ = paths.shape
    result = np.full(len(indices), np.inf)
    start = 0 if include_initial else 1
    for i, j in zip(*np.triu_indices(n, 1)):
        pair = np.linalg.norm(paths[i, :, None, start:steps+1, :2]
                              -paths[j, None, :, start:steps+1, :2], axis=-1).min(-1)
        result = np.minimum(result, pair[indices[:, i], indices[:, j]])
    return result


def rank_correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.ptp(a) < 1e-10 or np.ptp(b) < 1e-10:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def ranking_metrics(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    delta = predicted-actual
    row = dict(candidates=len(actual), dmin_mae=float(np.abs(delta).mean()),
               dmin_bias=float(delta.mean()), dmin_max_error=float(np.abs(delta).max()),
               distance_rank_spearman=rank_correlation(predicted, actual),
               risk_rank_spearman=rank_correlation(np.maximum(0., 1.-predicted/7.)**2,
                                                  np.maximum(0., 1.-actual/7.)**2))
    for threshold in (5, 7):
        danger, safe = actual < threshold, predicted >= threshold
        missed = int((danger & safe).sum())
        row.update({f'actual_danger_{threshold}': int(danger.sum()),
                    f'predicted_safe_{threshold}': int(safe.sum()),
                    f'missed_{threshold}': missed,
                    f'miss_rate_{threshold}': missed/int(danger.sum()) if danger.any() else None,
                    f'false_safe_rate_{threshold}': missed/int(safe.sum()) if safe.any() else None})
    return row


def chosen_indices(theta, choices):
    return tuple(int(np.flatnonzero(choices[i] == theta[i])[0]) for i in range(len(theta)))


def shortlist(case, states, action, boids, theta, options, budget=96):
    n = len(action)
    wanted = [tuple([0]*n)]
    selected = {}
    for steps in HORIZON_STEPS:
        gate, _ = GateController('predictive_joint', horizon=.2*steps).select(states, action, boids)
        selected[steps] = chosen_indices(gate, theta)
        wanted.append(selected[steps])
    gate, _ = GateController('reactive_joint').select(states, action, boids)
    wanted.append(chosen_indices(gate, theta))
    wanted.extend(itertools.product((1, 5), repeat=n))
    for i in range(n):
        for c in range(1, 6):
            item = [0]*n
            item[i] = c
            wanted.append(tuple(item))
    lookup = {tuple(row): i for i, row in enumerate(options)}
    indices = list(dict.fromkeys(lookup[x] for x in wanted))
    rng = np.random.default_rng(stable_number(case['meta']['case_id']) % (2**32))
    remaining = np.setdiff1d(np.arange(len(options)), indices)
    indices.extend(rng.permutation(remaining)[:max(0, min(budget, len(options))-len(indices))].tolist())
    return np.asarray(indices), {h: lookup[x] for h, x in selected.items()}


def oracle_key(result, initial_deviation):
    return (result['collision']==0, result['source_loss']==0, result['capture'],
            result['team_return'], result['minimum_distance'], -initial_deviation)


def diagnose_case(case, policy):
    env, observation = restore(case['state'])
    states, action, boids = snapshot(env), policy(observation), env.boids_actions.copy()
    theta, thrusts = GateController().candidates(action, boids)
    n, k, _ = thrusts.shape
    options = joint_indices(n, k)
    predicted = state_paths(np.repeat(states, k, axis=0), thrusts.reshape(-1, 2)).reshape(n, k, 11, 6)
    actual = held_paths(case, thrusts)
    nominal = held_paths(case, thrusts, nominal=True)
    position_error = float(np.max(np.abs(predicted[..., :2]-nominal[..., :2])))
    yaw_error = float(np.max(np.abs(np.angle(np.exp(1j*(predicted[..., 2]-nominal[..., 2]))))))
    double_paths = state_paths(np.repeat(states, k, axis=0), thrusts.astype(float).reshape(-1, 2)).reshape(n, k, 11, 6)
    double_error = float(np.max(np.abs(double_paths[..., :3]-nominal[..., :3])))
    # Frozen deployed model adds float32 thrusts; source converts each force to
    # float64 before addition. Check that roundoff separately from integration.
    if max(position_error, yaw_error) > 1e-6 or double_error > 1e-8:
        raise AssertionError(f'Nominal source-integrator mismatch {position_error}, {yaw_error}')
    audit = dict(case_id=case['meta']['case_id'], position_max_error=position_error,
                 heading_max_error=yaw_error, float64_integration_max_error=double_error, passed=True)
    candidate_ids, selected = shortlist(case, states, action, boids, theta, options)
    pulses, traces, deviations = [], [], []
    for candidate in candidate_ids:
        gate = theta[np.arange(n), options[candidate]]
        result, trace = branch(case, policy, first_theta=gate, stop_steps=10)
        pulses.append(result)
        traces.append(trace)
        force = thrusts[np.arange(n), options[candidate]]
        deviations.append(float(np.mean(((force-thrusts[:, 0])/1500.)**2)))
    choice = max(range(len(pulses)), key=lambda i: oracle_key(pulses[i], deviations[i]))
    oracle_theta = theta[np.arange(n), options[candidate_ids[choice]]]
    metrics, selections, candidates = [], [], []
    for steps in HORIZON_STEPS:
        predicted_distance = option_distances(predicted, options, steps)
        actual_distance = option_distances(actual, options, steps)
        base = dict(**case['meta'], horizon_steps=steps, horizon=.2*steps)
        row = dict(**base, execution='held', **ranking_metrics(predicted_distance, actual_distance))
        row['position_rmse'] = float(np.sqrt(np.mean(np.sum((predicted[:, :, 1:steps+1, :2]
                                                           -actual[:, :, 1:steps+1, :2])**2, axis=-1))))
        row['heading_mae'] = float(np.mean(np.abs(np.angle(np.exp(1j*(predicted[:, :, 1:steps+1, 2]
                                                                           -actual[:, :, 1:steps+1, 2]))))))
        metrics.append(row)
        complete = np.array([len(t) >= steps for t in traces])
        distances = np.array([min(s['distance'] for s in t[:steps]) for t in traces])
        # A terminal at this exact step is included; earlier termination is censored.
        if complete.any():
            metrics.append(dict(**base, execution='replan_complete', sampled_candidates=len(traces),
                                complete_fraction=float(complete.mean()),
                                **ranking_metrics(predicted_distance[candidate_ids[complete]], distances[complete])))
        common = min(steps, min(len(t) for t in traces))
        common_actual = np.array([min(s['distance'] for s in t[:common]) for t in traces])
        common_predicted = option_distances(predicted, options[candidate_ids], common)
        metrics.append(dict(**base, execution='replan_common_prefix', effective_steps=common,
                            **ranking_metrics(common_predicted, common_actual)))
        selected_id = selected[steps]
        selected_position = int(np.flatnonzero(candidate_ids == selected_id)[0])
        selections.append(dict(**base, selected_candidate=selected_id,
                               predicted_distance=float(predicted_distance[selected_id]),
                               held_distance=float(actual_distance[selected_id]),
                               replan_distance=float(distances[selected_position]),
                               held_clearance_regret=float(actual_distance.max()-actual_distance[selected_id]),
                               common_clearance_regret=float(common_actual.max()-common_actual[selected_position]),
                               replan_steps=min(steps, len(traces[selected_position])),
                               replan_outcome=traces[selected_position][min(steps,len(traces[selected_position]))-1]['done']))
        for i, candidate_id in enumerate(candidate_ids):
            end = traces[i][min(steps, len(traces[i]))-1]
            candidates.append(dict(case_id=case['meta']['case_id'], horizon_steps=steps,
                                   candidate_id=int(candidate_id), theta=json.dumps(theta[np.arange(n),options[candidate_id]].tolist()),
                                   predicted_dmin=float(predicted_distance[candidate_id]),
                                   held_dmin=float(actual_distance[candidate_id]), replan_dmin=float(distances[i]),
                                   complete=int(complete[i]), outcome=end['done'], team_return=end['team_return'],
                                   selected=int(candidate_id==selected_id), actual_shortlist_selected=int(i==choice)))
    selected_action = dict(case_id=case['meta']['case_id'], theta=oracle_theta.tolist(),
                           candidate_id=int(candidate_ids[choice]), shortlist_count=len(candidate_ids),
                           full_count=len(options), diagnostic_result=pulses[choice])
    return dict(audit=audit, metrics=metrics, selections=selections, candidates=candidates,
                oracle=selected_action)


def recover_case(case, oracle, policy):
    rows = []
    noise_seeds = [None]+[stable_number(f"recovery-{case['meta']['scene_seed']}-{r}") % (2**32)
                          for r in range(3)]
    for repeat, seed in enumerate(noise_seeds):
        # The same chosen oracle action is used for all three unseen future streams.
        for method in RECOVERY_METHODS:
            result, _ = branch(case, policy, method,
                               first_theta=oracle['theta'] if method=='actual_shortlist' else None,
                               noise_seed=seed)
            rows.append({**case['meta'], 'initial_minimum_distance': case['meta']['minimum_distance'],
                         'future': 'recorded' if seed is None else 'unseen', 'repeat': repeat,
                         'future_noise_seed': seed, **result})
    return rows
