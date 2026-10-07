"""Matched forecasts and receding gate interventions on frozen ARBoids states."""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from envs.TADgame import TADEnv
from envs.snapshot import preserved_random_state
from gate_diagnostics import (align_environment_noise, gate_options, minimum_distance,
                              trajectory_digest, validate_action)
from policy.interaction_prediction import InteractionPredictor, PredictionConfig


class NominalEnv(TADEnv):
    def generate_random_current(self):
        return np.zeros(3)

    def _apf_force_noise(self):
        return np.zeros(2)


def nominal_environment(snapshot):
    env = NominalEnv.__new__(NominalEnv)
    env.__dict__ = deepcopy(snapshot.environment.__dict__)
    # Match the predictor's measured-ground-velocity initialization. No future
    # disturbance or source episode outcome is supplied to this nominal model.
    for boat in [*env.defender_list, env.attacker]:
        boat.velocity_r = boat.motion_state()[3:].copy()
        boat.current_velocity = np.zeros(3)
        boat.update_velocity(np.zeros(3))
    return env


def mixture_thrusts(env, action, options):
    result = []
    for theta in options:
        thrust = env.action_to_thrust(action[:, :2].copy())
        for i in range(env.defender_num):
            thrust[i] = theta[i] * thrust[i] + (1 - theta[i]) * env.boids_actions[i]
        result.append(thrust)
    return np.stack(result)


def held_forecasts(env, action, options, horizon):
    predictor = InteractionPredictor(PredictionConfig(horizon=horizon))
    thrusts = mixture_thrusts(env, action, options)
    paths = predictor.trajectories(np.tile(env.prediction_snapshot(), (len(options), 1)),
                                   thrusts.reshape(-1, 2)).reshape(len(options), env.defender_num, -1, 2)
    a, b = np.triu_indices(env.defender_num, 1)
    distances = np.linalg.norm(paths[:, a] - paths[:, b], axis=-1)
    return paths, distances, thrusts


def choose_rolling_gate(env, action, *, horizon=2., safe_distance=7., change_penalty=.02,
                        grid=(0., .25, .5, .75, 1.)):
    if safe_distance <= env.Collision_R or not np.isfinite([safe_distance, change_penalty]).all() or change_penalty < 0:
        raise ValueError('Use a warning distance above collision radius and a nonnegative finite penalty.')
    options = gate_options(action[:, 2], grid)
    _, distances, thrusts = held_forecasts(env, action, options, horizon)
    minimum = distances[:, :, 1:].min(axis=(1, 2))
    risk = np.maximum(0., (safe_distance - minimum) / safe_distance)**2
    deviation = np.mean(((thrusts - thrusts[0]) / (env.max_thrust - env.min_thrust))**2, axis=(1, 2))
    # No forecast warning -> exact baseline, without gratuitous gate changes.
    index = 0 if minimum[0] >= safe_distance else int(np.argmin(risk + change_penalty * deviation))
    return options[index], dict(reference_predicted_min=float(minimum[0]),
                               chosen_predicted_min=float(minimum[index]),
                               candidate_id=index, warning=bool(minimum[0] < safe_distance))


@dataclass
class ControlTrace:
    summary: dict
    positions: np.ndarray  # [time, defender, xy], including the start
    gates: np.ndarray
    thrusts: np.ndarray


def run_controlled(case, policy, *, mode='baseline', theta=None, active_steps=1,
                   future_seed=None, horizon=2., safe_distance=7., change_penalty=.02,
                   grid=(0., .25, .5, .75, 1.), nominal=False, stop_after=None):
    """Run a full episode, or a labelled truncated prefix for model audits.

    pulse: one gate impulse; hold: fixed initial physical thrust; constant:
    fixed gates applied to fresh candidates; rolling: reselect every cycle;
    rolling_once: reselect only on the first predicted warning.
    A finite active_steps budget is measured from the captured state, not from
    the first alarm. None enables the rule throughout the remaining episode.
    """
    if mode not in ('baseline', 'pulse', 'hold', 'constant', 'rolling', 'rolling_once'):
        raise ValueError('Unknown intervention mode.')
    if active_steps is not None and (not isinstance(active_steps, int) or active_steps < 1):
        raise ValueError('Use positive active steps or None for the remaining episode.')
    if stop_after is not None and (not isinstance(stop_after, int) or stop_after < 1):
        raise ValueError('Use a positive prefix length.')
    theta = case.theta if theta is None else np.asarray(theta, dtype=case.action.dtype)
    if theta.shape != case.theta.shape or not np.isfinite(theta).all() or np.any((theta < 0) | (theta > 1)):
        raise ValueError('Expected bounded finite gates.')
    with preserved_random_state():
        env = nominal_environment(case.snapshot) if nominal else case.snapshot.restore(future_seed=future_seed)
        if not nominal and future_seed is not None:
            align_environment_noise(env, case.source_step)
        if env._isTerminate():
            raise ValueError('Cannot intervene after termination.')
        fixed_thrust = mixture_thrusts(env, case.action, theta[None])[0]
        observation = case.observation.copy()
        start = env.Current_T
        positions = [np.array([boat.pos for boat in env.defender_list]).copy()]
        gates, thrusts = [], []
        steps = done = altered = warnings = 0
        collision = breach = False
        first_warning = first_change = None
        task_return = 0.
        nearest = short_nearest = minimum_distance(env)
        while not done and (stop_after is None or steps < stop_after):
            action = case.action.copy() if steps == 0 else validate_action(policy(observation), env.defender_num)
            baseline = action.copy()
            enabled = active_steps is None or steps < active_steps
            override = None
            if mode == 'pulse' and steps == 0:
                action[:, 2] = theta
            elif mode == 'constant' and enabled:
                action[:, 2] = theta
            elif mode == 'hold' and enabled:
                override = fixed_thrust
                action[:, 2] = theta  # descriptive; physical override is authoritative
            elif (mode in ('rolling', 'rolling_once') and enabled
                  and (mode == 'rolling' or warnings == 0)):
                action[:, 2], forecast = choose_rolling_gate(
                    env, action, horizon=horizon, safe_distance=safe_distance,
                    change_penalty=change_penalty, grid=grid)
                if forecast['warning']:
                    warnings += 1
                    if first_warning is None:
                        first_warning = float(env.Current_T)
            if not np.array_equal(action[:, 2], baseline[:, 2]) or override is not None:
                altered += 1
                if first_change is None:
                    first_change = float(env.Current_T)
            observation, rewards, done, _ = env.step(action, 'AdaRes', defender_thrust=override)
            if not np.isfinite(observation).all() or not np.isfinite(rewards).all():
                raise FloatingPointError('Non-finite follow-up transition.')
            positions.append(np.array([boat.pos for boat in env.defender_list]).copy())
            gates.append(action[:, 2].copy())
            thrusts.append(np.array([[boat.left_thrust, boat.right_thrust] for boat in env.defender_list]))
            events = env.physical_events()
            collision |= events['collision']
            breach |= events['breach']
            task_return += float(np.mean(rewards))
            nearest = min(nearest, minimum_distance(env))
            if env.Current_T - start <= horizon + 1e-8:
                short_nearest = min(short_nearest, minimum_distance(env))
            steps += 1
            if steps > math.ceil((env.Total_T - start) / env.Action_T) + 1:
                raise RuntimeError('Follow-up exceeded task deadline.')
        summary = dict(success=int(done > 2), collision=int(collision), breach=int(breach),
                       capture=int(done == 3), timeout=int(done == 4), outcome_code=int(done),
                       terminated=bool(done), steps=steps, end_time=float(env.Current_T),
                       team_return=task_return, short_min_distance=short_nearest,
                       episode_min_distance=nearest, altered_steps=altered, warning_steps=warnings,
                       first_warning_time=first_warning, first_change_time=first_change,
                       trajectory_sha256=trajectory_digest(env))
        return ControlTrace(summary, np.stack(positions), np.asarray(gates), np.asarray(thrusts))


def prefix_metrics(predicted_path, trace, control_dt=.2, prediction_dt=.05, collision_radius=5.):
    """Compare the SAME control endpoints, stopping at actual termination."""
    stride = int(round(control_dt / prediction_dt))
    predicted = predicted_path[:, ::stride].transpose(1, 0, 2)[:len(trace.positions)]
    if predicted.shape != trace.positions.shape:
        raise ValueError('Prediction does not cover the observed prefix.')
    a, b = np.triu_indices(predicted.shape[1], 1)
    pd = np.linalg.norm(predicted[:, a] - predicted[:, b], axis=-1).min(axis=1)
    rd = np.linalg.norm(trace.positions[:, a] - trace.positions[:, b], axis=-1).min(axis=1)
    def first_hit(distances):
        hits = np.flatnonzero(distances <= collision_radius)
        return None if not len(hits) else float(hits[0] * control_dt)
    return dict(common_prefix_seconds=(len(pd) - 1) * control_dt,
                position_rmse=float(np.sqrt(np.mean(np.sum((predicted - trace.positions)**2, axis=-1)))),
                maximum_position_error=float(np.linalg.norm(predicted - trace.positions, axis=-1).max()),
                predicted_prefix_min=float(pd.min()), actual_prefix_min=float(rd.min()),
                predicted_first_collision=first_hit(pd), actual_first_collision=first_hit(rd))


def audit_case(case, policy, future_seeds, horizons=(1., 2., 3.), grid=(0., .25, .5, .75, 1.)):
    options = gate_options(case.theta, grid)
    max_horizon = max(horizons)
    env = case.snapshot.environment
    paths, distances, _ = held_forecasts(env, case.action, options, max_horizon)
    audits = []
    for horizon in horizons:
        limit = int(round(horizon / .05)) + 1
        minimum = distances[:, :, 1:limit].min(axis=(1, 2))
        best = int(np.argmax(minimum))
        for label, index in (('original', 0), ('best_held_prediction', best)):
            steps = int(round(horizon / env.Action_T))
            nominal = run_controlled(case, policy, mode='hold', theta=options[index],
                                     active_steps=steps, nominal=True, stop_after=steps)
            common = prefix_metrics(paths[index], nominal, collision_radius=env.Collision_R)
            audits.append(dict(case_id=case.case_id, source_seed=case.source_seed, kind=case.kind,
                lookback_seconds=case.lookback_seconds, horizon=horizon, label=label, candidate_id=index,
                execution='matched_zero_current', future_seed=None, **common))
            for seed in future_seeds:
                actual = run_controlled(case, policy, mode='hold', theta=options[index],
                                        active_steps=steps, future_seed=seed, stop_after=steps)
                audits.append(dict(case_id=case.case_id, source_seed=case.source_seed, kind=case.kind,
                    lookback_seconds=case.lookback_seconds, horizon=horizon, label=label, candidate_id=index,
                    execution='sampled_currents', future_seed=seed,
                    **prefix_metrics(paths[index], actual, collision_radius=env.Collision_R)))
    # Aligned pulse forecasts use the original control schedule: one altered
    # cycle followed by deterministic baseline feedback, in a noise-free model.
    forecasts = []
    steps = int(round(2. / env.Action_T))
    for index, theta in enumerate(options):
        trace = run_controlled(case, policy, mode='pulse', theta=theta, nominal=True,
                               horizon=2., stop_after=steps)
        forecasts.append(dict(case_id=case.case_id, source_seed=case.source_seed, kind=case.kind,
            lookback_seconds=case.lookback_seconds, candidate_id=index,
            held_prediction_min=float(distances[index, :, 1:int(round(2. / .05))+1].min()),
            pulse_prediction_min=trace.summary['short_min_distance'],
            pulse_prediction_collision=trace.summary['collision'],
            nominal_prefix_steps=trace.summary['steps'], nominal_terminated=trace.summary['terminated']))
    return audits, forecasts
