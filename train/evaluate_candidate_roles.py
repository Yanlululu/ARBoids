"""Matched task comparisons for the frozen interception-candidate controller."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from functools import lru_cache
from itertools import combinations
import multiprocessing as mp
import os
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from interaction_evaluation import OriginalPolicy, long_reference_episode
from interaction_rollout import CandidateRolePolicy, DeploymentPolicy
from policy.role_residual import RoleResidualPolicy, RESIDUAL_SETTINGS
from train_interaction import atomic_json, episode


class AssignmentControl(CandidateRolePolicy):
    """Keep all candidate controls fixed and change only their assignment."""
    def __init__(self, assignment):
        super().__init__()
        self.assignment = assignment
        self.fixed_choice = None

    def choose_action(self, packet, deterministic=True):
        candidates, cost = self.candidates(packet)
        n = len(cost)
        if self.assignment == 'distance':
            state, obs = np.asarray(packet['motion']), np.asarray(packet['obs'])
            bearing = state[:, 2] + obs[:, 3]
            target = (state[:, :2] + obs[:, 2, None] * np.column_stack(
                (np.cos(bearing), np.sin(bearing)))).mean(0)
            side = np.array([-target[1], target[0]]) / max(np.linalg.norm(target), 1e-9)
            goals = target + np.linspace(-self.width, self.width, n)[:, None] * side
            cost = np.linalg.norm(goals[None] - state[:, None, :2], axis=-1)
        if self.assignment == 'greedy':
            choice = np.empty(n, dtype=int)
            remaining = cost.copy()
            for _ in range(n):
                row, column = np.unravel_index(np.argmin(remaining), remaining.shape)
                choice[row] = column
                remaining[row, :] = np.inf
                remaining[:, column] = np.inf
        elif self.assignment == 'fixed' and self.fixed_choice is not None:
            choice = self.fixed_choice
        else:
            rows, columns = linear_sum_assignment(cost)
            choice = np.empty(n, dtype=int)
            choice[rows] = columns
            if self.assignment == 'fixed':
                self.fixed_choice = choice.copy()
        return candidates[np.arange(n), choice], 0.


class ConditionalGuardResidual(RoleResidualPolicy):
    """Reassign remaining guards when selected vessels take the capture role."""

    @staticmethod
    @lru_cache(maxsize=256)
    def layout(n, width, coefficients):
        coefficients = np.asarray(coefficients)
        if coefficients.shape != (n,) or not np.isin(coefficients, (0., 1.)).all():
            raise ValueError('This role-handover pilot uses binary compositions.')
        active, guards = np.where(coefficients == 1.)[0], np.where(coefficients == 0.)[0]
        if not len(active) or not len(guards):
            return active, guards, ()
        radii = abs(np.linspace(-width, width, n))
        cutoff = np.sort(radii)[len(active)-1]
        required = np.where(radii < cutoff-1e-9)[0].tolist()
        ties = np.where(np.isclose(radii, cutoff, rtol=0., atol=1e-9))[0].tolist()
        options = []
        for extra in combinations(ties, len(active)-len(required)):
            omitted = set(required+list(extra))
            options.append(np.array([j for j in range(n) if j not in omitted]))
        return active, guards, tuple(options)

    def compose(self, packet, coefficients):
        active, guards, options = self.layout(len(packet['motion']), self.width, tuple(coefficients))
        if not len(active):
            return self.candidate_controls(packet)[0]
        pursuit, _ = self.pursuit_controls(packet)
        action = pursuit.copy()
        if not len(guards):
            return action
        candidates, costs = self.candidates(packet)
        best = None
        for goals in options:
            rows, columns = linear_sum_assignment(costs[np.ix_(guards, goals)])
            value = float(costs[guards[rows], goals[columns]].sum())
            if best is None or value < best[0]:
                best = value, guards[rows], goals[columns]
        _, rows, columns = best
        action[rows] = candidates[rows, columns]
        return action.astype(np.float32)


def guard_composition_controller(n, independent=False, tail_steps=100):
    from role_prediction import RolePrediction
    controller = RolePrediction(n, tail_steps=tail_steps, tail_policy='candidate', additive=independent)
    controller.residual = ConditionalGuardResidual()
    return controller


def full_composition_controller(n, independent=False):
    from role_prediction import RolePrediction

    class FullCompositionPrediction(RolePrediction):
        def __init__(self):
            super().__init__(n, tail_policy='candidate')
            masks = []
            for count in range(n+1):
                for selected in combinations(range(n), count):
                    mask = np.zeros(n, dtype=np.float32)
                    mask[list(selected)] = 1.
                    masks.append(mask)
            self.options = dict(baseline=masks[0], **{str(i): m for i, m in enumerate(masks[1:], 1)})

        def predict(self, measurement, template):
            predicted = super().predict(measurement, template)
            if independent:
                mask = self.options[template]
                cost = 360.*predicted['score'][0]+180.*predicted['score'][1]+predicted['score'][2]
                if template == 'baseline':
                    self.base_cost, self.single_costs = cost, np.zeros(n)
                elif mask.sum() == 1:
                    self.single_costs[int(mask.argmax())] = cost-self.base_cost
                predicted['joint_score'] = predicted['score']
                predicted['score'] = (0, 0, self.base_cost+mask@self.single_costs)
            return predicted

    return FullCompositionPrediction()


SOURCE = None
JOINT = None


def initialize(source_path, joint_path):
    global SOURCE, JOINT
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore', message='.*Lgh is zero.*')
    SOURCE = OriginalPolicy(source_path)
    JOINT = DeploymentPolicy(joint_path)


def spaced_episode(method, n, agility, seed, minimum_spacing):
    from envs.TADgame import TADEnv
    from envs.snapshot import preserved_random_state, seed_random
    from feedback_joint_control import observe, minimum_separation
    from interaction_evaluation import long_reference_controller
    from interaction_rollout import public_packet, execute
    from role_prediction import RolePrediction
    from train_interaction import outcome_row
    supported = {'joint', 'fixed', 'role_sustained', 'role_independent', 'role_short', 'long_prediction_wide'}
    if method not in supported:
        raise ValueError('Unsupported controller in the declared initial-spacing study.')
    with preserved_random_state():
        seed_random(seed)
        env = TADEnv(n, protocol='paper-parameters-v1')
        obs, _ = env.reset(agility, noisy_agility=False)
        # Keep the original one-metre radial spread, angles, headings and noise.
        # For six vessels and a 7-m requirement the original reset is unchanged.
        offset = max(0., minimum_spacing/(2.*np.sin(np.pi/n))-7.)
        if offset > 1e-12:
            for boat in env.defender_list:
                boat.reset(boat.pos*(1.+offset/np.linalg.norm(boat.pos)), boat.theta)
            positions = np.array([b.pos for b in env.defender_list])
            headings = np.array([b.theta for b in env.defender_list])
            velocities = np.array([b.vel for b in env.defender_list])
            env.Pos_Def, env.Phi_Def = positions.flatten().copy(), headings.copy()
            env._Boid_navi_step(positions, velocities, headings, env.attacker.pos)
            obs, _ = env._get_obs()
            env.Rewards = env._get_rewards(done=0)
        separation = minimum_separation(observe(env, obs).defenders)
        if separation < minimum_spacing-1e-9:
            raise RuntimeError('Initial spacing requirement was not met.')
        feedback = None
        policy = JOINT if method == 'joint' else AssignmentControl('fixed')
        if method.startswith('role_'):
            feedback = RolePrediction(n, tail_steps=0 if method == 'role_short' else 100,
                tail_policy='candidate', additive=method == 'role_independent')
        elif method == 'long_prediction_wide':
            feedback = long_reference_controller(SOURCE, n, agility_bounds=(1., 8.))
        np.random.seed(seed+1000000)
        done = 0
        while not done:
            if feedback is None:
                packet = public_packet(env, obs)
                action, _ = policy.choose_action(packet, deterministic=True)
                packet, _, done, _, _, _ = execute(env, packet, action)
                obs = packet['obs']
            else:
                force, _ = feedback.control(observe(env, obs))
                action = SOURCE(obs) if method == 'long_prediction_wide' else np.zeros((n, 3))
                obs, _, done, _ = env.step(action, 'AdaRes', defender_thrust=force)
        return dict(outcome_row(env, done), initial_minimum_separation=separation,
                    initial_radius_offset=offset)


def run_one(task):
    method, n, agility, seed = task[:4]
    minimum_spacing = task[4] if len(task) > 4 else None
    started = time.perf_counter()
    if minimum_spacing is not None:
        row = spaced_episode(method, n, agility, seed, minimum_spacing)
    elif method in ('long_prediction', 'long_prediction_wide'):
        row, _ = long_reference_episode(SOURCE, seed, n, agility,
            agility_bounds=(1., 8.) if method == 'long_prediction_wide' else None)
    elif method in ('role_all_sustained', 'role_all_independent'):
        from distill_role_value import rollout
        if n != 6:
            raise ValueError('The full-composition pilot is limited to six vessels.')
        row, _ = rollout(seed, agility, lambda: full_composition_controller(n, method == 'role_all_independent'))
    elif method in ('role_guard_sustained', 'role_guard_independent', 'role_guard_short'):
        from distill_role_value import rollout
        if n != 6:
            raise ValueError('The role-handover pilot is limited to six vessels.')
        row, _ = rollout(seed, agility, lambda: guard_composition_controller(n,
            method == 'role_guard_independent', tail_steps=0 if method == 'role_guard_short' else 100))
    elif method in ('role_long_sustained', 'role_long_independent'):
        from distill_role_value import rollout
        from role_prediction import RolePrediction
        if n != 6:
            raise ValueError('The focused horizon pilot uses six vessels.')
        row, _ = rollout(seed, agility, lambda: RolePrediction(n, tail_steps=200,
            tail_policy='candidate', additive=method == 'role_long_independent'))
    elif method in ('role_predictive', 'role_short', 'role_sustained', 'role_independent'):
        from role_prediction import role_prediction_episode
        row, _ = role_prediction_episode(seed, n, agility, method)
    else:
        policy = (SOURCE if method == 'source' else JOINT if method == 'joint' else
                  CandidateRolePolicy(coordinated=False) if method == 'independent' else
                  RoleResidualPolicy(**RESIDUAL_SETTINGS[method]) if method in RESIDUAL_SETTINGS else
                  AssignmentControl(method))
        row, _ = episode(policy, seed, n, agility, safety=True)
    return dict(row, method=method, defenders=n, agility=agility, scene_seed=seed,
                evaluation_seconds=time.perf_counter() - started)


def summarize(rows, methods):
    result = {}
    for n, agility in sorted({(r['defenders'], r['agility']) for r in rows}):
        cell = [r for r in rows if r['defenders'] == n and r['agility'] == agility]
        result[f'n{n}-a{agility:g}'] = {
            method: dict(episodes=len(group), **{
                key: float(np.mean([r[key] for r in group]))
                for key in ('capture_time', 'capture', 'success', 'collision')})
            for method in methods if (group := [r for r in cell if r['method'] == method])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path(
        'train/experiments/paper-parameters-seed42-20260921/adares1.pth'))
    parser.add_argument('--joint', type=Path, default=Path(
        'train/experiments/ia-crrl-20261008-expanded-control/policy.pth'))
    parser.add_argument('--methods', nargs='+', default=['source', 'joint', 'independent',
                        'fixed', 'distance', 'greedy', 'long_prediction'])
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--scene-base', type=int, default=1020000000)
    parser.add_argument('--agilities', nargs='+', type=float, default=[4., 6.])
    parser.add_argument('--defenders', type=int, default=6)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--locked-confirmation', action='store_true')
    parser.add_argument('--primary-method', default='role_sustained')
    parser.add_argument('--replicate-interaction', action='store_true')
    parser.add_argument('--minimum-initial-separation', type=float)
    args = parser.parse_args()
    if args.replicate_interaction and (args.locked_confirmation or set(args.methods) != {'role_sustained', 'role_independent'}):
        parser.error('Interaction replication requires exactly the frozen joint and additive controls.')
    allowed = {'source', 'joint', 'independent', 'fixed', 'distance', 'greedy', 'long_prediction',
               'role_predictive', 'role_short', 'role_sustained', 'role_independent', 'long_prediction_wide',
               'role_long_sustained', 'role_long_independent'} | set(RESIDUAL_SETTINGS)
    allowed |= {'role_all_sustained', 'role_all_independent'}
    allowed |= {'role_guard_sustained', 'role_guard_independent', 'role_guard_short'}
    if not set(args.methods) <= allowed or len(set(args.methods)) != len(args.methods):
        parser.error('Unknown or duplicate methods.')
    if args.locked_confirmation and (args.primary_method not in args.methods or len(args.methods) < 2):
        parser.error('Locked confirmation requires its primary method and at least one reference.')
    if args.replicate_interaction and args.primary_method != 'role_sustained':
        parser.error('The original interaction replication has a fixed primary method.')
    if args.output.exists():
        parser.error('Use a new result directory; existing evidence is immutable.')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(range(24)))
    files = [Path(__file__), args.source, args.joint,
             Path('train/interaction_rollout.py'), Path('train/train_interaction.py'),
             Path('train/interaction_evaluation.py'), Path('train/envs/TADgame.py'),
             Path('train/policy/role_residual.py'), Path('train/role_prediction.py'),
             Path('train/distill_role_value.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    conditions = [(args.defenders, a, args.scene_base + i * 1000000)
                  for i, a in enumerate(args.agilities)]
    protocol = dict(methods=args.methods, count=args.count, conditions=conditions,
        files=hashes, frozen_before_execution=True, stage='development_comparator_challenge',
        environment='paper-parameters-v1', capped_capture_seconds=60,
        common_random_scene_and_disturbance=True, formal_evidence=False)
    if args.minimum_initial_separation is not None:
        if args.minimum_initial_separation <= 0 or args.locked_confirmation or args.replicate_interaction:
            parser.error('Declare a positive spacing as a new development task configuration.')
        protocol.update(minimum_initial_separation=args.minimum_initial_separation,
            initialization='Common radial offset max(0, separation/(2*sin(pi/N))-7); '
                'unchanged one-metre radial spread, angular layout, headings and random streams.')
    protocol['observer_bounds'] = dict(role_prediction=[1., 8.], long_prediction=[1.25, 3.25],
                                      long_prediction_wide=[1., 8.])
    if args.locked_confirmation or args.replicate_interaction:
        protocol.update(stage=('independent_interaction_replication' if args.replicate_interaction else 'independent_locked_confirmation'),
            primary_method=args.primary_method,
            primary_references=[m for m in args.methods if m != args.primary_method],
            primary_population=f'equal-weight {args.defenders}-vessel agility strata {args.agilities}',
            primary_gate='At least 10% lower pooled capped time against every primary reference; '
                'paired stratified simultaneous bootstrap intervals exclude zero; '
                'no higher observed collision rate; observed defense success no lower; '
                'one-sided 95% paired capture and success differences above -3 percentage points.',
            familywise_alpha=.05, bootstrap_repetitions=20000,
            exclusions='No dropped scenarios, checkpoint reselection, or parameter changes during evaluation.')
        if args.replicate_interaction:
            protocol.update(minimum_time_improvement=0.,
                primary_gate='A fresh single matched interaction contrast: paired stratified 95% time '
                    'interval below zero; capture and success noninferior within 3 percentage points; '
                    'observed success no lower and collisions no higher. This does not replace or '
                    'reclassify the earlier failed four-reference, 10%-against-each combined gate.',
                purpose='Replication of the fixed interaction effect, with no method or parameter search.')
    protocol['source_text'] = {str(p): p.read_text() for p in files if p.suffix == '.py'}
    atomic_json(args.output / 'protocol.json', protocol)
    tasks = [(method, n, a, bank + i, args.minimum_initial_separation) for n, a, bank in conditions
             for i in range(args.count) for method in args.methods]
    np.random.default_rng(20261008).shuffle(tasks)
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=initialize,
            initargs=(args.source, args.joint), mp_context=mp.get_context('spawn')) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 32 == 0 or len(rows) == len(tasks):
                atomic_json(args.output / 'results.json', dict(complete=False,
                    protocol=protocol, episodes=rows, summary=summarize(rows, args.methods),
                    wall_seconds=time.perf_counter()-started))
                print(json.dumps(dict(completed=len(rows), total=len(tasks))), flush=True)
    for name, expected in hashes.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen input changed during evaluation: ' + name)
    result = dict(complete=True, protocol=protocol, episodes=rows,
        summary=summarize(rows, args.methods), wall_seconds=time.perf_counter()-started)
    atomic_json(args.output / 'results.json', result)
    print(json.dumps(dict(complete=True, summary=result['summary'],
                         wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
