"""Parallel candidate forecasts with the unchanged residual composition policy."""
import study_runtime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from threading import Lock

from distill_role_value import initialize
from role_prediction import RolePrediction


RULES = {}
BOUND = None


def initialize_forecasting(bound):
    global BOUND
    initialize()
    BOUND = bound


def current_bound():
    return BOUND.value


def set_handover(controller, enabled):
    if enabled:
        from evaluate_candidate_roles import ConditionalGuardResidual
        controller.residual = ConditionalGuardResidual()
    return controller


def make_controller(n, tail, mode):
    if mode == 'adaptive':
        if tail != 100:
            raise ValueError('The adaptive pilot uses the frozen 20-second continuation.')
        from evaluate_candidate_roles import adaptive_guard_controller
        return adaptive_guard_controller(n)
    return set_handover(RolePrediction(n, tail_steps=tail, tail_policy='candidate'), mode == 'guard')


def forecast(request):
    n, tail, mode, measurement, previous, previous_attacker, agility, template = request
    key = n, tail, mode
    if key not in RULES:
        RULES[key] = make_controller(n, tail, mode)
        RULES[key].prediction_bound = current_bound if BOUND is not None else None
    controller = RULES[key]
    controller.previous_thrust = previous
    controller.previous_attacker_thrust = previous_attacker
    controller.attacker_agility = agility
    predicted = controller.predict(measurement, template)
    if BOUND is not None and not predicted.get('pruned') and predicted['score'][:2] == (0, 0):
        with BOUND.get_lock():
            BOUND.value = min(BOUND.value, predicted['score'][2])
    return predicted


class RolePredictionPool:
    def __init__(self, workers=22, prune=False):
        context = mp.get_context('spawn')
        self.bound = context.Value('d', float('inf')) if prune else None
        self.control_lock = Lock()
        self.executor = ProcessPoolExecutor(workers, initializer=initialize_forecasting,
            initargs=(self.bound,), mp_context=context)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=True, cancel_futures=True)


class ParallelRolePrediction(RolePrediction):
    def __init__(self, defenders, pool, tail_steps=100, additive=False, handover=False, adaptive=False):
        super().__init__(defenders, tail_steps=tail_steps, tail_policy='candidate', additive=additive)
        if pool.bound is not None and additive:
            raise ValueError('Bound pruning requires joint selection, not additive value reconstruction.')
        if adaptive and (additive or handover):
            raise ValueError('Adaptive timing uses its joint candidate map without another mode flag.')
        set_handover(self, handover)
        self.mode = 'adaptive' if adaptive else 'guard' if handover else 'point'
        if adaptive:
            candidate_map = make_controller(defenders, tail_steps, self.mode)
            self.options = candidate_map.options
            # Only the stateless candidate map is delegated; observer and plan
            # state remain in this parallel controller.
            self.nominal = candidate_map.nominal
        self.pool, self.batch_tick, self.batch = pool, None, {}

    def control(self, measurement):
        # A shared bound belongs to exactly one completed planning batch.
        with self.pool.control_lock:
            return super().control(measurement)

    def predict(self, measurement, template):
        if self.batch_tick != self.tick:
            self.batch_tick = self.tick
            ordering = list(self.options)
            if self.pool.bound is not None:
                self.pool.bound.value = float('inf')
                previous = getattr(self, 'template', 'baseline')
                if previous in ordering:
                    ordering.remove(previous)
                    ordering.insert(0, previous)
            self.batch = {t: self.pool.executor.submit(forecast, (self.defenders, self.tail_steps, self.mode,
                measurement, self.previous_thrust, self.previous_attacker_thrust, self.attacker_agility, t))
                for t in ordering}
        return self.batch[template].result()


def benchmark():
    import argparse
    import hashlib
    import os
    from pathlib import Path
    import time
    import numpy as np
    from envs.TADgame import TADEnv
    from envs.snapshot import seed_random
    from feedback_joint_control import observe
    from train_interaction import atomic_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=22)
    parser.add_argument('--prune', action='store_true')
    mode_flags = parser.add_mutually_exclusive_group()
    mode_flags.add_argument('--handover', action='store_true')
    mode_flags.add_argument('--adaptive-guard', action='store_true')
    parser.add_argument('--defenders', nargs='+', type=int, default=[6, 8])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Existing timing evidence must be retained.')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(range(24)))
    initialize()
    paths = [Path(__file__), Path('train/role_prediction.py'), Path('train/jit_rollout.py'),
             Path('train/policy/role_residual.py')]
    if args.handover or args.adaptive_guard:
        paths.append(Path('train/evaluate_candidate_roles.py'))
    mode = 'adaptive' if args.adaptive_guard else 'guard' if args.handover else 'point'
    protocol = dict(workers=args.workers, exact_bound_pruning=args.prune, mode=mode, handover=args.handover, conditions=[(n, a, 1020000000+c*1000000)
        for n in args.defenders for c, a in enumerate((4., 6.))], complete_episodes=True,
        source={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), text=p.read_text()) for p in paths})
    rows, started = [], time.perf_counter()
    with RolePredictionPool(args.workers, prune=args.prune) as pool:
        for n, agility, seed in protocol['conditions']:
            seed_random(seed)
            env = TADEnv(n, protocol='paper-parameters-v1')
            obs, _ = env.reset(agility, noisy_agility=False)
            serial = make_controller(n, 100, mode)
            parallel = ParallelRolePrediction(n, pool, handover=args.handover, adaptive=args.adaptive_guard)
            np.random.seed(seed+1000000)
            done, step = 0, 0
            while not done:
                measurement, controls, elapsed = observe(env, obs), {}, {}
                ordering = ('serial', 'parallel') if step % 2 == 0 else ('parallel', 'serial')
                for name in ordering:
                    controller = serial if name == 'serial' else parallel
                    before = time.perf_counter()
                    controls[name] = controller.control(measurement)
                    elapsed[name] = time.perf_counter()-before
                (a, info), (b, other) = controls['serial'], controls['parallel']
                np.testing.assert_array_equal(a, b)
                assert info['template'] == other['template']
                if info['planned']:
                    assert not parallel.plan.get('pruned')
                    for key in ('score', 'positions', 'attackers', 'commands'):
                        np.testing.assert_array_equal(serial.plan[key], parallel.plan[key])
                rows.append(dict(n=n, agility=agility, scene_seed=seed, step=step,
                    planned=info['planned'], cold=(agility == 4. and step == 0),
                    pruned_candidates=sum(bool(f.result().get('pruned')) for f in parallel.batch.values()) if info['planned'] else 0,
                    **elapsed))
                obs, _, done, _ = env.step(np.zeros((n, 3)), 'AdaRes', defender_thrust=a)
                step += 1
            print(dict(stage='parity', n=n, agility=agility, controls=step), flush=True)
    summary = {}
    for n in args.defenders:
        for planned in (True, False):
            for method in ('serial', 'parallel'):
                values = np.array([r[method] for r in rows if r['n'] == n and r['planned'] == planned and not r['cold']])
                summary[f'n{n}-{method}-planned-{planned}'] = dict(count=len(values),
                    median_ms=float(np.median(values)*1000.), p95_ms=float(np.quantile(values, .95)*1000.),
                    max_ms=float(values.max()*1000.), above_200ms=int((values > .2).sum()))
    for name, frozen in protocol['source'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != frozen['sha256']:
            raise RuntimeError('Frozen source changed: '+name)
    result = dict(protocol=protocol, rows=rows, summary=summary, exact_control_and_plan_parity=True,
        cold=[r for r in rows if r['cold']], cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
        wall_seconds=time.perf_counter()-started)
    atomic_json(args.output, result)
    print(dict(summary=summary, exact_parity=True, controls=len(rows), cold=result['cold']), flush=True)


if __name__ == '__main__':
    benchmark()
