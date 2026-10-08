"""Parallel candidate forecasts with the unchanged residual composition policy."""
import study_runtime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from distill_role_value import initialize
from role_prediction import RolePrediction


RULES = {}


def set_handover(controller, enabled):
    if enabled:
        from evaluate_candidate_roles import ConditionalGuardResidual
        controller.residual = ConditionalGuardResidual()
    return controller


def forecast(request):
    n, tail, handover, measurement, previous, previous_attacker, agility, template = request
    key = n, tail, handover
    if key not in RULES:
        RULES[key] = set_handover(RolePrediction(n, tail_steps=tail, tail_policy='candidate'), handover)
    controller = RULES[key]
    controller.previous_thrust = previous
    controller.previous_attacker_thrust = previous_attacker
    controller.attacker_agility = agility
    return controller.predict(measurement, template)


class RolePredictionPool:
    def __init__(self, workers=22):
        self.executor = ProcessPoolExecutor(workers, initializer=initialize, mp_context=mp.get_context('spawn'))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=True, cancel_futures=True)


class ParallelRolePrediction(RolePrediction):
    def __init__(self, defenders, pool, tail_steps=100, additive=False, handover=False):
        super().__init__(defenders, tail_steps=tail_steps, tail_policy='candidate', additive=additive)
        set_handover(self, handover)
        self.handover = handover
        self.pool, self.batch_tick, self.batch = pool, None, {}

    def predict(self, measurement, template):
        if self.batch_tick != self.tick:
            self.batch_tick = self.tick
            self.batch = {t: self.pool.executor.submit(forecast, (self.defenders, self.tail_steps, self.handover,
                measurement, self.previous_thrust, self.previous_attacker_thrust, self.attacker_agility, t))
                for t in self.options}
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
    parser.add_argument('--handover', action='store_true')
    parser.add_argument('--defenders', nargs='+', type=int, default=[6, 8])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Existing timing evidence must be retained.')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(range(24)))
    initialize()
    paths = [Path(__file__), Path('train/role_prediction.py'), Path('train/jit_rollout.py'),
             Path('train/policy/role_residual.py')]
    if args.handover:
        paths.append(Path('train/evaluate_candidate_roles.py'))
    protocol = dict(workers=args.workers, handover=args.handover, conditions=[(n, a, 1020000000+c*1000000)
        for n in args.defenders for c, a in enumerate((4., 6.))], complete_episodes=True,
        source={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), text=p.read_text()) for p in paths})
    rows, started = [], time.perf_counter()
    with RolePredictionPool(args.workers) as pool:
        for n, agility, seed in protocol['conditions']:
            seed_random(seed)
            env = TADEnv(n, protocol='paper-parameters-v1')
            obs, _ = env.reset(agility, noisy_agility=False)
            serial = set_handover(RolePrediction(n, tail_policy='candidate'), args.handover)
            parallel = ParallelRolePrediction(n, pool, handover=args.handover)
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
                    for key in ('score', 'positions', 'attackers', 'commands'):
                        np.testing.assert_array_equal(serial.plan[key], parallel.plan[key])
                rows.append(dict(n=n, agility=agility, scene_seed=seed, step=step,
                    planned=info['planned'], cold=(agility == 4. and step == 0), **elapsed))
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
