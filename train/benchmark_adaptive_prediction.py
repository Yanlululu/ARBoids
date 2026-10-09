"""Isolated full-episode latency and serial/parallel equivalence benchmark."""
import study_runtime
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from array_rollout import BatchedSourcePolicy
from benchmark_predictive_interception import summary
from develop_adaptive_prediction import CONFIGS
import evaluate_predictive_interception as prior
from feedback_joint_control import observe
from adaptive_interval_rollout import AdaptiveIntervalController
from parallel_adaptive_prediction import ParallelAdaptiveController
from parallel_jit_rollout import JitPredictionPool
from predictive_interception_v2 import TEMPLATES
from source_arboids import SourcePolicy, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=('predictive',), default='predictive')
    args = parser.parse_args()
    root = Path('train/experiments/adaptive-prediction-20261006')
    stem = 'runtime-adaptive'
    if (root/(stem+'.json')).exists():
        raise RuntimeError('An existing benchmark must not be overwritten.')
    torch.set_num_threads(1)
    started = time.perf_counter()
    policy = SourcePolicy(prior.old.CHECKPOINT)
    batch = BatchedSourcePolicy(policy)
    policy_initialization_ms = 1000*(time.perf_counter()-started)
    scenes = [s for s in prior.development_scenes() if s['defenders']==6]
    settings = dict(CONFIGS[args.method])
    records, outcomes, cold = [], [], []
    exact_steps = exact_plans = exact_candidates = 0
    code_names = ('array_rollout.py','compiled_source_policy.py','jit_nominal_environment.py',
                  'jit_rollout.py','parallel_jit_rollout.py','adaptive_interval_rollout.py',
                  'parallel_adaptive_prediction.py','benchmark_adaptive_prediction.py')
    code = {name:sha256(Path(__file__).with_name(name)) for name in code_names}
    env, obs = prior.source.setup_scene(scenes[0])
    warmup = (6, {k:v for k,v in settings.items() if k!='agility_threshold'}, observe(env, obs))
    with JitPredictionPool(prior.old.CHECKPOINT, workers=5, warmup=warmup) as pool:
        rules = dict(serial=AdaptiveIntervalController(6, policy, prediction_policy=batch, **settings),
                     parallel=ParallelAdaptiveController(6, policy, pool, **settings))
        for engine, rule in rules.items():
            began = time.perf_counter()
            rule.control(observe(env, obs))
            cold.append(dict(engine=engine, first_plan_ms=1000*(time.perf_counter()-began)))
        for scene_id, scene in enumerate(scenes):
            env, obs = prior.source.setup_scene(scene)
            for rule in rules.values():
                rule.reset()
            done = step = 0
            while not done:
                outputs = {}
                for engine in (('serial','parallel') if (step+scene_id)%2==0 else ('parallel','serial')):
                    began = time.perf_counter()
                    measurement = observe(env, obs)
                    force, info = rules[engine].control(measurement)
                    action = env.thrust_to_action(force)
                    elapsed = time.perf_counter()-began
                    outputs[engine] = force, info, action
                    records.append(dict(scene_seed=scene['scene_seed'], agility=scene['agility'],
                        step=step, engine=engine, planned=int(info['planned']),
                        interval_steps=info['execution_interval_steps'], seconds=elapsed))
                np.testing.assert_array_equal(outputs['serial'][0], outputs['parallel'][0])
                assert outputs['serial'][1]['template']==outputs['parallel'][1]['template']
                exact_steps += 1
                if outputs['serial'][1]['planned']:
                    for key in ('score','baseline_score','positions','commands','attackers'):
                        np.testing.assert_array_equal(rules['serial'].plan[key], rules['parallel'].plan[key])
                    exact_plans += 1
                    if scene_id == 0 and step == 0:
                        # All candidate forecasts are checked once, not just the selected plan.
                        original = rules['serial']
                        saved_def, saved_att = original.previous_thrust, original.previous_attacker_thrust
                        original.previous_thrust = original.previous_attacker_thrust = None
                        for template in TEMPLATES:
                            a = original.predict(measurement, template)
                            b = rules['parallel']._batch[template].result()
                            for key in ('score','positions','commands','attackers'):
                                np.testing.assert_array_equal(a[key], b[key])
                            exact_candidates += 1
                        original.previous_thrust, original.previous_attacker_thrust = saved_def, saved_att
                obs, _, done, _ = env.step(outputs['serial'][2], 'RL')
                step += 1
                if step > 401:
                    raise RuntimeError('Physical horizon exceeded.')
            outcomes.append(dict(**scene, outcome=int(done), capture=int(done==3), success=int(done>2),
                                 collision=int(done==2), duration=float(env.Current_T)))
            print(f'[ROBUST-TIMING] {scene_id+1}/32; steps={exact_steps}, plans={exact_plans}', flush=True)
    table = []
    for engine in ('serial','parallel'):
        for phase in ('all','planning','feedback'):
            values = [r['seconds'] for r in records if r['engine']==engine and
                      (phase=='all' or bool(r['planned'])==(phase=='planning'))]
            table.append(dict(engine=engine, phase=phase, **summary(values)))
    passed = exact_candidates==5 and all(sha256(Path(__file__).with_name(k))==v for k,v in code.items())
    result = dict(passed=passed, settings=settings, method=args.method, groups=table, cold=cold,
        policy_initialization_ms=policy_initialization_ms, exact_steps=exact_steps, exact_plans=exact_plans,
        exact_candidates=exact_candidates, outcomes=outcomes, platform=platform.platform(),
        workers=5, torch_threads=1, code=code, interval_plan_counts={h:sum(r['engine']=='parallel' and r['planned'] and r['interval_steps']==h for r in records) for h in (5,10)}, scene_seeds=[s['scene_seed'] for s in scenes],
        measurement='observe + complete controller call + command conversion; full episodes; no other experiment jobs',
        warmup='Every worker compiles the actor, safety QP and model before a shared startup barrier opens.')
    prior.source.write_csv(root/(stem+'.csv'), records)
    (root/(stem+'.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
