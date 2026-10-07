"""Independent latency and serial/parallel equivalence on old six-boat scenes."""
import study_runtime
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from array_rollout import ArrayRolloutController, BatchedSourcePolicy
from benchmark_predictive_interception import summary
import evaluate_predictive_interception as prior
from feedback_joint_control import observe
from parallel_array_rollout import ArrayPredictionPool, ParallelArrayController
from source_arboids import SourcePolicy, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tail-policy', choices=('baseline', 'candidate'), default='baseline')
    parser.add_argument('--tail-steps', type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path('train/experiments/robust-predictive-interception-20261006')
    root.mkdir(parents=True, exist_ok=True)
    policy = SourcePolicy(prior.old.CHECKPOINT)
    batch = BatchedSourcePolicy(policy)
    scenes = [next(s for s in prior.development_scenes() if s['defenders'] == 6 and s['agility'] == a)
              for a in (1.5, 2., 2.5, 3.)]
    settings = dict(block_steps=10, blend=1., tail_steps=args.tail_steps, tail_policy=args.tail_policy)
    records, cold = [], []
    equal_steps = equal_plans = 0
    with ArrayPredictionPool(prior.old.CHECKPOINT, workers=5) as pool:
        rules = dict(serial=ArrayRolloutController(6, policy, prediction_policy=batch, **settings),
                     parallel=ParallelArrayController(6, policy, pool, **settings))
        env, obs = prior.source.setup_scene(scenes[0])
        for engine, rule in rules.items():
            began = time.perf_counter()
            rule.control(observe(env, obs))
            cold.append(dict(engine=engine, first_plan_ms=1000 * (time.perf_counter() - began)))
        for scene in scenes:
            env, obs = prior.source.setup_scene(scene)
            for rule in rules.values():
                rule.reset()
            for step in range(100):
                outputs = {}
                for engine in (('serial', 'parallel') if step % 2 == 0 else ('parallel', 'serial')):
                    began = time.perf_counter()
                    force, info = rules[engine].control(observe(env, obs))
                    action = env.thrust_to_action(force)
                    seconds = time.perf_counter() - began
                    outputs[engine] = force, info, action
                    records.append(dict(scene_seed=scene['scene_seed'], step=step, engine=engine,
                                        planned=int(info['planned']), seconds=seconds))
                np.testing.assert_array_equal(outputs['serial'][0], outputs['parallel'][0])
                assert outputs['serial'][1]['template'] == outputs['parallel'][1]['template']
                equal_steps += 1
                if outputs['serial'][1]['planned']:
                    for key in ('score', 'baseline_score', 'positions', 'commands', 'attackers'):
                        np.testing.assert_array_equal(rules['serial'].plan[key], rules['parallel'].plan[key])
                    equal_plans += 1
                obs, _, done, _ = env.step(outputs['serial'][2], 'RL')
                if done:
                    break
            print(f'[ARRAY-TIMING] {scene["cell"]}; steps={equal_steps}, plans={equal_plans}', flush=True)
    table = []
    for engine in ('serial', 'parallel'):
        for phase in ('all', 'planning', 'feedback'):
            values = [r['seconds'] for r in records if r['engine'] == engine and
                      (phase == 'all' or bool(r['planned']) == (phase == 'planning'))]
            table.append(dict(engine=engine, phase=phase, **summary(values)))
    output = dict(passed=True, settings=settings, groups=table, cold=cold,
                  exact_steps=equal_steps, exact_plans=equal_plans,
                  scene_seeds=[s['scene_seed'] for s in scenes], platform=platform.platform(),
                  code={name: sha256(Path(__file__).with_name(name)) for name in
                        ('array_rollout.py', 'parallel_array_rollout.py', 'benchmark_array_rollout.py')})
    stem = f'array-runtime-{args.tail_policy}-{args.tail_steps}'
    prior.source.write_csv(root / (stem + '.csv'), records)
    (root / (stem + '.json')).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
