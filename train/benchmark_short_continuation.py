"""Time the fixed 8-second continuation without changing its arithmetic."""
import study_runtime
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

import evaluate_short_continuation as study
from benchmark_predictive_interception import summary
from feedback_joint_control import observe
from parallel_rollout_interception import PredictionPool, ParallelRolloutInterceptionController
from rollout_interception import RolloutInterceptionController
from source_arboids import sha256


ENGINES = ('reference_serial', 'compiled_serial', 'compiled_parallel')


def main():
    spec = json.loads((study.ROOT / 'specification.json').read_text(encoding='utf-8'))
    if spec != json.loads(json.dumps(study.frozen_inputs())):
        raise RuntimeError('Fixed follow-up inputs changed.')
    if not json.loads((study.ROOT / 'confirm_summary.json').read_text(encoding='utf-8'))['passed']:
        raise RuntimeError('Finish the confirmation before timing the controller.')
    initialization_started = time.perf_counter()
    study.initialize()
    actor_initialization_ms = 1000 * (time.perf_counter() - initialization_started)
    prior = study.prior
    scenes = [next(s for s in spec['development'] if s['defenders'] == 6 and s['agility'] == agility)
              for agility in (1.5, 2., 2.5, 3.)]
    settings = dict(block_steps=10, blend=1., tail_steps=40)
    records, cold = [], []
    equal_steps = equal_plans = 0
    with PredictionPool(prior.old.CHECKPOINT, workers=5) as pool:
        env, observations = prior.source.setup_scene(scenes[0])
        rules = {}
        for engine in ENGINES:
            began = time.perf_counter()
            if engine == 'reference_serial':
                rule = RolloutInterceptionController(6, prior.old._POLICY, **settings)
            elif engine == 'compiled_serial':
                rule = RolloutInterceptionController(6, study.FAST_POLICY, **settings)
            else:
                rule = ParallelRolloutInterceptionController(6, prior.old._POLICY, pool, **settings)
            rules[engine] = rule
            rule.control(observe(env, observations))
            cold.append(dict(engine=engine,
                             initialization_and_first_plan_ms=1000 * (time.perf_counter() - began)))
        for scene in scenes:
            env, observations = prior.source.setup_scene(scene)
            for rule in rules.values():
                rule.reset()
            for step in range(100):
                outputs = {}
                # Rotate the timing order; only the reference advances the world.
                order = ENGINES[step % 3:] + ENGINES[:step % 3]
                for engine in order:
                    began = time.perf_counter()
                    physical, info = rules[engine].control(observe(env, observations))
                    action = env.thrust_to_action(physical)
                    elapsed = time.perf_counter() - began
                    outputs[engine] = physical, info, action
                    records.append(dict(defenders=6, scene_seed=scene['scene_seed'], step=step,
                                        engine=engine, planned=int(info['planned']),
                                        template=info['template'], seconds=elapsed))
                reference = outputs['reference_serial']
                for engine in ENGINES[1:]:
                    np.testing.assert_array_equal(reference[0], outputs[engine][0])
                    if reference[1]['template'] != outputs[engine][1]['template']:
                        raise AssertionError('Candidate choice changed under compilation or scheduling.')
                equal_steps += 1
                if reference[1]['planned']:
                    for engine in ENGINES[1:]:
                        for key in ('positions', 'commands', 'attackers', 'score', 'baseline_score'):
                            np.testing.assert_array_equal(rules['reference_serial'].plan[key],
                                                          rules[engine].plan[key])
                    equal_plans += 1
                observations, _, done, _ = env.step(reference[2], 'RL')
                if done:
                    break
            print(f'[SHORT-LATENCY] {scene["cell"]}; exact steps={equal_steps}, plans={equal_plans}',
                  flush=True)
    table = []
    for engine in ENGINES:
        for phase in ('all', 'planning', 'feedback'):
            values = [r['seconds'] for r in records if r['engine'] == engine and
                      (phase == 'all' or bool(r['planned']) == (phase == 'planning'))]
            table.append(dict(defenders=6, engine=engine, phase=phase, **summary(values)))
    prior.source.write_csv(study.ROOT / 'runtime_samples.csv', records)
    prior.source.write_csv(study.ROOT / 'runtime_summary.csv', table)
    output = dict(
        passed=spec == json.loads(json.dumps(study.frozen_inputs())),
        platform=platform.platform(), processor=platform.processor(),
        python_version=platform.python_version(), torch_version=torch.__version__,
        numpy_version=np.__version__, torch_threads=torch.get_num_threads(),
        selected=study.SELECTED, actor_initialization_and_compilation_ms=actor_initialization_ms,
        scene_seeds=[s['scene_seed'] for s in scenes],
        maximum_steps_per_episode=100, cold=cold, groups=table,
        exact_command_steps=equal_steps, exact_plan_matches=equal_plans, parallel_workers=5,
        parallel_code_sha256=sha256(Path(__file__).with_name('parallel_rollout_interception.py')),
        compiled_actor_code_sha256=sha256(Path(__file__).with_name('compiled_source_policy.py')),
        benchmark_code_sha256=sha256(Path(__file__)),
        scope='Three arithmetically equivalent engines on four old development scenes; rotating timed order. '
              'Includes observation adapter, actor, candidate feedback, QPs and action mapping. '
              'Excludes environment integration, transport and initial compilation/pool startup. '
              'No evaluation workers active. Each actual state is advanced only once.')
    (study.ROOT / 'runtime_summary.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output), flush=True)
    if not output['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
