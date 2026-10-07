"""Serial/parallel latency and exact-equivalence audit on actual feedback."""
import study_runtime
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch

import evaluate_predictive_interception as study
from feedback_joint_control import observe
from parallel_rollout_interception import PredictionPool, ParallelRolloutInterceptionController
from source_arboids import sha256


def summary(seconds):
    values=np.asarray(seconds)*1000.
    return dict(samples=len(values),median_ms=float(np.median(values)),
        p95_ms=float(np.quantile(values,.95)),p99_ms=float(np.quantile(values,.99)),
        max_ms=float(values.max()),deadline_misses=int((values>200.).sum()))


def main():
    root=study.ROOT/'validation'
    spec=json.loads((root/'specification.json').read_text(encoding='utf-8'))
    if spec!=json.loads(json.dumps(study.frozen_inputs())):
        raise RuntimeError('Frozen study differs.')
    selected=json.loads((root/'frozen_parameters.json').read_text(encoding='utf-8'))['selected']
    study.initialize(selected)
    records,cold,roster=[],[],[]
    equal_steps=equal_plans=0
    with PredictionPool(study.old.CHECKPOINT,workers=5) as pool:
        for n in (2,3,4,5,6):
            scenes=[next(s for s in spec['development'] if s['defenders']==n and s['agility']==a)
                    for a in (1.5,2.,2.5,3.)]
            env,observations=study.source.setup_scene(scenes[0])
            rules={}
            for engine in ('serial','parallel'):
                began=time.perf_counter()
                rules[engine]=(study.factory(n,'predictive',10) if engine=='serial' else
                    ParallelRolloutInterceptionController(n,study.old._POLICY,pool,
                        **study.CONFIGS[selected['predictive']]))
                rules[engine].control(observe(env,observations))
                cold.append(dict(defenders=n,engine=engine,
                    initialization_and_first_plan_ms=1000*(time.perf_counter()-began)))
            for scene in scenes:
                roster.append(scene['scene_seed'])
                env,observations=study.source.setup_scene(scene)
                for rule in rules.values():
                    rule.reset()
                for step in range(100):
                    outputs={}
                    # Alternating order limits cache/thermal bias; both see exactly
                    # the same measurement and neither advances actual noise.
                    for engine in (('serial','parallel') if step%2==0 else ('parallel','serial')):
                        began=time.perf_counter()
                        physical,info=rules[engine].control(observe(env,observations))
                        action=env.thrust_to_action(physical)
                        elapsed=time.perf_counter()-began
                        outputs[engine]=(physical,info,action)
                        records.append(dict(defenders=n,scene_seed=scene['scene_seed'],step=step,engine=engine,
                            planned=int(info['planned']),template=info['template'],seconds=elapsed))
                    np.testing.assert_array_equal(outputs['serial'][0],outputs['parallel'][0])
                    if outputs['serial'][1]['template']!=outputs['parallel'][1]['template']:
                        raise AssertionError('Candidate choice changed under parallel scheduling.')
                    equal_steps+=1
                    if outputs['serial'][1]['planned']:
                        a,b=rules['serial'].plan,rules['parallel'].plan
                        for key in ('positions','commands','attackers','score','baseline_score'):
                            np.testing.assert_array_equal(a[key],b[key])
                        equal_plans+=1
                    observations,_,done,_=env.step(outputs['serial'][2],'RL')
                    if done:
                        break
            print(f'[LATENCY] N={n}, samples={sum(r["defenders"]==n for r in records)}',flush=True)
    table=[]
    for n in (2,3,4,5,6):
        for engine in ('serial','parallel'):
            for phase in ('all','planning','feedback'):
                values=[r['seconds'] for r in records if r['defenders']==n and r['engine']==engine and
                        (phase=='all' or bool(r['planned'])==(phase=='planning'))]
                table.append(dict(defenders=n,engine=engine,phase=phase,**summary(values)))
    study.source.write_csv(root/'runtime_samples.csv',records)
    study.source.write_csv(root/'runtime_summary.csv',table)
    output=dict(passed=spec==json.loads(json.dumps(study.frozen_inputs())),platform=platform.platform(),
        processor=platform.processor(),python_version=platform.python_version(),
        torch_version=torch.__version__,numpy_version=np.__version__,
        torch_threads=torch.get_num_threads(),selected=selected,
        scene_seeds=roster,maximum_steps_per_episode=100,cold=cold,groups=table,
        exact_command_steps=equal_steps,exact_plan_matches=equal_plans,parallel_workers=5,
        parallel_code_sha256=sha256(Path(__file__).with_name('parallel_rollout_interception.py')),
        compiled_actor_code_sha256=sha256(Path(__file__).with_name('compiled_source_policy.py')),
        compiled_actor=True,
        benchmark_code_sha256=sha256(Path(__file__)),
        scope='Serial and five prediction-worker implementations on identical states, alternating timed order; observation adapter, actor, all counterfactual feedback, QPs and command mapping; excludes environment integration and transport. No evaluation workers active.')
    (root/'runtime_summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output),flush=True)
    if not output['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
