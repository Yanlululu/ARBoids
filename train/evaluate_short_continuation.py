"""One fixed follow-up: robust development selection, then a fresh 6-boat set."""
import study_runtime
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import time

import evaluate_predictive_interception as prior
from compiled_source_policy import CompiledSourcePolicy
from predictive_interception_v2 import TaskTimePredictiveController
from rollout_interception import RolloutInterceptionController
from source_arboids import sha256


ROOT=prior.ROOT/'short-confirmation'
PROTOCOL=Path('docs/short-continuation-study.md')
METHODS=('original','cbf','long_value','predictive','short_value','held_prefix','best_fixed')
SELECTED=dict(predictive='rollout_t40_b100',fixed='fixed_lead_guard_b100')
RULES={}
FAST_POLICY=None


def initialize():
    global FAST_POLICY
    prior.initialize(SELECTED)
    FAST_POLICY=CompiledSourcePolicy(prior.old._POLICY)
    prior.old.new_controller=factory


def factory(n,method,h):
    key=n,method
    if key not in RULES:
        settings=dict(block_steps=10,blend=1.)
        if method=='short_value':
            rule=TaskTimePredictiveController(n,FAST_POLICY,**settings)
        elif method=='best_fixed':
            rule=TaskTimePredictiveController(n,prior.old._POLICY,**prior.FIXED[SELECTED['fixed']])
        else:
            rule=RolloutInterceptionController(n,FAST_POLICY,**settings,
                tail_steps=100 if method=='long_value' else 40,
                prediction='held' if method=='held_prefix' else 'feedback')
        RULES[key]=rule
    return RULES[key]


def scenes():
    return [dict(cell=f'n6-a{a:g}',defenders=6,agility=a,
            scene_seed=183160000+i*10000+r,noise_seed=184160000+i*10000+r)
            for i,a in enumerate((1.5,2.,2.5,3.)) for r in range(32)]


def frozen_inputs():
    previous=prior.frozen_inputs()
    frozen=json.loads((prior.ROOT/'validation/specification.json').read_text(encoding='utf-8'))
    if json.loads(json.dumps(previous))!=frozen:
        raise RuntimeError('The first study has changed.')
    code=dict(previous['code'])
    for name in ('compiled_source_policy.py','evaluate_short_continuation.py'):
        code[name]=sha256(Path(__file__).with_name(name))
    earlier={s['scene_seed'] for s in previous['development']+previous['confirmation']}
    if earlier & {s['scene_seed'] for s in scenes()}:
        raise AssertionError('Follow-up set overlaps earlier data.')
    return dict(code=code,source=previous['source'],checkpoint_sha256=previous['checkpoint_sha256'],
        protocol_sha256=sha256(PROTOCOL),first_confirmation_sha256=sha256(prior.ROOT/'validation/confirm.csv'),
        development=previous['development'],confirmation=scenes(),methods=list(METHODS),selected=SELECTED,
        confidence_level=.975,primary_defenders=6)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    spec=json.loads(json.dumps(frozen_inputs()))
    path=ROOT/'specification.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=spec:
            raise RuntimeError('Fixed follow-up changed.')
    else:
        path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    if (ROOT/'confirm.csv').exists():
        raise RuntimeError('This one follow-up has already completed.')
    (ROOT/'frozen_parameters.json').write_text(json.dumps(dict(selected=SELECTED,code=spec['code']),indent=2),encoding='utf-8')
    began=time.perf_counter()
    groups=[]
    with ProcessPoolExecutor(max_workers=4,initializer=initialize) as pool:
        pending={pool.submit(prior.scene_job,(scene,METHODS)):i for i,scene in enumerate(scenes())}
        for task in as_completed(pending):
            groups.append((pending[task],task.result()))
            print(f'[SHORT-CONFIRM] {len(groups)}/128; {time.perf_counter()-began:.1f}s',flush=True)
    rows=[r for _,group in sorted(groups) for r in group]
    prior.source.write_csv(ROOT/'confirm.csv',rows)
    passed=len(rows)==896 and json.loads(json.dumps(frozen_inputs()))==spec
    result=dict(passed=passed,scenes=128,rows=len(rows),elapsed_seconds=time.perf_counter()-began,selected=SELECTED)
    (ROOT/'confirm_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
