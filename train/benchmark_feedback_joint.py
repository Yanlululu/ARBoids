"""Single-process end-to-end latency on fixed captured states and real feedback."""
import study_runtime
import argparse
import json
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import torch

from source_arboids import SourcePolicy, sha256, verify_source
from gate_mechanism_diagnostics import restore, stable_number
from feedback_joint_control import observe
from feedback_joint_fast import FastFeedbackJointController
from evaluate_feedback_joint import CHECKPOINT, DIAGNOSTIC_ROOT, code_hashes
from evaluate_gate_contribution import write_csv


def summary(values):
    a=np.asarray(values)*1000.
    return dict(samples=len(a),median_ms=float(np.median(a)),p95_ms=float(np.quantile(a,.95)),
                p99_ms=float(np.quantile(a,.99)),maximum_ms=float(a.max()),
                deadline_misses=int(np.sum(a>200.)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('train/experiments/feedback-joint-20261006'))
    parser.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frozen=json.loads((args.output_dir/'frozen_method.json').read_text(encoding='utf-8'))
    if code_hashes()!=frozen['code']:
        raise RuntimeError('Frozen implementation changed.')
    source=verify_source()
    with (DIAGNOSTIC_ROOT/'states.pkl').open('rb') as stream:
        cases=pickle.load(stream)
    policy=SourcePolicy(CHECKPOINT)
    h=frozen['selected']['block_steps']
    audit=json.loads((args.output_dir/'acceleration_summary.json').read_text())
    if not audit['passed'] or audit['fast_code_sha256']!=sha256(Path(__file__).with_name('feedback_joint_fast.py')):
        raise RuntimeError('Accelerated predictor equivalence has not been verified.')
    rows=[]
    cold=[]
    selected=[]
    for n in (2,3,4,5):
        group=[c for c in cases if c['meta']['defenders']==n]
        group=sorted(group,key=lambda c:stable_number('latency-'+c['meta']['case_id']))[:8]
        selected.extend(c['meta']['case_id'] for c in group)
        env,obs=restore(group[0]['state'])
        began=time.perf_counter()
        rule=FastFeedbackJointController(n,policy,h)
        rule.control(observe(env,obs))
        cold.append(dict(defenders=n,first_initialization_and_plan_ms=1000*(time.perf_counter()-began)))
        for case in group:
            for repeat in range(args.repeats):
                env,obs=restore(case['state'])
                rule.reset(np.array([[b.left_thrust,b.right_thrust] for b in env.defender_list]))
                for step in range(h):
                    began=time.perf_counter()
                    physical,info=rule.control(observe(env,obs))
                    command=env.thrust_to_action(physical)
                    elapsed=time.perf_counter()-began
                    rows.append(dict(case_id=case['meta']['case_id'],defenders=n,repeat=repeat,
                                     step=step,planned=int(info['planned']),
                                     candidate_policies=info['candidate_policies'],seconds=elapsed))
                    obs,_,done,_=env.step(command,'RL')
                    if done:
                        break
        print(f'[BENCHMARK] N={n} states={len(group)} samples={sum(r["defenders"]==n for r in rows)}',flush=True)
    groups=[]
    for n in (2,3,4,5):
        for phase in ('all','planning','feedback'):
            subset=[r for r in rows if r['defenders']==n and
                    (phase=='all' or bool(r['planned'])==(phase=='planning'))]
            groups.append(dict(defenders=n,phase=phase,**summary([r['seconds'] for r in subset])))
    write_csv(args.output_dir/'runtime_samples.csv',rows)
    write_csv(args.output_dir/'runtime_summary.csv',groups)
    output=dict(passed=code_hashes()==frozen['code'] and verify_source()==source,
                platform=platform.platform(),processor=platform.processor(),torch_threads=torch.get_num_threads(),
                block_steps=h,repeats=args.repeats,state_ids=selected,
                prediction_engine='vectorized',fast_code_sha256=audit['fast_code_sha256'],
                scope='Single process; observe + actor + prediction + joint QP + command mapping; excludes environment integration and ROS transport',
                cold_start=cold,groups=groups,code_sha256=sha256(Path(__file__)))
    (args.output_dir/'runtime_summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output),flush=True)
    if not output['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
