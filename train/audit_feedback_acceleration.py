"""Numerical equivalence of the optimized predictor on all diagnostic states."""
import study_runtime
import argparse
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from feedback_joint_control import FeedbackJointController, observe, TEMPLATES
from feedback_joint_fast import FastFeedbackJointController
from evaluate_feedback_joint import CHECKPOINT, DIAGNOSTIC_ROOT, code_hashes
from gate_mechanism_diagnostics import restore
from source_arboids import SourcePolicy, sha256
from evaluate_gate_contribution import write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('train/experiments/feedback-joint-20261006'))
    args=parser.parse_args()
    frozen=json.loads((args.output_dir/'frozen_method.json').read_text(encoding='utf-8'))
    assert code_hashes()==frozen['code']
    with (DIAGNOSTIC_ROOT/'states.pkl').open('rb') as stream:
        cases=pickle.load(stream)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    policy=SourcePolicy(CHECKPOINT)
    rules={}
    rows=[]
    chosen=[]
    for case in cases:
        n=case['meta']['defenders']
        if n not in rules:
            rules[n]=(FeedbackJointController(n,policy,frozen['selected']['block_steps']),
                      FastFeedbackJointController(n,policy,frozen['selected']['block_steps']))
        reference,fast=rules[n]
        env,obs=restore(case['state'])
        past=np.array([[b.left_thrust,b.right_thrust] for b in env.defender_list])
        reference.reset(past);fast.reset(past)
        measurement=observe(env,obs)
        predictions=[[],[]]
        for template in TEMPLATES:
            start=time.perf_counter()
            a=reference.predict(measurement,template)
            reference_seconds=time.perf_counter()-start
            start=time.perf_counter()
            b=fast.predict(measurement,template)
            fast_seconds=time.perf_counter()-start
            predictions[0].append(a);predictions[1].append(b)
            same_length=a['positions'].shape==b['positions'].shape
            position_error=float(np.max(np.abs(a['positions']-b['positions']))) if same_length else float('inf')
            command_error=float(np.max(np.abs(a['commands']-b['commands']))) if same_length else float('inf')
            score_error=float(abs(a['score'][-1]-b['score'][-1]))
            rows.append(dict(case_id=case['meta']['case_id'],defenders=n,template=template,
                same_length=same_length,same_outcome=a['outcome']==b['outcome'],
                maximum_state_error=position_error,maximum_thrust_error=command_error,
                score_error=score_error,reference_seconds=reference_seconds,fast_seconds=fast_seconds))
        selections=[]
        for candidates in predictions:
            base=candidates[0]
            consider=candidates if base['minimum_distance']<8. or base['outcome']==1 else [base]
            selections.append(min(consider,key=lambda x:x['score'])['template'])
        chosen.append(dict(case_id=case['meta']['case_id'],reference=selections[0],fast=selections[1],
                           same_selection=selections[0]==selections[1]))
    summary=dict(passed=all(r['same_length'] and r['same_outcome'] and r['maximum_state_error']<1e-5
                           and r['maximum_thrust_error']<.02 and r['score_error']<1e-6 for r in rows)
                         and all(r['same_selection'] for r in chosen) and code_hashes()==frozen['code'],
                 states=len(cases),candidate_predictions=len(rows),
                 maximum_state_error=max(r['maximum_state_error'] for r in rows),
                 maximum_thrust_error=max(r['maximum_thrust_error'] for r in rows),
                 maximum_score_error=max(r['score_error'] for r in rows),
                 changed_selections=sum(not r['same_selection'] for r in chosen),
                 fast_code_sha256=sha256(Path(__file__).with_name('feedback_joint_fast.py')))
    write_csv(args.output_dir/'acceleration_predictions.csv',rows)
    write_csv(args.output_dir/'acceleration_selections.csv',chosen)
    (args.output_dir/'acceleration_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    if not summary['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
