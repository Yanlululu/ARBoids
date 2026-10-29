"""Controller timing on actual decision roots; never substitutes for mission results."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
import civ_rl as c
import civ_generalization as g
from civ_paper_evidence import OUTPUT, design, read
from civ_rl_experiment import _atomic_save, digest


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    path=OUTPUT/'controller-latency.pth'
    if path.exists():return
    specification=design();model=Path(specification['models']['CIV']['path'])
    policy=g.load(model,'cuda:0');rows=[];roots={}
    for n in (3,4,5,6,7,8):
        tasks=specification['cells']['confirmation' if n==6 else f'fleet-{n}'][:4]
        records=c.run_episodes(policy,tasks,collect_roots=True,stop_at_root=True)
        roots[n]=[r['root'] for r in records if r['root'] is not None]
        if not roots[n]:raise ValueError('No actual decision root for timing')
    def measure(controller,root,kind,setting,event):
        packet=c.d.mechanism_tensor(c.d.joint_packet(root['environment'],root['observation']),'cuda:0')
        executor=c.CIVExecutor(controller);samples=[]
        for repetition in range(10):
            state=c.ExecutionState(step=controller.config.decision_step if event=='joint-value-decision' else controller.config.decision_step+1,
                decided=event!='joint-value-decision')
            torch.cuda.synchronize();start=time.perf_counter()
            executor.actions(packet,[state]);torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            if repetition>=2:samples.append(elapsed)
        return dict(kind=kind,setting=setting,event=event,scene=root['scene'],root_time=root['environment'].Current_T,
            root_step=root['step'],seconds=samples,warmup_calls=2,config=asdict(controller.config))
    for n,bank in roots.items():
        for event in ('joint-value-decision','ordinary-control-step'):
            rows.extend(measure(policy,root,'fleet',n,event) for root in bank)
        print('LATENCY_FLEET',n,flush=True)
    original=dict(policy.source.actuator_fusion_refinement)
    try:
        for multiplier in (0.,.5,1.,2.):
            policy.source.actuator_fusion_refinement=dict(original,risk_weight=original.get('risk_weight',32.)*multiplier)
            controller=c.CIVController(policy.source,policy.config)
            rows.extend(measure(controller,root,'lambda',multiplier,'joint-value-decision') for root in roots[6])
            print('LATENCY_LAMBDA',multiplier,flush=True)
    finally:policy.source.actuator_fusion_refinement=original
    _atomic_save(dict(completed=True,rows=rows,policy_sha256=digest(model),design_sha256=digest(OUTPUT/'design.pth'),
        device='cuda:0',gpu=torch.cuda.get_device_name(),runtime=c.runtime_identity(),driver_sha256=digest(__file__),
        scope='Full stateful controller wall time on the first four fixed task roots per fleet, eight calls after two warmups; includes fixed-width padding and CPU output transfer; excludes Gazebo and transport; shared server load.'),path)
    print('LATENCY_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
