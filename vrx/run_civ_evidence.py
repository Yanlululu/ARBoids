"""Unified CIV executor through the existing measured Gazebo transport."""
from pathlib import Path
import hashlib
import json
import sys

import run_r20_evidence as transport

ROOT=Path(__file__).resolve().parents[1]


def load_controller(label,baseline_seed,manifest,inference_device):
    from civ_deployment import CIVMeasuredController
    path=Path(label).resolve()
    controller=CIVMeasuredController(path,inference_device=inference_device)
    source=ROOT/'train/experiments/main-seed42-20260921/adares1.pth'
    return controller,path,None,str(source)


def run_trial(extra,remaining,manifest,synchronized,policy_path,baseline_family,source_checkpoint):
    sys.argv=[sys.argv[0],'--method','joint','--joint-checkpoint',str(policy_path),
        '--checkpoint',source_checkpoint,'--controller','AdaRes','--termination-rule','source',
        '--device','cpu',*remaining]
    status=synchronized.main()
    source=synchronized.source
    if source.ACTIVE_ARGS is not None:
        args=source.ACTIVE_ARGS
        path=args.output_dir.resolve()/args.run_id/'result.json'
        result=json.loads(path.read_text())
        result.update(method='CIV-RL',algorithm='CIV-RL-unified-v1',policy=str(policy_path),
            policy_sha256=hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
            deployment_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            inference_device=extra.inference_device,
            policy_scope='The same unified CI candidate and CV selection executor used for paired training',
            policy_process_transport='Local Unix socket carrying public measurements and final thrust commands')
        result.pop('joint_checkpoint',None);result.pop('joint_checkpoint_sha256',None)
        path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return status


if __name__=='__main__':
    transport.__file__=__file__
    transport.load_controller=load_controller
    transport.run_trial=run_trial
    raise SystemExit(transport.main())
