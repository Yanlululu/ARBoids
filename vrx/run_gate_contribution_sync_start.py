"""Transport-only startup synchronization for original-source VRX controllers.

All models are inserted into a paused world before physics advances. This
removes method-dependent model insertion timing, without changing source
policy functions, observations, integration, or termination criteria.
"""
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import run_gate_contribution as source
from source_arboids import sha256


class SynchronizedStartTrial(source.SourceTrial):
    def additional_launch_arguments(self):
        self.world_started=False
        return ['paused:=true']

    def startup_tick(self,output,launch_env):
        if self.world_started:
            return
        log=(output/'gazebo.log').read_text(encoding='utf-8',errors='replace')
        # ros_gz_sim reports success only after the model has been inserted.
        spawned=len(re.findall(r'OK creation of entity\.',log))
        if spawned<self.num_robots or not all(p.get_subscription_count()>0 for p in self.publishers):
            return
        world=ET.parse(output/'world.sdf').getroot().find('world').attrib['name']
        command=['gz','service','-s',f'/world/{world}/control','--reqtype','gz.msgs.WorldControl',
                 '--reptype','gz.msgs.Boolean','--timeout','5000','--req','pause: false']
        result=subprocess.run(command,env=launch_env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=10)
        if result.returncode or 'data: true' not in result.stdout:
            raise RuntimeError(f'Could not start the fully inserted world: {result.stdout.strip()}')
        self.world_started=True
        print(f'[SYNCHRONIZED START] {spawned} models inserted before physics advances',flush=True)


def main():
    source.SourceTrial=SynchronizedStartTrial
    status=source.main()
    if source.ACTIVE_ARGS is not None:
        args=source.ACTIVE_ARGS
        path=args.output_dir.resolve()/args.run_id/'result.json'
        result=json.loads(path.read_text())
        result.update(synchronized_model_start=True,
                      startup_adapter_sha256=sha256(Path(__file__)),
                      shared_transport_sha256=sha256(Path(source.runner.__file__)))
        path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return status


if __name__=='__main__':
    raise SystemExit(main())
