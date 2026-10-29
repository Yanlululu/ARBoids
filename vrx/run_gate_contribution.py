"""Real Gazebo trial with untouched upstream ARBoids policy functions.

The existing runner supplies ROS transport, asset resolution and bounded process
lifecycle only. All original algorithm, observation and outcome functions come
from the pinned author source. New controllers are external additions.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'train'))
import study_runtime
import numpy as np
import torch
import rclpy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from rclpy.qos import qos_profile_sensor_data

import run_experiment as runner
from source_arboids import vrx_source, verify_source, verify_source_config, sha256
from gate_study_control import GateController, METHODS

AUTHOR = vrx_source().controller
MODELS = vrx_source().models
METHOD = 'original'
JOINT_CHECKPOINT = None
ACTIVE_ARGS = None
CBF_PROVENANCE = None


class SourceTrial(runner.Trial):
    get_observations = AUTHOR.ExperimentManager.get_observations
    generate_init_info = AUTHOR.ExperimentManager.generate_init_info

    def __init__(self, args):
        global ACTIVE_ARGS, CBF_PROVENANCE
        if args.controller != 'AdaRes' or args.termination_rule != 'source':
            raise ValueError('The contribution study requires original AdaRes and source termination.')
        if args.synchronize_feedback or args.message_delay_steps or args.message_drop_probability:
            raise ValueError('Modified observation/message handling is not part of this source comparison.')
        if args.run_id is None:
            raise ValueError('Provide --run-id so source provenance is saved beside the exact trial.')
        verify_source_config(Path(args.checkpoint).with_name('config.yaml'))
        AUTHOR.ExperimentManager.__init__(self, args.num_robots, False, device=args.device)
        self.args = ACTIVE_ARGS = args
        self.total_time = args.duration
        # Original constructor retains defend_r=5.5 and all other task constants.
        self.node = rclpy.create_node('arboids_source_trial')
        self.last_stamp = np.full(args.num_robots, np.nan)
        self.transport_stamp = np.full(args.num_robots, np.nan)
        self.last_received = np.zeros(args.num_robots)
        self.yaw_rates = np.zeros(args.num_robots)
        self.collision_seen = False
        self.publishers = [self.node.create_publisher(Float64, f'/wamv{i+1}/thrusters/{side}/thrust', 10)
                           for i in range(args.num_robots) for side in ('left','right')]
        self.subscribers = [self.node.create_subscription(TFMessage, f'/wamv{i+1}/pose', self.on_pose, 10)
                            for i in range(args.num_robots)]
        self.rows, self.frames = [], []
        self.latest_image = None
        if args.capture_frames:
            self.image_subscription = self.node.create_subscription(
                Image, '/arboids/overview/image', self.on_image, qos_profile_sensor_data)
        self.actor = None
        if METHOD != 'joint':
            self.actor = MODELS.ActorAdap(6, 8, 3, 512).to(self.device)
            self.actor.load(args.checkpoint)
            self.actor.eval()
        if METHOD == 'joint':
            from joint_fusion import JointMeasuredController
            self.rule = JointMeasuredController(JOINT_CHECKPOINT,args.checkpoint,inference_device=args.device)
        elif METHOD == 'cbf':
            from cbf_source_baseline import CBFController, dependency_provenance
            self.rule = CBFController(args.num_robots-1)
            n = args.num_robots-1
            sample = np.zeros((n, 6))
            sample[:, 0] = np.arange(n)*20.
            self.rule.control(sample, np.zeros((n, 3), dtype=np.float32), np.zeros((n, 2)))
            CBF_PROVENANCE = dependency_provenance()
        else:
            self.rule = GateController(METHOD)
        # Warm every policy/controller before starting Gazebo. Cold allocation
        # and compilation must not delay only one method's first command.
        n = args.num_robots-1
        dummy_observation = np.zeros((n,14+2*n))
        dummy_boids = np.zeros((n,2))
        if METHOD == 'joint':
            sample=np.zeros((n,6));sample[:,0]=np.arange(n)*20.
            target=np.array([120.,30.,np.pi,0.,0.,0.])
            boids,states=AUTHOR.Boids_navi_control(sample[:,:2],sample[:,3:5],sample[:,2],target[:2])
            observations=self.get_observations(sample[:,:2],sample[:,2],target[:2],target[3:5],states,boids)
            self.rule.control(sample,target,observations,boids,0.,self.total_time)
            self.rule.reset()
        elif METHOD == 'original':
            AUTHOR.RL_navi_control(self.actor,dummy_observation,dummy_boids,'AdaRes',self.device)
        else:
            with torch.no_grad():
                raw,_ = self.actor(torch.tensor(dummy_observation,dtype=torch.float,device=self.device),True,False)
            sample = np.zeros((n,6))
            sample[:,0] = np.arange(n)*20.
            self.rule.control(sample,raw.cpu().numpy(),dummy_boids)

    def on_pose(self, message):
        previous_stamp, previous_yaw = self.transport_stamp.copy(), self.curr_phi.copy()
        # Preserve the author's original delayed finite-difference velocity
        # calculation (including its fixed 0.05-second denominator).
        AUTHOR.ExperimentManager.robot_info_callback(self, message)
        for tf in message.transforms:
            name = tf.child_frame_id
            if name not in self.pose_data:
                continue
            i = int(name.removeprefix('wamv'))-1
            stamp = tf.header.stamp.sec+tf.header.stamp.nanosec*1e-9
            if np.isfinite(previous_stamp[i]) and stamp > previous_stamp[i]:
                angle = np.arctan2(np.sin(self.curr_phi[i]-previous_yaw[i]),
                                   np.cos(self.curr_phi[i]-previous_yaw[i]))
                self.yaw_rates[i] = angle/(stamp-previous_stamp[i])
            self.transport_stamp[i] = stamp
            # Identical passive settling interval for every method. It also
            # prevents camera startup from changing the compared initial time.
            if stamp >= 2.-1e-8:
                self.last_stamp[i] = stamp
            self.last_received[i] = time.monotonic()

    def outcome(self, elapsed):
        # Refresh the same geometric bookkeeping used by the original check.
        # The decision function itself is the byte-identical author function.
        defenders = self.curr_pos[1:]
        n = len(defenders)
        self.def_att_dists = np.linalg.norm(defenders-self.curr_pos[0], axis=-1)
        d = np.linalg.norm(defenders[:, None]-defenders[None], axis=-1)
        self.def_def_dists = d[~np.eye(n, dtype=bool)].reshape(n, n-1)
        self.curr_time = elapsed
        return AUTHOR.ExperimentManager.check_is_terminated(self)

    def control(self, elapsed):
        started = time.monotonic()
        limits = np.array([-500.,1000.])*self.agility
        actions = np.zeros((self.num_robots, 2))
        actions[0] = AUTHOR.APF_navi_control(self.curr_pos[0], np.zeros(2), self.curr_pos[1:],
                                            self.curr_phi[0], *limits)
        boids, states = AUTHOR.Boids_navi_control(self.curr_pos[1:], self.curr_vel[1:],
                                                 self.curr_phi[1:], self.curr_pos[0])
        observations = self.get_observations(self.curr_pos[1:], self.curr_phi[1:],
                                            self.curr_pos[0], self.curr_vel[0], states, boids)
        inference_started = time.monotonic()
        if METHOD == 'original':
            actions[1:] = AUTHOR.RL_navi_control(self.actor, observations, boids, 'AdaRes', self.device)
            theta = np.full(self.num_robots-1, np.nan)
            info = {'warning': False}
        elif METHOD == 'joint':
            snapshot=np.column_stack((self.curr_pos,self.curr_phi,self.curr_vel,self.yaw_rates))
            command,info=self.rule.control(snapshot[1:],snapshot[0],observations,boids,elapsed,self.total_time)
            theta=info['physical_action'][:,2]
            actions[1:]=np.clip(command,-500.,1000.)
            info['warning']=False
        else:
            with torch.no_grad():
                raw, _ = self.actor(torch.tensor(observations, dtype=torch.float, device=self.device), True, False)
            raw = raw.cpu().numpy()
            snapshot = np.column_stack((self.curr_pos[1:], self.curr_phi[1:],
                                        self.curr_vel[1:], self.yaw_rates[1:]))
            corrected, command, info = self.rule.control(snapshot, raw, boids)
            theta = corrected[:, 2]
            actions[1:] = np.clip(command, -500.,1000.)
        inference_seconds = time.monotonic()-inference_started
        if not all(np.isfinite(x).all() for x in (actions, self.curr_pos, self.curr_vel)):
            raise FloatingPointError('Non-finite source VRX state or control.')
        # Common physical ROS bridge, identical for all methods. Training uses
        # positive yaw for action[0] > action[1]; Gazebo uses ENU port/starboard.
        physical = actions[:, ::-1]
        for publisher, thrust in zip(self.publishers, physical.flat):
            publisher.publish(Float64(data=float(thrust)))
        self.rows.append(dict(timestamp=float(elapsed), AttPos=self.curr_pos[0].copy(),
            AttPhi=self.curr_phi[0].copy(), AttVel=self.curr_vel[0].copy(), AttAct=actions[0].copy(),
            DefPos=self.curr_pos[1:].flatten().copy(), DefPhi=self.curr_phi[1:].copy(),
            DefVel=self.curr_vel[1:].flatten().copy(), DefAct=actions[1:].flatten().copy(),
            PhysicalThrust=physical.flatten().copy(), GateTheta=theta.copy(),
            SimulationStamp=float(np.min(self.last_stamp)),
            GateWarning=bool(info['warning']),
            ConditionalInterventions=int(info.get('conditional_interventions',0)),
            CBFViolation=float(info.get('cbf_constraint_violation',0.)),
            CBFSaturation=float(info.get('cbf_control_saturation',0.)),
            ControlWallSeconds=time.monotonic()-started, PolicyWallSeconds=inference_seconds,
            FeedbackSkewSeconds=float(np.ptp(self.last_stamp))))


def main():
    global METHOD,JOINT_CHECKPOINT
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--method', choices=(*METHODS,'joint'), default='original')
    parser.add_argument('--joint-checkpoint',type=Path)
    extra, remaining = parser.parse_known_args()
    METHOD = extra.method
    JOINT_CHECKPOINT = extra.joint_checkpoint
    if METHOD=='joint' and JOINT_CHECKPOINT is None:parser.error('joint requires --joint-checkpoint')
    if '--duration' not in remaining:
        remaining += ['--duration','80' if METHOD=='joint' else '100']
    source = verify_source()
    code_files = [Path(__file__), Path(__file__).resolve().parents[1]/'train/gate_study_control.py',
                  Path(__file__).resolve().parents[1]/'train/source_arboids.py',
                  Path(__file__).resolve().parents[1]/'train/cbf_source_baseline.py']
    if METHOD=='joint':
        code_files.extend(Path(__file__).resolve().parents[1]/'train'/name
            for name in ('joint_fusion.py','actuator_fusion.py','mechanism_controls.py'))
    hashes = {p.name:sha256(p) for p in code_files}
    sys.argv = [sys.argv[0], *remaining]
    runner.Trial = SourceTrial
    status = runner.main()
    if ACTIVE_ARGS is not None:
        output = ACTIVE_ARGS.output_dir.resolve()/ACTIVE_ARGS.run_id/'result.json'
        result = json.loads(output.read_text())
        unchanged = verify_source()==source and hashes=={p.name:sha256(p) for p in code_files}
        result.update(method=METHOD, source=source, study_code=hashes, source_unchanged=unchanged,
                      capture_radius=5.5, original_observation_padding=True,
                      original_velocity_callback=True,
                      common_settle_seconds=2.,
                      policy_scope='Untouched author policy/Boids/APF/observation/outcome; shared ROS transport adapter',
                      external_baseline=CBF_PROVENANCE)
        if METHOD=='joint':
            result.update(joint_checkpoint=str(JOINT_CHECKPOINT),joint_checkpoint_sha256=sha256(JOINT_CHECKPOINT),
                policy_scope='Mechanism-one joint actuator fusion; shared numerical/ROS command decoder',
                inference_device=ACTIVE_ARGS.device, original_observation_padding=False,
                velocity_model='Measured ground velocity, zero-current nominal prediction')
        if not unchanged:
            result.update(passed=False, error='Source changed during trial.')
        output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(f'[SOURCE VERIFIED] method={METHOD} passed={result["passed"]} '
              f'outcome={result.get("outcome")}', flush=True)
        status = 0 if result['passed'] else 1
    return status


if __name__=='__main__':
    raise SystemExit(main())
