"""Run one bounded ARBoids experiment against a real ROS 2 / Gazebo world."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import struct
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import rclpy
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
import torch

from models import ActorAdap, ActorSAC
from tad_vrx_experiment import (
    ExperimentManager, APF_navi_control, Boids_navi_control, RL_navi_control,
)


class Trial(ExperimentManager):
    def __init__(self, args):
        super().__init__(args.num_robots, False, device=args.device)
        self.args = args
        self.total_time = args.duration
        self.node = rclpy.create_node('arboids_trial')
        self.last_stamp = np.full(args.num_robots, np.nan)
        self.last_received = np.zeros(args.num_robots)
        self.publishers = [
            self.node.create_publisher(Float64, f'/wamv{i + 1}/thrusters/{side}/thrust', 10)
            for i in range(args.num_robots) for side in ('left', 'right')
        ]
        self.subscribers = [
            self.node.create_subscription(TFMessage, f'/wamv{i + 1}/pose', self.on_pose, 10)
            for i in range(args.num_robots)
        ]
        self.rows = []
        self.latest_image = None
        self.frames = []
        if args.capture_frames:
            self.image_subscription = self.node.create_subscription(
                Image, '/arboids/overview/image', self.on_image, qos_profile_sensor_data)
        self.actor = None
        if args.controller != 'Boids':
            actor_type = ActorAdap if args.controller == 'AdaRes' else ActorSAC
            action_dim = 3 if args.controller == 'AdaRes' else 2
            self.actor = actor_type(6, 8, action_dim, hidden_dim=512).to(self.device)
            self.actor.load(args.checkpoint)
            self.actor.eval()

    def on_image(self, message):
        self.latest_image = message

    def save_frame(self, output, start_sim):
        message = self.latest_image
        if message is None:
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if self.frames and stamp <= self.frames[-1]['gazebo_timestamp']:
            return
        channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(message.encoding)
        if channels is None:
            raise RuntimeError(f'Unsupported camera encoding: {message.encoding}')
        pixels = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        pixels = pixels[:, :message.width*channels].reshape(message.height, message.width, channels)[:, :, :3]
        if message.encoding.startswith('bgr'):
            pixels = pixels[:, :, ::-1]
        if np.ptp(pixels) < 10:
            return
        def chunk(tag, payload):
            return struct.pack('!I', len(payload)) + tag + payload + struct.pack('!I', zlib.crc32(tag+payload) & 0xffffffff)
        raw = b''.join(b'\0' + row.tobytes() for row in pixels)
        png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', message.width, message.height, 8, 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress(raw, 3)) + chunk(b'IEND', b''))
        path = output / f'frame-{len(self.frames):03d}.png'
        path.write_bytes(png)
        self.frames.append({'file': path.name, 'simulation_seconds': stamp-start_sim, 'gazebo_timestamp': stamp})

    def on_pose(self, message):
        for tf in message.transforms:
            name = tf.child_frame_id
            if name not in self.pose_data:
                continue
            index = int(name.removeprefix('wamv')) - 1
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            if np.isfinite(self.last_stamp[index]) and stamp <= self.last_stamp[index]:
                continue
            xyz, q = tf.transform.translation, tf.transform.rotation
            position = np.array([xyz.x, xyz.y]) - self.origin
            if np.isfinite(self.last_stamp[index]):
                self.curr_vel[index] = (position - self.curr_pos[index]) / (stamp - self.last_stamp[index])
            self.curr_pos[index] = position
            self.curr_phi[index] = np.arctan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
            self.last_stamp[index] = stamp
            self.last_received[index] = time.monotonic()

    def outcome(self, elapsed):
        if np.linalg.norm(self.curr_pos[0]) < self.target_r:
            return 1
        defenders = self.curr_pos[1:]
        distances = np.linalg.norm(defenders[:, None] - defenders[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if np.any(distances < self.collision_r):
            return 2
        if np.any(np.linalg.norm(defenders - self.curr_pos[0], axis=1) < self.defend_r):
            return 3
        return 4 if elapsed >= self.total_time else 0

    def control(self, elapsed):
        actions = np.zeros((self.num_robots, 2))
        limits = np.array([-500., 1000.]) * self.agility
        actions[0] = APF_navi_control(self.curr_pos[0], np.zeros(2), self.curr_pos[1:],
                                      self.curr_phi[0], *limits)
        boids, states = Boids_navi_control(self.curr_pos[1:], self.curr_vel[1:],
                                          self.curr_phi[1:], self.curr_pos[0])
        if self.args.controller == 'Boids':
            actions[1:] = boids
        else:
            observation = self.get_observations(self.curr_pos[1:], self.curr_phi[1:],
                                                self.curr_pos[0], self.curr_vel[0], states, boids)
            actions[1:] = RL_navi_control(self.actor, observation, boids,
                                          self.args.controller, self.device)
        if not all(np.isfinite(x).all() for x in (actions, self.curr_pos, self.curr_vel)):
            raise FloatingPointError('Non-finite state or thrust command')
        # The training dynamics apply positive yaw for action[0] > action[1].
        # Gazebo's port thruster is at +y and produces negative ENU yaw, so
        # map training action[0] to starboard and action[1] to port.
        physical_actions = actions[:, ::-1]
        for publisher, thrust in zip(self.publishers, physical_actions.flat):
            publisher.publish(Float64(data=float(thrust)))
        self.rows.append({
            'timestamp': float(elapsed), 'AttPos': self.curr_pos[0].copy(),
            'AttPhi': self.curr_phi[0].copy(), 'AttVel': self.curr_vel[0].copy(),
            'AttAct': actions[0].copy(), 'DefPos': self.curr_pos[1:].flatten().copy(),
            'DefPhi': self.curr_phi[1:].copy(), 'DefVel': self.curr_vel[1:].flatten().copy(),
            'DefAct': actions[1:].flatten().copy(),
            'PhysicalThrust': physical_actions.flatten().copy(),
        })


def stop_process(process):
    """Stop the whole owned group, including children surviving ROS launch."""
    if process is None:
        return
    def group_alive():
        process.poll()
        for stat in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = stat.read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == process.pid and fields[0] != 'Z':
                    return True
            except (FileNotFoundError, ProcessLookupError):
                continue
        return False
    for sig, limit in ((signal.SIGINT, 8), (signal.SIGTERM, 5), (signal.SIGKILL, 3)):
        if not group_alive():
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + limit
        while group_alive() and time.monotonic() < deadline:
            time.sleep(.1)
    process.wait(timeout=5)
    if group_alive():
        raise RuntimeError(f'Gazebo process group {process.pid} did not stop')


def prepare_world(world, origin, output, capture_frames):
    source = Path(get_package_share_directory('vrx_gz')) / 'worlds' / f'{world}.sdf'
    tree = ET.parse(source)
    assets = Path(__file__).resolve().parents[1] / '.vrx-assets'
    manifest = json.loads((assets / 'fuel-manifest.json').read_text(encoding='utf-8'))
    models = {entry['url']: assets / 'resolved' / entry['directory'] for entry in manifest}
    for uri in tree.findall('.//include/uri'):
        if uri.text and uri.text.strip().startswith('https://fuel.gazebosim.org/'):
            model_path = models.get(uri.text.strip())
            if model_path is None or not (model_path / 'model.config').is_file():
                raise FileNotFoundError(f'Run scripts/fetch_fuel_assets.py for {uri.text}')
            # Resolve locally so evaluation never waits on Fuel network requests.
            uri.text = str(model_path)
    path = output / 'world.sdf'
    if not capture_frames:
        tree.write(path, encoding='utf-8', xml_declaration=True)
        return str(path)
    model = ET.SubElement(tree.getroot().find('world'), 'model', name='arboids_overview')
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = f'{origin[0]} {origin[1]} 120 0 1.5707963267948966 0'
    link = ET.SubElement(model, 'link', name='camera_link')
    sensor = ET.SubElement(link, 'sensor', name='overview', type='camera')
    ET.SubElement(sensor, 'always_on').text = 'true'
    ET.SubElement(sensor, 'update_rate').text = '1'
    ET.SubElement(sensor, 'topic').text = '/arboids/overview/image'
    camera = ET.SubElement(sensor, 'camera')
    ET.SubElement(camera, 'horizontal_fov').text = '1.4'
    image = ET.SubElement(camera, 'image')
    for name, value in (('width', '1280'), ('height', '960'), ('format', 'R8G8B8')):
        ET.SubElement(image, name).text = value
    clip = ET.SubElement(camera, 'clip')
    ET.SubElement(clip, 'near').text = '0.1'
    ET.SubElement(clip, 'far').text = '1000'
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', '--modelname', dest='checkpoint')
    parser.add_argument('--controller', choices=['AdaRes', 'Res', 'RL', 'Boids'], default='AdaRes')
    parser.add_argument('--setting', type=int, choices=[0, 1], default=1)
    parser.add_argument('--num-robots', '--num_robots', dest='num_robots', type=int, default=4)
    parser.add_argument('--agility', type=float, default=2.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--capture-frames', action='store_true', help='Save real Gazebo overview camera frames every five simulation seconds')
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--action-period', type=float, default=0.2)
    parser.add_argument('--startup-timeout', type=float, default=180.)
    parser.add_argument('--wall-timeout', type=float, default=900.)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--save_traj', action='store_true', help='Accepted for compatibility; trajectories are always saved')
    parser.add_argument('--save_file', type=Path, default=None)
    args = parser.parse_args()
    if args.num_robots < 3 or min(args.duration, args.action_period, args.agility) <= 0:
        parser.error('Use at least three robots and positive duration, action period and agility')
    if args.controller != 'Boids':
        if not args.checkpoint or not Path(args.checkpoint).is_file():
            parser.error('A trained policy checkpoint is required')
        args.checkpoint = str(Path(args.checkpoint).resolve())
    run_id = args.run_id or f'{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}'
    output = args.output_dir.resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    result = {'passed': False, 'seed': args.seed, 'setting': args.setting,
              'agility': args.agility, 'controller': args.controller, 'num_robots': args.num_robots,
              'duration_limit': args.duration, 'action_period': args.action_period,
              'checkpoint': args.checkpoint,
              'thruster_mapping': 'policy[0]->starboard; policy[1]->port (ENU yaw matching training)'}
    if args.checkpoint:
        result['checkpoint_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    process = image_bridge = trial = None
    started = time.monotonic()
    rclpy.init()
    try:
        trial = Trial(args)
        poses = trial.generate_init_info(args.agility, args.setting)
        result['initial_poses'] = poses
        world = 'sydney_regatta_original' + ('1' if args.setting == 1 else '')
        world = prepare_world(world, trial.origin, output, args.capture_frames)
        command = ['ros2', 'launch', 'vrx_gz', 'tad.launch.py', f'init_poses:={poses}',
                   f'world:={world}', f'headless:={str(args.headless).lower()}']
        if args.headless:
            command += ['extra_gz_args:=--headless-rendering']
        launch_env = dict(os.environ)
        launch_env['GZ_PARTITION'] = f'arboids-{os.getpid()}-{uuid.uuid4().hex[:8]}'
        result['gazebo_partition'] = launch_env['GZ_PARTITION']
        if preload := launch_env.get('ARBOIDS_OGRE_PRELOAD'):
            launch_env['LD_PRELOAD'] = preload + (':' + launch_env['LD_PRELOAD'] if launch_env.get('LD_PRELOAD') else '')
        with (output / 'gazebo.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, env=launch_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if args.capture_frames:
                image_bridge = subprocess.Popen(
                    ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge', '/arboids/overview/image@sensor_msgs/msg/Image[gz.msgs.Image'],
                    env=launch_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while not (np.isfinite(trial.last_stamp).all() and
                       all(p.get_subscription_count() > 0 for p in trial.publishers)):
                if process.poll() is not None:
                    raise RuntimeError(f'Gazebo launch exited ({process.returncode}); see gazebo.log')
                if time.monotonic() - started > args.startup_timeout:
                    raise TimeoutError('Waiting for all vessel poses and thrust bridge subscribers')
                rclpy.spin_once(trial.node, timeout_sec=0.05)
            start_sim = float(np.min(trial.last_stamp))
            next_action = 0.
            next_frame = 0.
            print(f'[READY] all {args.num_robots} vessels and thrust bridges at t={start_sim:.3f}', flush=True)
            while True:
                rclpy.spin_once(trial.node, timeout_sec=0.01)
                if process.poll() is not None:
                    raise RuntimeError('Gazebo exited before a valid task outcome')
                if time.monotonic() - started > args.wall_timeout:
                    raise TimeoutError('Trial exceeded wall-clock limit')
                if np.any(time.monotonic() - trial.last_received > 15.):
                    raise TimeoutError('A vessel stopped publishing pose feedback')
                elapsed = float(np.min(trial.last_stamp) - start_sim)
                code = trial.outcome(elapsed)
                if code:
                    if args.capture_frames:
                        trial.save_frame(output, start_sim)
                        if not trial.frames:
                            raise RuntimeError('No valid overview camera frames received')
                    if not trial.rows:
                        raise RuntimeError('Task ended before the controller produced a command')
                    attacker_positions = np.array([row['AttPos'] for row in trial.rows])
                    distance = float(np.linalg.norm(attacker_positions - attacker_positions[0], axis=1).max())
                    if distance < 1.0:
                        raise RuntimeError('No meaningful attacker motion; verify thrust delivery')
                    labels = {1: 'attacker_reached_target', 2: 'defender_collision',
                              3: 'defender_capture', 4: 'defended_until_timeout'}
                    result.update(passed=True, outcome_code=code, outcome=labels[code],
                                  success=code > 2, simulation_seconds=elapsed,
                                  attacker_max_displacement=distance)
                    break
                if elapsed + 1e-6 >= next_action:
                    trial.control(elapsed)
                    next_action = elapsed + args.action_period
                    if len(trial.rows) % 25 == 0:
                        print(f'[CONTROL] t={elapsed:.2f}s commands={len(trial.rows)}', flush=True)
                if args.capture_frames and elapsed >= next_frame:
                    trial.save_frame(output, start_sim)
                    next_frame = elapsed + 5.
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
    finally:
        stop_process(image_bridge)
        stop_process(process)
        log_path = output / 'gazebo.log'
        if log_path.exists():
            log_text = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text(encoding='utf-8', errors='replace'))
            diagnostics = [line for line in log_text.splitlines()
                           if any(message in line for message in ('[Err]', 'Segmentation fault', 'Assertion'))]
            if diagnostics:
                result.update(passed=False, error='Gazebo reported errors; see gazebo.log',
                              gazebo_errors=diagnostics[:20])
        result['wall_seconds'] = time.monotonic() - started
        if trial is not None:
            result['control_steps'] = len(trial.rows)
            result['frames'] = trial.frames
            if trial.rows:
                arrays = {key: np.asarray([row[key] for row in trial.rows]) for key in trial.rows[0]}
                path = args.save_file or output / 'trajectory.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                result['trajectory'] = str(path.resolve())
            trial.node.destroy_node()
            trial.unpause_signal_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
