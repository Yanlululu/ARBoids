"""Run one bounded ARBoids experiment against a real ROS 2 / Gazebo world."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import uuid

import numpy as np
import rclpy
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage
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
        self.actor = None
        if args.controller != 'Boids':
            actor_type = ActorAdap if args.controller == 'AdaRes' else ActorSAC
            action_dim = 3 if args.controller == 'AdaRes' else 2
            self.actor = actor_type(6, 8, action_dim, hidden_dim=512).to(self.device)
            self.actor.load(args.checkpoint)
            self.actor.eval()

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
        for publisher, thrust in zip(self.publishers, actions.flat):
            publisher.publish(Float64(data=float(thrust)))
        self.rows.append({
            'timestamp': float(elapsed), 'AttPos': self.curr_pos[0].copy(),
            'AttPhi': self.curr_phi[0].copy(), 'AttVel': self.curr_vel[0].copy(),
            'AttAct': actions[0].copy(), 'DefPos': self.curr_pos[1:].flatten().copy(),
            'DefPhi': self.curr_phi[1:].copy(), 'DefVel': self.curr_vel[1:].flatten().copy(),
            'DefAct': actions[1:].flatten().copy(),
        })


def stop_process(process):
    """Stop only the launch process group created by this trial."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


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
              'checkpoint': args.checkpoint}
    if args.checkpoint:
        result['checkpoint_sha256'] = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    process = trial = None
    started = time.monotonic()
    rclpy.init()
    try:
        trial = Trial(args)
        poses = trial.generate_init_info(args.agility, args.setting)
        result['initial_poses'] = poses
        world = 'sydney_regatta_original' + ('1' if args.setting == 1 else '')
        command = ['ros2', 'launch', 'vrx_gz', 'tad.launch.py', f'init_poses:={poses}',
                   f'world:={world}', f'headless:={str(args.headless).lower()}']
        if args.headless:
            command += ['extra_gz_args:=--headless-rendering']
        with (output / 'gazebo.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while not (np.isfinite(trial.last_stamp).all() and
                       all(p.get_subscription_count() > 0 for p in trial.publishers)):
                if process.poll() is not None:
                    raise RuntimeError(f'Gazebo launch exited ({process.returncode}); see gazebo.log')
                if time.monotonic() - started > args.startup_timeout:
                    raise TimeoutError('Waiting for all vessel poses and thrust bridge subscribers')
                rclpy.spin_once(trial.node, timeout_sec=0.05)
            start_sim = float(np.min(trial.last_stamp))
            next_action = 0.
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
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
    finally:
        stop_process(process)
        result['wall_seconds'] = time.monotonic() - started
        if trial is not None:
            result['control_steps'] = len(trial.rows)
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
