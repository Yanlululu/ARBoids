"""Measured-state VRX deployment of the final unified CITF task and policies."""
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import os
import re
import subprocess
import sys
import time
import math
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import torch
from std_msgs.msg import Float64

sys.path.insert(0, os.environ.get('CITF_CODE_ROOT', str(Path(__file__).resolve().parents[1]/'citf'/'code')))
from unified import IMPLEMENTATION_VERSION
from unified.environment import FleetEnv, TaskConfig, execute
from unified.networks import CITF, ModelConfig
from unified.baselines import PublicBaseline
from unified.train import seed_policy
from unified.protocol import (load_registry, validate_checkpoint, checkpoint_identity,
                              evaluation_identity, vrx_source_identity)
from run_experiment import Trial
from mappo_controller import synchronize_kinematics


class UnifiedTrial(Trial):
    def __init__(self, args):
        if (args.num_robots != 4 or args.duration != 60. or args.action_period != .2
                or args.agility != 2.25 or args.setting != 0 or args.termination_rule != 'paper'
                or args.scene_index < 0 or args.message_delay_steps or args.message_drop_probability):
            raise ValueError('Unified VRX uses 3 defenders, 60 s, 0.2 s control, agility 2.25, open-water setting 0 and paper events.')
        # Reuse ROS transport and measured-pose event handling, not legacy policies.
        transport_args = Namespace(**{**vars(args), 'controller': 'Boids'})
        super().__init__(transport_args)
        self.args = args
        torch.set_float32_matmul_precision('high')
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if payload.get('implementation_version') != IMPLEMENTATION_VERSION:
            raise ValueError('Use the matching unified checkpoint implementation.')
        if args.research_stage == 'formal' and (payload['arguments']['stage'] != 'formal'
                or payload['transitions'] != 1000000 or payload['arguments']['steps'] != 1000000
                or payload['arguments']['seed'] != 2101 or args.scene_seed != 74002
                or args.policy_seed != 84002 or args.scene_index not in range(100)):
            raise ValueError('Formal VRX uses the registered fixed-budget final main-training checkpoint.')
        registry = None
        if args.research_stage == 'formal':
            if not args.registry:
                raise ValueError('Formal VRX requires the common study registry.')
            registry = load_registry(args.registry)
        method = 'full' if args.controller == 'CITF' else 'arboids'
        validate_checkpoint(payload, registry, method, payload['arguments']['seed'],
                            1000000 if registry is not None else None)
        self.checkpoint_identity = checkpoint_identity(args.checkpoint, payload)
        self.adapter_source = vrx_source_identity(Path(__file__).parent)
        task = replace(TaskConfig(**payload['task']), agility=args.agility)
        self.result_identity = (None if registry is None else
            evaluation_identity(registry, method, payload['arguments']['seed'], self.checkpoint_identity,
                                asdict(task), args.scene_seed, args.policy_seed, 100, kind='vrx'))
        if task.defenders != 3 or task.duration != 60. or task.control_dt != .2:
            raise ValueError('The VRX checkpoint must use the common three-vessel main task.')
        if args.controller == 'CITF':
            cfg = ModelConfig(**payload['config'])
            if not cfg.action_interaction or cfg.feedback <= 0:
                raise ValueError('VRX compares the complete method and ARBoids only.')
            self.policy = CITF(cfg)
        else:
            if payload.get('method') != 'arboids':
                raise ValueError('ARBoidsUnified requires its trained public baseline checkpoint.')
            self.policy = PublicBaseline('arboids', payload['width'], task.thrust_delta)
        self.policy.to(self.device).load_state_dict(payload['model'])
        self.policy.eval()
        self.model_seed, self.model_steps = payload['arguments']['seed'], payload['transitions']
        del payload
        # Keep opponent dynamics/command bookkeeping off the inference device.
        bank = FleetEnv(args.scene_index+1, task, seed=args.scene_seed, device='cpu')
        self.common = FleetEnv(1, task, seed=args.scene_seed, device='cpu')
        for name in ('state', 'order', 'noise', 'agility'):
            getattr(self.common, name)[0].copy_(getattr(bank, name)[args.scene_index])
        del bank
        self.initial_state = self.common.state[0].detach().cpu().numpy().copy()
        self.applied = torch.zeros(1, 4, 2)
        self.agility = args.agility
        self.world_started = False
        # Allocate inference buffers before physics starts, then restore sampling.
        with torch.no_grad():
            self.policy.select(self.common.observation().to(self.device))
        seed_policy(args.policy_seed+args.scene_index, self.device)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='unified-defender')
        self.pending = None
        self.episode_id = uuid.uuid4().hex
        self.request_id = 0
        self.last_control_cycle = -1
        self.controller_closed = False

    def protocol_metadata(self):
        return dict(research_stage=self.args.research_stage, implementation_version=IMPLEMENTATION_VERSION,
                    identity=self.result_identity, checkpoint_metadata=self.checkpoint_identity,
                    adapter_source=self.adapter_source,
                    code_directory=str(Path(sys.modules['unified'].__file__).resolve().parents[1]),
                    scene_seed=self.args.scene_seed, scene_index=self.args.scene_index,
                    policy_seed=self.args.policy_seed+self.args.scene_index,
                    training_seed=self.model_seed, training_transitions=self.model_steps,
                    task=asdict(self.common.cfg),
                    matmul_precision=torch.get_float32_matmul_precision(),
                    apf_obstacle_rule='unified 15 m center-distance support',
                    synchronize_feedback=True, communication_scope='measured state; no impairment model',
                    thrust_increment=self.common.cfg.thrust_delta,
                    deadline_policy='one asynchronous defender query per cycle; discard late output and hold defender commands',
                    control_clocks='monotonic wall time for computation; simulation time for task and state age; either deadline expires a request',
                    request_identity='episode, request, control cycle, source simulation timestamp, wall and simulation deadlines',
                    terminal_detection='consume every 0.05 s observed simulation state in order; latch first event; reject missing intervals',
                    control_denominator='all scheduled simulation-time cycles, including unsubmitted cycles',
                    attacker_timing='APF advances each control cycle independently of defender inference',
                    synchronized_model_start=True)

    def generate_init_info(self, agility=2.25, setting=0):
        # Gazebo numbers attacker first; the unified controller numbers it last.
        physical = self.initial_state[[3, 0, 1, 2]]
        self.curr_pos[:] = physical[:, :2]
        self.curr_phi[:] = physical[:, 2]
        return ';'.join(f'{row[0]+self.origin[0]},{row[1]+self.origin[1]},{row[2]}' for row in physical)

    def additional_launch_arguments(self):
        return ['paused:=true']

    def startup_tick(self, output, launch_env):
        if self.world_started:
            return
        log = (output/'gazebo.log').read_text(encoding='utf-8', errors='replace')
        if len(re.findall(r'OK creation of entity\.', log)) < self.num_robots:
            return
        if not all(publisher.get_subscription_count() > 0 for publisher in self.publishers):
            return
        world = ET.parse(output/'world.sdf').getroot().find('world').attrib['name']
        command = ['gz', 'service', '-s', f'/world/{world}/control', '--reqtype', 'gz.msgs.WorldControl',
                   '--reptype', 'gz.msgs.Boolean', '--timeout', '5000', '--req', 'pause: false']
        result = subprocess.run(command, env=launch_env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=10)
        if result.returncode or 'data: true' not in result.stdout:
            raise RuntimeError(f'Could not start the fully inserted world: {result.stdout.strip()}')
        self.world_started = True

    def publish_commands(self, indices):
        commands = self.applied[0].detach().cpu().numpy()[[3, 0, 1, 2]][:, ::-1]
        if self.controller_closed or not np.isfinite(commands).all():
            raise RuntimeError('A closed controller or non-finite command cannot publish thrust.')
        for index in indices:
            for side in range(2):
                self.publishers[2*index+side].publish(Float64(data=float(commands[index, side])))

    def compute_defense(self, x, order, identity):
        started = time.monotonic()
        with torch.no_grad():
            ordered = self.policy.select(x.to(self.device))['action']
            physical = torch.zeros_like(ordered)
            physical.scatter_(1, order.to(ordered.device)[..., None].expand(-1, -1, 2), ordered)
            physical = physical.cpu()
        if not bool(torch.isfinite(physical).all()):
            raise FloatingPointError('Non-finite unified VRX defender command.')
        finished = time.monotonic()
        return physical, finished-started, finished, identity

    def poll_control(self, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid simulation time for a control result.')
        if self.pending is None:
            return
        pending = self.pending
        row = self.rows[pending['row']]
        identity = pending['identity']
        if identity['episode'] == self.episode_id and elapsed < pending['elapsed']-1e-9:
            raise ValueError('Simulation time moved backwards within the active control request.')
        expired = (time.monotonic() > identity['wall_deadline']
                   or elapsed > identity['sim_deadline']+1e-6)
        if expired:
            pending['expired'] = True
            row['MissedDeadline'] = 1
        if not pending['future'].done():
            return
        physical, inference_seconds, finished, returned_identity = pending['future'].result()
        row['PolicyWallSeconds'] = inference_seconds
        row['ControlWallSeconds'] = time.monotonic()-pending['started']
        late = pending['expired'] or finished > identity['wall_deadline']
        current = (not self.controller_closed and identity['episode'] == self.episode_id
                   and returned_identity == identity and identity['request'] == row['RequestId']
                   and identity['cycle'] == row['ControlCycle'])
        row['DiscardedLateOutput'] = int(late)
        row['DiscardedStaleOutput'] = int(not current)
        row['MissedDeadline'] = int(late)
        if not late and current:
            if not bool(torch.isfinite(physical).all()):
                raise FloatingPointError('Non-finite completed defender command.')
            with torch.no_grad():
                self.applied[:, :-1].copy_(execute(physical, self.applied[:, :-1], self.common.cfg.thrust_delta))
            self.publish_commands(range(1, 4))
            actions = self.applied[0].detach().cpu().numpy()[[3, 0, 1, 2]].copy()
            row['DefAct'] = actions[1:].flatten()
            row['PhysicalThrust'] = actions[:, ::-1].flatten()
            row['AppliedTimestamp'] = float(elapsed)
            row['DefenderOutputApplied'] = 1
        self.pending = None

    def close_controller(self, terminal_elapsed=None):
        # Running futures may refuse cancellation. Revoke execution eligibility
        # before waiting for their completion, then retain their timing only.
        self.controller_closed = True
        closed_at = time.monotonic()
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.pending is None:
            return
        pending = self.pending
        if pending['future'].cancelled():
            self.rows[pending['row']]['PolicyStarted'] = 0
            self.rows[pending['row']]['TerminalCancellation'] = 1
            self.pending = None
            return
        _, inference_seconds, finished, _ = pending['future'].result()
        row = self.rows[pending['row']]
        row['PolicyWallSeconds'] = inference_seconds
        row['ControlWallSeconds'] = finished-pending['started']
        row['TerminalCancellation'] = 1
        # No commands are published after the observed terminal event.
        if (closed_at > pending['identity']['wall_deadline']
                or terminal_elapsed is not None and terminal_elapsed > pending['identity']['sim_deadline']+1e-6):
            row['MissedDeadline'] = 1
            row['DiscardedLateOutput'] = 1
        self.pending = None

    def control_accounting(self, terminal_elapsed=None):
        recorded = self.last_control_cycle+1
        total = (recorded if terminal_elapsed is None else
                 max(recorded, math.ceil((terminal_elapsed-1e-9)/self.args.action_period)))
        skipped = sum(row['SkippedControlCycles'] for row in self.rows)+total-recorded
        return dict(control_steps=total, control_records=len(self.rows),
                    skipped_control_cycles=skipped,
                    control_deadline_misses=sum(row['MissedDeadline'] for row in self.rows)+skipped,
                    control_hold_cycles=sum(not row['DefenderOutputApplied'] for row in self.rows)+skipped,
                    policy_queries=sum(row['PolicyStarted'] for row in self.rows),
                    control_period_seconds=self.args.action_period,
                    control_denominator='all scheduled simulation-time cycles, including cycles without a query')

    def control(self, elapsed):
        if self.controller_closed or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Cannot start a control cycle after termination or at invalid simulation time.')
        cycle = math.floor((elapsed+1e-6)/self.args.action_period)
        if cycle <= self.last_control_cycle:
            raise ValueError('Control simulation time moved backwards or repeated a cycle.')
        skipped = cycle-self.last_control_cycle-1
        self.last_control_cycle = cycle
        started = time.monotonic()
        self.poll_control(elapsed)
        states, stamp, skew = synchronize_kinematics(self.curr_pos, self.curr_phi, self.curr_vel,
                                                    self.yaw_rates, self.last_stamp)
        with torch.no_grad():
            self.common.state[0].copy_(torch.as_tensor(states[[1, 2, 3, 0]], dtype=torch.float32))
            self.common.previous.copy_(self.applied)
            self.common.elapsed.fill_(elapsed)
            self.common.tick.fill_(min(int((elapsed+1e-6)/self.args.action_period), self.common.horizon-1))
            x = self.common.observation()
            # The opponent never waits for the defender's network/gradient work.
            self.applied[:, -1] = execute(self.common.attacker()/self.agility,
                                          self.applied[:, -1]/self.agility, self.common.cfg.thrust_delta)*self.agility
            actions = self.applied[0].cpu().numpy()[[3, 0, 1, 2]].copy()
            if not np.isfinite(actions).all():
                raise FloatingPointError('Non-finite unified VRX command.')
        self.publish_commands(range(4))
        physical = actions[:, ::-1]
        available = self.pending is None
        if available:
            self.request_id += 1
        identity = dict(episode=self.episode_id, request=self.request_id, cycle=cycle,
                        source_stamp=float(stamp), wall_deadline=started+self.args.action_period,
                        sim_deadline=float(elapsed)+self.args.action_period)
        self.rows.append(dict(timestamp=float(elapsed), AttPos=states[0, :2].copy(),
             AttPhi=states[0, 2].copy(), AttVel=states[0, 3:5].copy(), AttAct=actions[0].copy(),
             DefPos=states[1:, :2].flatten().copy(), DefPhi=states[1:, 2].copy(),
             DefVel=states[1:, 3:5].flatten().copy(), DefAct=actions[1:].flatten().copy(),
             PreviousDefAct=actions[1:].flatten().copy(), AppliedTimestamp=float(elapsed),
             PhysicalThrust=physical.flatten().copy(), SimulationStamp=stamp,
             ControlWallSeconds=time.monotonic()-started, PolicyWallSeconds=0.,
             FeedbackSkewSeconds=skew, DiscardedLateOutput=0, DiscardedStaleOutput=0, MissedDeadline=int(not available),
             EpisodeId=self.episode_id, RequestId=self.request_id if available else -1,
             ControlCycle=cycle, SkippedControlCycles=skipped,
             RequestWallDeadline=identity['wall_deadline'] if available else 0.,
             RequestSimulationDeadline=identity['sim_deadline'] if available else 0.,
             PolicyStarted=int(available), DefenderOutputApplied=0, TerminalCancellation=0))
        if available:
            future = self.executor.submit(self.compute_defense, x, self.common.order.clone(), identity.copy())
            self.pending = dict(future=future, row=len(self.rows)-1, started=started,
                                elapsed=elapsed, identity=identity, expired=False)
