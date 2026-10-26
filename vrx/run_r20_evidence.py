"""Deploy the indexed r20 runtime through the common source ROS transport."""
from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'train'))
import r20_independent_evidence as evidence


def load_controller(label, baseline_seed, manifest, inference_device):
    d, _, controls, _ = evidence.runtime(manifest)
    evidence.baseline_runtime()
    original_controller = d.JointMeasuredController
    policy_path = None
    baseline_family = label if label in ('boids', 'sac', 'rp', 'arboids', 'mappo', 'maac') else None
    source_checkpoint = str(d.SOURCE)

    if label in manifest['models']:
        item = manifest['models'][label]
        policy_path = Path(item['path'])
        if evidence.digest(policy_path) != item['sha256']:
            raise ValueError('The indexed r20 checkpoint changed.')

        class Controller(original_controller):
            def __init__(self, checkpoint, source):
                super().__init__(checkpoint, source)
                for module in self.actor.modules():
                    if hasattr(module, 'inference_device'):
                        module.inference_device = inference_device

    elif baseline_family is not None:
        family = baseline_family
        native_policy = None
        if family in ('sac', 'rp', 'arboids'):
            directory = evidence.BASE/'independent-training'/f'source-six-{family}-seed{baseline_seed}-20261021'
            training, config = controls.validate_author_source_repeat(directory)
            policy_path = directory/'best-development.pth'
            native_policy = controls.AuthorSourceRepeatPolicy(policy_path, config)
        elif family in ('mappo', 'maac'):
            directory = evidence.BASE/'r20-six-baseline-comparison'/f'{family}-seed{baseline_seed}'
            training = evidence.read(directory/'training-result.pth')
            if not training['completed'] or training['steps'] != training['budget']:
                raise ValueError('VRX requires the completed baseline training budget.')
            policy_path = directory/'best.pth'
            if evidence.digest(policy_path) != training['best_sha256']:
                raise ValueError('The selected original baseline changed.')
            endpoint = sys.modules['r20_endpoint_runtime']
            native_policy = endpoint.Actor(algorithm=family)
            native_policy.load_state_dict(evidence.read(policy_path)['actor'])
            native_policy.eval()

        class Controller:
            def __init__(self, checkpoint, source):
                self.steps = 0

            def reset(self):
                self.steps = 0

            def control(self, defenders, attacker, observations, boids, elapsed, total_time=80.):
                if family == 'boids':
                    command = boids.copy()
                elif family in ('sac', 'rp', 'arboids'):
                    from source_arboids import actor_thrust, source_mixture
                    with torch.no_grad():
                        raw = native_policy.actor(torch.as_tensor(observations,
                            dtype=torch.float32), True, False)[0].numpy()
                    if family == 'arboids':
                        command = source_mixture(raw, boids)
                    else:
                        command = actor_thrust(raw)+(boids if family == 'rp' else 0.)
                else:
                    # The author ROS observer appends an unused zero peer slot.
                    # Flat six-vessel actors consume the same 24 public features
                    # as their numerical training, without that padding slot.
                    if observations.shape != (6, 26) or np.any(observations[:, -2:] != 0):
                        raise ValueError('The source ROS observation padding changed.')
                    from source_arboids import actor_thrust
                    with torch.no_grad():
                        raw = native_policy(torch.as_tensor(observations[:, :-2],
                            dtype=torch.float32), deterministic=True)[0].numpy()
                    command = actor_thrust(raw)
                self.steps += 1
                return np.clip(command, -500., 1000.), dict(conditional_interventions=0,
                    control_steps=self.steps, physical_action=np.column_stack((
                        np.zeros((len(defenders), 2)), np.full(len(defenders), np.nan))))
    else:
        raise ValueError('Unknown frozen r20 or original-baseline variant: '+label)
    placeholder = policy_path or Path(manifest['models']['F00']['path'])
    return Controller(placeholder, source_checkpoint), policy_path, baseline_family, source_checkpoint


def policy_server(extra, manifest):
    controller, path, family, source = load_controller(extra.variant, extra.baseline_seed,
                                                       manifest, extra.inference_device)
    with Listener(extra.policy_server, family='AF_UNIX') as listener:
        with listener.accept() as connection:
            connection.send(dict(policy=str(path) if path else None, family=family, source=source))
            try:
                while True:
                    operation, values = connection.recv()
                    if operation == 'close':
                        break
                    if operation == 'reset':
                        controller.reset()
                        connection.send(None)
                        continue
                    defenders, attacker, observations, boids, elapsed, total_time = values
                    try:
                        command, info = controller.control(np.asarray(defenders), np.asarray(attacker),
                            np.asarray(observations), np.asarray(boids), elapsed, total_time)
                        info['physical_action'] = info['physical_action'].tolist()
                        connection.send(dict(command=command.tolist(), info=info))
                    except Exception as error:
                        connection.send(dict(error=repr(error)))
            finally:
                controller.reset()
    return 0


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--baseline-seed', type=int, default=42)
    parser.add_argument('--policy-server')
    parser.add_argument('--inference-device', default='cuda:0')
    extra, remaining = parser.parse_known_args()
    manifest = evidence.read(evidence.MANIFEST)
    if extra.policy_server:
        return policy_server(extra, manifest)

    import run_gate_contribution_sync_start as synchronized
    source = synchronized.source
    import develop_source_fusion as d
    label = extra.variant
    with tempfile.TemporaryDirectory(prefix='arboids-r20-vrx-policy-') as directory:
        socket = str(Path(directory)/'controller.sock')
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        command = [str(ROOT/'.venv/bin/python'), '-X', 'utf8', '-u', __file__,
            '--variant', label, '--baseline-seed', str(extra.baseline_seed),
            '--policy-server', socket, '--inference-device', extra.inference_device]
        process = subprocess.Popen(command, env=env)
        connection = None
        try:
            started = time.monotonic()
            while not Path(socket).exists():
                if process.poll() is not None or time.monotonic()-started > 90:
                    raise RuntimeError('The frozen policy process did not become ready.')
                time.sleep(.05)
            connection = Client(socket, family='AF_UNIX')
            info = connection.recv()
            path = Path(info['policy']) if info['policy'] else None

            class RemoteController:
                def __init__(self, checkpoint, source, inference_device='cpu'):
                    pass

                def reset(self):
                    connection.send(('reset', None))
                    connection.recv()

                def control(self, defenders, attacker, observations, boids, elapsed, total_time=80.):
                    connection.send(('control', (defenders.tolist(), attacker.tolist(),
                        observations.tolist(), boids.tolist(), elapsed, total_time)))
                    response = connection.recv()
                    if 'error' in response:
                        raise RuntimeError(response['error'])
                    response['info']['physical_action'] = np.asarray(response['info']['physical_action'])
                    return np.asarray(response['command']), response['info']

            d.JointMeasuredController = RemoteController
            return run_trial(extra, remaining, manifest, synchronized, path, info['family'], info['source'])
        finally:
            if connection is not None:
                try:
                    connection.send(('close', None))
                except (EOFError, BrokenPipeError):
                    pass
                connection.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=15)


def run_trial(extra, remaining, manifest, synchronized, policy_path, baseline_family, source_checkpoint):
    source = synchronized.source
    label = extra.variant
    placeholder = policy_path or Path(manifest['models']['F00']['path'])
    sys.argv = [sys.argv[0], '--method', 'joint', '--joint-checkpoint', str(placeholder),
                '--checkpoint', source_checkpoint, '--controller', 'AdaRes',
                '--termination-rule', 'source', '--device', 'cpu', *remaining]
    status = synchronized.main()
    if source.ACTIVE_ARGS is not None:
        args = source.ACTIVE_ARGS
        path = args.output_dir.resolve()/args.run_id/'result.json'
        result = json.loads(path.read_text())
        result.update(method=label, baseline_family=baseline_family,
            baseline_seed=extra.baseline_seed if baseline_family and baseline_family != 'boids' else None,
            policy=str(policy_path) if policy_path else None,
            policy_sha256=evidence.digest(policy_path) if policy_path else None,
            r20_manifest_sha256=evidence.digest(evidence.MANIFEST),
            frozen_runtime_sha256=manifest['runtime_sha256'],
            baseline_runtime_sha256=evidence.digest(evidence.OUTPUT/'baseline-runtime.pth'),
            deployment_adapter_sha256=evidence.digest(__file__),
            includes_mechanism_one=label in ('F10', 'F11', 'ordinary_with_M1', 'M1_independent'),
            includes_mechanism_two=label in ('F01', 'F11'),
            inference_device=extra.inference_device,
            policy_process_transport='local Unix socket; identical public measurements and command decoder',
            policy_scope=('Original baseline with its own frozen actor and command decoder'
                          if baseline_family else 'Indexed frozen r20 model; inference device adaptation only'))
        result['transport_file_sha256'] = result.pop('study_code')
        result.pop('joint_checkpoint', None)
        result.pop('joint_checkpoint_sha256', None)
        path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
