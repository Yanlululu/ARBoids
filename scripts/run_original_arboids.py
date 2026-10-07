"""Run the author's pinned release without importing the modified working tree.

The original training loop, environment, dynamics, networks and SAC updates are
loaded byte-for-byte from Git. Only the author's existing configuration switches
are used for the SAC, RP, reward and curriculum comparisons.
"""

import argparse
import contextlib
import copy
import csv
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = '9eb6df464808af6b9729f39644b4d658e1695e2d'
UPSTREAM_TREE = '2660ea907a91fc889c84ff0f68dad02d1ffdc7f6'
UPSTREAM_URL = 'https://github.com/taojy687/ARBoids'
SOURCE_FILES = (
    'train/train.py', 'train/envs/TADgame.py', 'train/envs/modules.py',
    'train/policy/SAC.py', 'train/policy/networks.py',
    'train/utils/config.py', 'train/utils/manager.py', 'train/configs/train.yaml',
)
VARIANTS = {
    'arboids': {},
    'sac': {'residual': False, 'adaptive': False, 'boid_state': False},
    'rp': {'adaptive': False},
    'no_formation_reward': {'form_reward': False},
    'no_curriculum': {'curriculum': False},
    'boids': {},
}
MODULES = ('train', 'envs.TADgame', 'envs.modules', 'policy.SAC',
           'policy.networks', 'utils.config', 'utils.manager')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class ReleaseLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import exact Git blobs in memory; no source checkout or edits are needed."""

    def __init__(self, blobs):
        self.blobs = blobs

    @staticmethod
    def relative(name):
        return 'train/train.py' if name == 'train' else 'train/' + name.replace('.', '/') + '.py'

    def find_spec(self, fullname, path=None, target=None):
        if fullname in MODULES or fullname in ('envs', 'policy', 'utils'):
            return importlib.util.spec_from_loader(
                fullname, self, is_package=fullname in ('envs', 'policy', 'utils'))
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        relative = self.relative(module.__name__)
        module.__file__ = f'git:{UPSTREAM_REVISION}:{relative}'
        if module.__name__ in MODULES:
            exec(compile(self.blobs[relative], module.__file__, 'exec'), module.__dict__)

    def get_source(self, fullname):
        return self.blobs[self.relative(fullname)].decode('utf-8')


@contextlib.contextmanager
def original_release():
    contaminated = [name for name in sys.modules if name == 'train' or
                    name.split('.')[0] in ('envs', 'policy', 'utils')]
    if contaminated:
        raise RuntimeError('Use a fresh process; modules already loaded: ' + ', '.join(contaminated))
    tree = subprocess.run(
        ['git', '-C', str(ROOT), 'rev-parse', f'{UPSTREAM_REVISION}^{{tree}}'],
        capture_output=True, check=True, text=True).stdout.strip()
    if tree != UPSTREAM_TREE:
        raise RuntimeError('The full author release tree does not match the pinned tree.')
    source = {relative: subprocess.run(
        ['git', '-C', str(ROOT), 'show', f'{UPSTREAM_REVISION}:{relative}'],
        capture_output=True, check=True).stdout for relative in (*SOURCE_FILES, 'requirements.txt')}
    hashes = {relative: sha256(blob) for relative, blob in source.items()}
    loader = ReleaseLoader(source)
    sys.meta_path.insert(0, loader)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        loaded = {name: importlib.import_module(name) for name in MODULES}
        verify_release(source, hashes, loaded)
        yield source, hashes, loaded
        verify_release(source, hashes, loaded)
    finally:
        sys.meta_path.remove(loader)
        sys.dont_write_bytecode = previous_bytecode
        for name in list(sys.modules):
            if name in MODULES or name in ('envs', 'policy', 'utils'):
                del sys.modules[name]


def verify_release(source, hashes, modules):
    for relative, expected in hashes.items():
        if sha256(source[relative]) != expected:
            raise RuntimeError('Author source was changed: ' + relative)
    for name, module in modules.items():
        expected = f'git:{UPSTREAM_REVISION}:{ReleaseLoader.relative(name)}'
        if module.__file__ != expected or not isinstance(module.__loader__, ReleaseLoader):
            raise RuntimeError('Imported a working-tree module: ' + name)


def configuration(source, modules, variant):
    import yaml
    utility = modules['utils.config']
    cfg = utility._dict_to_namespace(yaml.safe_load(source['train/configs/train.yaml']))
    original = utility._namespace_to_dict(cfg)
    for key, value in VARIANTS[variant].items():
        setattr(cfg.agent, key, value)
    expected = copy.deepcopy(original)
    expected['agent'].update(VARIANTS[variant])
    if utility._namespace_to_dict(cfg) != expected:
        raise RuntimeError('Unexpected change to the original configuration.')
    return cfg


def controller_for(variant, cfg):
    if variant == 'boids':
        return 'Boids'
    if not cfg.agent.residual:
        return 'RL'
    return 'AdaRes' if cfg.agent.adaptive else 'Res'


def release_identity(source):
    return dict(upstream_url=UPSTREAM_URL, upstream_revision=UPSTREAM_REVISION,
                upstream_tree=UPSTREAM_TREE,
                source_sha256={name: sha256(blob) for name, blob in source.items()})


def runtime_identity(source, device):
    import torch
    from packaging.requirements import Requirement
    packages = {}
    for line in source['requirements.txt'].decode('utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        requirement = Requirement(line)
        version = importlib.metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f'Author requirement {line} is not satisfied: {version}')
        packages[requirement.name] = version
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages=packages, device=str(device), torch_cuda=torch.version.cuda,
                torch_num_threads=torch.get_num_threads())


def evaluation_steps(cfg):
    # In the released main(), validation is inside train_steps >= warm_steps.
    interval = cfg.training.eval_interval
    first = ((max(1, cfg.training.warm_steps) + interval - 1) // interval) * interval
    return list(range(first, cfg.training.total_steps + 1, interval))


def validate_metrics(path, cfg):
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    schedule = evaluation_steps(cfg)
    if not schedule or schedule[-1] != cfg.training.total_steps:
        raise RuntimeError('The released schedule must save at the training endpoint.')
    if len(rows) != len(schedule) or [int(row['num']) for row in rows] != list(range(1, len(schedule) + 1)):
        raise RuntimeError('Original training did not complete its evaluation schedule.')
    if any(not math.isfinite(float(row[key])) for row in rows for key in ('def_sr', 'reward', 'time')):
        raise RuntimeError('Non-finite original training metrics.')
    if any(not 0 <= float(row['def_sr']) <= 1 or float(row['time']) < 0 for row in rows):
        raise RuntimeError('Invalid original training metrics.')
    return len(rows)


def validate_config(path, cfg, modules):
    import yaml
    expected = modules['utils.config']._namespace_to_dict(cfg)
    if yaml.safe_load(path.read_text(encoding='utf-8')) != expected:
        raise RuntimeError('Saved configuration differs from the author configuration.')


def checkpoint_state(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(
            torch.is_tensor(value) and torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError('Expected a finite original actor state dictionary.')
    return state


def validate_training_origin(checkpoint, variant, cfg, source, modules):
    """Reject modified-framework or interrupted-run models, even if shapes match."""
    record_path = checkpoint.parent / 'completed.json'
    if not record_path.is_file():
        raise ValueError('Missing completed.json from original-code training; '
                         'existing modified-framework checkpoints cannot be used.')
    record = json.loads(record_path.read_text(encoding='utf-8'))
    expected = dict(kind='author-release-training', schema_version=1,
                    passed=True, completed=True, initialization='from_scratch',
                    variant=variant, allowed_agent_changes=VARIANTS[variant],
                    training_steps=cfg.training.total_steps,
                    config=modules['utils.config']._namespace_to_dict(cfg),
                    evaluation_count=len(evaluation_steps(cfg)), **release_identity(source))
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Checkpoint provenance does not match the original release and variant.')
    if type(record.get('seed')) is not int or not 0 <= record['seed'] < 2 ** 32:
        raise ValueError('Checkpoint provenance has no valid training seed.')
    if checkpoint.name != record.get('checkpoint_file') or sha256(checkpoint.read_bytes()) != record.get('checkpoint_sha256'):
        raise ValueError('Checkpoint hash or filename does not match original training.')
    for name, key in (('config.yaml', 'config_sha256'), ('metrics1.csv', 'metrics_sha256')):
        artifact = checkpoint.parent / name
        if not artifact.is_file() or sha256(artifact.read_bytes()) != record.get(key):
            raise ValueError('Original training artifact is missing or changed: ' + name)
    validate_config(checkpoint.parent / 'config.yaml', cfg, modules)
    validate_metrics(checkpoint.parent / 'metrics1.csv', cfg)
    return record


def make_agent(cfg, modules, device):
    import torch
    env = modules['envs.TADgame'].TADEnv(
        cfg.agent.defender_num, cfg.agent.boid_state, cfg.agent.form_reward)
    dimensions = env.action_dim + int(cfg.agent.adaptive)
    agent = modules['policy.SAC'].SAC(
        cfg, env.feature1_dim, env.feature2_dim, dimensions,
        adaptive=cfg.agent.adaptive, device=torch.device(device))
    return env, agent


def check(source, hashes, modules):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    runtime = runtime_identity(source, 'cpu')
    checks = []
    for variant in VARIANTS:
        cfg = configuration(source, modules, variant)
        modules['utils.manager'].set_seed(101)
        env, agent = make_agent(cfg, modules, 'cpu')
        if env.Total_T != 80.0 or hasattr(env, 'protocol'):
            raise RuntimeError('Expected the unchanged released environment.')
        observation, _ = env.reset(2.0, noisy_agility=False)
        controller = controller_for(variant, cfg)
        for step in range(8):
            action = None if variant == 'boids' else agent.choose_action(
                observation, True).reshape(env.defender_num, -1)
            observation, reward, done, _ = env.step(action, controller)
            if not np.isfinite(observation).all() or not np.isfinite(reward).all():
                raise RuntimeError('Non-finite original-code integration result.')
            if done:
                break
        checks.append(dict(variant=variant, controller=controller,
                           allowed_agent_changes=VARIANTS[variant], steps=step + 1))
    verify_release(source, hashes, modules)
    return dict(passed=True, **release_identity(source), runtime=runtime,
                checks=checks, expected_evaluations=len(evaluation_steps(cfg)),
                formal_experiment=False)


def prepare_output(output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing experiment: ' + str(output))
    return output


def train(args, source, modules):
    import torch
    if args.variant == 'boids':
        raise ValueError('Boids has no training; use evaluate.')
    if args.checkpoint is not None:
        raise ValueError('Original training starts from scratch and accepts no checkpoint.')
    output = prepare_output(args.output)
    cfg = configuration(source, modules, args.variant)
    runtime = runtime_identity(source, args.device)
    modules['utils.manager'].set_seed(args.seed)
    manager = modules['utils.manager'].ExperimentManager(
        cfg, base_dir=str(output.parent), run_id=output.name, repeat_idx=1)
    # The author implementation supplies every training update and its schedule.
    started = time.perf_counter()
    modules['train'].main(cfg, manager, torch.device(args.device))
    elapsed = time.perf_counter() - started
    checkpoint = output / 'adares1.pth'
    metrics = output / 'metrics1.csv'
    evaluation_count = validate_metrics(metrics, cfg)
    validate_config(output / 'config.yaml', cfg, modules)
    if not checkpoint.is_file() or checkpoint.stat().st_mtime_ns < metrics.stat().st_mtime_ns:
        raise RuntimeError('Original trainer did not save the final checkpoint.')
    checkpoint_state(checkpoint)
    verify_release(source, release_identity(source)['source_sha256'], modules)
    result = dict(kind='author-release-training', schema_version=1,
                passed=True, completed=True, formal_experiment=True,
                variant=args.variant, seed=args.seed, initialization='from_scratch',
                allowed_agent_changes=VARIANTS[args.variant],
                config=modules['utils.config']._namespace_to_dict(cfg),
                runtime=runtime, elapsed_seconds=elapsed, evaluation_count=evaluation_count,
                training_steps=cfg.training.total_steps, checkpoint=str(checkpoint),
                checkpoint_file=checkpoint.name,
                checkpoint_sha256=sha256(checkpoint.read_bytes()),
                config_sha256=sha256((output / 'config.yaml').read_bytes()),
                metrics_sha256=sha256(metrics.read_bytes()),
                runner_sha256=sha256(Path(__file__).read_bytes()), **release_identity(source))
    (output / 'completed.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def evaluate(args, source, modules):
    import numpy as np
    import torch
    output = prepare_output(args.output)
    cfg = configuration(source, modules, args.variant)
    runtime = runtime_identity(source, args.device)
    provenance = None
    training_record_hash = None
    if args.variant != 'boids':
        if args.checkpoint is None:
            raise ValueError('A trained original-code checkpoint is required.')
        provenance = validate_training_origin(args.checkpoint, args.variant, cfg, source, modules)
        training_record_hash = sha256((args.checkpoint.parent / 'completed.json').read_bytes())
    env, agent = make_agent(cfg, modules, args.device)
    if args.variant != 'boids':
        state = checkpoint_state(args.checkpoint)
        agent.actor.load_state_dict(state)
    elif args.checkpoint is not None:
        raise ValueError('Boids does not use a checkpoint.')
    controller = controller_for(args.variant, cfg)
    output.mkdir(parents=True)
    counts = dict(success=0, collision=0, attacker_win=0, capture=0, timeout=0,
                  physical_breach=0, early_attacker_win=0)
    fields = ['seed', *counts, 'outcome_code', 'steps']
    with (output / 'episodes.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed in range(args.first_scene_seed, args.first_scene_seed + args.episodes):
            np.random.seed(seed)
            observation, _ = env.reset(args.agility, noisy_agility=False)
            done, steps = 0, 0
            while not done:
                action = None if args.variant == 'boids' else agent.choose_action(
                    observation, True).reshape(env.defender_num, -1)
                observation, reward, done, _ = env.step(action, controller)
                steps += 1
                if not np.isfinite(observation).all() or not np.isfinite(reward).all():
                    raise RuntimeError('Non-finite original-code evaluation.')
            breach = int(np.linalg.norm(env.attacker.pos) < env.Target_R)
            row = dict(seed=seed, success=int(done > 2), collision=int(done == 2),
                       attacker_win=int(done == 1), capture=int(done == 3), timeout=int(done == 4),
                       physical_breach=breach, early_attacker_win=int(done == 1 and not breach),
                       outcome_code=int(done), steps=steps)
            writer.writerow(row)
            for key in counts:
                counts[key] += row[key]
    if provenance and (
            sha256(args.checkpoint.read_bytes()) != provenance['checkpoint_sha256'] or
            sha256((args.checkpoint.parent / 'completed.json').read_bytes()) != training_record_hash):
        raise RuntimeError('Original checkpoint or training record changed during evaluation.')
    result = dict(passed=True, completed=True, formal_experiment=True, variant=args.variant,
                  **release_identity(source), episodes=args.episodes, runtime=runtime,
                  training_seed=provenance['seed'] if provenance else None,
                  first_scene_seed=args.first_scene_seed, agility=args.agility,
                  checkpoint_sha256=provenance['checkpoint_sha256'] if provenance else None,
                  training_record_sha256=training_record_hash,
                  episode_data_sha256=sha256((output / 'episodes.csv').read_bytes()),
                  runner_sha256=sha256(Path(__file__).read_bytes()),
                  evaluation_protocol='author-release-80s',
                  counts=counts, rates={k: v / args.episodes for k, v in counts.items()})
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'train', 'evaluate'))
    parser.add_argument('--variant', choices=tuple(VARIANTS), default='arboids')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--first-scene-seed', type=int, default=91000000)
    parser.add_argument('--agility', type=float, default=2.0)
    args = parser.parse_args()
    if args.action != 'check' and args.output is None:
        parser.error('--output is required for an experiment.')
    if args.episodes < 1:
        parser.error('--episodes must be positive.')
    if not math.isfinite(args.agility) or args.agility <= 0:
        parser.error('--agility must be positive and finite.')
    if not 0 <= args.first_scene_seed <= args.first_scene_seed + args.episodes - 1 < 2 ** 32:
        parser.error('Scene seeds must fit the original NumPy seed range.')
    if not 0 <= args.seed < 2 ** 32:
        parser.error('--seed must fit the original NumPy seed range.')
    with original_release() as (source, hashes, modules):
        print(json.dumps(dict(upstream_revision=UPSTREAM_REVISION, source_sha256=hashes)), flush=True)
        if args.action == 'check':
            result = check(source, hashes, modules)
        elif args.action == 'train':
            result = train(args, source, modules)
        else:
            result = evaluate(args, source, modules)
        result.update(upstream_revision=UPSTREAM_REVISION, source_sha256=hashes)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
